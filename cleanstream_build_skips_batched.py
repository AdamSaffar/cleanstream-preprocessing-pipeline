"""
batched build_skips
=============================

This version of build_skips() processes several frames at once with transformers. It uses the same detection prompt and range-building helpers as cleanstream_build_skips.py, but batching can make the scan faster on a GPU.

Example:
    !pip install -q git+https://github.com/huggingface/transformers accelerate
    !pip install -q "qwen-vl-utils[decord]" pillow
    import sys; sys.path.insert(0, "/content/drive/MyDrive")
    from cleanstream_build_skips_batched import build_skips_batched

    result = build_skips_batched(
        video_path="/content/drive/MyDrive/81188617_video_small_2.mp4",
        coarse_interval_s=3.0,
        batch_size=8,
    )
    # elapsed_s can be used to compare batch sizes.
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# reuse the prompt and range-building helpers from the main skip module.
from cleanstream_build_skips import (
    DETECTION_PROMPT,
    SEVERITY_THRESHOLD,
    _extract_frames_range,
    _video_duration_s,
    _flags_to_skip_ranges,
    QwenVLJudge,          # Reuse its JSON parsing method.
)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_PIXELS = 768 * 768


class QwenVLBatchedJudge:
    """Load Qwen2.5-VL-7B with transformers and judge frames in batches."""

    def __init__(self, model_id: str = MODEL_ID, max_pixels: int = MAX_PIXELS):
        self.model_id = model_id
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None

    def load(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"Loading {self.model_id} (batched transformers)")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto")
        self.model.eval()
        # Left padding keeps the batch aligned for generation.
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, max_pixels=self.max_pixels, padding_side="left")
        # Add a pad token if the tokenizer does not have one.
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        print("Qwen2.5-VL ready (batched).")

    def judge_batch(self, image_paths: List[str]) -> List[Dict]:
        """Judge a batch of frames and return one verdict per input frame."""
        import torch
        from qwen_vl_utils import process_vision_info

        # Build one prompt message for each frame.
        batch_messages = []
        for p in image_paths:
            batch_messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{p}"},
                    {"type": "text", "text": DETECTION_PROMPT},
                ],
            }])

        # Apply the chat template and collect the image inputs.
        texts = [self.processor.apply_chat_template(
                    m, tokenize=False, add_generation_prompt=True)
                 for m in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)

        inputs = self.processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        # Remove the prompt tokens from each generated response.
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return [QwenVLJudge._parse(d) for d in decoded]


def build_skips_batched(
    video_path: str,
    coarse_interval_s: float = 3.0,
    batch_size: int = 8,
    judge: Optional[QwenVLBatchedJudge] = None,
    severity_threshold: int = SEVERITY_THRESHOLD,
    keep_frames: bool = False,
    verbose: bool = True,
) -> Dict:
    """Build skip ranges by judging frames in batches.

    The returned data matches build_skips() and also includes timing values.
    """
    t_start = time.time()
    video_p = Path(video_path)
    if not video_p.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    dur_s = _video_duration_s(video_p)
    if verbose and dur_s:
        n = int(dur_s / coarse_interval_s)
        print(f"{video_p.name}: {dur_s:.0f}s -> ~{n} frames at {coarse_interval_s}s")

    if judge is None:
        judge = QwenVLBatchedJudge()
        judge.load()

    tmp = Path(tempfile.mkdtemp(prefix="cs_batched_"))
    flags: List[Dict] = []
    t_infer = 0.0
    try:
        frames = _extract_frames_range(video_p, tmp / "coarse", coarse_interval_s)
        if verbose:
            print(f"judging {len(frames)} frames in batches of {batch_size}")

        t0 = time.time()
        for bstart in range(0, len(frames), batch_size):
            batch = frames[bstart:bstart + batch_size]
            paths = [fr["path"] for fr in batch]
            verdicts = judge.judge_batch(paths)
            for fr, v in zip(batch, verdicts):
                if v["flag"] and v["severity"] >= severity_threshold:
                    flags.append({"t_s": fr["t_s"], "severity": int(v["severity"]),
                                  "confidence": v.get("confidence"),
                                  "reason": v.get("reason", "")})
                    if verbose:
                        mm, ss = divmod(int(fr["t_s"]), 60)
                        cf = v.get("confidence")
                        cf_s = f" conf={cf}" if cf is not None else ""
                        print(f"   🚩 {mm}:{ss:02d} sev={v['severity']}{cf_s} {v['reason']}")
            if verbose:
                done = min(bstart + batch_size, len(frames))
                print(f" {done}/{len(frames)}")
        t_infer = time.time() - t0

        skip_ranges = _flags_to_skip_ranges(flags, coarse_interval_s, dur_s)
        total = sum(r["end"] - r["start"] for r in skip_ranges) / 1000.0
        if verbose:
            print(f"{len(skip_ranges)} skip window(s), {total:.1f}s total, "
                  f"from {len(flags)} flagged frames.")
            print(f"inference wall-clock: {t_infer:.1f}s "
                  f"({len(frames)/max(t_infer,1e-9):.2f} frames/s)")
    finally:
        if not keep_frames:
            for p in tmp.rglob("*.jpg"):
                p.unlink(missing_ok=True)
            try:
                for d in sorted(tmp.rglob("*"), reverse=True):
                    if d.is_dir():
                        d.rmdir()
                tmp.rmdir()
            except OSError:
                pass

    return {
        "skip_ranges": skip_ranges,
        "video_duration_ms": int(round((dur_s or 0) * 1000)),
        "coarse_interval_s": coarse_interval_s,
        "scene_source": "qwen2.5vl_scene_batched_transformers",
        "num_flagged_frames": len(flags),
        "elapsed_s": round(t_infer, 1),
        "total_elapsed_s": round(time.time() - t_start, 1),
        "batch_size": batch_size,
    }




# The orchestrator imports build_skips, so point it to the batched version.
build_skips = build_skips_batched
QwenVLBatchJudge = QwenVLBatchedJudge   # Alias used by the orchestrator.


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--coarse", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = build_skips_batched(video_path=a.video, coarse_interval_s=a.coarse,
                              batch_size=a.batch_size)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"wrote {a.out}")
    print(f"\nelapsed(infer)={res['elapsed_s']}s  skips={len(res['skip_ranges'])}")