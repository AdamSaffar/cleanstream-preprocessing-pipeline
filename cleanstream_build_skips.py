"""
build_skips
=========================

This module finds scenes that should be skipped. The orchestrator calls build_skips() with a video path and uses the returned ranges in the filter JSON.

Frames are sampled across the video, checked by Qwen2.5-VL, then grouped into skip ranges. Each range includes severity, confidence, and reason information from the flagged frames.

Example:
    from cleanstream_build_skips import build_skips, QwenVLJudge
    judge = QwenVLJudge(); judge.load()
    result = build_skips(video_path=..., coarse_interval_s=3.0, judge=judge)
    # Skip ranges include timing and model details.

Reuse a loaded judge for multiple titles so the model is not loaded again.
build_vlm_scene_filter() is also available for running the full script on its own.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

"""
The original detection policy is not included in the public version because
content standards are subjective and should be decided by the person running
the pipeline.

To use this script, replace the POLICY section below with the content types
you want the model to detect. Be specific about what should and should not
be flagged, then keep the required JSON response format at the end.

Example policy categories might include violence, sexual content, profanity,
flashing lights, or other viewer preferences.
"""
DETECTION_PROMPT = (
    "You are a content filter for a video app. Review this video frame using "
    "the policy below.\n\n"
    "POLICY:\n"
    "[Replace this text with your own filtering rules.]\n\n"
    "Respond with only a compact JSON object and nothing else:\n"
    '{"flag": true or false, "severity": 0 to 3, '
    '"confidence": 0 to 100, "reason": "few words"}\n'
    "severity: 0=none, 1=low, 2=moderate, 3=high.\n"
    "confidence: how certain you are, from 0 to 100."
)

# A frame must be flagged and meet this severity level.
SEVERITY_THRESHOLD = 1


# --- range settings ---
# Add buffer before and after flagged frames, then merge nearby ranges.
LEAD_PAD_S = 6.0
TAIL_PAD_S = 6.0
MERGE_GAP_S = 25.0
MIN_SKIP_S = 2.0

# Ignore isolated flags. A skip needs at least MIN_CLUSTER flags within the configured time window.
MIN_CLUSTER = 2
CLUSTER_WINDOW_S = 20.0


# --- frame extract ----
def _extract_frames_range(video_path: Path, out_dir: Path, interval_s: float,
                          start_s: float = 0.0, end_s: Optional[float] = None
                          ) -> List[Dict]:
    """Sample video frames at the requested interval."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / interval_s
    pattern = str(out_dir / "f_%06d.jpg")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    if end_s is not None and end_s > start_s:
        cmd += ["-t", f"{end_s - start_s:.3f}"]
    cmd += ["-i", str(video_path), "-vf", f"fps={fps:.6f}", "-q:v", "3", pattern, "-y"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-400:]}")
    files = sorted(out_dir.glob("f_*.jpg"))
    return [{"path": str(f), "t_s": start_s + i * interval_s}
            for i, f in enumerate(files)]


def _video_duration_s(video_path: Path) -> Optional[float]:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except (ValueError, AttributeError):
        return None


# --- the VLM ---
class QwenVLJudge:
    """Load Qwen2.5-VL once and use it to judge video frames."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 max_pixels: int = 768 * 768):
        self.model_id = model_id
        self.max_pixels = max_pixels
        self.model = None
        self.processor = None

    def load(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print(f"Loading {self.model_id}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto")
        # Limit visual tokens to keep inference faster.
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, max_pixels=self.max_pixels)
        print("Qwen2.5-VL ready.")

    def judge(self, image_path: str) -> Dict:
        """Return {'flag','severity','confidence','reason'} for one frame."""
        from qwen_vl_utils import process_vision_info
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": DETECTION_PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs,
                                videos=video_inputs, padding=True,
                                return_tensors="pt").to(self.model.device)
        gen = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        return self._parse(out)

    @staticmethod
    def _parse(text: str) -> Dict:
        """Read a JSON verdict from the model response when possible.
        Confidence is the model's own estimate, not a calibrated probability"""
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                conf = obj.get("confidence", None)
                try:
                    conf = int(conf)
                    conf = max(0, min(100, conf))
                except (TypeError, ValueError):
                    conf = None
                return {
                    "flag": bool(obj.get("flag", False)),
                    "severity": int(obj.get("severity", 0) or 0),
                    "confidence": conf,
                    "reason": str(obj.get("reason", ""))[:80],
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # Fall back to a few keywords when the JSON cannot be read.
        low = text.lower()
        flag = ("true" in low and "flag" in low) or "sexual" in low or "nud" in low
        return {"flag": flag, "severity": 2 if flag else 0,
                "confidence": None, "reason": "parse-fallback"}


# --- range bridging ---
def _cluster_filter(flagged_ts: List[float], min_cluster: int = MIN_CLUSTER,
                    window_s: float = CLUSTER_WINDOW_S) -> List[float]:
    """Keep flags that have enough nearby flags in the time window."""
    if min_cluster <= 1:
        return flagged_ts
    kept = []
    for t in flagged_ts:
        nearby = sum(1 for f in flagged_ts if abs(f - t) <= window_s)
        if nearby >= min_cluster:
            kept.append(t)
    return kept

def _flags_to_skip_ranges(flags: List[Dict], interval_s: float,
                          duration_s: Optional[float],
                          lead_pad_s: float = LEAD_PAD_S,
                          tail_pad_s: float = TAIL_PAD_S,
                          merge_gap_s: float = MERGE_GAP_S,
                          min_skip_s: float = MIN_SKIP_S,
                          min_cluster: int = MIN_CLUSTER,
                          cluster_window_s: float = CLUSTER_WINDOW_S) -> List[Dict]:
    """Turn flagged frames into padded, merged skip ranges in milliseconds.

    The output also keeps the main severity, confidence, and reasons from the
    frames in each range.
    """
    if not flags:
        return []
    flags = sorted(flags, key=lambda f: f["t_s"])
    all_ts = [f["t_s"] for f in flags]

    # Remove isolated flags before adding padding.
    kept_ts = set(_cluster_filter(all_ts, min_cluster, cluster_window_s))
    if not kept_ts:
        return []
    kept_flags = [f for f in flags if f["t_s"] in kept_ts]

    # Each sample covers one interval. Add padding around it.
    windows = []
    for f in kept_flags:
        s = max(0.0, f["t_s"] - lead_pad_s)
        e = f["t_s"] + interval_s + tail_pad_s
        if duration_s:
            e = min(e, duration_s)
        windows.append({"s": s, "e": e, "flag": f})
    windows.sort(key=lambda w: w["s"])

    # Merge overlapping or nearby ranges and keep their flags together.
    merged = []
    for w in windows:
        if merged and w["s"] <= merged[-1]["e"] + merge_gap_s:
            merged[-1]["e"] = max(merged[-1]["e"], w["e"])
            merged[-1]["members"].append(w["flag"])
        else:
            merged.append({"s": w["s"], "e": w["e"], "members": [w["flag"]]})

    out = []
    for m in merged:
        if m["e"] - m["s"] < min_skip_s:
            continue
        members = m["members"]
        # Use the highest-severity frame for the main details.
        peak = max(members, key=lambda f: (f.get("severity", 0),
                                           f.get("confidence") or 0))
        # Keep up to five different reasons from the range.
        seen = set(); samples = []
        for f in sorted(members, key=lambda f: -(f.get("severity", 0))):
            r = (f.get("reason") or "").strip()
            if r and r.lower() not in seen and r != "parse-fallback":
                seen.add(r.lower()); samples.append(r)
            if len(samples) >= 5:
                break
        rng = {
            "start": int(round(m["s"] * 1000)),
            "end": int(round(m["e"] * 1000)),
            "severity": int(peak.get("severity", 0)),
            "reason": (peak.get("reason") or "").strip(),
        }
        pc = peak.get("confidence", None)
        if pc is not None:
            rng["confidence"] = int(pc)
        if samples:
            rng["sample_reasons"] = samples
        out.append(rng)
    return out


# ---------------------------------------------------------------- driver

def build_vlm_scene_filter(
    video_path: str,
    out_path: str,
    movie_id: str = "",
    title: str = "",
    episode: str = "",
    coarse_interval_s: float = 3.0,
    fine_interval_s: float = 0.5,
    two_pass: bool = False,
    severity_threshold: int = SEVERITY_THRESHOLD,
    merge_into: Optional[str] = None,
    judge: Optional[QwenVLJudge] = None,
    keep_frames: bool = False,
) -> Dict:
    """Run the full VLM pipeline and write a filter JSON file.

    The optional second pass samples more frames near the first pass's hits.
    """
    video_p = Path(video_path)
    dur_s = _video_duration_s(video_p)
    if dur_s:
        n_coarse = int(dur_s / coarse_interval_s)
        print(f"duration {dur_s:.0f}s -> coarse pass ~{n_coarse} frames "
              f"at {coarse_interval_s}s")

    if judge is None:
        judge = QwenVLJudge()
        judge.load()

    # --- First pass: scan the full video ---
    tmp = Path(tempfile.mkdtemp(prefix="cs_vlm_"))
    flags: List[Dict] = []      # Timestamp and model details for flagged frames.
    try:
        coarse = _extract_frames_range(video_p, tmp / "coarse", coarse_interval_s)
        print(f"coarse: judging {len(coarse)} frames")
        for i, fr in enumerate(coarse):
            v = judge.judge(fr["path"])
            if v["flag"] and v["severity"] >= severity_threshold:
                flags.append({"t_s": fr["t_s"], "severity": int(v["severity"]),
                              "confidence": v.get("confidence"),
                              "reason": v.get("reason", "")})
                mm, ss = divmod(int(fr["t_s"]), 60)
                cf = v.get("confidence")
                cf_s = f" conf={cf}" if cf is not None else ""
                print(f"{mm}:{ss:02d} sev={v['severity']}{cf_s} {v['reason']}")
            if (i + 1) % 50 == 0:
                print(f" {i+1}/{len(coarse)}")
        print(f"   coarse hits: {len(flags)}")

        # --- Second pass: scan more closely around hits ---
        if two_pass and flags:
            print(f"fine pass around {len(flags)} hit(s) at {fine_interval_s}s")
            pad = coarse_interval_s
            regions = []
            for t in sorted(f["t_s"] for f in flags):
                rs = max(0.0, t - pad)
                re_ = t + pad
                if regions and rs <= regions[-1][1]:
                    regions[-1][1] = max(regions[-1][1], re_)
                else:
                    regions.append([rs, re_])
            fine_flags: List[Dict] = []
            for ri, (rs, re_) in enumerate(regions):
                frs = _extract_frames_range(video_p, tmp / f"fine{ri}",
                                            fine_interval_s, start_s=rs, end_s=re_)
                for fr in frs:
                    v = judge.judge(fr["path"])
                    if v["flag"] and v["severity"] >= severity_threshold:
                        fine_flags.append({"t_s": fr["t_s"],
                                           "severity": int(v["severity"]),
                                           "confidence": v.get("confidence"),
                                           "reason": v.get("reason", "")})
            # Add the closer samples to the original flags.
            flags = sorted(fine_flags + flags, key=lambda f: f["t_s"])
            print(f"   fine hits: {len(fine_flags)}")

        skip_ranges = _flags_to_skip_ranges(
            flags, min(fine_interval_s, coarse_interval_s), dur_s)
        total = sum(w["end"] - w["start"] for w in skip_ranges) / 1000.0
        print(f"{len(skip_ranges)} skip window(s), {total:.1f}s total.")
    finally:
        if not keep_frames:
            for p in tmp.rglob("*.jpg"):
                p.unlink(missing_ok=True)

    # --- assemble / merge ---
    if merge_into and Path(merge_into).exists():
        obj = json.loads(Path(merge_into).read_text(encoding="utf-8"))
        obj["skip_ranges"] = skip_ranges
        if dur_s:
            obj["video_duration_ms"] = int(round(dur_s * 1000))
        obj.setdefault("movieID", movie_id)
        obj["_scene_source"] = "qwen2.5vl_scene_batch"
        print(f"merged skip_ranges into {merge_into} (captions/mutes preserved).")
    else:
        obj = {
            "movieID": movie_id, "title": title, "episode": episode,
            "video_duration_ms": int(round((dur_s or 0) * 1000)),
            "_scene_source": "qwen2.5vl_scene_batch",
            "captions": [], "mute_windows": [], "skip_ranges": skip_ranges,
        }
        print("wrote scenes-only filter (no captions/mutes).")

    Path(out_path).write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"wrote {out_path}: skips={len(skip_ranges)}")
    return obj




# --- Orchestrator entry point ---
def build_skips(
    video_path: str,
    coarse_interval_s: float = 3.0,
    judge: Optional["QwenVLJudge"] = None,
    two_pass: bool = False,
    severity_threshold: int = SEVERITY_THRESHOLD,
    keep_frames: bool = False,
    verbose: bool = True,
) -> Dict:
    """Build skip ranges for the CleanStream orchestrator.

    This scans a video and returns skip ranges with details from the model. It
    does not write a filter file; the orchestrator handles that step.

    Args:
        video_path: Path to the video file.
        coarse_interval_s: Seconds between sampled frames.
        judge: A loaded judge to reuse. A new one is loaded if needed.
        two_pass: Whether to run a closer second scan around flagged frames.
        severity_threshold: Minimum severity needed to flag a frame.
        keep_frames: Keep extracted JPEGs for debugging.
        verbose: Whether to print progress messages.

    Returns a dict:
        {
          "skip_ranges": [ {start,end,severity,confidence,reason,sample_reasons}, ... ],
          "video_duration_ms": int,
          "coarse_interval_s": float,
          "scene_source": "qwen2.5vl_scene_batch",
          "num_flagged_frames": int,
        }
    """
    video_p = Path(video_path)
    if not video_p.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    dur_s = _video_duration_s(video_p)
    if verbose and dur_s:
        n = int(dur_s / coarse_interval_s)
        print(f"{video_p.name}: {dur_s:.0f}s -> ~{n} frames at {coarse_interval_s}s")

    if judge is None:
        judge = QwenVLJudge()
        judge.load()

    tmp = Path(tempfile.mkdtemp(prefix="cs_vlm_"))
    flags: List[Dict] = []
    try:
        frames = _extract_frames_range(video_p, tmp / "coarse", coarse_interval_s)
        if verbose:
            print(f"judging {len(frames)} frames")
        for i, fr in enumerate(frames):
            v = judge.judge(fr["path"])
            if v["flag"] and v["severity"] >= severity_threshold:
                flags.append({"t_s": fr["t_s"], "severity": int(v["severity"]),
                              "confidence": v.get("confidence"),
                              "reason": v.get("reason", "")})
                if verbose:
                    mm, ss = divmod(int(fr["t_s"]), 60)
                    cf = v.get("confidence")
                    cf_s = f" conf={cf}" if cf is not None else ""
                    print(f"{mm}:{ss:02d} sev={v['severity']}{cf_s} {v['reason']}")
            if verbose and (i + 1) % 50 == 0:
                print(f" {i+1}/{len(frames)}")

        # Optionally sample more frames near the flagged areas
        if two_pass and flags:
            pad = coarse_interval_s
            regions = []
            for t in sorted(f["t_s"] for f in flags):
                rs = max(0.0, t - pad); re_ = t + pad
                if regions and rs <= regions[-1][1]:
                    regions[-1][1] = max(regions[-1][1], re_)
                else:
                    regions.append([rs, re_])
            for ri, (rs, re_) in enumerate(regions):
                frs = _extract_frames_range(video_p, tmp / f"fine{ri}",
                                            0.5, start_s=rs, end_s=re_)
                for fr in frs:
                    v = judge.judge(fr["path"])
                    if v["flag"] and v["severity"] >= severity_threshold:
                        flags.append({"t_s": fr["t_s"], "severity": int(v["severity"]),
                                      "confidence": v.get("confidence"),
                                      "reason": v.get("reason", "")})

        skip_ranges = _flags_to_skip_ranges(flags, coarse_interval_s, dur_s)
        total = sum(r["end"] - r["start"] for r in skip_ranges) / 1000.0
        if verbose:
            print(f"{len(skip_ranges)} skip window(s), {total:.1f}s total, "
                  f"from {len(flags)} flagged frames.")
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
        "scene_source": "qwen2.5vl_scene_batch",
        "num_flagged_frames": len(flags),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--movie-id", default="")
    ap.add_argument("--coarse", type=float, default=3.0)
    ap.add_argument("--fine", type=float, default=0.5)
    ap.add_argument("--merge-into", default=None)
    a = ap.parse_args()
    build_vlm_scene_filter(video_path=a.video, out_path=a.out, movie_id=a.movie_id,
                           coarse_interval_s=a.coarse, fine_interval_s=a.fine,
                           merge_into=a.merge_into)
