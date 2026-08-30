"""
Qwen3 forced-alignment engine
==========================================

This module uses Qwen3-ForcedAligner to find profanity in subtitle cue windows.
It slices the needed audio, aligns it with the cue text, adds padding to matched
words, and returns merged mute windows in the full audio timeline.

The model is loaded through qwen-asr. Mute windows receive 155 ms of padding on
each side to allow for small timing differences.

Each run saves its audio snippets, cue text, model output, and final mute windows
under debug_output/<watch_id>/.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# GPU dependencies are imported in load() so the module can be imported without
# the full GPU environment.

from ttml_profanity import (
    Cue,
    plan_snippet_windows,
    slice_to_wav,
    token_is_profane,
    wav_duration_seconds,
)

logger = logging.getLogger("uvicorn.error")

MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
MUTE_PAD_S = 0.155  # Padding added before and after each matched word.
DEBUG_ROOT = Path("debug_output")

# Map subtitle language codes to the names used by Qwen3.
LANGUAGE_MAP = {
    "en": "English", "zh": "Chinese", "yue": "Cantonese", "fr": "French",
    "de": "German", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "pt": "Portuguese", "ru": "Russian", "es": "Spanish",
}


class Qwen3AlignmentEngine:
    """Load Qwen3 once and use it to align audio snippets."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None  # Set after load() finishes.

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info("Loading %s on %s (bfloat16)", MODEL_ID, device)
        self.model = Qwen3ForcedAligner.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map=device,
        )
        logger.info("Qwen3-ForcedAligner ready.")

    # ------------------------------------------------------------ alignment

    def align_snippet(self, wav_path: Path, text: str, language: str) -> List[Dict[str, Any]]:
        """Align one audio snippet and return local word timestamps."""
        if self.model is None:
            raise RuntimeError("Engine not loaded")
        results = self.model.align(audio=str(wav_path), text=text, language=language)
        # qwen-asr results can differ between package versions, so normalize them.
        units = self._extract_units(results)
        words: List[Dict[str, Any]] = []
        for it in units or []:
            if isinstance(it, dict):
                token, start, end = it.get("text"), it.get("start_time"), it.get("end_time")
            else:
                token = getattr(it, "text", None)
                start = getattr(it, "start_time", None)
                end = getattr(it, "end_time", None)
            if token is None or start is None or end is None:
                continue
            words.append({"text": str(token), "start_s": float(start), "end_s": float(end)})

        return self._normalize_units(words, wav_duration_seconds(wav_path))

    @staticmethod
    def _extract_units(results: Any) -> List[Any]:
        """Return the word units from the alignment result."""
        if results is None:
            return []
        # Unwrap a single-item batch list.
        first = results
        if isinstance(results, (list, tuple)):
            if not results:
                return []
            first = results[0]
        # Handle results that expose an items attribute.
        items = getattr(first, "items", None)
        if items is not None:
            return list(items)
        # A list or tuple may already contain the units.
        if isinstance(first, (list, tuple)):
            return list(first)
        # Otherwise, treat it as one unit.
        return [first]

    @staticmethod
    def _normalize_units(words: List[Dict[str, Any]], snippet_dur_s: float) -> List[Dict[str, Any]]:
        """Normalize timestamps to seconds when qwen-asr returns milliseconds."""
        if not words:
            return words
        max_end = max(w["end_s"] for w in words)
        if max_end > max(snippet_dur_s * 2.0, snippet_dur_s + 5.0):
            for w in words:
                w["start_s"] /= 1000.0
                w["end_s"] /= 1000.0
        return words


# --------------------------------------------------------- title processing

def build_align_text(cues: List[Cue], win_start_s: float, win_end_s: float) -> str:
    """Join the alignment text from cues that overlap the current window."""
    parts = [
        (c.align_text or c.text) for c in cues
        if (c.align_text or c.text) and c.end_s > win_start_s and c.start_s < win_end_s
    ]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def merge_windows(windows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    windows = sorted(windows, key=lambda w: w["start"])
    merged: List[Dict[str, float]] = []
    for w in windows:
        if merged and w["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
        else:
            merged.append(dict(w))
    return merged


def process_title(
    engine: Qwen3AlignmentEngine,
    watch_id: str,
    audio_path: Path,
    cues: List[Cue],
    prof_re: re.Pattern,
    duration_s: Optional[float] = None,
    language_code: str = "en",
    progress_cb=None,
) -> List[Dict[str, float]]:
    """Create mute windows for one title.

    Audio sections with flagged TTML cues are aligned with Qwen3 and returned as
    merged mute windows in seconds:
        [{"start": 12.345, "end": 13.101}]
    """
    language = LANGUAGE_MAP.get((language_code or "en").lower().split("-")[0], "English")
    debug_dir = DEBUG_ROOT / str(watch_id)
    debug_dir.mkdir(parents=True, exist_ok=True)

    snippet_plan = plan_snippet_windows(cues, prof_re, duration_s=duration_s)
    logger.info("Cloud Slicer: %d cues -> %d snippet windows.", len(cues), len(snippet_plan))
    if not snippet_plan:
        (debug_dir / "mute_windows.json").write_text("[]", encoding="utf-8")
        return []

    mute_windows: List[Dict[str, float]] = []
    total = len(snippet_plan)

    for i, win in enumerate(snippet_plan):
        offset_s = float(win["start"])
        dur_s = float(win["end"]) - offset_s
        stem = f"snippet_{i:04d}"

        # 1. Slice the audio for this window.
        wav_path = slice_to_wav(audio_path, offset_s, dur_s, debug_dir / f"{stem}.wav")

        # 2. Get the cleaned cue text for this window.
        align_text = build_align_text(cues, win["start"], win["end"])
        (debug_dir / f"{stem}_text.txt").write_text(align_text + "\n", encoding="utf-8")
        if not align_text:
            logger.warning("Snippet %d/%d at %.2fs has no cue text; skipping.", i + 1, total, offset_s)
            continue

        # 3. Align the audio using local snippet timestamps.
        words = engine.align_snippet(wav_path, align_text, language)

        # 4. Add padding to profanity matches and shift them to the full timeline.
        hits = []
        for w in words:
            if not token_is_profane(w["text"], prof_re):
                continue
            hits.append(w["text"])
            mute_windows.append({
                "start": max(0.0, offset_s + w["start_s"] - MUTE_PAD_S),
                "end": offset_s + w["end_s"] + MUTE_PAD_S,
            })

        # 5. Save the model output for debugging.
        (debug_dir / f"{stem}_aligned.json").write_text(
            json.dumps({
                "watch_id": watch_id,
                "snippet_index": i,
                "global_offset_s": offset_s,
                "window": win,
                "language": language,
                "align_text": align_text,
                "qwen3_words_local_s": words,
                "profane_hits": hits,
                "mute_pad_s": MUTE_PAD_S,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("Snippet %d/%d (global %.2fs, %.1fs): %d words aligned, %d profane.",
                    i + 1, total, offset_s, dur_s, len(words), len(hits))
        if progress_cb:
            progress_cb(i + 1, total)

    merged = merge_windows(mute_windows)
    (debug_dir / "mute_windows.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    logger.info("%d raw hits -> %d merged mute windows. Debug artifacts: %s",
                len(mute_windows), len(merged), debug_dir)
    return merged