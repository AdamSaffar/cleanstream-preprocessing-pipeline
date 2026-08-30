"""
build_mutes
=========================

This module is the mute-generation part of the CleanStream pipeline. The
orchestrator calls build_mutes() to process a title's audio and TTML subtitles.

The forced-alignment work is handled by the existing backend modules:
  - ttml_profanity.py parses captions and prepares the profanity regex.
  - qwen3_engine.py loads Qwen3 and finds the mute windows.

build_mutes() returns mute windows, captions, and a few values the orchestrator uses for checks and logging. It does not write the final filter JSON.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from ttml_profanity import (
    compile_profanity_regex,
    load_profanity_words,
    media_duration_seconds,
    parse_ttml_cues,
)
from qwen3_engine import Qwen3AlignmentEngine, process_title


def _windows_s_to_ms(windows: List[Dict[str, float]]) -> List[Dict]:
    """convert merged mute windows from seconds to milliseconds"""
    out: List[Dict] = []
    for w in windows:
        st = int(round(float(w["start"]) * 1000.0))
        en = int(round(float(w["end"]) * 1000.0))
        if en > st:
            out.append({"start": st, "end": en})
    return out


def _cues_to_caption_json(cues) -> List[Dict]:
    """
    Convert non-empty caption cues from seconds to milliseconds.

    cue.text is the display text, so accessibility notes such as [squeals] are
    kept. Captions stay uncensored here; Censor.java handles that in the app.
    """
    out: List[Dict] = []
    for c in cues:
        if not c.text:
            continue
        out.append({
            "start": int(round(c.start_s * 1000.0)),
            "end": int(round(c.end_s * 1000.0)),
            "text": c.text,
        })
    return out


def build_mutes(
    audio_path: str,
    ttml_path: str,
    profanity_path: str = "profanity.txt",
    language_code: str = "en",
    engine: Optional[Qwen3AlignmentEngine] = None,
    watch_id: str = "",
    verbose: bool = True,
) -> Dict:
    """
    Build mute windows and captions for the CleanStream orchestrator.

    This runs Qwen3 forced alignment on a title's audio and TTML file. It returns
    profanity mute windows in milliseconds but does not write the final filter.
    The orchestrator adds the results to the filter JSON.

    Args:
        audio_path: Path to the extracted audio file.
        ttml_path: Path to the TTML subtitle file.
        profanity_path: Path to the profanity word list.
        language_code: Subtitle language, such as "en" or "es".
        engine: A loaded engine to reuse. A new one is loaded if needed.
        watch_id: Name used for the aligner's debug output folder. Defaults to the TTML file name.
        verbose: Whether to print progress messages.

    Returns a dict:
        {
          "mute_windows": [ {start, end}, ... ],   # milliseconds
          "captions":     [ {start, end, text}, ... ],  # milliseconds
          "audio_duration_ms": int,                # from the audio file
          "num_flagged_cues": int,                 # captions with profanity
          "num_captions": int,
          "num_mute_windows": int,
          "mute_source": "qwen3_forced_aligner",
          "elapsed_s": float,                      # processing time
        }
    """
    t_start = time.time()
    audio_p = Path(audio_path)
    ttml_p = Path(ttml_path)
    if not audio_p.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    if not ttml_p.exists():
        raise FileNotFoundError(f"ttml not found: {ttml_path}")

    # 1. cues (cleaned) + profanity regex
    ttml_xml = ttml_p.read_text(encoding="utf-8", errors="ignore")
    cues = parse_ttml_cues(ttml_xml)
    prof_words = load_profanity_words(Path(profanity_path))
    prof_re = compile_profanity_regex(prof_words)
    if prof_re is None:
        raise RuntimeError("profanity regex empty — check profanity.txt")

    n_flagged = sum(1 for c in cues
                    if (c.align_text or c.text) and prof_re.search(c.align_text or c.text))
    if verbose:
        print(f"{len(cues)} cues parsed, {n_flagged} contain flagged words.")

    # 2. audio duration (the authoritative timeline; also a Stage-0 health signal)
    audio_dur_s = media_duration_seconds(audio_p)
    audio_dur_ms = int(round(audio_dur_s * 1000.0)) if audio_dur_s else 0
    if verbose and audio_dur_s:
        print(f"audio duration: {audio_dur_s:.1f}s")

    # short-circuit: nothing flagged -> no alignment work, no mutes
    if n_flagged == 0:
        if verbose:
            print("no flagged cues; 0 mute windows (skipping alignment).")
        return {
            "mute_windows": [],
            "captions": _cues_to_caption_json(cues),
            "audio_duration_ms": audio_dur_ms,
            "num_flagged_cues": 0,
            "num_captions": sum(1 for c in cues if c.text),
            "num_mute_windows": 0,
            "mute_source": "qwen3_forced_aligner",
            "elapsed_s": round(time.time() - t_start, 1),
        }

    # 3. Qwen3 forced alignment over flagged windows (reuse a loaded engine).
    if engine is None:
        engine = Qwen3AlignmentEngine()
        engine.load()
    wid = watch_id or ttml_p.stem
    mute_windows_s = process_title(
        engine=engine,
        watch_id=wid,
        audio_path=audio_p,
        cues=cues,
        prof_re=prof_re,
        duration_s=audio_dur_s,
        language_code=language_code,
    )
    mute_json = _windows_s_to_ms(mute_windows_s)
    if verbose:
        print(f"{len(mute_json)} merged mute windows aligned.")

    captions_json = _cues_to_caption_json(cues)
    return {
        "mute_windows": mute_json,
        "captions": captions_json,
        "audio_duration_ms": audio_dur_ms,
        "num_flagged_cues": n_flagged,
        "num_captions": len(captions_json),
        "num_mute_windows": len(mute_json),
        "mute_source": "qwen3_forced_aligner",
        "elapsed_s": round(time.time() - t_start, 1),
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="CleanStream build_mutes")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ttml", required=True)
    ap.add_argument("--profanity", default="profanity.txt")
    ap.add_argument("--language", default="en")
    ap.add_argument("--out", default=None, help="optional: write result JSON here")
    a = ap.parse_args()
    res = build_mutes(audio_path=a.audio, ttml_path=a.ttml,
                      profanity_path=a.profanity, language_code=a.language)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"wrote {a.out}")
    print(f"\nmutes={res['num_mute_windows']}  "
          f"flagged_cues={res['num_flagged_cues']}  elapsed={res['elapsed_s']}s")
