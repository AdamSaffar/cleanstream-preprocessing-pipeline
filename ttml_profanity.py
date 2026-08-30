"""
TTML parsing, profanity matching, and audio helpers
==============================================================

This module parses Netflix-style TTML captions, builds the profanity regex, and
contains the audio helpers used by the alignment pipeline. Times are converted
to seconds internally. Flagged caption sections are cut into 16 kHz mono WAV
files before they are sent to the aligner.
"""

from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_TICK_RATE = 10_000_000  # Default number of TTML ticks per second.

# --- text helpers ---

BRACKET_RE = re.compile(r"\[[^\]]*?\]")
PAREN_RE = re.compile(r"\([^)]*?\)")
MUSIC_RE = re.compile(r"[♪♫♩♬♭♯]|â™ª")
MULTISPACE_RE = re.compile(r"[ \t]+")


def clean_caption_text(s: str) -> str:
    """Clean caption text for Qwen3 alignment by removing non-spoken notes."""
    s = (s or "").strip()
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = BRACKET_RE.sub("", s)
    s = PAREN_RE.sub("", s)
    s = MUSIC_RE.sub("", s)
    s = s.replace("\r", "").replace("\n", " ")
    s = re.sub(r"(^|\s)-\s*", r"\1", s)  # Remove leading dialogue dashes.
    s = MULTISPACE_RE.sub(" ", s).strip()
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    if not re.search(r"[A-Za-z0-9']", s):
        return ""
    return s


def display_caption_text(s: str) -> str:
    """Clean caption text for display while keeping accessibility notes."""
    s = (s or "").strip()
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = MUSIC_RE.sub("", s)                       # Remove standalone music symbols.
    s = s.replace("\r", "").replace("\n", " ")
    s = re.sub(r"(^|\s)-\s*", r"\1", s)          # Remove leading dialogue dashes.
    s = MULTISPACE_RE.sub(" ", s).strip()
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    # Keep captions that have text, numbers, or caption tags.
    if not re.search(r"[A-Za-z0-9'\[\]()]", s):
        return ""
    return s


# --- TTML cues ---

@dataclass(frozen=True)
class Cue:
    idx: int
    cue_id: str
    start_s: float
    end_s: float
    text: str        # Text shown in the caption.
    align_text: str = ""  # Cleaned text used for Qwen3 alignment.


def _localname(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _text_with_br(elem: ET.Element) -> str:
    out: List[str] = []
    if elem.text:
        out.append(elem.text)
    for child in list(elem):
        if _localname(child.tag).lower() == "br":
            out.append(" ")
        else:
            out.append(_text_with_br(child))
        if child.tail:
            out.append(child.tail)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _parse_clock(expr: str, tick_rate: int) -> float:
    """Convert a TTML time value to seconds."""
    expr = (expr or "").strip()
    if expr.endswith("t"):
        return int(expr[:-1]) / float(tick_rate)
    if expr.endswith("ms"):
        return float(expr[:-2]) / 1000.0
    if expr.endswith("s"):
        return float(expr[:-1])
    if ":" in expr:
        h, m, s = expr.split(":")
        return float(h) * 3600.0 + float(m) * 60.0 + float(s)
    return float(expr)


def parse_ttml_cues(ttml_xml: str) -> List[Cue]:
    """Parse a TTML document into cleaned cues with second-based timestamps."""
    root = ET.fromstring((ttml_xml or "").lstrip())

    tick_rate = DEFAULT_TICK_RATE
    for k, v in root.attrib.items():
        if k.endswith("tickRate"):
            try:
                tick_rate = int(float(v))
            except ValueError:
                pass
            break

    cues: List[Cue] = []
    idx = 0
    for p in root.findall(".//{*}p"):
        begin, end = p.attrib.get("begin"), p.attrib.get("end")
        if not begin or not end:
            continue
        start_s = _parse_clock(begin, tick_rate)
        end_s = _parse_clock(end, tick_rate)
        if end_s <= start_s:
            continue
        raw_line = _text_with_br(p)
        disp = display_caption_text(raw_line)     # Keep notes for displayed captions.
        aln = clean_caption_text(raw_line)        # Remove notes for alignment.
        cue_id = (
            p.attrib.get("{http://www.w3.org/XML/1998/namespace}id", "")
            or p.attrib.get("xml:id", "")
        )
        cues.append(Cue(idx=idx, cue_id=cue_id, start_s=start_s, end_s=end_s,
                        text=disp, align_text=aln))
        idx += 1
    return cues


# --- profanity matching ---

def load_profanity_words(path: Path) -> List[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _normalize_to_base(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (term or "").strip().lower())


def _fuzzy_core(base: str) -> str:
    """Match a word even if punctuation or spaces split its letters."""
    sep = r"(?:[\s_\-.*\"'“”‘’`]+|[‐-―−])*"
    return sep.join(re.escape(ch) for ch in base)


_COMMON_SUFFIX = r"(?:['’]?\s*(?:s|es|ed|er|ers|ing|in['’]?))?"


def compile_profanity_regex(words: List[str], min_base_len: int = 3) -> Optional[re.Pattern]:
    """Build one regex for matching word forms and simple obfuscations."""
    seen = set()
    patterns = []
    for w in words:
        w = (w or "").strip()
        if not w or w.startswith("#"):
            continue
        is_star = w.endswith("*")
        base = _normalize_to_base(w[:-1] if is_star else w)
        if len(base) < min_base_len or (base, is_star) in seen:
            continue
        seen.add((base, is_star))
        tail = r"(?:\w*)" if is_star else ""
        patterns.append(
            rf"(?<![a-z0-9]){_fuzzy_core(base)}{_COMMON_SUFFIX}{tail}(?![a-z0-9])"
        )
    if not patterns:
        return None
    return re.compile("(?:%s)" % "|".join(patterns), re.IGNORECASE)


def text_has_profanity(text: str, prof_re: re.Pattern) -> bool:
    return bool(text and prof_re.search(text))


def _collapse_elongation(word: str) -> str:
    """Reduce three or more repeated letters to one."""
    return re.sub(r"(.)\1{2,}", r"\1", word or "")


def token_is_profane(token: str, prof_re: re.Pattern) -> bool:
    """Check an aligned word against the profanity regex."""
    t = re.sub(r"[^A-Za-z0-9'’\-]+", "", token or "").strip("-")
    if not t:
        return False
    if prof_re.search(t):
        return True
    collapsed = _collapse_elongation(t)
    if collapsed != t and prof_re.search(collapsed):
        return True
    return False


# --- snippet planning ---
# Only align short sections around flagged cues, then merge nearby sections.

SNIPPET_PAD_S = 1.5
MIN_SNIPPET_S = 10.0
SNIPPET_MERGE_GAP_S = 2.0


def plan_snippet_windows(
    cues: List[Cue],
    prof_re: re.Pattern,
    duration_s: Optional[float] = None,
    pad_s: float = SNIPPET_PAD_S,
    min_len_s: float = MIN_SNIPPET_S,
    merge_gap_s: float = SNIPPET_MERGE_GAP_S,
) -> List[Dict[str, float]]:
    """Create padded, merged audio windows around flagged cues."""
    windows: List[Dict[str, float]] = []
    for c in cues:
        # Match spoken text rather than display-only annotations.
        if not text_has_profanity(c.align_text or c.text, prof_re):
            continue
        start = c.start_s - pad_s
        end = c.end_s + pad_s
        if end - start < min_len_s:
            grow = (min_len_s - (end - start)) / 2.0
            start -= grow
            end += grow
        start = max(0.0, start)
        if duration_s and duration_s > 0:
            end = min(end, duration_s)
        if end > start:
            windows.append({"start": start, "end": end})

    windows.sort(key=lambda w: w["start"])
    merged: List[Dict[str, float]] = []
    for w in windows:
        if merged and w["start"] <= merged[-1]["end"] + merge_gap_s:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
        else:
            merged.append(dict(w))
    return merged



def slice_to_wav(
    source_audio: Path, start_s: float, dur_s: float, dest_wav: Path,
    sample_rate: int = 16000,
) -> Path:
    """Use ffmpeg to cut a section into a 16 kHz mono WAV file."""
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start_s):.3f}",
        "-t", f"{max(0.0, dur_s):.3f}",
        "-i", str(source_audio),
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
        str(dest_wav), "-y",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dest_wav.exists():
        raise RuntimeError(f"ffmpeg slice failed at {start_s:.2f}s: {proc.stderr[-500:]}")
    return dest_wav


def media_duration_seconds(media_path: Path) -> Optional[float]:
    """Return the media duration in seconds, or None if ffprobe fails."""
    try:
        return wav_duration_seconds(media_path)
    except Exception:  # noqa: BLE001
        return None


def wav_duration_seconds(wav_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr[-300:]}")
    return float(proc.stdout.strip())
