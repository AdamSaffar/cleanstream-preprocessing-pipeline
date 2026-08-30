"""
orchestrator
=====================

This script watches an inbox folder and processes each title through the full
pipeline. Source folders use the name <netflix_id>_<tmdb_id> and contain video,
audio, and subtitle files.

For each title, the script checks the source files, reads queue metadata, builds
mutes and skips, creates a filter JSON file, and verifies the output. Failed
folders are moved to failed/ so their source files can be checked later.

The GPU models are loaded once and reused for all titles. Batch mode processes
the current inbox once; watch mode keeps checking for new folders.

Example (Colab):
    !pip install -q git+https://github.com/huggingface/transformers accelerate
    !pip install -q "qwen-vl-utils[decord]" pillow qwen-asr
    import sys; sys.path.insert(0, "/content/drive/MyDrive")
    from cleanstream_orchestrator import run
    run(root="/content/drive/MyDrive/CleanStream")
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

SCHEMA_VERSION = 1

# --- settings that can be changed through run() or the CLI ---
DURATION_MISMATCH_TOL_S = 5.0      # Maximum allowed audio/video duration difference.
STABILITY_WAIT_S = 5.0             # Wait before checking whether a synced folder changed.
DEFAULT_COARSE_INTERVAL_S = 4.0    # Frame interval when the queue has no value.
SUBTITLE_EXTS = {".ttml", ".xml", ".dfxp", ".vtt", ".srt"}
VIDEO_HINTS = ("video", "vid")     # Used only when ffprobe cannot identify the video.
AUDIO_HINTS = ("audio", "aud")
WATCH_POLL_S = 15.0                # Time between inbox checks in watch mode.


# --- utilities ---

def _log(msg: str, log_file: Optional[Path] = None) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    if log_file is not None:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def ffprobe_duration(path: Path) -> Optional[float]:
    """Return the media duration in seconds, or None if ffprobe fails."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return None


def has_video_stream(path: Path) -> bool:
    """Return whether a file has at least one video stream."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60)
        return "video" in out.stdout.lower()
    except subprocess.SubprocessError:
        return False


def video_codec(path: Path) -> str:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60)
        return out.stdout.strip().lower()
    except subprocess.SubprocessError:
        return ""


def video_height(path: Path) -> int:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60)
        return int(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 0


# --- folder handling ---

def parse_folder_name(name: str) -> Tuple[str, str]:
    """Read Netflix and TMDB IDs from a folder name.

    Examples:
      "81188617_66049"      -> ("81188617", "66049")
      "81188617"            -> ("81188617", "")      (tmdb missing; warn later)
      "81188617_66049_x"    -> ("81188617", "66049") (extra suffix ignored)
    """
    parts = re.findall(r"\d+", name)
    netflix_id = parts[0] if len(parts) >= 1 else ""
    tmdb_id = parts[1] if len(parts) >= 2 else ""
    return netflix_id, tmdb_id


def identify_source_files(folder: Path) -> Dict[str, Path]:
    """Find the video, audio, and subtitle files in a title folder.

    ffprobe is used to tell video and audio files apart. File names are used as
    a fallback when needed.
    """
    files = [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")]
    subtitle = next((p for p in files if p.suffix.lower() in SUBTITLE_EXTS), None)
    media = [p for p in files if p.suffix.lower() not in SUBTITLE_EXTS]

    if subtitle is None:
        raise ValueError(f"no subtitle file (expected one of {sorted(SUBTITLE_EXTS)})")
    if len(media) < 2:
        raise ValueError(f"need a video AND an audio file; found media: "
                         f"{[p.name for p in media]}")

    with_video = [p for p in media if has_video_stream(p)]
    without_video = [p for p in media if p not in with_video]

    video = audio = None
    if len(with_video) == 1 and len(without_video) >= 1:
        video = with_video[0]
        audio = without_video[0]
    else:
        # Use file-name hints if ffprobe could not decide.
        def hinted(cands, hints):
            return next((p for p in cands
                         if any(h in p.stem.lower() for h in hints)), None)
        video = hinted(media, VIDEO_HINTS)
        audio = hinted(media, AUDIO_HINTS)
        if video is None or audio is None or video == audio:
            raise ValueError(
                "could not tell video from audio: "
                f"{[p.name for p in media]} "
                f"(video-stream count={len(with_video)}). "
                "Name them with 'video'/'audio' in the filename to disambiguate.")
    return {"video": video, "audio": audio, "subtitle": subtitle}


def folder_signature(folder: Path) -> Dict[str, int]:
    """Return file names and sizes for the folder stability check."""
    sig = {}
    try:
        for p in folder.iterdir():
            try:
                if p.is_file() and not p.name.startswith("."):
                    sig[p.name] = p.stat().st_size
            except FileNotFoundError:
                continue
    except FileNotFoundError:
        return {}
    return sig


def is_folder_ready(folder: Path, stability_wait_s: float = STABILITY_WAIT_S) -> bool:
    """Return whether the folder has the needed files and is done syncing."""
    try:
        # Check that the expected source files are present.
        identify_source_files(folder)
    except (ValueError, FileNotFoundError):
        return False
    sig1 = folder_signature(folder)
    if not sig1 or any(sz == 0 for sz in sig1.values()):
        return False
    time.sleep(stability_wait_s)
    sig2 = folder_signature(folder)
    return sig1 == sig2


# --- metadata lookup ---

def load_queue(config_dir: Path) -> Dict[str, dict]:
    """Load titles_queue.json as a map keyed by TMDB ID."""
    qp = config_dir / "titles_queue.json"
    if not qp.exists():
        return {}
    try:
        data = json.loads(qp.read_text(encoding="utf-8"))
        rows = data.get("titles", data if isinstance(data, list) else [])
        return {str(r.get("tmdb_id")): r for r in rows if r.get("tmdb_id") is not None}
    except (json.JSONDecodeError, OSError):
        return {}


def metadata_for(tmdb_id: str, queue: Dict[str, dict]) -> dict:
    """Return title metadata, using defaults if it is not in the queue."""
    row = queue.get(str(tmdb_id), {})
    return {
        "title": row.get("title", ""),
        "episode": row.get("episode", ""),
        "genre": row.get("genre", ""),
        "certification": row.get("certification", row.get("rating_band", "")),
        "original_language": row.get("original_language", ""),
        "runtime_ms": row.get("runtime_ms", 0),
        "coarse_interval_s": row.get("coarse_interval_s", DEFAULT_COARSE_INTERVAL_S),
        "run_skip_vlm": row.get("run_skip_vlm", True),
        "sex_nudity_hint": row.get("sex_nudity_hint"),  # "none" means the VLM scan can be skipped.
        "skip": bool(row.get("skip", False)),  # Skip this title completely.
        "poster_url": row.get("poster_url", ""),
        "backdrop_url": row.get("backdrop_url", ""),
        "overview": row.get("overview", ""),
        "vote_average": row.get("vote_average"),
        "release_date": row.get("release_date", ""),
        "_in_queue": bool(row),
    }


# -- Stage 0: validation ---

def stage0_validate(folder: Path, tol_s: float, log_file: Optional[Path],
                    transcode_dir: Optional[Path] = None) -> Dict:
    """Identify source files and check that audio and video durations match."""
    srcs = identify_source_files(folder)
    v_dur = ffprobe_duration(srcs["video"])
    a_dur = ffprobe_duration(srcs["audio"])
    if v_dur is None or a_dur is None:
        raise ValueError(f"ffprobe couldn't read durations "
                         f"(video={v_dur}, audio={a_dur})")

    delta = abs(v_dur - a_dur)
    _log(f"   Stage 0: video={v_dur:.1f}s audio={a_dur:.1f}s (Δ={delta:.2f}s)", log_file)
    if delta > tol_s:
        raise ValueError(
            f"audio/video duration mismatch Δ={delta:.1f}s > {tol_s}s — likely a "
            f"wrong-file pairing (video={srcs['video'].name}, audio={srcs['audio'].name}). "
            "Aborting to avoid a misaligned filter.")

    video_path = srcs["video"]
    # Transcode only when requested and the video needs it.
    if transcode_dir is not None:
        codec = video_codec(video_path)
        height = video_height(video_path)
        if codec not in ("h264", "") or (height and height > 720):
            transcode_dir.mkdir(parents=True, exist_ok=True)
            out_v = transcode_dir / (video_path.stem + "_h264_720p.mp4")
            _log(f"   Stage 0: transcoding {codec} {height}p -> H.264 720p", log_file)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                   "-i", str(video_path), "-vf", "scale=-2:720",
                   "-c:v", "libx264", "-crf", "20", "-preset", "fast",
                   "-an", str(out_v)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and out_v.exists():
                video_path = out_v
            else:
                _log(f"   Stage 0: transcode failed, using original "
                     f"({r.stderr[-200:]})", log_file)

    return {
        "video": video_path,
        "audio": srcs["audio"],
        "subtitle": srcs["subtitle"],
        "audio_duration_ms": int(round(a_dur * 1000.0)),
        "video_duration_ms": int(round(v_dur * 1000.0)),
    }


# --- Stage 4: filter assembly ---


def vlm_decision(meta: dict, do_skips: bool) -> Tuple[bool, str]:
    """Decide whether to run the VLM scan and return the reason."""
    if not do_skips:
        return False, "vlm_disabled_globally"          # --no-skips was used.
    if meta.get("run_skip_vlm") is False:
        return False, "run_skip_vlm=false"             # Per-title scan setting is off.
    hint = meta.get("sex_nudity_hint")
    if isinstance(hint, str) and hint.strip().lower() == "none":
        # The title was checked manually, so skip the scan.
        return False, "sex_nudity_hint=none (verified clean)"
    return True, "scanned"                              # Run the VLM by default.


def assemble_filter(netflix_id: str, tmdb_id: str, meta: dict,
                    mutes: dict, skips: dict, stage0: dict) -> dict:
    """Combine metadata, captions, mutes, and skips into a filter JSON object."""
    return {
        "schema_version": SCHEMA_VERSION,
        "movieID": netflix_id,          # Field read by the current engine.
        "netflix_id": netflix_id,       # Main ID field for the filter.
        "tmdb_id": tmdb_id,
        "title": meta.get("title", ""),
        "episode": meta.get("episode", ""),
        "genre": meta.get("genre", ""),
        "certification": meta.get("certification", ""),
        "original_language": meta.get("original_language", ""),
        # Display duration and measured media durations.
        "duration_ms": meta.get("runtime_ms", 0),            # From TMDB.
        "audio_duration_ms": stage0["audio_duration_ms"],    # From ffprobe.
        "video_duration_ms": stage0["video_duration_ms"],    # From ffprobe.
        "duration_source": "ffprobe",
        "coarse_interval_s": skips.get("coarse_interval_s"),
        "scene_scan": skips.get("scene_scan", "scanned"),  # Scan result or skip reason.
        # All content timestamps use milliseconds.
        "captions": mutes.get("captions", []),
        "mute_windows": mutes.get("mute_windows", []),
        "skip_ranges": skips.get("skip_ranges", []),
        # Source information and display metadata.
        "_mute_source": mutes.get("mute_source"),
        "_scene_source": skips.get("scene_source"),
        "num_flagged_cues": mutes.get("num_flagged_cues"),
        "num_flagged_frames": skips.get("num_flagged_frames"),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "poster_url": meta.get("poster_url", ""),
        "backdrop_url": meta.get("backdrop_url", ""),
        "overview": meta.get("overview", ""),
        "vote_average": meta.get("vote_average"),
        "release_date": meta.get("release_date", ""),
    }


def verify_filter(path: Path, require_captions: bool = True) -> Tuple[bool, str]:
    """Read the output again and check that the expected lists are present."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return False, f"unreadable/invalid JSON: {e}"
    for key in ("captions", "mute_windows", "skip_ranges"):
        if key not in obj or not isinstance(obj[key], list):
            return False, f"missing/invalid '{key}'"
    if require_captions and len(obj["captions"]) == 0:
        return False, "captions empty (likely a parse failure)"
    return True, "ok"


# --- status tracking ---

def update_status(output_dir: Path, netflix_id: str, entry: dict) -> None:
    """Update output/_status.json without changing the queue file."""
    sp = output_dir / "_status.json"
    data = {}
    if sp.exists():
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data[str(netflix_id)] = entry
    try:
        sp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


# --- per-title processing ---

def process_folder(
    folder: Path,
    dirs: Dict[str, Path],
    queue: Dict[str, dict],
    profanity_path: Path,
    judge=None,
    engine=None,
    do_skips: bool = True,
    do_mutes: bool = True,
    tol_s: float = DURATION_MISMATCH_TOL_S,
    language_code: str = "en",
    transcode: bool = False,
    delete_on_success: bool = False,
) -> Dict:
    """Process one title folder and return its result status."""
    from cleanstream_build_mutes import build_mutes
    from cleanstream_build_skips_batched import build_skips

    log_file = dirs["logs"] / "orchestrator.log"
    netflix_id, tmdb_id = parse_folder_name(folder.name)
    _log(f"{folder.name}  (netflix={netflix_id or '?'}, tmdb={tmdb_id or '?'})", log_file)
    if not netflix_id:
        return _fail(folder, dirs, "?", "folder name has no netflix_id", log_file)

    # Move the folder so it is marked as being processed.
    work = dirs["processing"] / folder.name
    try:
        if folder.parent != dirs["processing"]:
            if work.exists():
                shutil.rmtree(work, ignore_errors=True)
            shutil.move(str(folder), str(work))
    except OSError as e:
        return _fail(folder, dirs, netflix_id, f"couldn't move to processing: {e}", log_file)

    t0 = time.time()
    try:
        # Stage 0: validate the source files.
        tdir = (work / "_transcode") if transcode else None
        stage0 = stage0_validate(work, tol_s, log_file, transcode_dir=tdir)

        # Stage 1: load metadata.
        meta = metadata_for(tmdb_id, queue)
        if not meta["_in_queue"]:
            _log(f"tmdb {tmdb_id or '(none)'} not in queue; using defaults "
                 f"(coarse={meta['coarse_interval_s']}s).", log_file)
        # Do not process titles marked as skipped in the queue.
        if meta.get("skip"):
            _log(f"SKIP flagged in queue for tmdb {tmdb_id} — not processing.", log_file)
            skipped_dir = dirs["failed"].parent / "skipped"
            skipped_dir.mkdir(parents=True, exist_ok=True)
            dest = skipped_dir / folder.name
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            try:
                shutil.move(str(work), str(dest))
            except OSError:
                pass
            update_status(dirs["output"], netflix_id, {
                "status": "skipped", "tmdb_id": tmdb_id,
                "at": datetime.now(timezone.utc).isoformat()})
            return {"status": "skipped", "netflix_id": netflix_id}
        coarse = meta["coarse_interval_s"]

        # Stage 2: build mutes and captions.
        if do_mutes:
            m = build_mutes(
                audio_path=str(stage0["audio"]),
                ttml_path=str(stage0["subtitle"]),
                profanity_path=str(profanity_path),
                language_code=language_code,
                engine=engine,
                watch_id=netflix_id,
                verbose=True,
            )
        else:
            m = {"captions": [], "mute_windows": [], "num_flagged_cues": 0,
                 "mute_source": None}

        # Stage 3: build skip ranges when the VLM scan is enabled.
        run_vlm, skip_reason = vlm_decision(meta, do_skips)
        if run_vlm:
            s = build_skips(
                video_path=str(stage0["video"]),
                coarse_interval_s=coarse,
                judge=judge,
                verbose=True,
            )
            s["scene_scan"] = "scanned"
        else:
            _log(f"VLM skipped: {skip_reason}", log_file)
            s = {"skip_ranges": [], "coarse_interval_s": coarse,
                 "scene_source": None, "num_flagged_frames": 0,
                 "scene_scan": skip_reason}

        # Stage 4: assemble the filter.
        filt = assemble_filter(netflix_id, tmdb_id, meta, m, s, stage0)

        # Stage 5: write and verify the output.
        out_path = dirs["output"] / f"filter_{netflix_id}.json"
        out_path.write_text(json.dumps(filt, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        ok, reason = verify_filter(out_path)
        if not ok:
            out_path.unlink(missing_ok=True)
            raise RuntimeError(f"output verification failed: {reason}")

        elapsed = time.time() - t0
        _log(f"{out_path.name}: caps={len(filt['captions'])} "
             f"mutes={len(filt['mute_windows'])} skips={len(filt['skip_ranges'])} "
             f"({elapsed:.0f}s)", log_file)

        # Remove source files only after the output passes verification.
        if delete_on_success:
            shutil.rmtree(work, ignore_errors=True)
        else:
            done = dirs["output"] / f"_src_{folder.name}"
            shutil.move(str(work), str(done))

        update_status(dirs["output"], netflix_id, {
            "status": "done", "tmdb_id": tmdb_id, "title": meta.get("title", ""),
            "captions": len(filt["captions"]), "mutes": len(filt["mute_windows"]),
            "skips": len(filt["skip_ranges"]), "elapsed_s": round(elapsed, 1),
            "at": datetime.now(timezone.utc).isoformat(),
        })
        return {"status": "done", "netflix_id": netflix_id, "filter": out_path.name,
                "captions": len(filt["captions"]), "mutes": len(filt["mute_windows"]),
                "skips": len(filt["skip_ranges"])}

    except Exception as e:
        tb = traceback.format_exc()
        return _fail(work, dirs, netflix_id, f"{type(e).__name__}: {e}", log_file, tb)


def _fail(folder: Path, dirs: Dict[str, Path], netflix_id: str, reason: str,
          log_file: Optional[Path], tb: str = "") -> Dict:
    """Move a failed folder and record its error."""
    _log(f"FAILED: {reason}", log_file)
    if tb:
        try:
            (dirs["logs"] / f"error_{netflix_id or folder.name}.log").write_text(
                tb, encoding="utf-8")
        except OSError:
            pass
    try:
        if folder.exists() and folder.parent != dirs["failed"]:
            dest = dirs["failed"] / folder.name
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            shutil.move(str(folder), str(dest))
    except OSError as e:
        _log(f"   (couldn't move to failed/: {e})", log_file)
    update_status(dirs["output"], netflix_id, {
        "status": "failed", "reason": reason,
        "at": datetime.now(timezone.utc).isoformat(),
    })
    return {"status": "failed", "netflix_id": netflix_id, "reason": reason}


# --- main entry ---

def _dirs(root: Path) -> Dict[str, Path]:
    d = {name: root / name for name in
         ("inbox", "processing", "output", "logs", "config", "failed")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _load_models(do_skips: bool, do_mutes: bool):
    """Load the requested GPU models and return the judge and aligner."""
    judge = engine = None
    if do_mutes:
        from cleanstream_build_mutes import Qwen3AlignmentEngine
        engine = Qwen3AlignmentEngine()
        engine.load()
    if do_skips:
        from cleanstream_build_skips_batched import QwenVLBatchedJudge
        judge = QwenVLBatchedJudge()
        judge.load()
    return judge, engine


def run(root: str, watch: bool = False, do_skips: bool = True, do_mutes: bool = True,
        tol_s: float = DURATION_MISMATCH_TOL_S, language_code: str = "en",
        transcode: bool = False, delete_on_success: bool = False,
        load_models: bool = True) -> None:
    """Process the inbox once or keep watching for new folders.

    Set load_models to False to test the folder flow without loading the GPU
    models.
    """
    root_p = Path(root)
    dirs = _dirs(root_p)

    # Return folders left in processing/ after a previous interruption.
    try:
        stranded = [p for p in dirs["processing"].iterdir() if p.is_dir()]
    except FileNotFoundError:
        stranded = []
    for sp in stranded:
        try:
            dest = dirs["inbox"] / sp.name
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            shutil.move(str(sp), str(dest))
            _log(f"recovered stranded folder from processing/: {sp.name}")
        except OSError as e:
            _log(f"  (couldn't recover {sp.name}: {e})")

    profanity = dirs["config"] / "profanity.txt"
    if not profanity.exists():
        _log(f"profanity list not found at {profanity} — mutes will error. "
             "Put profanity_flat.txt there as profanity.txt.")
    queue = load_queue(dirs["config"])
    _log(f"CleanStream orchestrator @ {root_p}")
    _log(f"  queue: {len(queue)} titles | mode: {'watch' if watch else 'batch'} | "
         f"skips={do_skips} mutes={do_mutes}")

    judge = engine = None
    if load_models and (do_skips or do_mutes):
        _log("Loading models (once)")
        judge, engine = _load_models(do_skips, do_mutes)

    def drain_once() -> int:
        try:
            folders = sorted([p for p in dirs["inbox"].iterdir() if p.is_dir()])
        except FileNotFoundError:
            # The inbox may briefly be unavailable while Drive is syncing.
            _log("inbox not available this scan (Drive hiccup); will retry.")
            return 0
        results = []
        for folder in folders:
            # A synced folder can disappear between listing and processing.
            try:
                if not folder.exists():
                    continue                      # It disappeared after listing.
                # Ignore empty folders.
                try:
                    if not any(folder.iterdir()):
                        continue
                except FileNotFoundError:
                    continue
                if not is_folder_ready(folder):
                    _log(f"{folder.name}: not ready (incomplete or still syncing); skipping.")
                    continue
                res = process_folder(
                    folder, dirs, queue, profanity, judge=judge, engine=engine,
                    do_skips=do_skips, do_mutes=do_mutes, tol_s=tol_s,
                    language_code=language_code, transcode=transcode,
                    delete_on_success=delete_on_success)
                results.append(res)
            except FileNotFoundError as e:
                _log(f"{folder.name}: vanished mid-scan ({e}); skipping.")
                continue
            except Exception as e:
                # Log an unexpected folder error and continue with the next one.
                _log(f"{folder.name}: unexpected error ({type(e).__name__}: {e}); skipping.")
                continue
        # Print a summary after the batch.
        if results:
            done = [r for r in results if r.get("status") == "done"]
            failed = [r for r in results if r.get("status") == "failed"]
            skipped = [r for r in results if r.get("status") == "skipped"]
            _log("")
            _log("=" * 60)
            _log(f"BATCH SUMMARY: {len(done)} done, {len(failed)} failed, {len(skipped)} skipped")
            _log("=" * 60)
            for r in done:
                _log(f"{r['netflix_id']}: caps={r.get('captions')} "
                     f"mutes={r.get('mutes')} skips={r.get('skips')}")
            for r in failed:
                _log(f"{r['netflix_id']}: {r.get('reason','')[:70]}")
        return len(results)

    if not watch:
        done = drain_once()
        _log(f"Batch complete: processed {done} folder(s).")
        return

    _log(f"Watching {dirs['inbox']} (poll every {WATCH_POLL_S:.0f}s). Ctrl+C to stop.")
    try:
        while True:
            drain_once()
            time.sleep(WATCH_POLL_S)
    except KeyboardInterrupt:
        _log("Watch stopped.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CleanStream orchestrator")
    ap.add_argument("--root", required=True, help="CleanStream root folder")
    ap.add_argument("--watch", action="store_true", help="stay running")
    ap.add_argument("--no-skips", action="store_true", help="skip the VLM stage")
    ap.add_argument("--no-mutes", action="store_true", help="skip the aligner stage")
    ap.add_argument("--tol", type=float, default=DURATION_MISMATCH_TOL_S)
    ap.add_argument("--language", default="en")
    ap.add_argument("--transcode", action="store_true",
                    help="transcode odd/large videos to H.264 720p")
    ap.add_argument("--delete-sources", action="store_true",
                    help="delete source files after successful processing")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the flow without loading GPU models")
    a = ap.parse_args()
    run(root=a.root, watch=a.watch, do_skips=not a.no_skips, do_mutes=not a.no_mutes,
        tol_s=a.tol, language_code=a.language, transcode=a.transcode,
        delete_on_success=a.delete_sources, load_models=not a.dry_run)