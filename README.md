# CleanStream preprocessing pipeline

CleanStream is an offline preprocessing pipeline for building timestamped
content filters from locally available video, audio, and subtitle files. It
creates a filter JSON file containing:

- captions, preserved for display in the app;
- mute windows for configured words or phrases; and
- skip ranges for scenes identified by a vision-language model.

The pipeline is meant to run before playback. It does not modify the source
video. Its output is a compact JSON file that a separate client application can
read during playback.

## How it works

The pipeline takes a local video file, audio file, and subtitle file for each
title. The orchestrator coordinates the work and produces one filter JSON file
containing captions, mute windows, and visual skip ranges.

```text
Video + audio + subtitles
            ↓
Caption and word alignment
            ↓
Visual scene detection
            ↓
Filter JSON for the client application
```

### Captions and mute windows

The mute stage parses the subtitle file, checks its spoken text against a local
word list, and aligns matching sections with the audio using Qwen3 Forced
Aligner. It returns caption data and mute windows in milliseconds.

The word list is not limited to profanity. It can contain any words or phrases
that the user wants to detect and mute.

### Why Qwen3 Forced Aligner?

The mute problem is mostly an alignment problem, not a transcription problem:
the subtitle text is already available, but the pipeline needs reliable word
boundaries in the audio so it can mute the correct moment.

I tested WhisperX, Stable-TS, and Qwen3 Forced Aligner in a small benchmark.
The comparison focused on word-level timing, runtime, failed clips, and
transcription errors. On the two matched Qwen3-versus-WhisperX clips, Qwen3
was faster and had lower typical boundary error:

| Test clip | Qwen3 inference | WhisperX inference | Qwen3 median boundary error | WhisperX median boundary error | Qwen3 p95 boundary error | WhisperX p95 boundary error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s0102a | 9.35 s | 14.75 s | 37 ms | 61 ms | 155 ms | 579 ms |
| s0102b | 8.17 s | 11.29 s | 38 ms | 53 ms | 118 ms | 180 ms |

Stable-TS was not selected because it was much slower in this test and had clip
failures on both of its runs. WhisperX produced excellent text accuracy, but the
Qwen3 results were a better fit for this pipeline because timing precision and
speed matter more than generating a fresh transcript. The tests also showed a
small number of Qwen3 hallucinations and rare large timestamp outliers, so the
pipeline uses padded, merged windows instead of treating a single model boundary
as exact.

### Visual skip ranges

The skip path samples frames from the local video and evaluates them with
Qwen2.5-VL. `cleanstream_build_skips_batched.py` processes several frames in a
batch to use the GPU more efficiently.

The model returns a small JSON verdict for each frame: `flag`, `severity`,
`confidence`, and `reason`. Those frame-level decisions are combined into skip
ranges for the final filter.

The public version intentionally contains a policy placeholder in
`DETECTION_PROMPT`. Before running scene detection, replace that placeholder
with rules that reflect the categories you want to identify. These choices are
subjective, so the repository does not present one policy as a universal default.

### Filter output

The orchestrator combines captions, mute windows, skip ranges, media durations,
and optional metadata into `filter_<netflix_id>.json`.

Successful source folders are preserved by default and moved to
`output/_src_<folder-name>`. To delete source folders only after successful
verification, pass `--delete-sources`.

## Installation

### Prerequisites

- Python 3.10 or newer
- `ffmpeg` and `ffprobe` installed and available on your `PATH`
- A CUDA-capable GPU is strongly recommended for the Qwen models
- Git, because the current `requirements.txt` installs Transformers from its
  upstream repository

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Create a `config` folder in the pipeline root. The mute stage requires a local
word list at `config/profanity.txt`:

```text
# Add one word or phrase per line.
# The list is not limited to profanity. It can include any word or phrase that
# you want the pipeline to detect and mute.
# Lines beginning with # are ignored.
```

### Optional title queue

`config/titles_queue.json` is optional. Without it, the pipeline uses default
metadata and sampling settings. When present, it can provide title metadata and
per-title controls such as `coarse_interval_s`, `run_skip_vlm`, and `skip`.

## Usage

Choose a working root outside the repository. The orchestrator creates its
runtime folders there:

```text
CleanStream-work/
├── config/
│   └── profanity.txt
└── inbox/
    └── <netflix_id>_<tmdb_id>/
        ├── video file
        ├── audio file
        └── subtitle file
```

Run one batch:

```powershell
python cleanstream_orchestrator.py --root "C:\path\to\CleanStream-work"
```

Useful options:

```powershell
# Keep watching for new folders.
python cleanstream_orchestrator.py --root "C:\path\to\CleanStream-work" --watch

# Run only the mute/caption stages.
python cleanstream_orchestrator.py --root "C:\path\to\CleanStream-work" --no-skips

# Run only the visual skip stage.
python cleanstream_orchestrator.py --root "C:\path\to\CleanStream-work" --no-mutes

# Delete source folders after verified output. This is off by default.
python cleanstream_orchestrator.py --root "C:\path\to\CleanStream-work" --delete-sources
```

