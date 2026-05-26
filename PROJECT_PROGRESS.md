# CudaVox-Transcriber Project Progress

Last updated: 2026-05-26

## Current State

The project is a Chinese speech transcription and speaker diarization pipeline based on `Qwen3-ASR`, `Qwen3.6`, `pyannote.audio`, and `CAM++`.

The main workflow is already functional:

- Normalize input audio to `16kHz / mono / wav`.
- Run speaker diarization with `pyannote/speaker-diarization-community-1`.
- Merge adjacent same-speaker diarization segments before ASR.
- Dictate Chinese speech with `Qwen3-ASR-1.7B`.
- Summarize transcript text with `Qwen3.6-27B`; per-segment refinement is disabled by default for speed.
- Extract speaker embeddings with `CAM++`.
- Persist and reuse speaker IDs across audio files.
- Export `json`, `txt`, and `srt` results.
- Run standalone workflow scripts for audio transcription and SRT-based audio cutting.
- Export voiceprint review samples from historical JSON outputs.

## Recent Changes

- Unified `.flake8` ignore settings with the broader project convention.
- Moved logging configuration out of the project root into `FunASRNano/logging_config.py`.
- Added `FunASRNano/logging_utils.py` as a project compatibility wrapper.
- Preserved this project's existing log behavior:
  - logs are written under the repository `log/` directory;
  - callers can still request one-time log reset behavior through `setup_project_logger(..., reset_log=True)`.
- Updated imports to use `FunASRNano.logging_config` and `FunASRNano.logging_utils`.
- Removed the root-level `logging_config.py` reference from `pyrightconfig.json`.
- Updated `README.md` with the current logging structure.
- Replaced the FunASR transcription backend with a Qwen local-model backend.
- Added `FunASRNano/qwen_service.py` for OpenAI-compatible local API calls.
- Added JSON metadata for `dictation_model`, `text_model`, and optional `summary`.
- Optimized Qwen runtime path:
  - disabled per-segment Qwen3.6 text refinement by default;
  - lowered refinement/summary token limits;
  - capped whole-file summary input at 5000 characters to stay under the 8192-token local context;
  - merged diarization segments before cutting audio and calling Qwen3-ASR;
  - preserved raw diarization segments with `extras.transcribed_segment_index`.
- Added standalone workflow scripts:
  - `scripts/transcribe_audio.py` for audio-to-`TranscriptDocument` outputs;
  - `scripts/cut_audio_by_srt.py` for SRT timeline based audio cutting and CSV manifests.
- Split `CudaVoxPipeline.transcribe_file(...)` from `process_file(...)`, so callers can obtain a `TranscriptDocument` before deciding how to write or post-process it.

## Validation

The following checks are used for the current branch:

```powershell
python -m compileall FunASRNano scripts main.py
python main.py --help
python -m flake8 FunASRNano scripts main.py
```

Current local validation notes:

- `compileall` passed with the `cudavox` conda environment.
- `scripts/transcribe_audio.py --help` passed.
- `scripts/cut_audio_by_srt.py --help` passed.
- `scripts/cut_audio_by_srt.py` smoke test wrote 1 clip and `clips.csv` using an existing SRT timeline and the currently available local mp3.
- `flake8` is not installed in the current `cudavox` conda environment, so the lint command could not run there.
- `input/2026-04-13 09_46_37.mp3` completed successfully with Qwen3-ASR + Qwen3.6.
- The output JSON contains 66 merged ASR segments, 80 raw diarization segments, 0 empty final text segments, and a generated whole-file summary.

## Notes

- This document supersedes the earlier misspelled `PROJECET_PROGRESS.md` file.
- `logging_config.py` is not an entrypoint and should remain inside the package.
- New code should avoid importing `logging_config` from the repository root.
- The main executable entrypoints remain `main.py` and `python -m FunASRNano`.
- For Qwen speed-sensitive runs, keep `QWEN_ENABLE_TEXT_REFINEMENT=false` and reserve Qwen3.6 for whole-file summary.
- If throughput becomes more important than Qwen-only behavior, keep a fast ASR backend such as FunASR or SenseVoice as an optional path.

## Performance Baseline

- Before optimization, `input/2026-04-13 09_46_37.mp3` took about 24 minutes with Qwen3-ASR plus per-segment Qwen3.6 refinement.
- After optimization, the same input completed in 608.20 seconds, about 10.14 minutes.
- The optimized run used 66 Qwen3-ASR requests after merging 80 raw diarization segments, with `QWEN_ENABLE_TEXT_REFINEMENT=false`.

## Next Candidates

- Add a true transcription-only workflow mode that skips diarization and voiceprint matching when speaker attribution is not needed.
- Add focused tests around configuration loading, transcript serialization, and logging setup compatibility.
