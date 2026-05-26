# CudaVox-Transcriber Project Progress

Last updated: 2026-05-26

## Current State

The project is a Chinese speech transcription and speaker diarization pipeline based on `Qwen3-ASR`, `Qwen3.6`, `pyannote.audio`, and `CAM++`.

The main workflow is already functional:

- Normalize input audio to `16kHz / mono / wav`.
- Run speaker diarization with `pyannote/speaker-diarization-community-1`.
- Split per-segment temporary audio.
- Dictate Chinese speech with `Qwen3-ASR-1.7B`.
- Refine and summarize transcript text with `Qwen3.6-27B`.
- Extract speaker embeddings with `CAM++`.
- Persist and reuse speaker IDs across audio files.
- Export `json`, `txt`, and `srt` results.
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

## Validation

The following checks passed after the logging move and README update:

```powershell
python -m compileall FunASRNano scripts main.py
python main.py --help
python -m flake8 FunASRNano scripts main.py
```

## Notes

- The file name is intentionally `PROJECET_PROGRESS.md` to match the requested artifact name.
- `logging_config.py` is not an entrypoint and should remain inside the package.
- New code should avoid importing `logging_config` from the repository root.
- The main executable entrypoints remain `main.py` and `python -m FunASRNano`.

## Next Candidates

- Add an independent `cut_audio_by_srt.py` workflow using existing SRT parsing and audio cutting helpers.
- Add a standalone transcription-only workflow for cases that do not need diarization or voiceprint matching.
- Add focused tests around configuration loading, transcript serialization, and logging setup compatibility.
