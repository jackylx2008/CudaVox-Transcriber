# TODO

## Qwen3-ASR Runtime Findings

- Current `llama.cpp + Qwen3-ASR GGUF` ASR path is slow mainly because the pipeline sends many small, serial HTTP requests.
- The `2026-05-25 16_14_10.mp3` run produced 784 raw diarization segments and 617 merged ASR requests before interruption.
- GPU-Z may show low load because the Qwen3-ASR main model can be offloaded to GPU, while the audio encoder path logged `clip_ctx: CLIP using CPU backend`.
- ASR server logs showed repeated per-request fixed cost:
  - audio encode/decode around 1.0-1.2 seconds per segment;
  - prompt eval around 1.1-1.5 seconds per segment;
  - occasional segments hit the previous 1024-token output cap and took much longer.
- The `llama-server --list-devices` check did not list CUDA devices even though `ggml-cuda.dll` exists, so CUDA backend loading should be verified separately.
- The 27B model is not needed during ASR when `QWEN_ENABLE_TEXT_REFINEMENT=false`; it is only needed for whole-file summary when `QWEN_ENABLE_SUMMARY=true`.

## Active Optimization Plan

- Reduce ASR segment count:
  - merge by resolved `speaker_id` / `speaker_name` instead of only pyannote local labels;
  - increase merge gap to 2.0 seconds;
  - absorb short same-speaker segments when possible;
  - cap merged ASR segment duration to avoid overly long requests.
- Lower ASR output token cap from 1024 to 256.
- Disable llama.cpp prompt cache for ASR server startup with `--cache-ram 0`.
- Add ASR request concurrency so multiple llama-server slots can work in parallel.
- Run audio cutting inside the ASR worker so cutting and HTTP requests overlap instead of serializing all cutting first.
- Keep `QWEN_ENABLE_SUMMARY=false` for speed tests; run 27B summary separately later if needed.

## 2026-05-27 Optimization Result

- `input/2026-05-25 16_14_10.mp3` completed successfully with Qwen3-ASR-only settings.
- Total elapsed time was 1119.8 seconds, about 18.66 minutes.
- Raw diarization segments stayed at 784; merged Qwen3-ASR target segments dropped to 545.
- Output contained 0 empty text segments.
- Runtime settings were `QWEN_ASR_MAX_TOKENS=256`, `QWEN_ASR_CONCURRENCY=3`, `QWEN_ENABLE_TEXT_REFINEMENT=false`, and `QWEN_ENABLE_SUMMARY=false`.
- The ASR `llama-server` was started with `--cache-ram 0`; Qwen3.6-27B was not required for this ASR-only run.

## Remaining Follow-ups

- Verify whether the local `llama-server` build is actually using CUDA for Qwen3-ASR beyond loading `ggml-cuda.dll`.
- Check whether Qwen3-ASR can use a GPU audio encoder path in the available llama.cpp build; current logs still show `clip_ctx: CLIP using CPU backend`.
- If higher throughput is required, keep FunASR or SenseVoice as an optional fast ASR backend and reserve Qwen3-ASR for cases that need its flexibility.
