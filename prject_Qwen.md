# FunASR / SenseVoice + Qwen3.6 Hybrid Plan

## Goal

Use a dedicated ASR backend as the default fast and accurate dictation layer, and use Qwen3.6 only for language understanding, correction, summarization, and structured output.

Recommended default pipeline:

1. Audio normalization: `mp3/m4a/wav -> 16kHz mono wav`
2. Speaker diarization: `pyannote.audio`
3. Segment merge for ASR efficiency
4. Dictation: `FunASR` by default, optionally `SenseVoice`
5. Speaker voiceprint reuse: `CAM++`
6. Text post-processing: `Qwen3.6-27B`
7. Export: `json / txt / srt / structured summary`

This keeps the strongest part of each model:

- `FunASR` / `SenseVoice`: audio recognition, Chinese speech accuracy, fast throughput
- `Qwen3.6`: context-aware correction, meeting notes, summaries, action items, structured extraction

## Why Not Default to Qwen3-ASR

The recent comparison on `input/2026-05-25 16_14_10.mp3` showed that the main-branch FunASR result is visibly better than the Qwen3-ASR result.

Reasons:

- Qwen3-ASR is more flexible, but the current local GGUF + llama.cpp path has high per-segment HTTP overhead.
- The audio encoder currently logs CPU backend behavior, so the runtime path is not ideal for throughput.
- Short diarization slices lose speech context, making ASR more error-prone.
- Dedicated ASR pipelines are trained and optimized for direct audio-to-text recognition.
- Qwen3.6 is better used after recognition, where it can see longer context and correct text conservatively.

## Target Architecture

Introduce an ASR backend abstraction:

```text
Audio file
  -> normalize audio
  -> diarize speakers
  -> merge ASR segments
  -> ASR backend
       - funasr
       - sensevoice
       - qwen_asr optional
  -> TranscriptDocument
  -> Qwen3.6 post-processing
       - text cleanup
       - meeting summary
       - action items
       - topic extraction
       - structured JSON
  -> json / txt / srt / summary.md
```

## Configuration Design

Add an explicit backend selector:

```yaml
asr:
  backend: ${ASR_BACKEND:-funasr}
  merge_gap_seconds: ${ASR_MERGE_GAP_SECONDS:-1.0}
  min_segment_seconds: ${ASR_MIN_SEGMENT_SECONDS:-2.0}
  max_segment_seconds: ${ASR_MAX_SEGMENT_SECONDS:-30.0}
```

Keep existing FunASR config:

```yaml
funasr:
  model: ${FUNASR_MODEL:-FunAudioLLM/Fun-ASR-Nano-2512}
  hub: ${FUNASR_HUB:-ms}
  batch_size_s: ${FUNASR_BATCH_SIZE_S:-120}
  language: ${FUNASR_LANGUAGE:-}
  itn: ${FUNASR_ITN:-true}
```

Add SenseVoice optional config:

```yaml
sensevoice:
  model: ${SENSEVOICE_MODEL:-iic/SenseVoiceSmall}
  hub: ${SENSEVOICE_HUB:-ms}
  language: ${SENSEVOICE_LANGUAGE:-zh}
  itn: ${SENSEVOICE_ITN:-true}
```

Add Qwen text post-processing config:

```yaml
qwen_text:
  enabled: ${QWEN_TEXT_ENABLED:-true}
  base_url: ${QWEN_LLM_BASE_URL:-http://127.0.0.1:8080/v1}
  model: ${QWEN_LLM_MODEL:-Qwen3.6-27B-Q4_K_M}
  enable_segment_cleanup: ${QWEN_ENABLE_SEGMENT_CLEANUP:-false}
  enable_summary: ${QWEN_ENABLE_SUMMARY:-true}
  enable_structured_output: ${QWEN_ENABLE_STRUCTURED_OUTPUT:-true}
  segment_cleanup_max_tokens: ${QWEN_SEGMENT_CLEANUP_MAX_TOKENS:-256}
  summary_max_tokens: ${QWEN_SUMMARY_MAX_TOKENS:-1024}
  summary_input_max_chars: ${QWEN_SUMMARY_INPUT_MAX_CHARS:-12000}
```

## Code Changes

### 1. Add ASR Interface

Create `FunASRNano/asr/base.py`:

```python
class AsrBackend(Protocol):
    def transcribe_segment(self, audio_path: Path, segment: TranscriptSegment) -> str:
        ...
```

Concrete backends:

- `FunASRNano/asr/funasr_backend.py`
- `FunASRNano/asr/sensevoice_backend.py`
- `FunASRNano/asr/qwen_asr_backend.py` optional

Pipeline should depend on the interface, not a specific implementation.

### 2. Preserve GPU Device Routing

FunASR and SenseVoice should use the project's existing device resolution path.

Current local environment check:

```text
torch.cuda.is_available() = True
device0 = NVIDIA GeForce RTX 5090 D v2
```

The current main-branch FunASR path already passes the resolved device into `AutoModel`:

```python
AutoModel(
    model=settings.funasr.model,
    hub=settings.funasr.hub,
    device=resolved_device,
)
```

Keep this behavior in the refactor. The new ASR backend interface must accept or store the resolved device, so every backend can make an explicit GPU/CPU decision.

Recommended backend constructor shape:

```python
class FunAsrBackend:
    def __init__(self, settings: FunASRSettings, device: str, logger) -> None:
        ...

class SenseVoiceBackend:
    def __init__(self, settings: SenseVoiceSettings, device: str, logger) -> None:
        ...
```

Expected device behavior:

- `FunASR`: use `device="cuda:0"` when CUDA is available.
- `SenseVoice`: use `device="cuda:0"` through FunASR `AutoModel` or ModelScope pipeline, depending on implementation.
- `Qwen3-ASR`: do not assume GPU is active just because `-ngl 999` is set; verify `llama-server --list-devices` and startup logs.

Important distinction:

- FunASR/SenseVoice use PyTorch CUDA and already detect the RTX GPU in the current `cudavox` environment.
- Qwen3-ASR currently uses a `llama.cpp` GGUF server path. Local checks showed `llama-server --list-devices` did not list CUDA devices, so Qwen3-ASR's audio encoder was still logging CPU backend behavior.

This means FunASR/SenseVoice are the safer GPU-backed ASR choices for this project.

GPU does not eliminate all CPU work:

- ffmpeg normalization and audio slicing are CPU-bound;
- Python orchestration and file I/O are CPU-bound;
- post-processing and output writing are CPU-bound;
- many tiny segments can underutilize the GPU because per-call overhead dominates.

Therefore, the refactor should combine GPU-backed ASR with segment merging and batching where possible.

### 3. Keep FunASR as Default

Default `ASR_BACKEND=funasr`.

Do not remove the existing `FunASRTranscriber`; wrap it behind the new backend interface first. This keeps the change low-risk.

### 4. Add SenseVoice as Optional Fast Backend

SenseVoice can be added as a second dedicated ASR path.

Expected use cases:

- better robustness on noisy speech;
- multilingual or code-switching audio;
- quick comparison against FunASR.

Use config to switch:

```powershell
$env:ASR_BACKEND="sensevoice"
python main.py --input ".\input\example.mp3"
```

Implementation note: verify SenseVoice GPU use with a small local run and watch both logs and GPU-Z. The code should log the selected backend and resolved device before model loading.

### 5. Move Qwen3.6 to Text Post-Processing

Create `FunASRNano/qwen_text_service.py` or reuse the Qwen service from the feature branch.

Post-processing modes:

- `cleanup`: correct obvious ASR typos, punctuation, numbers, names, and domain terms
- `summary`: summarize the full transcript
- `structured`: output agenda, decisions, risks, action items, speaker views

Default policy:

- `QWEN_ENABLE_SEGMENT_CLEANUP=false`
- `QWEN_ENABLE_SUMMARY=true`
- `QWEN_ENABLE_STRUCTURED_OUTPUT=true`

Reason: segment-by-segment cleanup is expensive and can change meaning if there is too little context. Whole-file post-processing gives Qwen3.6 more context and fewer requests.

### 6. Add Standalone Workflow Scripts

Avoid making `pipeline.py` larger.

Recommended scripts:

- `scripts/transcribe_audio.py`
  - audio -> `TranscriptDocument`
  - uses selected ASR backend
- `scripts/summarize_transcript.py`
  - existing JSON/TXT/SRT -> Qwen3.6 summary and structured JSON
- `scripts/compare_asr_backends.py`
  - run the same input with FunASR and SenseVoice for quality/runtime comparison
- `scripts/cut_audio_by_srt.py`
  - SRT -> audio clips

## Output Design

Keep raw ASR text and post-processed text separate.

Recommended JSON fields:

```json
{
  "segments": [
    {
      "text": "post-processed display text",
      "raw_text": "original ASR text",
      "speaker_id": "speaker_0001",
      "start": 0.0,
      "end": 10.0,
      "asr_backend": "funasr"
    }
  ],
  "metadata": {
    "asr_backend": "funasr",
    "asr_model": "FunAudioLLM/Fun-ASR-Nano-2512",
    "text_model": "Qwen3.6-27B-Q4_K_M",
    "summary": "...",
    "structured": {
      "topics": [],
      "decisions": [],
      "action_items": []
    }
  }
}
```

Important rule: Qwen3.6 must not overwrite raw ASR evidence. Store both raw and cleaned text.

## Prompt Policy for Qwen3.6

Use conservative correction prompts.

Requirements:

- Do not add facts that are not in the transcript.
- Do not invent speaker names.
- Preserve timestamps and speaker IDs.
- Correct only obvious recognition errors, punctuation, repeated fillers, and domain terms.
- Mark uncertain corrections instead of silently rewriting.

For meeting summaries, Qwen3.6 should output:

- concise summary;
- topic list;
- decisions;
- action items with owner if explicitly mentioned;
- unresolved questions;
- risks or follow-ups.

## Runtime Strategy

Default production-like run:

```powershell
$env:DEVICE="cuda:0"
$env:ASR_BACKEND="funasr"
$env:QWEN_ENABLE_SEGMENT_CLEANUP="false"
$env:QWEN_ENABLE_SUMMARY="true"
$env:QWEN_ENABLE_STRUCTURED_OUTPUT="true"
python main.py --input ".\input\2026-05-25 16_14_10.mp3"
```

Fast ASR-only run:

```powershell
$env:DEVICE="cuda:0"
$env:ASR_BACKEND="funasr"
$env:QWEN_TEXT_ENABLED="false"
python main.py --input ".\input\2026-05-25 16_14_10.mp3"
```

Backend comparison run:

```powershell
$env:DEVICE="cuda:0"
$env:ASR_BACKEND="funasr"
python scripts\transcribe_audio.py --input ".\input\sample.mp3" --output-dir ".\output\compare\funasr"

$env:ASR_BACKEND="sensevoice"
python scripts\transcribe_audio.py --input ".\input\sample.mp3" --output-dir ".\output\compare\sensevoice"
```

## Evaluation Plan

Use the existing `input/2026-05-25 16_14_10.mp3` as the long-audio benchmark.

Record for each backend:

- total runtime;
- raw diarization segment count;
- ASR target segment count;
- final merged segment count;
- empty text segment count;
- obvious recognition errors;
- speaker attribution consistency;
- GPU/CPU utilization notes.

GPU verification checklist:

- `python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"` returns the target GPU.
- Runtime log contains `使用设备: cuda:0`.
- Backend log contains selected ASR backend and device.
- GPU-Z or `nvidia-smi` shows memory allocation and non-zero utilization during ASR forward passes.
- If utilization is low but runtime is progressing, check segment count, average segment duration, and batching before assuming the model is CPU-only.

Suggested baseline table:

| Backend | Runtime | Raw Segments | ASR Segments | Output Segments | Empty Segments | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FunASR main branch | 28.79 min | 784 | 784 | 438 | 0 | Better recognition quality in current test |
| Qwen3-ASR optimized branch | 18.66 min | 784 | 545 | 545 | 0 | Faster than main FunASR run, but recognition quality worse |
| SenseVoice | TBD | TBD | TBD | TBD | TBD | Add after backend implementation |

## Implementation Phases

### Phase 1: Low-Risk Refactor

- Add ASR backend selector.
- Wrap existing FunASR implementation as default backend.
- Keep current output behavior unchanged.
- Add metadata fields for backend and model.

### Phase 2: Qwen3.6 Post-Processing

- Add Qwen text service.
- Add whole-file summary and structured output.
- Keep segment cleanup disabled by default.
- Write Qwen outputs into metadata and separate summary files.

### Phase 3: SenseVoice Optional Backend

- Add SenseVoice backend.
- Add config and README docs.
- Run the same input through FunASR and SenseVoice.
- Compare quality and runtime.

### Phase 4: Quality Controls

- Preserve `raw_text`.
- Add domain vocabulary and speaker-name hints.
- Add uncertainty markers for Qwen corrections.
- Add tests for serialization, backend selection, and summary output.

## Recommended Final Direction

Do not make Qwen3-ASR the default backend for this project right now.

Recommended default:

- ASR: `FunASR`
- Optional ASR: `SenseVoice`
- Optional experimental ASR: `Qwen3-ASR`
- Understanding and summary: `Qwen3.6-27B`

This gives the best balance of recognition quality, speed, and downstream intelligence.
