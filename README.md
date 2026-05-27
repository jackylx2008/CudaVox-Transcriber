# CudaVox-Transcriber

基于 `Qwen3-ASR + Qwen3.6 + pyannote.audio + CAM++` 的中文语音转写与说话人区分项目。

当前版本在保留原有主流程的基础上，已经完成一轮面向后续扩展的重构：核心结果不再只是“说话人分离片段”，而是统一收敛到可复用的转写领域模型，便于后续继续增加：

- 基于现有 `SRT` 结果切割音频
- 只复用语音识别能力的独立工作流
- 读取外部字幕再做后处理的工作流

## 当前能力

- 使用 CUDA 运行中文语音识别
- 使用 `pyannote.audio` 做说话人分离
- 使用 `CAM++` 生成与持久化声纹
- 跨音频尽量复用同一个说话人 ID
- 输出 `json / txt / srt`
- 从历史转写结果中导出人工核对声纹样本

## 主流程

1. 把输入音频统一转成 `16kHz / mono / wav`
2. 用 `pyannote/speaker-diarization-community-1` 做说话人分离
3. 先完成声纹归一化匹配
4. 合并相邻同说话人的短片段，减少 ASR 请求次数
5. 按合并后的片段切分临时音频
6. 用 `Qwen3-ASR-1.7B` 做原始听写
7. 默认跳过逐段 `Qwen3.6-27B` 文本整理；速度测试时也可关闭整文件摘要
8. 把同一文件里的本地说话人片段拼成 profile 音频
9. 用 `CAM++` 提取声纹 embedding
10. 与本地声纹库做余弦相似度比对，命中则复用已有说话人 ID，否则创建新说话人
11. 将结果写出为统一的转写文档，再导出 `json / txt / srt`

## 重构后的结构

当前代码大致分成下面几层：

- 入口层：
  [main.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/main.py)、
  [cli.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/cli.py)
- 编排层：
  [pipeline.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/pipeline.py)
- 模型适配层：
  [qwen_service.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/qwen_service.py)、
  [pyannote_service.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/pyannote_service.py)、
  [voiceprint_service.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/voiceprint_service.py)
- 通用音频工具：
  [audio.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/audio.py)
- 日志配置：
  [logging_config.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/logging_config.py)、
  [logging_utils.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/logging_utils.py)
- 转写领域模型：
  [schemas.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/schemas.py)
- 转写读写层：
  [transcript_io.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/transcript_io.py)
- 后处理脚本：
  [transcribe_audio.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/scripts/transcribe_audio.py)、
  [cut_audio_by_srt.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/scripts/cut_audio_by_srt.py)、
  [export_voiceprint_samples.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/scripts/export_voiceprint_samples.py)

项目进度和当前维护状态记录在 [PROJECT_PROGRESS.md](/d:/CloudStation/Python/Project/CudaVox-Transcriber/PROJECT_PROGRESS.md)。

## 核心数据模型

重构后新增了两个核心对象：

- `TranscriptSegment`
  统一表示一个带时间轴的转写片段，可同时承载文本、说话人、来源、切片音频路径等信息
- `TranscriptDocument`
  统一表示一个音频文件的完整转写结果，包含 `segments`、`raw_segments`、输入音频路径、标准化音频路径和元数据

这层模型定义在 [schemas.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/schemas.py)。

兼容性说明：

- 代码里仍保留了 `DiarizedSegment` 兼容别名，便于逐步迁移
- JSON 输出仍保留 `local_speaker`，同时新增更通用的 `speaker_label`

## SRT 相关基础能力

当前已经有一层可复用的 SRT 读写能力，位于 [transcript_io.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/transcript_io.py)：

- 可把统一转写文档写出为 `json / txt / srt`
- 可把已有 `SRT` 解析为 `TranscriptSegment` 列表

已经单独暴露“根据 SRT 切音频”的命令行脚本，位于 [cut_audio_by_srt.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/scripts/cut_audio_by_srt.py)。它复用 `load_srt_segments(...)` 读取字幕，复用 `cut_audio_clip(...)` 按时间切音频，并输出 `wav` 片段和 `clips.csv` 清单。

## 推荐环境

推荐使用 `Python 3.10` 或 `Python 3.11`。

```powershell
conda create -n cudavox python=3.10 -y
conda activate cudavox
```

先安装与你 CUDA 版本匹配的 PyTorch：

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

再安装项目依赖：

```powershell
pip install -r requirements.txt
```

语音听写和文本整理通过本地 OpenAI 兼容 HTTP API 调用，具体本地 `llama-server` 配置按本机 `LOCAL_AI_RUNTIME_SETUP.md` 的约定设置。

## 重要前置条件

### 1. 安装 ffmpeg

项目会用 `ffmpeg` 做音频标准化、片段切分和样本导出。

```powershell
ffmpeg -version
```

### 2. 准备 Hugging Face Token

`pyannote.audio` 的 `community-1` 需要：

- 先接受 `pyannote/speaker-diarization-community-1` 的使用条款
- 再创建 Hugging Face access token

把 token 填到 [common.env](/d:/CloudStation/Python/Project/CudaVox-Transcriber/common.env)：

```env
HUGGINGFACE_TOKEN=hf_xxx
```

## 运行方式

在 [common.env](/d:/CloudStation/Python/Project/CudaVox-Transcriber/common.env) 里指定待处理文件：

```env
INPUT_FILES=.\input\2026-03-16 17_54_59_example.mp3
```

多个文件可用英文分号、逗号或换行分隔：

```env
INPUT_FILES=.\input\a.mp3;.\input\b.wav
```

直接运行：

```powershell
python main.py
```

如果 `INPUT_FILES` 为空，才会回退为处理整个 `input/` 目录。

指定单个音频：

```powershell
python main.py --input ".\input\2026-03-25 21_50_00.mp3"
```

指定配置文件：

```powershell
python main.py --config .\config.yaml
```

也可以用模块方式：

```powershell
python -m FunASRNano --input ".\input\2026-03-25 21_50_00.mp3"
```

## 独立工作流脚本

只运行“音频 -> TranscriptDocument -> json/txt/srt”的独立转写工作流：

```powershell
python .\scripts\transcribe_audio.py --input ".\input\2026-03-25 21_50_00.mp3"
```

根据已有 SRT 切音频片段，并生成 `clips.csv` 清单：

```powershell
python .\scripts\cut_audio_by_srt.py `
  --audio ".\input\2026-03-25 21_50_00.mp3" `
  --srt ".\output\2026-03-25 21_50_00\2026-03-25 21_50_00.srt"
```

## 声纹样本导出

导出人工核对声纹的人声样本：

```powershell
python .\scripts\export_voiceprint_samples.py
```

默认行为：

- 扫描 `output/` 下所有转写结果 JSON
- 基于 `raw_segments` 按 `speaker_id` 导出最多 3 段样本
- 跳过空文本片段
- 只导出单段时长不少于 10 秒的片段
- 从原始 `input_file` 重新切出 `wav`
- 输出目录为 `output/voiceprint_samples/`
- 同时生成 `output/voiceprint_samples/voiceprint_samples.csv`

## 输出目录

默认输出到 `./output`：

- `output/<音频名>/<音频名>.json`
- `output/<音频名>/<音频名>.txt`
- `output/<音频名>/<音频名>.srt`
- `output/voiceprints/speakers.json`
- `output/voiceprints/<speaker_id>.npy`
- `output/voiceprint_samples/`

## 日志配置

日志配置不放在项目根目录，统一放在包内：

- [FunASRNano/logging_config.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/logging_config.py)：项目复用的统一日志配置
- [FunASRNano/logging_utils.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/logging_utils.py)：本项目的兼容封装，负责保留项目内 `log/` 输出目录和 `reset_log` 行为

代码中应通过 `FunASRNano.logging_config` 或 `FunASRNano.logging_utils` 导入日志能力，不再从项目根目录导入 `logging_config`。

## JSON 输出说明

JSON 仍兼容原有字段，同时补充了更通用的字段，便于后续新增工作流：

- `segments`: 合并后的片段
- `raw_segments`: 合并前的原始片段
- `speaker_label`: 当前文件内的本地说话人标签
- `local_speaker`: 兼容旧逻辑保留的别名字段
- `segment_audio_path`: 片段音频路径
- `source`: 片段来源，例如 `diarization`
- `metadata`: 文档级元数据，包含 `dictation_model`、`text_model`，以及启用时的 `summary`

当前为了减少 Qwen HTTP 请求，程序先按声纹归一化后的说话人身份合并 diarization 片段，再切音频并调用 Qwen3-ASR。`raw_segments` 仍保留原始 diarization 片段，并在 `extras.transcribed_segment_index` 中记录它归属的合并听写片段。

Qwen3-ASR 速度敏感配置：

- `qwen.asr_max_tokens` 默认降为 `256`，避免个别短片段生成到 1024 token 上限。
- `qwen.asr_concurrency` 默认 `3`，并发请求本地 `llama-server`，让多个 slot 同时处理 ASR。
- `pipeline.merge_gap_seconds` 默认 `2.0`，并按声纹身份而不是 pyannote 本地标签做合并。
- `pipeline.min_asr_segment_seconds` 默认 `4.0`，尽量吸收同一说话人的短片段。
- `pipeline.max_asr_segment_seconds` 默认 `30.0`，避免合并出过长 ASR 请求。

## 当前性能记录

使用 `input/2026-04-13 09_46_37.mp3` 在本机 Qwen3-ASR + Qwen3.6 配置下测试：

- 优化前：约 24 分钟，主要耗时来自逐段 Qwen3.6 文本整理和 80 次片段化 ASR 调用。
- 优化后：608.20 秒，约 10.14 分钟；80 个原始 diarization 片段先合并为 66 个 ASR 片段，且逐段文本整理关闭，只保留整文件摘要。

`input/2026-05-25 16_14_10.mp3` 的中断测试暴露了更典型的长音频瓶颈：pyannote 产生 784 个原始片段，旧合并逻辑仍产生 617 个 ASR 请求；Qwen3-ASR 日志显示 audio encoder 使用 CPU backend，并且 ASR HTTP 请求串行执行。

按“减少片段数 + 降 token + 关闭 cache + 并发请求”继续优化后，同一文件在本机 ASR-only 配置下完成：

- 总耗时：1119.8 秒，约 18.66 分钟。
- 输出：784 个原始 diarization 片段，545 个 Qwen3-ASR 目标片段，0 个空文本片段。
- 配置：`QWEN_ASR_MAX_TOKENS=256`、`QWEN_ASR_CONCURRENCY=3`、`QWEN_ENABLE_TEXT_REFINEMENT=false`、`QWEN_ENABLE_SUMMARY=false`。
- 服务：Qwen3-ASR `llama-server` 使用 `--cache-ram 0`；由于关闭总结，本轮不需要启动 Qwen3.6-27B。

## 声纹姓名映射

如果你希望转写结果直接显示人名，而不是 `speaker_0001` 这类 ID，可以单独放到私密文件 `voiceprint_name_map.env` 里：

```text
speaker_0001=张三
speaker_0002=李四
```

说明：

- `voiceprint_name_map.env` 默认不会同步到 git，适合放人名这类敏感映射
- 使用格式 `speaker_id=姓名`
- 一行一个映射，方便人工维护
- 只需要维护 `speaker_id` 到姓名的对应关系，不要手动改 `.npy` 文件名
- 程序启动时会自动先加载 [common.env](/d:/CloudStation/Python/Project/CudaVox-Transcriber/common.env)，再加载 `voiceprint_name_map.env`，并把这个映射同步到 `output/voiceprints/speakers.json`
- 后续输出的 `json / txt / srt` 会直接带上这个姓名

如果映射里写了不存在的 `speaker_id`，程序会跳过并在日志里提示。

## 配置说明

核心配置在 [config.yaml](/d:/CloudStation/Python/Project/CudaVox-Transcriber/config.yaml)，环境变量在 [common.env](/d:/CloudStation/Python/Project/CudaVox-Transcriber/common.env)。

几个常用项：

- `device.preferred`: 默认 `cuda:0`
- `qwen.asr_base_url`: Qwen3-ASR OpenAI 兼容 API 地址，默认 `http://127.0.0.1:8081/v1`
- `qwen.asr_model`: 听写模型，默认 `Qwen3-ASR-1.7B`
- `qwen.asr_endpoint`: ASR 调用方式，默认 `chat_completions`，也可设为 `audio_transcriptions`
- `qwen.asr_max_tokens`: ASR 输出 token 上限，默认 `256`
- `qwen.asr_concurrency`: ASR 并发请求数，默认 `3`
- `qwen.llm_base_url`: 文本整理模型 API 地址，默认读取 `LLAMACPP_BASE_URL`
- `qwen.llm_model`: 文本整理和总结模型，默认读取 `LLAMACPP_MODEL`
- `qwen.enable_text_refinement`: 是否用 Qwen3.6 整理每段听写文本，默认 `false`
- `qwen.refinement_max_tokens`: 逐段整理 token 上限，默认 `256`
- `qwen.summary_max_tokens`: 整文件总结 token 上限，默认 `512`
- `qwen.summary_input_max_chars`: 送入整文件总结的最大字符数，默认 `5000`
- `qwen.refinement_min_duration_seconds`: 短于该时长的片段跳过逐段整理，默认 `2.0`
- `qwen.enable_summary`: 是否在 JSON `metadata.summary` 中写入整文件摘要；速度测试可设为 `false`
- `campp.similarity_threshold`: 声纹命中阈值，默认 `0.72`
- `campp.relaxed_similarity_threshold`: 对已有稳定声纹启用保守复用的次级阈值，默认 `0.69`
- `campp.named_similarity_threshold`: 对已命名说话人的跨音频复用阈值，默认 `0.64`
- `pyannote.num_speakers`: 已知说话人数时可直接指定
- `pipeline.merge_gap_seconds`: 合并相邻同说话人片段的时间间隔，默认 `2.0`
- `pipeline.min_asr_segment_seconds`: 同一说话人的短片段吸收阈值，默认 `4.0`
- `pipeline.max_asr_segment_seconds`: 合并后 ASR 片段最长时长，默认 `30.0`

启动 Qwen3-ASR 的 `llama-server` 时建议关闭 prompt cache，避免大量不同音频片段反复保存/淘汰缓存：

```powershell
D:\llama-cpp-cu12\llama-server.exe `
  -m "C:\Users\bcjt_\.lmstudio\models\ggml-org\Qwen3-ASR-1.7B-GGUF\Qwen3-ASR-1.7B-Q8_0.gguf" `
  --mmproj "C:\Users\bcjt_\.lmstudio\models\ggml-org\Qwen3-ASR-1.7B-GGUF\mmproj-Qwen3-ASR-1.7B-bf16.gguf" `
  --host 127.0.0.1 `
  --port 8081 `
  -c 8192 `
  -ngl 999 `
  --alias Qwen3-ASR-1.7B `
  --cache-ram 0
```

## 后续扩展建议

如果继续沿当前重构方向扩展，建议按独立工作流增加脚本，而不是继续让 `pipeline.py` 变胖：

- [transcribe_audio.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/scripts/transcribe_audio.py)
  负责“音频 -> TranscriptDocument”
- [cut_audio_by_srt.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/scripts/cut_audio_by_srt.py)
  负责“读取 SRT -> 切音频”
- [export_voiceprint_samples.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/scripts/export_voiceprint_samples.py)
  负责“历史结果 -> 声纹人工复核样本”
- 可选保留 `FunASR` 或 `SenseVoice` 作为快速 ASR 后端。Qwen3-ASR 更灵活，但本地多模态 LLM 的 HTTP/片段化调用成本高；专用 ASR pipeline 在吞吐上通常更快。

## 参考资料

- Qwen3-ASR 模型配置参考本机 `LOCAL_AI_RUNTIME_SETUP.md`
- pyannote.audio 官方 README: https://github.com/pyannote/pyannote-audio
- 3D-Speaker / CAM++ 官方仓库: https://github.com/modelscope/3D-Speaker
