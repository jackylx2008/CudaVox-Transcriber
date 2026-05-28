# CudaVox-Transcriber

基于 `FunASR + pyannote.audio + CAM++` 的中文语音转写与说话人区分项目。

当前版本在保留原有主流程的基础上，已经完成一轮面向后续扩展的重构：核心结果不再只是“说话人分离片段”，而是统一收敛到可复用的转写领域模型，便于后续继续增加：

- 基于现有 `SRT` 结果切割音频
- 只复用语音识别能力的独立工作流
- 读取外部字幕再做后处理的工作流

## 当前能力

- 使用 CUDA 运行中文语音识别
- 使用 `pyannote.audio` 做说话人分离
- 使用 `CAM++` 生成与持久化声纹
- 跨音频尽量复用同一个说话人 ID
- 默认使用专用 ASR 后端 `FunASR`，可通过配置切换到 `SenseVoice`
- 可选使用 `Qwen3.6` 做整文件摘要和结构化输出
- 输出 `json / txt / srt`
- 从历史转写结果中导出人工核对声纹样本

## 主流程

1. 把输入音频统一转成 `16kHz / mono / wav`
2. 用 `pyannote/speaker-diarization-community-1` 做说话人分离
3. 按片段切分临时音频
4. 用配置的 ASR backend 做中文转写，默认 `FunASR`，可选 `SenseVoice`
5. 把同一文件里的本地说话人片段拼成 profile 音频
6. 用 `CAM++` 提取声纹 embedding
7. 与本地声纹库做余弦相似度比对，命中则复用已有说话人 ID，否则创建新说话人
8. 将结果写出为统一的转写文档，再导出 `json / txt / srt`

## 重构后的结构

当前代码大致分成下面几层：

- 入口层：
  [main.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/main.py)、
  [Deprecated_main.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/Deprecated_main.py)、
  [cli.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/cli.py)
- 编排层：
  [pipeline.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/pipeline.py)
- 模型适配层：
  [funasr_service.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/funasr_service.py)、
  [pyannote_service.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/pyannote_service.py)、
  [voiceprint_service.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/voiceprint_service.py)
- 通用音频工具：
  [audio.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/audio.py)
- 转写领域模型：
  [schemas.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/schemas.py)
- 转写读写层：
  [transcript_io.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/FunASRNano/transcript_io.py)
- 后处理脚本：
  [export_voiceprint_samples.py](/d:/CloudStation/Python/Project/CudaVox-Transcriber/scripts/export_voiceprint_samples.py)

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

注意：

- 目前仓库里还没有单独暴露“根据 SRT 切音频”的命令行脚本
- 但底层能力已经具备：`load_srt_segments(...)` 可读字幕，`cut_audio_clip(...)` 可按时间切音频
- 后续如果要加这个工作流，建议新增独立脚本，而不是继续塞进现有 `pipeline.py`

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

兼容旧入口：

```powershell
python .\Deprecated_main.py --input ".\input\2026-03-25 21_50_00.mp3"
```

`Deprecated_main.py` is deprecated. New commands should use `main.py`. `scripts/transcribe_audio.py` is kept as an internal workflow entrypoint and should not be used as the documented project entry.

日志文件：

- 保留原始分散日志，例如入口日志、`qwen_asr_server.err.log`、`llama_server.err.log`
- 运行 `main.py` 时，主要 INFO 级运行记录会同时写入 `log/main.log`

Qwen3.6 llama.cpp 服务生命周期：

- `LLAMACPP_AUTOSTART=true` 时，如果 8080 服务未启动，程序会自动启动 `llama-server`
- 默认只关闭本次自动启动的 `llama-server`
- 如果需要运行结束后也关闭已经存在的 8080 服务，设置 `LLAMACPP_SHUTDOWN_EXISTING_ON_EXIT=true`

切换 ASR backend：

```powershell
$env:DEVICE="cuda:0"
$env:ASR_BACKEND="funasr"
python main.py --input ".\input\2026-03-25 21_50_00.mp3"

$env:ASR_BACKEND="sensevoice"
python main.py --input ".\input\2026-03-25 21_50_00.mp3"
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

## JSON 输出说明

JSON 仍兼容原有字段，同时补充了更通用的字段，便于后续新增工作流：

- `segments`: 合并后的片段
- `raw_segments`: 合并前的原始片段
- `speaker_label`: 当前文件内的本地说话人标签
- `local_speaker`: 兼容旧逻辑保留的别名字段
- `segment_audio_path`: 片段音频路径
- `raw_text`: ASR 原始识别文本
- `source`: 片段来源，例如 `diarization`
- `metadata`: 文档级元数据，包含 `asr_backend`、`asr_model`、`text_model`，以及启用时的 `summary` 和 `structured`

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
- `asr.backend`: ASR 后端，默认 `funasr`，可选 `sensevoice`
- `funasr.model`: 默认 `FunAudioLLM/Fun-ASR-Nano-2512`
- `funasr.language`: 默认 `中文`
- `funasr.itn`: 默认 `true`
- `funasr.trust_remote_code`: 默认 `false`
- `sensevoice.model`: 默认 `iic/SenseVoiceSmall`
- `sensevoice.language`: 默认 `zh`
- `qwen_text.enabled`: 是否启用 Qwen3.6 文本后处理
- `qwen_text.enable_segment_cleanup`: 是否逐段校对文本，默认 `false`
- `qwen_text.enable_summary`: 是否生成整文件摘要
- `qwen_text.enable_structured_output`: 是否生成结构化 metadata
- `campp.similarity_threshold`: 声纹命中阈值，默认 `0.72`
- `campp.relaxed_similarity_threshold`: 对已有稳定声纹启用保守复用的次级阈值，默认 `0.69`
- `campp.named_similarity_threshold`: 对已命名说话人的跨音频复用阈值，默认 `0.64`
- `pyannote.num_speakers`: 已知说话人数时可直接指定
- `pipeline.merge_gap_seconds`: 合并相邻同说话人片段的时间间隔

## 后续扩展建议

如果继续沿当前重构方向扩展，建议按内部工作流增加脚本，而不是继续让 `pipeline.py` 变胖：

- `transcribe_audio.py`
  内部入口，负责“音频 -> TranscriptDocument”
- `cut_audio_by_srt.py`
  负责“读取 SRT -> 切音频”
- `export_voiceprint_samples.py`
  负责“历史结果 -> 声纹人工复核样本”

## 参考资料

- FunASR 官方 README: https://github.com/modelscope/FunASR
- pyannote.audio 官方 README: https://github.com/pyannote/pyannote-audio
- 3D-Speaker / CAM++ 官方仓库: https://github.com/modelscope/3D-Speaker
