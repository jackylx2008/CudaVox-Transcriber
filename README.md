# CudaVox-Transcriber

基于 `FunASR + pyannote.audio + CAM++` 的中文语音转写与说话人区分项目。

能力包括：

- 使用 CUDA 运行中文语音识别
- 使用 `pyannote.audio` 做说话人分离
- 使用 `CAM++` 生成与持久化声纹
- 跨音频尽量复用同一个说话人 ID
- 输出 `json / txt / srt`

## 方案结构

1. 先把输入音频统一转成 `16kHz / mono / wav`
2. 用 `pyannote/speaker-diarization-community-1` 做说话人分离
3. 按说话人片段切分音频
4. 用 `FunASR` 的 `paraformer-zh + fsmn-vad + ct-punc` 做中文转写
5. 把同一文件里的本地说话人片段拼成 profile 音频
6. 用 `iic/speech_campplus_sv_zh-cn_16k-common` 提取平均声纹
7. 与本地声纹库做余弦相似度比对，命中则复用已有说话人 ID，否则创建新说话人

## 推荐环境

这套组合更建议使用 `Python 3.10` 或 `Python 3.11`。你当前机器默认是 `Python 3.12.13`，代码我已经写成了兼容式封装，但安装依赖时建议单独建环境。

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

项目会用 `ffmpeg` 统一音频格式。

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
python -m cudavox_transcriber --input ".\input\2026-03-25 21_50_00.mp3"
```

## 输出目录

默认输出到 `./output`：

- `output/<音频名>/<音频名>.json`
- `output/<音频名>/<音频名>.txt`
- `output/<音频名>/<音频名>.srt`
- `output/voiceprints/speakers.json`
- `output/voiceprints/<speaker_id>.npy`

## 配置说明

核心配置在 [config.yaml](/d:/CloudStation/Python/Project/CudaVox-Transcriber/config.yaml)，环境变量在 [common.env](/d:/CloudStation/Python/Project/CudaVox-Transcriber/common.env)。

几个常用项：

- `device.preferred`: 默认 `cuda:0`
- `campp.similarity_threshold`: 声纹命中阈值，默认 `0.72`
- `pyannote.num_speakers`: 已知说话人数时可直接指定
- `pipeline.merge_gap_seconds`: 合并相邻同说话人片段的时间间隔

## 参考资料

- FunASR 官方 README: https://github.com/modelscope/FunASR
- pyannote.audio 官方 README: https://github.com/pyannote/pyannote-audio
- 3D-Speaker / CAM++ 官方仓库: https://github.com/modelscope/3D-Speaker
