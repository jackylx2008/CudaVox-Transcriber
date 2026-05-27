"""Project data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AppSettings:
    log_level: str = "INFO"
    input_path: str = "./input"
    input_files: list[str] = field(default_factory=list)
    output_dir: str = "./output"
    temp_dir: str = "./output/tmp"
    cleanup_temp: bool = True
    ffmpeg_bin: str = "ffmpeg"
    supported_extensions: list[str] = field(
        default_factory=lambda: [".wav", ".mp3", ".m4a", ".flac", ".aac"]
    )


@dataclass
class DeviceSettings:
    preferred: str = "cuda:0"
    allow_cpu_fallback: bool = True


@dataclass
class FunASRSettings:
    model: str = "FunAudioLLM/Fun-ASR-Nano-2512"
    vad_model: str = ""
    punc_model: str = ""
    hub: str = "ms"
    batch_size_s: int = 120
    hotword: str = ""
    max_single_segment_time: int = 30000
    language: str = ""
    itn: Optional[bool] = None
    trust_remote_code: Optional[bool] = None


@dataclass
class AsrSettings:
    backend: str = "funasr"


@dataclass
class SenseVoiceSettings:
    model: str = "iic/SenseVoiceSmall"
    hub: str = "ms"
    batch_size_s: int = 120
    hotword: str = ""
    language: str = "zh"
    itn: Optional[bool] = True
    trust_remote_code: Optional[bool] = True


@dataclass
class QwenTextSettings:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "Qwen3.6-27B-Q4_K_M"
    api_key: str = ""
    request_timeout_seconds: int = 60
    temperature: float = 0.0
    enable_segment_cleanup: bool = False
    enable_summary: bool = True
    enable_structured_output: bool = True
    segment_cleanup_max_tokens: int = 256
    summary_max_tokens: int = 1024
    structured_max_tokens: int = 1024
    summary_input_max_chars: int = 12000
    cleanup_prompt: str = (
        "你是中文会议转写校对助手。只修正明显的 ASR 错字、同音词、标点和格式，"
        "不要新增事实，不要改变原意，不要编造说话人。只输出修正后的文本。"
    )
    summary_prompt: str = (
        "请基于下面的会议转写内容生成简洁中文摘要，包含主题、关键结论、待办事项和未解决问题。"
        "不要添加转写中不存在的信息。"
    )
    structured_prompt: str = (
        "请基于下面的会议转写内容输出 JSON，对象包含 topics、decisions、action_items、"
        "open_questions、risks。不要添加转写中不存在的信息。"
    )


@dataclass
class PyannoteSettings:
    model: str = "pyannote/speaker-diarization-community-1"
    token: str = ""
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    min_segment_seconds: float = 0.6


@dataclass
class CamppSettings:
    model: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    similarity_threshold: float = 0.72
    relaxed_similarity_threshold: float = 0.69
    relaxed_similarity_margin: float = 0.04
    relaxed_min_samples: int = 2
    named_similarity_threshold: float = 0.64
    named_similarity_margin: float = 0.08
    named_min_samples: int = 5
    db_dir: str = "./output/voiceprints"
    metadata_file: str = "speakers.json"
    speaker_prefix: str = "speaker_"
    speaker_name_map: dict[str, str] = field(default_factory=dict)
    min_profile_seconds: float = 1.5
    max_profile_audio_seconds: float = 30.0


@dataclass
class PipelineSettings:
    merge_gap_seconds: float = 0.4


@dataclass
class OutputSettings:
    write_json: bool = True
    write_txt: bool = True
    write_srt: bool = True


@dataclass
class Settings:
    app: AppSettings = field(default_factory=AppSettings)
    device: DeviceSettings = field(default_factory=DeviceSettings)
    asr: AsrSettings = field(default_factory=AsrSettings)
    funasr: FunASRSettings = field(default_factory=FunASRSettings)
    sensevoice: SenseVoiceSettings = field(default_factory=SenseVoiceSettings)
    qwen_text: QwenTextSettings = field(default_factory=QwenTextSettings)
    pyannote: PyannoteSettings = field(default_factory=PyannoteSettings)
    campp: CamppSettings = field(default_factory=CamppSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    output: OutputSettings = field(default_factory=OutputSettings)


@dataclass
class VoiceprintIdentity:
    speaker_id: str
    speaker_name: str
    similarity: Optional[float]
    is_new: bool
    embedding_path: Optional[str] = None
    transient: bool = False


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str = ""
    raw_text: str = ""
    speaker_label: str = ""
    speaker_id: str = ""
    speaker_name: str = ""
    speaker_similarity: Optional[float] = None
    segment_audio_path: Optional[str] = None
    source: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def local_speaker(self) -> str:
        return self.speaker_label

    @local_speaker.setter
    def local_speaker(self, value: str) -> None:
        self.speaker_label = value

    @property
    def segment_wav(self) -> Optional[str]:
        return self.segment_audio_path

    @segment_wav.setter
    def segment_wav(self, value: Optional[str]) -> None:
        self.segment_audio_path = value


@dataclass
class TranscriptDocument:
    input_file: str
    normalized_wav: str
    device: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    raw_segments: list[TranscriptSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def segment_count(self) -> int:
        return len(self.segments)


class DiarizedSegment(TranscriptSegment):
    """Backward-compatible alias for older diarization-oriented naming."""
