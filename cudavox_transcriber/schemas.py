"""Project data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
    db_dir: str = "./output/voiceprints"
    metadata_file: str = "speakers.json"
    speaker_prefix: str = "speaker_"
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
    funasr: FunASRSettings = field(default_factory=FunASRSettings)
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
class DiarizedSegment:
    start: float
    end: float
    local_speaker: str
    text: str = ""
    speaker_id: str = ""
    speaker_name: str = ""
    speaker_similarity: Optional[float] = None
    segment_wav: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)
