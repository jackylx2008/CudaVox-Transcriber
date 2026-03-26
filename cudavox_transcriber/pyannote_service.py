"""pyannote speaker diarization service."""

from __future__ import annotations

from pathlib import Path

from cudavox_transcriber.schemas import DiarizedSegment, PyannoteSettings


class PyannoteDiarizer:
    def __init__(self, settings: PyannoteSettings, device: str, logger) -> None:
        self.settings = settings
        self.device = device
        self.logger = logger
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._load()
        return self._pipeline

    def _load(self) -> None:
        if not self.settings.token:
            raise RuntimeError(
                "pyannote 需要 Hugging Face token。请在 common.env 中填写 HUGGINGFACE_TOKEN，"
                "并先接受 pyannote/speaker-diarization-community-1 的使用条款。"
            )

        import torch
        from pyannote.audio import Pipeline

        self.logger.info("加载 pyannote 模型: %s", self.settings.model)
        self._pipeline = Pipeline.from_pretrained(
            self.settings.model,
            token=self.settings.token,
        )
        self._pipeline.to(torch.device(self.device))

    @staticmethod
    def _build_audio_input(audio_path: str | Path):
        import soundfile as sf
        import torch

        waveform, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )
        tensor = torch.from_numpy(waveform.T)
        return {
            "waveform": tensor,
            "sample_rate": sample_rate,
            "uri": Path(audio_path).stem,
        }

    def diarize(self, audio_path: str | Path) -> list[DiarizedSegment]:
        kwargs = {}
        if self.settings.num_speakers is not None:
            kwargs["num_speakers"] = self.settings.num_speakers
        if self.settings.min_speakers is not None:
            kwargs["min_speakers"] = self.settings.min_speakers
        if self.settings.max_speakers is not None:
            kwargs["max_speakers"] = self.settings.max_speakers

        self.logger.info("开始说话人分离: %s", Path(audio_path).resolve())
        self.logger.debug("pyannote 参数: %s", kwargs or "default")
        pipeline_input = self._build_audio_input(audio_path)
        self.logger.debug(
            "已将音频预加载到内存供 pyannote 使用: uri=%s, sample_rate=%s, shape=%s",
            pipeline_input["uri"],
            pipeline_input["sample_rate"],
            tuple(pipeline_input["waveform"].shape),
        )
        result = self.pipeline(pipeline_input, **kwargs)
        annotation = (
            getattr(result, "exclusive_speaker_diarization", None)
            or getattr(result, "speaker_diarization", None)
            or result
        )

        segments: list[DiarizedSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start = round(float(turn.start), 3)
            end = round(float(turn.end), 3)
            if end - start < self.settings.min_segment_seconds:
                self.logger.debug(
                    "跳过过短片段: speaker=%s, start=%.3f, end=%.3f",
                    speaker,
                    start,
                    end,
                )
                continue
            segments.append(
                DiarizedSegment(
                    start=start,
                    end=end,
                    local_speaker=str(speaker),
                )
            )

        if not segments:
            raise RuntimeError("pyannote 没有检测到有效说话人片段。")

        self.logger.info("pyannote 检测到 %s 个说话片段", len(segments))
        return sorted(segments, key=lambda item: (item.start, item.end))
