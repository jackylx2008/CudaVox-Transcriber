"""Main transcription pipeline."""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from FunASRNano.audio import (
    build_profile_wav,
    convert_to_wav,
    ensure_dir,
    extract_wav_segment,
)
from FunASRNano.funasr_service import FunASRTranscriber
from FunASRNano.logging_config import get_logger
from FunASRNano.logging_utils import setup_project_logger
from FunASRNano.pyannote_service import PyannoteDiarizer
from FunASRNano.runtime import resolve_device
from FunASRNano.schemas import Settings, TranscriptDocument, TranscriptSegment
from FunASRNano.transcript_io import write_transcript_outputs
from FunASRNano.voiceprint_service import VoiceprintStore


class CudaVoxPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        level = getattr(logging, settings.app.log_level.upper(), logging.INFO)
        setup_project_logger(log_level=level, reset_log=True)
        self.logger = get_logger(__name__)
        self.device = resolve_device(
            settings.device.preferred,
            settings.device.allow_cpu_fallback,
        )
        self.logger.info(
            "Pipeline 初始化完成: device=%s, output_dir=%s, temp_dir=%s",
            self.device,
            Path(settings.app.output_dir).resolve(),
            Path(settings.app.temp_dir).resolve(),
        )

        self.diarizer = PyannoteDiarizer(settings.pyannote, self.device, self.logger)
        self.transcriber = FunASRTranscriber(settings.funasr, self.device, self.logger)
        self.voiceprints = VoiceprintStore(settings.campp, self.device, self.logger)

    def process_file(self, input_path: str | Path) -> dict[str, str]:
        input_file = Path(input_path)
        stem = input_file.stem
        output_dir = ensure_dir(Path(self.settings.app.output_dir) / stem)
        temp_dir = ensure_dir(Path(self.settings.app.temp_dir) / stem)
        normalized_wav = temp_dir / f"{stem}_normalized.wav"

        self.logger.info("开始处理: %s", input_file)
        self.logger.debug("输出目录: %s, 临时目录: %s", output_dir.resolve(), temp_dir.resolve())
        convert_to_wav(
            input_file,
            normalized_wav,
            ffmpeg_bin=self.settings.app.ffmpeg_bin,
        )

        segments = self.diarizer.diarize(normalized_wav)
        self.logger.info("说话人分离完成: %s 个片段", len(segments))
        for index, segment in enumerate(segments, start=1):
            speaker_label = segment.speaker_label or f"speaker_{index:04d}"
            segment_path = temp_dir / f"segment_{index:04d}_{speaker_label}.wav"
            extract_wav_segment(normalized_wav, segment.start, segment.end, segment_path)
            segment.segment_audio_path = str(segment_path)

        identities = self._resolve_voiceprints(
            normalized_wav=normalized_wav,
            segments=segments,
            temp_dir=temp_dir,
            source_file=input_file.name,
        )

        for segment in segments:
            speaker_label = segment.speaker_label
            identity = identities.get(speaker_label)
            if identity is None:
                identity = self.voiceprints.create_transient_identity(speaker_label)

            if segment.segment_audio_path:
                segment.text = self.transcriber.transcribe(segment.segment_audio_path)
            segment.speaker_id = identity.speaker_id
            segment.speaker_name = identity.speaker_name
            segment.speaker_similarity = identity.similarity

        raw_segments = [replace(segment) for segment in segments]
        merged_segments = self._merge_segments(segments)
        self.logger.info(
            "文本合并完成: 原始片段=%s, 合并后片段=%s",
            len(raw_segments),
            len(merged_segments),
        )
        transcript = TranscriptDocument(
            input_file=str(input_file.resolve()),
            normalized_wav=str(normalized_wav.resolve()),
            device=self.device,
            segments=merged_segments,
            raw_segments=raw_segments,
            metadata={
                "workflow": "diarization_transcription",
                "merge_gap_seconds": self.settings.pipeline.merge_gap_seconds,
            },
        )
        written_files = write_transcript_outputs(
            document=transcript,
            output_dir=output_dir,
            output_stem=input_file.stem,
            settings=self.settings.output,
            logger=self.logger,
        )

        if self.settings.app.cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.debug("已清理临时目录: %s", temp_dir.resolve())
        else:
            self.logger.info("保留临时目录: %s", temp_dir.resolve())

        self.logger.info("处理完成: %s", input_file)
        return written_files

    def _resolve_voiceprints(
        self,
        normalized_wav: Path,
        segments: list[TranscriptSegment],
        temp_dir: Path,
        source_file: str,
    ):
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for segment in segments:
            grouped[segment.speaker_label].append((segment.start, segment.end))

        self.logger.info("开始声纹归一化匹配，本地说话人数=%s", len(grouped))
        identities = {}
        for speaker_label, spans in grouped.items():
            profile_path = temp_dir / f"profile_{speaker_label}.wav"
            total_seconds = build_profile_wav(
                normalized_wav,
                spans,
                profile_path,
                max_duration_seconds=self.settings.campp.max_profile_audio_seconds,
            )
            if total_seconds < self.settings.campp.min_profile_seconds:
                self.logger.warning(
                    "说话人 %s 可用时长不足 %.2fs，改用临时身份。",
                    speaker_label,
                    self.settings.campp.min_profile_seconds,
                )
                identities[speaker_label] = self.voiceprints.create_transient_identity(
                    speaker_label
                )
                continue

            embedding = self.voiceprints.extract_embedding(profile_path)
            identity = self.voiceprints.match_or_register(
                embedding=embedding,
                source_file=source_file,
                local_speaker=speaker_label,
            )
            identities[speaker_label] = identity
            self.logger.info(
                "说话人 %s 归属为 %s，相似度=%s，新建=%s",
                speaker_label,
                identity.speaker_id,
                identity.similarity,
                identity.is_new,
            )
        return identities

    def _merge_segments(
        self,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        if not segments:
            self.logger.warning("没有可合并的片段。")
            return []

        merged: list[TranscriptSegment] = []
        for current in sorted(segments, key=lambda item: (item.start, item.end)):
            segment = replace(current)
            if not merged:
                merged.append(segment)
                continue

            previous = merged[-1]
            gap = segment.start - previous.end
            if (
                self._speaker_key(previous) == self._speaker_key(segment)
                and gap <= self.settings.pipeline.merge_gap_seconds
            ):
                previous.end = segment.end
                previous.text = self._join_text(previous.text, segment.text)
                if previous.speaker_similarity is None:
                    previous.speaker_similarity = segment.speaker_similarity
                continue

            merged.append(segment)

        return merged

    @staticmethod
    def _join_text(left: str, right: str) -> str:
        left = left.strip()
        right = right.strip()
        if not left:
            return right
        if not right:
            return left
        return f"{left}{right}"

    @staticmethod
    def _speaker_key(segment: TranscriptSegment) -> tuple[str, str, str]:
        return (
            segment.speaker_id,
            segment.speaker_name,
            segment.speaker_label,
        )
