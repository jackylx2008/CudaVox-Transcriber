"""Main transcription pipeline."""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from FunASRNano.audio import (
    build_profile_wav,
    convert_to_wav,
    ensure_dir,
    extract_wav_segment,
)
from FunASRNano.logging_config import get_logger
from FunASRNano.logging_utils import setup_project_logger
from FunASRNano.pyannote_service import PyannoteDiarizer
from FunASRNano.qwen_service import QwenTranscriber
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
        self.transcriber = QwenTranscriber(settings.qwen, self.logger)
        self.voiceprints = VoiceprintStore(settings.campp, self.device, self.logger)

    def transcribe_file(self, input_path: str | Path) -> TranscriptDocument:
        input_file = Path(input_path)
        stem = input_file.stem
        temp_dir = ensure_dir(Path(self.settings.app.temp_dir) / stem)
        normalized_wav = temp_dir / f"{stem}_normalized.wav"

        self.logger.info("开始处理: %s", input_file)
        self.logger.debug("临时目录: %s", temp_dir.resolve())
        convert_to_wav(
            input_file,
            normalized_wav,
            ffmpeg_bin=self.settings.app.ffmpeg_bin,
        )

        segments = self.diarizer.diarize(normalized_wav)
        self.logger.info("说话人分离完成: %s 个片段", len(segments))
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

            segment.speaker_id = identity.speaker_id
            segment.speaker_name = identity.speaker_name
            segment.speaker_similarity = identity.similarity

        raw_segments = [self._copy_segment(segment) for segment in segments]
        merged_segments = self._merge_segments(segments)
        self.logger.info(
            "ASR 前合并完成: 原始片段=%s, ASR 目标片段=%s, 并发=%s",
            len(raw_segments),
            len(merged_segments),
            self.settings.qwen.asr_concurrency,
        )

        self._transcribe_segments(merged_segments, normalized_wav, temp_dir)

        self._annotate_raw_segments(raw_segments, merged_segments)
        self.logger.info(
            "文本转写完成: 原始片段=%s, 输出片段=%s",
            len(raw_segments),
            len(merged_segments),
        )
        metadata = {
            "workflow": "diarization_transcription",
            "merge_gap_seconds": self.settings.pipeline.merge_gap_seconds,
            "dictation_model": self.settings.qwen.asr_model,
            "text_model": self.settings.qwen.llm_model,
            "raw_segment_count": len(raw_segments),
            "asr_segment_count": len(merged_segments),
            "text_refinement_enabled": self.settings.qwen.enable_text_refinement,
            "asr_concurrency": self.settings.qwen.asr_concurrency,
            "asr_max_tokens": self.settings.qwen.asr_max_tokens,
            "min_asr_segment_seconds": self.settings.pipeline.min_asr_segment_seconds,
            "max_asr_segment_seconds": self.settings.pipeline.max_asr_segment_seconds,
        }
        summary = self.transcriber.summarize(merged_segments)
        if summary:
            metadata["summary"] = summary

        transcript = TranscriptDocument(
            input_file=str(input_file.resolve()),
            normalized_wav=str(normalized_wav.resolve()),
            device=self.device,
            segments=merged_segments,
            raw_segments=raw_segments,
            metadata=metadata,
        )
        return transcript

    def process_file(self, input_path: str | Path) -> dict[str, str]:
        input_file = Path(input_path)
        output_dir = ensure_dir(Path(self.settings.app.output_dir) / input_file.stem)
        self.logger.debug("输出目录: %s", output_dir.resolve())
        transcript = self.transcribe_file(input_file)
        written_files = write_transcript_outputs(
            document=transcript,
            output_dir=output_dir,
            output_stem=input_file.stem,
            settings=self.settings.output,
            logger=self.logger,
        )

        temp_dir = Path(self.settings.app.temp_dir) / input_file.stem
        if self.settings.app.cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.debug("已清理临时目录: %s", temp_dir.resolve())
        else:
            self.logger.info("保留临时目录: %s", temp_dir.resolve())

        self.logger.info("处理完成: %s", input_file)
        return written_files

    def _transcribe_segments(
        self,
        segments: list[TranscriptSegment],
        normalized_wav: Path,
        temp_dir: Path,
    ) -> None:
        workers = max(1, int(self.settings.qwen.asr_concurrency))
        if workers == 1 or len(segments) <= 1:
            for index, segment in enumerate(segments, start=1):
                self._prepare_and_transcribe_segment(index, segment, normalized_wav, temp_dir)
            return

        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._prepare_and_transcribe_segment,
                    index,
                    segment,
                    normalized_wav,
                    temp_dir,
                ): segment
                for index, segment in enumerate(segments, start=1)
            }
            for future in as_completed(futures):
                segment = futures[future]
                future.result()
                completed += 1
                if completed % 20 == 0 or completed == len(segments):
                    self.logger.info(
                        "ASR 并发转写进度: %s/%s",
                        completed,
                        len(segments),
                    )

    def _prepare_and_transcribe_segment(
        self,
        index: int,
        segment: TranscriptSegment,
        normalized_wav: Path,
        temp_dir: Path,
    ) -> None:
        speaker_label = segment.speaker_label or f"speaker_{index:04d}"
        segment_path = temp_dir / f"asr_segment_{index:04d}_{speaker_label}.wav"
        extract_wav_segment(normalized_wav, segment.start, segment.end, segment_path)
        segment.segment_audio_path = str(segment_path)
        segment.text = self.transcriber.transcribe_segment(segment)

    @staticmethod
    def _copy_segment(segment: TranscriptSegment) -> TranscriptSegment:
        copied = replace(segment)
        copied.extras = dict(segment.extras)
        return copied

    @staticmethod
    def _annotate_raw_segments(
        raw_segments: list[TranscriptSegment],
        merged_segments: list[TranscriptSegment],
    ) -> None:
        for raw_segment in raw_segments:
            for index, merged_segment in enumerate(merged_segments, start=1):
                if (
                    raw_segment.start >= merged_segment.start
                    and raw_segment.end <= merged_segment.end
                    and CudaVoxPipeline._speaker_key(raw_segment)
                    == CudaVoxPipeline._speaker_key(merged_segment)
                ):
                    raw_segment.text = merged_segment.text
                    raw_segment.extras["transcribed_segment_index"] = index
                    raw_segment.extras["transcribed_segment_audio_path"] = (
                        merged_segment.segment_audio_path
                    )
                    break

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
            segment = self._copy_segment(current)
            if not merged:
                merged.append(segment)
                continue

            previous = merged[-1]
            gap = segment.start - previous.end
            merged_duration = segment.end - previous.start
            short_merge_gap = max(
                self.settings.pipeline.merge_gap_seconds,
                self.settings.pipeline.min_asr_segment_seconds,
            )
            can_merge = (
                self._speaker_key(previous) == self._speaker_key(segment)
                and gap <= self.settings.pipeline.merge_gap_seconds
                and merged_duration <= self.settings.pipeline.max_asr_segment_seconds
            )
            should_merge_short = (
                self._speaker_key(previous) == self._speaker_key(segment)
                and (
                    previous.duration < self.settings.pipeline.min_asr_segment_seconds
                    or segment.duration < self.settings.pipeline.min_asr_segment_seconds
                )
                and gap <= short_merge_gap
                and merged_duration <= self.settings.pipeline.max_asr_segment_seconds
            )
            if can_merge or should_merge_short:
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
    def _speaker_key(segment: TranscriptSegment) -> tuple[str, str]:
        if segment.speaker_id:
            return ("speaker_id", segment.speaker_id)
        if segment.speaker_name:
            return ("speaker_name", segment.speaker_name)
        return ("speaker_label", segment.speaker_label)
