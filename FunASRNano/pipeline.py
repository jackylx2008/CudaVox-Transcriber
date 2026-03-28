"""Main transcription pipeline."""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from logging_config import get_logger, setup_logger

from FunASRNano.audio import (
    build_profile_wav,
    convert_to_wav,
    ensure_dir,
    extract_wav_segment,
    format_timestamp,
)
from FunASRNano.funasr_service import FunASRTranscriber
from FunASRNano.pyannote_service import PyannoteDiarizer
from FunASRNano.runtime import resolve_device
from FunASRNano.schemas import DiarizedSegment, Settings
from FunASRNano.voiceprint_service import VoiceprintStore


class CudaVoxPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        level = getattr(logging, settings.app.log_level.upper(), logging.INFO)
        setup_logger(log_level=level)
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
            segment_path = temp_dir / f"segment_{index:04d}_{segment.local_speaker}.wav"
            extract_wav_segment(normalized_wav, segment.start, segment.end, segment_path)
            segment.segment_wav = str(segment_path)

        identities = self._resolve_voiceprints(
            normalized_wav=normalized_wav,
            segments=segments,
            temp_dir=temp_dir,
            source_file=input_file.name,
        )

        for segment in segments:
            identity = identities.get(segment.local_speaker)
            if identity is None:
                identity = self.voiceprints.create_transient_identity(segment.local_speaker)

            if segment.segment_wav:
                segment.text = self.transcriber.transcribe(segment.segment_wav)
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
        written_files = self._write_outputs(
            input_file=input_file,
            output_dir=output_dir,
            normalized_wav=normalized_wav,
            segments=merged_segments,
            raw_segments=raw_segments,
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
        segments: list[DiarizedSegment],
        temp_dir: Path,
        source_file: str,
    ):
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for segment in segments:
            grouped[segment.local_speaker].append((segment.start, segment.end))

        self.logger.info("开始声纹归一化匹配，本地说话人数=%s", len(grouped))
        identities = {}
        for local_speaker, spans in grouped.items():
            profile_path = temp_dir / f"profile_{local_speaker}.wav"
            total_seconds = build_profile_wav(
                normalized_wav,
                spans,
                profile_path,
                max_duration_seconds=self.settings.campp.max_profile_audio_seconds,
            )
            if total_seconds < self.settings.campp.min_profile_seconds:
                self.logger.warning(
                    "说话人 %s 可用时长不足 %.2fs，改用临时身份。",
                    local_speaker,
                    self.settings.campp.min_profile_seconds,
                )
                identities[local_speaker] = self.voiceprints.create_transient_identity(
                    local_speaker
                )
                continue

            embedding = self.voiceprints.extract_embedding(profile_path)
            identity = self.voiceprints.match_or_register(
                embedding=embedding,
                source_file=source_file,
                local_speaker=local_speaker,
            )
            identities[local_speaker] = identity
            self.logger.info(
                "说话人 %s 归属为 %s，相似度=%s，新建=%s",
                local_speaker,
                identity.speaker_id,
                identity.similarity,
                identity.is_new,
            )
        return identities

    def _merge_segments(self, segments: list[DiarizedSegment]) -> list[DiarizedSegment]:
        if not segments:
            self.logger.warning("没有可合并的片段。")
            return []

        merged: list[DiarizedSegment] = []
        for current in sorted(segments, key=lambda item: (item.start, item.end)):
            segment = replace(current)
            if not merged:
                merged.append(segment)
                continue

            previous = merged[-1]
            gap = segment.start - previous.end
            if (
                previous.speaker_id == segment.speaker_id
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

    def _write_outputs(
        self,
        input_file: Path,
        output_dir: Path,
        normalized_wav: Path,
        segments: list[DiarizedSegment],
        raw_segments: list[DiarizedSegment],
    ) -> dict[str, str]:
        payload = {
            "input_file": str(input_file.resolve()),
            "normalized_wav": str(normalized_wav.resolve()),
            "device": self.device,
            "segment_count": len(segments),
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "duration": round(segment.duration, 3),
                    "local_speaker": segment.local_speaker,
                    "speaker_id": segment.speaker_id,
                    "speaker_name": segment.speaker_name,
                    "speaker_similarity": segment.speaker_similarity,
                    "text": segment.text,
                }
                for segment in segments
            ],
            "raw_segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "duration": round(segment.duration, 3),
                    "local_speaker": segment.local_speaker,
                    "speaker_id": segment.speaker_id,
                    "speaker_name": segment.speaker_name,
                    "speaker_similarity": segment.speaker_similarity,
                    "text": segment.text,
                }
                for segment in raw_segments
            ],
        }

        written_files: dict[str, str] = {}
        if self.settings.output.write_json:
            json_path = output_dir / f"{input_file.stem}.json"
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written_files["json"] = str(json_path.resolve())
            self.logger.info("已写出 JSON: %s", json_path.resolve())

        if self.settings.output.write_txt:
            txt_path = output_dir / f"{input_file.stem}.txt"
            txt_lines = []
            for index, segment in enumerate(segments, start=1):
                txt_lines.append(str(index))
                txt_lines.append(
                    f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)} "
                    f"（{segment.speaker_name}）"
                )
                txt_lines.append("")
                txt_lines.append(segment.text or "")
                txt_lines.append("")
            txt_path.write_text("\n".join(txt_lines).strip() + "\n", encoding="utf-8")
            written_files["txt"] = str(txt_path.resolve())
            self.logger.info("已写出 TXT: %s", txt_path.resolve())

        if self.settings.output.write_srt:
            srt_path = output_dir / f"{input_file.stem}.srt"
            blocks = []
            for index, segment in enumerate(segments, start=1):
                blocks.append(str(index))
                blocks.append(
                    f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}"
                )
                blocks.append(segment.speaker_name)
                blocks.append(segment.text or "")
                blocks.append("")
            srt_path.write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")
            written_files["srt"] = str(srt_path.resolve())
            self.logger.info("已写出 SRT: %s", srt_path.resolve())

        return written_files
