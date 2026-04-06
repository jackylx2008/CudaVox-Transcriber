"""Transcript serialization helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path

from FunASRNano.audio import format_timestamp
from FunASRNano.schemas import OutputSettings, TranscriptDocument, TranscriptSegment

TIMECODE_LINE_PATTERN = re.compile(
    r"^\s*(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*$"
)


def segment_to_payload(
    segment: TranscriptSegment,
    include_legacy_keys: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "start": segment.start,
        "end": segment.end,
        "duration": round(segment.duration, 3),
        "speaker_label": segment.speaker_label,
        "speaker_id": segment.speaker_id,
        "speaker_name": segment.speaker_name,
        "speaker_similarity": segment.speaker_similarity,
        "segment_audio_path": segment.segment_audio_path,
        "text": segment.text,
        "source": segment.source,
    }
    if segment.extras:
        payload["extras"] = segment.extras
    if include_legacy_keys:
        payload["local_speaker"] = segment.speaker_label
    return payload


def document_to_payload(
    document: TranscriptDocument,
    include_legacy_keys: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "input_file": document.input_file,
        "normalized_wav": document.normalized_wav,
        "device": document.device,
        "segment_count": document.segment_count,
        "segments": [
            segment_to_payload(segment, include_legacy_keys=include_legacy_keys)
            for segment in document.segments
        ],
        "raw_segments": [
            segment_to_payload(segment, include_legacy_keys=include_legacy_keys)
            for segment in document.raw_segments
        ],
    }
    if document.metadata:
        payload["metadata"] = document.metadata
    return payload


def _display_speaker(segment: TranscriptSegment) -> str:
    return segment.speaker_name or segment.speaker_id or segment.speaker_label


def write_transcript_json(document: TranscriptDocument, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.write_text(
        json.dumps(document_to_payload(document), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def write_transcript_txt(document: TranscriptDocument, output_path: str | Path) -> Path:
    target = Path(output_path)
    lines: list[str] = []
    for index, segment in enumerate(document.segments, start=1):
        speaker = _display_speaker(segment)
        lines.append(str(index))
        if speaker:
            lines.append(
                f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)} "
                f"（{speaker}）"
            )
        else:
            lines.append(
                f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}"
            )
        lines.append("")
        lines.append(segment.text or "")
        lines.append("")
    target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return target


def write_transcript_srt(document: TranscriptDocument, output_path: str | Path) -> Path:
    target = Path(output_path)
    blocks: list[str] = []
    for index, segment in enumerate(document.segments, start=1):
        blocks.append(str(index))
        blocks.append(
            f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}"
        )
        speaker = _display_speaker(segment)
        if speaker:
            blocks.append(speaker)
        blocks.append(segment.text or "")
        blocks.append("")
    target.write_text("\n".join(blocks).strip() + "\n", encoding="utf-8")
    return target


def write_transcript_outputs(
    document: TranscriptDocument,
    output_dir: str | Path,
    output_stem: str,
    settings: OutputSettings,
    logger,
) -> dict[str, str]:
    target_dir = Path(output_dir)
    written_files: dict[str, str] = {}
    if settings.write_json:
        json_path = write_transcript_json(document, target_dir / f"{output_stem}.json")
        written_files["json"] = str(json_path.resolve())
        logger.info("已写出 JSON: %s", json_path.resolve())
    if settings.write_txt:
        txt_path = write_transcript_txt(document, target_dir / f"{output_stem}.txt")
        written_files["txt"] = str(txt_path.resolve())
        logger.info("已写出 TXT: %s", txt_path.resolve())
    if settings.write_srt:
        srt_path = write_transcript_srt(document, target_dir / f"{output_stem}.srt")
        written_files["srt"] = str(srt_path.resolve())
        logger.info("已写出 SRT: %s", srt_path.resolve())
    return written_files


def parse_srt_timestamp(raw_value: str) -> float:
    hours = int(raw_value[0:2])
    minutes = int(raw_value[3:5])
    seconds = int(raw_value[6:8])
    millis = int(raw_value[9:12])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def parse_srt_text(raw_text: str) -> list[TranscriptSegment]:
    normalized = (
        raw_text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n").strip()
    )
    if not normalized:
        return []

    segments: list[TranscriptSegment] = []
    for block in re.split(r"\n\s*\n", normalized):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 2:
            continue

        cursor = 0
        if lines[0].isdigit():
            cursor = 1
        if cursor >= len(lines):
            continue

        match = TIMECODE_LINE_PATTERN.match(lines[cursor])
        if match is None:
            continue
        cursor += 1

        speaker_name = ""
        text_lines = lines[cursor:]
        if len(text_lines) >= 2:
            speaker_name = text_lines[0]
            text_lines = text_lines[1:]

        segments.append(
            TranscriptSegment(
                start=parse_srt_timestamp(match.group(1)),
                end=parse_srt_timestamp(match.group(2)),
                text="\n".join(text_lines).strip(),
                speaker_name=speaker_name,
                source="imported_srt",
            )
        )
    return segments


def load_srt_segments(srt_path: str | Path) -> list[TranscriptSegment]:
    return parse_srt_text(Path(srt_path).read_text(encoding="utf-8"))
