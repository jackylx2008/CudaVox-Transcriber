"""Audio utilities."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from FunASRNano.logging_config import get_logger

LOGGER = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("确保目录存在: %s", target.resolve())
    return target


def ensure_ffmpeg(ffmpeg_bin: str = "ffmpeg") -> None:
    if shutil.which(ffmpeg_bin) is None:
        raise RuntimeError(
            f"找不到 `{ffmpeg_bin}`，请先安装 ffmpeg 并确保它在 PATH 中。"
        )
    LOGGER.debug("检测到 ffmpeg 可执行文件: %s", ffmpeg_bin)


def convert_to_wav(
    input_path: str | Path,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    ensure_ffmpeg(ffmpeg_bin)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "开始音频标准化: input=%s, output=%s, sample_rate=%s, channels=%s",
        Path(input_path).resolve(),
        output_file.resolve(),
        sample_rate,
        channels,
    )

    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        str(output_file),
    ]
    LOGGER.debug("执行 ffmpeg 命令: %s", " ".join(command))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 转码失败: {result.stderr.strip() or result.stdout.strip()}"
        )
    LOGGER.info("音频标准化完成: %s", output_file.resolve())
    return output_file


def extract_wav_segment(
    wav_path: str | Path,
    start: float,
    end: float,
    output_path: str | Path,
) -> Path:
    import soundfile as sf

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(str(wav_path)) as audio_file:
        sample_rate = audio_file.samplerate
        start_frame = max(0, int(start * sample_rate))
        frame_count = max(1, int((end - start) * sample_rate))
        audio_file.seek(start_frame)
        data = audio_file.read(frame_count, dtype="float32", always_2d=False)

    sf.write(str(output_file), data, sample_rate)
    LOGGER.debug(
        "导出切片音频: %s [%.3fs - %.3fs]",
        output_file.resolve(),
        start,
        end,
    )
    return output_file


def cut_audio_clip(
    input_path: str | Path,
    start: float,
    duration: float,
    output_path: str | Path,
    ffmpeg_bin: str = "ffmpeg",
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    ensure_ffmpeg(ffmpeg_bin)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        str(output_file),
    ]
    LOGGER.debug("执行 ffmpeg 切片命令: %s", " ".join(command))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 切片失败: {result.stderr.strip() or result.stdout.strip()}"
        )

    LOGGER.debug(
        "导出音频切片: %s [%.3fs + %.3fs]",
        output_file.resolve(),
        start,
        duration,
    )
    return output_file


def build_profile_wav(
    wav_path: str | Path,
    spans: Iterable[tuple[float, float]],
    output_path: str | Path,
    max_duration_seconds: float = 30.0,
) -> float:
    import numpy as np
    import soundfile as sf

    chunks = []
    total_seconds = 0.0
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(str(wav_path)) as audio_file:
        sample_rate = audio_file.samplerate
        for start, end in spans:
            if total_seconds >= max_duration_seconds:
                break
            remain = max_duration_seconds - total_seconds
            actual_end = min(end, start + remain)
            start_frame = max(0, int(start * sample_rate))
            frame_count = max(1, int((actual_end - start) * sample_rate))
            audio_file.seek(start_frame)
            data = audio_file.read(frame_count, dtype="float32", always_2d=False)
            if len(data) == 0:
                continue
            chunks.append(data)
            total_seconds += frame_count / sample_rate

    if not chunks:
        sf.write(str(output_file), np.zeros(1600, dtype="float32"), 16000)
        LOGGER.warning("未能为声纹建模提取到有效音频，写入静音占位文件: %s", output_file)
        return 0.0

    merged = np.concatenate(chunks, axis=0)
    sf.write(str(output_file), merged, sample_rate)
    LOGGER.debug(
        "生成声纹 profile 音频: %s, 时长=%.3fs",
        output_file.resolve(),
        total_seconds,
    )
    return total_seconds


def list_audio_files(input_path: str | Path, supported_extensions: list[str]) -> list[Path]:
    target = Path(input_path)
    if not target.exists():
        raise FileNotFoundError(f"输入路径不存在: {target}")

    extensions = {ext.lower() for ext in supported_extensions}
    if target.is_file():
        if target.suffix.lower() not in extensions:
            raise ValueError(f"暂不支持的音频格式: {target.suffix}")
        LOGGER.info("检测到单文件输入: %s", target.resolve())
        return [target]

    files = sorted(
        path
        for path in target.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not files:
        raise FileNotFoundError(f"在目录 {target} 中没有找到可处理音频。")
    LOGGER.info("扫描输入目录完成: %s, 匹配文件数=%s", target.resolve(), len(files))
    return files


def resolve_audio_files(
    input_files: list[str],
    input_path: str | Path,
    supported_extensions: list[str],
) -> list[Path]:
    if input_files:
        LOGGER.info("优先使用配置中的指定文件列表。")
        resolved: list[Path] = []
        seen: set[str] = set()
        for item in input_files:
            for path in list_audio_files(item, supported_extensions):
                key = str(path.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                resolved.append(path)
        if not resolved:
            raise FileNotFoundError("INPUT_FILES 已配置，但未解析到可处理音频。")
        LOGGER.info("指定文件解析完成，最终待处理文件数=%s", len(resolved))
        return resolved

    return list_audio_files(input_path, supported_extensions)


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remain = divmod(total_ms, 3600_000)
    minutes, remain = divmod(remain, 60_000)
    sec, millis = divmod(remain, 1000)
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{millis:03d}"


def format_seconds(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, remain = divmod(total_ms, 3600_000)
    minutes, remain = divmod(remain, 60_000)
    sec, millis = divmod(remain, 1000)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}.{millis:03d}"
