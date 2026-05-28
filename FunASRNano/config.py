"""Configuration loader."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from logging_config import get_logger

from FunASRNano.schemas import (
    AppSettings,
    AsrSettings,
    CamppSettings,
    DeviceSettings,
    FunASRSettings,
    LlamaCppSettings,
    OutputSettings,
    PipelineSettings,
    PyannoteSettings,
    QwenAsrSettings,
    QwenTextSettings,
    SenseVoiceSettings,
    Settings,
)

ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")
DEFAULT_VOICEPRINT_NAME_MAP_ENV = "voiceprint_name_map.env"
LOGGER = get_logger(__name__)


def _expand_env_vars(raw_text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2)
        value = os.getenv(key)
        if value is None or value == "":
            return "" if default is None else default
        return value

    return ENV_PATTERN.sub(_replace, raw_text)


def _load_yaml(config_path: Path) -> dict[str, Any]:
    import yaml

    LOGGER.debug("读取 YAML 配置文件: %s", config_path.resolve())
    raw_text = config_path.read_text(encoding="utf-8")
    expanded = _expand_env_vars(raw_text)
    data = yaml.safe_load(expanded) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml 顶层必须是对象。")
    return data


def _merge_legacy_config(data: dict[str, Any]) -> dict[str, Any]:
    if "app" in data:
        return data

    app = {
        "log_level": data.get("log_level", "INFO"),
        "input_path": data.get("input_file", "./input"),
        "output_dir": data.get("output_path", "./output"),
    }
    return {"app": app}


def _drop_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if v not in ("", None)}


def _parse_input_files(raw_value: str) -> list[str]:
    parts = re.split(r"[;\r\n,]+", raw_value)
    return [part.strip() for part in parts if part.strip()]


def _parse_speaker_name_map(raw_value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not raw_value.strip():
        return mapping

    for part in re.split(r"[;\r\n]+", raw_value):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "VOICEPRINT_NAME_MAP 格式错误，必须使用 speaker_id:姓名，例如 "
                "speaker_0001:张三;speaker_0002:李四"
            )
        speaker_id, speaker_name = item.split(":", 1)
        speaker_id = speaker_id.strip()
        speaker_name = speaker_name.strip()
        if not speaker_id or not speaker_name:
            raise ValueError(
                "VOICEPRINT_NAME_MAP 格式错误，speaker_id 和 姓名 都不能为空。"
            )
        mapping[speaker_id] = speaker_name
    return mapping


def _parse_speaker_name_map_file(file_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        file_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("VOICEPRINT_NAME_MAP="):
            legacy_value = line.split("=", 1)[1].strip()
            mapping.update(_parse_speaker_name_map(legacy_value))
            continue

        if "=" not in line:
            raise ValueError(
                f"{file_path.name} 第 {line_number} 行格式错误，必须使用 "
                "speaker_id=姓名，例如 speaker_0001=张三"
            )

        speaker_id, speaker_name = line.split("=", 1)
        speaker_id = speaker_id.strip()
        speaker_name = speaker_name.strip()
        if not speaker_id or not speaker_name:
            raise ValueError(
                f"{file_path.name} 第 {line_number} 行格式错误，speaker_id 和 姓名 都不能为空。"
            )
        mapping[speaker_id] = speaker_name
    return mapping


def _resolve_secret_env_file(
    base_env_file: Path,
    secret_file_name: str = DEFAULT_VOICEPRINT_NAME_MAP_ENV,
) -> Path:
    secret_path = Path(secret_file_name)
    if secret_path.is_absolute():
        return secret_path
    return base_env_file.parent / secret_path


def load_settings(
    config_path: str | Path = "config.yaml",
    env_path: str | Path = "common.env",
) -> Settings:
    from dotenv import load_dotenv

    env_file = Path(env_path)
    if env_file.exists():
        LOGGER.info("加载环境变量文件: %s", env_file.resolve())
        load_dotenv(env_file, override=False)
    else:
        LOGGER.warning("环境变量文件不存在，跳过加载: %s", env_file.resolve())

    secret_env_file = _resolve_secret_env_file(env_file)

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_file}")

    LOGGER.info("加载配置文件: %s", config_file.resolve())
    data = _merge_legacy_config(_load_yaml(config_file))
    settings = Settings(
        app=AppSettings(**_drop_empty(data.get("app", {}))),
        device=DeviceSettings(**_drop_empty(data.get("device", {}))),
        asr=AsrSettings(**_drop_empty(data.get("asr", {}))),
        funasr=FunASRSettings(**_drop_empty(data.get("funasr", {}))),
        sensevoice=SenseVoiceSettings(**_drop_empty(data.get("sensevoice", {}))),
        qwen_asr=QwenAsrSettings(**_drop_empty(data.get("qwen_asr", {}))),
        qwen_text=QwenTextSettings(**_drop_empty(data.get("qwen_text", {}))),
        llamacpp=LlamaCppSettings(**_drop_empty(data.get("llamacpp", {}))),
        pyannote=PyannoteSettings(**_drop_empty(data.get("pyannote", {}))),
        campp=CamppSettings(**_drop_empty(data.get("campp", {}))),
        pipeline=PipelineSettings(**_drop_empty(data.get("pipeline", {}))),
        output=OutputSettings(**_drop_empty(data.get("output", {}))),
    )
    raw_input_files = os.getenv("INPUT_FILES", "").strip()
    if raw_input_files:
        settings.app.input_files = _parse_input_files(raw_input_files)
        LOGGER.info("从 .env 读取到待处理文件数量: %s", len(settings.app.input_files))
    raw_speaker_name_map = os.getenv("VOICEPRINT_NAME_MAP", "").strip()
    if raw_speaker_name_map:
        settings.campp.speaker_name_map = _parse_speaker_name_map(raw_speaker_name_map)
        LOGGER.info(
            "从环境变量读取到声纹姓名映射数量: %s",
            len(settings.campp.speaker_name_map),
        )
    if secret_env_file.exists():
        settings.campp.speaker_name_map = _parse_speaker_name_map_file(secret_env_file)
        LOGGER.info(
            "从私密映射文件读取到声纹姓名映射数量: %s",
            len(settings.campp.speaker_name_map),
        )
    LOGGER.info(
        "配置加载完成: input=%s, input_files=%s, output=%s, preferred_device=%s, log_level=%s, asr_backend=%s",
        settings.app.input_path,
        len(settings.app.input_files),
        settings.app.output_dir,
        settings.device.preferred,
        settings.app.log_level,
        settings.asr.backend,
    )
    return settings
