"""Runtime helpers."""

from __future__ import annotations

from logging_config import get_logger

LOGGER = get_logger(__name__)


def resolve_device(preferred: str, allow_cpu_fallback: bool = True) -> str:
    LOGGER.info(
        "开始解析推理设备: preferred=%s, allow_cpu_fallback=%s",
        preferred,
        allow_cpu_fallback,
    )
    try:
        import torch
    except ImportError as exc:
        if preferred.startswith("cuda"):
            LOGGER.exception("未安装 PyTorch，无法按请求启用 CUDA。")
            raise RuntimeError(
                "当前未安装 PyTorch，无法启用 CUDA。请先安装对应 CUDA 版本的 torch/torchaudio。"
            ) from exc
        LOGGER.warning("未安装 PyTorch，回退到 CPU。")
        return "cpu"

    if preferred.startswith("cuda"):
        if torch.cuda.is_available():
            LOGGER.info("检测到可用 CUDA，使用设备: %s", preferred)
            return preferred
        if allow_cpu_fallback:
            LOGGER.warning("未检测到可用 CUDA，已自动回退到 CPU。")
            return "cpu"
        raise RuntimeError("当前未检测到可用 CUDA 设备，且已禁用 CPU 回退。")

    resolved = preferred or ("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info("使用推理设备: %s", resolved)
    return resolved
