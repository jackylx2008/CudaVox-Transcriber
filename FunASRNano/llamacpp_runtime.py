"""llama.cpp server lifecycle helpers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from FunASRNano.schemas import LlamaCppSettings


class LlamaCppServerManager:
    def __init__(
        self,
        settings: LlamaCppSettings,
        logger,
        project_root: Path | None = None,
    ) -> None:
        self.settings = settings
        self.logger = logger
        self.project_root = project_root or Path.cwd()
        self.process: subprocess.Popen | None = None
        self.existing_pid: int | None = None

    def ensure_server(self) -> None:
        if self._is_ready():
            self.logger.info("Qwen3.6 llama.cpp 服务已可用: %s", self.settings.base_url)
            if self.settings.shutdown_existing_on_exit:
                self.existing_pid = self._listening_pid()
                if self.existing_pid:
                    self.logger.info(
                        "已记录现有 Qwen3.6 llama.cpp 服务 PID=%s，退出时将按配置关闭。",
                        self.existing_pid,
                    )
            return

        if not self.settings.autostart:
            raise RuntimeError(
                f"Qwen3.6 服务不可用，且 LLAMACPP_AUTOSTART=false: {self.settings.base_url}"
            )

        self._start_server()
        self._wait_until_ready()

    def shutdown_started_server(self) -> None:
        if self.process is None:
            self._shutdown_existing_server()
            return
        if self.process.poll() is not None:
            self.process = None
            self._shutdown_existing_server()
            return
        self.logger.info("关闭本次自动启动的 llama.cpp 服务。")
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=20)
        finally:
            self.process = None

    def _shutdown_existing_server(self) -> None:
        if not self.settings.shutdown_existing_on_exit:
            return
        pid = self.existing_pid or self._listening_pid()
        if not pid:
            self.logger.info("未找到需要关闭的现有 llama.cpp 服务。")
            return
        if self.process and self.process.pid == pid:
            return

        self.logger.info(
            "按 LLAMACPP_SHUTDOWN_EXISTING_ON_EXIT=true 关闭现有 llama.cpp 服务: PID=%s",
            pid,
        )
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self.existing_pid = None

    def _listening_pid(self) -> int | None:
        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None

        port_suffix = f":{self.settings.port}"
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            protocol, local_address, _remote_address, state, pid_text = parts[:5]
            if protocol.upper() != "TCP" or state.upper() != "LISTENING":
                continue
            if not local_address.endswith(port_suffix):
                continue
            try:
                return int(pid_text)
            except ValueError:
                return None
        return None

    def _start_server(self) -> None:
        server_path = self._resolve_path(self.settings.server_path)
        model_path = self._resolve_path(self.settings.model_path)
        if not server_path or not server_path.exists():
            raise FileNotFoundError(f"LLAMACPP_SERVER_PATH 不存在: {server_path}")
        if not model_path or not model_path.exists():
            raise FileNotFoundError(f"LLAMACPP_MODEL_PATH 不存在: {model_path}")

        command = [
            str(server_path),
            "-m",
            str(model_path),
            "--alias",
            self.settings.model,
            "-c",
            str(self.settings.ctx_size),
            "-ngl",
            str(self.settings.n_gpu_layers),
            "--host",
            self.settings.host,
            "--port",
            str(self.settings.port),
        ]
        mmproj_path = self._resolve_path(self.settings.mmproj_path)
        if mmproj_path:
            if not mmproj_path.exists():
                raise FileNotFoundError(f"LLAMACPP_MMPROJ_PATH 不存在: {mmproj_path}")
            command.extend(["--mmproj", str(mmproj_path)])
        if self.settings.reasoning:
            command.extend(["--reasoning", self.settings.reasoning])
        command.extend(["--reasoning-budget", str(self.settings.reasoning_budget)])

        log_dir = self.project_root / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "llama_server.out.log"
        stderr_path = log_dir / "llama_server.err.log"
        env = os.environ.copy()
        env["PATH"] = self._build_path(server_path.parent, env.get("PATH", ""))

        self.logger.info("自动启动 Qwen3.6 llama.cpp 服务: %s", self.settings.base_url)
        stdout_file = stdout_path.open("ab")
        stderr_file = stderr_path.open("ab")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            command,
            cwd=str(server_path.parent),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            creationflags=creationflags,
        )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.settings.startup_timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    "llama.cpp 服务启动后立即退出，查看 log/llama_server.err.log"
                )
            try:
                if self._is_ready():
                    self.logger.info("Qwen3.6 llama.cpp 服务启动完成。")
                    return
            except Exception as exc:  # noqa: BLE001 - keep polling startup diagnostics.
                last_error = str(exc)
            time.sleep(2)
        raise TimeoutError(
            "等待 llama.cpp 服务启动超时"
            f" ({self.settings.startup_timeout_seconds}s): {last_error}"
        )

    def _is_ready(self) -> bool:
        try:
            self._request_json(self._health_url(), timeout=3)
            models = self._request_json(self._url("models"), timeout=3)
        except Exception:
            return False
        model_ids = self._extract_model_ids(models)
        if self.settings.model not in model_ids:
            raise RuntimeError(
                f"Qwen3.6 服务模型不匹配: 期望 {self.settings.model}, 实际 {model_ids}"
            )
        return True

    def _build_path(self, server_dir: Path, existing_path: str) -> str:
        paths = [str(server_dir)]
        for raw_path in self.settings.extra_dll_dirs.split(";"):
            raw_path = raw_path.strip()
            if not raw_path:
                continue
            path = self._resolve_path(raw_path)
            if path:
                paths.append(str(path))
        if existing_path:
            paths.append(existing_path)
        return os.pathsep.join(paths)

    def _resolve_path(self, value: str) -> Path | None:
        value = value.strip()
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _health_url(self) -> str:
        base_url = self.settings.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            return base_url[:-3] + "/health"
        return base_url + "/health"

    def _url(self, endpoint: str) -> str:
        return f"{self.settings.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    @staticmethod
    def _request_json(url: str, timeout: int) -> Any:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _extract_model_ids(payload: Any) -> list[str]:
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if not isinstance(data, list):
            return []
        model_ids = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                model_ids.append(item["id"])
        return model_ids
