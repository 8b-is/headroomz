"""Crush MCP registrar (Crush + crush-dev-local).

Crush stores MCP server configuration under ``~/.config/crush/crush.json``
(or ``$CRUSH_CONFIG``) under the top-level ``mcp`` key using ``type: "stdio"``,
``command: "<bin>"``, and ``args: [...]``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .base import MCPRegistrar, RegisterResult, RegisterStatus, ServerSpec
from .opencode import (
    _diff_specs,
    _MalformedConfigError,
    _read_json,
    _read_json_for_write,
    _specs_equivalent,
    _write_json,
)


def crush_config_path() -> Path:
    """Return the active Crush configuration file path."""
    env_cfg = os.environ.get("CRUSH_CONFIG", "").strip()
    if env_cfg:
        return Path(env_cfg).expanduser()
    return Path.home() / ".config" / "crush" / "crush.json"


def _entry_to_spec(name: str, entry: dict[str, Any]) -> ServerSpec:
    command_value = entry.get("command")
    raw_args = entry.get("args")
    if isinstance(command_value, list):
        command = str(command_value[0]) if command_value else ""
        args = tuple(str(x) for x in command_value[1:])
    else:
        command = str(command_value) if command_value else ""
        args = tuple(str(x) for x in raw_args) if isinstance(raw_args, list) else ()
    env_value = entry.get("env", entry.get("environment", {}))
    env: dict[str, str] = {}
    if isinstance(env_value, dict):
        env = {str(k): str(v) for k, v in env_value.items()}
    return ServerSpec(name=name, command=command, args=args, env=env)


def _spec_to_entry(spec: ServerSpec) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": "stdio",
        "command": spec.command,
        "args": list(spec.args),
    }
    if spec.env:
        entry["env"] = dict(spec.env)
    return entry


class CrushRegistrar(MCPRegistrar):
    """Register MCP servers with Crush and crush-dev-local."""

    name = "crush"
    display_name = "Crush"

    def __init__(self, *, config_path: Path | None = None) -> None:
        self._config_path = config_path or crush_config_path()

    def detect(self) -> bool:
        for binary in ("crush", "crush-dev-local", "crush-love-dev"):
            if shutil.which(binary):
                return True
        return self._config_path.parent.is_dir()

    def get_server(self, server_name: str) -> ServerSpec | None:
        data = _read_json(self._config_path)
        mcp = data.get("mcp", {})
        if not isinstance(mcp, dict):
            return None
        entry = mcp.get(server_name)
        if not isinstance(entry, dict):
            return None
        return _entry_to_spec(server_name, entry)

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        existing = self.get_server(spec.name)

        if existing is not None and _specs_equivalent(existing, spec):
            return RegisterResult(RegisterStatus.ALREADY, "matches current configuration")

        if existing is not None and not force:
            return RegisterResult(
                RegisterStatus.MISMATCH,
                _diff_specs(existing, spec),
            )

        if existing is not None and force:
            self.unregister_server(spec.name)

        return self._write_entry(spec)

    def unregister_server(self, server_name: str) -> bool:
        data = _read_json(self._config_path)
        mcp = data.get("mcp", {})
        if not isinstance(mcp, dict) or server_name not in mcp:
            return False
        del mcp[server_name]
        if not mcp:
            data.pop("mcp", None)
        try:
            _write_json(self._config_path, data)
        except OSError:
            return False
        return True

    def _write_entry(self, spec: ServerSpec) -> RegisterResult:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            data = _read_json_for_write(self._config_path)
            mcp = data.setdefault("mcp", {})
            if not isinstance(mcp, dict):
                mcp = {}
                data["mcp"] = mcp
            mcp[spec.name] = _spec_to_entry(spec)
            _write_json(self._config_path, data)
        except _MalformedConfigError as exc:
            return RegisterResult(
                RegisterStatus.FAILED,
                f"{self._config_path} exists but is not valid JSON ({exc}); "
                "refusing to overwrite. Fix or remove the file, then re-run.",
            )
        except OSError as exc:
            return RegisterResult(
                RegisterStatus.FAILED, f"could not write {self._config_path}: {exc}"
            )
        return RegisterResult(RegisterStatus.REGISTERED, f"wrote to {self._config_path}")
