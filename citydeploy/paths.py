"""Workspace-relative inputs and outputs, independent of the installation path."""
from __future__ import annotations
import os
from pathlib import Path


def workspace_root() -> Path:
    return Path(os.environ.get("CITYDEPLOY_WORKSPACE", Path.cwd())).expanduser().resolve()


def config_path(name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Built-in configuration names must be relative and cannot contain '..'.")
    local = workspace_root() / "configs" / relative
    return local if local.is_file() else Path(__file__).resolve().parent / "configs" / relative


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "configs":
        return config_path(str(Path(*path.parts[1:])))
    return (workspace_root() / path).resolve()
