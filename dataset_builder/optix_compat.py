"""Isolated Windows Dr.Jit 1.2.0 backport for the R600+ OptiX compiler bug.

The upstream fix adds an unused predicate output to warp shuffle instructions:
https://github.com/mitsuba-renderer/drjit-core/commit/82f56afd25fac4421b01ff73d7031bf776dff917

Do not replace installed packages or executable instructions. For ONE verified
Windows DLL, backport its three embedded PTX text templates to a cache copy and
preload that copy before importing Dr.Jit. Text slots retain their exact size,
so PE sections, offsets, exports and the C++ ABI remain unchanged. No propagation
settings, random seeds, JIT optimizations or ray counts are changed.
"""

from __future__ import annotations

import ctypes
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading


SOURCE_SHA256 = "450918d99d620d7755a237352a93a5edb03412751fc5d96a4b894fba8d9f8d89"
from citydeploy.paths import workspace_root

CACHE_ROOT = workspace_root() / ".cache" / "runtime_compat"
_state: dict | None = None
_library = None  # Retain the DLL handle for the lifetime of the process.
_dll_directory = None
_lock = threading.Lock()


def patch_ptx_templates(source: bytes) -> bytes:
    """Apply the upstream text-only fix, rejecting any unverified DLL."""
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Unknown Dr.Jit DLL checksum; refusing to patch it.")
    result = bytearray(source)
    counts = []
    for match in re.finditer(rb"[^\x00]+", source):
        template = match.group()
        if b"shfl.sync." not in template:
            continue
        if not template.startswith((b".func reduce_$s_$s(", b".func (.param .u32 rv) reduce_inc_u32 ")):
            raise RuntimeError("Unexpected PTX template; refusing to patch it.")
        # Remove indentation only to make room in the original constant slot.
        fixed = re.sub(rb"(?m)^ +", b"", template)
        fixed, declarations = re.subn(
            rb"(\.reg \.pred %leader[^;]*);", rb"\1, %unused;", fixed
        )
        fixed, shuffles = re.subn(
            rb"(shfl\.sync\.(?:bfly|idx)\.b32 %\w+),", rb"\1|%unused,", fixed
        )
        if declarations != 1 or shuffles not in {1, 6, 12} or len(fixed) > len(template):
            raise RuntimeError("PTX backport validation failed; refusing to patch it.")
        # Spaces are legal PTX whitespace; preserve the original trailing NUL.
        result[match.start():match.end()] = fixed.ljust(len(template), b" ")
        counts.append(shuffles)
    if sorted(counts) != [1, 6, 12] or len(result) != len(source):
        raise RuntimeError("Expected exactly three PTX templates and 19 shuffle fixes.")
    return bytes(result)


def _driver_versions() -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return [line.strip() for line in completed.stdout.splitlines()
                if re.fullmatch(r"\d+\.\d+", line.strip())]
    except (OSError, subprocess.SubprocessError):
        return []


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        try:
            os.replace(temporary, path)
        except PermissionError:
            # Another process may have published and loaded the same DLL while
            # this one was preparing it. Never overwrite a loaded library.
            if not path.exists() or path.read_bytes() != data:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def _loaded_core_path() -> Path | None:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    handle = kernel.GetModuleHandleW("drjit-core.dll")
    if not handle:
        return None
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    kernel.GetModuleFileNameW.restype = ctypes.c_uint32
    buffer = ctypes.create_unicode_buffer(32768)
    if not kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
        raise ctypes.WinError(ctypes.get_last_error())
    return Path(buffer.value).resolve()


def prepare_optix_compatibility() -> dict:
    """Must run before any Dr.Jit/Mitsuba/Sionna import; idempotent and offline."""
    global _state, _library, _dll_directory
    with _lock:
        if _state is not None:
            return deepcopy(_state)
        state = {"applied": False}
        if sys.platform != "win32":
            state["reason"] = "not_windows"
        elif os.environ.get("CITYDEPLOY_OPTIX_COMPAT", "auto").lower() == "off":
            state["reason"] = "disabled_by_environment"
        else:
            drivers = _driver_versions()
            state["nvidia_driver_versions"] = drivers
            if drivers and all(int(version.split(".")[0]) < 600 for version in drivers):
                state["reason"] = "driver_not_affected"
            else:
                # If nvidia-smi is absent, this known-safe text fix still works.
                # Read distribution metadata without importing the extension.
                distribution = importlib.metadata.distribution("drjit")
                state["drjit_version"] = distribution.version
                if distribution.version != "1.2.0":
                    state["reason"] = "not_the_pinned_drjit_version"
                else:
                    original = Path(distribution.locate_file("drjit/drjit-core.dll")).resolve()
                    source = original.read_bytes()
                    fixed = patch_ptx_templates(source)
                    fixed_hash = hashlib.sha256(fixed).hexdigest()
                    folder = CACHE_ROOT / f"drjit-1.2.0-{fixed_hash[:16]}"
                    library_path = folder / "drjit-core.dll"
                    loaded = _loaded_core_path()
                    if loaded is not None and loaded != library_path.resolve():
                        raise RuntimeError(
                            "The original Dr.Jit DLL is already loaded. Restart Python and "
                            "call dataset_builder.optix_compat.prepare_optix_compatibility() "
                            "before importing drjit, mitsuba or sionna."
                        )
                    if library_path.exists():
                        if hashlib.sha256(library_path.read_bytes()).hexdigest() != fixed_hash:
                            raise RuntimeError(f"Corrupt OptiX compatibility cache: {library_path}")
                    else:
                        _atomic_write(library_path, fixed)
                    state.update(
                        applied=True, reason="upstream_shuffle_predicate_backport",
                        original_library=str(original), original_sha256=SOURCE_SHA256,
                        loaded_library=str(library_path), patched_sha256=fixed_hash,
                    )
                    if not (folder / "manifest.json").exists():
                        _atomic_write(folder / "manifest.json", json.dumps(state, indent=2).encode())
                    # nanothread.dll remains in the original wheel directory.
                    _dll_directory = os.add_dll_directory(str(original.parent))
                    _library = ctypes.CDLL(str(library_path))
                    if _loaded_core_path() != library_path.resolve():
                        raise RuntimeError("Windows did not load the verified compatibility DLL.")
                    print("[INFO] OptiX driver compatibility enabled (Dr.Jit 1.2.0; original environment unchanged).")
        _state = state
        return deepcopy(state)
