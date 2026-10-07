"""Audit a clean source distribution for private paths and generated artifacts."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import re
import struct
import zlib

TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".cff", ".txt", ".svg", ".in"}
FORBIDDEN_PARTS = {".git", "__pycache__", ".cache", ".venv", "node_modules", "datasets", "outputs", "runs", "dist", "build"}
FORBIDDEN_SUFFIXES = {".pyc", ".pt", ".pth", ".ckpt", ".log", ".npz", ".npy", ".parquet", ".dll", ".zip"}
FIGURE_ASSETS = {"docs/assets/benchmark_overview.png", "docs/assets/scene_morphology.png",
                 "docs/assets/deployment_example.png"}
PATTERNS = {
    "absolute_windows_path": re.compile(r"\b[A-Za-z]:[\\/]"),
    "private_unix_path": re.compile(r"/(?:Users|home)/[\w.-]+"),
    "email_address": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "credential": re.compile(r"(?:gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_]{20,}"),
    "recorded_run_timestamp": re.compile(r"\b20\d{6}(?:T\d{6}Z|_\d{6})\b"),
}


def inspect_figure(path: Path):
    """Accept static PNG pixels only; reject metadata, animation and appended data."""
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "invalid_figure"
    offset, chunks, width, height = 8, [], 0, 0
    while offset < len(data):
        if offset + 12 > len(data):
            return "invalid_figure"
        size = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + size
        if end > len(data):
            return "invalid_figure"
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:end - 4]
        checksum = struct.unpack_from(">I", data, end - 4)[0]
        if zlib.crc32(kind + body) & 0xffffffff != checksum:
            return "invalid_figure"
        if kind not in {b"IHDR", b"IDAT", b"IEND"}:
            return "figure_metadata_or_unreviewed_chunk"
        if kind == b"IHDR":
            if chunks or size != 13:
                return "invalid_figure"
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", body)
            if not (0 < width <= 10000 and 0 < height <= 10000) or (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                return "invalid_figure"
        elif not chunks:
            return "invalid_figure"
        elif kind == b"IEND":
            if size or end != len(data) or b"IDAT" not in chunks:
                return "invalid_figure"
        chunks.append(kind)
        offset = end
    if not chunks or chunks[-1] != b"IEND":
        return "invalid_figure"
    return None


def inspect_source(root: Path):
    root = Path(root).resolve()
    findings, count = [], 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        config_directory = relative.parts[:2] == ("citydeploy", "configs") or relative.parts[:1] == ("configs",)
        forbidden_location = any(part in FORBIDDEN_PARTS and not (part == "datasets" and config_directory)
                                 for part in relative.parts)
        if path.is_symlink():
            findings.append({"file": relative.as_posix(), "kind": "symlink"})
            continue
        if path.is_dir():
            if (path.name in FORBIDDEN_PARTS and not (path.name == "datasets" and config_directory)) or path.name.endswith(".egg-info"):
                findings.append({"file": relative.as_posix(), "kind": "generated_or_private_directory"})
            continue
        count += 1
        if forbidden_location or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append({"file": relative.as_posix(), "kind": "generated_or_private_artifact"})
            continue
        if relative.as_posix() in FIGURE_ASSETS:
            issue = inspect_figure(path)
            if issue:
                findings.append({"file": relative.as_posix(), "kind": issue})
        elif path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            try:
                source = path.read_text(encoding="utf-8")
                if path.suffix == ".py":
                    ast.parse(source, filename=relative.as_posix())
            except (UnicodeError, SyntaxError) as exc:
                findings.append({"file": relative.as_posix(), "kind": type(exc).__name__})
                continue
            for label, pattern in PATTERNS.items():
                for match in pattern.finditer(source):
                    findings.append({"file": relative.as_posix(), "kind": label,
                                     "line": source.count("\n", 0, match.start()) + 1})
        else:
            findings.append({"file": relative.as_posix(), "kind": "unreviewed_file_type"})
    return {"passed": bool(count) and not findings, "files_checked": count, "findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    report = inspect_source(args.root)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
