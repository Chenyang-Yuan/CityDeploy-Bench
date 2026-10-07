"""Inspect the installed runtime without requiring datasets or model weights."""
import argparse
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys


def inspect_core():
    packages = {}
    for name in ("numpy", "scipy", "torch", "pyarrow", "pandas", "matplotlib", "Pillow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {"python": platform.python_version(), "platform": platform.system(),
              "packages": packages, "passed": all(packages.values()) and (3, 10) <= sys.version_info[:2] < (3, 12)}
    if report["passed"]:
        try:
            import pyarrow
            import torch
            report.update(torch_cuda=torch.version.cuda, model_cuda_available=torch.cuda.is_available())
        except Exception as exc:
            report.update(passed=False, import_error=str(exc))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radio", action="store_true", help="Also load and check the independent Sionna backend")
    parser.add_argument("--ray-device", choices=["cpu", "gpu", "auto"], default="auto")
    args = parser.parse_args()
    report = inspect_core()
    if args.radio:
        # Isolate native-library failures so the diagnostics process survives.
        code = "import json; from dataset_builder.runtime import require_supported_runtime; print(json.dumps(require_supported_runtime(%r)))" % args.ray_device
        child = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=60)
        report["radio_passed"] = child.returncode == 0
        report["radio_details"] = child.stdout.strip() if child.returncode == 0 else child.stderr[-2500:]
        report["passed"] &= child.returncode == 0
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
