"""Detect the actual Sionna RT execution backend before an expensive run."""

from __future__ import annotations

import platform
from typing import Any

from dataset_builder.optix_compat import prepare_optix_compatibility


def detect_sionna_runtime(requested_device: str = "auto") -> dict[str, Any]:
    compatibility = prepare_optix_compatibility()
    import drjit as dr
    import mitsuba as mi

    requested = requested_device.strip().lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unknown Sionna RT device: {requested_device}")
    if mi.variant() is None and requested == "cpu":
        mi.set_variant("llvm_ad_mono_polarized")
    elif mi.variant() is None and requested == "gpu":
        try:
            mi.set_variant("cuda_ad_mono_polarized")
        except ImportError as exc:
            raise RuntimeError("GPU mode requested, but the Mitsuba CUDA variant is unavailable.") from exc
    import sionna
    import sionna.rt  # noqa: F401 - initializes the Mitsuba variant

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "sionna_version": getattr(sionna, "__version__", "unknown"),
        "mitsuba_version": getattr(mi, "__version__", "unknown"),
        "drjit_version": getattr(dr, "__version__", "unknown"),
        "optix_compatibility": compatibility,
        "requested_device": requested,
        "mitsuba_variant": mi.variant(),
        "cuda_backend_available": bool(dr.has_backend(dr.JitBackend.CUDA)),
        "llvm_backend_available": bool(dr.has_backend(dr.JitBackend.LLVM)),
    }


def require_supported_runtime(requested_device: str = "auto", require_cuda: bool = False) -> dict[str, Any]:
    info = detect_sionna_runtime(requested_device=requested_device)
    if not info["cuda_backend_available"] and not info["llvm_backend_available"]:
        raise RuntimeError("Neither the CUDA nor LLVM Dr.Jit backend is available for Sionna RT.")
    if (require_cuda or requested_device == "gpu") and (
        not info["cuda_backend_available"] or "cuda" not in str(info["mitsuba_variant"])
    ):
        raise RuntimeError(
            "CUDA production mode was requested, but Dr.Jit has no CUDA backend. "
            "Use Linux with an NVIDIA GPU for bulk generation, or omit --require-cuda for a CPU smoke test."
        )
    return info


def gpu_compiler_self_test() -> dict[str, Any]:
    """Compile and check the three affected PTX templates, without any dataset writes."""
    prepare_optix_compatibility()
    import drjit as dr
    import mitsuba as mi
    import numpy as np

    if "cuda" not in str(mi.variant()):
        raise RuntimeError("The OptiX self-test requires an initialized CUDA Mitsuba variant.")
    checked = []
    with dr.scoped_set_flag(dr.JitFlag.ForceOptiX, True):
        # Both full and partially active warps, with contended and distinct bins.
        for count, bins in ((64, 1), (257, 11)):
            index_np = np.arange(count, dtype=np.uint32) % bins
            active_np = np.arange(count) % 5 != 1
            index = mi.UInt(index_np)
            active = mi.Bool(active_np)
            target = dr.zeros(mi.UInt, bins)
            previous = dr.scatter_inc(target, index, active)
            dr.eval(target, previous)
            dr.sync_thread()
            expected = np.bincount(index_np[active_np], minlength=bins)
            np.testing.assert_array_equal(np.asarray(target), expected)
            previous_np = np.asarray(previous)
            for slot in range(bins):
                np.testing.assert_array_equal(
                    np.sort(previous_np[active_np & (index_np == slot)]),
                    np.arange(expected[slot]),
                )
        checked.append("scatter_inc_u32")
        for dtype in (mi.Float, dr.cuda.ad.Float64):
            values = dtype(np.arange(257) % 7 + 1)
            target = dr.zeros(dtype, 11)
            index = mi.UInt(np.arange(257, dtype=np.uint32) % 11)
            active = mi.Bool(np.arange(257) % 5 != 1)
            dr.scatter_reduce(dr.ReduceOp.Add, target, values, index, active, mode=dr.ReduceMode.Local)
            dr.eval(target)
            dr.sync_thread()
            expected = np.bincount(
                np.arange(257)[np.arange(257) % 5 != 1] % 11,
                weights=(np.arange(257) % 7 + 1)[np.arange(257) % 5 != 1], minlength=11,
            )
            np.testing.assert_array_equal(np.asarray(target), expected)
            checked.append(f"scatter_add_{dtype.__name__}")
    return {"passed": True, "checked": checked}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Check Sionna RT and its actual GPU compiler.")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="gpu")
    parser.add_argument("--self-test", action="store_true", help="Compile and verify OptiX atomic kernels")
    args = parser.parse_args()
    info = require_supported_runtime(args.device)
    if args.self_test:
        info["gpu_compiler_self_test"] = gpu_compiler_self_test()
    print(json.dumps(info, indent=2))
