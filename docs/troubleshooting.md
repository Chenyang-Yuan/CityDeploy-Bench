# Troubleshooting

## Platform validation

The initial source release passed the Linux automated unit tests, synthetic
model/sampler smoke test and package build. Its Windows CI run passed 87 of 88
unit tests. The remaining test compares a resolved temporary-directory path to
its Windows short-name alias. The two spellings can identify the same directory,
but that assertion compares path spellings. This is a known test portability
limitation, not evidence of a failed model computation. Windows is not currently
advertised as fully CI-validated; Linux is the validated automated baseline.

## Common issues

| Symptom | Check |
| --- | --- |
| `citydeploy` is not found | Activate the environment used for installation; `python -m citydeploy` is equivalent. |
| Dataset or scene is missing | Paths resolve from the working directory or global `--workspace`, not from the installed package. Public data are not bundled. |
| Several model paths are rejected | Use plural `--checkpoints`, not singular `--checkpoint`. |
| Training destination exists | Select a new `--model-root` and `--output-root`; existing weights are protected. |
| Schema is not recognized | Use data from this distribution; for a known earlier namespace, see the explicit read-only compatibility option in `data.md`. |
| Torch has no CUDA device | Run `citydeploy doctor`; verify the installed PyTorch build and NVIDIA driver. CPU model training still works. |
| LLVM initialization warning | This concerns the CPU radio backend. A successful CUDA/OptiX check is separate. Do not select CPU ray tracing without a usable LLVM backend. |
| Native OptiX/PTX compilation failure | Run `citydeploy doctor --radio --ray-device gpu` in a fresh process. Keep the pinned radio stack; retain the complete error and version report. |
| OSM/Overpass timeout | This is geographic data retrieval, not model training. Retry later or use the generator's configured endpoint options; do not silently replace missing geometry. |
| Training or ray tracing is slow/out of memory | First use a one-condition pilot. Reduce training batch size or search particles; record changed budgets. Lower RT rays only for smoke tests, not mixed benchmark comparisons. |
| Different coverage after an upgrade | Compare scene/RF hashes, backend versions, ray counts and seeds. Do not mix changed simulation contracts into one experiment. |

## Windows OptiX compatibility

`dataset_builder/optix_compat.py` implements a verified, text-template-only compatibility backport for one Dr.Jit 1.2.0 Windows DLL on newer driver families, following the [upstream shuffle-instruction fix](https://github.com/mitsuba-renderer/drjit-core/commit/82f56afd25fac4421b01ff73d7031bf776dff917). A cached copy is stored under the workspace's `.cache/runtime_compat/`; the installed library remains untouched. Unknown hashes, corrupt cached binaries, and an already-imported incompatible core are rejected. No arbitrary DLL patching is attempted.

Set `CITYDEPLOY_OPTIX_COMPAT=off` only to diagnose or when using a verified unaffected runtime. This does not repair an incompatible driver/compiler combination. An upstream-compatible stack should be validated as a unit before replacing the pinned versions.

The compatibility helper is not a solution to all native failures. Start from a fresh process, avoid loading multiple Dr.Jit/Mitsuba variants in the same interpreter, and retain a small reproducer. Do not disable propagation mechanisms to hide a compiler error in benchmark runs.

## Security and shared artifacts

Only load trusted model and scene files. Do not include API tokens, runtime logs, local paths, weights, private data or cache binaries in source uploads. Public dependency URLs and required third-party attributions are intentionally retained. Run the source audit and inspect the archive before publishing. Automated scanning is defense in depth, not proof that arbitrary future contributions contain no sensitive information.
