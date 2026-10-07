"""Lazy command dispatch for model learning, planning and physical evaluation."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import runpy
import sys

COMMANDS = {
    "doctor": ("citydeploy.doctor", "Check dependencies and optional radio backends"),
    "init": ("citydeploy.initialize", "Copy editable configurations into a workspace"),
    "demo": ("citydeploy.demo", "Train and search a small synthetic task, without Sionna"),
    "demo-data": ("citydeploy.demo_data", "Create a small analytic training dataset and scenes"),
    "radio-demo": ("citydeploy.radio_demo", "Verify a synthetic scene with real ray tracing"),
    "train": ("scripts.train_model", "Train one of the four utility models"),
    "plan": ("samplers.infer", "Save continuous search candidates (no physical verification)"),
    "evaluate": ("samplers.evaluate_single", "Search, repair and physically verify a deployment"),
    "benchmark": ("samplers.benchmark_fixed_tx", "Run a fixed-cardinality experiment plan"),
    "target": ("samplers.evaluate_target", "Search ascending TX counts for a coverage target"),
    "adapt": ("energy_model.adapt_rewards", "Run few-shot followed by full-pool adaptation"),
    "build-scene": ("scene_builder.generate", "Construct a scene from a geographic bounding box"),
    "build-data": ("dataset_builder.generate", "Simulate deployment families in supplied scenes"),
    "validate-data": ("dataset_builder.validate", "Validate dataset integrity and geographic splits"),
    "inspect-data": ("citydeploy.inspect_data", "Read schema, counts and model-input compatibility"),
    "manual": ("manual_deployment.run", "Place TXs interactively and verify their service"),
    "summarize": ("citydeploy.summarize", "Aggregate successful fixed-TX evaluation records"),
    "audit-release": ("citydeploy.audit", "Scan source files for private paths and artifacts"),
}
ALIASES = {"ws-g": "smc_gaussian", "ws-l": "smc_langevin", "br-snis": "br_snis"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="citydeploy", description=__doc__,
        epilog="Commands:\n" + "\n".join(f"  {k:16} {v[1]}" for k, v in COMMANDS.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, help="Input/output root (default: current directory)")
    parser.add_argument("--legacy-schema-prefix", help="Opt-in read-only compatibility for an earlier namespace")
    parser.add_argument("--version", action="version", version="CityDeploy-Bench 0.1.0")
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    if args.workspace:
        os.environ["CITYDEPLOY_WORKSPACE"] = str(args.workspace.expanduser().resolve())
    if args.legacy_schema_prefix:
        os.environ["CITYDEPLOY_LEGACY_SCHEMA_PREFIX"] = args.legacy_schema_prefix
    from citydeploy.paths import workspace_root
    # Relative explicit arguments and plan entries share one documented root.
    root = workspace_root()
    if not root.is_dir():
        parser.error("Workspace does not exist; create it before running commands.")
    os.chdir(root)
    forwarded = args.arguments
    for i, token in enumerate(forwarded):
        if token.startswith("--sampler="):
            value = token.split("=", 1)[1]
            forwarded[i] = "--sampler=" + ALIASES.get(value.lower(), value)
    for i, token in enumerate(forwarded[:-1]):
        if token == "--sampler":
            forwarded[i + 1] = ALIASES.get(forwarded[i + 1].lower(), forwarded[i + 1])
    module = COMMANDS[args.command][0]
    sys.argv = [f"citydeploy {args.command}", *forwarded]
    try:
        runpy.run_module(module, run_name="__main__")
    except ModuleNotFoundError as exc:
        parser.exit(2, f"Missing dependency {exc.name!r}. See docs/installation.md for core, radio and scene extras.\n")
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
