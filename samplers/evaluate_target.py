"""Sequential TX-count search with Sionna-verified stopping and resumable trials.

The returned count is the first successful count *tested by this search*, not a
proof of the globally minimal feasible deployment. Native method budgets differ.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

PROJECT_ROOT = workspace_root()
DEFAULT_PLAN = config_path("experiments/paris_07_hpem_target70.json")
COVERAGES = ("joint_4metric_coverage", "pathloss_coverage", "ss_rsrp_coverage",
             "sinr_coverage", "effective_throughput_coverage")
COSTS = ("num_reward_evaluations", "total_model_evaluations", "inference_seconds",
         "raytracing_seconds", "wall_seconds", "raytracing_calls", "nominal_launched_rays")
MAP_KEYS = ("path_loss_db", "ss_rsrp_dbm", "sinr_db", "effective_throughput_mbps", "joint", "raw_npz")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return resolve_path(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    # Atomic replacement prevents a stopped process from leaving half a summary.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_plan(plan: dict) -> None:
    target = plan["target_joint_coverage"]
    if not isinstance(target, (int, float)) or not math.isfinite(target) or not 0 < target <= 1:
        raise ValueError("target_joint_coverage must be a fraction in (0, 1], e.g. 0.70")
    low, high = plan["min_num_tx"], plan["max_num_tx"]
    if type(low) is not int or type(high) is not int or not 1 <= low <= high:
        raise ValueError("TX range must be positive integers with min <= max")
    seeds = plan["seeds"]
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("seeds must be nonempty uint32 integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    if not plan["scene"] or Path(plan["scene"]).name != plan["scene"]:
        raise ValueError("scene must be one scene directory name")
    for key in ("min_tx_distance_m", "nondeployable_weight", "minimum_distance_weight"):
        if not math.isfinite(plan[key]) or plan[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")


def prepare_manifest(plan_path: Path) -> dict:
    # Keep summary/test helpers free of the GPU stack. pyarrow must precede torch.
    import pyarrow  # noqa: F401
    import torch
    from energy_model.guidance import load_guidance_checkpoint
    from samplers.runner import config_to_dict, sampler_config_from_dict

    plan = read_json(plan_path)
    validate_plan(plan)
    for key in ("checkpoint", "scene_root", "rf_profile", "methods_config", "output_root"):
        plan[key] = str(resolve(plan[key]))
    kind, model, checkpoint = load_guidance_checkpoint(Path(plan["checkpoint"]), torch.device("cpu"))
    if plan["max_num_tx"] > model.config.max_num_tx:
        raise ValueError(f"max_num_tx exceeds checkpoint limit {model.config.max_num_tx}")
    methods_spec = read_json(Path(plan["methods_config"]))
    methods = copy.deepcopy(methods_spec["methods"])
    names = [method["sampler"] for method in methods]
    if not names or len(names) != len(set(names)):
        raise ValueError("methods must be nonempty and unique")
    for method in methods:
        if type(method["num_particles"]) is not int or method["num_particles"] <= 0:
            raise ValueError("num_particles must be a positive integer")
        method["config"] = config_to_dict(sampler_config_from_dict(method["sampler"], method["config"]))
    rf = read_json(Path(plan["rf_profile"]))
    rf["coverage_thresholds"]["joint_coverage_target"] = plan["target_joint_coverage"]
    scene_dir = Path(plan["scene_root"]) / plan["scene"]
    metadata = read_json(scene_dir / "inputs/metadata.json")
    scene_files = [scene_dir / "scene.xml"]
    for directory in ("inputs", "mesh"):
        scene_files.extend(p for p in (scene_dir / directory).rglob("*") if p.is_file())
    source_files = []
    for directory in ("samplers", "energy_model", "dataset_builder", "radio_backend", "citydeploy", "scene_builder", "visualization"):
        source_files.extend((Path(__file__).resolve().parents[1] / directory).glob("*.py"))
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "sionna-rt", "mitsuba", "drjit")}
    manifest = {
        "schema": "citydeploy.target-coverage", "plan": plan, "methods": methods,
        "budget_policy": methods_spec["budget_policy"], "model_kind": kind,
        "reward_model": "hpem" if kind == "hypergraph_potential" else kind,
        "model_config": checkpoint["model_config"], "rf_profile": rf,
        "map_size_m": [metadata["map_x_m"], metadata["map_y_m"]],
        "checkpoint_sha256": file_hash(Path(plan["checkpoint"])),
        "scene_sha256": {str(p.relative_to(scene_dir)): file_hash(p) for p in sorted(scene_files)},
        "code_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): file_hash(p) for p in sorted(source_files)},
        "package_versions": versions,
        "protocol": {
            "stopping": "Sionna joint_4metric_coverage >= target on road evaluation cells",
            "search": "independent full-deployment search at each ascending integer TX count",
            "verification": "one reward-selected, road-snapped plan per count; no RT reranking",
            "seed_coupling": "same seed for search and ray tracing, paired across methods and counts",
            "claim": "first successful tested count within range and native budget, not global optimum",
            "failure": "not_reached means no successful plan found within the tested range/budget",
            "uncertainty": "sample std across successful seeds only; report success rate separately",
            "timing": "GPU-synchronized inference; materialized RT; wall includes startup/load/save/plots",
            "evaluation_count": "num_reward_evaluations counts sampler calls by candidate, including MC/grad calls; total_model_evaluations also counts 2 invariance probes and 1 final prediction",
        },
    }
    manifest["fingerprint"] = fingerprint(manifest)
    return manifest


def summarize_chain(plan: dict, sampler: str, seed: int, trials: list[dict], error: bool = False) -> dict:
    by_count = {trial["num_tx"]: trial for trial in trials}
    tested, reached = [], None
    for count in range(plan["min_num_tx"], plan["max_num_tx"] + 1):
        if count not in by_count:
            break  # Never infer success/minimality across a missing smaller count.
        trial = by_count[count]
        tested.append(trial)
        if trial["joint_4metric_coverage"] >= plan["target_joint_coverage"]:
            reached = trial
            break
    complete_range = len(tested) == plan["max_num_tx"] - plan["min_num_tx"] + 1
    status = "reached" if reached else "not_reached" if complete_range else "error" if error else "pending"
    result = {
        "sampler": sampler, "seed": seed, "status": status,
        "target_joint_coverage": plan["target_joint_coverage"],
        "first_reached_num_tx": reached["num_tx"] if reached else None,
        "tested_counts": len(tested), "last_tested_num_tx": tested[-1]["num_tx"] if tested else None,
        "best_joint_coverage": max((t["joint_4metric_coverage"] for t in tested), default=None),
        "reached_result_json": reached["result_json"] if reached else None,
    }
    result.update({f"reached_{key}": reached[key] if reached else None for key in COVERAGES})
    result.update({f"cumulative_{key}": sum_known([t[key] for t in tested]) for key in COSTS})
    return result


def sum_known(values: list) -> float | None:
    return None if any(value is None for value in values) else sum(values)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if any(value is None for value in values):
        return None, None
    return (statistics.mean(values) if values else None,
            statistics.stdev(values) if len(values) > 1 else None)


def summarize_methods(methods: list[dict], chains: list[dict]) -> list[dict]:
    result = []
    for method in methods:
        rows = [row for row in chains if row["sampler"] == method["sampler"]]
        successful = [row for row in rows if row["status"] == "reached"]
        completed = [row for row in rows if row["status"] in ("reached", "not_reached")]
        mean, std = mean_std([row["first_reached_num_tx"] for row in successful])
        summary = {
            "sampler": method["sampler"], "num_particles": method["num_particles"],
            "planned_seeds": len(rows), "completed_seeds": len(completed),
            "successful_seeds": len(successful),
            "not_reached_seeds": sum(row["status"] == "not_reached" for row in rows),
            "error_seeds": sum(row["status"] == "error" for row in rows),
            "pending_seeds": sum(row["status"] == "pending" for row in rows),
            "success_rate_completed": len(successful) / len(completed) if completed else None,
            "first_reached_tx_mean_successful": mean, "first_reached_tx_std_successful": std,
        }
        for key in COVERAGES:
            mean, std = mean_std([row[f"reached_{key}"] for row in successful])
            summary[f"reached_{key}_mean_successful"] = mean
            summary[f"reached_{key}_std_successful"] = std
        for key in COSTS:
            summary[f"total_{key}"] = sum_known([row[f"cumulative_{key}"] for row in rows])
            mean, std = mean_std([row[f"cumulative_{key}"] for row in completed])
            summary[f"cumulative_{key}_mean_completed"] = mean
            summary[f"cumulative_{key}_std_completed"] = std
        result.append(summary)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def refresh_reports(root: Path, manifest: dict) -> dict:
    plan, trials, chains = manifest["plan"], [], []
    for method in manifest["methods"]:
        for seed in plan["seeds"]:
            chain_dir = root / method["sampler"] / f"seed_{seed}"
            chain_trials = [read_json(p) for p in sorted(chain_dir.glob("ntx_*/trial.json"))]
            for trial in chain_trials:
                if trial["fingerprint"] != manifest["fingerprint"]:
                    raise ValueError(f"Trial fingerprint mismatch: {chain_dir}")
            trials.extend(chain_trials)
            chains.append(summarize_chain(plan, method["sampler"], seed, chain_trials,
                                          any(chain_dir.glob("ntx_*/error.json"))))
    methods = summarize_methods(manifest["methods"], chains)
    complete = all(row["status"] in ("reached", "not_reached") for row in chains)
    summary = {"status": "complete" if complete else "incomplete", "root": str(root),
               "budget_policy": manifest["budget_policy"], "num_trials": len(trials),
               "completed_chains": sum(m["completed_seeds"] for m in methods),
               "planned_chains": len(chains), "methods": methods}
    write_json(root / "summary.json", summary)
    write_csv(root / "trials.csv", trials)
    write_csv(root / "seed_summary.csv", chains)
    write_csv(root / "method_summary.csv", methods)
    return summary


def validate_result(result: dict, manifest: dict, method: dict, seed: int, count: int) -> None:
    plan = manifest["plan"]
    expected = {"scene_id": plan["scene"], "num_tx": count, "seed": seed, "ray_seed": seed,
                "sampler": method["sampler"], "num_particles": method["num_particles"],
                "sampler_config": method["config"], "rf_profile_snapshot": manifest["rf_profile"],
                "model_kind": manifest["model_kind"]}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Result mismatch for {key}")
    if resolve(result["checkpoint"]) != Path(plan["checkpoint"]):
        raise ValueError("Result checkpoint mismatch")
    for key in COVERAGES:
        if not math.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"Invalid coverage {key}={result[key]}")
    for key in ("inference_seconds", "raytracing_seconds", "num_reward_evaluations", "total_model_evaluations"):
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"Invalid budget {key}")
    if len(result["tx_positions_m"]) != count or len(result["tx_xy_norm01"]) != count:
        raise ValueError("Result TX coordinates have wrong cardinality")
    parent = Path(result["result_json"]).parent
    paths = [Path(result["radio_maps"][key]) for key in MAP_KEYS]
    paths += [parent / "candidates.npz", parent / "trace.json"]
    if any(not p.is_file() or p.stat().st_size == 0 for p in paths):
        raise ValueError("Result is missing maps, arrays, candidates, or trace")


def child_command(manifest: dict, root: Path, method: dict, seed: int, count: int, output: Path) -> list[str]:
    plan = manifest["plan"]
    command = [sys.executable, "-u", "-m", "samplers.evaluate_single", "--checkpoint", plan["checkpoint"],
               "--scene", plan["scene"], "--scene-root", plan["scene_root"], "--num-tx", str(count),
               "--sampler", method["sampler"], "--num-particles", str(method["num_particles"]),
               "--sampler-config", str(root / "method_configs" / f"{method['sampler']}.json"),
               "--rf-profile", str(root / "rf_profile_resolved.json"), "--seed", str(seed),
               "--num-seeds", "1", "--run-dir", str(output)]
    for key in ("device", "ray_device", "min_tx_distance_m", "nondeployable_weight", "minimum_distance_weight"):
        command.extend(["--" + key.replace("_", "-"), str(plan[key])])
    return command


def execute_trial(root: Path, manifest: dict, method: dict, seed: int, count: int) -> dict:
    case = root / method["sampler"] / f"seed_{seed}" / f"ntx_{count:02d}"
    case.mkdir(parents=True, exist_ok=True)
    if (case / "trial.json").exists():
        trial = read_json(case / "trial.json")
        if trial["fingerprint"] != manifest["fingerprint"]:
            raise ValueError(f"Trial fingerprint mismatch: {case}")
        validate_result(read_json(Path(trial["result_json"])), manifest, method, seed, count)
        return trial
    # Recover a completed child after the parent stopped before committing trial.json.
    recovered = []
    for path in sorted(case.glob("attempt_*/result.json")):
        run = read_json(path)
        validate_result(run, manifest, method, seed, count)
        recovered.append((path, run))
    if recovered:
        result_path, result = recovered[-1]
        timing_path = case / f"{result_path.parent.name}_timing.json"
        # Native result is authoritative even if the parent died before saving timing.
        # Missing wall time remains null, never fabricated or zero-filled.
        wall = read_json(timing_path)["wall_seconds"] if timing_path.exists() else None
    else:
        index = len(list(case.glob("attempt_*_command.json"))) + 1
        output = case / f"attempt_{index:03d}"
        command = child_command(manifest, root, method, seed, count, output)
        log_path = case / f"{output.name}.log"
        write_json(case / f"{output.name}_command.json", {"command": command, "fingerprint": manifest["fingerprint"]})
        print(f"RUN {method['sampler']} seed={seed} ntx={count}; log={log_path}", flush=True)
        started = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    returncode = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            wall = time.perf_counter() - started
            write_json(case / f"{output.name}_timing.json", {"wall_seconds": wall, "returncode": returncode})
            if returncode:
                raise RuntimeError(f"Child exited {returncode}; inspect {log_path}")
            result_path = output / "result.json"
            result = read_json(result_path)
            validate_result(result, manifest, method, seed, count)
        except BaseException as exc:
            write_json(case / "error.json", {"message": str(exc), "log": str(log_path), "attempt": output.name})
            raise
    plan = manifest["plan"]
    trial = {"scene_id": plan["scene"], "reward_model": manifest["reward_model"],
             "sampler": method["sampler"], "seed": seed, "num_tx": count,
             "target_joint_coverage": plan["target_joint_coverage"],
             "reached": result["joint_4metric_coverage"] >= plan["target_joint_coverage"],
             **{key: result[key] for key in COVERAGES}, "predicted_reward": result["predicted_reward"],
             "num_particles": method["num_particles"],
             **{key: result[key] for key in ("num_reward_evaluations", "total_model_evaluations", "inference_seconds", "raytracing_seconds")},
             "wall_seconds": wall, "raytracing_calls": 1,
             "nominal_launched_rays": count * manifest["rf_profile"]["ray_tracing"]["samples_per_tx"],
             "result_json": str(result_path), "source_parameters": method.get("source_result", ""),
             "budget_policy": manifest["budget_policy"], "fingerprint": manifest["fingerprint"]}
    write_json(case / "trial.json", trial)
    # Retain earlier failed attempts as audit artifacts; error marker is no longer active.
    if (case / "error.json").exists():
        (case / "error.json").replace(case / f"resolved_error_{result_path.parent.name}.json")
    print(f"DONE {method['sampler']} seed={seed} ntx={count} joint={100*trial['joint_4metric_coverage']:.2f}% "
          f"reached={trial['reached']} evals={trial['num_reward_evaluations']} "
          f"search={trial['inference_seconds']:.2f}s RT={trial['raytracing_seconds']:.2f}s wall={wall}s", flush=True)
    return trial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path, help="Exact experiment directory; validate identity and skip completed trials")
    parser.add_argument("--max-trials", type=int, default=0, help="Limit new trials for a pilot; 0 runs until complete")
    args = parser.parse_args()
    if args.max_trials < 0:
        parser.error("--max-trials must be >= 0")
    manifest = prepare_manifest(resolve(args.plan))
    plan = manifest["plan"]
    if args.dry_run:
        print(json.dumps({key: manifest[key] for key in ("plan", "methods", "budget_policy", "model_kind", "map_size_m", "protocol", "fingerprint")}, indent=2))
        return
    if args.resume:
        root = resolve(args.resume)
        previous = read_json(root / "experiment.json")
        if previous["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Resume identity changed (plan/config/checkpoint/scene/code/packages); start a separate experiment")
        if read_json(root / "rf_profile_resolved.json") != manifest["rf_profile"]:
            raise ValueError("Saved RF snapshot changed")
        for method in manifest["methods"]:
            if read_json(root / "method_configs" / f"{method['sampler']}.json") != method["config"]:
                raise ValueError("Saved method snapshot changed")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = (f"{plan['scene']}_{manifest['reward_model']}_target{100*plan['target_joint_coverage']:g}_"
                f"tx{plan['min_num_tx']}-{plan['max_num_tx']}_{stamp}")
        root = Path(plan["output_root"]) / name
        root.mkdir(parents=True, exist_ok=False)
        (root / "method_configs").mkdir()
        write_json(root / "experiment.json", manifest)
        write_json(root / "rf_profile_resolved.json", manifest["rf_profile"])
        for method in manifest["methods"]:
            write_json(root / "method_configs" / f"{method['sampler']}.json", method["config"])
    print(f"Experiment: {root}\nBudget policy: {manifest['budget_policy']}", flush=True)
    refresh_reports(root, manifest)
    new_trials = 0
    try:
        for method in manifest["methods"]:
            for seed in plan["seeds"]:
                for count in range(plan["min_num_tx"], plan["max_num_tx"] + 1):
                    existing = root / method["sampler"] / f"seed_{seed}" / f"ntx_{count:02d}" / "trial.json"
                    if not existing.exists() and args.max_trials and new_trials >= args.max_trials:
                        print(f"Pilot limit reached; resume with --resume {root}", flush=True)
                        return
                    is_new = not existing.exists()
                    trial = execute_trial(root, manifest, method, seed, count)
                    new_trials += int(is_new)
                    refresh_reports(root, manifest)
                    if trial["reached"]:
                        break
    finally:
        summary = refresh_reports(root, manifest)
        print(f"{summary['status']}: {summary['completed_chains']}/{summary['planned_chains']} seed chains, "
              f"{summary['num_trials']} trials. Summary: {root / 'method_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
