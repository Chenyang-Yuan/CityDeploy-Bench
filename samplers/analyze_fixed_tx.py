#!/usr/bin/env python3
"""Aggregate fixed-TX benchmark trials into paper-ready tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from citydeploy.paths import workspace_root, resolve_path, config_path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = workspace_root()
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "fixed_tx_seen"
METRICS = (
    "joint_4metric_coverage",
    "pathloss_coverage",
    "ss_rsrp_coverage",
    "sinr_coverage",
    "effective_throughput_coverage",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze the fixed-TX planning benchmark")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _mean_std(values):
    data = np.asarray(values, dtype=float)
    return float(data.mean()), float(data.std(ddof=1)) if len(data) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _paper_joint_rows(condition_summary: list[dict]) -> list[dict]:
    """Pivot per-condition joint coverage into a paper-friendly mean +/- std table."""
    conditions = defaultdict(dict)
    labels = {}
    for row in condition_summary:
        key = (row["city"], row["scene_id"], int(row["size_m"]), int(row["num_tx"]))
        planner_id = row["planner_id"]
        labels[planner_id] = row["planner_label"]
        conditions[key][planner_id] = (
            float(row["joint_4metric_coverage_mean"]),
            float(row["joint_4metric_coverage_std"]),
        )

    planner_ids = sorted(labels, key=lambda planner_id: labels[planner_id])
    output = []
    for (city, scene_id, size_m, num_tx), values in sorted(conditions.items()):
        record = {
            "city": city,
            "scene_id": scene_id,
            "size_m": size_m,
            "num_tx": num_tx,
        }
        for planner_id in planner_ids:
            if planner_id in values:
                mean, std = values[planner_id]
                record[labels[planner_id]] = f"{100 * mean:.2f} ± {100 * std:.2f}"
            else:
                record[labels[planner_id]] = ""
        output.append(record)
    return output


def _paper_metric_rows(condition_summary: list[dict]) -> list[dict]:
    """Keep all coverage metrics as formatted percentages for manuscript tables."""
    output = []
    for row in condition_summary:
        record = {
            key: row[key]
            for key in ("city", "scene_id", "size_m", "num_tx", "planner_id", "planner_label", "trials")
        }
        for metric in METRICS:
            mean = 100 * float(row[f"{metric}_mean"])
            std = 100 * float(row[f"{metric}_std"])
            record[metric] = f"{mean:.2f} ± {std:.2f}"
        output.append(record)
    return output


def _summary_rows(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)
    output = []
    for key, items in sorted(grouped.items()):
        record = dict(zip(group_keys, key))
        record["trials"] = len(items)
        for metric in METRICS:
            mean, std = _mean_std([item[metric] for item in items])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
        for metric in ("inference_seconds", "raytracing_seconds", "num_reward_evaluations"):
            mean, std = _mean_std([item[metric] for item in items])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
        output.append(record)
    return output


def _paired_rows(rows: list[dict], baseline: str, samples: int, seed: int) -> list[dict]:
    by_method = defaultdict(dict)
    labels = {}
    for row in rows:
        key = (row["scene_id"], int(row["num_tx"]), int(row["seed"]))
        by_method[row["planner_id"]][key] = float(row["joint_4metric_coverage"])
        labels[row["planner_id"]] = row["planner_label"]
    reference = by_method.get(baseline, {})
    rng = np.random.default_rng(seed)
    output = []
    for method, values in sorted(by_method.items()):
        common = sorted(set(reference) & set(values))
        if not common:
            continue
        difference = np.asarray([values[key] - reference[key] for key in common])
        if samples > 0:
            indices = rng.integers(0, len(difference), size=(samples, len(difference)))
            bootstrap = difference[indices].mean(1)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
        else:
            low = high = float("nan")
        output.append({
            "planner_id": method,
            "planner_label": labels[method],
            "baseline": baseline,
            "paired_trials": len(common),
            "joint_difference_mean": float(difference.mean()),
            "bootstrap_ci_low": float(low),
            "bootstrap_ci_high": float(high),
            "win_rate": float(np.mean(difference > 0)),
            "tie_rate": float(np.mean(difference == 0)),
        })
    return output


def _figures(output: Path, planner_summary: list[dict], size_summary: list[dict]) -> None:
    labels = [row["planner_label"] for row in planner_summary]
    joint = np.asarray([row["joint_4metric_coverage_mean"] for row in planner_summary])
    runtime = np.asarray([row["inference_seconds_mean"] for row in planner_summary])
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    axis.scatter(runtime, joint * 100, s=70, color="#d94841")
    for x, y, label in zip(runtime, joint * 100, labels):
        axis.annotate(label, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xscale("log")
    axis.set_xlabel("Mean planning time (s, log scale)")
    axis.set_ylabel("Mean Sionna joint coverage (%)")
    axis.grid(alpha=0.25)
    fig.savefig(output / "quality_runtime.png", dpi=220)
    plt.close(fig)

    methods = list(dict.fromkeys(row["planner_id"] for row in size_summary))
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for method in methods:
        items = sorted((row for row in size_summary if row["planner_id"] == method), key=lambda row: int(row["size_m"]))
        axis.plot(
            [int(row["size_m"]) for row in items],
            [100 * row["joint_4metric_coverage_mean"] for row in items],
            marker="o", linewidth=1.5, label=items[0]["planner_label"],
        )
    axis.set_xticks([128, 256, 512])
    axis.set_xlabel("Scene size (m)")
    axis.set_ylabel("Mean Sionna joint coverage (%)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(output / "joint_coverage_by_size.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("*/*/*/seed_*/result.json")]
    if not rows:
        raise RuntimeError(f"no completed trials under {root}")
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    planner_summary = _summary_rows(rows, ("planner_id", "planner_label"))
    size_summary = _summary_rows(rows, ("planner_id", "planner_label", "size_m", "num_tx"))
    city_summary = _summary_rows(rows, ("planner_id", "planner_label", "city"))
    condition_summary = _summary_rows(
        rows,
        ("city", "scene_id", "size_m", "num_tx", "planner_id", "planner_label"),
    )
    paired = _paired_rows(rows, "diffusion_hpem", args.bootstrap_samples, args.seed)
    _write_csv(output / "planner_summary.csv", planner_summary)
    _write_csv(output / "size_summary.csv", size_summary)
    _write_csv(output / "city_summary.csv", city_summary)
    _write_csv(output / "condition_summary.csv", condition_summary)
    _write_csv(output / "paper_joint_coverage.csv", _paper_joint_rows(condition_summary))
    _write_csv(output / "paper_all_metrics.csv", _paper_metric_rows(condition_summary))
    _write_csv(output / "paired_vs_diffusion_hpem.csv", paired)
    _figures(output, planner_summary, size_summary)
    expected = None
    summary_path = root / "summary.json"
    if summary_path.exists():
        expected = json.loads(summary_path.read_text(encoding="utf-8")).get("expected_trials")
    report = {
        "completed_trials": len(rows), "expected_trials": expected,
        "complete": expected is not None and len(rows) == expected,
        "warning": None if expected is not None and len(rows) == expected else "Incomplete benchmark: statistics are provisional.",
        "best_observed_mean": max(planner_summary, key=lambda item: item["joint_4metric_coverage_mean"])["planner_id"],
        "outputs": [
            "planner_summary.csv", "size_summary.csv", "city_summary.csv",
            "condition_summary.csv", "paper_joint_coverage.csv", "paper_all_metrics.csv",
            "paired_vs_diffusion_hpem.csv", "quality_runtime.png", "joint_coverage_by_size.png",
        ],
    }
    (output / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
