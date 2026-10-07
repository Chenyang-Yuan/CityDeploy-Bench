"""Create reproducible aggregate statistics for the seen/unseen benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "hypergraph_potential": "Hypergraph potential",
    "reward_predictor": "Reward predictor",
}


def _scene_splits(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for group in payload["groups"].values():
        for scene_id in group["scenes"]:
            result[scene_id] = group["split"]
    return result


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def analyze(results_csv: Path, split_manifest: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(results_csv)
    frame["data_split"] = frame["scene_id"].map(_scene_splits(split_manifest)).fillna("external")
    frame.to_csv(output_dir / "results_with_split_source.csv", index=False)

    grouped = (
        frame.groupby(["model", "status", "size_m"], as_index=False)
        .agg(
            num_cases=("scene_id", "size"),
            actual_joint_mean=("joint_4metric_coverage", "mean"),
            prediction_mae=("prediction_error", lambda x: x.abs().mean()),
            prediction_bias=("prediction_error", "mean"),
        )
    )
    grouped.to_csv(output_dir / "model_status_size.csv", index=False)

    key = ["city", "size_m", "status", "scene_id", "data_split"]
    paired = frame.pivot(index=key, columns="model", values="joint_4metric_coverage").reset_index()
    paired["hpem_minus_reward"] = (
        paired["hypergraph_potential"] - paired["reward_predictor"]
    )
    paired.to_csv(output_dir / "paired_conditions.csv", index=False)
    delta = paired["hpem_minus_reward"].to_numpy()

    model_stats = {}
    for model, part in frame.groupby("model"):
        error = part["prediction_error"]
        model_stats[model] = {
            "num_cases": int(len(part)),
            "actual_joint_mean": float(part["joint_4metric_coverage"].mean()),
            "prediction_mae": float(error.abs().mean()),
            "prediction_bias": float(error.mean()),
            "pearson_prediction_vs_actual": float(
                part["predicted_reward"].corr(part["joint_4metric_coverage"], method="pearson")
            ),
            "spearman_prediction_vs_actual": float(
                part["predicted_reward"].corr(part["joint_4metric_coverage"], method="spearman")
            ),
            "large_overprediction_count": int((error > 0.20).sum()),
        }

    external = frame[frame["data_split"] == "external"]
    external_stats = (
        external.groupby("model")
        .agg(
            num_cases=("scene_id", "size"),
            actual_joint_mean=("joint_4metric_coverage", "mean"),
            prediction_mae=("prediction_error", lambda x: x.abs().mean()),
        )
        .reset_index()
    )
    report = {
        "num_results": int(len(frame)),
        "num_paired_conditions": int(len(paired)),
        "model_stats": model_stats,
        "paired_comparison": {
            "hpem_wins": int((delta > 0).sum()),
            "reward_wins": int((delta < 0).sum()),
            "ties": int((delta == 0).sum()),
            "mean_hpem_minus_reward": float(delta.mean()),
            "median_hpem_minus_reward": float(np.median(delta)),
            "bootstrap_95pct_ci_mean_difference": _bootstrap_mean_ci(delta),
        },
        "strict_external_scenes": _records(external_stats),
        "grouped_model_status_size": _records(grouped),
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    colors = {"hypergraph_potential": "#3268a8", "reward_predictor": "#e0823d"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    x = np.arange(3)
    width = 0.18
    offsets = {("seen", 0): -1.5, ("unseen", 0): 0.5}
    for model_idx, model in enumerate(MODEL_LABELS):
        for status in ("seen", "unseen"):
            values = []
            for size in (128, 256, 512):
                row = grouped[
                    (grouped["model"] == model)
                    & (grouped["status"] == status)
                    & (grouped["size_m"] == size)
                ]
                values.append(float(row["actual_joint_mean"].iloc[0]))
            slot = model_idx + (0 if status == "seen" else 2)
            pos = x + (slot - 1.5) * width
            axes[0].bar(
                pos,
                values,
                width,
                color=colors[model],
                alpha=1.0 if status == "seen" else 0.55,
                hatch="" if status == "seen" else "//",
                label=f"{MODEL_LABELS[model]} - {status}",
            )
    axes[0].set_xticks(x, ["128 m / 2 TX", "256 m / 3 TX", "512 m / 7 TX"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Ray-traced joint coverage")
    axes[0].set_title("Deployment quality by size and scene visibility")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(fontsize=8, ncols=2, loc="lower left")

    matrix = paired.pivot_table(
        index="city", columns="size_m", values="hpem_minus_reward", aggfunc="mean"
    ).reindex(columns=[128, 256, 512])
    vmax = float(np.abs(matrix.to_numpy()).max())
    image = axes[1].imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1].set_xticks(range(3), ["128 m", "256 m", "512 m"])
    axes[1].set_yticks(range(len(matrix.index)), [x.replace("_", " ").title() for x in matrix.index])
    axes[1].set_title("HPEM advantage (paired coverage difference)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            axes[1].text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes[1], label="HPEM minus reward predictor")
    fig.savefig(output_dir / "benchmark_summary.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results_csv.parent / "analysis"
    analyze(args.results_csv.resolve(), args.split_manifest.resolve(), output_dir.resolve())


if __name__ == "__main__":
    main()
