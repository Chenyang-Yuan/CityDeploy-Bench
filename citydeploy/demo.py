"""Run the four training entry points and demonstrate learned-utility search on CPU."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import pyarrow
import numpy as np
import torch
from citydeploy.demo_data import create_dataset, write_json
from energy_model.guidance import load_guidance_checkpoint, predict_reward
from energy_model.dataset import load_scene_feature_tensor, set_seed
from samplers.runner import run_sampler, sampler_config_from_dict
from examples.smoke_test import SMALL_BUDGETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-samplers", action="store_true", help="Exercise all 36 model/search combinations")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    output = args.output.resolve()
    data, scenes = create_dataset(output, seed=args.seed)
    env = dict(os.environ, OMP_NUM_THREADS="2", MPLBACKEND="Agg")
    torch.set_num_threads(2)
    for model in ("hpem", "regressor", "in", "nri"):
        subprocess.run([sys.executable, "-m", "citydeploy", "train", "--model", model,
            "--dataset-root", str(data), "--scene-root", str(scenes), "--epochs", str(args.epochs),
            "--batch-size", "8", "--device", "cpu", "--num-workers", "0", "--seed", str(args.seed),
            "--output-root", str(output / "training"), "--model-root", str(output / "models")],
            check=True, env=env)
    feature, meta = load_scene_feature_tensor(scenes / "synthetic_2")
    scene_tensor = torch.from_numpy(feature)[None]
    size = torch.tensor([[meta["map_x_m"], meta["map_y_m"]]])
    records, selected = [], []
    for model in ("hpem", "regressor", "in", "nri"):
        kind, network, _ = load_guidance_checkpoint(output / "models" / model / "checkpoint_best.pt", torch.device("cpu"))
        def reward(points):
            return predict_reward(kind, network, points, torch.ones(points.shape[:2], dtype=torch.bool),
                                  scene_tensor, size.expand(len(points), -1))
        budgets = SMALL_BUDGETS if args.all_samplers else {"diffusion": SMALL_BUDGETS["diffusion"]}
        for method, budget in budgets.items():
            set_seed(args.seed)
            result = run_sampler(reward, sampler=method, num_particles=4, num_tx=2,
                                 device=torch.device("cpu"), config=sampler_config_from_dict(method, budget))
            values = result["best_reward"].detach().cpu().numpy().reshape(-1)
            points = result["best_tx_xy_norm01"][int(np.argmax(values))].detach().cpu().numpy()
            record = {"model": model, "sampler": method, "seed": args.seed, "num_tx": 2,
                      "predicted_analytic_utility": float(values.max()), "tx_xy_norm01": points.tolist(),
                      "budget": budget, "physical_verification": False}
            records.append(record)
            if method == "diffusion":
                selected.append((model, points))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), layout="constrained")
    for ax, (model, points) in zip(axes, selected):
        ax.imshow(feature[0], cmap="Greys", vmin=0, vmax=2, extent=(0, 1, 0, 1), origin="lower")
        ax.scatter(*points.T, c="#df6c32", s=65, marker="*", edgecolors="black", linewidths=.5)
        ax.set(title=model.upper(), xlabel="Normalized x", ylabel="Normalized y", xlim=(0, 1), ylim=(0, 1))
    fig.suptitle("Synthetic interface demonstration — not radio-verified deployments", fontsize=10)
    fig.savefig(output / "preview.png", dpi=180)
    plt.close(fig)
    write_json(output / "summary.json", {"label_source": "analytic_demo_not_radio", "epochs": args.epochs,
                "note": "One-epoch smoke training is not an accuracy or algorithm-ranking experiment.", "results": records})
    print(f"Completed {len(records)} synthetic searches. Results: {output}")


if __name__ == "__main__":
    main()
