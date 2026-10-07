"""Train a configured utility model and export its best state to a stable path."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from citydeploy.paths import config_path, resolve_path, workspace_root

MODEL_NAMES = {"hpem": "hpem", "regressor": "reward", "reward": "reward", "in": "in", "nri": "nri"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--config", type=Path, help="Training preset; architecture overrides supported by the selected trainer only")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/raytracing/urban_multicity"))
    parser.add_argument("--scene-root", type=Path, default=Path("datasets/scenes"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    for name in ("epochs", "max-samples", "batch-size", "max-num-tx", "seed", "num-workers"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/training"))
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    name = MODEL_NAMES[args.model]
    public_name = "regressor" if name == "reward" else name
    preset = resolve_path(args.config) if args.config else config_path(f"models/{name}.json")
    specification = json.loads(preset.read_text(encoding="utf-8"))
    if "architecture" in specification:
        parser.error("Use training options for supported overrides. Fixed layer defaults are documented as architecture_reference, not an executable architecture override.")
    expected = {"hpem": "energy_model.train", "reward": "energy_model.train_reward", "in": "energy_model.train_relational", "nri": "energy_model.train_relational"}[name]
    if specification["module"] != expected:
        parser.error("Training preset module disagrees with the requested model.")
    options = dict(specification["training"])
    expected_relational = {"in": "interaction_network", "nri": "nri"}.get(name)
    if expected_relational and options.get("model") != expected_relational:
        parser.error("Relational model kind in preset disagrees with the requested model.")
    options.update(dataset_root=resolve_path(args.dataset_root), scene_root=resolve_path(args.scene_root),
                   save_dir=resolve_path(args.output_root) / public_name, device=args.device)
    for key in ("epochs", "max_samples", "batch_size", "max_num_tx", "seed", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    command = [sys.executable, "-B", "-m", expected]
    for key, value in options.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if args.dry_run:
        print(json.dumps({"module": expected, "options": options}, default=str, indent=2))
        return
    target = resolve_path(args.model_root) / public_name / "checkpoint_best.pt"
    if target.exists():
        parser.error("Model destination exists. Select another --model-root; existing weights are never overwritten.")
    if not options["dataset_root"].is_dir():
        parser.error("Dataset is missing. Run demo-data for the tutorial or supply the published dataset via --dataset-root.")
    run_root = Path(options["save_dir"])
    previous = set(run_root.glob("*/checkpoint_best.pt"))
    subprocess.run(command, cwd=workspace_root(), check=True)
    produced = set(run_root.glob("*/checkpoint_best.pt")) - previous
    if len(produced) != 1:
        raise RuntimeError("Training must produce exactly one new best model state.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced.pop(), target)
    print(f"Best model: {target}")


if __name__ == "__main__":
    main()
