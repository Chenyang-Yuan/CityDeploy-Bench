"""Aggregate fixed-cardinality physical evaluations without pooling different protocols."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

FIELDS = ("joint_4metric_coverage", "pathloss_coverage", "ss_rsrp_coverage", "sinr_coverage",
          "effective_throughput_coverage", "inference_seconds", "raytracing_seconds", "num_reward_evaluations")


def aggregate(root: Path):
    groups = {}
    seen = set()
    for path in sorted(Path(root).rglob("result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        required = ("scene_id", "num_tx", "sampler", "model_kind", "seed", "tx_positions_m", *FIELDS)
        if not all(k in row for k in required):
            raise ValueError(f"Not a complete fixed-TX evaluation record: {path}")
        if any(not math.isfinite(float(row[k])) for k in FIELDS):
            raise ValueError(f"Non-finite metric in {path}")
        protocol = {k: row.get(k) for k in ("checkpoint", "rf_profile_snapshot", "sampler_config", "num_particles")}
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
        key = (row["scene_id"], row["num_tx"], row["sampler"], row["model_kind"], fingerprint)
        identity = (*key, row["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate condition and seed; select one experiment root: {path}")
        seen.add(identity)
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ValueError("No complete result.json records found.")
    result = []
    for key, rows in sorted(groups.items()):
        record = dict(zip(("scene", "num_tx", "sampler", "model", "protocol_id"), key))
        record.update(n=len(rows), seeds=";".join(str(r["seed"]) for r in sorted(rows, key=lambda r: r["seed"])))
        for field in FIELDS:
            values = [float(row[field]) for row in rows]
            record[f"{field}_mean"] = mean(values)
            record[f"{field}_std"] = stdev(values) if len(values) > 1 else None
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/summary.csv"))
    args = parser.parse_args()
    rows = aggregate(args.root)
    if args.output.exists():
        parser.error("Output exists; choose a new path.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} separate conditions. Coverage is a fraction; std is sample std (ddof=1).")


if __name__ == "__main__":
    main()
