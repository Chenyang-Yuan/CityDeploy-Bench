"""Read dataset inventory without modifying labels, manifests or partitions."""
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
from citydeploy.schema import canonical_schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    counts = {}
    for file in sorted((args.dataset / "data").glob("*/*.parquet")):
        for row in pq.read_table(file, columns=["scene_id", "num_tx"]).to_pylist():
            key = (file.parent.name, row["scene_id"], row["num_tx"])
            counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"schema": manifest.get("schema_version"),
        "reader_schema": canonical_schema(manifest.get("schema_version")),
        "declared_rows": manifest.get("num_samples_total"), "observed_rows": sum(counts.values()),
        "conditions": [dict(split=k[0], scene=k[1], num_tx=k[2], rows=v) for k, v in sorted(counts.items())]}, indent=2))


if __name__ == "__main__":
    main()
