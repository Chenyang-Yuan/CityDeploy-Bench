#!/usr/bin/env python3
"""Pack V4 rich radio-map NPZ files into streamable WebDataset TAR shards."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--files-per-shard", type=int, default=256)
    args = parser.parse_args()
    root = args.dataset.resolve()
    shard_root = root / "radio_maps" / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for split in ("train", "validation", "test"):
        files = sorted((root / "radio_maps" / split).glob("*.npz"))
        for start in range(0, len(files), args.files_per_shard):
            selected = files[start : start + args.files_per_shard]
            shard_index = start // args.files_per_shard
            shard_path = shard_root / f"{split}-{shard_index:05d}.tar"
            temporary = shard_path.with_suffix(".tar.tmp")
            with tarfile.open(temporary, "w") as archive:
                for path in selected:
                    data = path.read_bytes()
                    info = tarfile.TarInfo(name=f"{path.stem}.npz")
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
                    metadata = json.dumps({"sample_id": path.stem, "split": split}).encode("utf-8")
                    meta_info = tarfile.TarInfo(name=f"{path.stem}.json")
                    meta_info.size = len(metadata)
                    archive.addfile(meta_info, io.BytesIO(metadata))
                    index_rows.append(
                        {
                            "sample_id": path.stem,
                            "split": split,
                            "source_path": path.relative_to(root).as_posix(),
                            "webdataset_shard": shard_path.relative_to(root).as_posix(),
                            "member": f"{path.stem}.npz",
                        }
                    )
            temporary.replace(shard_path)
    index_path = root / "radio_maps" / "index.parquet"
    pq.write_table(pa.Table.from_pylist(index_rows), index_path, compression="zstd")
    print(json.dumps({"radio_maps": len(index_rows), "index": str(index_path)}, indent=2))


if __name__ == "__main__":
    main()
