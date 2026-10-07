"""Materialize editable presets without overwriting existing configurations."""
import argparse
from pathlib import Path
import shutil
from citydeploy.paths import workspace_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    source = Path(__file__).parent / "configs"
    target = workspace_root() / "configs"
    created = skipped = 0
    for path in source.rglob("*.json"):
        destination = target / path.relative_to(source)
        if destination.exists():
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        created += 1
    print(f"Configurations copied: {created}; existing files preserved: {skipped}")


if __name__ == "__main__":
    main()
