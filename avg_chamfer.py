from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the average Chamfer from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the Chamfer CSV file")
    args = parser.parse_args()

    p = Path(args.csv_file)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    with p.open(newline="") as f:
        rows = list(csv.DictReader(f))

    vals = []
    for r in rows:
        v = r.get("chamfer_squared")
        if v not in ("", None):
            vals.append(float(v))

    if not vals:
        raise RuntimeError("No chamfer values found in the CSV.")

    mean = sum(vals) / len(vals)
    print(f"count: {len(vals)}")
    print(f"mean chamfer_squared: {mean:.12f}")


if __name__ == "__main__":
    main()