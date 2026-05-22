from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from myutils.blender_io import file_pair_pair_name, load_sequence_for_directory, load_sequence_for_file

from myutils.chamfer import (
    canonical_prefix,
    chamfer_squared,
    pair_directories,
    summarize_rows,
    write_csv,
    write_json,
)

from myutils.chamfer import load_mesh_trimesh, sample_surface_points


NUM_POINTS = 10000
BLENDER_BIN = "/users/angelyhe/apps/blender/blender-3.6.23-linux-x64/blender"


def default_export_path(input_a: Path, input_b: Path) -> Path:
    out_name = f"{input_a.stem}_vs_{input_b.stem}.csv"
    return Path.cwd() / "chamfer_results" / out_name


def evaluate_file_pair(input_a: Path, input_b: Path, n_points: int, seed: int, blender_bin: str) -> list[dict]:
    seq_a = load_sequence_for_file(input_a, n_points=n_points, seed=seed, blender_bin=blender_bin)
    seq_b = load_sequence_for_file(input_b, n_points=n_points, seed=seed + 10000, blender_bin=blender_bin)

    n = min(len(seq_a), len(seq_b))
    if n == 0:
        raise RuntimeError("No overlapping frames to compare.")

    if len(seq_a) != len(seq_b):
        print(f"Warning: sequence length mismatch ({len(seq_a)} vs {len(seq_b)}); comparing first {n} frame(s).")

    pair_name = file_pair_pair_name(input_a, input_b)
    rows = []
    for t in range(n):
        cd = chamfer_squared(seq_a[t], seq_b[t])
        rows.append(
            {
                "pair_name": pair_name,
                "frame": t,
                "path_a": str(input_a),
                "path_b": str(input_b),
                "num_points": n_points,
                "chamfer_squared": cd,
            }
        )
    return rows


def evaluate_directory_pair(input_a: Path, input_b: Path, n_points: int, seed: int, blender_bin: str):
    pairs, only_a, only_b = pair_directories(input_a, input_b)

    if not pairs:
        raise RuntimeError(
            f"No matching mesh filenames found between directories:\n"
            f"  A: {input_a}\n"
            f"  B: {input_b}"
        )

    rows = []
    print(f"Found {len(pairs)} matched mesh pair(s).")
    if only_a:
        print(f"Warning: {len(only_a)} mesh(es) only in A.")
    if only_b:
        print(f"Warning: {len(only_b)} mesh(es) only in B.")

    for i, (path_a, path_b) in enumerate(pairs):
        print(f"[{i + 1}/{len(pairs)}] {path_a.name} <-> {path_b.name}")
        rows.extend(evaluate_file_pair(path_a, path_b, n_points, seed + i, blender_bin))

    return rows, only_a, only_b


def main():
    parser = argparse.ArgumentParser(
        description="Compute symmetric squared Chamfer distance between two mesh files, or matched meshes in two directories."
    )
    parser.add_argument("input_a", type=str, help="Mesh file or directory of mesh files")
    parser.add_argument("input_b", type=str, help="Mesh file or directory of mesh files")
    parser.add_argument("--num_points", type=int, default=NUM_POINTS, help="Surface points sampled per mesh")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--export",
        nargs="?",
        const=True,
        default=False,
        help="Export results. Use '--export' for a default CSV filename, or '--export path/to/results.csv' / '.json'.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if directory inputs contain unmatched mesh names.",
    )
    parser.add_argument(
        "--blender_bin",
        type=str,
        default=BLENDER_BIN,
        help="Blender executable used for animated FBX/GLB/GLTF inputs.",
    )

    args = parser.parse_args()

    input_a = Path(args.input_a)
    input_b = Path(args.input_b)

    if not input_a.exists():
        raise FileNotFoundError(f"Input A does not exist: {input_a}")
    if not input_b.exists():
        raise FileNotFoundError(f"Input B does not exist: {input_b}")

    rows: list[dict] = []
    unmatched_a: list[str] = []
    unmatched_b: list[str] = []

    if input_a.is_file() and input_b.is_file():
        rows = evaluate_file_pair(input_a, input_b, args.num_points, args.seed, args.blender_bin)

    elif input_a.is_dir() and input_b.is_dir():
        rows, unmatched_a, unmatched_b = evaluate_directory_pair(
            input_a, input_b, args.num_points, args.seed, args.blender_bin
        )

        if args.strict and (unmatched_a or unmatched_b):
            raise RuntimeError(
                f"Unmatched files found.\nOnly in A: {unmatched_a}\nOnly in B: {unmatched_b}"
            )

    else:
        raise ValueError(
            "Inputs must be either both files or both directories.\n"
            f"Got:\n"
            f"  input_a={input_a} exists={input_a.exists()} is_file={input_a.is_file()} is_dir={input_a.is_dir()}\n"
            f"  input_b={input_b} exists={input_b.exists()} is_file={input_b.is_file()} is_dir={input_b.is_dir()}"
        )

    summary = summarize_rows(rows)

    print("\nResults")
    print("-------")
    for row in rows:
        if row["frame"] == "" or row["frame"] is None:
            print(f"{row['pair_name']}: {row['chamfer_squared']:.8f}")
        else:
            print(f"{row['pair_name']} frame {int(row['frame']):04d}: {row['chamfer_squared']:.8f}")

    print("\nSummary")
    print("-------")
    print(f"num_pairs: {summary['num_pairs']}")
    print(f"mean_chamfer_squared: {summary['mean_chamfer_squared']:.8f}")
    print(f"std_chamfer_squared: {summary['std_chamfer_squared']:.8f}")
    print(f"min_chamfer_squared: {summary['min_chamfer_squared']:.8f}")
    print(f"max_chamfer_squared: {summary['max_chamfer_squared']:.8f}")

    if unmatched_a:
        print("\nUnmatched only in A:")
        for x in unmatched_a:
            print(f"  {x}")

    if unmatched_b:
        print("\nUnmatched only in B:")
        for x in unmatched_b:
            print(f"  {x}")

    if args.export:
        if args.export is True:
            output_path = default_export_path(input_a, input_b)
        else:
            output_path = Path(args.export)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == ".json":
            write_json(rows, output_path, summary)
        else:
            write_csv(rows, output_path)

        print(f"\nExported results to: {output_path}")


if __name__ == "__main__":
    main()