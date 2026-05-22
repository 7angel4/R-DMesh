from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh

MESH_EXTS = {".obj", ".ply", ".stl", ".off", ".dae", ".glb", ".gltf", ".fbx"}
CONTAINER_EXTS = {".fbx", ".glb", ".gltf"}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def is_mesh_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MESH_EXTS


def is_container_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in CONTAINER_EXTS


def canonical_prefix(path: Path) -> str:
    """
    Match common frame-suffix patterns:
      eat_0001.obj   -> eat
      eat-0001.obj   -> eat
      eat.0001.obj   -> eat
    Falls back to the full stem if no trailing frame index is present.
    """
    stem = path.stem
    return re.sub(r"([._-])\d+$", "", stem)


def list_meshes(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if is_mesh_file(p)]
    return sorted(files, key=lambda p: natural_key(p.name))


def load_mesh_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load as mesh: {path}")

    if mesh.vertices is None or mesh.faces is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no vertices/faces: {path}")

    return mesh


def sample_surface_points(mesh: trimesh.Trimesh, n_points: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(pts, dtype=np.float64)


def chamfer_squared(a_pts: np.ndarray, b_pts: np.ndarray, chunk: int = 2048) -> float:
    """
    Symmetric squared Chamfer:
      mean_{a in A} min_{b in B} ||a-b||^2 + mean_{b in B} min_{a in A} ||b-a||^2
    """

    def min_sq_dists(x: np.ndarray, y: np.ndarray):
        out = []
        for i in range(0, len(x), chunk):
            xi = x[i : i + chunk]
            d2 = ((xi[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
            out.append(d2.min(axis=1))
        return np.concatenate(out, axis=0)

    d_ab = min_sq_dists(a_pts, b_pts)
    d_ba = min_sq_dists(b_pts, a_pts)
    return float(d_ab.mean() + d_ba.mean())


def summarize_rows(rows: list[dict], score_key: str = "chamfer_squared") -> dict:
    scores = np.array([r[score_key] for r in rows], dtype=np.float64)
    return {
        "num_pairs": int(len(rows)),
        "mean_chamfer_squared": float(scores.mean()) if len(scores) else None,
        "std_chamfer_squared": float(scores.std()) if len(scores) else None,
        "min_chamfer_squared": float(scores.min()) if len(scores) else None,
        "max_chamfer_squared": float(scores.max()) if len(scores) else None,
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = ["pair_name", "frame", "path_a", "path_b", "num_points", "chamfer_squared"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict], output_path: Path, summary: dict) -> None:
    payload = {"summary": summary, "results": rows}
    output_path.write_text(json.dumps(payload, indent=2))


def pair_directories(dir_a: Path, dir_b: Path):
    """
    Match files by canonical prefix.
    Returns:
      pairs: list[(Path, Path)]
      only_a: list[str]
      only_b: list[str]
    """
    meshes_a = list_meshes(dir_a)
    meshes_b = list_meshes(dir_b)

    map_a = {canonical_prefix(p): p for p in meshes_a}
    map_b = {canonical_prefix(p): p for p in meshes_b}

    common = sorted(set(map_a) & set(map_b), key=natural_key)
    only_a = sorted(set(map_a) - set(map_b), key=natural_key)
    only_b = sorted(set(map_b) - set(map_a), key=natural_key)

    pairs = [(map_a[k], map_b[k]) for k in common]
    return pairs, only_a, only_b