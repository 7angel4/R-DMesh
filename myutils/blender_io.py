from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np

from myutils.chamfer import (
    canonical_prefix,
    is_container_file,
    is_mesh_file,
    load_mesh_trimesh,
    sample_surface_points,
)


def resolve_blender_bin(blender_bin: str) -> Path:
    """
    Accepts:
      - an absolute path to the Blender executable
      - a directory containing 'blender'
      - a bare command on PATH
    """
    p = Path(blender_bin).expanduser()

    if p.name == "blender" and p.exists():
        return p

    if p.exists() and p.is_dir():
        candidate = p / "blender"
        if candidate.exists():
            return candidate

    which = shutil.which(str(p))
    if which:
        return Path(which)

    raise FileNotFoundError(
        f"Blender executable not found: {blender_bin}\n"
        "Pass either the full executable path, a directory containing 'blender', or a PATH-resolvable command."
    )


def _run_blender_container_sampler(container_path: Path, n_points: int, seed: int, blender_bin: str) -> list[np.ndarray]:
    blender_exe = resolve_blender_bin(blender_bin)

    script = textwrap.dedent(
        r"""
        import bpy
        import sys
        import numpy as np

        def sample_scene_points(n_points, seed):
            rng = np.random.default_rng(seed)
            depsgraph = bpy.context.evaluated_depsgraph_get()

            all_verts = []
            all_tris = []
            vert_offset = 0

            mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
            if not mesh_objects:
                raise RuntimeError("No mesh objects found in scene.")

            for obj in mesh_objects:
                obj_eval = obj.evaluated_get(depsgraph)
                mesh = obj_eval.to_mesh()
                try:
                    mesh.calc_loop_triangles()

                    verts = np.array([(obj_eval.matrix_world @ v.co)[:] for v in mesh.vertices], dtype=np.float64)
                    tris = np.array([tri.vertices for tri in mesh.loop_triangles], dtype=np.int64)

                    if len(tris) > 0:
                        all_verts.append(verts)
                        all_tris.append(tris + vert_offset)
                        vert_offset += len(verts)
                finally:
                    obj_eval.to_mesh_clear()

            if not all_verts:
                raise RuntimeError("No triangulated mesh data found.")

            verts = np.concatenate(all_verts, axis=0)
            tris = np.concatenate(all_tris, axis=0)

            v0 = verts[tris[:, 0]]
            v1 = verts[tris[:, 1]]
            v2 = verts[tris[:, 2]]

            areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            total = areas.sum()
            if total <= 0:
                raise RuntimeError("Imported scene has zero surface area.")

            probs = areas / total
            tri_idx = rng.choice(len(tris), size=n_points, p=probs)

            a = v0[tri_idx]
            b = v1[tri_idx]
            c = v2[tri_idx]

            r1 = rng.random(n_points)
            r2 = rng.random(n_points)
            sr1 = np.sqrt(r1)

            pts = (1 - sr1)[:, None] * a + (sr1 * (1 - r2))[:, None] * b + (sr1 * r2)[:, None] * c
            return pts.astype(np.float64)

        argv = sys.argv
        argv = argv[argv.index("--") + 1:]
        input_path = argv[0]
        n_points = int(argv[1])
        seed = int(argv[2])
        out_path = argv[3]

        bpy.ops.wm.read_factory_settings(use_empty=True)

        ext = input_path.lower().split(".")[-1]
        if ext == "fbx":
            bpy.ops.import_scene.fbx(filepath=input_path)
        elif ext in {"glb", "gltf"}:
            bpy.ops.import_scene.gltf(filepath=input_path)
        else:
            raise RuntimeError(f"Unsupported animated container: {input_path}")

        scene = bpy.context.scene
        start = int(scene.frame_start)
        end = int(scene.frame_end)
        if end < start:
            end = start

        frames = []
        for f in range(start, end + 1):
            scene.frame_set(f)
            pts = sample_scene_points(n_points, seed + f)
            frames.append(pts)

        print(f"Saving sampled frames to {out_path}")
        payload = {f"f{i:04d}": frames[i] for i in range(len(frames))}
        np.savez_compressed(out_path, **payload)
        print("Saved sampled frames successfully")
        """
    )

    

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = Path(f.name)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        out_path = Path(f.name)

    try:
        cmd = [
            str(blender_exe),
            "--background",
            "--python",
            str(script_path),
            "--",
            str(container_path),
            str(n_points),
            str(seed),
            str(out_path),
        ]
        print("Running Blender sampler for:", container_path)
        subprocess.run(cmd, check=True)
        print("Blender finished, loading:", out_path)

        if not out_path.exists():
            raise FileNotFoundError(f"Blender did not write output file: {out_path}")

        data = np.load(out_path, allow_pickle=False)
        keys = sorted(data.files)
        print("Loaded frames:", len(keys))
        frames = [np.asarray(data[k], dtype=np.float64) for k in keys]
        print("Frame shapes:", [f.shape for f in frames[:3]])
        return frames
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass


def load_sequence_for_file(spec: str | Path, n_points: int, seed: int, blender_bin: str = "blender") -> list[np.ndarray]:
    """
    Returns a list of per-frame point clouds.

    - static mesh file -> one frame
    - animated container (fbx/glb/gltf) -> many frames sampled via Blender
    """
    p = Path(spec)

    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    if is_container_file(p):
        return _run_blender_container_sampler(p, n_points=n_points, seed=seed, blender_bin=blender_bin)

    if is_mesh_file(p):
        mesh = load_mesh_trimesh(p)
        pts = sample_surface_points(mesh, n_points=n_points, seed=seed)
        return [pts]

    raise ValueError(f"Unsupported file type: {p}")


def load_sequence_for_directory(directory: str | Path, n_points: int, seed: int) -> list[tuple[Path, np.ndarray]]:
    """
    For a directory of meshes, returns per-file point clouds.
    This is the directory-vs-directory mode used for static mesh comparison.
    """
    from chamfer import list_meshes

    d = Path(directory)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {d}")

    files = list_meshes(d)
    if not files:
        raise FileNotFoundError(f"No mesh files found in directory: {d}")

    out = []
    for i, fp in enumerate(files):
        mesh = load_mesh_trimesh(fp)
        pts = sample_surface_points(mesh, n_points=n_points, seed=seed + i)
        out.append((fp, pts))
    return out


def file_pair_pair_name(path_a: Path, path_b: Path) -> str:
    return f"{canonical_prefix(path_a)}__{canonical_prefix(path_b)}"
