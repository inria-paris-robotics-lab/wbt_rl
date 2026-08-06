"""gen_mesh_collision.py — pack a robot mesh with INSCRIBED collision spheres.

Same idea HoloNew already uses to pull real geometry out of a compiled MJCF
(`HoloNew/HoloNew/src/holosoma/mujoco_utils.py::_mesh_local_vf`) applied to a new
job: instead of hand-placed capsules ("boudins") approximating a body part, derive
the collision set from the part's own geometry.

The subtlety that makes or breaks this: a sphere CENTRED on the surface sticks out
by its full radius, inflating the whole part. On the G1's paddle hand -- 14mm
thick -- 16mm-radius surface-centred spheres made the collision hand ~46mm thick,
so a grasped object would visibly float off the mesh. Each sphere here is instead
pushed INWARD along the surface normal by exactly its own radius, so it is
internally tangent: its outermost point lands on the true surface and the
collision hull never exceeds the visual one.

Radius is tied to sample spacing rather than to any inside/outside query. That is
deliberate: the G1 STLs are non-watertight CAD shells, on which both
`signed_distance` (measures the few-mm shell WALL) and `voxelized().fill()` (a
silent no-op -- it cannot find an interior to fill) give wrong answers. Surface
normals stay reliable on a shell, so the method only relies on those.

Denser sampling therefore means smaller spheres and a tighter fit, which is the
knob to turn: the generation is offline and pays no runtime cost, though the geom
count it produces does show up in every physics step.

Needs numpy + trimesh, which live in SPIDER's venv:

    modules/third_party/spider/.venv/bin/python scripts/gen_mesh_collision.py \\
        --mesh modules/third_party/holosoma_custom/.../meshes/left_rubber_hand.STL \\
        --name left_hand_collision --n-points 120

Prints ready-to-paste <geom> lines plus a measured worst-case overflow, so a bad
normal cannot pass unnoticed. It does not edit any XML itself -- the calling site
(which body it belongs in, whether it replaces or supplements an existing
primitive) is a per-file judgment call.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import trimesh

# Surface samples drawn before farthest-point-sampling picks the final subset.
# FPS needs a dense, evenly-distributed pool: CAD STL vertices bunch up around
# fillets and edges, so sampling the surface beats reusing vertices.
_POOL = 40000


@dataclass
class Config:
    mesh: str
    name: str
    n_points: int = 120
    radius_max: float = 0.02
    radius_min: float = 0.003
    # Sphere radius as a fraction of the local sample spacing. At 0.5 neighbouring
    # spheres just touch; above that they overlap and close the gaps between them
    # at the cost of a coarser fit in thin regions.
    fill: float = 0.6
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    rgba: str | None = None
    seed: int = 0


def _quat_to_R(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )


def farthest_point_sample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Greedy FPS -- even surface coverage, free of the vertex-density bias."""
    rng = np.random.default_rng(seed)
    n = min(n, len(points))
    idx = [int(rng.integers(len(points)))]
    d = np.linalg.norm(points - points[idx[0]], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(d))
        idx.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return np.asarray(idx, dtype=int)


def pack_spheres(cfg: Config) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (centres_local, radii, stats): spheres tangent to the mesh from inside."""
    mesh = trimesh.load(cfg.mesh, process=False)

    pool, face_ids = trimesh.sample.sample_surface(mesh, _POOL, seed=cfg.seed)
    pool = np.asarray(pool, dtype=np.float64)
    keep = farthest_point_sample(pool, cfg.n_points, seed=cfg.seed)
    pts = pool[keep]
    normals = np.asarray(mesh.face_normals[face_ids[keep]], dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # Orient each normal outward INDIVIDUALLY. `fix_normals()` cannot be trusted
    # here: it needs a consistently orientable surface, and on these open CAD
    # shells it leaves whole patches inverted -- which pushed those spheres
    # outward instead of inward and inflated the part by two full radii.
    # Counting ray hits along -n settles it per sample without assuming closure:
    # more material behind the point than in front means -n really is inward.
    eps = 1e-5
    back = mesh.ray.intersects_first(pts - normals * eps, -normals)
    front = mesh.ray.intersects_first(pts + normals * eps, normals)
    flip = (back < 0) & (front >= 0)
    normals[flip] *= -1.0

    # Local spacing: distance to the nearest other sample. FPS makes this fairly
    # uniform, but thin features still end up denser, and they get the smaller
    # spheres they need as a result.
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    spacing = d.min(axis=1)

    # Cap by half the local thickness, or a sphere sunk into a plate thinner than
    # its own diameter comes straight out the far side -- the residual inflation
    # left over once the centring error is gone. The cap needs the LAST hit along
    # the inward ray, not the first: on a hollow shell the first hit is the inner
    # wall a couple of mm away, which would shrink every sphere to the floor.
    half_thickness = np.full(len(pts), np.inf)
    hits, ray_ids, _ = mesh.ray.intersects_location(
        pts - normals * 1e-5, -normals, multiple_hits=True
    )
    if len(hits):
        depth = np.linalg.norm(hits - pts[ray_ids], axis=1)
        np.maximum.at(half_thickness, ray_ids, -np.inf)  # seed touched rays
        for rid, dep in zip(ray_ids, depth, strict=True):
            half_thickness[rid] = max(half_thickness[rid], dep / 2.0)

    radii = np.clip(
        np.minimum(spacing * cfg.fill, half_thickness), cfg.radius_min, cfg.radius_max
    )

    centres_local = pts - normals * radii[:, None]

    # Final retraction pass. A handful of samples land on razor edges where the
    # inward ray misses the part altogether, so no thickness cap applies and the
    # sphere ends up straddling the surface. Shrinking those to their centre's
    # actual clearance is safe everywhere: in a hollow region it merely yields a
    # thinner shrink-wrap of the outer wall, which is all the collision needs.
    # Shrinking moves the centre, which can expose a new violation, so repeat
    # until it settles -- it converges in two or three passes.
    for _ in range(4):
        _, clearance, _ = trimesh.proximity.closest_point(mesh, centres_local)
        tight = clearance < radii - 1e-6
        if not tight.any():
            break
        radii[tight] = np.maximum(clearance[tight], 1e-4)
        centres_local = pts - normals * radii[:, None]

    # Self-check by support function, which is what "does it stick out" actually
    # means and, unlike a nearest-surface query, is not fooled by the hollow
    # interior of a shell: along every probe direction, the spheres must not reach
    # further than the mesh itself.
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(256, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mesh_support = np.asarray(mesh.vertices) @ dirs.T          # (V, D)
    sph_support = centres_local @ dirs.T + radii[:, None]      # (N, D)
    excess = sph_support.max(axis=0) - mesh_support.max(axis=0)

    stats = {
        "n": len(radii),
        "r_min": float(radii.min()),
        "r_max": float(radii.max()),
        "r_med": float(np.median(radii)),
        "excess_max_mm": float(excess.max() * 1000.0),
        "excess_med_mm": float(np.median(excess) * 1000.0),
    }
    return centres_local, radii, stats


def generate(cfg: Config) -> tuple[list[str], dict]:
    centres_local, radii, stats = pack_spheres(cfg)

    R = _quat_to_R(cfg.quat_wxyz)
    centres = centres_local @ R.T + np.asarray(cfg.pos, dtype=np.float64)

    rgba_attr = f' rgba="{cfg.rgba}"' if cfg.rgba else ""
    lines = [
        f'<geom name="{cfg.name}_{i}" class="collision" type="sphere" '
        f'size="{r:.4f}" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}"{rgba_attr}/>'
        for i, (c, r) in enumerate(zip(centres, radii, strict=True))
    ]
    return lines, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mesh", required=True, help="STL/OBJ file, in the body's own local frame")
    ap.add_argument("--name", required=True, help="geom name prefix, e.g. left_hand_collision")
    ap.add_argument("--n-points", type=int, default=120, help="maximum number of spheres")
    ap.add_argument("--radius-max", type=float, default=0.02)
    ap.add_argument("--radius-min", type=float, default=0.003)
    ap.add_argument("--fill", type=float, default=0.6, help="radius as a fraction of sample spacing")
    ap.add_argument(
        "--pos", type=float, nargs=3, default=(0.0, 0.0, 0.0),
        help='offset of the source visual geom relative to the body (its own pos="...")',
    )
    ap.add_argument(
        "--quat", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0), metavar=("W", "X", "Y", "Z"),
        help='orientation of the source visual geom relative to the body (its own quat="...")',
    )
    ap.add_argument("--rgba", default=None)
    args = ap.parse_args()

    cfg = Config(
        mesh=args.mesh,
        name=args.name,
        n_points=args.n_points,
        radius_max=args.radius_max,
        radius_min=args.radius_min,
        fill=args.fill,
        pos=tuple(args.pos),
        quat_wxyz=tuple(args.quat),
        rgba=args.rgba,
    )
    lines, st = generate(cfg)
    print(
        f"<!-- {st['n']} inscribed spheres, r={st['r_min']:.4f}..{st['r_max']:.4f}m "
        f"(median {st['r_med']:.4f}), outer excess max {st['excess_max_mm']:.2f}mm -->"
    )
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
