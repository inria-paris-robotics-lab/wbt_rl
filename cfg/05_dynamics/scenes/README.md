# Scene manifests (stage 05_dynamics)

One yaml per clip, describing **what else is in the world** besides the robot.

The retargeted motion npz carries object *trajectories* but nothing about their
physics — no mass, no friction, no collision geometry. A manifest supplies that
missing half, by pointing at the URDFs the rest of the pipeline already uses.

```bash
python scripts/enrich_dynamics.py --scene cfg/05_dynamics/scenes/femto14_box36.yaml ...
```

---

## Fields

```yaml
objects:
  - name: box36            # becomes the MuJoCo body/geom name
    kind: dynamic          # dynamic | static
    urdf: <path>           # mass + collision mesh + friction are read from it
    pos_key: object_pos_w  # (T,3) key in the motion npz
    quat_key: object_quat_w  # (T,4) wxyz

  - name: support
    kind: static
    mesh: <path>           # stands in for a urdf when none exists
    pos_key: support_pos_w   # (3,) — a static object's pose is read once
    quat_key: support_quat_w # (4,) wxyz
    rgba: "0.6 0.4 0.2 1"

contact:
  robot_geoms: [lh, rh]  # robot geoms allowed to touch the objects
  object_object: true    # pair every object with every other
  object_floor: true     # pair dynamic objects with the ground
  condim: 4
```

| Field | Meaning |
|-------|---------|
| `kind: dynamic` | Free to move. Adds **6 DOF** (3 slide + 3 hinge) to the qpos tail and 6 guidance actuators. |
| `kind: static` | Welded in place. Costs no DOF — use it for tables, walls, platforms. |
| `urdf` | Preferred source of geometry. Reads `<mass>`, the collision `<mesh>`/`<box>`, and `<contact><lateral_friction>`. |
| `mesh` / `half_extents` | Direct geometry, for objects with no URDF. Overrides the URDF's. |
| `mass` | Overrides the URDF's mass. |
| `pos_key` / `quat_key` | Where in the motion npz this object's pose lives. Quaternions are **wxyz**. |

Paths are repo-root-relative unless absolute.

## Adding objects

Append entries to `objects`. Each **dynamic** one grows `qpos` by 6 and `ctrl`
by 6; static ones are free. Nothing else needs changing — the embodiment reads
`num_objects` and slices accordingly.

## The contact block matters more than it looks

The G1 MJCF **disables collision globally** (`contype=0` on the robot's default
class) and re-enables it through an explicit `<pair>` list. Any object not
listed in `contact.robot_geoms` pairing is therefore *invisible* to the robot —
its hands pass straight through, silently, with no error. If a grasp mysteriously
fails to hold, check this block first.

`robot_geoms` defaults to the two hand geoms (`lh`, `rh`). Add foot geoms
(`lf0`…`rf3`) for clips where the robot stands or climbs on an object.
