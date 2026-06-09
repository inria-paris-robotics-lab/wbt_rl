from __future__ import annotations

from pathlib import Path

_URDF_TEMPLATE = """\
<?xml version="1.0" ?>
<robot name="{name}">
  <link name="baseLink">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_filename}" scale="{scale}"/>
      </geometry>
      <material name="mat">
        <color rgba="0.7 0.8 0.9 1"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_filename}" scale="{scale}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def obj_to_urdf(obj_path: str | Path, scale: float = 1.0) -> Path:
    """Generate a URDF alongside an .obj mesh file, reusing it if already up-to-date.

    The URDF is written next to the .obj so IsaacSim's URDF converter can resolve
    the mesh filename as a relative path.

    Args:
        obj_path: Path to the .obj mesh file.
        scale: Uniform scale applied to the mesh (e.g. obj_scale from OMOMO data).
    """
    obj_path = Path(obj_path).resolve()
    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    scale_str = f"{scale} {scale} {scale}"
    urdf_path = obj_path.with_suffix(".urdf")

    # Regenerate if missing, stale, or scale changed
    if urdf_path.exists() and urdf_path.stat().st_mtime >= obj_path.stat().st_mtime:
        if scale_str in urdf_path.read_text():
            return urdf_path

    urdf_path.write_text(
        _URDF_TEMPLATE.format(name=obj_path.stem, mesh_filename=obj_path.name, scale=scale_str)
    )
    return urdf_path
