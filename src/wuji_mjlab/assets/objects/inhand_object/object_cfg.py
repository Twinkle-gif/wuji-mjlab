# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Common in-hand object config helpers."""

from functools import partial
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

_ASSET_DIR = Path(__file__).resolve().parent

INHAND_OBJECT_XML: Path = _ASSET_DIR / "xmls" / "cube.xml"
assert INHAND_OBJECT_XML.exists(), f"Missing in-hand object XML: {INHAND_OBJECT_XML}"

# Baseline cube — reflects values inside cube.xml. If cube.xml changes, update here.
BASELINE_EDGE_M = 0.054
BASELINE_MASS_KG = 0.120
CUBE_DENSITY = BASELINE_MASS_KG / (BASELINE_EDGE_M**3)  # ≈ 683 kg/m^3

INHAND_OBJECT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(-0.1404, 0.0042, 0.52),
  rot=(0.3894, -0.3796, -0.5980, -0.5888),
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
)


def _get_spec(edge_m: float | None = None) -> mujoco.MjSpec:
  """Load the cube spec; optionally scale to a different edge length.

  When ``edge_m`` is None, return the spec exactly as the XML defines it
  (54 mm baseline). When set, mutate the named ``cube`` geom's box size +
  mass and the ``cube_mesh`` mesh scale so that the cube becomes a uniform
  same-density solid with the requested edge length.
  """
  spec = mujoco.MjSpec.from_file(str(INHAND_OBJECT_XML))
  if edge_m is None:
    return spec

  half = edge_m / 2.0
  mass = CUBE_DENSITY * (edge_m**3)

  cube_geom = next((g for g in spec.geoms if g.name == "cube"), None)
  if cube_geom is None:
    raise ValueError(f"{INHAND_OBJECT_XML} must contain a geom named 'cube'")
  cube_geom.size = [half, half, half]
  cube_geom.mass = mass

  cube_mesh = next((m for m in spec.meshes if m.name == "cube_mesh"), None)
  if cube_mesh is None:
    raise ValueError(f"{INHAND_OBJECT_XML} must contain a mesh named 'cube_mesh'")
  cube_mesh.scale = [half, half, half]

  return spec


def get_inhand_object_cfg(edge_m: float | None = None) -> EntityCfg:
  return EntityCfg(
    init_state=INHAND_OBJECT_INIT_STATE,
    spec_fn=partial(_get_spec, edge_m),
  )


def _get_can_spec(object_dir: Path, scale: float) -> mujoco.MjSpec:
  """Build a free-floating can (coacd mesh) spec for the AnyGrasp task.

  The mesh is the CoACD ``decomposed.obj``, re-exported to STL so MuJoCo reads
  all its ``o`` groups (MuJoCo's OBJ importer only reads the first group).
  Mirrors ``scripts/visualize_grasps.py`` but adds a massed collision geom so the
  hand can actually grasp it.
  """
  import trimesh

  mesh_path = object_dir / "coacd" / "decomposed.obj"
  if not mesh_path.exists():
    raise FileNotFoundError(f"Missing can mesh: {mesh_path}")
  mesh = trimesh.load(mesh_path, process=False)

  spec = mujoco.MjSpec()
  spec.assets["can.stl"] = mesh.export(file_type="stl")
  spec.add_mesh(name="can_mesh", file="can.stl", scale=[scale] * 3)

  body = spec.worldbody.add_body(name="can_body")
  body.add_freejoint(name="can_freejoint")
  body.add_geom(
    name="can_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="can_mesh",
    rgba=[0.80, 0.33, 0.28, 1.0],
    group=2,
    contype=0,
    conaffinity=0,
  )
  col = body.add_geom(
    name="can_collision",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="can_mesh",
    group=3,
    contype=1,
    conaffinity=2,
  )
  col.mass = 0.15
  col.friction = [0.3, 0.3, 0.3]
  col.condim = 3
  col.priority = 0
  col.solref = [0.015, 1.0]
  col.solimp = [0.85, 0.85, 0.001, 0.5, 2.0]
  return spec


def get_inhand_can_cfg(object_dir: Path, scale: float) -> EntityCfg:
  """Free-floating can entity for the AnyGrasp-to-AnyGrasp task.

  ``object_dir`` is the meshdata object directory (contains ``coacd/``) and
  ``scale`` is the GraspQP mesh scale (``data["scale"]``), mapping the normalized
  CoACD mesh onto the real object.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.55),
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
    ),
    spec_fn=partial(_get_can_spec, object_dir, scale),
  )
