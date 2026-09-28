# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Inspect and visualize GraspQP grasp datasets (``meshdata_20``).

``meshdata_20`` is *not* produced by this repo. It is the output of the upstream
GraspQP pipeline (CoRL 2025, leggedrobotics) run on the DexGraspNet
``meshdatav3`` objects, with the Wuji Hand added as a supported hand. Each object
directory looks like this::

    <object>/
      coacd/                                    # CoACD convex decomposition
        decomposed.obj   coacd.urdf  coacd.usd  coacd_convex_piece_*.obj
        config.yaml      model.config           .asset_hash
      grasp_predictions/<hand>/<n_contacts>_contacts/graspqp/scale_<s>/<type>/
        <object>.dexgrasp.pt                    # optimized grasps (pre-grasp pose)
        <object>_step_<N>.dexgrasp.pt           # optimizer snapshots
        succ_grasps_*.pt / failed_grasps_*.pt   # sim-verified subsets
        dexgrasp_eval_*.csv                     # simulator evaluation metrics

A ``.dexgrasp.pt`` is a dict of batched tensors (one row per candidate grasp;
256 per file for the files shipped here). See ``--describe`` for the full key /
shape dump.

Coordinate conventions (verified numerically against the bundled
``fingertip_info`` ground truth, see ``_ROT_GRASPP_TO_MJCF``):

* ``parameters['root_pose']`` is ``[px, py, pz, qw, qx, qy, qz]`` -- the palm
  origin and orientation expressed in the *GraspQP object frame*, where the
  object sits at the origin with identity orientation.
* The GraspQP palm frame is ``X = thumb``, ``Y = index``, ``Z = palm normal``,
  whereas the Wuji MJCF palm link uses ``X = palm normal``, ``Y = thumb``,
  ``Z = index``. Hence the axis permutation below.
* ``parameters['root_pose'][:3]`` is the GraspQP palm origin, which is *not* the
  MJCF palm-link origin; a fixed ``OFFSET_PALM_LOCAL`` in the GraspQP palm frame
  bridges the two.
* Distances are metres. ``scale`` maps the normalized CoACD mesh onto the real
  object, so the mesh must be multiplied by ``scale``.

Usage::

    # what is in these files?
    python scripts/visualize_grasps.py --describe

    # render the 6 highest-scoring grasps of one object
    python scripts/visualize_grasps.py --object core-can-129880fda38f3f2ba1ab68e159bfb347

    # interactive viewer (needs a display; WSLg works)
    python scripts/visualize_grasps.py --object core-bottle-* --viewer

Rendering is offscreen via EGL, so it works headless. Set ``MUJOCO_GL`` to
override (e.g. ``MUJOCO_GL=glfw`` to use the WSLg/X11 window system).
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path

# Must be set before mujoco imports its GL backend. EGL gives headless rendering
# on the CUDA device; fall back to the display-backed backend otherwise.
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import trimesh
from scipy.spatial.transform import Rotation

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HAND_XML = (
  _REPO_ROOT / "src" / "wuji_mjlab" / "assets" / "robots" / "wuji_hand" / "mjcf" / "right_mjlab.xml"
)

# GraspQP palm frame (X=thumb, Y=index, Z=palm normal) -> Wuji MJCF palm-link
# frame (X=palm normal, Y=thumb, Z=index). Column permutation matrix.
_ROT_GRASPP_TO_MJCF = np.array(
  [
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
  ]
)

# Offset from the GraspQP palm origin to the Wuji MJCF palm-link origin,
# expressed in the GraspQP palm frame (thumb, index, normal).
OFFSET_PALM_LOCAL = np.array([0.0, -0.08, -0.01])

# Order of the hand links that own contact slots (documented by
# ``contact_links``: 12 slots on the palm, then 4 per fingertip).


def _quat_to_mat(qwxyz: np.ndarray) -> np.ndarray:
  """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix.

  Uses scipy (not ``mujoco.mju_quat2Mat``) so the module stays statically
  type-checkable; both agree to machine precision on the Hamilton convention.
  """
  x, y, z, w = qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]
  return Rotation.from_quat([x, y, z, w]).as_matrix()


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
  """3x3 rotation matrix -> unit quaternion (w, x, y, z)."""
  x, y, z, w = Rotation.from_matrix(np.asarray(mat, dtype=float).reshape(3, 3)).as_quat()
  return np.array([w, x, y, z])


@dataclass
class GraspBatch:
  """One ``.dexgrasp.pt`` file plus everything needed to interpret it."""

  path: Path
  data: dict
  scale: float
  n_grasps: int

  @property
  def values(self) -> np.ndarray:
    return np.asarray(self.data["values"], dtype=np.float32)

  @property
  def joint_names(self) -> list[str]:
    return [k for k in self.data["parameters"] if k != "root_pose"]

  def joints(self, idx: int) -> np.ndarray:
    """(20,) joint angles for grasp ``idx``, in the Wuji joint order."""
    return np.array([float(self.data["parameters"][n][idx]) for n in self.joint_names])

  def root_pose(self, idx: int) -> np.ndarray:
    """(7,) ``[px, py, pz, qw, qx, qy, qz]`` in the GraspQP object frame."""
    return np.asarray(self.data["parameters"]["root_pose"][idx], dtype=float)

  def mjcf_palm_pose(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``root_pose`` into the Wuji MJCF palm-link world pose.

    The object frame is treated as the world frame (the object sits at the
    origin with identity orientation upstream).
    """
    rp = self.root_pose(idx)
    rot_graspp = _quat_to_mat(rp[3:])
    pos = rp[:3] + rot_graspp @ OFFSET_PALM_LOCAL
    rot_mjcf = rot_graspp @ _ROT_GRASPP_TO_MJCF
    return pos, _mat_to_quat(rot_mjcf)

  @property
  def has_contacts(self) -> bool:
    return "fingertip_info" in self.data


def find_objects(data_path: Path) -> list[Path]:
  """Object directories (those with a ``coacd/`` subdir), sorted by name."""
  return sorted(p for p in data_path.iterdir() if p.is_dir() and (p / "coacd").is_dir())


def resolve_object(data_path: Path, name: str | None) -> Path:
  """Match an object by exact name, unique prefix, or unique glob."""
  objects = find_objects(data_path)
  if not objects:
    raise FileNotFoundError(f"No object directories found under {data_path}")
  if name is None:
    return objects[0]

  by_name = {p.name: p for p in objects}
  if name in by_name:
    return by_name[name]

  matches = [p for p in objects if p.name.startswith(name)]
  matches += [Path(m) for m in sorted(glob.glob(str(data_path / name))) if Path(m).is_dir()]
  matches = sorted(set(matches))
  if len(matches) == 1:
    return matches[0]
  if not matches:
    raise FileNotFoundError(
      f"No object matches {name!r}. Available:\n  " + "\n  ".join(by_name)
    )
  raise FileNotFoundError(
    f"{name!r} is ambiguous ({len(matches)} matches): " + ", ".join(p.name for p in matches)
  )


def find_grasp_file(object_dir: Path, source: str) -> Path:
  """Locate the requested ``.dexgrasp.pt`` for an object."""
  if source == "dexgrasp":
    hits = sorted(p for p in object_dir.rglob("*.dexgrasp.pt") if "_step_" not in p.name)
  elif source == "succ":
    hits = sorted(object_dir.rglob("succ_grasps_*.pt"))
  elif source == "failed":
    hits = sorted(object_dir.rglob("failed_grasps_*.pt"))
  else:
    raise ValueError(f"Unknown source {source!r}")
  if not hits:
    raise FileNotFoundError(f"No {source!r} grasp file under {object_dir}")
  return hits[0]


def _resolve_scale(path: Path, data: dict) -> float:
  """Recover the object scale for batches that do not record it themselves.

  ``succ_grasps_*.pt`` / ``failed_grasps_*.pt`` inherit the scale from the
  sibling ``*.dexgrasp.pt`` produced by the same run; fall back to the
  ``scale_<value>`` directory name.
  """
  if "scale" in data:
    return float(data["scale"])

  siblings = sorted(p for p in path.parent.glob("*.dexgrasp.pt") if "_step_" not in p.name)
  for sib in siblings:
    other = torch.load(sib, map_location="cpu", weights_only=False)
    if "scale" in other:
      return float(other["scale"])

  for part in path.parts[::-1]:
    if part.startswith("scale_"):
      return float(part[len("scale_") :])
  raise KeyError(f"Cannot determine object scale for {path}")


def load_batch(path: Path) -> GraspBatch:
  data = torch.load(path, map_location="cpu", weights_only=False)
  if "parameters" not in data or "root_pose" not in data["parameters"]:
    raise KeyError(f"{path} has no parameters/root_pose; not a GraspQP grasp batch")
  scale = _resolve_scale(path, data)
  n = int(data["parameters"]["root_pose"].shape[0])
  return GraspBatch(path=path, data=data, scale=scale, n_grasps=n)


def build_scene(
  object_dir: Path,
  scale: float,
  n_contact_markers: int = 0,
) -> mujoco.MjSpec:
  """Compose the Wuji Hand + scaled object into one MjSpec.

  The hand stays at the origin; the object is a free body so a grasp can be
  written into ``qpos`` at runtime. ``n_contact_markers`` sphere geoms named
  ``contact_0..N-1`` are created at the object origin for
  :func:`write_contact_markers` to move.
  """
  spec = mujoco.MjSpec.from_file(str(_HAND_XML))

  obj_mesh = object_dir / "coacd" / "decomposed.obj"
  if not obj_mesh.exists():
    raise FileNotFoundError(f"Missing mesh: {obj_mesh}")
  # decomposed.obj contains 8 `o` groups (convex_0..7); MuJoCo's OBJ importer
  # only reads the first group's faces, so re-export a single unified mesh.
  mesh = trimesh.load(obj_mesh, process=False)
  spec.assets["grasp_object.stl"] = mesh.export(file_type="stl")
  spec.add_mesh(name="grasp_object", file="grasp_object.stl", scale=[scale] * 3)

  body = spec.worldbody.add_body(name="grasp_object_body")
  body.add_freejoint(name="grasp_object_free")
  body.add_geom(
    name="grasp_object_geom",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="grasp_object",
    rgba=[0.80, 0.33, 0.28, 1.0],
    group=1,
    contype=0,
    conaffinity=0,
  )

  # Contact markers (small spheres) so the recorded contact locations are visible.
  for k in range(n_contact_markers):
    body.add_geom(
      name=f"contact_{k}",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=[0.0022],
      pos=[0.0, 0.0, 0.0],
      rgba=[0.20, 0.85, 0.25, 1.0],
      group=1,
      contype=0,
      conaffinity=0,
    )

  # The hand XML ships without lights; add a key light + fill for legible renders.
  spec.worldbody.add_light(
    name="key",
    pos=[0.35, -0.45, 0.55],
    dir=[-0.4, 0.5, -0.6],
    diffuse=[1.0, 1.0, 1.0],
    specular=[0.4, 0.4, 0.4],
    castshadow=True,
  )
  spec.worldbody.add_light(
    name="fill",
    pos=[-0.45, 0.35, 0.25],
    dir=[0.5, -0.4, -0.3],
    diffuse=[0.45, 0.45, 0.5],
  )
  return spec


def object_pose_in_hand_frame(batch: GraspBatch, idx: int) -> tuple[np.ndarray, np.ndarray]:
  """Object pose expressed in the (fixed, origin-located) Wuji palm frame.

  The hand is kept at the world origin in the compiled scene, so the grasp is
  realised by moving the object instead. ``root_pose`` maps hand-local points
  into the object frame, hence the inverse below.
  """
  pos_h, quat_h = batch.mjcf_palm_pose(idx)
  rot_h = _quat_to_mat(quat_h)
  return -(rot_h.T @ pos_h), _mat_to_quat(rot_h.T)


def write_grasp(model: mujoco.MjModel, data: mujoco.MjData, batch: GraspBatch, idx: int) -> None:
  """Write grasp ``idx`` into ``data.qpos`` (hand joints + free object pose)."""
  obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "grasp_object_free")
  if obj_jid < 0 or model.jnt_type[obj_jid] != mujoco.mjtJoint.mjJNT_FREE:
    raise RuntimeError("Scene is missing the object freejoint.")
  adr = model.jnt_qposadr[obj_jid]
  pos, quat = object_pose_in_hand_frame(batch, idx)
  data.qpos[adr : adr + 3] = pos
  data.qpos[adr + 3 : adr + 7] = quat

  for name, value in zip(batch.joint_names, batch.joints(idx), strict=True):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      raise KeyError(f"Joint {name!r} not present in {_HAND_XML.name}")
    data.qpos[model.jnt_qposadr[jid]] = float(value)
  mujoco.mj_forward(model, data)


def _make_hand_transparent(model: mujoco.MjModel, alpha: float = 0.4) -> None:
  """Make every hand geom translucent so the grasped object shows through.

  The object geom (``grasp_object_geom``) and the contact markers stay opaque.
  """
  for gid in range(model.ngeom):
    gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    if gname == "grasp_object_geom" or gname.startswith("contact_"):
      continue
    model.geom_rgba[gid, 3] = alpha


def compile_grasp_scene(
  object_dir: Path,
  batch: GraspBatch,
  transparent_hand: bool = True,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
  """Compile the hand+object scene and return it with an ``MjData`` buffer."""
  n_markers = (
    int(np.asarray(batch.data["fingertip_info"]["nearest_pts"]).shape[1])
    if batch.has_contacts
    else 0
  )
  spec = build_scene(object_dir, batch.scale, n_markers)
  model = spec.compile()
  if transparent_hand:
    _make_hand_transparent(model)
  return model, mujoco.MjData(model)


def write_contact_markers(
  model: mujoco.MjModel,
  batch: GraspBatch,
  idx: int,
  radius: float = 0.0022,
) -> None:
  """Move the 12 contact-marker spheres onto grasp ``idx`` (object-local coords)."""
  if not batch.has_contacts:
    return
  pts = np.asarray(batch.data["fingertip_info"]["nearest_pts"][idx])
  for k in range(pts.shape[0]):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"contact_{k}")
    if gid < 0:
      continue
    model.geom_pos[gid] = pts[k]
    model.geom_size[gid][0] = radius


def apply_grasp(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  batch: GraspBatch,
  idx: int,
  contacts: bool = True,
) -> None:
  """Move the contact markers, then write the grasp and forward kinematics."""
  if contacts:
    write_contact_markers(model, batch, idx)
  write_grasp(model, data, batch, idx)


def describe(batch: GraspBatch, object_dir: Path) -> None:
  """Print the structure of the grasp file so the format is self-explanatory."""
  bar = "=" * 78
  print(bar)
  print(f"OBJECT   {object_dir.name}")
  print(f"GRASPS   {batch.path}")
  print(bar)
  print(f"grasp records : {batch.n_grasps}")
  print(f"scale         : {batch.scale:.6g}  (multiply the CoACD mesh by this -> metres)")

  mesh = trimesh.load(object_dir / "coacd" / "decomposed.obj", process=False)
  ext = (mesh.bounds[1] - mesh.bounds[0]) * batch.scale
  print(f"object extents: {np.round(ext, 4)} m  (real size after scaling)")

  print("\n-- top-level keys --")
  for k, v in batch.data.items():
    if torch.is_tensor(v):
      print(f"  {k:26s} tensor {tuple(v.shape)} {v.dtype}")
    elif isinstance(v, dict):
      print(f"  {k:26s} dict   ({len(v)} entries)")
    else:
      print(f"  {k:26s} {type(v).__name__:6s} {v!r}"[:110])

  print(f"\n-- parameters (batch of {batch.n_grasps}) --")
  for name in batch.joint_names:
    v = np.asarray(batch.data["parameters"][name], dtype=float)
    print(f"  {name:26s} {tuple(v.shape)}  range [{v.min():+.3f}, {v.max():+.3f}] rad")
  rp = np.asarray(batch.data["parameters"]["root_pose"], dtype=float)
  print(f"  {'root_pose':26s} {tuple(rp.shape)}  = [px py pz qw qx qy qz] (GraspQP object frame)")
  print(f"  {'  translation extent':26s} {np.round(rp[:, :3].max(0) - rp[:, :3].min(0), 4)} m")
  print(f"  {'  |quaternion| mean':26s} {np.linalg.norm(rp[:, 3:], axis=1).mean():.6f}")

  for key in ("grasp_velocities", "grasp_velocities_full", "grasp_velocities_off", "grasp_velocities_rl"):
    if key in batch.data:
      d = batch.data[key]
      shp = tuple(next(iter(d.values())).shape)
      print(f"\n-- {key} -- {len(d)} joints x {shp} (closing-velocity profiles)")

  if "contact_links" in batch.data:
    print("\n-- contact_links (contact slots per hand link) --")
    for name, info in batch.data["contact_links"].items():
      print(f"  {name:26s} n_points={info['n_points']}")

  if batch.has_contacts:
    fi = batch.data["fingertip_info"]
    print("\n-- fingertip_info (ground truth used to validate conventions) --")
    for k, v in fi.items():
      print(f"  {k:26s} tensor {tuple(v.shape)} {v.dtype}")
    d = np.asarray(fi["distances"], dtype=float)
    print("  distances are signed: <0 separated outside, ~0 in contact")
    print(f"    range [{d.min() * 1000:+.1f}, {d.max() * 1000:+.1f}] mm")
    print(f"    fraction |d|<1mm (true contacts): {np.mean(np.abs(d) < 1e-3) * 100:.1f}%")

  v = batch.values
  print("\n-- values (quality score, higher = better) --")
  print(f"  min {v.min():.3f}  median {np.median(v):.3f}  max {v.max():.3f}")
  print(f"  best 5 indices: {np.argsort(v)[::-1][:5].tolist()}")


def render_grid(
  object_dir: Path,
  batch: GraspBatch,
  idxs: list[int],
  out_path: Path,
  width: int = 480,
  height: int = 480,
  azimuths: tuple[float, ...] = (135.0, 45.0),
  elevation: float = -20.0,
  contacts: bool = True,
) -> None:
  """Render one row per grasp (one column per azimuth) into a PNG."""
  model, data = compile_grasp_scene(object_dir, batch)
  rows, ncols = len(idxs), len(azimuths)
  fig = plt.figure(figsize=(3.6 * ncols, 3.6 * rows))
  for r, idx in enumerate(idxs):
    apply_grasp(model, data, batch, idx, contacts=contacts)
    for c, azim in enumerate(azimuths):
      img = render_grasp(model, data, width, height, azim=azim, elev=elevation)
      ax = fig.add_subplot(rows, ncols, r * ncols + c + 1)
      ax.imshow(img)
      ax.set_xticks([])
      ax.set_yticks([])
      if c == 0:
        ax.set_ylabel(f"#{idx}  value {batch.values[idx]:.2f}", fontsize=9)
      if r == 0:
        ax.set_title(f"azimuth {azim:g} deg", fontsize=9)
  plt.tight_layout()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(out_path, dpi=110)
  plt.close(fig)
  print(f"[INFO] wrote {out_path}  ({rows} grasps x {ncols} views)")


def run_viewer(
  object_dir: Path,
  batch: GraspBatch,
  idx: int,
  contacts: bool = True,
) -> None:
  """Open MuJoCo's interactive viewer; ``[``/``]`` cycle grasps, ``b`` best-first."""
  import time

  import mujoco.viewer

  model, data = compile_grasp_scene(object_dir, batch)
  best_first = np.argsort(batch.values)[::-1]
  state = {"idx": idx, "b": 0}

  def refresh() -> None:
    apply_grasp(model, data, batch, state["idx"], contacts=contacts)
    value = batch.values[state["idx"]]
    print(f"  grasp #{state['idx']:4d}  value={value:8.3f}")

  def key_callback(keycode: int) -> None:
    if keycode in (ord("]"), 265):
      state["idx"] = (state["idx"] + 1) % batch.n_grasps
      refresh()
    elif keycode in (ord("["), 264):
      state["idx"] = (state["idx"] - 1) % batch.n_grasps
      refresh()
    elif keycode == ord("b"):
      state["idx"] = int(best_first[min(state["b"], len(best_first) - 1)])
      state["b"] += 1
      refresh()

  refresh()
  print("[INFO] '[' / ']' cycle grasps, 'b' walks best-first, close the window to quit.")
  with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
      viewer.sync()
      time.sleep(0.02)


def _scene_bounds(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float]:
  """World-space centre and radius covering every mesh geom (hand + object)."""
  lo = np.full(3, np.inf)
  hi = np.full(3, -np.inf)
  for gid in range(model.ngeom):
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    mid = model.geom_dataid[gid]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = model.mesh_vert[va : va + vn]
    rot = data.geom_xmat[gid].reshape(3, 3)
    world = verts @ rot.T + data.geom_xpos[gid]
    lo = np.minimum(lo, world.min(axis=0))
    hi = np.maximum(hi, world.max(axis=0))
  center = (lo + hi) / 2.0
  radius = float(np.linalg.norm(hi - lo) / 2.0)
  return center, max(radius * 2.4, 0.16)


def render_grasp(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  width: int,
  height: int,
  azim: float = 135.0,
  elev: float = -20.0,
) -> np.ndarray:
  """Offscreen render of the given ``data`` state from a free camera.

  ``data`` must already be forwarded (see :func:`apply_grasp`); this function
  does not mutate it beyond re-running ``mj_forward`` for safety.
  """
  mujoco.mj_forward(model, data)
  center, radius = _scene_bounds(model, data)

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = center
  cam.distance = radius
  cam.azimuth = azim
  cam.elevation = elev

  renderer = mujoco.Renderer(model, height=height, width=width)
  renderer.update_scene(data, camera=cam)
  img = renderer.render().copy()
  del renderer
  return img


def main() -> int:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--data-path", type=Path, default=_REPO_ROOT / "meshdata_20",
    help="root of the GraspQP output tree (default: <repo>/meshdata_20)",
  )
  parser.add_argument(
    "--object", type=str, default=None,
    help="object directory name, unique prefix, or glob (default: first found)",
  )
  parser.add_argument(
    "--source", choices=("dexgrasp", "succ", "failed"), default="dexgrasp",
    help="which .pt artifact to load",
  )
  parser.add_argument(
    "--grasp-index", type=int, default=None,
    help="record index inside the batch (default: highest 'values')",
  )
  parser.add_argument("--top", type=int, default=6,
                      help="how many highest-scoring grasps to render")
  parser.add_argument("--out", type=Path, default=_REPO_ROOT / "tmp" / "grasp_grid.png",
                      help="output PNG for the grid")
  parser.add_argument("--width", type=int, default=480, help="render width per panel")
  parser.add_argument("--height", type=int, default=480, help="render height per panel")
  parser.add_argument("--describe", action="store_true",
                      help="print the data structure and exit")
  parser.add_argument("--viewer", action="store_true", help="open the interactive viewer")
  parser.add_argument("--no-contacts", action="store_true", help="hide contact markers")
  parser.add_argument("--list-objects", action="store_true",
                      help="list available objects and exit")
  args = parser.parse_args()

  data_path = args.data_path.resolve()
  if args.list_objects:
    for p in find_objects(data_path):
      print(p.name)
    return 0

  object_dir = resolve_object(data_path, args.object)
  batch = load_batch(find_grasp_file(object_dir, args.source))

  if args.describe:
    describe(batch, object_dir)
    return 0

  idx = int(np.argmax(batch.values)) if args.grasp_index is None else args.grasp_index
  if not 0 <= idx < batch.n_grasps:
    raise IndexError(f"--grasp-index {idx} out of range [0, {batch.n_grasps})")

  if args.viewer:
    run_viewer(object_dir, batch, idx, contacts=not args.no_contacts)
    return 0

  order = np.argsort(batch.values)[::-1][: args.top].tolist()
  if idx not in order:
    order[0] = idx
  render_grid(
    object_dir,
    batch,
    order,
    args.out,
    width=args.width,
    height=args.height,
    contacts=not args.no_contacts,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())


