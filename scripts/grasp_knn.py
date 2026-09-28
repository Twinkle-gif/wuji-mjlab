# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Find visually-similar grasps in a GraspQP batch with k-nearest neighbours.

Grasp poses live in ``<object>/grasp_predictions/wuji/12_contacts/graspqp/
scale_*/<type>/*.dexgrasp.pt`` under ``meshdata_20/``. Each grasp is
``root_pose`` (7 = position + wxyz quaternion) + 20 joint angles.

This script:
  1. builds one feature vector per grasp (position + sign-fixed quaternion +
     joint angles),
  2. uses a KD-tree (k-NN) to find the ``--k`` nearest neighbours of a query,
  3. renders query + neighbours side-by-side so you can eyeball similarity.

The data is plain tensors with no Isaac-specific serialization, so it renders
with MuJoCo — no Isaac Sim required. The GraspQP->Wuji-MJCF coordinate mapping
is implemented in ``scripts/visualize_grasps.py``.

Usage::

    pixi run python scripts/grasp_knn.py \
        --object core-can-129880fda38f3f2ba1ab68e159bfb347 --k 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize_grasps import (
  apply_grasp,
  compile_grasp_scene,
  find_grasp_file,
  load_batch,
  render_grasp,
  resolve_object,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _fix_quat_sign(quat: np.ndarray) -> np.ndarray:
  """Canonicalize a wxyz quaternion so ``q`` and ``-q`` map to the same vector."""
  q = np.asarray(quat, dtype=float).copy()
  if q[0] < 0.0:
    q = -q
  return q


def grasp_feature(batch, idx: int) -> np.ndarray:
  """(27,) feature = [position(3), sign-fixed quaternion(4), joints(20)]."""
  rp = batch.root_pose(idx)
  pos = np.asarray(rp[:3], dtype=float)
  quat = _fix_quat_sign(rp[3:])
  joints = batch.joints(idx).astype(float)
  return np.concatenate([pos, quat, joints])


def grasp_breakdown(batch, a: int, b: int) -> tuple[float, float, float]:
  """Interpretable (position, orientation, joint) distances between two grasps."""
  ra, rb = batch.root_pose(a), batch.root_pose(b)
  pos_d = float(np.linalg.norm(ra[:3] - rb[:3]))
  qa, qb = _fix_quat_sign(ra[3:]), _fix_quat_sign(rb[3:])
  dot = float(np.clip(np.abs(np.dot(qa, qb)), 0.0, 1.0))
  rot_d = float(2.0 * np.arccos(dot))          # geodesic angle on SO(3), radians
  joint_d = float(np.linalg.norm(batch.joints(a) - batch.joints(b)))
  return pos_d, rot_d, joint_d


def main() -> int:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--data-path", type=Path, default=_REPO_ROOT / "meshdata_20",
    help="root of the GraspQP output tree (default: <repo>/meshdata_20)",
  )
  parser.add_argument("--object", type=str, default=None,
                      help="object directory name / unique prefix (default: first)")
  parser.add_argument("--source", choices=("dexgrasp", "succ", "failed"),
                      default="dexgrasp")
  parser.add_argument("--query-index", type=int, default=None,
                      help="grasp index to query (default: highest 'values')")
  parser.add_argument("--k", type=int, default=3, help="number of neighbours")
  parser.add_argument("--out", type=Path, default=_REPO_ROOT / "tmp" / "grasp_knn.png")
  parser.add_argument("--no-contacts", action="store_true",
                      help="hide contact markers for a cleaner comparison")
  parser.add_argument("--width", type=int, default=420)
  parser.add_argument("--height", type=int, default=420)
  args = parser.parse_args()

  object_dir = resolve_object(args.data_path.resolve(), args.object)
  batch = load_batch(find_grasp_file(object_dir, args.source))
  n = batch.n_grasps

  # 1) Build the feature matrix and the KD-tree.
  features = np.stack([grasp_feature(batch, i) for i in range(n)])  # (N, 27)
  tree = KDTree(features)

  # 2) Query.
  qi = int(np.argmax(batch.values)) if args.query_index is None else args.query_index
  if not 0 <= qi < n:
    raise IndexError(f"--query-index {qi} out of range [0, {n})")

  k_plus_self = min(args.k + 1, n)
  dists, idxs = tree.query(features[qi], k=k_plus_self)
  dists, idxs = np.atleast_1d(dists), np.atleast_1d(idxs)
  keep = idxs != qi                     # drop the query itself (distance 0)
  idxs = idxs[keep][: args.k]
  dists = dists[keep][: args.k]

  print(f"object      : {object_dir.name}")
  print(f"grasp count : {n}")
  print(f"query       : #{qi}  (value={batch.values[qi]:.3f})")
  print(f"\n{'rank':>4s} {'idx':>5s} {'L2 dist':>9s} {'pos(m)':>8s} "
        f"{'rot(deg)':>9s} {'joint(rad)':>11s}")
  for r, (j, d) in enumerate(zip(idxs, dists), start=1):
    p, rot, jd = grasp_breakdown(batch, qi, int(j))
    print(f"{r:4d} {int(j):5d} {d:9.4f} {p:8.4f} {np.degrees(rot):9.2f} {jd:11.4f}")

  # 3) Render query + neighbours side by side (two camera azimuths).
  contacts = not args.no_contacts
  model, data = compile_grasp_scene(object_dir, batch)
  columns = [("query", qi)] + [(f"NN{r}", int(j)) for r, j in enumerate(idxs, start=1)]

  fig = plt.figure(figsize=(3.4 * len(columns), 6.8))
  for row, azim in enumerate((135.0, 45.0), start=1):
    for col, (label, gi) in enumerate(columns, start=1):
      apply_grasp(model, data, batch, gi, contacts=contacts)
      img = render_grasp(model, data, args.width, args.height, azim=azim)
      ax = fig.add_subplot(2, len(columns), (row - 1) * len(columns) + col)
      ax.imshow(img)
      ax.set_xticks([])
      ax.set_yticks([])
      if row == 1:
        dist = "query" if col == 1 else f"{dists[col - 2]:.3f}"
        ax.set_title(f"{label}  (#{gi})\nL2={dist}", fontsize=9)
      if col == 1:
        ax.set_ylabel(f"azimuth {azim:g}", fontsize=9)

  plt.tight_layout()
  args.out.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(args.out, dpi=110)
  plt.close(fig)
  print(f"\n[INFO] wrote {args.out}  (query + {len(idxs)} neighbours x 2 views)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

