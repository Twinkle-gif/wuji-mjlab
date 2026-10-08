# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""GraspQP <-> Wuji MJCF coordinate mapping.

Mirrors the conventions verified in ``scripts/visualize_grasps.py``:

* ``root_pose`` is ``[px, py, pz, qw, qx, qy, qz]`` -- the palm origin +
  orientation expressed in the *GraspQP object frame* (object at origin,
  identity orientation).
* GraspQP palm frame: X=thumb, Y=index, Z=palm normal.
* Wuji MJCF palm-link frame: X=palm normal, Y=thumb, Z=index.

A grasp is therefore ``root_pose`` (7) + 20 joint angles -- not just the joints.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# GraspQP palm frame -> Wuji MJCF palm-link frame (column permutation matrix).
ROT_GRASPP_TO_MJCF = np.array(
  [
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
  ]
)

# Offset from the GraspQP palm origin to the MJCF palm-link origin, expressed in
# the GraspQP palm frame.
OFFSET_PALM_LOCAL = np.array([0.0, -0.08, -0.01])


def quat_to_mat(qwxyz: np.ndarray) -> np.ndarray:
  """Unit quaternion (w, x, y, z) -> 3x3 rotation matrix."""
  x, y, z, w = qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]
  return Rotation.from_quat([x, y, z, w]).as_matrix()


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
  """3x3 rotation matrix -> unit quaternion (w, x, y, z)."""
  x, y, z, w = Rotation.from_matrix(np.asarray(mat, float).reshape(3, 3)).as_quat()
  return np.array([w, x, y, z])


def fix_quat_sign(qwxyz: np.ndarray) -> np.ndarray:
  """Canonicalize a wxyz quaternion so ``q`` and ``-q`` map to the same vector."""
  q = np.asarray(qwxyz, float).copy()
  if q[0] < 0.0:
    q = -q
  return q


def grasp_feature(root_pose: np.ndarray, joints: np.ndarray) -> np.ndarray:
  """(27,) feature = [position(3), sign-fixed quaternion(4), joints(20)]."""
  pos = np.asarray(root_pose[:3], float)
  quat = fix_quat_sign(root_pose[3:])
  return np.concatenate([pos, quat, np.asarray(joints, float)])


def mjcf_palm_pose(root_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """``root_pose`` (GraspQP object frame) -> MJCF palm-link (pos, quat) in object frame."""
  rp = np.asarray(root_pose, float)
  rot_graspp = quat_to_mat(rp[3:])
  pos = rp[:3] + rot_graspp @ OFFSET_PALM_LOCAL
  rot_mjcf = rot_graspp @ ROT_GRASPP_TO_MJCF
  return pos, mat_to_quat(rot_mjcf)


def object_pose_in_hand_frame(root_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Object pose in the (origin-located) MJCF palm frame that realises ``root_pose``.

  The hand is kept fixed at the origin, so the grasp is realised by moving the
  object instead. ``root_pose`` maps hand-local points into the object frame,
  hence the inverse here.
  """
  pos_h, quat_h = mjcf_palm_pose(root_pose)
  rot_h = quat_to_mat(quat_h)
  return -(rot_h.T @ pos_h), mat_to_quat(rot_h.T)


def root_pose_from_mjcf(pos_mjcf: np.ndarray, quat_mjcf: np.ndarray) -> np.ndarray:
  """Inverse of :func:`mjcf_palm_pose`: MJCF palm pose -> (7,) GraspQP root_pose."""
  rot_mjcf = quat_to_mat(quat_mjcf)
  rot_graspp = rot_mjcf @ ROT_GRASPP_TO_MJCF.T
  pos_graspp = np.asarray(pos_mjcf, float) - rot_graspp @ OFFSET_PALM_LOCAL
  return np.concatenate([pos_graspp, mat_to_quat(rot_graspp)])
