# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""AnyGrasp-to-AnyGrasp goal command.

Samples target grasps from a GraspQP ``.dexgrasp.pt`` batch (the "grasp cache"),
following DexterityGen's goal-dynamics idea: the RL policy reconfigures the hand
from the current grasp toward a sampled target grasp.

A grasp is the *full* hand configuration -- ``root_pose`` (7: palm pose in the
object frame) + 20 joint angles -- not just the joints. The next target is
sampled by a KD-tree k-NN search over a 27-d feature (position + sign-fixed
quaternion + joints), mirroring ``scripts/grasp_knn.py``, so goals stay at a
"moderate" distance from the current one instead of being drawn uniformly at
random.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from scipy.spatial import KDTree

from wuji_mjlab.tasks.reorient.mdp.grasp_geometry import grasp_feature

# Finger-major order, matching the Wuji MJCF joint order.
JOINT_NAMES = [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]


def _build_fingertip_targets(
  contact_pts: np.ndarray, contact_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Average the 12 active contact points into per-fingertip targets.

  ``contact_idx`` slot layout (from ``contact_links``): 0-11 palm, then 4 slots
  per fingertip (12-15 finger1, 16-19 finger2, ..., 28-31 finger5). Returns
  ``(target, mask)`` of shapes (N, 5, 3) and (N, 5).
  """
  n = contact_pts.shape[0]
  target = np.zeros((n, 5, 3), dtype=np.float64)
  count = np.zeros((n, 5), dtype=np.int64)
  for g in range(n):
    for j in range(contact_idx.shape[1]):
      slot = int(contact_idx[g, j])
      if slot >= 12:
        finger = (slot - 12) // 4
        target[g, finger] += contact_pts[g, j]
        count[g, finger] += 1
  mask = count > 0
  target = np.where(mask[..., None], target / np.maximum(count[..., None], 1), 0.0)
  return target, mask.astype(np.float64)


class GraspGoalCommand(CommandTerm):
  """Command that holds a sampled target grasp (root_pose + joints) per env."""

  cfg: "GraspGoalCommandCfg"

  def __init__(self, cfg: "GraspGoalCommandCfg", env):
    super().__init__(cfg, env)

    data = torch.load(cfg.data_path, map_location="cpu", weights_only=False)
    params = data["parameters"]
    root_pose = params["root_pose"].numpy()  # (N, 7)
    joints = np.stack([params[n].numpy() for n in JOINT_NAMES], axis=-1)  # (N, 20)
    contact_pts = data["fingertip_info"]["nearest_pts"].numpy()  # (N, 12, 3)
    contact_idx = data["contact_idx"].numpy()  # (N, 12)

    self._grasp_root_pose = root_pose.astype(np.float64)
    self._grasp_joints = joints.astype(np.float64)
    self._n_grasps = int(self._grasp_joints.shape[0])
    self._grasp_fingertip_target, self._grasp_fingertip_mask = _build_fingertip_targets(
      contact_pts, contact_idx
    )

    # 27-d features + KD-tree over the whole grasp cache.
    self._features = np.stack(
      [grasp_feature(root_pose[i], joints[i]) for i in range(self._n_grasps)]
    )
    self._tree = KDTree(self._features)

    # Normalized joints for the reward (robot soft limits, env 0; limits are
    # shared across envs here since joint-limit DR is off).
    robot = env.scene[cfg.robot_entity_name]
    soft = robot.data.soft_joint_pos_limits[0].cpu().numpy()  # (20, 2)
    center = 0.5 * (soft[:, 0] + soft[:, 1])
    half = 0.5 * (soft[:, 1] - soft[:, 0])
    self._grasp_joints_norm = ((joints - center) / (half + 1e-6)).clip(-1.0, 1.0)

    self.goal_root_pose = torch.zeros(self.num_envs, 7, device=self.device)
    self.goal_joints = torch.zeros(self.num_envs, 20, device=self.device)
    self.goal_joints_norm = torch.zeros(self.num_envs, 20, device=self.device)
    self.goal_fingertip_target = torch.zeros(self.num_envs, 5, 3, device=self.device)
    self.goal_fingertip_mask = torch.zeros(self.num_envs, 5, device=self.device)
    self._goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    self.metrics["joint_err"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["pose_err"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.goal_joints_norm

  def _set_goal(self, env_ids: torch.Tensor, idx: np.ndarray) -> None:
    """Write the grasp at cache indices ``idx`` into the per-env goal buffers."""
    idx = np.asarray(idx)
    self._goal_idx[env_ids] = torch.as_tensor(idx, device=self.device)
    self.goal_root_pose[env_ids] = torch.as_tensor(
      self._grasp_root_pose[idx], device=self.device, dtype=torch.float32
    )
    self.goal_joints[env_ids] = torch.as_tensor(
      self._grasp_joints[idx], device=self.device, dtype=torch.float32
    )
    self.goal_joints_norm[env_ids] = torch.as_tensor(
      self._grasp_joints_norm[idx], device=self.device, dtype=torch.float32
    )
    self.goal_fingertip_target[env_ids] = torch.as_tensor(
      self._grasp_fingertip_target[idx], device=self.device, dtype=torch.float32
    )
    self.goal_fingertip_mask[env_ids] = torch.as_tensor(
      self._grasp_fingertip_mask[idx], device=self.device, dtype=torch.float32
    )

  def _sample_random(self, env_ids: torch.Tensor) -> None:
    """Uniform random target (used for the initial grasp on reset)."""
    idx = np.random.randint(self._n_grasps, size=len(env_ids))
    self._set_goal(env_ids, idx)

  def _sample_knn(self, env_ids: torch.Tensor) -> None:
    """Sample the next target as a k-NN neighbour of the *current* target.

    Queries the KD-tree with the current goal's 27-d feature and picks uniformly
    from its ``knn_k`` nearest neighbours (self excluded), keeping consecutive
    goals at a moderate distance -- the "goal dynamics" from DexterityGen.
    """
    q_idx = self._goal_idx[env_ids].cpu().numpy()
    query = self._features[q_idx]
    k = min(self.cfg.knn_k + 1, self._n_grasps)  # +1 for the query itself
    _, nn = self._tree.query(query, k=k)
    nn = np.atleast_2d(nn)

    picks = np.empty(len(env_ids), dtype=np.int64)
    for i in range(len(env_ids)):
      cands = nn[i][nn[i] != q_idx[i]]  # drop self
      picks[i] = cands[np.random.randint(len(cands))]
    self._set_goal(env_ids, picks)

  def set_initial_grasp_idx(self, env_ids: torch.Tensor, idx: np.ndarray) -> None:
    """Record the grasp index of the initial state (set by the reset event).

    The KNN goal is then sampled from this grasp so the first target is a
    "moderate distance" neighbour of where the hand actually starts.
    """
    self._goal_idx[env_ids] = torch.as_tensor(np.asarray(idx), device=self.device)

  def reset(self, env_ids: torch.Tensor):
    """Base reset: the KNN goal is sampled from the initial grasp index."""
    return super().reset(env_ids)

  def _update_metrics(self) -> None:
    robot = self._env.scene[self.cfg.robot_entity_name]
    joint_pos = robot.data.joint_pos
    soft = robot.data.soft_joint_pos_limits
    center = 0.5 * (soft[..., 0] + soft[..., 1])
    half = 0.5 * (soft[..., 1] - soft[..., 0])
    q_norm = ((joint_pos - center) / (half + 1e-6)).clamp(-1.0, 1.0)
    self.metrics["joint_err"] = torch.norm(q_norm - self.goal_joints_norm, dim=-1)
    # pose error is filled by the reward term (which has the object pose).

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sample_mode == "random":
      self._sample_random(env_ids)
    else:
      self._sample_knn(env_ids)

  def _update_command(self) -> None:
    # Goal switching is driven by ``resampling_time_range`` in the base class.
    pass


@dataclass(kw_only=True)
class GraspGoalCommandCfg(CommandTermCfg):
  data_path: Path
  robot_entity_name: str = "robot"
  sample_mode: str = "knn"  # "knn" | "random"
  knn_k: int = 10

  def build(self, env) -> GraspGoalCommand:
    return GraspGoalCommand(self, env)

