# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Random-grasp reset for the AnyGrasp-to-AnyGrasp task."""

from __future__ import annotations

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from scipy.spatial.transform import Rotation

from wuji_mjlab.tasks.reorient.mdp.event_utils import resolve_env_ids
from wuji_mjlab.tasks.reorient.mdp.grasp_geometry import object_pose_in_hand_frame

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def _compose(
  pos1: np.ndarray, quat1: np.ndarray, pos2: np.ndarray, quat2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """(pos, wxyz quat) of ``transform1 @ transform2`` (batched)."""
  r1 = Rotation.from_quat(np.concatenate([quat1[..., 1:4], quat1[..., :1]], axis=-1))  # xyzw
  r2 = Rotation.from_quat(np.concatenate([quat2[..., 1:4], quat2[..., :1]], axis=-1))
  r = r1 * r2
  pos = pos1 + r1.apply(pos2)
  q = r.as_quat()  # xyzw
  return pos, np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)  # wxyz


def reset_to_random_grasp(
  env,
  env_ids,
  command_name: str = "anygrasp_command",
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  object_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
) -> None:
  """Place the hand (joints) + object into a uniformly sampled grasp from the cache.

  The hand is a fixed-base mocap entity, so the grasp's ``root_pose`` (palm
  relative to object) is realised by moving the *object*::

      object_world = palm_world @ object_pose_in_hand_frame(root_pose)
  """
  env_ids = resolve_env_ids(env, env_ids)
  if env_ids.numel() == 0:
    return

  cmd = env.command_manager.get_term(command_name)
  robot: Entity = env.scene[robot_cfg.name]
  obj: Entity = env.scene[object_cfg.name]

  idx = np.random.randint(cmd._n_grasps, size=len(env_ids))
  root_poses = cmd._grasp_root_pose[idx]  # (n, 7)
  joints = cmd._grasp_joints[idx]  # (n, 20)

  obj_pos_in_hand = np.stack([object_pose_in_hand_frame(rp)[0] for rp in root_poses])
  obj_quat_in_hand = np.stack([object_pose_in_hand_frame(rp)[1] for rp in root_poses])

  palm_pos = robot.data.root_link_pos_w[env_ids].cpu().numpy()
  palm_quat = robot.data.root_link_quat_w[env_ids].cpu().numpy()

  obj_pos_w, obj_quat_w = _compose(palm_pos, palm_quat, obj_pos_in_hand, obj_quat_in_hand)

  obj.write_root_link_pose_to_sim(
    torch.tensor(
      np.concatenate([obj_pos_w, obj_quat_w], axis=-1),
      device=env.device,
      dtype=torch.float32,
    ),
    env_ids=env_ids,
  )
  obj.write_root_link_velocity_to_sim(
    torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids
  )
  robot.write_joint_state_to_sim(
    torch.tensor(joints, device=env.device, dtype=torch.float32),
    torch.zeros(len(env_ids), 20, device=env.device),
    env_ids=env_ids,
  )
  # Record the initial grasp so the command's KNN goal is sampled from it
  # (the first target becomes a moderate-distance neighbour of the start state).
  cmd.set_initial_grasp_idx(env_ids, idx)
