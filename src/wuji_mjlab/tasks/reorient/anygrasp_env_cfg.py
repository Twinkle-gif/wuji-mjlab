# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""AnyGrasp-to-AnyGrasp task configuration.

Reference: DexterityGen (arXiv:2502.04307). The RL policy reconfigures the Wuji
Hand from one grasp to another sampled from a GraspQP grasp cache. A grasp is
the full hand configuration (``root_pose`` + 20 joints); the next target is
sampled by a KD-tree k-NN search (goal dynamics), and the goal reward matches
both the joint angles and the relative hand pose.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from wuji_mjlab.tasks.reorient import mdp
from wuji_mjlab.tasks.reorient.mdp.grasp_goal_command import GraspGoalCommandCfg
from wuji_mjlab.tasks.reorient.reorient_terms import build_reorient_actions

_REPO_ROOT = Path(__file__).resolve().parents[4]

_DEFAULT_GRASP_DATA = (
  _REPO_ROOT
  / "meshdata_20"
  / "core-can-129880fda38f3f2ba1ab68e159bfb347"
  / "grasp_predictions"
  / "wuji"
  / "12_contacts"
  / "graspqp"
  / "scale_0.0661"
  / "full"
  / "core-can-129880fda38f3f2ba1ab68e159bfb347.dexgrasp.pt"
)


def _resolve_object_dir(data_path: Path) -> Path:
  """Walk up from a grasp file to the object dir (the one containing ``coacd/``)."""
  p = data_path
  while not (p / "coacd").is_dir() and p != p.parent:
    p = p.parent
  return p


_DEFAULT_OBJECT_DIR = _resolve_object_dir(_DEFAULT_GRASP_DATA)
# data["scale"] of the core-can mesh: maps the normalized CoACD mesh onto metres.
_DEFAULT_SCALE = 0.06605165239217067


def build_anygrasp_commands(data_path: Path | None = None) -> dict[str, CommandTermCfg]:
  return {
    "anygrasp_command": GraspGoalCommandCfg(
      data_path=data_path or _DEFAULT_GRASP_DATA,
      robot_entity_name="robot",
      resampling_time_range=(2.0, 2.0),
      sample_mode="knn",
      knn_k=10,
    )
  }


def build_anygrasp_observations() -> dict[str, ObservationGroupCfg]:
  policy_terms = {
    "joint_angles": ObservationTermCfg(func=mdp.joint_pos_limit_normalized),
    "qpos_error": ObservationTermCfg(func=mdp.joint_pos_target_error),
  }
  return {
    "policy": ObservationGroupCfg(terms=policy_terms, concatenate_terms=True),
    "critic": ObservationGroupCfg(terms=policy_terms, concatenate_terms=True),
  }


def build_anygrasp_rewards() -> dict[str, RewardTermCfg]:
  return {
    "grasp_goal_reward": RewardTermCfg(
      func=mdp.grasp_goal_reward,
      weight=15.0,
      params={
        "command_name": "anygrasp_command",
        "lambda_joint": 2.0,
        "lambda_pose": 2.0,
      },
    ),
    "grasp_contact_reward": RewardTermCfg(
      func=mdp.grasp_contact_reward,
      weight=2.0,
      params={"command_name": "anygrasp_command", "lambda_contact": 10.0},
    ),
    "torque": RewardTermCfg(func=mdp.torque_penalty, weight=-24.0),
    "action_rate": RewardTermCfg(func=mdp.action_rate_combined, weight=-1.0),
    "hand_pose": RewardTermCfg(func=mdp.hand_pose_penalty, weight=-0.2),
    "work_penalty": RewardTermCfg(func=mdp.work_penalty, weight=-1.0),
    "fingertip_velocity_penalty": RewardTermCfg(
      func=mdp.fingertip_velocity_penalty, weight=-0.5
    ),
  }


def build_anygrasp_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }


def build_anygrasp_events() -> dict[str, EventTermCfg]:
  return {
    "reset_to_random_grasp": EventTermCfg(
      func=mdp.reset_to_random_grasp,
      mode="reset",
      params={"command_name": "anygrasp_command"},
    ),
  }


def make_anygrasp_env_cfg(
  num_envs: int = 64,
  data_path: Path | None = None,
) -> ManagerBasedRlEnvCfg:
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=num_envs,
      env_spacing=0.75,
      extent=0.8,
    ),
    observations=build_anygrasp_observations(),
    actions=build_reorient_actions(),
    commands=build_anygrasp_commands(data_path=data_path),
    rewards=build_anygrasp_rewards(),
    terminations=build_anygrasp_terminations(),
    events=build_anygrasp_events(),
    sim=SimulationCfg(
      nconmax=180,
      njmax=1500,
      mujoco=MujocoCfg(timestep=0.01, iterations=10, ls_iterations=20),
    ),
    decimation=5,
    episode_length_s=50.0,
  )
