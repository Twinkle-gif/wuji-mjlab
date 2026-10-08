# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Debug entry: build the AnyGrasp-to-AnyGrasp env and step it a few times.

Reference: DexterityGen (arXiv:2502.04307). See
``src/wuji_mjlab/tasks/reorient/anygrasp_env_cfg.py`` for the task design.

Usage::

    pixi run python scripts/debug_anygrasp_env.py
"""

from __future__ import annotations

import torch
import wuji_mjlab.tasks  # noqa: F401  (triggers task registration)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_anygrasp_env_cfg,
)


def main() -> int:
  print("[1] registered:", [t for t in list_tasks() if "AnyGrasp" in t])

  cfg = wuji_hand_anygrasp_env_cfg(num_envs=4)
  print("[2] command terms:", list(cfg.commands))
  print("[2] reward terms:", list(cfg.rewards))
  print("[2] obs groups:", list(cfg.observations))

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  print("[3] env built on", device, "action space:", env.action_space.shape)

  cmd = env.command_manager.get_term("anygrasp_command")
  action = torch.zeros(env.num_envs, 20, device=device)
  env.reset()
  print("[4] goal_joints_norm shape:", tuple(cmd.goal_joints_norm.shape))
  print("[4] goal_root_pose (env0):", [f"{x:.3f}" for x in cmd.goal_root_pose[0].tolist()])
  print("[4] goal idx (envs):", cmd._goal_idx.tolist())
  for i in range(5):
    obs, rewards, terminated, truncated, info = env.step(action)  # noqa: F841
    joint_err = cmd.metrics["joint_err"].mean().item()
    print(f"[4] step {i}: reward mean={rewards.mean().item():.4f}  joint_err={joint_err:.4f}")

  env.close()
  print("[done]")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
