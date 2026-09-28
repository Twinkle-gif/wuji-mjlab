# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Debug/learning entry point: build the reorient env and step it a few times.

Point the VS Code debugger at this file (see ``.vscode/launch.json``) and set
breakpoints anywhere in ``mjlab`` / ``wuji_mjlab`` / ``rsl_rl`` to trace the
full call flow without launching an 8192-env training run (which needs ~20 GB
VRAM).

It uses ``play=True`` (4 envs, no domain-randomization / disturbance events,
curriculum cleared) so it is light enough for a laptop GPU.

Good breakpoints to start from::

    wuji_mjlab.tasks.reorient.mdp.commands.InHandReorientCommand._update_command
    wuji_mjlab.tasks.reorient.mdp.actions.JointPositionOffsetEMAAction.process_actions
    wuji_mjlab.tasks.reorient.mdp.rewards.orientation_alignment
    mjlab.envs.manager_based_rl_env.ManagerBasedRlEnv.step

Usage::

    pixi run python scripts/debug_reorient_env.py      # run normally
    # or press F5 in VS Code with the "Debug: reorient env" configuration
"""

from __future__ import annotations

import torch

import wuji_mjlab.tasks  # noqa: F401  (triggers task registration)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks
from wuji_mjlab.utils.task_cfg_utils import prepare_task_cfgs


def main() -> int:
  task_id = "WujiHand_Reorient"

  # [1] Registration: what tasks are available after `import wuji_mjlab.tasks`
  print("[1] registered tasks:", list_tasks())

  # [2] Config assembly: env_cfg + agent_cfg (no sim spawned yet)
  env_cfg, agent_cfg = prepare_task_cfgs(task_id, overrides=[], play=True)
  print("[2] num_envs:", env_cfg.scene.num_envs)
  print("[2] action terms:", list(env_cfg.actions))
  print("[2] obs groups:", list(env_cfg.observations))
  print("[2] reward terms:", list(env_cfg.rewards))
  print("[2] command terms:", list(env_cfg.commands))
  print("[2] event terms:", list(env_cfg.events))

  # [3] Env build: compiles the MuJoCo-Warp scene on GPU
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  print("[3] env built on", device)
  print("[3] action space:", env.action_space.shape)

  # [4] Step the env a few times with zero actions so you can trace the
  #     reset -> command update -> obs -> reward -> termination pipeline.
  action = torch.zeros(env.num_envs, 20, device=device)
  env.reset()
  for i in range(5):
    obs, rewards, terminated, truncated, info = env.step(action)  # noqa: F841
    print(f"[4] step {i}: reward mean={rewards.mean().item():.4f} "
          f"terminated={int(terminated.sum().item())}")

  env.close()
  print("[done]")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
