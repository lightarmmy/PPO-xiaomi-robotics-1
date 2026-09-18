#!/usr/bin/env python3
"""Real XR-1 checkpoint GPU preflight for the RLinf PPO adapter."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch


def move(value):
    if isinstance(value, torch.Tensor):
        return value.to(device="cuda", non_blocking=True)
    if isinstance(value, dict):
        return {key: move(item) for key, item in value.items()}
    return value


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the XR-1 preflight")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "xr1"))
    sys.path.insert(0, os.path.join(root, "third_party", "RLinf"))

    from mibot.rlinf.builders import build_xr1_policy

    ckpt = os.environ.get(
        "XR1_ROBOCASA365_CKPT",
        os.path.join(root, "checkpoints", "Xiaomi-Robotics-1-RoboCasa365"),
    )
    cfg = SimpleNamespace(
        model_path=ckpt,
        local_files_only=True,
        attn_implementation=os.environ.get("XR1_ATTN", "eager"),
        action_dim=60,
        action_horizon=16,
        state_dim=60,
        state_length=4,
        video_history=1,
        robot_type="robocasa365",
        decode_actions=True,
        env_action_dim=12,
        add_value_head=True,
        trainable_mode=os.environ.get("XR1_TRAINABLE_MODE", "action_expert"),
        action_std=0.2,
        clip_normalized_action=1.5,
    )
    print(f"[preflight] loading checkpoint: {ckpt}", flush=True)
    policy = build_xr1_policy(cfg, torch_dtype=torch.bfloat16).cuda()
    policy.train()
    print(f"[preflight] parameters: {policy.parameter_report()}", flush=True)

    rng = np.random.default_rng(0)
    # The adapter accepts the same batched keys emitted by Robocasa365Env.
    frame = rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8)
    env_obs = {
        "main_images": frame,
        "wrist_images": frame.copy(),
        "extra_view_images": frame[:, None].copy(),
        "states": np.zeros((1, 60), dtype=np.float32),
        "task_descriptions": ["Open the cabinet door"],
    }
    batch = move(policy.obs_to_batch(env_obs))
    print(
        "[preflight] batch shapes:",
        {key: tuple(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
        flush=True,
    )
    # Rollout API accepts environment observations and performs the adapter
    # conversion internally.  The already-built batch is retained below for
    # the differentiable PPO forward/backward check.
    actions, result = policy.predict_action_batch(env_obs, mode="train")
    print(f"[preflight] rollout action shape={actions.shape}", flush=True)
    forward_inputs = move(result["forward_inputs"])
    output = policy.default_forward(
        forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
    )
    for key in ("xr1_mean_abs_diff", "xr1_mean_max_diff", "xr1_logprob_abs_diff", "xr1_logprob_max_diff"):
        if key in output:
            print(f"[preflight] {key}={float(output[key].detach()):.6e}", flush=True)
    loss = output["logprobs"].float().mean() + output["values"].float().mean()
    if output["entropy"] is not None:
        loss = loss - 1e-3 * output["entropy"].float().mean()
    loss.backward()
    finite = True
    grad_total = 0.0
    grad_tensors = 0
    for name, param in policy.named_parameters():
        if param.grad is None:
            continue
        grad_tensors += 1
        grad_total += float(param.grad.float().abs().sum().item())
        finite = finite and bool(torch.isfinite(param.grad).all().item())
    print(
        f"[preflight] loss={float(loss.detach()):.6f} grad_tensors={grad_tensors} "
        f"grad_abs_sum={grad_total:.6e} finite={finite}",
        flush=True,
    )
    if not finite or grad_tensors == 0 or grad_total == 0.0:
        raise RuntimeError("PPO backward produced no finite non-zero gradients")
    print("XR1 real-checkpoint PPO preflight PASSED", flush=True)


if __name__ == "__main__":
    main()
