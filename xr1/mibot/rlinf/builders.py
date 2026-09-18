"""RLinf model builder for a locally checked-out XR-1 model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .ppo_policy import XR1PPOPolicy
from .robocasa365_adapter import XR1RoboCasa365ObservationAdapter


def build_xr1_policy(cfg: Any, torch_dtype: torch.dtype | None = None) -> XR1PPOPolicy:
    """Build the native XR-1 model and wrap it for RLinf.

    ``cfg.model_path`` points at a Xiaomi HF checkpoint directory.  The native
    custom-code model is loaded with ``trust_remote_code``.  A ``state_dict``
    checkpoint can be supplied through ``cfg.checkpoint`` when using the
    Lightning-style XR-1 implementation instead.
    """
    model_path = str(cfg.model_path)
    from transformers import AutoModel

    dtype = torch_dtype or torch.bfloat16
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=getattr(cfg, "attn_implementation", "eager"),
        local_files_only=bool(getattr(cfg, "local_files_only", True)),
    )
    checkpoint = getattr(cfg, "checkpoint", None)
    if checkpoint:
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        state = state.get("module", state.get("state_dict", state))
        model.load_state_dict(state, strict=False)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=bool(getattr(cfg, "local_files_only", True)),
    )
    action_decoder = None
    if bool(getattr(cfg, "decode_actions", False)):
        robot_type = str(getattr(cfg, "robot_type", "robocasa365"))

        def action_decoder(actions: torch.Tensor) -> torch.Tensor:
            return processor.decode_action(actions, robot_type=robot_type)

    action_horizon = int(getattr(cfg, "action_horizon", 16))
    obs_to_batch = XR1RoboCasa365ObservationAdapter(
        processor,
        robot_type=str(getattr(cfg, "robot_type", "robocasa365")),
        state_dim=int(getattr(cfg, "state_dim", 60)),
        state_length=int(getattr(cfg, "state_length", 4)),
        video_history=int(getattr(cfg, "video_history", 1)),
        history_interval=int(getattr(cfg, "history_interval", 2)),
        crop_ratio=float(getattr(cfg, "crop_ratio", 0.95)),
        action_horizon=action_horizon,
    )

    return XR1PPOPolicy(
        model,
        action_dim=int(getattr(cfg, "action_dim", 60)),
        # RoboCasa365's processor emits a 16-step x 60-dim action mask;
        # native XR-1 post-training batches use the original 30-step horizon.
        action_horizon=action_horizon,
        add_value_head=bool(getattr(cfg, "add_value_head", True)),
        trainable_mode=str(getattr(cfg, "trainable_mode", "full")),
        obs_to_batch=obs_to_batch,
        action_std=float(getattr(cfg, "action_std", 0.2)),
        learn_action_std=bool(getattr(cfg, "learn_action_std", False)),
        action_to_env=action_decoder,
        env_action_dim=getattr(cfg, "env_action_dim", 12),
        clip_normalized_action=getattr(cfg, "clip_normalized_action", None),
        likelihood_mode=str(getattr(cfg, "likelihood_mode", "gaussian")),
        flow_trace_samples=int(getattr(cfg, "flow_trace_samples", 1)),
    )
