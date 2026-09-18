# Copyright (C) 2026 Xiaomi Corporation.
"""CPU checks for the RLinf embodied PPO math path used by XR-1."""

import torch

from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss


def test_embodied_gae_and_actor_critic_loss_backpropagate():
    # Rollout tensors use RLinf's [time, batch, action_chunks] layout.
    steps, batch_size, chunks, action_dim = 3, 1, 1, 4
    rewards = torch.tensor([[[0.0]], [[0.5]], [[1.0]]])
    dones = torch.zeros(steps + 1, batch_size, chunks, dtype=torch.bool)
    dones[-1] = True
    values = torch.zeros(steps + 1, batch_size, chunks)
    rollout_mask = torch.ones(steps, batch_size, chunks, dtype=torch.bool)

    advantages = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="gae",
        rewards=rewards,
        dones=dones,
        values=values,
        num_action_chunks=chunks,
        gamma=0.99,
        gae_lambda=0.95,
        group_size=1,
        reward_type="chunk_level",
        loss_mask=rollout_mask,
        normalize_advantages=True,
    )
    assert advantages["advantages"].shape == (steps, batch_size, chunks)
    assert advantages["returns"].shape == (steps, batch_size, chunks)
    assert torch.isfinite(advantages["advantages"]).all()

    logprobs = torch.randn(steps, action_dim, requires_grad=True)
    # Keep values inside the PPO value-clip interval so this regression test
    # always exercises the critic gradient rather than randomly clipping it.
    current_values = torch.full((steps,), 0.1, requires_grad=True)
    old_logprobs = logprobs.detach() + 0.01
    loss, metrics = policy_loss(
        loss_type="actor_critic",
        task_type="embodied",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=action_dim,
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages["advantages"],
        values=current_values,
        returns=advantages["returns"],
        prev_values=torch.zeros_like(current_values),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=torch.ones(steps, 1, dtype=torch.bool),
        max_episode_steps=steps,
        critic_warmup=False,
    )
    assert metrics
    assert torch.isfinite(loss)
    loss.backward()
    assert logprobs.grad is not None and torch.isfinite(logprobs.grad).all()
    assert current_values.grad is not None and torch.isfinite(current_values.grad).all()
    assert float(logprobs.grad.abs().sum()) > 0.0
    assert float(current_values.grad.abs().sum()) > 0.0
