# XR-1 RLinf PPO adapter

`XR1PPOPolicy` implements RLinf's embodied `BasePolicy` contract:

* `predict_action_batch()` returns environment actions plus `prev_logprobs`,
  `prev_values`, and replayable `forward_inputs`.
* `default_forward()` recomputes PPO log-probabilities and values.
* `set_trainable_mode("full")` enables all XR-1 parameters; `"action_expert"`
  freezes the VLM and choice heads and leaves the DiT/action path trainable.

The released XR-1 sampler is an ODE generator without an explicit density.
Two likelihood modes are available:

* `gaussian` (default) retains the diagonal-Gaussian surrogate around the
  generated action.
* `flow_hutchinson` treats the sampler as a probability-flow ODE. Rollout
  actions are the ODE outputs from standard Gaussian initial noise; PPO
  re-integrates backwards from the fixed action and estimates the accumulated
  velocity divergence with cached Rademacher Hutchinson probes. This removes
  the extra action-space Gaussian, but remains an estimator because XR-1 uses
  a finite-step Euler solver and a stochastic trace estimate.

For `flow_hutchinson`, set `clip_normalized_action: null`: clipping changes
the action distribution and invalidates the change-of-variables likelihood.
Start with `xr1_robocasa365_ppo_flow_smoke`, one trace probe, and one PPO
update; its Hessian-vector products are materially more expensive than the
surrogate path.

Example (from the `xr1/` checkout after installing RLinf):

```python
from mibot.rlinf import XR1PPOPolicy
policy = XR1PPOPolicy(model, action_dim=60, action_horizon=30,
                      trainable_mode="action_expert")
print(policy.parameter_report())
```

The vendored RLinf checkout registers `xr1_ppo` automatically. Use
`actor.model.model_type: xr1_ppo` and set
`actor.model.model_path` to the local `Xiaomi-Robotics-1-RoboCasa365` directory.
The builder loads the Hugging Face custom-code model and accepts optional
`checkpoint`, `trainable_mode`, `action_std`, `action_dim`, and `action_horizon`
fields. Its default `action_horizon` is 16 and `env_action_dim` is 12 for
RoboCasa365; set `action_horizon: 30` and `decode_actions: false` for the
native XR-1 post-training batch format. With `decode_actions: true`, the
builder loads the checkpoint processor, applies `robot_type` normalization
(default `robocasa365`), and returns decoded environment actions while keeping
60D normalized actions in the PPO replay fields. For an external RLinf
checkout, import and call `register_xr1_model()` once before starting the
runner.
