"""CPU-only contract smoke test for the XR-1 RLinf policy wrapper.

This intentionally uses a tiny stand-in model; it validates imports, shape
contracts and full/action-expert parameter masks without loading the 5B model.
"""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xr1"))
sys.path.insert(0, str(ROOT / "third_party" / "RLinf"))

from mibot.rlinf import XR1PPOPolicy  # noqa: E402


class TinyXR1(torch.nn.Module):
    state_shape = (1, 4)

    def __init__(self):
        super().__init__()
        self.vlm = torch.nn.Linear(4, 4)
        self.dit = torch.nn.Linear(4, 4)

    def generate(self, batch, noise=None):
        return batch["action"].float() * 0 + 1

    def generate_with_grad(self, batch, noise=None):
        return self.generate(batch, noise)


def main():
    policy = XR1PPOPolicy(TinyXR1(), action_dim=4, action_horizon=2)
    policy.set_trainable_mode("action_expert")
    assert not policy.xr1_model.vlm.weight.requires_grad
    assert policy.xr1_model.dit.weight.requires_grad
    batch = {"action": torch.zeros(2, 2, 4), "state": torch.zeros(2, 1, 4)}
    actions, result = policy.predict_action_batch({"xr1_batch": batch}, mode="eval")
    assert actions.shape == (2, 2, 4)
    assert result["prev_logprobs"].shape == (2, 2, 4)
    output = policy.default_forward(result["forward_inputs"], compute_values=True)
    assert output["logprobs"].shape == (2, 2, 4)
    print("XR-1 RLinf PPO wrapper smoke passed", policy.parameter_report())


if __name__ == "__main__":
    main()
