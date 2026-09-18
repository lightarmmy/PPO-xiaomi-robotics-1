"""Optional RLinf registry hook for the XR-1 adapter.

Import this module from an RLinf launcher after both repositories are on
``PYTHONPATH``.  Registration is opt-in so installing the Xiaomi package does
not mutate a user's global RLinf model registry.
"""

from __future__ import annotations

from typing import Any


def register_xr1_model() -> None:
    from rlinf.models import register_model

    from .builders import build_xr1_policy

    register_model("xr1_ppo", build_xr1_policy, category="embodied", force=True)


__all__ = ["register_xr1_model"]
