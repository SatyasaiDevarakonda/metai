"""OpenEnv environment for QStorePrice — canonical server-side entry point.

Follows the standard OpenEnv project layout (step 2 of "Build Your Own
Environment"): the server imports the Environment subclass from
`server.environment`, which exposes reset(), step(), and state().

The simulation logic lives in `freshprice_env/openenv_adapter.py` so the
training pipeline, evaluator, and Gradio demo can import it without pulling
in server dependencies. This module re-exports everything that server/app.py
and external tools need.
"""

from __future__ import annotations

from freshprice_env.openenv_adapter import (
    BriefAction,
    BriefObservation,
    FreshPriceOpenEnv,
    FreshPriceState,
)

__all__ = ["FreshPriceOpenEnv", "BriefAction", "BriefObservation", "FreshPriceState"]
