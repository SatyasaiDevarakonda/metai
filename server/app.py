"""FastAPI server for the QStorePrice OpenEnv environment.

Exposes the canonical OpenEnv HTTP/WebSocket endpoints required by the
hackathon (PDF page 41):

    GET  /health   - liveness probe
    POST /reset    - start a new episode
    POST /step     - submit an Operating Brief, advance the simulation
    GET  /state    - current FreshPriceState snapshot
    WS   /ws       - persistent session used by the async client
    GET  /docs     - OpenAPI documentation

The server itself is built by openenv-core's `create_fastapi_app` so that we
inherit the standard request/response schemas and don't reinvent the wheel
(hackathon non-negotiable: "Build on top of the framework").

Run:

    # Local development (with reload)
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

    # Production (matches the Dockerfile CMD)
    python -m server.app
"""

from __future__ import annotations

import os

try:
    from openenv.core.env_server import create_fastapi_app
except ImportError as exc:
    raise ImportError(
        "openenv-core is not installed. Run: pip install openenv-core"
    ) from exc

from server.environment import BriefAction, BriefObservation, FreshPriceOpenEnv

# create_fastapi_app takes a factory callable (not an instance) so that the
# session layer can construct per-session envs, plus explicit Action/Observation
# types for request/response schema generation.
app = create_fastapi_app(
    env=FreshPriceOpenEnv,
    action_cls=BriefAction,
    observation_cls=BriefObservation,
)


def main() -> None:
    """CLI entry point used by `python -m server.app` and the Dockerfile."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=workers,
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
