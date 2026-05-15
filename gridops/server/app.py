"""
FastAPI application for the GridOps Microgrid Environment.

Uses OpenEnv's create_app for standard /ws, /health, /schema, /web endpoints.
Adds custom STATEFUL /reset and /step endpoints for the dashboard (HTTP).
"""

from __future__ import annotations

import copy
import os
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openenv.core.env_server.http_server import create_app

from gridops.episode_logging import episode_logger
from gridops.models import GridOpsAction, GridOpsObservation
from gridops.server.environment import GridOpsEnvironment
from gridops.tasks.definitions import TASKS
from gridops.tool_agent import (
    DEFAULT_COMPARE_HORIZON,
    DEFAULT_OPTIMIZER_HORIZON,
    PlanInputs,
    action_dict,
    optimize_action,
    plan_action,
    previous_outcome_from_observation,
    validate_action_payload,
)

mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("image/svg+xml", ".svg")

# Create the OpenEnv app (provides /ws, /health, /schema, /web, /docs)
app = create_app(
    GridOpsEnvironment,
    GridOpsAction,
    GridOpsObservation,
    env_name="gridops",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "10")),
)


@app.middleware("http")
async def add_static_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(("/assets/", "/evals/")):
        response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
    return response

# ── Shared stateful environment for HTTP dashboard ───────────────────────
# OpenEnv HTTP /reset and /step are stateless (new env per request).
# The dashboard needs persistent state between reset → step → step...
# We maintain a single shared environment instance for HTTP usage.

_dashboard_env = GridOpsEnvironment()


def _dashboard_observation_snapshot() -> dict[str, Any]:
    """Return current dashboard observation without advancing live forecast RNG."""
    env_copy = copy.deepcopy(_dashboard_env)
    return env_copy._make_observation(0.0, env_copy.state.done, "").model_dump()  # noqa: SLF001


class ResetBody(BaseModel):
    seed: int | None = 42
    task_id: str = "task_1_normal"


class StepBody(BaseModel):
    action: dict[str, Any]


class OptimizeBody(BaseModel):
    task_id: str | None = None
    observation: dict[str, Any] | None = None
    previous_outcome: dict[str, Any] | None = None
    horizon: int = DEFAULT_OPTIMIZER_HORIZON


class ValidateBody(BaseModel):
    action: dict[str, Any] | None = None
    completion: str | None = None


class PlanBody(BaseModel):
    task_id: str | None = None
    observation: dict[str, Any] | None = None
    previous_action: dict[str, Any] | None = None
    previous_outcome: dict[str, Any] | None = None
    model_action: dict[str, Any] | None = None
    model_completion: str | None = None
    strategy: dict[str, Any] | str | None = None
    use_llm: bool = False
    optimizer_horizon: int = DEFAULT_OPTIMIZER_HORIZON
    compare_horizon: int = DEFAULT_COMPARE_HORIZON


@app.post("/api/reset")
def dashboard_reset(body: ResetBody):
    """Reset the shared dashboard environment."""
    obs = _dashboard_env.reset(seed=body.seed, task_id=body.task_id)
    episode_logger.append(
        _dashboard_env.state.episode_id,
        "reset",
        {
            "task_id": _dashboard_env.state.task_id,
            "seed": body.seed,
            "observation": obs.model_dump(),
        },
    )
    return {"observation": obs.model_dump()}


@app.post("/api/step")
def dashboard_step(body: StepBody):
    """Execute one step in the shared dashboard environment."""
    before = _dashboard_env.state.model_dump()
    obs_before = _dashboard_observation_snapshot()
    action = GridOpsAction(**body.action)
    obs = _dashboard_env.step(action)
    episode_logger.append(
        _dashboard_env.state.episode_id,
        "step",
        {
            "task_id": _dashboard_env.state.task_id,
            "hour_before": before.get("hour"),
            "action": action_dict(action),
            "observation_before": obs_before,
            "observation": obs.model_dump(),
            "grade": _dashboard_env.state.grade,
        },
    )
    return {"observation": obs.model_dump()}


@app.get("/api/state")
def dashboard_state():
    """Get current state of the shared dashboard environment."""
    return _dashboard_env.state.model_dump()


@app.post("/api/tools/optimize")
def tool_optimize(body: OptimizeBody):
    """Return a causal LP optimizer action without stepping the environment."""
    task_id = body.task_id or _dashboard_env.state.task_id
    obs = body.observation or _dashboard_observation_snapshot()
    previous_outcome = body.previous_outcome or previous_outcome_from_observation(obs)
    action, info = optimize_action(obs, task_id, previous_outcome=previous_outcome, horizon=body.horizon)
    return {
        "task_id": task_id,
        "action": action_dict(action),
        "info": info,
    }


@app.post("/api/tools/validate")
def tool_validate(body: ValidateBody):
    """Validate a raw action dict or model completion."""
    payload = body.completion if body.completion is not None else body.action
    return validate_action_payload(payload)


@app.post("/api/plan")
def dashboard_plan(body: PlanBody):
    """Plan one action from the current dashboard state without stepping."""
    task_id = body.task_id or _dashboard_env.state.task_id
    obs = body.observation or _dashboard_observation_snapshot()
    result = plan_action(
        _dashboard_env,
        PlanInputs(
            task_id=task_id,
            observation=obs,
            previous_action=body.previous_action,
            previous_outcome=body.previous_outcome,
            model_action=body.model_action,
            model_completion=body.model_completion,
            strategy=body.strategy,
            use_llm=body.use_llm,
            optimizer_horizon=body.optimizer_horizon,
            compare_horizon=body.compare_horizon,
        ),
    )
    episode_logger.append(
        _dashboard_env.state.episode_id,
        "plan",
        {
            "task_id": task_id,
            "observation": obs,
            "plan": result,
        },
    )
    return result


@app.post("/api/strategy/plan")
def dashboard_strategy_plan(body: PlanBody):
    """Plan one strategy-mediated action without stepping."""
    return dashboard_plan(body)


# ── Custom endpoints ─────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """List available tasks with their descriptions."""
    return {
        "tasks": [
            {
                "id": "task_1_normal",
                "name": "Normal Summer",
                "difficulty": "Easy",
                "description": "Clear skies, ~100 kW avg demand, Rs 3-12 prices. Tests basic battery arbitrage.",
                "oracle_score": 0.79,
            },
            {
                "id": "task_2_heatwave",
                "name": "Heatwave + Price Spike",
                "difficulty": "Medium",
                "description": "Day 2-3 heatwave (+30% demand), Rs 20 price spike. Tests temporal planning via forecast.",
                "oracle_score": 0.81,
            },
            {
                "id": "task_3_crisis",
                "name": "Extreme Crisis + Grid Outage",
                "difficulty": "Hard",
                "description": "Full 3-day heatwave, -30% solar, +50% demand, limited diesel, 6-hour grid outage. Tests islanding.",
                "oracle_score": 0.70,
            },
        ]
    }


# ── Root and project pages ───────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
EVALS_DIR = REPO_ROOT / "evals"

@app.get("/")
def root_serve():
    """Serve dashboard directly at root so HF Space iframe shows the UI."""
    index = STATIC_DIR / "index.html"
    return HTMLResponse(content=index.read_text(), status_code=200)


@app.get("/web")
def web_serve():
    """Serve dashboard at /web (OpenEnv default web UI path)."""
    index = STATIC_DIR / "index.html"
    return HTMLResponse(content=index.read_text(), status_code=200)


@app.get("/case-study")
def case_study_serve():
    """Serve the Capabl Machines GridOps project case study."""
    index = STATIC_DIR / "case-study.html"
    return HTMLResponse(content=index.read_text(), status_code=200)


# ── Serve dashboard static files ────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(STATIC_DIR), html=True), name="dashboard")
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
if EVALS_DIR.exists():
    app.mount("/evals", StaticFiles(directory=str(EVALS_DIR)), name="evals")


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
