"""FastAPI application for the Portfolio Reasoning OpenEnv.

Uses OpenEnv's `create_app` to expose standard endpoints:
  - WebSocket  /ws       — preferred (one env instance per session)
  - HTTP       /reset    — stateless episode start
  - HTTP       /step     — stateless step (each call = new env)
  - HTTP       /state    — current state of last env
  - HTTP       /metadata — env description, version, README
  - HTTP       /schema   — Pydantic schemas of Action / Observation / State
  - HTTP       /health   — liveness check
  - HTTP       /web      — bundled inspector UI

Plus a small dashboard route mounted from `static/` if present.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openenv.core.env_server.http_server import create_app

from portfolio_env import (
    PortfolioAction, PortfolioEnv, PortfolioObs,
)


# ── OpenEnv-standard FastAPI app (provides /ws /health /schema /web /docs /metadata) ──
app: FastAPI = create_app(
    PortfolioEnv,
    PortfolioAction,
    PortfolioObs,
    env_name='portfolio-env',
    max_concurrent_envs=int(os.environ.get('MAX_CONCURRENT_ENVS', '10')),
)


# ── Stateful HTTP routes for dashboard / Greenberg Terminal UI ──
# Each WebSocket session already gets its own PortfolioEnv via create_app.
# These routes are for HTTP clients (notably brother's UI) that want a
# persistent env across reset/step calls without managing WebSocket state.

_dashboard_env = PortfolioEnv()


class ResetBody(BaseModel):
    seed: int | None = 42
    phase: int = 3


class StepBody(BaseModel):
    action: dict[str, Any]
    completion: str = ''


@app.post('/api/reset')
def dashboard_reset(body: ResetBody):
    obs = _dashboard_env.reset(seed=body.seed, phase=body.phase)
    return {'observation': obs.model_dump()}


@app.post('/api/step')
def dashboard_step(body: StepBody):
    action = PortfolioAction(**body.action)
    obs = _dashboard_env.step(action, completion=body.completion)
    return {'observation': obs.model_dump()}


@app.get('/api/state')
def dashboard_state():
    return _dashboard_env.state.model_dump()


# ── Friendly tasks / phases listing ──

@app.get('/phases')
def list_phases():
    return {
        'phases': [
            {'id': 1, 'name': 'Format + regret', 'difficulty': 'easy',
             'description': '4Q episodes, easy shocks only (6 in pool, sample 2). '
                            'Tests basic JSON shape + baseline beating.'},
            {'id': 2, 'name': 'Ambiguity', 'difficulty': 'medium',
             'description': '8Q episodes, easy + ambiguous shocks (12, sample 3). '
                            'Adds drawdown penalty + infra_commit intervention.'},
            {'id': 3, 'name': 'Full task', 'difficulty': 'hard',
             'description': '12Q episodes (full bull-bear cycle), all 17 shocks (sample 5). '
                            'All 4 interventions, full carbon penalty. The submission target.'},
        ]
    }


# ── Static dashboard mount (Greenberg Terminal lives here once brother ships it) ──

STATIC_DIR = Path(__file__).parent / 'static'
if STATIC_DIR.exists():
    app.mount('/dashboard', StaticFiles(directory=str(STATIC_DIR), html=True), name='dashboard')


@app.get('/')
def root():
    """Serve the dashboard at root if present, else a minimal landing page."""
    index = STATIC_DIR / 'index.html'
    if index.exists():
        return HTMLResponse(content=index.read_text(), status_code=200)
    return HTMLResponse(
        content=(
            '<html><body style="font-family:monospace;background:#0a0e14;color:#b3b1ad;padding:2em">'
            '<h1>Portfolio Reasoning OpenEnv</h1>'
            '<p>This Space hosts an OpenEnv-compliant environment.</p>'
            '<ul>'
            '<li><a href="/docs" style="color:#7fdbca">/docs</a> — interactive API docs</li>'
            '<li><a href="/schema" style="color:#7fdbca">/schema</a> — Pydantic schemas</li>'
            '<li><a href="/metadata" style="color:#7fdbca">/metadata</a> — env description + README</li>'
            '<li><a href="/health" style="color:#7fdbca">/health</a> — liveness</li>'
            '<li><a href="/web" style="color:#7fdbca">/web</a> — OpenEnv inspector UI</li>'
            '<li><a href="/phases" style="color:#7fdbca">/phases</a> — curriculum phases</li>'
            '</ul>'
            '<p>WebSocket protocol available at <code>/ws</code>.</p>'
            '</body></html>'
        ),
        status_code=200,
    )


def main(host: str = '0.0.0.0', port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()
    main(port=args.port)
