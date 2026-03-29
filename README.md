---
title: GridOps
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - microgrid
  - energy
---

# GridOps — Community Microgrid Bridge Operator

An OpenEnv reinforcement learning environment where an AI agent operates a **100-home community microgrid** in an Indian city during summer. The agent must balance solar generation, battery storage, diesel backup, and grid trade to keep the lights on while minimizing cost and emissions.

## Why This Matters

Community microgrid operation is a real job in India under the RDSS (Revamped Distribution Sector Scheme). IEX prosumer bidding is live. The tension between local energy independence and national grid economics creates genuine multi-day planning challenges that simple heuristics cannot solve.

## Action Space (3 continuous dimensions)

| Action | Range | Description |
|--------|-------|-------------|
| `battery_dispatch` | -1.0 to +1.0 | Charge (-100 kW) or discharge (+100 kW) the community battery |
| `diesel_dispatch` | 0.0 to 1.0 | Diesel generator output (0-100 kW). Rs 100 startup cost. |
| `demand_shedding` | 0.0 to 1.0 | Request residents reduce usage (0-20%). 50% rebounds next hour. |

**The grid is NOT an action** — it automatically absorbs the residual (capped at ±200 kW). If demand exceeds all supply sources, that's a blackout.

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `hour` | float | Current hour in episode (0-72) |
| `demand_kw` | float | Current aggregate demand (kW) |
| `solar_kw` | float | Current solar generation (kW) |
| `battery_soc` | float | Battery state-of-charge (0-1) |
| `grid_price` | float | Current IEX price (Rs/kWh) |
| `diesel_fuel_remaining` | float | Diesel fuel level (0-1) |
| `diesel_is_on` | bool | Whether diesel was running last step |
| `demand_forecast_4h` | list[4] | Noisy demand forecast (±15%) |
| `solar_forecast_4h` | list[4] | Noisy solar forecast |
| `price_forecast_4h` | list[4] | Noisy price forecast |
| `cumulative_blackout_kwh` | float | Total unmet demand |
| `cumulative_cost` | float | Net cost so far (Rs) |
| `day_of_episode` | int | Current day (1-3) |

## Episode Structure

- **Duration**: 3 days (72 hours, 1-hour steps)
- **Why 3 days**: Day 1 = learn the cycle. Day 2 = heatwave hits. Day 3 = multi-day planning tested.
- **Determinism**: Seeded RNG → identical episodes per seed.

## Tasks (Easy → Medium → Hard)

| Task | Conditions | Challenge |
|------|-----------|-----------|
| **Task 1: Normal Summer** | Clear skies, ~100 kW avg demand, Rs 3-12 prices | Battery arbitrage + evening peak management |
| **Task 2: Heatwave + Clouds** | Day 2-3 heatwave (+30% demand), intermittent clouds, price spikes Rs 18 | Multi-day battery planning under uncertainty |
| **Task 3: Extreme Crisis** | Full 3-day heatwave, -30% solar, +50% demand, Rs 8-20, limited diesel | Survival mode — all resources constrained |

## Grading (0.0 - 1.0)

```
score = 0.50 × cost_efficiency + 0.25 × reliability + 0.25 × green_score
```

- **Cost efficiency**: How much cheaper than a dumb "max grid import, no intelligence" baseline
- **Reliability**: Fraction of demand met (blackouts penalized via Rs 150/kWh VoLL)
- **Green score**: 1 - (diesel_used / total_demand)

## Baseline Scores

| Strategy | Task 1 | Task 2 | Task 3 |
|----------|--------|--------|--------|
| **Oracle** | 0.81 | 0.82 | 0.77 |
| Do-Nothing | 0.58 | 0.51 | 0.46 |
| Always-Discharge | 0.58 | 0.51 | 0.47 |
| Always-Diesel | 0.42 | 0.42 | 0.45 |

## Key Physics

- **Battery**: 500 kWh, 100 kW max, 90% round-trip efficiency, **Rs 2.5/kWh degradation cost**
- **Diesel**: 100 kW, Rs 25/kWh, **Rs 100 startup cost** (penalizes on-off cycling)
- **Demand shedding**: Up to 20%, but **50% rebounds next hour** (no free lunch)
- **VoLL**: Rs 150/kWh blackout penalty (smooth gradient, no hard gate)

## Setup

```bash
# Install
pip install -e .

# Run server
uvicorn gridops.server.app:app --port 8000

# Dashboard
open http://localhost:8000/dashboard/

# Oracle test
python scripts/oracle_test.py

# Inference
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your-token"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
python inference.py
```

## Docker

```bash
docker build -t gridops .
docker run -p 8000:8000 gridops
```

## Project Structure

```
gridops/
├── inference.py                # LLM baseline (uses OpenAI client)
├── openenv.yaml                # OpenEnv manifest
├── Dockerfile
├── server/app.py               # Root entry point (OpenEnv validate)
├── gridops/
│   ├── models.py               # GridOpsAction, GridOpsObservation (Pydantic)
│   ├── simulation/
│   │   ├── physics.py          # Energy balance, battery, VoLL, degradation
│   │   └── scenarios.py        # Demand/solar/price curve generators
│   ├── tasks/
│   │   ├── definitions.py      # 3 task configs
│   │   └── graders.py          # 0-1 grading with cost/reliability/green
│   └── server/
│       ├── app.py              # FastAPI + OpenEnv create_app
│       ├── environment.py      # OpenEnv Environment class
│       └── static/index.html   # Interactive dashboard
└── scripts/
    └── oracle_test.py          # Oracle validation + determinism check
```
