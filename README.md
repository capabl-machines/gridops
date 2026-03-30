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

> Keep the lights on for 100 homes. Don't go broke. Don't pollute.

An OpenEnv RL environment where an AI agent operates a **community microgrid in an Indian city during summer**. Every hour for 3 days, the agent decides how to use the battery, whether to run diesel, and whether to ask residents to cut usage — while the national grid automatically covers whatever is left over (up to its 200 kW limit).

**Live dashboard**: [77ethers-gridops.hf.space/dashboard/](https://77ethers-gridops.hf.space/dashboard/)

---

## Why This Environment Exists

Community microgrid operation is a **real job** in India under the [RDSS](https://rdss.gov.in/) (Revamped Distribution Sector Scheme). IEX prosumer bidding is live. Over 50 million Indian homes will have rooftop solar by 2030, and someone needs to manage the battery-grid-diesel tradeoff in real time.

This environment captures the core tension: **sell surplus solar during expensive evening peaks for profit — but if a heatwave continues tomorrow and your battery is empty, the neighborhood goes dark.**

Simple heuristics ("always discharge at peak", "always run diesel when low") provably fail. The agent must learn multi-hour planning, price forecasting, and constraint management.

---

## The Problem at a Glance

**You have:**
- **Solar panels** — 250 kW peak, free, but only during daylight
- **Community battery** — 500 kWh storage, 100 kW max charge/discharge
- **Diesel generator** — 100 kW, but Rs 25/kWh + Rs 100 startup cost
- **National grid** — auto-imports/exports as slack (capped at 200 kW)

**You control (3 continuous actions):**

| Action | Range | What it does |
|--------|-------|-------------|
| `battery_dispatch` | -1 to +1 | Charge (-100 kW) or discharge (+100 kW). Rs 2.5/kWh degradation. |
| `diesel_dispatch` | 0 to 1 | Diesel output (0-100 kW). Rs 25/kWh + Rs 100 startup if was off. |
| `demand_shedding` | 0 to 1 | Ask residents to cut 0-20% usage. **100% rebounds next hour.** Rs 40/kWh penalty. |

**You do NOT control the grid.** It automatically absorbs whatever energy gap remains after your decisions. If the gap exceeds 200 kW, that's a **blackout** (Rs 150/kWh penalty).

---

## The Critical Bottleneck

At **8 PM every evening**, demand hits **250 kW** but the grid maxes out at **200 kW** and solar is zero.

The **50 kW gap** must come from your battery. If you discharged it for profit during the day, the neighborhood goes dark.

On a heatwave day (Task 2-3), demand spikes to **325-375 kW**. Now the gap is **125-175 kW** — you need battery + diesel + shedding just to survive. And in Task 3, the grid goes down entirely for 6 hours.

---

## What the Agent Sees (Observation)

| Field | Description |
|-------|-------------|
| `hour` | Current hour in episode (0-72, starting 6 AM) |
| `demand_kw` | What the 100 homes need right now |
| `solar_kw` | Free solar power available (0 at night, up to 250 kW midday) |
| `battery_soc` | Battery charge level (0-1, i.e. 0-500 kWh) |
| `grid_price` | Current IEX electricity price (Rs 3-20/kWh) |
| `diesel_fuel_remaining` | Diesel tank level (0-1) |
| `diesel_is_on` | Was diesel running last step? (startup cost if turning on) |
| `demand_forecast_4h` | Noisy 4-hour demand forecast (+-15%) |
| `solar_forecast_4h` | Noisy 4-hour solar forecast |
| `price_forecast_4h` | Noisy 4-hour price forecast |
| `cumulative_blackout_kwh` | Total blackout energy so far |
| `cumulative_cost` | Total money spent so far (Rs) |
| `flow_*` | Detailed energy flows (solar, grid import/export, battery in/out, diesel, demand) |

**Partial observability**: forecasts have +-15% Gaussian noise. The agent cannot perfectly predict heatwave intensity, cloud cover, or price spikes.

---

## 3 Tasks (Each Tests a Different RL Capability)

### Task 1: Normal Summer (Easy) — *Tests basic arbitrage*
- Clear skies, standard demand (~100 kW avg, 250 kW peak)
- Grid prices Rs 3-12 with clear cheap night / expensive evening pattern
- **What the agent must learn**: charge battery at night (cheap grid), discharge during evening peak (expensive grid), let solar cover midday

### Task 2: Heatwave + Price Spike (Medium) — *Tests temporal planning*
- Day 2-3 heatwave (+30% demand), intermittent clouds
- **Rs 20 price spike** on Day 2 evening — visible in 4-hour forecast
- **What the agent must learn**: read the forecast, hold battery charge for the spike instead of greedily discharging early. A greedy policy discharges mid-afternoon; an RL agent that reads the forecast holds until 6 PM.

### Task 3: Extreme Crisis + Grid Outage (Hard) — *Tests constraint management*
- Full 3-day heatwave, -30% solar from haze, +50% demand
- Limited diesel (33% tank = ~8 hours at full power)
- **6-hour grid outage** on Day 2 afternoon — grid cap drops to 0 kW
- **What the agent must learn**: aggressively pre-charge battery before the outage, ration diesel across the outage window, shed demand strategically to stretch resources. This is true microgrid islanding.

---

## Grading (0.0 - 1.0)

```
score = 0.50 x cost_efficiency + 0.25 x reliability + 0.25 x green_score
```

| Component | Formula | What it rewards |
|-----------|---------|----------------|
| **Cost efficiency** (50%) | `1 - (agent_cost / baseline_cost)` | Spending less than a dumb "max grid import" baseline |
| **Reliability** (25%) | `(demand_met - blackout) / demand_met` | Keeping the lights on |
| **Green score** (25%) | `1 - (diesel_used / total_demand)` | Minimizing diesel emissions |

**Baseline**: "import max grid every hour, no battery/diesel/shedding" — physically possible, but expensive and suffers blackouts during peak hours and grid outages.

**VoLL (Value of Lost Load)**: Rs 150/kWh blackout penalty. This is a smooth gradient — no hard reliability cliff. The agent always gets signal for reducing blackouts incrementally.

---

## Why Heuristics Fail

| Strategy | Why it fails |
|----------|-------------|
| "Always discharge battery" | Empty by evening peak. 50 kW gap = blackout. Score collapses. |
| "Always run diesel" | Rs 25/kWh vs Rs 5 grid at night. Hemorrhages money. Green score = 0. |
| "Shed demand whenever short" | Rs 40/kWh cost + 100% rebounds next hour. More expensive than diesel. |
| "Discharge when price > X" | Ignores battery state. Drains SOC before the real peak. |
| "Do nothing" | Grid alone can't cover evening peak. 3.6% blackout rate. |

The oracle (rule-based, time-of-day + price-aware) scores 0.70-0.81. There's a clear **0.20-0.35 gap** between heuristics and the oracle, proving the environment has real optimization headroom.

---

## Anti-Gaming Design

The environment has 5 mechanisms that prevent reward hacking:

1. **Shedding is expensive** — Rs 40/kWh + 100% rebound. Costlier than diesel. True emergency only.
2. **Battery degradation** — Rs 2.5/kWh throughput. Prevents infinite cycling for tiny arbitrage.
3. **Diesel startup cost** — Rs 100 per on-switch. Prevents on/off toggling.
4. **VoLL is smooth** — Rs 150/kWh with no cliff. Agent can't exploit a binary gate.
5. **Grid is capped** — 200 kW max (0 during outages). Can't just buy everything.

---

## Baseline Scores

| Strategy | Task 1 | Task 2 | Task 3 | What it does |
|----------|--------|--------|--------|-------------|
| **Oracle (rule-based)** | **0.79** | **0.81** | **0.70** | Time-of-day + price + SOC aware |
| Do-Nothing (grid only) | 0.58 | 0.51 | 0.45 | Grid covers everything it can |
| Always-Discharge | 0.59 | 0.51 | 0.45 | Drains battery, empty by evening |
| Always-Diesel | 0.42 | 0.42 | 0.44 | Rs 25/kWh burns money |

- **Deterministic**: identical scores across 3 runs (seeded RNG)
- **Oracle ceiling < 1.0**: real physics constraints, not inflated scores
- **Clear separation**: oracle >> heuristics on every task (0.20-0.35 gap)
- **Task 3 hardest**: grid outage drops oracle from 0.81 to 0.70

---

## Key Physics

| Component | Spec | Cost |
|-----------|------|------|
| **Solar** | 250 kW peak, bell curve 6 AM - 6 PM | Free |
| **Battery** | 500 kWh, 100 kW max, 90% round-trip (sqrt each way) | Rs 2.5/kWh degradation |
| **Diesel** | 100 kW max | Rs 25/kWh + Rs 100 startup |
| **Grid** | 200 kW max import/export (slack variable) | Market price Rs 3-20/kWh |
| **Blackout** | Unmet demand when all sources exhausted | Rs 150/kWh VoLL penalty |
| **Shedding** | Up to 20% demand reduction | Rs 40/kWh + 100% rebound next hour |

**Energy balance every step:**
```
supply  = solar + grid_import + battery_discharge + diesel
consume = effective_demand + grid_export + battery_charge
```
Supply always equals consumption. Any unmet demand beyond grid cap = blackout.

---

## Setup & Usage

```bash
# Install
pip install -e .

# Run server
uvicorn gridops.server.app:app --port 8000

# Interactive dashboard
open http://localhost:8000/dashboard/

# Validate oracle + determinism
python scripts/oracle_test.py

# Run LLM baseline
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

## OpenEnv Validation

```bash
# Local structure check
openenv validate

# Runtime check (against live server)
openenv validate --url http://localhost:8000
```

---

## Project Structure

```
gridops/
├── inference.py                 # LLM baseline (API_BASE_URL, MODEL_NAME, HF_TOKEN)
├── openenv.yaml                 # OpenEnv manifest
├── Dockerfile                   # Docker deployment
├── server/app.py                # Root entry point (openenv validate)
├── gridops/
│   ├── models.py                # GridOpsAction, GridOpsObservation (Pydantic)
│   ├── simulation/
│   │   ├── physics.py           # Energy balance, battery, VoLL, degradation, outages
│   │   └── scenarios.py         # Demand/solar/price curve generators
│   ├── tasks/
│   │   ├── definitions.py       # 3 task configs (normal, heatwave, crisis+outage)
│   │   └── graders.py           # 0-1 scoring: cost + reliability + green
│   └── server/
│       ├── app.py               # FastAPI + OpenEnv create_app
│       ├── environment.py       # OpenEnv Environment class
│       └── static/index.html    # Interactive dashboard with energy flows
└── scripts/
    └── oracle_test.py           # Oracle + heuristic validation + determinism check
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/schema` | GET | Action/observation/state JSON schemas |
| `/metadata` | GET | Environment name and description |
| `/reset` | POST | Reset environment (OpenEnv standard) |
| `/step` | POST | Execute action (OpenEnv standard) |
| `/state` | GET | Current state (OpenEnv standard) |
| `/ws` | WebSocket | Persistent session (OpenEnv standard) |
| `/api/reset` | POST | Stateful reset (dashboard) |
| `/api/step` | POST | Stateful step (dashboard) |
| `/api/state` | GET | Stateful state (dashboard) |
| `/tasks` | GET | List available tasks |
| `/dashboard/` | GET | Interactive web UI |
