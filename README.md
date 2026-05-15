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

> A production-grade OpenEnv RL environment for Indian community microgrid operation. Plug-and-play. Deterministic. Benchmarkable.

**Live demo**: [77ethers-gridops.hf.space/dashboard/](https://77ethers-gridops.hf.space/dashboard/) | **HF Space**: [huggingface.co/spaces/77ethers/gridops](https://huggingface.co/spaces/77ethers/gridops)

---

## At a Glance

| | |
|---|---|
| **Domain** | Real-world Indian community microgrid operation (100 homes, summer) |
| **Interface** | Full OpenEnv spec: `reset()` -> `step(action)` -> `state()`, typed Pydantic models |
| **Actions** | 3D continuous: `battery_dispatch [-1,1]`, `diesel_dispatch [0,1]`, `demand_shedding [0,1]` |
| **Observations** | 30+ fields: demand, solar, price, SOC, forecasts, energy flows. Partial observability (noisy forecasts). |
| **Tasks** | 3 tasks (easy -> medium -> hard), each testing a different RL capability |
| **Grading** | Deterministic, programmatic, 0.0-1.0. Same seed = same score, every run. |
| **Reward** | Dense per-step signal, aligned with episode grader (50% cost + 25% reliability + 25% green) |
| **Anti-gaming** | 5 mechanisms: degradation, startup costs, rebound, smooth VoLL, grid cap |
| **Baseline** | Grok-4 LLM: 0.80/0.82/0.72 — beats hand-coded oracle on all tasks |
| **Deployment** | Docker + HF Space + `openenv validate` 6/6 pass |

---

## SFT Training Pipeline Upgrade

This branch adds a CarbonAlpha-style training harness around the original GridOps environment without changing the public OpenEnv API.

| Artifact | Link |
|---|---|
| Shared prompt/action contract | [`gridops/prompting.py`](gridops/prompting.py) |
| Reusable oracle + adversarial policies | [`gridops/policies.py`](gridops/policies.py) |
| 1,200-row curriculum dataset | [`sft_traces/gridops_curriculum_1200.jsonl`](sft_traces/gridops_curriculum_1200.jsonl) |
| Trace generator | [`scripts/generate_sft_traces.py`](scripts/generate_sft_traces.py) |
| OpenRouter/DeepSeek trace generator | [`scripts/generate_openrouter_deepseek_traces.py`](scripts/generate_openrouter_deepseek_traces.py) |
| Trace validator | [`scripts/validate_traces.py`](scripts/validate_traces.py) |
| Holdout/adversarial evaluator | [`scripts/evaluate_gridops_model.py`](scripts/evaluate_gridops_model.py) |
| Local adapter evaluator | [`scripts/evaluate_gridops_adapter.py`](scripts/evaluate_gridops_adapter.py) |
| Guarded SFT script | [`scripts/hf_sft_gridops.py`](scripts/hf_sft_gridops.py) |
| Eval plotter | [`scripts/plot_gridops_evals.py`](scripts/plot_gridops_evals.py) |
| Colab-ready notebook | [`notebooks/gridops_sft_pipeline.ipynb`](notebooks/gridops_sft_pipeline.ipynb) |
| Model card | [`GRIDOPS_MODEL_CARD.md`](GRIDOPS_MODEL_CARD.md) |

The first milestone is **SFT only**: teach a compact model to emit valid JSON actions for each hourly observation. The first adapter passed the SFT gate on held-out seeds `7001,7002,7003`.

| Model | Avg score | Valid JSON | Task 1 | Task 2 | Task 3 |
|---|---:|---:|---:|---:|---:|
| Do-nothing | 0.5133 | 100.00% | 0.5820 | 0.5057 | 0.4522 |
| GridOps SFT v1 | 0.6854 | 99.85% | 0.6615 | 0.7300 | 0.6648 |
| Oracle | 0.7688 | 100.00% | 0.7932 | 0.8087 | 0.7046 |

| Gate | Target |
|---|---:|
| Valid JSON action rate | >= 98% |
| Average holdout score | >= 0.65 |
| No task below do-nothing baseline | required |
| Task 3 crisis score | >= 0.55 |
| Fixed-seed determinism | stable |

Final SFT v1 artifact:

```text
Qwen/Qwen2.5-3B-Instruct -> QLoRA SFT adapter:
77ethers/gridops-models/sft_qwen25_3b_gridops_mixed1418_v1
```

Evidence:

- [SFT training curve](evals/plots/gridops_sft_training_curve.png)
- [Holdout scores](evals/plots/gridops_holdout_scores.png)
- [Battery throughput](evals/plots/gridops_battery_throughput.png)
- [Blackout reduction](evals/plots/gridops_blackout_kwh.png)
- [Holdout summary JSON](evals/plots/gridops_holdout_summary.json)

The existing leaderboard remains historical. The table above is reported separately as **GridOps SFT v1**.

---

## Why This Environment Exists

Community microgrid operation is a **real job** in India under the [RDSS](https://rdss.gov.in/) (Revamped Distribution Sector Scheme). IEX prosumer bidding is live. Over 50 million Indian homes will have rooftop solar by 2030, and someone — or some agent — needs to manage the battery-grid-diesel tradeoff in real time.

This is not a toy problem. This is what a microgrid operator at an Indian housing society actually decides every hour:

- **Should I charge the battery now** (grid is cheap at Rs 4/kWh) **or save capacity for tonight** (price will spike to Rs 15)?
- **Should I run diesel** (Rs 25/kWh + Rs 100 startup) **or risk a blackout** (Rs 150/kWh VoLL penalty)?
- **Should I ask residents to reduce AC usage** (Rs 40/kWh + 100% rebounds tomorrow)?

Simple heuristics provably fail. The environment requires multi-hour planning, price forecasting, and constraint management under partial observability.

### What makes this a strong benchmark

- **Any agent can plug in immediately** — typed JSON actions in, typed observations out, no custom hacks
- **Fully deterministic** — same seed, same actions = identical trajectory every time. Leaderboard-ready.
- **Tasks differentiate agents** — Do-Nothing scores 0.45-0.58, Oracle 0.70-0.81, Grok-4 LLM 0.72-0.82. Clear skill gradient.
- **Can't be gamed** — 5 anti-exploit mechanisms prevent reward hacking (detailed below)
- **Grader = ground truth** — programmatic, deterministic, partial credit, aligned with per-step reward

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
| **Grok-4 (LLM)** | **0.80** | **0.82** | **0.72** | Reads observations, reasons about tradeoffs |
| **Oracle (rule-based)** | 0.79 | 0.81 | 0.70 | Time-of-day + price + SOC heuristic |
| Do-Nothing (grid only) | 0.58 | 0.51 | 0.45 | Grid covers everything it can |
| Always-Discharge | 0.59 | 0.51 | 0.45 | Drains battery, empty by evening |
| Always-Diesel | 0.42 | 0.42 | 0.44 | Rs 25/kWh burns money |

- **LLM beats oracle**: Grok-4 matched or exceeded the hand-coded oracle on every task
- **Deterministic**: identical scores across 3 runs (seeded RNG)
- **Oracle ceiling < 1.0**: real physics constraints, not inflated scores
- **Clear separation**: LLM > oracle >> heuristics (0.20-0.38 gap from best to worst)
- **Task 3 hardest**: grid outage makes it genuinely challenging even for frontier LLMs

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

# Generate and validate the SFT curriculum
python scripts/generate_sft_traces.py
python scripts/validate_traces.py sft_traces/gridops_curriculum_1200.jsonl

# Optional: generate 10-at-a-time teacher traces with DeepSeek on OpenRouter
export API_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_API_KEY="your-token"
python scripts/generate_openrouter_deepseek_traces.py --model deepseek/deepseek-v4-pro

# Evaluate reusable policies on holdout seeds
python scripts/evaluate_gridops_model.py --policy oracle
python scripts/evaluate_gridops_model.py --policy do_nothing

# Evaluate an API-hosted or HF-router model with the SFT prompt contract
export HF_API_TOKEN="your-token"
export MODEL_NAME="your-gridops-sft-endpoint-or-model"
python scripts/evaluate_gridops_model.py --model-name "$MODEL_NAME"

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
