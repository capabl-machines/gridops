"""
Inference Script — GridOps Microgrid Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import json
import os
import sys

from openai import OpenAI

# ── Env vars (as required by hackathon) ──────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")

# ── Environment import (runs in-process, no server needed) ──────────────
sys.path.insert(0, os.path.dirname(__file__))
from gridops.server.environment import GridOpsEnvironment
from gridops.models import GridOpsAction

TASKS = ["task_1_normal", "task_2_heatwave", "task_3_crisis"]
MAX_STEPS = 72
TEMPERATURE = 0.1
MAX_TOKENS = 150

SYSTEM_PROMPT = """\
You are an expert microgrid operator managing a 100-home community in India during summer.

You control three actions each hour:
- battery_dispatch: -1 (charge 100 kW from grid) to +1 (discharge 100 kW to community)
- diesel_dispatch: 0 (off) to 1 (100 kW). Costs Rs 25/kWh + Rs 100 startup if was off.
- demand_shedding: 0 (none) to 1 (shed 20% of demand). WARNING: 50% rebounds next hour.

The GRID automatically absorbs the residual (capped at ±200 kW).
If demand exceeds grid + solar + battery + diesel → BLACKOUT (Rs 150/kWh penalty!).

Key economics:
- Grid prices vary Rs 3-20/kWh. Cheap at night, expensive evening.
- Battery: 500 kWh, 100 kW max, 90% round-trip efficiency, Rs 2.5/kWh degradation.
- Solar: 250 kW peak (free!), bell curve 6AM-6PM, zero at night.
- Demand: ~100 kW avg, 250 kW evening peak. Grid cap = 200 kW → need battery for gap.

Strategy:
1. Night (0-6h): charge battery (cheap grid, low demand)
2. Solar (6-15h): surplus charges battery or exports
3. Pre-peak (15-17h): ensure battery > 70%
4. Evening peak (18-22h): discharge battery to cover gap above grid 200 kW cap
5. Diesel: only when battery empty AND peak demand. Avoid startup costs.

Respond ONLY with valid JSON: {"battery_dispatch": float, "diesel_dispatch": float, "demand_shedding": float}"""


def format_observation(obs: dict) -> str:
    """Format observation into a readable prompt for the LLM."""
    return (
        f"Hour {obs['hour']:.0f}/72 (Day {obs.get('day_of_episode', '?')})\n"
        f"Demand: {obs['demand_kw']:.0f} kW | Solar: {obs['solar_kw']:.0f} kW\n"
        f"Battery SOC: {obs['battery_soc']*100:.0f}% | Grid Price: Rs {obs['grid_price']:.1f}/kWh\n"
        f"Diesel Fuel: {obs['diesel_fuel_remaining']*100:.0f}% | Diesel On: {obs.get('diesel_is_on', False)}\n"
        f"Grid import last step: {obs.get('grid_kw_this_step', 0):.0f} kW\n"
        f"Forecasts (next 4h):\n"
        f"  Demand: {[f'{v:.0f}' for v in obs.get('demand_forecast_4h', [])]}\n"
        f"  Solar:  {[f'{v:.0f}' for v in obs.get('solar_forecast_4h', [])]}\n"
        f"  Price:  {[f'{v:.1f}' for v in obs.get('price_forecast_4h', [])]}\n"
        f"Cumulative: blackout={obs['cumulative_blackout_kwh']:.1f} kWh, cost=Rs {obs['cumulative_cost']:.0f}\n"
        f"{obs.get('narration', '')}\n"
        f"\nWhat action? Reply with JSON only."
    )


def parse_action(text: str) -> dict:
    """Extract action JSON from LLM response."""
    text = text.strip()
    for start, end in [("{", "}"), ("```json", "```")]:
        idx = text.find(start)
        if idx >= 0:
            if end == "}":
                eidx = text.rfind("}") + 1
            else:
                eidx = text.find(end, idx + len(start))
            try:
                return json.loads(text[idx:eidx])
            except json.JSONDecodeError:
                continue
    return {"battery_dispatch": 0.0, "diesel_dispatch": 0.0, "demand_shedding": 0.0}


def run_task(client: OpenAI, env: GridOpsEnvironment, task_id: str, seed: int = 42) -> dict:
    """Run one full episode on a task, return grade."""
    obs = env.reset(seed=seed, task_id=task_id)
    obs_dict = obs.model_dump()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step_idx in range(MAX_STEPS):
        user_msg = format_observation(obs_dict)
        messages.append({"role": "user", "content": user_msg})

        # Keep context manageable
        if len(messages) > 21:
            messages = [messages[0]] + messages[-20:]

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            reply = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"    LLM error at step {step_idx}: {e}")
            reply = "{}"

        messages.append({"role": "assistant", "content": reply})

        action_dict = parse_action(reply)
        action = GridOpsAction(
            battery_dispatch=float(action_dict.get("battery_dispatch", 0.0)),
            diesel_dispatch=float(action_dict.get("diesel_dispatch", 0.0)),
            demand_shedding=float(action_dict.get("demand_shedding", 0.0)),
        )
        obs = env.step(action)
        obs_dict = obs.model_dump()

        if step_idx % 24 == 0:
            print(f"    Hour {obs_dict['hour']:.0f}: SOC={obs_dict['battery_soc']*100:.0f}% "
                  f"cost=Rs {obs_dict['cumulative_cost']:.0f} "
                  f"blackout={obs_dict['cumulative_blackout_kwh']:.1f}")

        if obs_dict.get("done", False):
            break

    grade = env.state.grade
    return grade


def main():
    print("=" * 60)
    print("  GridOps — LLM Baseline Inference")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API:   {API_BASE_URL}")
    print("=" * 60)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = GridOpsEnvironment()

    results = {}
    for task_id in TASKS:
        print(f"\n--- {task_id} ---")
        grade = run_task(client, env, task_id)
        results[task_id] = grade
        if grade:
            print(f"  Score:       {grade['score']}")
            print(f"  Reliability: {grade['reliability']}")
            print(f"  Cost Eff:    {grade['cost_efficiency']}")
            print(f"  Green:       {grade['green_score']}")
            print(f"  Cost:        Rs {grade['actual_cost']:.0f} (baseline Rs {grade['baseline_cost']:.0f})")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for task_id, grade in results.items():
        score = grade["score"] if grade else "ERROR"
        print(f"  {task_id}: {score}")
    print("=" * 60)


if __name__ == "__main__":
    main()
