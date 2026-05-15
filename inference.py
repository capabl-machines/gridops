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
from gridops.prompting import SYSTEM_PROMPT, format_observation, parse_action

TASKS = ["task_1_normal", "task_2_heatwave", "task_3_crisis"]
MAX_STEPS = 72
TEMPERATURE = 0.1
MAX_TOKENS = 500  # higher to support reasoning models that emit thinking before JSON


def run_task(client: OpenAI, env: GridOpsEnvironment, task_id: str, seed: int = 42) -> dict:
    """Run one full episode on a task, return grade."""
    obs = env.reset(seed=seed, task_id=task_id)
    obs_dict = obs.model_dump()

    # ── [START] structured output ──
    print(f"[START] task={task_id}", flush=True)

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
            reply = "{}"

        messages.append({"role": "assistant", "content": reply})

        action = parse_action(reply, default=GridOpsAction())
        obs = env.step(action)
        obs_dict = obs.model_dump()

        # ── [STEP] structured output ──
        reward = obs_dict.get("reward", 0.0)
        print(f"[STEP] step={step_idx + 1} reward={reward:.4f}", flush=True)

        if obs_dict.get("done", False):
            break

    grade = env.state.grade
    score = grade["score"] if grade else 0.0
    steps = step_idx + 1

    # ── [END] structured output ──
    print(f"[END] task={task_id} score={score:.4f} steps={steps}", flush=True)

    return grade


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = GridOpsEnvironment()

    results = {}
    for task_id in TASKS:
        grade = run_task(client, env, task_id)
        results[task_id] = grade

    # Summary
    for task_id, grade in results.items():
        score = grade["score"] if grade else 0.0
        print(f"[SUMMARY] task={task_id} score={score:.4f}", flush=True)


if __name__ == "__main__":
    main()
