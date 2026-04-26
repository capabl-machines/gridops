# CarbonAlpha Engineering Process

This note records how we built CarbonAlpha end to end: first as an
OpenEnv-compliant climate portfolio environment, then as a model-training
pipeline for reasoning under constraints. It is intentionally written as an
engineering record rather than a polished paper: what we decided, why we
decided it, what broke, what survived contact with real training runs, and
what we would do differently next time.

## 1. Project Thesis

CarbonAlpha is a single-agent OpenEnv environment where an LLM acts as a
climate-aware portfolio manager. The agent sees one macro-news event and must
commit one portfolio allocation that is held through a 12-quarter, three-year
macro cycle.

The core challenge is not raw market forecasting. The challenge is constrained
reasoning:

- preserve real returns against an equal-weight baseline;
- respect a hard carbon budget;
- reason through 1st-, 2nd-, and 3rd-order macro effects;
- avoid overusing optional interventions such as hedges and carbon offsets;
- produce a strict `<think>...</think>` plus JSON `PortfolioAction`.

The project evolved from a broad “agent arena” brainstorm into a focused
Reasoning-Under-Constraints OpenEnv. We deliberately chose a flattened
single-turn decision format for training: the model makes one action, then the
environment rolls that action through the 12-quarter path. This avoided
immature multi-turn GRPO complexity while still preserving path-dependent
state inside the environment.

## 2. Environment Creation Process

### 2.1 OpenEnv Contract First

The environment implementation lives in `portfolio_env/`. The first
engineering priority was to satisfy the OpenEnv contract cleanly:

- `PortfolioEnv.reset(...)`
- `PortfolioEnv.step(...)`
- `PortfolioEnv.state`
- `PortfolioEnv.get_metadata()`
- Pydantic action / observation / state models
- FastAPI/OpenEnv server compatibility

The key files are:

- `portfolio_env/env.py` — path-dependent environment dynamics
- `portfolio_env/models.py` — `PortfolioAction`, `PortfolioObs`, `PortfolioState`
- `portfolio_env/constants.py` — assets, caps, return assumptions, intervention limits
- `portfolio_env/shocks.py` — easy / ambiguous / hard macro shock pool
- `portfolio_env/rewards.py` — reward components and parsing helpers
- `portfolio_env/prompt.py` — single source of truth for the model prompt

We treated the schema as the boundary between the model, the environment, and
the demo UI. This mattered later: the same `PortfolioAction` contract could be
used for Gemini trace generation, SFT training, holdout simulation, and the
Hugging Face Space walkthrough.

### 2.2 Flattened MDP, Path-Dependent Simulator

The final environment shape is:

1. `reset()` samples a 12-quarter episode plan.
2. The model receives one macro-news item and returns one allocation.
3. `step(action)` advances one quarter at a time using the same locked action.
4. The environment tracks NAV, inflation, carbon, baseline NAV, shock regime,
   hedge state, infra lockups, offsets, and final reward components.

The model does not choose a fresh action each quarter. That is intentional. It
keeps training tractable and forces the model to make a macro-cycle allocation
rather than repeatedly reacting with hindsight.

The environment itself is still path-dependent:

- Transaction costs apply when allocation changes from the baseline.
- Carbon accumulates every quarter from exposure and NAV.
- `infra_commit` locks capital for four quarters and pays only if transition
  shocks arrive during the lockup.
- Physical-risk shocks penalize infrastructure lockups.
- Put hedges bleed premium and only help on severe portfolio-level downside.
- Inflation regimes compound into real NAV.
- Shocks can switch the future inflation regime.

This gave us a simple training interface with a non-trivial simulator behind
it.

### 2.3 Shock Design

The shock pool is organized into curriculum tiers:

- `easy` — direct 1st-order asset moves;
- `ambiguous` — conflicting signals where naive interpretation can lose;
- `hard` — 2nd/3rd-order effects dominate.

Examples:

- stagflation: bonds and long-duration assets suffer despite their usual
  “safe” role;
- rare-earth export controls: green supply chains can suffer even though the
  headline sounds climate-related;
- carbon offset fraud: offsets fall, but real abatement assets can rerate
  upward.

This tiering served two purposes:

1. Environment curriculum: phases 1/2/3 expose easy, ambiguous, then hard
   shocks.
2. Data curriculum: Gemini trace generation can request easy / medium / hard
   examples in controlled proportions.

### 2.4 Reward Stack

The reward design landed on five components:

- `r_format` — validates `<think>...</think>` and JSON shape.
- `r_regret` — primary objective: final real return minus equal-weighted
  baseline real return.
- `r_sharpe` — secondary risk-adjusted return signal.
- `r_carbon` — quadratic penalty above the 25 kg carbon cap, phase-weighted.
- `r_drawdown` — max drawdown penalty.

The reward stack was kept modular because GRPO trainers accept reward
functions as separate callables, and because per-component logging is essential
for debugging reward hacking.

The baseline is not a dummy. Equal-weighted allocation is intentionally strong:
it gives a robust comparison point and prevents us from declaring victory just
because the model outputs valid JSON.

### 2.5 Adversarial Reward Testing

Before trusting the environment, we attacked the rewards with hand-written
policies. This caught real design bugs:

- `all_oil` exposed that the original carbon cap was too loose.
- Infrastructure had a double-count / no-downside issue.
- Put hedge logic could be farmed if it triggered on a single asset instead of
  portfolio-level downside.
- Infra needed a physical-risk counter-penalty.

The v0.7 environment patches came directly from these tests:

- carbon cap tightened to 25 kg;
- infra payoff became return-only instead of double-counting principal;
- infra loses value under physical-risk shocks;
- put hedge triggers on portfolio drawdown, not an individual asset move.

This was the most important environment engineering loop: break the game
ourselves before asking an RL algorithm to optimize it.

### 2.6 Prompt as an Environment Artifact

`portfolio_env/prompt.py` became a central file rather than a training-script
detail. This was a key process decision.

The same prompt is used for:

- SFT trace generation;
- SFT training;
- GRPO prompts;
- holdout generation;
- demo inference.

We made this single-source because prompt mismatch is a classic SFT/RL failure
mode. If the SFT model learns one prompt distribution and GRPO samples from a
different one, the policy can collapse before rewards have a chance to help.

### 2.7 Demo-Specific Environment Binding

The live custom demo added one important UX/environment bridge. In the normal
environment, reset samples a hidden shock plan. In the walkthrough demo, if a
user enters or selects a Q1 macro event, that visible macro event should be
the shock that resolves when they press “Advance Quarter.”

So the Space app maps the selected/custom Q1 headline to a canonical shock and
injects it into quarter 0 of the episode plan. This keeps the demo honest:
the news the user sees is the news the simulator scores.

## 3. Model Training Pipeline

### 3.1 Training Objective

The model’s job is not to predict returns numerically. Its job is to produce a
valid and useful `PortfolioAction`:

```json
{
  "weights": [w_tech, w_oil, w_green, w_real_estate, w_bonds],
  "infra_commit": 0.0,
  "carbon_offset_buy": 0.0,
  "put_hedge": 0.0,
  "tech_bet": "status_quo"
}
```

The completion contract is:

```text
<think>
macro-cycle reasoning
</think>
{JSON action}
```

We trained for two things:

1. Format control: closed thinking tags, parseable JSON, valid action ranges.
2. Allocation quality: positive regret against the equal-weighted baseline on
   held-out seeds.

### 3.2 SFT Trace Generation

The SFT data pipeline began with manually aligned traces, then moved to
Gemini-generated curriculum traces. The final generator is
`sft_traces/generate_curriculum_traces.py`.

Important design choices:

- It uses the same prompt schema as model inference.
- It generates 10 traces per API call.
- It rotates across configured Gemini API keys.
- It validates every trace before writing.
- It preserves the row schema used by `merged_v6_aligned.jsonl`.
- Easy / medium / hard map directly to environment shock tiers:
  - easy → `Shock.tier == "easy"` → phase 1
  - medium → `Shock.tier == "ambiguous"` → phase 2
  - hard → `Shock.tier == "hard"` → phase 3

Each accepted trace stores:

- `id`
- `seed_id`
- `seed_year`
- `seed_category`
- `prompt`
- `completion`
- `raw`

The `raw` object stores curriculum metadata without breaking older SFT
loaders.

The successful large curriculum file was:

```text
sft_traces/curriculum_400_e80_m160_h160.jsonl
```

with:

- 80 easy traces
- 160 medium traces
- 160 hard traces

### 3.3 SFT Lineage

The training process went through several model/data/recipe iterations.

Early runs established three lessons:

1. More diverse traces mattered.
2. Full 16-bit / non-over-aggressive LoRA settings performed better than an
   overly canonical recipe copied from larger datasets.
3. GRPO could damage a good SFT model if rollout generation was unhealthy.

The strongest safe model from the Qwen3 line was:

```text
77ethers/CarbonAlpha/v6_sft_only_v2
```

It used:

- Qwen3-4B-Instruct
- LoRA rank 16
- `lora_alpha=16`
- SFT on `merged_v6_aligned.jsonl`
- 5/5 valid holdout format
- mean holdout regret `+0.034`
- beat baseline on 3/5 holdout seeds

This became the final-safe model. We explicitly preserved it and avoided
overwriting that subfolder.

### 3.4 Qwen2.5-7B SFT Rescue

After the Qwen3/GRPO path proved unstable, we ran a cleaner SFT rescue using:

```text
unsloth/Qwen2.5-7B-Instruct
```

The entrypoint is:

```text
scripts/hf_sft_qwen25_7b.py
```

The run used:

- QLoRA SFT
- LoRA rank 16
- `lora_alpha=16`
- 220 SFT steps
- effective batch size 4
- curriculum 400 trace file
- HF Jobs on L40S

The artifact landed at:

```text
77ethers/CarbonAlpha/sft_qwen25_7b_curriculum400_v1
```

Results:

- generation sanity: 5/5 valid, closed `<think></think>`;
- holdout: 5/5 valid;
- mean holdout regret `+0.02796`;
- beats baseline on 3/5 seeds.

This did not beat the v6 SFT model numerically on mean regret, but it produced
cleaner demo behavior and had stronger instruction-following ergonomics. We
therefore used it for the live custom Space while keeping v6 as the numerical
safe baseline.

### 3.5 Hugging Face Jobs Pipeline

We moved from ad hoc RunPod execution toward HF Jobs for repeatability.

The HF Jobs scripts follow this pattern:

1. Load `.env` locally only for convenience.
2. Require `HF_API_TOKEN` for private Hugging Face access.
3. Set `HF_TOKEN=$HF_API_TOKEN` inside the job process because some libraries
   implicitly read `HF_TOKEN`.
4. Verify auth with `HfApi.whoami`.
5. Confirm access to:
   - `77ethers/CarbonAlpha`
   - `77ethers/CarbonAlpha-train`
6. Download the code bundle from the private dataset repo.
7. Train.
8. Run generation sanity checks.
9. Run holdout evaluation.
10. Upload artifact and metrics into a new subfolder of
    `77ethers/CarbonAlpha`.

Two operational rules became non-negotiable:

- never use stale `HF_TOKEN` / `HF2_TOKEN` from `.env`;
- never overwrite a known-good model subfolder.

### 3.6 Holdout Evaluation

Holdout seeds are reserved:

```text
100, 200, 300, 400, 500
```

The holdout loop:

1. Selects a shock from the phase-3 pool using the seed.
2. Prompts the model with that shock news.
3. Parses the model output into `PortfolioAction`.
4. Simulates a 12-quarter episode.
5. Computes regret versus the equal-weight baseline.

Acceptance was deliberately strict:

- valid action count matters;
- mean regret must be positive;
- beating baseline on individual seeds matters;
- demo samples must be interpretable, not just numerically lucky.

This is why v6 SFT remains the safe numerical model and Qwen2.5-7B became the
better demo model.

## 4. GRPO Attempts and Failure Analysis

### 4.1 Original GRPO Plan

The intended pipeline was:

1. SFT warm-start.
2. Phase 1 GRPO on easy shocks.
3. Phase 2 GRPO on ambiguous shocks.
4. Phase 3 GRPO on all shocks.
5. Select best checkpoint by holdout regret, not training reward.

The script `notebooks/grpo_training.py` was the original SFT + GRPO driver.

### 4.2 Qwen3 Instruct GRPO Failure

GRPO failed under the Unsloth/vLLM stack in several distinct ways:

- `matmul_lora` dtype mismatch between fp16 and bf16;
- fp16 sampling collapse on Blackwell;
- vLLM 0.19.x graph-erase compile failure;
- after pinning to vLLM 0.15.1, rollouts still collapsed to 1-token
  completions.

The worst symptom was:

```text
completions/mean_length: 1.0
loss: 0.0
grad_norm: 0.0
reward std: 0.0
```

That means no real policy learning. Worse, GRPO could degrade an SFT-warmed
model because it was optimizing from broken rollouts.

At that point, the decision was to stop treating Qwen3/vLLM GRPO as the
production path and preserve the SFT-only model.

### 4.3 Qwen3 Base GRPO Rescue Attempt

We then implemented a separate GRPO rescue script:

```text
scripts/hf_grpo_qwen3_base.py
```

This was intentionally isolated from the v6 Instruct pipeline. It followed the
official Unsloth Qwen3 4B GRPO notebook mechanics more closely:

- `unsloth/Qwen3-4B-Base`
- custom chat template
- generation prompt opens with `<think>\n`
- SFT warm-start retemplated through the Base template
- vLLM pinned to `0.15.1`
- `transformers==4.56.2`
- `trl==0.22.2`
- explicit sampling parameters

The reward wrapper treats completions as if prefixed with `<think>\n` because
the opening tag is in the prompt rather than generated by the model.

This path was valuable because it gave us a principled GRPO rescue route, but
we did not let it endanger the safe SFT artifact.

### 4.4 GRPO Smoke Gate

The GRPO smoke gate was:

- SFT warm-start first;
- 5-prompt generation sanity check;
- 5–10 GRPO steps only;
- continue only if:
  - completion mean length > 50;
  - min/max completion length are not stuck at 1;
  - gradient norm is nonzero and non-NaN;
  - at least one reward component has nonzero std;
  - at least 3/5 sampled completions parse into valid `PortfolioAction`.

If one-token collapse reappears, abort GRPO and ship SFT. This rule kept the
project from burning the working model while chasing unstable RL.

### 4.5 Qwen2.5 Adapter GRPO Rescue

After the Qwen3/vLLM path failed, we tried a more conservative RL route:
continue from the already-good Qwen2.5-7B SFT adapter and avoid vLLM entirely.

The entrypoint is:

```text
scripts/hf_grpo_qwen25_adapter.py
```

The key design choice was to use TRL GRPO with plain Transformers generation:

```text
use_vllm=False
```

This mattered because the previous failures were rollout failures. The SFT
adapter could already produce valid, long, closed-`<think>` completions; the
risk was not reasoning capacity but generation plumbing. Removing vLLM gave us
a slower but much easier-to-debug training loop.

The model stack was:

```text
base model:     unsloth/Qwen2.5-7B-Instruct
warm start:     77ethers/CarbonAlpha/sft_qwen25_7b_curriculum400_v1
new GRPO path:  separate subfolders only
```

The reward functions used in this path were intentionally narrower than the
full environment reward:

- format reward for closed `<think>...</think>` and JSON;
- action-contract reward for valid ranges and schema;
- reasoning-shape reward for useful but bounded reasoning length;
- phase-1 regret reward from the simulator;
- carbon-guard reward.

The script also binds the selected prompt shock into quarter 0 of the
simulator before reward scoring. This avoids a subtle mismatch where the model
would answer one news event while the reward environment scored a different
sampled shock.

### 4.6 TRL qLoRA Patch

The first Qwen2.5 GRPO launches loaded the model correctly and passed
pre-GRPO sanity, but failed at `GRPOTrainer` construction. The failure came
from TRL 0.22.2's qLoRA preparation path:

```text
ValueError: 'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time
```

The root cause was not our explicit config. TRL internally called
`dataclasses.replace(args, gradient_checkpointing=False)` inside
`prepare_peft_model(...)` for the qLoRA model. That re-ran
`GRPOConfig.__post_init__` after TRL had already populated generated batching
fields, causing the config to invalidate itself.

The fix was:

1. prepare the 4-bit base model ourselves with
   `prepare_model_for_kbit_training(..., use_gradient_checkpointing=False)`;
2. load the SFT LoRA adapter as a trainable `PeftModel`;
3. monkeypatch TRL's `prepare_peft_model` call to return the already-loaded
   adapter unchanged.

This is a local compatibility patch for this specific preloaded-adapter GRPO
path. It should not be generalized blindly to fresh PEFT training scripts.

### 4.7 Successful GRPO Runs

The first healthy smoke run was:

```text
77ethers/CarbonAlpha/grpo_qwen25_7b_adapter_smoke_v3
```

Smoke results:

- 8 GRPO steps;
- pre-GRPO sanity: 5/5 valid actions, 5/5 closed `<think>`;
- completion lengths stayed in the normal 100+ token range;
- `grad_norm` was nonzero;
- reward standard deviation was nonzero through the regret component;
- post-GRPO sanity: 5/5 valid actions;
- holdout mean regret: `+0.0761`;
- smoke gate passed.

After that, we launched a longer Phase 1 run:

```text
77ethers/CarbonAlpha/grpo_qwen25_7b_adapter_phase1_100_v1
```

Results:

- 100 GRPO steps;
- post-GRPO sanity: 5/5 valid actions, 5/5 closed `<think>`;
- post-GRPO mean completion length: `164` tokens;
- holdout: 5/5 valid;
- mean holdout regret: `+0.1058`;
- beat baseline on 5/5 holdout seeds;
- v6 SFT numerical bar: `+0.034`.

This is the first run where GRPO was not merely "attempted" but actually
healthy by the smoke criteria and better than the previous numerical holdout
bar. We still treat it as Phase 1 GRPO, not as a final production replacement,
because the GRPO dataset is currently easy-shock weighted and the manual eval
found two targeted reasoning weaknesses.

## 5. Manual Macro Evaluation

The environment holdout is necessary but not sufficient. A model can score
well on simulator regret and still fail a human-facing macro reasoning test.
To catch that, we created a 10-question eval set:

```text
evals/macro_eval_10.jsonl
```

The eval covers:

- benign disinflation/productivity;
- oil supply inflation shock;
- credit freeze;
- AI efficiency second-order paradox;
- carbon offset fraud;
- rare-earth export controls;
- insurance retreat / climate physical risk;
- global deflation pulse;
- crypto policy noise;
- yen carry unwind.

Each row includes:

- `id`
- `difficulty`
- `category`
- `question`
- `expected_focus`
- `red_flags`

The launcher is:

```text
scripts/launch_macro_eval.py
```

It runs `scripts/hf_compare_qwen25.py` on Hugging Face Jobs with the eval
cases passed through `CARBON_ALPHA_COMPARE_CASES_JSON`.

The first GRPO eval report is:

```text
evals/macro_eval_10_grpo_report.json
```

Summary:

- GRPO adapter: 10/10 valid JSON actions;
- GRPO adapter: 10/10 closed `<think>`;
- base model: 9/10 valid JSON actions;
- GRPO was much stronger on:
  - rare-earth export controls;
  - global deflation pulse;
  - yen carry unwind.

The eval also found two weaknesses:

- `q02_oil_chokepoint_inflation`: the model understood the inflation regime
  and hedged, but underweighted OIL despite the direct oil supply shock.
- `q04_ai_efficiency_paradox`: the model correctly liked TECH and cut
  REAL_ESTATE, but still gave GREEN too much weight despite lower data-center
  power-demand expectations.

These weaknesses are useful because they are specific. The next data/reward
iteration should add targeted traces and/or reward shaping for:

- direct energy supply shocks where OIL is the first-order beneficiary;
- AI efficiency shocks where lower power demand hurts GREEN infrastructure and
  data-center real estate even while software margins improve.

## 6. Demo and Deployment Process

The demo started as a simpler inference Space and then evolved into a custom
FastAPI/HTML walkthrough.

The current Space code is saved in:

```text
carbonalpha_demo_space/
```

Important files:

- `carbonalpha_demo_space/app.py`
- `carbonalpha_demo_space/static/index.html`
- `carbonalpha_demo_space/Dockerfile`
- `carbonalpha_demo_space/requirements.txt`

The Space loads:

```text
77ethers/CarbonAlpha/sft_qwen25_7b_curriculum400_v1
```

as a LoRA adapter over:

```text
unsloth/Qwen2.5-7B-Instruct
```

The UI pattern was inspired by the Round 1 GridOps cockpit:

- left control rail;
- central environment walkthrough;
- right score rail;
- quarter strip;
- completed-quarter ledger;
- review back/forward controls.

The most important UX correction was changing the app from “generate a full
future report immediately” to “lock one allocation, then let the user advance
the environment quarter by quarter.”

The Space currently still points to the Qwen2.5 SFT adapter. Before switching
the live demo to the GRPO adapter, we should review the completed 10-question
macro eval and run a walkthrough UX pass against:

```text
77ethers/CarbonAlpha/grpo_qwen25_7b_adapter_phase1_100_v1
```

That prevents us from promoting a numerically better model that has a visible
demo regression on user-entered macro news.

## 7. Engineering Lessons

### Environment Lessons

- The environment must be adversarial-tested before model training. RL will
  find reward exploits faster than humans expect.
- A simple action interface can still produce rich dynamics if the simulator
  is path-dependent.
- Prompt construction should be part of the environment package, not a random
  training-script string.
- A strong baseline is a feature, not an inconvenience.
- Demo UX must make hidden environment state explicit enough that users can
  trust what is being scored.

### Training Lessons

- SFT quality and prompt alignment mattered more than rushing into GRPO.
- `lora_alpha=16` was safer than `alpha=32` for the trace scale we had.
- GRPO smoke metrics must be checked before trusting any reward curve.
- One-token rollout collapse is an immediate abort condition.
- vLLM is not mandatory for GRPO. When rollout health is the problem, a slower
  plain-Transformers GRPO loop can be the safer engineering path.
- A successful GRPO smoke does not remove the need for manual macro evals.
  Human-facing weaknesses show up in different places than simulator regret.
- HF Jobs was more reproducible than repeatedly rebuilding RunPod state.
- Keep artifacts isolated by subfolder; never overwrite the known-good model.

### Deployment Lessons

- Custom FastAPI + Docker gave us better control than Gradio for this demo.
- The Space needs `HF_API_TOKEN` as a secret because the model repo is private.
- In-memory sessions are acceptable for a single-replica demo, but a production
  version should persist sessions if multiple workers or restarts matter.

## 8. Current Artifact Map

Safe numerical model:

```text
77ethers/CarbonAlpha/v6_sft_only_v2
```

Current demo model:

```text
77ethers/CarbonAlpha/sft_qwen25_7b_curriculum400_v1
```

Successful GRPO smoke model:

```text
77ethers/CarbonAlpha/grpo_qwen25_7b_adapter_smoke_v3
```

Successful 100-step Phase 1 GRPO model:

```text
77ethers/CarbonAlpha/grpo_qwen25_7b_adapter_phase1_100_v1
```

Training data:

```text
sft_traces/merged_v6_aligned.jsonl
sft_traces/curriculum_400_e80_m160_h160.jsonl
```

Training scripts:

```text
scripts/hf_train.py
scripts/hf_sft_qwen25_7b.py
scripts/hf_grpo_qwen25_adapter.py
scripts/hf_grpo_qwen3_base.py
scripts/hf_compare_qwen25.py
scripts/launch_macro_eval.py
notebooks/grpo_training.py
```

Evaluation artifacts:

```text
evals/macro_eval_10.jsonl
evals/macro_eval_10_grpo_report.json
```

Environment:

```text
portfolio_env/
```

Demo Space source:

```text
carbonalpha_demo_space/
```

Failure record:

```text
TRAINING_ERRORS.md
```

## 9. Final Position

The strongest story is not "we ran one training recipe." The strongest story
is that we built a real OpenEnv environment, broke and patched its reward
mechanics, generated a curriculum of reasoning traces, trained multiple model
lineages, preserved known-good artifacts, recovered a working GRPO path after
the first RL stack failed, and deployed a custom walkthrough that makes the
environment understandable quarter by quarter.

CarbonAlpha’s current best production stance is:

- keep `v6_sft_only_v2` as the safe numerical baseline;
- keep `sft_qwen25_7b_curriculum400_v1` as the current live demo model until
  the GRPO model clears a demo-focused review;
- treat `grpo_qwen25_7b_adapter_phase1_100_v1` as the current best research
  model by holdout regret;
- document Qwen3/vLLM GRPO as a failed path, but Qwen2.5 adapter GRPO with
  `use_vllm=False` as a successful Phase 1 rescue;
- fix the two manual-eval weaknesses before promoting the GRPO adapter to the
  public walkthrough.
