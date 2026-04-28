---
name: train-custom-model
description: Plan and execute reproducible custom model-training projects for new use cases, including task framing, environment/harness design, dataset curriculum, synthetic trace generation, SFT, RL/GRPO/DPO-style reward design, evals, artifact management, Hugging Face/Colab publishing, and demo/readme evidence. Use when the user wants to train, fine-tune, evaluate, rescue, document, or submit a new AI model pipeline.
---

# Train Custom Model

## Overview

Use this skill to turn a vague model idea into a reproducible training pipeline with evidence. Favor small, isolated experiments, explicit artifacts, and evals that match the real user-facing task.

For detailed checklists, read `references/model-training-checklist.md`. For reusable documentation tables and model-card snippets, read `references/artifact-template.md`.

## Operating Principles

- Preserve known-good models. Never overwrite a working baseline; write every experiment to a new path or subfolder.
- Treat the runtime around the model as a harness: state, tools, schema, guardrails, rewards, evals, logs, and demo.
- Use one source of truth for prompts, schemas, reward functions, and eval logic. Training, inference, and demo code should not silently diverge.
- Make reward functions executable and domain-connected. Do not reward "sounds good" when the task needs "works in the environment."
- Keep the first RL run tiny. Gate on parse rate, completion length, gradient health, reward variance, and a holdout smoke eval before scaling.
- Publish enough evidence that another person can verify training happened: plots, logs, config, model lineage, data description, and rerun instructions.
- If current library versions, model availability, HF APIs, or competition requirements matter, verify with current official docs or package metadata before deciding.

## Workflow

### 1. Define The Task Contract

Capture the task as an executable contract:

- user input shape;
- model output schema;
- valid/invalid actions;
- environment state and step semantics;
- success metric;
- baseline to beat;
- failure modes that matter to users.

Create a minimal environment or harness before training. For decision tasks, the environment should score real actions, not prose. For generation tasks, define parsers, validators, and rubric/eval functions early.

### 2. Establish Baselines

Pick at least three baselines when feasible:

- a non-LLM or heuristic baseline;
- an untouched base/instruct model;
- a safe SFT-only model.

Record baseline metrics before RL. If RL fails, the SFT-only model should remain shippable.

### 3. Build A Curriculum Dataset

Design traces by difficulty:

- easy: teaches schema, first-order behavior, and common cases;
- medium: adds ambiguity, partial observability, and competing signals;
- hard: forces second-order/third-order reasoning and counterintuitive actions.

Validate every trace before training:

- prompt matches inference prompt;
- completion matches output schema;
- JSON/action parses;
- constraints are satisfied;
- labels and difficulty metadata are present;
- duplicates and degenerate examples are removed.

If using a frontier model to generate traces, batch small, validate aggressively, and save rejected rows separately.

### 4. SFT First

Use SFT to teach the model:

- the exact output format;
- bounded reasoning style;
- task vocabulary;
- base-rate behavior;
- valid schema completion.

Keep SFT artifacts separate from final/RL artifacts. Run holdout eval after SFT and save examples of good and bad completions.

### 5. RL Only After The Format Is Stable

Use RL/GRPO/DPO-style refinement only when SFT already produces parseable completions. Start with a smoke run:

- 5-10 steps for a new setup;
- small prompt count;
- 2 generations per prompt if using GRPO;
- short max tokens until completion health is known;
- no overwrite of SFT or previous final model.

Abort and preserve the SFT model if there is collapse:

- completion length stuck at 1 or near-zero;
- parse rate crashes;
- gradient norm is zero/NaN;
- reward variance is zero;
- outputs get verbose without improving the environment metric.

### 6. Design Rewards From The Domain

Every reward component should answer: "What user-visible or environment-visible behavior does this improve?"

Typical components:

- format/schema reward;
- constraint reward;
- task objective reward;
- regret or improvement over baseline;
- risk/robustness reward;
- brevity or reasoning-shape reward;
- penalty for invalid, degenerate, or unsafe actions.

Keep rewards inspectable. Log component means/stds, not only the aggregate reward.

### 7. Evaluate In Layers

Use layered evals:

- smoke eval: parseability, length, reward variance, gradient health;
- holdout simulation/task eval: fixed seeds and metrics not used in training;
- manual challenge set: 10-20 examples that reveal real reasoning failures;
- demo eval: whether the UI/API output feels useful and honest.

Document weaknesses explicitly. A useful eval names the next data and reward changes.

### 8. Publish Reproducibly

Prepare:

- Colab or notebook that verifies repos, runs smoke checks, loads metrics, and can relaunch guarded jobs;
- README with Space/demo link, model link, dataset link, blog/video link, metrics, and screenshots;
- model card with lineage, intended use, limitations, training plots, logs, and eval table;
- public demo if required;
- artifact ledger with exact subfolders and run labels.

For Hugging Face, prefer:

- `HF_API_TOKEN` as the canonical token;
- public Space for judging;
- public model repo if evaluators need plots/model card/logs;
- public or well-described dataset repo depending on sensitivity.

## Output Shape

When asked to plan a new training project, produce:

- a short problem contract;
- environment/harness design;
- dataset/curriculum plan;
- model candidates and GPU fit;
- SFT plan;
- RL/reward plan;
- eval plan;
- artifact naming plan;
- risks and smoke gates;
- first commands/files to implement.

When asked to implement, create the files, scripts, notebook, and docs rather than stopping at advice.
