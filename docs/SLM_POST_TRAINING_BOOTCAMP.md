# Build Your Own AI: Post-Training SLM Bootcamp Playbook

This bootcamp is not about fine-tuning a chatbot.

It is about enabling students, builders, founders, researchers, and operators to
build their own domain AI systems for their own use cases: energy, water,
gaming, sustainability, agriculture, robotics, rocketry, logistics,
manufacturing, education, finance, and any other field where decisions can be
defined, checked, improved, and deployed.

GridOps and CarbonAlpha are case studies. The real lesson is the reusable
process behind them.

```text
Use case -> environment -> schema -> data curriculum -> SFT -> evals -> RL -> demo
```

The bootcamp goal is to help each participant leave with a repeatable mental
model and a working starter kit for building a small, useful, domain-trained AI.

## The Core Belief

India does not only need people who can call an API. India needs builders who
can turn local domain knowledge into capable machines.

That might mean:

- a housing society AI that schedules solar battery usage;
- a farmer-support AI that recommends irrigation actions under water scarcity;
- a robotics AI that selects safe warehouse movement primitives;
- a gaming AI that learns a custom NPC strategy policy;
- a rocketry AI that validates test-stand procedures;
- a factory AI that detects and responds to machine operating states;
- a sustainability AI that manages carbon-aware procurement;
- a logistics AI that allocates vehicles under fuel and route constraints.

The common pattern is the same:

```text
The model must read state, choose an action, obey constraints, and improve a
measurable outcome.
```

When students understand that pattern, they are no longer limited to copying
notebooks. They can build their own AI for their own world.

## What "Build Your Own AI" Means

In this bootcamp, "from scratch" does not mean pretraining a foundation model
from zero tokens. That is expensive and unnecessary for most teams.

It means building the whole applied AI system from first principles:

- define the problem;
- create the environment or task harness;
- define the input and output contract;
- generate or collect domain traces;
- validate every training example;
- post-train a small language model;
- evaluate it against baselines;
- package the result as a model, dataset, demo, and model card.

The model is only one part. The real product is the harness around the model.

## The Universal Process

### 1. Choose A Real Use Case

Start with a problem where better decisions matter.

Good bootcamp use cases have:

- a clear user or operator;
- a state that can be observed;
- an action that can be taken;
- constraints that must not be violated;
- an objective that can be measured;
- enough examples, simulations, logs, or expert rules to create traces.

Weak use cases sound like:

```text
Make a general assistant for X.
```

Strong use cases sound like:

```text
Given this state of X, choose the next valid action that improves metric Y
without violating constraint Z.
```

Examples:

| Domain | State | Action | Objective |
| --- | --- | --- | --- |
| Energy | demand, solar, battery SOC, price | battery/diesel/shedding action | lower cost, avoid blackout |
| Water | tank level, rainfall, demand, pump health | pump schedule | avoid shortage, reduce power cost |
| Gaming | player state, enemy state, map | next move or strategy | win rate, fun, difficulty balance |
| Robotics | sensors, pose, obstacle map | movement primitive | reach goal safely |
| Rocketry | test readings, procedure stage | next checklist action | safe test progression |
| Agriculture | crop, soil, weather, water | irrigation/fertilizer action | yield, water efficiency |
| Logistics | orders, vehicles, traffic, fuel | dispatch route | on-time delivery, lower cost |
| Sustainability | supplier, price, carbon, demand | procurement allocation | cost under carbon budget |

### 2. Convert The Use Case Into A Contract

Every serious training project needs a contract before it needs a model.

Define:

- input schema;
- output schema;
- valid action bounds;
- invalid action behavior;
- success metric;
- failure modes;
- baseline policies;
- evaluation seeds or holdout set.

Example output contracts:

```json
{"battery_dispatch": 0.0, "diesel_dispatch": 0.0, "demand_shedding": 0.0}
```

```json
{"pump_1": "on", "pump_2": "off", "duration_minutes": 30}
```

```json
{"move": "advance", "speed": 0.4, "turn_degrees": -15}
```

Rule:

```text
If the output cannot be parsed, validated, and scored, it is not ready for
training.
```

### 3. Build The Environment Or Harness

The harness is where the model meets reality.

Depending on the use case, the harness can be:

- a simulator;
- a game engine;
- a spreadsheet-like evaluator;
- a Python rules engine;
- a robotics sandbox;
- a historical backtest;
- an API wrapper around real tools;
- a human review interface with validators.

The harness should answer:

- did the action parse?
- was it valid?
- what happened after the action?
- what score did it get?
- which baseline did it beat or lose to?

This is why OpenEnv-style thinking is powerful. It forces the project to become
an environment, not just a prompt.

### 4. Establish Baselines

Never train without baselines.

Minimum baselines:

- do-nothing policy;
- simple heuristic;
- domain expert or oracle policy;
- untouched base/instruct model;
- SFT-only model;
- later, RL-refined model if useful.

Baselines make the project honest. If a simple heuristic is better than the
model, that is not failure. It is information. The next dataset should teach the
model exactly where the heuristic wins.

### 5. Create A Curriculum Dataset

A good dataset is a teaching plan.

Split traces into:

- easy: format, common cases, obvious actions;
- medium: competing signals, partial ambiguity, timing tradeoffs;
- hard: rare failures, counterintuitive moves, resource scarcity, edge cases.

Each trace should contain:

- trace ID;
- domain/task;
- seed or source;
- observation;
- target action;
- baseline action;
- score context;
- difficulty;
- focus tags;
- validation status;
- source of label.

Trace sources can include:

- expert demonstrations;
- rule-based oracle;
- simulator search;
- historical logs;
- human annotation;
- frontier model proposal;
- failure replay from earlier model errors.

The bootcamp should teach students this sentence:

```text
Do not just generate more data. Generate the next data your model needs.
```

### 6. Use Frontier Models As Assistants, Not Judges

Large frontier models can help create candidate traces, explain failures, and
suggest edge cases. They should not be treated as truth.

Recommended pattern:

1. sample states from the environment;
2. ask a strong model for candidate actions in strict JSON;
3. validate the JSON locally;
4. score candidates in the environment;
5. compare against baseline and oracle;
6. accept useful candidates;
7. store rejected candidates for failure analysis.

For our GridOps v3 work:

- DeepSeek V4 Flash proposed cheap candidate actions;
- Gemma 4 26B A4B added diversity;
- DeepSeek V4 Pro was reserved for harder judgments;
- the Python simulator, not the LLM, decided what survived.

This pattern generalizes to any domain.

### 7. Validate Before Training

Validation protects GPU time.

Check:

- prompts match inference format;
- completions parse as JSON or required schema;
- actions satisfy bounds;
- no duplicate trace IDs;
- task distribution is intentional;
- hard cases exist;
- labels are not constant;
- rejected examples are preserved separately;
- small samples are manually inspected.

Validation is not paperwork. It is model quality.

### 8. Train With SFT First

SFT teaches:

- output format;
- domain vocabulary;
- common actions;
- base-rate behavior;
- schema discipline;
- compact reasoning or no-reasoning style.

Recommended starting models:

- `Qwen/Qwen2.5-1.5B-Instruct` for low-cost experiments;
- `Qwen/Qwen2.5-3B-Instruct` for stronger small-model performance;
- 7B models only when GPU budget allows.

Training style:

- LoRA or QLoRA;
- small batch size;
- gradient accumulation;
- fixed run label;
- no overwrite of previous artifacts;
- upload to Hugging Face under a new subfolder.

Healthy SFT signals:

- loss decreases;
- token accuracy rises;
- grad norm is nonzero;
- samples remain valid;
- model does not collapse to one constant answer.

### 9. Evaluate Like The Real User

Loss is not the final metric. The real metric is whether the model helps the
domain.

Evaluation layers:

- parse and validity rate;
- task score;
- cost, safety, regret, or domain metric;
- performance by difficulty;
- performance by scenario;
- comparison against do-nothing;
- comparison against heuristic/oracle;
- manual challenge set;
- demo behavior.

A strong eval names the next improvement:

```text
The model handles normal cases, but fails under low-resource crisis states.
Next data: crisis traces with low resource and rebound examples.
```

### 10. Use RL Only After SFT Is Stable

RL is not a magic first step. It is a refinement tool.

Use RL only when:

- the model already emits valid actions;
- reward functions are executable;
- there is reward variance;
- holdout eval can catch regressions;
- the SFT model remains preserved.

Reward components should map to the domain:

- schema reward;
- valid action reward;
- task score reward;
- regret against baseline;
- safety or constraint penalty;
- cost penalty;
- resource-use penalty;
- robustness reward.

If RL hurts reliability, ship the SFT model and document RL as an experiment.

### 11. Package The AI System

A serious project should publish:

- dataset or dataset card;
- training script;
- notebook for Colab/Kaggle;
- model or adapter;
- model card;
- evaluation table;
- plots from a real run;
- demo or API;
- README with links;
- known limitations.

This is what separates a weekend experiment from a credible AI system.

## Case Studies

### GridOps

Problem:

```text
Given hourly microgrid state, choose battery, diesel, and shedding actions.
```

Why it matters:

India is rapidly adding solar, batteries, EV charging, and distributed energy
assets. Apartments, societies, campuses, villages, and industrial parks will
need intelligence layers that reduce cost and keep power reliable without
requiring a full-time expert operator everywhere.

What we built:

- OpenEnv-compatible microgrid environment;
- strict JSON action schema;
- oracle and adversarial policies;
- curriculum traces across normal, heatwave, and crisis tasks;
- OpenRouter-assisted data factory;
- Qwen2.5 SFT pipeline;
- Kaggle training runner;
- holdout evaluation plan.

Current important files:

```text
docs/KAGGLE_SFT_V3.md
notebooks/gridops_kaggle_sft_v3.py
scripts/kaggle_sft_v3_gridops.sh
scripts/generate_openrouter_tool_augmented_traces.py
scripts/build_gridops_v3_curriculum.py
scripts/hf_sft_gridops.py
sft_traces/gridops_curriculum_v3_tool_augmented.jsonl
evals/gridops_curriculum_v3_tool_augmented_summary.json
```

Current Hugging Face target:

```text
77ethers/gridops-models/sft_qwen25_3b_gridops_tool_augmented_v3
```

### CarbonAlpha

Problem:

```text
Given macro news and portfolio state, allocate under return, risk, and carbon
constraints.
```

What it taught us:

- preserve a safe SFT model before RL;
- make rewards match the domain;
- keep model artifacts isolated;
- use holdout regret and parseability as gates;
- document training evidence with plots, logs, model cards, and demo links.

CarbonAlpha shows how the same process applies outside energy operations:
environment, schema, traces, SFT, RL reward design, evals, model card, demo.

## Bootcamp Formats

### One-Day Bootcamp

Morning:

- mindset: from chatbot to domain AI system;
- use-case selection;
- environment and schema design;
- baselines and validators;
- case study walkthrough.

Afternoon:

- curriculum dataset design;
- trace validation;
- SFT launch on Kaggle/Colab;
- reading training logs;
- eval and model card template;
- student use-case planning clinic.

Outcome:

- students understand the full loop;
- each team has a problem contract and dataset plan;
- one shared training run reaches a visible checkpoint.

### Two-Day Bootcamp

Day 1:

- choose a domain problem;
- define state, action, constraints, and metric;
- build or adapt a small harness;
- create baselines;
- design easy/medium/hard traces;
- generate or inspect a starter dataset.

Day 2:

- validate traces;
- run SFT;
- evaluate against baselines;
- inspect failures;
- design next data;
- package model card and demo;
- optional RL/reward design session.

Outcome:

- every team leaves with a reproducible project skeleton;
- stronger teams leave with a trained adapter and eval table;
- everyone understands how to continue improving the model after the bootcamp.

## Student Deliverables

Each team should produce:

- problem statement;
- input/output schema;
- environment or evaluator;
- baseline policies;
- curriculum dataset plan;
- at least 50-200 validated starter traces;
- SFT training command or notebook;
- eval table;
- 5 failure cases;
- next-data plan;
- README or model card.

Stretch deliverables:

- 1,000+ trace curriculum;
- trained LoRA adapter;
- Hugging Face model card;
- public demo;
- RL reward design;
- short video walkthrough.

## Reusable Project Skeleton

```text
my-domain-ai/
  README.md
  pyproject.toml
  domain_env/
    models.py
    env.py
    scoring.py
    prompting.py
  scripts/
    generate_traces.py
    validate_traces.py
    train_sft.py
    evaluate_model.py
  sft_traces/
    curriculum_v1.jsonl
  evals/
    baseline_summary.json
    model_holdout.json
  notebooks/
    train_on_kaggle.py
  docs/
    process.md
    model_card.md
```

## The Bootcamp Promise

By the end, students should be able to say:

```text
I know how to convert a domain problem into an AI training problem.
I know how to build the harness, generate the data, train the model, evaluate
it, and publish the result.
```

That is the real crux.

The future is not only bigger models. It is millions of smaller, sharper,
domain-trained machines built by people who understand their own problems.
