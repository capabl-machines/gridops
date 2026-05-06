# Post-Training Small Language Models: GridOps Bootcamp Playbook

This document captures the end-to-end process we used across CarbonAlpha and
GridOps, shaped into a practical 1-2 day bootcamp for Indian AI students who
want to post-train a small language model for a real decision environment.

The core lesson is simple: do not train a model on vibes. Build an environment,
define the action contract, generate traces through a curriculum, train a small
model to obey the contract, evaluate it against baselines, and publish enough
evidence that someone else can reproduce the work.

## Why This Bootcamp Should Exist

India is adding solar, batteries, EV charging, and distributed energy assets at
neighbourhood scale. Apartments, societies, campuses, villages, industrial
parks, and small microgrid operators increasingly face the same operational
question every hour:

```text
How do we keep the lights on, use the battery wisely, avoid diesel overuse,
and reduce cost without needing a full-time expert operator?
```

That is a strong AI training problem because it has:

- a real environment with measurable outcomes;
- a compact action schema;
- obvious baselines to beat;
- simulator feedback;
- high social and commercial impact;
- a path from model to usable operations tool.

The bootcamp goal is not to create a chatbot. The goal is to create a compact
"capabl machine": a model-backed decision layer that can read structured state,
emit valid actions, and improve an operational score.

## Case Study: GridOps

GridOps is a community microgrid optimization environment. At each step the
model receives an hourly observation and must output a JSON action:

```json
{"battery_dispatch": 0.0, "diesel_dispatch": 0.0, "demand_shedding": 0.0}
```

The environment then scores whether the action made sense. The model is not
rewarded for sounding intelligent. It is rewarded for operational behavior:

- battery use aligned with solar, price, demand, and state of charge;
- diesel used when necessary but not wasted;
- demand shedding minimized and used only under stress;
- blackout risk controlled;
- cost controlled;
- task score improved over do-nothing and weak policies.

The first milestone is SFT, not RL. The model must first become reliable at
emitting valid JSON actions before any reinforcement learning is worth doing.

## The End-To-End Process

### 1. Define The Environment Contract

Before training, write down the machine-readable contract:

- observation schema;
- action schema;
- valid action bounds;
- task IDs;
- reset and step semantics;
- scoring function;
- baselines;
- failure modes.

For GridOps, the contract lives around:

- `gridops.models.GridOpsObservation`
- `gridops.models.GridOpsAction`
- `/api/reset`
- `/api/step`
- `/ws`
- the dashboard
- the oracle and adversarial policies

Rule for students: if the output cannot be parsed and scored, it is not a
training target yet.

### 2. Build Baselines First

Baselines turn model performance into an honest comparison.

Minimum baselines:

- do-nothing or grid-only policy;
- simple heuristic or oracle policy;
- untouched base/instruct model;
- SFT-only model;
- later, RL-refined model if SFT passes gates.

For GridOps, adversarial policies are useful teaching tools:

- always charge;
- always discharge;
- always diesel;
- shed-farmer;
- diesel-chatter;
- blackout-acceptor;
- price-greedy;
- grid-only.

These policies reveal what bad control looks like. Students can inspect why
they fail before trusting a trained model.

### 3. Create A Curriculum Dataset

A good post-training dataset is not just a pile of examples. It is a syllabus
for the model.

GridOps v3 curriculum shape:

- format anchors: teach exact JSON output;
- normal states: teach battery timing and cost control;
- heatwave states: teach demand pressure and rebound behavior;
- crisis states: teach outage, low SOC, diesel scarcity, and shedding tradeoffs;
- failure replay: teach corrections from known bad policies;
- tool-augmented traces: allow LLM proposers, but let the simulator decide.

Difficulty tags:

- easy: common states, obvious valid action, format learning;
- medium: competing signals such as high demand but low price, or solar about to
  arrive;
- hard: outage, low SOC, diesel scarcity, rebound, or multiple risks at once.

Every trace should include:

- task;
- seed;
- hour;
- observation snapshot;
- chosen action;
- oracle action;
- score context;
- difficulty;
- focus tags;
- validation status;
- source policy or proposer model.

For this repo, the current v3 dataset is:

```text
sft_traces/gridops_curriculum_v3_tool_augmented.jsonl
```

Current size:

```text
2,111 traces
```

### 4. Use Frontier Models As Proposers, Not Truth

DeepSeek, Gemma, Gemini, Claude, GPT, or any other strong model can help propose
candidate actions or edge cases. But the simulator remains the judge.

The pattern:

1. sample observations from fixed seeds;
2. send 10 observations per API call when practical;
3. request strict JSON actions;
4. validate through Pydantic;
5. score in the simulator;
6. compare against oracle and baselines;
7. accept, reject, or store as failure replay.

For GridOps v3:

- primary proposer: DeepSeek V4 Flash through OpenRouter;
- diversity proposer: Gemma 4 26B A4B through OpenRouter;
- escalation judge: DeepSeek V4 Pro for hard or close cases;
- storage: accepted and rejected JSONL files plus summary metrics.

Important rule: SFT completions remain JSON-only. Reasoning can be stored in
metadata if useful, but the trained model should emit the exact action contract.

### 5. Validate Before Training

Validation catches expensive mistakes before the GPU starts.

Checklist:

- every completion parses as JSON;
- every action validates through `GridOpsAction`;
- prompts match the inference prompt format;
- no duplicate trace IDs;
- task distribution is intentional;
- hard examples are present;
- rejected traces are preserved separately;
- a small sample is manually inspected.

Command:

```bash
python scripts/validate_traces.py sft_traces/gridops_curriculum_v3_tool_augmented.jsonl
```

### 6. Train With SFT First

Recommended first model:

```text
Qwen/Qwen2.5-3B-Instruct
```

Fallback for tight GPUs:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Training style:

- QLoRA adapter;
- batch size 1;
- gradient accumulation;
- max length 1024-1280 on Kaggle T4;
- gradient checkpointing on;
- upload adapter to a new Hugging Face subfolder.

Kaggle runner:

```bash
bash scripts/kaggle_sft_v3_gridops.sh
```

Kaggle notes:

- T4 x2 means two 15 GB GPUs, not one 30 GB GPU;
- assume one 15 GB GPU unless the training stack clearly uses both;
- start with 100 steps as a smoke run;
- scale only after loss and evals look healthy.

Healthy SFT signals:

- loss decreases and stabilizes;
- mean token accuracy rises;
- grad norm is nonzero and not NaN;
- samples remain valid JSON;
- the model does not collapse to constant actions.

### 7. Evaluate The Model Like An Operator

Low loss is not enough. The model must operate the grid.

Evaluation metrics:

- valid action rate;
- average score by task;
- score against do-nothing;
- score against oracle;
- blackout kWh;
- diesel kWh;
- demand shed kWh;
- total cost;
- battery dispatch behavior by SOC, solar, price, and outage status.

Promotion gates:

- valid action rate at least 99.5 percent;
- task 1 keeps battery improvement;
- task 2 handles heatwave and rebound;
- task 3 recovers crisis performance;
- average holdout beats v1 and v2;
- no task falls below do-nothing baseline.

Manual challenge set:

- low SOC before outage;
- excess solar at noon;
- high price evening ramp;
- heatwave demand spike;
- diesel scarcity;
- rebound after shedding;
- cloudy day with limited solar;
- normal night demand;
- battery almost full;
- grid outage with critical load.

### 8. Only Then Consider RL

RL is useful only after the model reliably emits valid actions.

Reward components should map to portfolio or grid reality, not abstract text
quality:

- schema reward;
- valid action reward;
- task score reward;
- regret against oracle;
- blackout penalty;
- diesel overuse penalty;
- demand shedding penalty;
- cost penalty;
- battery sanity reward.

RL smoke gate:

- run 5-10 steps first;
- completion length is not collapsed;
- parse rate stays high;
- grad norm is nonzero;
- reward variance is nonzero;
- at least one holdout metric improves.

If RL breaks reliability, keep the SFT model. The safe model is an asset.

## Bootcamp Format

### One-Day Version

Morning:

- problem framing: from chatbot to decision machine;
- GridOps environment walkthrough;
- action schema and scoring;
- baselines and adversarial policies;
- trace format and curriculum design.

Afternoon:

- generate or inspect traces;
- validate dataset;
- launch QLoRA SFT on Kaggle or Colab;
- inspect training logs;
- run holdout eval;
- package model card and demo notes.

Outcome:

- each student understands the full loop;
- one shared model run completes or reaches a useful checkpoint;
- students leave with a reproducible template.

### Two-Day Version

Day 1:

- environment contract;
- baselines;
- oracle and adversarial policies;
- curriculum dataset design;
- OpenRouter or Gemini-based trace proposal;
- validation and dataset ledger.

Day 2:

- SFT training;
- evals and failure analysis;
- model card;
- Hugging Face upload;
- demo/dashboard integration;
- optional RL design discussion.

Outcome:

- students build a complete post-training project;
- teams compare model behavior across tasks;
- best teams submit a model card, eval table, and demo video.

## Student Project Template

Every team should produce:

- `README.md`: problem, environment, results, links;
- `docs/process.md`: task contract and training process;
- `sft_traces/*.jsonl`: validated curriculum;
- `scripts/validate_traces.py`: parser and validator;
- `scripts/train_sft.py`: reproducible SFT runner;
- `scripts/evaluate_model.py`: holdout eval;
- `notebooks/*.ipynb` or notebook-style `.py`: Colab/Kaggle run;
- Hugging Face model card;
- 5-10 failure examples and what they teach.

## Artifact Ledger For GridOps

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

## Teaching Philosophy

Students should leave with one deep mental model:

```text
The model is only one component. The real product is the harness around it.
```

The harness includes:

- environment;
- schema;
- data factory;
- validators;
- baselines;
- rewards;
- evals;
- logs;
- model card;
- demo;
- deployment path.

That is the difference between a prompt experiment and a serious AI system.
