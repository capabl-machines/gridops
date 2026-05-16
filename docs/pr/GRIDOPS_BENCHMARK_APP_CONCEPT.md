# GridOps Benchmark App Concept

## Purpose

Turn GridOps from a single model release into a public benchmark for climate AI
operators.

The benchmark should let people test different models, prompts, strategy
policies, and tool-assisted agents on the same simulated Indian community
microgrid environment.

The positioning:

```text
Bring your model. Operate the microgrid. Beat the benchmark.
```

## Why This Helps PR

A model card says:

```text
Here is what we built.
```

A benchmark says:

```text
Here is the problem. Come compete.
```

That moves Capabl Machines from participant to category-builder.

The soft-power goal is to become known for:

- climate AI environments;
- physical-system evals;
- model + optimizer harnesses;
- honest baseline comparisons;
- India-relevant infrastructure problems.

## Benchmark User Flow

### 1. Select System

Options:

```text
Deterministic strategy-controller
Untuned Qwen 2.5 1.5B + strategy harness
Released v7.3 trained selector
Custom model endpoint
Custom strategy JSON policy
Manual strategy input
```

### 2. Select Scenario

Options:

```text
Task 1: normal operation
Task 2: heatwave
Task 3: crisis / outage
Full holdout suite
```

### 3. Run Episode

For each hour:

```text
observation -> strategy -> optimizer -> action -> environment step
```

### 4. Show Results

Metrics:

```text
average score
task score
valid strategy rate
valid action rate
blackout kWh
diesel kWh
cost
LP ceiling capture
```

Charts:

```text
SOC over time
demand vs solar
grid/diesel/battery dispatch
blackout kWh
cost accumulation
strategy mode timeline
```

### 5. Share Result

Generate a shareable run card:

```text
Model: GPT / Claude / Gemini / Qwen / local / custom
Scenario: crisis
Score: 0.7517
LP capture: 95.01%
Blackout: 356.85 kWh
Diesel: 757.0 kWh
Valid strategy: 100%
```

## Leaderboard

Table:

| Rank | System | Avg | Normal | Heatwave | Crisis | Validity | LP Capture | Blackout | Diesel | Cost |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | full LP ceiling | 0.8233 | 0.8372 | 0.8416 | 0.7912 | - | 100.00% | - | - | - |
| 2 | untuned Qwen 2.5 1.5B + harness | 0.7911 | 0.7993 | 0.8223 | 0.7517 | 100% | 96.09% | 356.85 | 757.0 | 216568 |
| 3 | v7 deterministic controller | 0.7907 | 0.7995 | 0.8224 | 0.7503 | 100% | 96.04% | 338.70 | 760.2 | 217690 |
| 4 | v7.3 trained selector | 0.7888 | 0.7993 | 0.8223 | 0.7449 | 100% | 95.81% | 404.68 | 760.2 | 222097 |
| 5 | v5.1 direct-action model | 0.7354 | 0.7896 | 0.7681 | 0.6484 | 99.69% | - | - | - | - |

## App Sections

### Hero

Title:

```text
GridOps Benchmark
```

Subtitle:

```text
Test AI agents on community microgrid operation across normal days, heatwaves,
and outage crises.
```

CTA:

```text
Run Benchmark
Submit A Model
```

### How It Works

```text
1. The environment provides a 72-hour microgrid state.
2. The model chooses a strategy or action.
3. The harness validates and executes it.
4. OpenEnv scores cost, reliability, diesel use, and blackout.
5. Results are compared against baselines and LP ceiling.
```

### Why It Matters

```text
Climate AI should be judged by operational outcomes, not fluent text.
GridOps makes that measurable for one India-relevant energy problem.
```

## Submission Modes

### Mode A: Strategy JSON Endpoint

The user provides an API endpoint:

```text
POST /strategy
input: GridOps observation
output: GridOpsStrategy JSON
```

### Mode B: Prompted Model

The app calls a model provider:

```text
OpenAI / Anthropic / Gemini / OpenRouter / Hugging Face endpoint
```

The app owns the prompt and parser.

### Mode C: Local Policy File

User uploads a Python policy:

```python
def policy(observation: dict) -> dict:
    return {
        "mode": "cost_saving",
        "risk_level": "low",
        "battery_bias": "charge",
        "diesel_policy": "avoid",
        "shedding_policy": "never",
    }
```

Use this only in sandboxed/offline mode.

## MVP Scope

Keep the first version simple:

- static leaderboard;
- run deterministic/controller baselines;
- allow manual strategy testing;
- allow one model endpoint integration;
- export JSON report;
- show charts.

Do not start with multi-user auth, payments, or a full competition system.

## Technical Architecture

```text
frontend
  -> scenario selector
  -> model/policy selector
  -> run viewer
  -> leaderboard

backend
  -> GridOps environment
  -> strategy parser
  -> optimizer/controller
  -> evaluator
  -> report writer
```

## PR Launch Sequence

1. Publish the two essays:
   - `What We Learned Building an AI Operator for Indian Microgrids`
   - `Why Climate AI Needs Harnesses, Not Just Models`
2. Publish the benchmark app.
3. Post the leaderboard with a challenge:

```text
Can your model operate an Indian community microgrid better than our harness?
```

4. Invite model builders:

```text
We will test GPT, Claude, Gemini, Qwen, Gemma, Kimi, DeepSeek, and open-source
agents on the same GridOps scenarios.
```

5. Share failure cases:

```text
Where models waste diesel.
Where models allow blackout.
Where models violate schema.
Where optimizer-backed systems recover.
```

## Long-Term Expansion

Once GridOps works, repeat the same benchmark pattern for:

- water pump scheduling;
- cold storage operation;
- EV depot charging;
- irrigation planning;
- factory energy optimization;
- disaster-resilient infrastructure.

The category becomes:

```text
ClimateOps Benchmarks by Capabl Machines
```

