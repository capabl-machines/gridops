# Why Climate AI Needs Harnesses, Not Just Models

Most AI demos are built around answers.

Climate systems need decisions.

An answer can be fluent and still be useless. A decision has to survive contact
with constraints: physics, cost, risk, safety, demand, weather, equipment,
uncertainty, and human consequences.

That is why climate AI needs harnesses, not just models.

## A Model Is Not An Operating System

A model can read a situation and produce text. That is powerful.

But a climate-heavy workflow usually needs more:

- a state representation;
- a valid action schema;
- a simulator or environment;
- deterministic tools;
- validators;
- reward functions;
- holdout scenarios;
- baselines;
- failure logs;
- deployment guardrails.

Without that harness, the model is only guessing in prose.

With the harness, the model becomes one component in an accountable operating
loop.

```text
state
  -> model intent
  -> tool-backed decision
  -> validator
  -> environment score
  -> learning data
```

## Climate Problems Are Control Problems

Many climate workflows are not pure prediction problems.

They are control problems.

Energy systems decide when to store, dispatch, shed, or conserve. Water systems
decide when to pump, irrigate, refill, or ration. Cold chains decide when to
pre-cool, preserve, or spend energy. EV depots decide when to charge, delay, or
protect transformer limits.

These systems are full of trade-offs:

```text
cost vs reliability
emissions vs resilience
short-term savings vs long-term risk
local action vs system rebound
```

A generic model can discuss those trade-offs. But discussion is not operation.

Operation requires a harness that turns judgment into constrained action.

## What A Harness Does

A climate AI harness gives the model a job it can actually perform.

In GridOps, we first asked the model to output exact microgrid dispatch values:

```json
{"battery_dispatch": 0.0, "diesel_dispatch": 0.0, "demand_shedding": 0.0}
```

That made the model responsible for numerical control. It was brittle.

The better interface was strategy:

```json
{
  "mode": "reliability",
  "risk_level": "high",
  "battery_bias": "preserve",
  "diesel_policy": "allow_if_blackout",
  "shedding_policy": "last_resort"
}
```

The harness then did the rest:

- the optimizer converted strategy into dispatch;
- the validator checked bounds;
- the OpenEnv environment scored outcomes;
- the evaluation script compared against baselines and LP ceiling.

The model did not need to become the whole controller. It needed the right role.

## The GridOps Result

GridOps is a simulated Indian community microgrid environment. It tests normal
operation, heatwave operation, and crisis operation with haze, outage, limited
diesel, and reliability pressure.

The direct-action model improved over simple baselines, but stayed weak in
crisis.

The strategy harness changed the result.

An untuned `Qwen/Qwen2.5-1.5B-Instruct` model, placed inside the v7 strategy
harness, reached:

```text
0.7911 average score
100% valid strategy outputs
96.09% LP ceiling capture
```

The trained v7.3 selector reached:

```text
0.7888 average score
100% valid strategy outputs
95.81% LP ceiling capture
```

That result is important because it is honest.

The fine-tuned model did not beat the base model. The harness made the base
model capable.

For a company building useful climate AI, that is not embarrassing. That is the
insight.

## The Product Is The Operating Loop

The wrong conclusion would be:

```text
Fine-tuning failed, so there is nothing here.
```

The better conclusion is:

```text
The model alone was not the product. The operating loop was.
```

Capabl Machines should build the full loop:

```text
environment
  -> datasets
  -> model layer
  -> tools
  -> validators
  -> optimizer
  -> rewards
  -> evals
  -> demo
  -> deployment pathway
```

Sometimes post-training will create the edge. Sometimes a base model will be
good enough once the harness is designed well. Sometimes a deterministic
optimizer should lead and the model should only choose intent.

The point is not to force one technique everywhere.

The point is to deliver a reliable system.

## What This Means For Climate AI

The next wave of climate AI should not only produce reports, summaries, or
recommendations.

It should be tested inside operational environments.

For energy:

```text
Can it reduce blackout without wasting diesel?
```

For water:

```text
Can it meet demand without over-pumping?
```

For agriculture:

```text
Can it preserve yield while saving water and power?
```

For cold chains:

```text
Can it protect goods while reducing compressor cost?
```

For EV depots:

```text
Can it charge vehicles without breaking transformer limits?
```

These are not vibes. They are measurable outcomes.

## The Benchmark Opportunity

The best way to build soft power is to stop only releasing models and start
releasing environments.

GridOps can become a public benchmark for climate AI agents:

```text
Submit a model or strategy.
Run it through normal, heatwave, and crisis scenarios.
Compare cost, blackout, diesel, validity, and LP capture.
```

That changes the conversation.

Instead of saying:

```text
Look at our model.
```

We can say:

```text
Here is a climate operations benchmark. Bring your model. Beat our harness.
```

That is how Capabl Machines can become known for serious work.

## Closing

Climate AI will not be won by prompts alone.

It will be won by teams that can turn messy operational domains into tested
AI harnesses.

That means building the world around the model:

- the environment;
- the tools;
- the validators;
- the reward functions;
- the datasets;
- the evals;
- the demo;
- the failure analysis.

The model is still important. But for climate-heavy systems, the harness is
what makes the model useful.

