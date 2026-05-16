# becapable.in AI Pages Update: Capabl Machines Positioning

This brief is for updating:

- `https://becapable.in/ai`
- `https://becapable.in/ai/energy`

The positioning should be broader than "we fine-tune models." The sharper
message is:

```text
Capabl Machines builds AI operating harnesses for climate-heavy systems.
The model is one part. The environment, tools, optimizer, validator, critic,
dataset, and evaluation loop are the rest of the product.
```

## Page 1: `/ai`

### Page Goal

Make visitors understand the Capabl Machines AI thesis:

```text
Useful AI for physical systems must be trained, tooled, and evaluated inside
the world it claims to operate.
```

This page should introduce the general method and then point to Energy/GridOps
as the flagship case study.

### Hero

Headline:

```text
AI systems for climate-heavy operations.
```

Subheadline:

```text
We build domain harnesses where small models, optimizers, simulators,
validators, and reward systems work together to make measurable decisions.
```

Body:

```text
Most AI demos stop at answers. Real-world infrastructure needs decisions:
when to charge, when to conserve, when to dispatch, when to recover, and when
to ask for human intervention.

Capabl Machines builds the operating loop around the model so climate and
infrastructure AI can be tested before it is trusted.
```

Primary CTA:

```text
Explore Energy AI
```

Secondary CTA:

```text
View Hugging Face Work
```

Link targets:

- `Explore Energy AI` -> `/ai/energy`
- `View Hugging Face Work` -> `https://huggingface.co/capabl-machines`

### Core Thesis Section

Title:

```text
The harness is the product.
```

Copy:

```text
A checkpoint alone cannot operate a physical system.

Climate-heavy domains need an environment where actions have consequences, a
schema for valid decisions, tools for physics and constraints, critics for
feedback, datasets for learning, and evals that show whether the system
actually improved.

Our work combines all of that into one deployable AI operating harness.
```

Diagram:

```text
World state
  -> model chooses intent
  -> tools solve constraints
  -> validator checks safety
  -> environment scores outcome
  -> data improves the next model
```

### What We Build

Use four cards.

Card 1:

```text
Environments
Simulated worlds where decisions can be tested repeatedly before touching a
real system.
```

Card 2:

```text
Tool-backed models
Small models that choose strategy, call tools, and stay inside strict schemas.
```

Card 3:

```text
Optimizers and critics
Deterministic solvers, validators, and reward functions that ground AI in
physics, cost, risk, and reliability.
```

Card 4:

```text
Evaluation evidence
Holdout scenarios, baseline comparisons, failure analysis, and public artifacts
that show what worked and what did not.
```

### Case Study Preview

Title:

```text
Case study: GridOps
```

Copy:

```text
GridOps is our energy case study: a simulated Indian community microgrid with
solar, battery, grid imports, diesel backup, heatwaves, outages, and demand
stress.

The breakthrough was not asking a model to output raw dispatch floats. It was
building a strategy-first harness: the model chooses operating intent, an
optimizer handles the physical dispatch, and OpenEnv scores the result.
```

Metric strip:

```text
100% valid strategy JSON
96.09% LP ceiling capture with an untuned 1.5B model + harness
95.81% LP ceiling capture with the released trained selector
3 operating regimes: normal, heatwave, crisis
```

CTA:

```text
Open GridOps Energy Case Study
```

Target:

```text
/ai/energy
```

### Closing Section

Title:

```text
From climate models to climate machines.
```

Copy:

```text
We are interested in AI that can help operate the real world: energy systems,
water systems, farms, factories, logistics networks, robotics, and resilient
infrastructure.

The goal is not to claim that the model is magic. The goal is to build the
whole loop that makes AI useful, measurable, and safe enough to improve.
```

## Page 2: `/ai/energy`

### Page Goal

Position GridOps as the first energy case study and clearly explain the
architecture, result, and honest learning:

```text
The model did not win because fine-tuning alone was magic. The system worked
because the harness gave the model the right role.
```

### Hero

Headline:

```text
Energy AI for community microgrids.
```

Subheadline:

```text
GridOps shows how a small model, strategy schema, optimizer, and OpenEnv
simulation can operate through normal days, heatwaves, and outage crises.
```

Body:

```text
India does not only need more solar panels. It needs intelligence between the
panel, the battery, the grid, the diesel backup, and the people depending on
power.

GridOps is our attempt to build that intelligence layer for community energy:
not as a standalone model, but as a tested operating harness.
```

Primary CTA:

```text
View Model Card
```

Target:

```text
https://huggingface.co/capabl-machines/gridops-strategy-selector-v7
```

Secondary CTA:

```text
Open Demo Artifact
```

Target:

```text
https://huggingface.co/spaces/capabl-machines/gridops-demo
```

Add note below demo CTA:

```text
The demo Space may be paused to avoid unnecessary compute cost.
```

### Problem Section

Title:

```text
Distributed solar creates an operations problem.
```

Copy:

```text
Apartments, societies, campuses, and local energy systems are becoming small
power systems. They may have rooftop solar, batteries, grid imports, diesel
backup, time-of-day pricing, demand spikes, and outage risk.

The hard question is no longer only installation. It is operation.
```

Bullets:

```text
- Charge the battery too early and it may be full before solar peaks.
- Discharge too early and evening demand can create blackouts.
- Avoid diesel too aggressively and crisis reliability suffers.
- Use diesel too freely and cost and emissions rise.
- Shed demand without planning and rebound demand arrives later.
```

### Architecture Section

Title:

```text
The architecture that worked: model for strategy, tools for dispatch.
```

Copy:

```text
Our early direct-action models tried to output exact battery, diesel, and
demand-shedding values. They learned JSON, but the numerical control problem
was too brittle for a small language model.

GridOps v7 changes the interface. The model outputs strict strategy JSON. The
optimizer converts that strategy into a bounded dispatch action. OpenEnv scores
the outcome.
```

Diagram:

```text
Microgrid observation
  -> GridOpsStrategy JSON
  -> causal LP/MPC optimizer
  -> GridOpsAction
  -> OpenEnv score
```

Strategy JSON example:

```json
{
  "mode": "outage_prepare",
  "risk_level": "critical",
  "battery_bias": "preserve",
  "diesel_policy": "allow_if_blackout",
  "shedding_policy": "last_resort"
}
```

### Results Section

Title:

```text
What the results actually say.
```

Copy:

```text
The most important result was not that a LoRA beat every baseline. It did not.

The most important result was that the strategy harness made the task tractable:
even an untuned 1.5B instruction model, when placed inside the right strategy
schema and optimizer loop, captured 96.09% of the LP ceiling with 100% valid
strategy outputs.

That is the Capabl Machines lesson: the harness and the model interface are
often the breakthrough.
```

Metrics:

```text
v5.1 direct-action model: 0.7354 average score
v7 deterministic controller: 0.7907 average score
untuned 1.5B + v7 harness: 0.7911 average score
v7.3 trained selector: 0.7888 average score
full LP ceiling: 0.8233 average score
```

Table:

| System | Avg score | Task 1 normal | Task 2 heatwave | Task 3 crisis | Validity | LP capture |
|---|---:|---:|---:|---:|---:|---:|
| v5.1 direct-action model | 0.7354 | 0.7896 | 0.7681 | 0.6484 | 99.69% action | - |
| v7 deterministic controller | 0.7907 | 0.7995 | 0.8224 | 0.7503 | 100% action | 96.04% |
| untuned Qwen 2.5 1.5B + harness | 0.7911 | 0.7993 | 0.8223 | 0.7517 | 100% strategy | 96.09% |
| v7.3 trained selector | 0.7888 | 0.7993 | 0.8223 | 0.7449 | 100% strategy | 95.81% |
| full LP ceiling | 0.8233 | 0.8372 | 0.8416 | 0.7912 | - | 100% |

### Honest Learning Section

Title:

```text
The trained model was not the whole story.
```

Copy:

```text
We trained SFT and DPO strategy selectors, and they stayed valid and stable.
But the untouched base model with the same harness performed slightly better on
the holdout set.

That changes the positioning in a useful way. GridOps is not evidence that
fine-tuning should be forced into every domain. It is evidence that the right
environment, schema, optimizer, and evaluator can turn even a small general
model into a capable operating component.

When a domain needs post-training, we will train. When the harness is enough,
we will use the simpler system. The product is the reliable operating loop.
```

### Visual Placement

Use these assets:

```text
assets/case_study/capabl_india_microgrid_hero.webp
assets/case_study/gridops_environment_loop.webp
assets/case_study/capabl_energy_journey_infographic.svg
hf_release/capabl_machines/model_assets/gridops_v7_task_scores.png
hf_release/capabl_machines/model_assets/gridops_v7_lp_capture.png
hf_release/capabl_machines/model_assets/gridops_v7_crisis_footprint.png
```

Recommended order:

1. Hero image: Indian community microgrid visual.
2. Architecture: environment loop visual.
3. Results: task score chart.
4. Results: LP capture chart.
5. Crisis section: operational footprint chart.
6. Closing: energy journey infographic.

### Final CTA

Title:

```text
Want to build a climate AI operating harness?
```

Copy:

```text
GridOps is one case study. The same pattern can apply to building energy
systems, water pumps, agriculture, cold chains, EV depots, factories, robotics,
and disaster-resilient infrastructure.
```

CTA:

```text
Talk to Capabl Machines
```

## Site-Wide Notes

Use these phrases:

```text
AI operating harness
climate-heavy operations
model plus tools
environment-tested AI
strategy-first control
optimizer-backed decisions
measurable physical outcomes
```

Avoid these phrases:

```text
fully autonomous grid operator
guaranteed savings
AI replaces operators
the model solved microgrids
fine-tuning is always necessary
```

## Links To Include

- Hugging Face org: https://huggingface.co/capabl-machines
- GridOps model card: https://huggingface.co/capabl-machines/gridops-strategy-selector-v7
- GridOps demo artifact: https://huggingface.co/spaces/capabl-machines/gridops-demo
- Source repo: https://github.com/capabl-machines/gridops

