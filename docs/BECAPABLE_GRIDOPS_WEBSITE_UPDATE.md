# becapable.in Website Update Brief: GridOps Release

This document is a ready-to-use content and structure brief for updating
`becapable.in` with the GridOps / Capabl Machines release story.

## Goal

Make visitors understand that Capabl Machines is not just fine-tuning models.
We are building AI systems for real-world physical operations: environment,
model, optimizer, evaluator, and demo evidence together.

GridOps should be positioned as the flagship case study:

```text
AI for sustainable infrastructure operations.
Small model for judgment. Tools for physics. OpenEnv for truth.
```

## Primary Links

Use these links throughout the page:

- Model: https://huggingface.co/capabl-machines/gridops-strategy-selector-v7
- Demo Space, paused to save cost: https://huggingface.co/spaces/capabl-machines/gridops-demo
- Source branch/repo: https://github.com/capabl-machines/gridops
- Capabl Machines HF org: https://huggingface.co/capabl-machines

## Homepage Hero Copy

Recommended homepage hero:

```text
Capabl Machines

AI systems for the real world.

We build small, specialized models that operate inside environments, use tools
for physics and constraints, and are evaluated by what they actually do.
```

CTA buttons:

```text
Explore GridOps
View Hugging Face Release
```

Short supporting line:

```text
Our first release, GridOps, explores AI-assisted operation for Indian community
microgrids: solar, battery, grid, diesel, demand response, cost, reliability,
and sustainability in one tested loop.
```

## Homepage Section: Why Capabl Machines

Section title:

```text
From model demos to operating machines
```

Body:

```text
Most AI demos stop at answers. Physical systems need decisions.

Energy, water, agriculture, robotics, logistics, and climate infrastructure all
need models that can observe state, choose intent, call tools, respect
constraints, and learn from executable feedback.

Capabl Machines builds that full loop:
environment -> data -> model -> tools -> evaluation -> release.
```

Use a simple diagram:

```text
World state
  -> small model chooses intent
  -> deterministic tools execute safely
  -> environment scores the outcome
  -> data improves the next model
```

## GridOps Case Study Page

Suggested route:

```text
/work/gridops
```

Suggested SEO title:

```text
GridOps: AI for Community Microgrid Operations | Capabl Machines
```

Suggested meta description:

```text
GridOps is a strategy-first AI system for simulated Indian community microgrids,
combining a small learned model, causal optimizer, OpenEnv environment, and
evaluation on cost, reliability, diesel use, and blackout reduction.
```

## GridOps Page Hero

Title:

```text
GridOps
```

Subtitle:

```text
Teaching a small AI system to operate a community microgrid through normal
days, heatwaves, and grid-outage crises.
```

Hero body:

```text
India does not only need more solar panels. It needs intelligence between the
panel, the battery, the grid, the diesel backup, and the people depending on
power.

GridOps is our attempt to build that intelligence layer: a small learned
strategy selector, a deterministic optimizer, and an OpenEnv world where every
decision is scored by cost, reliability, diesel use, and blackout impact.
```

Primary CTA:

```text
View Model Card
```

Secondary CTA:

```text
Read Technical Release
```

## Problem Section

Title:

```text
The new problem created by distributed solar
```

Body:

```text
Apartments, societies, campuses, and local energy systems are beginning to look
like small power plants. They may have rooftop solar, batteries, grid imports,
diesel backup, time-of-day pricing, demand spikes, and outage risk.

The hard question is operational:

When should the system charge the battery, preserve it, discharge it, run
diesel, or ask users to reduce demand?

Bad control turns clean infrastructure into higher bills, battery misuse,
blackouts, or unnecessary diesel. Good control makes the same hardware more
valuable.
```

Use bullets:

```text
- Solar is free, but only available during the day.
- Batteries help, but only if they are preserved for the right hours.
- Diesel protects reliability, but increases cost and emissions.
- Demand response can help, but rebounds later.
- Grid outages turn local planning into a survival problem.
```

## System Section

Title:

```text
The architecture that finally worked
```

Body:

```text
We first tried to make the model output exact battery, diesel, and demand
shedding numbers directly. That was the wrong abstraction.

The model could learn JSON, but it was being asked to become the entire grid
engineer.

GridOps v7 uses a better split:
```

Diagram:

```text
Microgrid state
  -> learned strategy selector
  -> causal optimizer
  -> bounded GridOps action
  -> OpenEnv score
```

Copy below diagram:

```text
The model learns operating intent. The optimizer handles physics and
constraints. OpenEnv judges the result.
```

## Results Section

Title:

```text
Release results
```

Lead:

```text
The strongest deployable system is the deterministic v7 strategy-controller.
The learned v7.3 selector nearly matches it while producing 100% valid strategy
JSON on the holdout set.
```

Metric cards:

```text
100%
valid strategy outputs

96.04%
LP ceiling capture by strategy-controller

95.81%
LP ceiling capture by learned selector

3
operating regimes tested: normal, heatwave, crisis
```

Table:

| System | Avg score | Task 1 | Task 2 | Task 3 | Validity | LP capture |
|---|---:|---:|---:|---:|---:|---:|
| v5.1 direct action model | 0.7354 | 0.7896 | 0.7681 | 0.6484 | 99.69% action | - |
| v7 deterministic controller | 0.7907 | 0.7995 | 0.8224 | 0.7503 | 100% action | 96.04% |
| v7.3 learned selector | 0.7888 | 0.7993 | 0.8223 | 0.7449 | 100% strategy | 95.81% |
| Full LP ceiling | 0.8233 | 0.8372 | 0.8416 | 0.7912 | - | 100% |

## Engineering Journey Section

Title:

```text
What we learned
```

Body:

```text
The breakthrough was not one checkpoint. It was finding the right interface
between model and machine.
```

Timeline:

```text
v4    direct action SFT from reasoning traces
v5    causal LP teacher imitation
v5.1  crisis repair continuation
v6    tool-corrected action SFT, not promoted
v6.1  clean LP-critic action SFT, not promoted
v7    strategy-first harness
v7.1  SFT strategy selector
v7.2  DPO preference tuning
v7.3  crisis-weighted DPO release checkpoint
```

Quote block:

```text
The model does not need to become the entire operator. It needs to learn the
operating language that lets deterministic tools act safely.
```

## Sustainability / Impact Section

Title:

```text
Why this matters beyond GridOps
```

Body:

```text
GridOps is one case study in a broader thesis: useful AI for physical systems
should be trained and evaluated inside the world it claims to operate.

The same pattern can apply to apartment energy systems, water pumps, cold
chains, EV charging depots, factory energy use, farm irrigation, robotics, and
disaster-resilient infrastructure.
```

Closing line:

```text
Capabl Machines is building AI that can operate real-world systems, not just
talk about them.
```

## Visual Assets To Use

Preferred existing visuals from the repo:

```text
assets/case_study/capabl_india_microgrid_hero.webp
assets/case_study/gridops_environment_loop.webp
assets/case_study/capabl_energy_journey_infographic.svg
hf_release/capabl_machines/model_assets/gridops_v7_task_scores.png
hf_release/capabl_machines/model_assets/gridops_v7_lp_capture.png
hf_release/capabl_machines/model_assets/gridops_v7_crisis_footprint.png
```

Suggested placement:

1. Hero: `capabl_india_microgrid_hero.webp`
2. Architecture section: `gridops_environment_loop.webp`
3. Results section: `gridops_v7_task_scores.png`
4. Results section: `gridops_v7_lp_capture.png`
5. Impact section: `capabl_energy_journey_infographic.svg`

## Navigation Update

Add top-nav item:

```text
Work
```

Dropdown or page links:

```text
GridOps
CarbonAlpha
Training Process
```

If keeping it simple:

```text
Home | Work | Process | Contact
```

## Homepage Project Card

Card title:

```text
GridOps
```

Card description:

```text
A strategy-first AI operator for Indian community microgrids. The system
combines a small learned model, causal optimizer, and OpenEnv evaluation across
normal, heatwave, and crisis scenarios.
```

Stats on card:

```text
100% valid strategy outputs
96.04% LP ceiling capture
OpenEnv + optimizer + learned selector
```

CTA:

```text
View case study
```

## LinkedIn Launch Copy

Short post:

```text
India does not only need more solar panels.

It needs intelligence between the panel, the battery, the grid, the diesel
backup, and the people depending on power.

We built GridOps: a strategy-first AI system for simulated Indian community
microgrids.

The model does not directly output dispatch floats. It chooses operating
strategy. A causal optimizer handles physics and constraints. OpenEnv scores
the result on cost, reliability, diesel, and blackout impact.

Results:
- 100% valid strategy outputs
- 96.04% LP ceiling capture by the strategy-controller
- tested across normal, heatwave, and grid-outage crisis regimes

This is the kind of AI we want to build at Capabl Machines: small models,
real environments, tool-backed execution, measurable impact.
```

Long post opening:

```text
Most AI demos stop at answers. Physical systems need decisions.

For the last few weeks, we have been building GridOps: an AI operating pattern
for community microgrids in India.
```

## Tone Guidelines

Use:

```text
simulated
strategy-controller
toward deployable local energy intelligence
physical systems AI
environment-tested
tool-backed
```

Avoid:

```text
solves India’s energy problem
autonomous grid operator
ready for live grid deployment
guaranteed cost savings
```

## Implementation Checklist

- Add `/work/gridops` page.
- Add GridOps card on homepage.
- Add model/repo/HF org links.
- Add three charts from `hf_release/capabl_machines/model_assets`.
- Keep the HF Space link visible but mark it as demo artifact, not always-on.
- Include a note that the Space may be paused to avoid unnecessary compute cost.
- Add the final metrics table exactly as shown above.
- Use the architecture diagram early; it is the core idea.
- End with the broader Capabl Machines thesis.

