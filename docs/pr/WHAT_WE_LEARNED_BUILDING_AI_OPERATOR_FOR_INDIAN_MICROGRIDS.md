# What We Learned Building an AI Operator for Indian Microgrids

India does not only need more solar panels. It needs intelligence between the
panel, the battery, the grid, the diesel backup, and the people depending on
power.

That was the starting point for GridOps.

We wanted to build a small AI system that could operate a simulated community
microgrid through normal days, heatwaves, and outage crises. The system had to
make decisions that mattered: when to charge the battery, when to preserve it,
when to discharge it, when to allow diesel, and when reliability mattered more
than cost.

The original instinct was simple: train a model to output the final action.

```json
{"battery_dispatch": 0.0, "diesel_dispatch": 0.0, "demand_shedding": 0.0}
```

That was the wrong abstraction.

## The Direct Model Was Too Brittle

We trained models to emit dispatch values directly. They learned the JSON
format. They learned some battery behavior. They improved over do-nothing
baselines.

But they were still brittle where the real problem mattered most.

In crisis settings, a small language model was being asked to do too much at
once:

- read the microgrid state;
- infer the time context;
- understand solar and demand;
- reason about outage risk;
- preserve battery state-of-charge;
- manage limited diesel;
- avoid invalid outputs;
- choose three exact numeric dispatch values.

That is not just language. That is constrained control.

The model could imitate examples, but it was not reliably becoming the grid
engineer.

## The Breakthrough Was Changing The Interface

The key shift was to stop asking the model for exact dispatch floats.

Instead, we made the model choose operating strategy:

```json
{
  "mode": "outage_prepare",
  "risk_level": "critical",
  "battery_bias": "preserve",
  "diesel_policy": "allow_if_blackout",
  "shedding_policy": "last_resort"
}
```

Then a causal optimizer converted that strategy into the final bounded action.
OpenEnv scored the result.

The loop became:

```text
microgrid observation
  -> strategy JSON
  -> optimizer
  -> validated dispatch action
  -> OpenEnv score
```

That changed everything.

The model no longer had to become the entire operator. It only had to choose
the right operating intent. The optimizer handled physics and constraints. The
validator handled schema safety. The environment judged the result.

## The Unexpected Result

We trained SFT and DPO versions of the strategy selector. They were stable and
produced 100% valid strategy JSON on the holdout set.

But then we ran the uncomfortable sanity check: the untouched base
`Qwen/Qwen2.5-1.5B-Instruct` model with the same strategy prompt and controller.

It performed slightly better than the trained adapter:

| System | Avg score | Task 1 normal | Task 2 heatwave | Task 3 crisis | LP capture |
|---|---:|---:|---:|---:|---:|
| v5.1 direct-action model | 0.7354 | 0.7896 | 0.7681 | 0.6484 | - |
| v7 deterministic controller | 0.7907 | 0.7995 | 0.8224 | 0.7503 | 96.04% |
| untuned Qwen 2.5 1.5B + harness | 0.7911 | 0.7993 | 0.8223 | 0.7517 | 96.09% |
| v7.3 trained selector | 0.7888 | 0.7993 | 0.8223 | 0.7449 | 95.81% |
| full LP ceiling | 0.8233 | 0.8372 | 0.8416 | 0.7912 | 100.00% |

If the only goal was "fine-tune a model that beats base," this was not the win
we expected.

But from an engineering perspective, it was more valuable.

It showed that the strategy harness was doing the real work. Once the interface
was right, even a small general model could become a useful operating component.

## The Lesson

The lesson is not that training is useless. The lesson is that training should
not be used to compensate for a bad interface.

For physical systems, the first question should not be:

```text
How do we make the model smarter?
```

It should be:

```text
What role should the model play inside the operating loop?
```

In GridOps, the answer was not "make the model the controller." The answer was:

```text
model for strategy
optimizer for physics
validator for safety
environment for truth
metrics for accountability
```

That is the pattern we now care about.

## Why This Matters For India

Apartments, societies, campuses, farms, factories, and local energy systems are
starting to look like small power systems. They may have rooftop solar,
batteries, diesel backup, variable tariffs, demand spikes, and outage risk.

The hardware is becoming distributed.

The intelligence layer has to become distributed too.

Most places will not have a full-time expert energy operator watching every
microgrid. But they can have a tested AI operating harness that:

- observes the state;
- chooses the operating strategy;
- uses deterministic tools for dispatch;
- validates safety;
- records outcomes;
- improves over time.

GridOps is a first case study in that direction.

## What We Would Build Next

The next step is not another model checkpoint for its own sake.

The next step is a public benchmark app where different models and strategies
can be tested on the same GridOps environment:

```text
model -> strategy -> optimizer -> action -> OpenEnv score
```

Then we can compare small models, frontier models, rule-based strategies, and
trained adapters on the same operational metrics:

- cost;
- blackout kWh;
- diesel kWh;
- valid action rate;
- crisis recovery;
- LP ceiling capture.

That is how climate AI should be evaluated: not by how convincing the answer
sounds, but by what the decision does inside a world.

## Closing

GridOps did not teach us that a fine-tuned checkpoint is always the product.

It taught us something more useful:

> For climate-heavy operations, the harness is often the product.

The model matters. But the environment, optimizer, validator, critic, dataset,
and evaluation loop matter just as much.

That is the direction for Capabl Machines.

