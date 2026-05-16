# Capabl Machines

Capabl Machines builds domain AI systems that combine small language models,
simulators, optimizers, validators, and real evaluation harnesses.

Our release philosophy:

```text
model + environment + tools + evals + demo
```

## GridOps

GridOps is a strategy-first microgrid operator for Indian community energy
systems. It uses OpenEnv as the world, a causal optimizer as the controller, and
a small learned model as a strategy selector.

- Demo: [GridOps Space](https://huggingface.co/spaces/capabl-machines/gridops-demo)
- Model: [GridOps Strategy Selector v7](https://huggingface.co/capabl-machines/gridops-strategy-selector-v7)

Key result:

```text
v7 deterministic strategy-controller: 96.04% LP ceiling capture
v7.3 learned strategy selector:       100% valid strategy JSON
```
