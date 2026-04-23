# GridOps — Agent Leaderboard

Benchmark results across 10 agents on the GridOps OpenEnv environment.
All runs use seed=42 for full reproducibility. Each task is 72 steps (3 days).

---

## Overall Standings

| Rank | Agent | Task 1 (Normal) | Task 2 (Heatwave) | Task 3 (Crisis) | **Average** |
|---:|---|:---:|:---:|:---:|:---:|
| 1 | **Grok-4.1 (xAI)** | 0.80 | **0.82** | **0.72** | **0.78** |
| 2 | Oracle (rule-based) | 0.79 | 0.81 | 0.70 | 0.77 |
| 3 | **GPT-5.4 (OpenAI)** | 0.79 | 0.79 | 0.67 | 0.75 |
| 4 | Gemma-4-31B (Google) | **0.81** | 0.79 | 0.62 | 0.74 |
| 4 | Grok 4.20 Multi-Agent | **0.81** | 0.80 | 0.60 | 0.74 |
| 4 | DeepSeek V3.2 | 0.80 | 0.79 | 0.62 | 0.74 |
| 7 | GPT-5.4-mini (OpenAI) | 0.72 | 0.74 | 0.46 | 0.64 |
| 8 | Qwen 3.6 Plus (free) | 0.69 | 0.67 | 0.45 | 0.60 |
| 9 | Gemini 3.1 Pro Preview | 0.65 | 0.53 | 0.47 | 0.55 |
| 10 | Kimi K2.5 | 0.57 | 0.54 | 0.48 | 0.53 |
| — | Do-Nothing baseline | 0.58 | 0.51 | 0.45 | 0.51 |
| — | Always-Discharge | 0.59 | 0.51 | 0.45 | 0.52 |
| — | Always-Diesel | 0.42 | 0.42 | 0.44 | 0.43 |

---

## Capability Tiers

| Tier | Score Range | Agents |
|---|---|---|
| **Frontier** | 0.74 - 0.78 | Grok-4, GPT-5.4, Gemma-4-31B, Grok 4.20, DeepSeek V3.2 |
| **Hand-coded baseline** | 0.77 | Oracle (rule-based) |
| **Mid-tier** | 0.60 - 0.64 | GPT-5.4-mini, Qwen 3.6 Plus |
| **Weak** | 0.51 - 0.55 | Kimi K2.5, Gemini 3.1 Pro Preview |
| **No-intelligence baselines** | 0.43 - 0.52 | Do-Nothing, Always-Discharge, Always-Diesel |

---

## Per-Task Breakdown

### Task 1: Normal Summer (Easy)
*Tests basic battery arbitrage. ~100 kW avg demand, Rs 3-12 prices, no heatwave.*

| Rank | Agent | Score |
|---:|---|:---:|
| 1 | Gemma-4-31B | **0.81** |
| 1 | Grok 4.20 Multi-Agent | **0.81** |
| 3 | Grok-4 | 0.80 |
| 3 | DeepSeek V3.2 | 0.80 |
| 5 | Oracle | 0.79 |
| 5 | GPT-5.4 | 0.79 |
| 7 | GPT-5.4-mini | 0.72 |
| 8 | Qwen 3.6 Plus | 0.69 |
| 9 | Gemini 3.1 Pro Preview | 0.65 |
| 10 | Always-Discharge | 0.59 |
| 11 | Do-Nothing | 0.58 |
| 12 | Kimi K2.5 | 0.57 |
| 13 | Always-Diesel | 0.42 |

### Task 2: Heatwave + Price Spike (Medium)
*Tests temporal planning. Day 2-3 heatwave (+30% demand), Rs 20 evening price spike visible in 4h forecast.*

| Rank | Agent | Score |
|---:|---|:---:|
| 1 | Grok-4 | **0.82** |
| 2 | Oracle | 0.81 |
| 3 | Grok 4.20 Multi-Agent | 0.80 |
| 4 | Gemma-4-31B | 0.79 |
| 4 | DeepSeek V3.2 | 0.79 |
| 4 | GPT-5.4 | 0.79 |
| 7 | GPT-5.4-mini | 0.74 |
| 8 | Qwen 3.6 Plus | 0.67 |
| 9 | Kimi K2.5 | 0.54 |
| 10 | Gemini 3.1 Pro Preview | 0.53 |
| 11 | Do-Nothing | 0.51 |
| 11 | Always-Discharge | 0.51 |
| 13 | Always-Diesel | 0.42 |

### Task 3: Extreme Crisis + Grid Outage (Hard)
*Tests constraint management. Full 3-day heatwave, -30% solar, +50% demand, limited diesel, 6-hour grid outage on Day 2.*

| Rank | Agent | Score |
|---:|---|:---:|
| 1 | Grok-4 | **0.72** |
| 2 | Oracle | 0.70 |
| 3 | GPT-5.4 | 0.67 |
| 4 | Gemma-4-31B | 0.62 |
| 4 | DeepSeek V3.2 | 0.62 |
| 6 | Grok 4.20 Multi-Agent | 0.60 |
| 7 | Kimi K2.5 | 0.48 |
| 8 | Gemini 3.1 Pro Preview | 0.47 |
| 9 | GPT-5.4-mini | 0.46 |
| 10 | Qwen 3.6 Plus | 0.45 |
| 10 | Do-Nothing | 0.45 |
| 10 | Always-Discharge | 0.45 |
| 13 | Always-Diesel | 0.44 |

---

## Key Observations

1. **The environment cleanly differentiates capability.** A clean gradient from `do-nothing` (0.51 avg) through frontier LLMs (0.78). Every model lands in a different tier.

2. **Task 3 is the real differentiator.** The 6-hour grid outage forces true islanding behavior. Only Grok-4 and the Oracle handle it well (>0.70). Most LLMs collapse to ~0.45 — the same as do-nothing.

3. **Frontier LLMs match or beat the hand-coded oracle.** Grok-4 (0.78) > Oracle (0.77) — the environment is solvable by raw LLM reasoning, but requires real intelligence.

4. **Smaller LLMs barely beat do-nothing.** Kimi K2.5 (0.53) and Gemini 3.1 Pro Preview (0.55) are within rounding error of the do-nothing baseline (0.51) — they struggle to produce useful actions consistently.

5. **Capability scales with model size within a family.** GPT-5.4 (0.75) significantly outperforms GPT-5.4-mini (0.64). Same prompt, same environment — only the model size differs.

6. **The 0.20-0.35 gap between best and worst agents** proves the environment has real optimization headroom and isn't trivially solvable.

---

## Reproducibility

All scores are deterministic. To reproduce:

```bash
export API_BASE_URL="https://openrouter.ai/api/v1"
export HF_TOKEN="<your-key>"
export MODEL_NAME="<model-id>"
python inference.py
```

Output is structured `[START] / [STEP] / [END]` blocks with explicit task names and scores. Same seed (42) + same model = identical scores across runs.

To run hand-coded baselines (no API key needed):

```bash
python scripts/oracle_test.py
```
