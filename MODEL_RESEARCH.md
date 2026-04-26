# CarbonAlpha model rescue notes

Last updated: 2026-04-25

## Current safe model

- `77ethers/CarbonAlpha/v6_sft_only_v2`
- Qwen3-4B-Instruct + LoRA SFT-only
- Holdout: 5/5 valid, mean regret `+0.034`, beats equal-weight baseline on 3/5 seeds
- Do not overwrite this path.

## Active experiment

- `unsloth/Qwen3-4B-Base`
- Target upload folder: `77ethers/CarbonAlpha/grpo_qwen3_4b_base_smoke_v1`
- Reason: matches Unsloth's official Qwen3 4B GRPO recipe: Base model, custom chat template, SFT pre-formatting, vLLM rollouts.
- Smoke result so far: rollout mechanics are healthy, with no 1-token collapse. The remaining issue is verbosity and failure to finish valid JSON inside the token cap.

## Best pivot if Qwen3 Base stays too verbose

### 1. Qwen2.5-7B-Instruct

Recommended repo options:

- `Qwen/Qwen2.5-7B-Instruct`
- `unsloth/Qwen2.5-7B-Instruct`
- `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` if job memory becomes tight

Why it is the best rescue candidate:

- Strong instruction following and JSON compliance.
- Large enough to improve macro-event reasoning over 4B, while still realistic on an L40S with QLoRA.
- Avoids the Qwen3 Base custom-template `<think>` bootstrapping problem.
- Good fit for an SFT-heavy strategy, with optional short GRPO after format stability is proven.

HF Hub signal observed on 2026-04-25:

- `Qwen/Qwen2.5-7B-Instruct`: ~12.16M downloads, 1237 likes.
- `unsloth/Qwen2.5-7B-Instruct`: available as an Unsloth-compatible path.

Run result on 2026-04-25:

- Run label: `sft_qwen25_7b_curriculum400_v1`
- Base: `unsloth/Qwen2.5-7B-Instruct`
- Traces: `sft_traces/curriculum_400_e80_m160_h160.jsonl`
- Recipe: QLoRA SFT, r=16, alpha=16, 220 steps, effective batch size 4
- Artifact: `77ethers/CarbonAlpha/sft_qwen25_7b_curriculum400_v1`
- Generation sanity: 5/5 valid actions, 5/5 closed `<think></think>`, concise 127-169 token samples
- Holdout: 5/5 valid, mean regret `+0.02796`, beats baseline on 3/5 seeds
- Decision: strong demo/format candidate, but it does not beat v6 SFT's mean holdout regret bar of `+0.034`.

## Stronger but riskier

### 2. Qwen3-8B / Qwen3-8B-Base

Recommended repo options:

- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-8B-Base`
- `unsloth/Qwen3-8B-Base`

Why to consider it:

- Better reasoning capacity than Qwen3-4B.
- Same family as the current work, so behavior may transfer.

Risks:

- More expensive and slower.
- Could amplify the same verbosity and `<think>` completion-control problems seen with Qwen3 Base.
- Base variant still needs careful SFT pre-formatting before it can answer reliably.

HF Hub signal observed on 2026-04-25:

- `Qwen/Qwen3-8B`: ~8.83M downloads, 1065 likes.
- `Qwen/Qwen3-8B-Base`: ~523K downloads, 99 likes.

## Fast stable fallback

### 3. Llama-3.2-3B-Instruct

Recommended repo options:

- `meta-llama/Llama-3.2-3B-Instruct`
- `unsloth/Llama-3.2-3B-Instruct`

Why to consider it:

- Fast, stable, likely good at concise instruction-following and JSON.
- Useful as a sanity baseline if Qwen training behavior remains unstable.

Risks:

- Weaker macro-news reasoning than Qwen2.5-7B or Qwen3-8B.
- May need more traces to reach the same financial allocation quality.

HF Hub signal observed on 2026-04-25:

- `meta-llama/Llama-3.2-3B-Instruct`: ~2.92M downloads, 2112 likes.

## Not first choice

### Phi-4-mini-reasoning

- Good reasoning model, but likely to overproduce reasoning for this constrained JSON task.
- Smaller Hub/adaptation signal for the current Unsloth workflow.

### Gemma 3 4B IT

- Strong general model and clean size class.
- Less directly aligned with the current Unsloth/Qwen rescue path and may require more integration time.

## Recommended decision ladder

1. Continue current `unsloth/Qwen3-4B-Base` experiment with shorter curriculum traces and a reward/format gate that favors finished JSON.
2. If Qwen3 Base still fails the 3/5 valid-completion gate after trace/verbosity fixes, run an SFT-first smoke on `Qwen2.5-7B-Instruct`.
3. If Qwen2.5-7B beats v6 SFT on holdout or produces clearly better demo answers, ship that.
4. If none of the rescue paths beat `v6_sft_only_v2`, ship the current v6 SFT model and document GRPO as attempted but unstable.

## Instruct models and `<think></think>` control

For instruct models, prefer completions that contain the full closed reasoning block plus JSON:

```text
<think>
Short causal reasoning: shock, sector exposure, hedge, carbon constraint.
</think>
{"weights": {...}, "rationale": "..."}
```

This is different from the Qwen3 Base GRPO path, where the chat template appends only the opening `<think>` to the prompt. Instruct models should learn the full output contract directly from SFT.

Most controllable instruct candidates:

1. `Qwen/Qwen2.5-7B-Instruct`
   - Best format-control pivot.
   - Likely to close `</think>` and emit valid JSON more reliably than Qwen3 Base.
   - Heavier than 4B, but realistic on L40S with QLoRA.
2. `Qwen/Qwen3-4B-Instruct` or `Qwen/Qwen3-8B`
   - Closer to native reasoning style.
   - Stronger visible reasoning, but higher verbosity risk.
   - SFT-only works; GRPO needs careful smoke gates.
3. `meta-llama/Llama-3.2-3B-Instruct`
   - Compact, fast, stable formatting baseline.
   - Can learn the tag contract with SFT, but likely weaker on macro-finance reasoning.
4. `microsoft/Phi-4-mini-reasoning`
   - Reasoning-native, but likely to require aggressive brevity and finish rewards.
5. `google/gemma-3-4b-it`
   - Viable small instruct option, but less aligned with the current Qwen/Unsloth training path.

## GRPO reward shaping update

The Base GRPO smoke should not rely on `format + regret` only. That lets a rollout be financially scoreable while still being unusable in the demo.

The isolated Base script now uses five reward components:

- `format`: existing environment format reward for `<think>` plus parseable JSON.
- `structure`: rewards exactly one closed `<think>...</think>` block followed by JSON, and penalizes markdown fences or unfinished thought tags.
- `brevity`: rewards compact reasoning, roughly 45-180 words in the think block and 350-1200 total characters; penalizes rambling beyond the cap.
- `action`: rewards bounded, valid, non-degenerate `PortfolioAction` JSON with legal intervention ranges and known `tech_bet`.
- `regret`: primary environment reward for beating equal-weight baseline.

Smoke gate still requires valid sampled completions, non-collapsed completion lengths, nonzero grad norm, and nonzero reward variance. Reward-std detection now checks the structural/action rewards too, not only regret.
