# Model Training Checklist

## Discovery

- Define the user-facing task in one paragraph.
- Identify whether the output must be executable, parseable, ranked, judged, or free-form.
- List constraints that must never be violated.
- Pick the metric a simple baseline must beat.
- Decide where artifacts will live before training begins.

## Harness / Environment

- Implement a parser and validator before model training.
- Implement deterministic rewards or scoring where possible.
- Add a tiny smoke test for `reset`, `step`, parse, reward, and holdout eval.
- Keep prompt construction in one module used by trace generation, training, evaluation, and demo.

## Dataset

- Start from a schema example row.
- Add `id`, `difficulty`, source/seed metadata, prompt, completion, and raw structured fields.
- Split into easy/medium/hard or equivalent curriculum tiers.
- Generate traces in small batches.
- Validate every row and write failures to a separate file.
- Sample 10 rows manually before SFT.

## SFT

- Choose model by task fit, context length, license, GPU memory, tokenizer/chat-template fit, and existing training ecosystem.
- Start with QLoRA unless full fine-tuning is justified.
- Save to a new model subfolder.
- Run generation sanity checks before and after SFT.
- Run holdout eval and preserve the SFT model as a fallback.

## RL / GRPO

- Start from the best SFT checkpoint, not from a fragile base model, unless intentionally testing base-model RL.
- Use a new run label/subfolder.
- Begin with 5-10 smoke steps.
- Log aggregate reward and component rewards.
- Gate on completion length, parse rate, gradient norm, reward variance, and holdout result.
- Scale only after smoke metrics are healthy.

## Evaluation

- Keep holdout seeds/data fixed and outside training.
- Use a manual challenge set with at least 10 examples.
- Report validity rate, objective metric, baseline comparison, and known failure modes.
- Include example completions from base, SFT, and RL models when useful.

## Publishing

- README links: demo Space, model repo, dataset repo, Colab, blog/video, plots, logs.
- Model card: lineage, intended use, limitations, metrics, evidence plots, raw logs.
- Colab: read-only checks by default; training/eval reruns guarded by booleans.
- HF Space: runnable and public if required.
- Avoid large video files in repos; link public video URLs instead.

## Safety Rules

- Never overwrite the best known model.
- Never hide failed experiments; summarize what failed and why.
- Never trust a single metric for model quality.
- Never ship RL if it fails the smoke gate.
- Never let demo code use a different prompt/schema than training without documenting the change.
