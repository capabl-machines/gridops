"""One-time deep-research call to Gemini with full context + Google grounding.

Invoked once before we start building. Feeds Gemini:
- The full v0.6 design doc
- All the strategic decisions we've made (Gemma 4 E4B, TRL v1.0 GRPO with
  DAPO default, Unsloth 4-bit, 3-phase curriculum, SFT warm-start)
- The remaining open risks we want stress-tested

With Google Search grounding enabled so Gemini fact-checks against
current (April 2026) web state on library versions, Gemma 4 support,
Colab T4 throughput, recent RL-reasoning papers, etc.

Usage: python gemini_deep_research.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from google import genai
from google.genai import types


# ── Load .env ───────────────────────────────────────────────────────
_env_path = Path(__file__).parent / '.env'
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())


# ── Load the design doc as context ──────────────────────────────────
DESIGN_DOC = (Path(__file__).parent / 'portfolio_env_design.md').read_text()


SYSTEM_PROMPT = """\
You are a senior ML engineering strategist with deep expertise in:
(a) Hugging Face TRL v1.0 and GRPO/DAPO-style RL training,
(b) Unsloth 4-bit QLoRA, Gemma 4 family, and memory-efficient training on consumer GPUs,
(c) Reward engineering for RLVR with verifiable rewards,
(d) Colab / T4 practical constraints for 48-hour hackathon windows,
(e) Meta PyTorch Hackathon judging criteria and OpenEnv ecosystem.

You have been given the full v0.6 design doc for a Round 2 Meta PyTorch
Hackathon submission (Climate-Stressed Portfolio Manager — a
Reasoning-Under-Constraints OpenEnv). Ekansh + brother's team ships
April 25-26 in Bangalore.

You have Google Search access via grounding. USE IT. Verify claims
about library support, model capabilities, and recent research. Don't
speculate — search and cite.

RULES:
1. No filler, no "great question." Engineering-grade analysis only.
2. If you find the plan contradicts current (April 2026) library state
   or recent research, say exactly what's wrong and how to fix it.
3. Cite sources (URIs) when you make a factual claim about a library
   version, API surface, or paper.
4. Quantify when possible — token budgets, VRAM, iter times.
5. Call out blind spots we haven't considered.
"""


RESEARCH_QUESTION = f"""\
I need one deep research pass on our v0.6 plan before we start building
tonight. Use Google grounding to verify everything — don't speculate on
library state or recent papers.

## PART A — Full design doc (v0.6)

{DESIGN_DOC}

## PART B — Training stack decisions

We are committing to:
- **Model**: `google/gemma-4-E4B-it` (4B effective params, 4-bit QLoRA via Unsloth)
- **Framework**: TRL v1.0 `GRPOTrainer` with `loss_type="dapo"` default, `beta=0.0`
- **Efficiency**: Unsloth 4-bit QLoRA; skip Muon optimizer (integration risk);
  use AdamW-8bit as the optimizer
- **Compute**: Colab T4 (16GB VRAM)
- **Curriculum**: 3 phases (easy only → + ambiguous → + hard + interventions)
  with entry-criterion gates on median regret
- **SFT warm-start**: 15 expert `<think>` traces generated via GPT-4/Claude/Gemini
  + brother's domain reasoning, SFT ~50 steps before Phase 3 GRPO
- **Reward funcs**: 5 composite (format, regret-vs-equal-weighted-baseline on
  real inflation-adjusted returns, Sharpe 0.3×, non-linear carbon penalty
  phase-weighted, drawdown)
- **Plus DAPO technique #4**: overlong-think reward shaping implemented as a
  6th reward function (penalize `<think>` blocks > 150 tokens)
- **Episode**: 12 quarters, 5 shocks per episode from 17-shock pool with
  3-tier taxonomy (easy/ambiguous/hard), regime coherence, path-dep state

## PART C — Research questions — deep + grounded

Please address each of the following. For each, search current state,
cite sources, give a verdict + concrete recommendation.

### 1. Gemma 4 E4B + Unsloth + TRL GRPO — integration reality check
Is this combination actually working well in stable releases as of
April 2026? Search for known issues, GitHub issues, recent benchmark
reports. Specifically:
- Does Unsloth's GRPO notebook for Gemma 4 E4B run cleanly end-to-end
  on T4 without known crashes?
- Does `loss_type="dapo"` in TRL v1.0 behave well with Gemma's chat
  template?
- Any reports of tokenizer / special-token issues specific to Gemma 4?

### 2. Realistic Colab T4 throughput for our exact setup
We projected 300 tok/s for Llama 3.1 8B, 600 tok/s for Gemma 4 E4B
(both Unsloth 4-bit QLoRA during GRPO rollouts). Ground-truth check.
Search for benchmarks / user reports of actual T4 throughput with
Unsloth + Gemma 4 E4B + GRPOTrainer. Also: does T4 support FP8
quantization or is that A100+ only?

### 3. DAPO technique #4 (Overlong Reward Shaping) — implementation
recipe
Search for open implementations or the original DAPO paper's recipe.
We plan to add it as a 6th reward function that applies a smooth
penalty to completions with `<think>` blocks exceeding N tokens.
Recommend the exact shape of the penalty (linear, quadratic, sigmoid)
and the threshold N based on recent research.

### 4. SFT warm-start: does it actually help GRPO on 4B models?
Search for papers / blog posts that quantify the benefit of SFT
warm-start before GRPO specifically on small (<=4B) models. DeepSeek-R1
used it at 32B+ — does the same recipe carry to 4B? How many expert
traces is the threshold for statistically significant warm-start
benefit?

### 5. Reward hacking failure modes specific to portfolio / financial RL
Search for recent (2025-2026) papers or blog posts on reward hacking
in financial / portfolio RL envs. Our composite reward (§9) has 5
components. Are there exploits specific to this domain that we haven't
considered? Especially around inflation-real-return calculations and
put-hedge triggers.

### 6. Recent Meta PyTorch hackathon winning submissions (OpenEnv)
Search for any public information about past Meta / HF RL hackathon
winners, their submission patterns, what judges highlighted as
impressive. We're banking hard on "reasoning under ambiguity +
interventions" as our innovation angle — is there precedent for
similar angles winning?

### 7. Gemma 4 tokenizer + chat template gotchas
Our action parsing relies on extracting `<think>...</think>` and a JSON
block. Search for Gemma 4's native chat template and whether our
`<think>` tag strategy conflicts with any special tokens. Does Gemma 4
have a built-in reasoning token?

### 8. Any overlooked v0.6 design flaws
Do a fresh independent read of the design doc (§0-19) and flag
anything we've missed. Focus on: reward reward exploits we haven't
anticipated, training instability risks, demo narrative weaknesses,
HF Spaces deployment gotchas for OpenEnv-compliant environments.

### 9. Publication bias / "this looks like research we should know about"
Search arXiv and HuggingFace papers from Q1 2026 that specifically
relate to: (a) causal reasoning in LLMs trained via RL, (b)
long-horizon credit assignment with GRPO, (c) environments for
training agentic financial reasoning. Flag anything published in the
last 60 days that we should either cite or be aware of.

## PART D — Deliverable

Please produce a STRUCTURED report with:
- ✅ confirmed claims (with citations)
- ⚠️ risks found (with severity: HIGH / MEDIUM / LOW + mitigation)
- 🔁 suggested changes to v0.6 before we start building
- 📚 papers/repos worth reading tonight

Be ruthless. I'd rather scrap a section tonight than fail at hour 30.
"""


def ask_gemini_with_grounding(prompt: str) -> str:
    client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))

    grounding_tool = types.Tool(google_search=types.GoogleSearch())

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[grounding_tool],
        thinking_config=types.ThinkingConfig(thinking_level='HIGH'),
    )

    content = types.Content(
        role='user',
        parts=[types.Part.from_text(text=prompt)],
    )

    full_text: list[str] = []
    print('─' * 70)
    print('  Gemini deep research — grounded, thinking HIGH')
    print('─' * 70)
    print()

    for chunk in client.models.generate_content_stream(
        model='gemini-3-pro-preview',
        contents=[content],
        config=config,
    ):
        if chunk.text:
            full_text.append(chunk.text)
            print(chunk.text, end='', flush=True)

    print()
    print()
    print('─' * 70)

    # Print grounding metadata if available
    try:
        # last chunk's candidates carry metadata
        for chunk in [chunk]:  # keep the last iteration's value
            md = chunk.candidates[0].grounding_metadata
            if md and md.grounding_chunks:
                print('  Grounding sources:')
                for i, c in enumerate(md.grounding_chunks, 1):
                    if hasattr(c, 'web') and c.web:
                        print(f'    [{i}] {c.web.title} — {c.web.uri}')
    except Exception as exc:
        print(f'  (grounding metadata not exposed: {exc})')

    return ''.join(full_text)


if __name__ == '__main__':
    out = ask_gemini_with_grounding(RESEARCH_QUESTION)
    # Save transcript for later review
    out_path = Path(__file__).parent / 'gemini_deep_research_output.md'
    out_path.write_text(f'# Gemini deep research — portfolio_env v0.6\n\n{out}\n')
    print(f'\nTranscript saved to {out_path}')
