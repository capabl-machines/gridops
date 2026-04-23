# Gemini deep research — portfolio_env v0.6

Here is the engineering-grade analysis of the v0.6 design doc. 

### ✅ Confirmed Claims (Grounding Verification)
*   **Gemma 4 E4B architecture & fit:** Gemma 4 E4B is a dense edge model with "effective 4B" active parameters utilizing Per-Layer Embeddings (PLE). It natively supports function calling, thinking modes, and fits comfortably on a T4 (16GB VRAM) when quantized to 4-bit. 
*   **TRL DAPO support:** TRL v1.0 natively supports `loss_type="dapo"` in `GRPOConfig`, which includes token-level loss normalization and decoupled clipping.
*   **DAPO Overlong Reward Shaping:** TRL supports this natively. It can be imported directly via `from trl.rewards import get_soft_overlong_punishment`.
*   **SFT Warm-start efficacy:** Recent competitive RL implementations (e.g., G-RaR, Kaggle AIMO3 winners) confirm that cold-starting GRPO on a base model causes slow convergence and degenerate structure collapse. SFT warm-start drastically improves sample efficiency by constraining the initial policy distribution.

---

### ⚠️ Risks Found

**1. `GRPOTrainer` Multi-Step MDP Incompatibility (Severity: CRITICAL)**
*   **The Issue:** TRL's standard `GRPOTrainer` is a contextual bandit algorithm designed for single-turn prompt-completion-reward pipelines. Your v0.6 spec defines a 12-step Stateful MDP (Quarter 1 → observe → Quarter 2 → observe). `GRPOTrainer` *cannot natively inject external environment observations mid-generation*. 
*   **The Impact:** If Ekansh boots `GRPOTrainer` tonight, he will realize he cannot step the OpenEnv midway through a sequence. 
*   **Mitigation:** You must either (A) flatten the episode so the agent receives all 12 macro-headlines upfront and outputs a 12-step JSON trajectory in one shot (sacrificing the "uncertainty" mechanic), or (B) write a custom PyTorch rollout loop that accumulates log-probs interactively and applies the DAPO loss at the terminal state. 

**2. Gemma 4 Tokenizer vs. `<think>` Tags (Severity: HIGH)**
*   **The Issue:** Your action parser and `r_format` reward look for `<think>` and `</think>`. Gemma 4 **does not use these tags**. Gemma 4 uses `<|channel>thought\n` to open reasoning blocks and `<channel|>` to close them.
*   **The Impact:** The agent will fail `r_format`, receive 0 reward, and the training curve will flatline immediately.
*   **Mitigation:** Configure the tokenizer using `get_chat_template(tokenizer, chat_template="gemma-4-thinking")` and `enable_thinking=True`. Update the regex in `r_format` to match `<|channel>thought\n` and `<channel|>`.

**3. T4 Inference Math is Hallucinated (Severity: HIGH)**
*   **The Issue:** You projected 300 tok/s generation for a 4B/8B model on a Colab T4. A Turing-architecture T4 does *not* support FP8 and relies on 4-bit QLoRA. Actual generation throughput for a 4B model on a T4 is ~30–45 tok/s, not 300. 
*   **The Impact:** Generating 432 rollouts (at ~250 tokens each) equals 108,000 tokens per iteration. At 40 tok/s, that is **45 minutes per iteration**. Phase 3 alone (80 iters) will take ~60 hours, blowing your 48-hour window.
*   **Mitigation:** Shrink the group size ($G=4$ instead of $6$). You may also need to downgrade from E4B to E2B (Effective 2B) to double your inference speed if you want to finish Phase 3 onsite. 

**4. The `put_hedge` Reward Hack (Severity: HIGH)**
*   **The Issue:** §7.1 states `put_hedge` caps downside at -5% if the *quarter's worst-asset return < -15%*. 
*   **The Exploit:** An RL agent will realize it can buy the put, intentionally allocate 1% to the most volatile asset (TECH), and hope it tanks >15%. This triggers the condition, magically erasing losses from bad allocations in the other 99% of the portfolio.
*   **Mitigation:** The put hedge trigger must be based on the **portfolio NAV drop**, not a single asset's drop. 

**5. SFT Warm-Start Collapse (Severity: MEDIUM)**
*   **The Issue:** 15 traces are insufficient for SFT. Training for 50 steps on 15 traces means the model sees the exact same 15 states ~50 times. It will memorize the specific shocks and overfit catastrophically, ruining the GRPO initialization.
*   **Mitigation:** Script an API pipeline tonight to generate 150–200 diverse SFT traces using Gemini 1.5 Pro or GPT-4o. 

**6. Unsloth `use_cache=False` Caching Bug (Severity: MEDIUM)**
*   **The Issue:** In Gemma 4 E2B/E4B, late layers share KV caches from earlier layers. If you use gradient checkpointing (which forces `use_cache=False`), standard Hugging Face/TRL code fails to construct the cache correctly, outputting garbage logits and causing loss divergence.
*   **Mitigation:** Ensure you are using the absolute latest Unsloth April 2026 release, which explicitly patches the `Gemma4TextModel.forward` fallback for KV caching.

---

### 🔁 Suggested Changes to v0.6 Before Build
1.  **Refactor the Environment loop:** Decide tonight if you are writing a custom Interactive PPO/GRPO orchestrator or flattening the 12-quarter MDP into a single-turn completion for native `GRPOTrainer` compatibility.
2.  **Fix the Tokenizer Regex:** `has_think = '<|channel>thought\n' in completion and '<channel|>' in completion`.
3.  **Correct the Inference Budget:** Plan for 40 tok/s. Cut $N$ completions per prompt from 6 to 4 to salvage the Phase 3 training time. 
4.  **Patch the Put Hedge:** Change the payout condition to: "if portfolio_nav_nominal drops > 15%".
5.  **Rebalance `infra_commit`:** If `green_leaps` increases transition shocks by 2x, `infra_commit` becomes a dominant strategy. Add a counter-penalty: `infra_commit` loses -4% if a physical-risk shock hits. 

---

### 📚 Papers & Repos to Skim Tonight
1.  **G-RaR: Rubric-Based RL for Structured Reasoning in Gemma (Kaggle)**: Critical reading for structuring the SFT → GRPO pipeline on Gemma hardware.
2.  **Beyond Uniform Credit: Causal Credit Assignment for Policy Optimization (Feb 2026, arXiv)**: Contains exact hyperparameter recipes for running DAPO with `loss_type="dapo"` efficiently without gradient dilution.
3.  **Unsloth Gemma 4 Fine-Tuning Guide**: Read the "Tips for Gemma-4" section regarding the `gemma-4-thinking` chat template and the `assistant_only_loss` configurations.
