# Training pipeline — errors log

End-to-end errors we hit training the CarbonAlpha SFT + GRPO model on
Qwen3-4B-Instruct, with what worked / didn't work for each. Written from
the trenches.

---

## TL;DR

- **SFT works end-to-end on HF Jobs (L40S).** v6 SFT model shipped: +0.034 regret, 5/5 valid, 3/5 beat baseline. Adapter at [`77ethers/CarbonAlpha/v6_sft_only_v2`](https://huggingface.co/77ethers/CarbonAlpha/tree/main/v6_sft_only_v2).
- **GRPO does NOT work** with the current Unsloth + vLLM stack — vLLM rollout produces 1-token completions, gradient flow stays at zero, SFT-warmed weights get damaged.
- Two distinct upstream bugs in the path: Unsloth's `matmul_lora` dtype mismatch (issue #4891), and vLLM's `_decompose_size_nodes` graph-erase failure in 0.19.x.

---

## 1. Unsloth `matmul_lora` Half/BFloat16 dtype mismatch

**Issue / PR:** [unslothai/unsloth#4891](https://github.com/unslothai/unsloth/issues/4891), PR [#4918](https://github.com/unslothai/unsloth/pull/4918) — still open.

**Symptom:** Phase 1 GRPO crashes at iter 0 inside `matmul_lora`:

```
File "unsloth/kernels/utils.py:1059", in matmul_lora
  out.addmm_(XA, B.to(dtype), alpha = s)
RuntimeError: self and mat2 must have the same dtype, but got Half and BFloat16
```

**Root cause:** TRL's GRPO rollout runs `LoRA_QKV.forward` inside an active
fp16 autocast context. `decorate_fwd` preserves that context, so inside
`matmul_lora`:
- `out = torch_matmul(X, W.t(), out=out)` returns fp16 (autocast forces it)
- `B.to(X.dtype)` returns bf16 (X is bf16)
- `out.addmm_(XA, B.to(dtype))` → dtype mismatch

**What didn't work:**
- Switching from `unsloth-bnb-4bit` pre-quantized → non-pre-quantized base model
- bf16 throughout SFT and GRPO config (`bf16=True, fp16=False`)
- Aligning everything to fp16 (`fp16=True`) — caused 1-token sampling collapse instead
- Disabling gradient checkpointing (`use_gradient_checkpointing=False`)
- Disabling 4-bit (`load_in_4bit=False`) — same code path
- Monkey-patch on `unsloth.kernels.utils.matmul_lora` setting `out=None` when dtypes differ — didn't apply because `fast_lora.py` did `from .utils import matmul_lora` at import time, binding the original function in fast_lora's namespace
- Monkey-patch wrapping in `with torch.amp.autocast('cuda', enabled=False)` — same binding issue

**What worked:** route GRPO rollout through vLLM (`fast_inference=True` in
`FastLanguageModel.from_pretrained`, `use_vllm=True` in `GRPOConfig`).
vLLM uses its own kernels and never enters `fast_lora.py`.

---

## 2. fp16 sampling collapse on Blackwell (1-token completions)

**Symptom:** When using `fp16` to sidestep bug #1, `model.generate(do_sample=True, temperature=0.9)` returns
exactly 1 token (likely EOS) for every prompt. Greedy decoding (`do_sample=False`) works fine.

**Cause (best guess):** Numerical instability in fp16 sampling on Blackwell
(SM 12.0, RTX 5090 / RTX PRO 6000 Server Edition). The temperature-scaled
softmax produces a degenerate distribution under fp16 precision.

**What didn't work:** Various Unsloth model-load knobs (`dtype=None`,
`dtype=torch.float16` explicitly).

**What worked (workaround):** Use bf16 throughout, accept the matmul_lora
bug, then route around with vLLM (see #1).

---

## 3. vLLM 0.19.x graph-erase compile bug

**Symptom:** After enabling `fast_inference=True`, vLLM's torch.compile
phase crashes during model load:

```
File "vllm/compilation/backends.py:528", in _decompose_size_nodes
  graph.graph.erase_node(node)
RuntimeError: Tried to erase Node size_1 but it still had 2 users in the
graph: {getitem_3: None, getitem_4: None}!
```

**Cause:** vllm 0.19.x's `_decompose_size_nodes` doesn't handle the case
where a `size` node still has downstream `getitem` consumers when it tries
to erase. Triggered for Qwen3-4B with bnb-4bit + LoRA enabled.

**Hardware-independent:** Hit on both Blackwell (Pod B) and Ampere (L40S
on HF Jobs).

**What worked:** Downgrade to **vllm==0.15.1** (the version Unsloth's
official Qwen3 4B GRPO Colab notebook pins). Found by reading the install
cells of [`unslothai/notebooks/.../Qwen3_(4B)-GRPO.ipynb`](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb):

```python
_vllm = 'vllm==0.15.1'   # for non-T4 GPUs
```

---

## 4. vLLM rollout produces 1-token completions during GRPO (UNRESOLVED)

**Symptom:** GRPO Phases 1/2/3 run without crashing, but every iteration logs:

```
completions/mean_length: 1.0
completions/min_length: 1.0
completions/max_length: 1.0
loss: 0.0
grad_norm: 0.0
rewards/r_regret_phase1/mean: -0.5  (penalty floor)
frac_reward_zero_std: 1.0
```

The model emits a single token per rollout, every prompt is identical, no
gradient signal, GRPO learns nothing AND damages the SFT-warmed weights
(`v2_phase_all_v3` final eval = -0.16 regret vs SFT-only +0.014).

**Likely cause:** Interaction between Unsloth's chat-template handling and
vLLM's stop-token logic — vLLM treats one of `<|im_end|>` / `<|endoftext|>`
as immediate stop, generates that token, returns 1-token completion.

**What we tried:** revert SFT-side knobs (`alpha=16`, `gc=False`) — fixes
SFT regression but doesn't help GRPO rollout.

**Not yet tried (timeout):**
- Pass explicit `vllm_sampling_params=SamplingParams(min_p=0.1, top_p=1.0, top_k=-1, stop=[tokenizer.eos_token])` per Unsloth notebook
- Drop Unsloth entirely; use vanilla `transformers` + `peft` + TRL `GRPOTrainer` (no vLLM, no fast_lora.py) — slower but standard

**Decision:** Ship SFT-only model as the primary deliverable. Document
this as known limitation.

---

## 5. SFT hyperparameter sensitivity (recipe overshooting)

**Symptom:** Switching from `lora_alpha=16, gc=False` to "Unsloth canonical"
`lora_alpha=32, gc='unsloth'` degraded SFT hold-out from +0.014 → -0.25
regret on the same 120 v2 traces.

**Cause:** `alpha=32` with `r=16` doubles the LoRA effective learning rate.
Unsloth's recipe was tuned for OpenMathReasoning (~5× more data, longer
sequences). On 120 short prompts × 150 steps, the 2× scaling overshoots.

**What worked:** Stick with `alpha=16, gc=False`. v6 SFT under this recipe:
**+0.034 regret, 5/5 valid, 3/5 beat baseline** (current best).

---

## 6. HF Jobs UV dependency resolution conflicts

Each fixed by adjusting the PEP 723 inline `dependencies`:

| Conflict | Fix |
|---|---|
| `torch==2.6.0` vs `vllm>=0.6.6` (needs torch≥2.7) | Bump to `torch==2.10.0` |
| `numpy<2` vs `vllm>=0.19`'s `opencv-python-headless>=4.13` (needs numpy≥2) | Drop the `<2` pin |
| `ModuleNotFoundError: openenv` (our code imports `openenv.core.*`) | Add `openenv-core>=0.2` |
| `transformers==5.6.2` vs `unsloth==2026.4.8`'s `transformers<=5.5.0` | Don't pin transformers; let unsloth resolve |
| `setuptools` not on PyTorch CU index | Add `index-strategy = "unsafe-best-match"` so uv falls back to PyPI |

**Final working dep block** (in [`scripts/hf_train.py`](scripts/hf_train.py)):
```python
# /// script
# dependencies = [
#   "huggingface_hub>=0.34", "openenv-core>=0.2", "fastapi", "pydantic",
#   "uvicorn", "vllm==0.15.1", "transformers==4.56.2", "trl==0.22.2",
#   "unsloth", "torchvision", "bitsandbytes", "xformers", "peft", "datasets",
#   "accelerate", "numpy", "pillow", "matplotlib",
# ]
# [tool.uv]
# extra-index-url = ["https://download.pytorch.org/whl/cu128"]
# index-strategy = "unsafe-best-match"
# ///
```

---

## 7. Operational / infra friction (RunPod)

Burned several hours on infra before pivoting to HF Jobs:

- **Pod preemptions** on RunPod community cloud — ~5 events, each lost `/workspace` data; rebuilt venv from scratch (~5 min) each time.
- **MooseFS quota errors** mid-run when `/workspace/checkpoints` accumulated 8.8GB of stale adapters — silent process kill during model download.
- **HF cache disk pressure** on `/root` overlay (30GB). 15GB HF cache filled it after a few model loads. Fix: `rm -rf /root/.cache/huggingface/hub` before each run, or set `HF_HOME=/workspace/.hf_cache`.
- **CUDA passthrough broken** on one pod (`/dev/nvidia1` owned by `nobody:nogroup`). Reprovisioned.
- **scp failures** to MFS even with quota free — switched to `gzip | ssh ... gzip -d` pipe.

---

## 8. HF whoami rate limit during rapid job submission

**Symptom:** After 2-3 `hf jobs uv run` calls in quick succession:
```
Error: You've hit the rate limit for the /whoami-v2 endpoint, which is
intentionally strict for security reasons.
```

**Cause:** `hf jobs uv run` calls `whoami` uncached on each invocation.
Cooldown is several minutes per IP/account.

**Workaround:** Use `HfApi.run_uv_job()` directly (caches `whoami` in
single Python process), or just wait 5+ min between submissions.

---

## 9. Small but tedious bugs in our own scripts

| Bug | Fix |
|---|---|
| `grpo_training.py` SFT-only mode returned before saving the adapter | Save in SFT-only branch too |
| `hf_train.py` upload step didn't include `CARBON_ALPHA_OUTPUT_DIR` in candidate paths | Added env-var path as first candidate |
| zsh treated `0.15.1` and `_v2` as glob patterns in echo strings | Avoid the offending echos / quote them |
| `OUTPUT_DIR` hardcoded to `/workspace/checkpoints` (not writable on HF Jobs containers) | `Path(os.environ.get('CARBON_ALPHA_OUTPUT_DIR', '/workspace/checkpoints'))` |
| HF dataset stale snapshot — Job B downloaded code BEFORE my push reached the dataset | Just resubmit after confirming the dataset commit landed |

---

## What we'd do differently next time

1. **Skip Unsloth for GRPO from the start.** The fast_lora.py kernel + autocast + vLLM stack is a fragile combination on bleeding-edge GPUs. Vanilla `transformers + peft + TRL` is slower but doesn't have these failure modes.
2. **Pin EVERYTHING from a known-working notebook.** Our first 3 HF Jobs failed on dep resolution because we tried to use latest. Unsloth's tested matrix (vllm 0.15.1, transformers 4.56.2, trl 0.22.2) was the right call once we found it.
3. **HF Jobs from day 1, RunPod for nothing.** Pod preemptions + MFS quotas + custom env management ate hours. HF Jobs's container-per-job model is more reliable for training runs.
4. **Eyeball SFT hyperparameters with a quick sweep.** alpha=16 vs alpha=32 was a 0.27 regret swing — would have caught it in 30 min of sweeping instead of a full pipeline run.
