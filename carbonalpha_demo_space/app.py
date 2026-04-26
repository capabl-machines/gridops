"""CarbonAlpha custom walkthrough Space.

FastAPI + vanilla HTML/JS UI. Loads the Qwen2.5-7B-Instruct LoRA adapter and
uses the real PortfolioEnv OpenEnv class for reset/step/state/metadata during
an interactive 12-quarter walkthrough.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from peft import PeftModel
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from portfolio_env import PortfolioAction, PortfolioEnv, parse_json_action
from portfolio_env.constants import ASSETS, CARBON_CAP, CARBON_INTENSITY
from portfolio_env.rewards import r_carbon, r_drawdown, r_format, r_regret, r_sharpe
from portfolio_env.shocks import SHOCKS_BY_ID, Shock, shocks_available
from portfolio_env.prompt import SYSTEM_PROMPT, build_user_prompt


MODEL_REPO = "77ethers/CarbonAlpha"
# Two adapters are loaded so the demo can show the full progression:
#   base Qwen (no adapter) → SFT-warmed → GRPO-tuned.
GRPO_SUBFOLDER = os.environ.get("MODEL_SUBFOLDER", "grpo_qwen25_7b_adapter_phase1_100_v1")
SFT_SUBFOLDER  = os.environ.get("SFT_SUBFOLDER",   "sft_qwen25_7b_curriculum400_v1")
SUBFOLDER = GRPO_SUBFOLDER  # primary adapter for /health and back-compat
BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct"
HF_TOKEN = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NEWS_POOL: list[str] = [
    "Cat-5 hurricane forecast for US Gulf Coast. Insurers downgrade REIT exposure. Gulf refineries at risk. FEMA preparing $80B reconstruction package. Fed hints at emergency rate cut.",
    "Fed minutes leaked: PCE core unexpectedly at 5.8%. Committee signals sustained tightening into 2027. 10-year yields climb 80bp. Dollar rallies against EM. Oil services announce capacity expansion.",
    "China announces 80% reduction in rare-earth exports over 18 months citing domestic demand. US semiconductor export controls tighten. Renewable manufacturers warn of 3-quarter supply chain disruption.",
    "Routine earnings season. Tech majors beat estimates by 2.8% on average. Bond yields steady. No macro surprises.",
    "Two leading ratings agencies find 40% of voluntary carbon offsets invalid. EU CBAM enforcement agency announces retroactive audit. Offset market prices plunge while compliance-market credits spike 3x.",
    "EU CBAM enters full enforcement phase. €23/ton carbon tariff applied to imported steel, aluminum, cement, fertilizer, hydrogen. Emerging-market exporters scramble; domestic green steel premium widens.",
    "India approves 30 GW of new coal capacity over 5 years citing energy security. Climate trajectory diverges from 1.5C target. Adaptation sector and air-quality tech see record inflows.",
    "First commercial small modular reactor (SMR) fleet enters operation in US, UK, Poland. Uranium prices rally 35%. Hyperscaler PPAs signed for 24/7 carbon-free electricity. Grid baseload economics reset.",
    "AI data center demand exceeds US Western Interconnect capacity. Phoenix and Northern Virginia experience rolling brownouts. Hyperscalers pause $40B in new capex. Nuclear and gas peaker stocks rally.",
    "Green hydrogen production hits $2/kg at scale via electrolyzer cost crash. ArcelorMittal, Thyssenkrupp lock 10-year offtakes. Iron ore demand mix shifts toward DR-grade pellets.",
    "US methane fee of $1,500/ton kicks in. Permian leak rates forced down via mandatory monitoring. Marginal upstream operators announce shut-ins. Oil supply tightens paradoxically; LNG export economics improve.",
    "LFP battery oversupply pushes utility-scale prices to $45/kWh. Grid storage deployment triples in 12 months. Natural gas peaker plants accelerate retirement schedules. Renewable curtailment falls sharply.",
    "Atmospheric river drenches California for 8 consecutive weeks. Reservoirs at 110% capacity but levee failures flood Central Valley. $50B agriculture loss. Insurance retreats from coastal CA.",
    "Chinese property developer Country Garden defaults on $20B in offshore bonds. Contagion fears spread to regional banks and trust products. PBOC announces 100bp RRR cut. Yuan slides 4%.",
    "ECB hikes 50bp and signals end of negative-rates era. Bund yields touch 4%. Carry-trade unwind hits EM FX. Periphery spreads widen. Real-estate financing costs jump 30%.",
    "TSMC's Arizona Fab 21 opens at scale, producing 4nm and 3nm at US-domestic content. CHIPS Act subsidies validated. Nvidia, AMD shift 15% of advanced production stateside.",
    "Russian Druzhba pipeline restarts after 18-month outage. EU receives 1.6 mbpd of crude at discount. European refining margins compress. Renewable installation pace slows on cheap energy.",
    "Iceland's Bardarbunga volcano erupts; ash cloud halts trans-Atlantic flights for 2 weeks. Stratospheric SO2 injection cools global temperatures 0.3C for 18 months. Soft commodity yields spike.",
    "Autonomous AI trading agents now drive 35% of retail equity flows. SEC announces inquiry into reflexive volatility patterns. Liquidity bifurcates between AI-favored and orphaned tickers.",
    "Saudi Arabia's $500B Manafa sovereign fund launches with mandate to corner critical-minerals supply chains. Lithium, cobalt, copper miners announce off-take agreements. Western strategic reserves expand.",
    "Dutch court orders 10 oil majors to cut emissions 50% by 2035, replicating the 2021 Shell ruling. Insurance retrenches from upstream. Pension funds initiate $200B divestment. Stranded-asset writedowns begin.",
    "BoJ exits zero-rates after 17 years. Yen rallies 15% in 6 weeks; carry-trade unwind sends global rates higher. Japanese institutional investors repatriate $700B in foreign assets. JGB yields touch 1.8%.",
    "Once-in-a-century drought hits Latin America. Brazilian coffee output down 40%; Chilean copper concentrators shut. Soft commodity prices spike. EM food inflation accelerates 8%.",
    "UN Loss & Damage fund finally operationalizes at $40B/year, funded by financial-transaction tax on G20 nations. Climate-vulnerable countries gain leverage in WTO disputes. Adaptation tech sector rallies.",
]

# Fallback shock mapping when env stepping is invoked. Keys are hash-stable
# fragments matched against the news; default to a routine shock.
NEWS_SHOCK_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("ambig_hurricane_gulf", ("hurricane", "gulf", "fema", "reconstruction")),
    ("ambig_stagflation_trigger", ("stagflation", "pce", "tightening", "10-year")),
    ("hard_rare_earth_rotation", ("rare-earth", "rare earth", "export controls")),
    ("hard_carbon_offset_fraud", ("carbon offset", "offsets", "cbam audit")),
    ("hard_taiwan_water_chip", ("tsmc", "taiwan", "chips act", "fab 21")),
    ("ambig_insurance_retreat", ("insurance retreats", "reinsurance", "property markets")),
    ("easy_oil_opec_cut", ("opec", "production cut", "spot crude", "druzhba")),
    ("easy_ev_penetration", ("ev registrations", "charging network", "fleet electrification")),
    ("hard_deflation_pulse", ("deflation", "rrr cut", "yuan slides")),
]

print("Loading CarbonAlpha model...", flush=True)
_load_t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, subfolder=SUBFOLDER, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if DEVICE == "cuda":
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization,
        device_map="auto",
        token=HF_TOKEN,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        token=HF_TOKEN,
    )

model = PeftModel.from_pretrained(
    base_model, MODEL_REPO, subfolder=GRPO_SUBFOLDER, adapter_name="grpo", token=HF_TOKEN,
)
# Mount the SFT adapter alongside GRPO so we can A/B at runtime via set_adapter().
try:
    model.load_adapter(MODEL_REPO, subfolder=SFT_SUBFOLDER, adapter_name="sft", token=HF_TOKEN)
    SFT_LOADED = True
    print(f"SFT adapter loaded ({SFT_SUBFOLDER})", flush=True)
except Exception as _sft_err:
    SFT_LOADED = False
    print(f"WARN: SFT adapter not loaded ({_sft_err}); demo will skip SFT column.", flush=True)
model.set_adapter("grpo")
model.eval()
print(f"Model loaded in {time.time() - _load_t0:.1f}s on {DEVICE} (primary adapter: grpo)", flush=True)

app = FastAPI(title="CarbonAlpha Walkthrough")
STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


class RunRequest(BaseModel):
    news: str = Field(default="")
    example: str = Field(default="")
    seed: int = Field(default=100, ge=0, le=999999)
    phase: int = Field(default=3, ge=1, le=3)
    max_new_tokens: int = Field(default=420, ge=64, le=700)


class StepRequest(BaseModel):
    session_id: str


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
SESSIONS: dict[str, dict[str, Any]] = {}


def split_completion(completion: str) -> tuple[str, dict[str, Any] | None]:
    match = THINK_RE.search(completion)
    reasoning = match.group(1).strip() if match else ""
    payload = parse_json_action(completion)
    return reasoning, payload


def normalize_action(payload: dict[str, Any] | None) -> PortfolioAction:
    if not isinstance(payload, dict):
        return PortfolioAction(weights=[0.2] * 5)
    weights = payload.get("weights")
    if not isinstance(weights, list) or len(weights) != 5:
        weights = [0.2] * 5
    try:
        weights = [max(0.0, float(x)) for x in weights]
    except Exception:
        weights = [0.2] * 5
    try:
        return PortfolioAction(
            weights=weights,
            infra_commit=float(payload.get("infra_commit", 0.0) or 0.0),
            carbon_offset_buy=float(payload.get("carbon_offset_buy", 0.0) or 0.0),
            put_hedge=float(payload.get("put_hedge", 0.0) or 0.0),
            tech_bet=payload.get("tech_bet", "status_quo"),
        )
    except Exception:
        return PortfolioAction(weights=[0.2] * 5)


COUNTERFACTUAL_STRATEGIES = [
    {
        "id": "benchmark",
        "label": "Benchmark",
        "color": "#6bb8d6",
        "action": PortfolioAction(weights=[0.2, 0.2, 0.2, 0.2, 0.2]),
        "blurb": "Equal-weighted basket of the 5 ETFs — quarter-wise average of all index returns.",
    },
]


# ETF backtest data (12 quarters of % real returns — Q1 2022 through Q4 2024).
# Each strategy is replayed against this same returns matrix so the only varying
# factor across the 4 NAV lines is the allocation choice.
ETF_TICKERS: dict[str, str] = {
    "TECH":        "IXN",   # iShares Global Tech ETF
    "OIL":         "DBO",   # Invesco DB Oil Fund
    "GREEN":       "ICLN",  # iShares Global Clean Energy ETF
    "REAL_ESTATE": "VNQI",  # Vanguard Global ex-US Real Estate ETF
    "BONDS":       "BNDW",  # Vanguard Total World Bond ETF (USD-hedged)
}

QUARTERLY_RETURNS_PCT: dict[str, list[float]] = {
    "TECH":        [20.5,  7.0,  -5.6, 16.7,  11.4, -3.9, 13.8,  4.8,  9.2, -4.1,  6.8, 11.2],
    "OIL":         [-0.3, -7.6,  -4.3,  8.2,  11.6,  5.5, -4.7, -4.1,  2.1, -8.4, -3.9, -1.8],
    "GREEN":       [ 2.6,  3.3, -22.5, -3.2,  -7.3,  2.8, -9.1,-12.7, 14.8,  8.6, 13.2,  6.8],
    "REAL_ESTATE": [ 3.4,  1.5,  -5.2,  7.4,   3.7, -4.2,  7.0, -8.6,  6.1,  3.8,  5.4,  4.4],
    "BONDS":       [ 3.2,  0.8,  -3.5,  6.6,   0.9, -1.3,  5.2, -2.3,  1.4,  0.6,  1.8,  1.1],
}


def run_etf_trajectory(action: PortfolioAction) -> dict[str, Any]:
    """Pure backtest: apply locked weights to the real ETF quarterly returns.

    All counterfactual strategies AND the model's own projection use this same
    deterministic returns matrix — the only thing varying across them is the
    allocation, so the comparison is exactly fair.
    """
    weights = list(action.weights)
    intensities = [CARBON_INTENSITY[a] for a in ASSETS]
    nav = [1.0]
    carbon = [0.0]
    for q in range(12):
        portfolio_return = sum(
            weights[i] * (QUARTERLY_RETURNS_PCT[a][q] / 100.0)
            for i, a in enumerate(ASSETS)
        )
        nav.append(nav[-1] * (1.0 + portfolio_return))
        carbon_step = sum(weights[i] * intensities[i] * nav[-1] for i in range(len(ASSETS)))
        carbon.append(carbon[-1] + carbon_step)
    nav = [round(x, 4) for x in nav]
    carbon = [round(x, 4) for x in carbon]
    return {
        "nav_real": nav,
        "nav_nominal": nav,  # ETF returns already real
        "baseline_nav_real": nav,
        "carbon": carbon,
        "carbon_by_asset": carbon_attribution(weights, nav),
        "final_nav_real": nav[-1],
        "final_carbon": carbon[-1],
        "real_return_pct": round((nav[-1] - 1.0) * 100, 2),
        "carbon_over_cap": carbon[-1] > CARBON_CAP,
    }


def carbon_attribution(weights: list[float], nav_nominal_series: list[float]) -> dict[str, list[float]]:
    """Per-quarter, per-asset cumulative carbon kg, given the locked weights.

    Mirrors env.py: carbon_q = sum_i (w[i] * intensity[i] * nav_nominal_q).
    Produces cumulative series so the frontend stacked-area reads as a thermometer
    of how each asset added to the budget over time.
    """
    per_asset_quarterly: dict[str, list[float]] = {a: [] for a in ASSETS}
    intensities = [CARBON_INTENSITY[a] for a in ASSETS]
    # nav_nominal_series includes the starting 1.0 at index 0; per-quarter carbon
    # in env.py uses s.nav_nominal AT THE TIME of the step, which is the post-step
    # nav for that quarter. For attribution display we approximate using the
    # series we collected post-step (skipping index 0).
    for q_idx, nav in enumerate(nav_nominal_series[1:], start=1):
        for i, asset in enumerate(ASSETS):
            per_asset_quarterly[asset].append(float(weights[i]) * intensities[i] * float(nav))
    cumulative: dict[str, list[float]] = {a: [0.0] for a in ASSETS}
    running = {a: 0.0 for a in ASSETS}
    for q_idx in range(len(nav_nominal_series) - 1):
        for asset in ASSETS:
            running[asset] += per_asset_quarterly[asset][q_idx]
            cumulative[asset].append(running[asset])
    return cumulative


def run_counterfactual(seed: int, phase: int, plan, action: PortfolioAction) -> dict[str, Any]:
    """Replay the same shock schedule with a different fixed allocation.

    Same seed+phase+plan ensures identical shock sequence and (because rng state
    is at the post-reset, post-plan-generation point in both envs) identical
    return draws — so the comparison is apples-to-apples on macro path,
    differences are pure allocation choice.
    """
    env = PortfolioEnv(phase=phase, seed=seed)
    env.reset(seed=seed)
    env._plan = copy.deepcopy(plan)
    nav_real = [1.0]
    nav_nominal = [1.0]
    baseline = [1.0]
    carbon = [0.0]
    for _ in range(12):
        obs = env.step(action, completion="")
        nav_real.append(round(float(obs.portfolio_nav_real), 4))
        nav_nominal.append(round(float(obs.portfolio_nav_nominal), 4))
        baseline.append(round(float(obs.baseline_nav_real), 4))
        carbon.append(round(float(obs.carbon_footprint_accumulated), 4))
        if obs.done:
            break
    final_real_return_pct = (nav_real[-1] - 1.0) * 100.0
    return {
        "nav_real": nav_real,
        "nav_nominal": nav_nominal,
        "baseline_nav_real": baseline,
        "carbon": carbon,
        "carbon_by_asset": carbon_attribution(list(action.weights), nav_nominal),
        "final_nav_real": round(nav_real[-1], 4),
        "final_carbon": round(carbon[-1], 4),
        "real_return_pct": round(final_real_return_pct, 2),
        "carbon_over_cap": carbon[-1] > CARBON_CAP,
    }


def shock_for_news(news: str) -> Shock:
    """Pick a representative env shock matching the news text (best-effort)."""
    text = news.lower()
    explicit_id = "easy_tech_earnings"
    for shock_id, needles in NEWS_SHOCK_KEYWORDS:
        if any(needle in text for needle in needles):
            explicit_id = shock_id
            break
    base = SHOCKS_BY_ID.get(explicit_id)
    return Shock(
        id=f"demo_{base.id}",
        tier=base.tier,
        news=news,
        impacts=dict(base.impacts),
        regime_shift=base.regime_shift,
        tags=list(base.tags),
    )


@torch.no_grad()
def generate_completion(news: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(news)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _build_inputs(news: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(news)},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").to(DEVICE)


def _stream_with_streamer(streamer, gen_kwargs, mode: str):
    """Run generate() in a thread; collect tokens via the streamer for the
    main thread to yield. `mode` selects which model variant runs:
      - "grpo" : LoRA active = GRPO adapter (default training output)
      - "sft"  : LoRA active = SFT adapter (warm-start, pre-RL)
      - "base" : LoRA disabled, raw base Qwen 2.5 7B Instruct
    """

    def _run():
        if mode == "base":
            with model.disable_adapter():
                model.generate(**gen_kwargs)
        elif mode == "sft":
            if not SFT_LOADED:
                # Should never reach here — caller checks SFT_LOADED first.
                model.generate(**gen_kwargs)
            else:
                prev = getattr(model, "active_adapter", "grpo")
                try:
                    model.set_adapter("sft")
                    model.generate(**gen_kwargs)
                finally:
                    model.set_adapter(prev if prev in ("grpo", "sft") else "grpo")
        else:  # "grpo" or anything unrecognized -> grpo
            prev = getattr(model, "active_adapter", "grpo")
            try:
                model.set_adapter("grpo")
                model.generate(**gen_kwargs)
            finally:
                model.set_adapter(prev if prev in ("grpo", "sft") else "grpo")

    thread = Thread(target=_run)
    thread.start()
    full = ""
    for chunk in streamer:
        full += chunk
        yield chunk, full
    thread.join()


def obs_row(q: int, news_seen: str, obs, env: PortfolioEnv) -> dict[str, Any]:
    snapshot = (obs.metadata or {}).get("snapshot", {}) if hasattr(obs, "metadata") else {}
    shock_id = (obs.metadata or {}).get("shock_fired") if hasattr(obs, "metadata") else None
    if shock_id:
        narration = f"Completed Q{q}: shock fired ({shock_id}). Regime now {getattr(obs, 'current_regime', 'normal')}."
    else:
        narration = f"Completed Q{q}: routine market path. Regime now {getattr(obs, 'current_regime', 'normal')}."
    return {
        "quarter": q,
        "news": news_seen,
        "shock": shock_id or "routine",
        "regime": getattr(obs, "current_regime", "normal"),
        "nav_real": round(float(snapshot.get("nav_real", obs.portfolio_nav_real)), 4),
        "baseline_nav_real": round(float(snapshot.get("baseline_nav_real", obs.baseline_nav_real)), 4),
        "step_reward": round(float(obs.reward or 0.0), 5),
        "regret_so_far": round(float(snapshot.get("regret_so_far", obs.last_quarter_regret)), 5),
        "carbon": round(float(snapshot.get("carbon_accumulated", obs.carbon_footprint_accumulated)), 4),
        "carbon_remaining": round(float(obs.carbon_budget_remaining), 4),
        "narration": narration,
    }


def grade_episode(env: PortfolioEnv, completion: str) -> dict[str, Any]:
    traj = env.trajectory
    grade = env.state.final_grade or {
        "r_format": float(r_format(completion)),
        "r_regret": float(r_regret(traj)),
        "r_sharpe": float(r_sharpe(traj)),
        "r_carbon": float(r_carbon(traj, phase_weight=1.0)),
        "r_drawdown": float(r_drawdown(traj)),
        "final_nav_real": float(traj.nav_real_series[-1]),
        "baseline_nav_real": float(traj.baseline_nav_real_series[-1]),
    }
    return {k: round(float(v), 5) if isinstance(v, (int, float)) else v for k, v in grade.items()}


def empty_grade(env: PortfolioEnv, completion: str) -> dict[str, Any]:
    traj = env.trajectory
    return {
        "r_format": round(float(r_format(completion)), 5),
        "r_regret": round(float(r_regret(traj)), 5),
        "r_sharpe": round(float(r_sharpe(traj)), 5),
        "r_carbon": round(float(r_carbon(traj, phase_weight=1.0)), 5),
        "r_drawdown": round(float(r_drawdown(traj)), 5),
        "final_nav_real": round(float(traj.nav_real_series[-1]), 5),
        "baseline_nav_real": round(float(traj.baseline_nav_real_series[-1]), 5),
    }


def episode_payload(session: dict[str, Any]) -> dict[str, Any]:
    env: PortfolioEnv = session["env"]
    timeline = session["timeline"]
    grade = grade_episode(env, session["completion"]) if env.state.done else empty_grade(env, session["completion"])
    cfs = session.get("counterfactuals") or {}
    model_carbon_attr = session.get("model_carbon_attribution") or {a: [0.0] for a in ASSETS}

    # Pareto plot points: (real_return_pct, final_carbon_kg). Model uses what the
    # session has played so far; if not yet complete, fall back to the projected
    # path under the locked allocation (which we precomputed as a counterfactual).
    model_projection = session.get("model_projection") or {}
    pareto = [
        {
            "id": "model",
            "label": "CarbonAlpha (locked)",
            "color": "#62d29f",
            "real_return_pct": round(((model_projection.get("final_nav_real", 1.0)) - 1.0) * 100, 2),
            "final_carbon": round(float(model_projection.get("final_carbon", 0.0)), 3),
            "carbon_over_cap": bool(model_projection.get("carbon_over_cap", False)),
        }
    ] + [
        {
            "id": cf_id,
            "label": cf["label"],
            "color": cf["color"],
            "real_return_pct": cf["data"]["real_return_pct"],
            "final_carbon": cf["data"]["final_carbon"],
            "carbon_over_cap": cf["data"]["carbon_over_cap"],
        }
        for cf_id, cf in cfs.items()
    ]

    return {
        "metadata": {
            "name": env.get_metadata().name,
            "version": env.get_metadata().version,
            "phase": session["phase"],
            "seed": session["seed"],
        },
        "session_id": session["id"],
        "current_news": session["obs"].news,
        "state": env.state.model_dump(),
        "timeline": timeline,
        "grade": grade,
        "series": {
            "quarters": [0] + [row["quarter"] for row in timeline],
            "nav_real": [1.0] + [row["nav_real"] for row in timeline],
            "baseline_nav_real": [1.0] + [row["baseline_nav_real"] for row in timeline],
            "carbon": [0.0] + [row["carbon"] for row in timeline],
            "carbon_cap": [CARBON_CAP] * (len(timeline) + 1),
            "step_reward": [row["step_reward"] for row in timeline],
            "cumulative_reward": [row["cumulative_reward"] for row in timeline],
        },
        "model_projection": {
            "quarters": list(range(13)),
            "nav_real": model_projection.get("nav_real", [1.0]),
            "carbon": model_projection.get("carbon", [0.0]),
            "final_nav_real": model_projection.get("final_nav_real", 1.0),
            "final_carbon": model_projection.get("final_carbon", 0.0),
            "real_return_pct": model_projection.get("real_return_pct", 0.0),
            "carbon_over_cap": model_projection.get("carbon_over_cap", False),
        },
        "counterfactuals": {
            cf_id: {
                "label": cf["label"],
                "color": cf["color"],
                "blurb": cf["blurb"],
                "quarters": list(range(13)),
                "nav_real": cf["data"]["nav_real"],
                "carbon": cf["data"]["carbon"],
                "final_nav_real": cf["data"]["final_nav_real"],
                "final_carbon": cf["data"]["final_carbon"],
                "real_return_pct": cf["data"]["real_return_pct"],
                "carbon_over_cap": cf["data"]["carbon_over_cap"],
            }
            for cf_id, cf in cfs.items()
        },
        "carbon_attribution": {
            "assets": list(ASSETS),
            "tickers": [ETF_TICKERS[a] for a in ASSETS],
            "quarters": list(range(13)),
            "cumulative_by_asset": {asset: model_carbon_attr.get(asset, [0.0]) for asset in ASSETS},
            "cap": CARBON_CAP,
        },
        "pareto": pareto,
        "carbon_cap": CARBON_CAP,
        "etf_tickers": ETF_TICKERS,
        "backtest_window": "Q1 2022 – Q4 2024",
    }


def start_session(news: str, example: str, action: PortfolioAction, completion: str, seed: int, phase: int) -> dict[str, Any]:
    env = PortfolioEnv(phase=phase, seed=seed)
    obs = env.reset(seed=seed)
    if getattr(env, "_plan", None) is not None:
        env._plan.shocks_by_quarter[0] = shock_for_news(news)

    # All projection trajectories come from a deterministic ETF backtest —
    # same returns matrix across model + counterfactuals, so the only varying
    # factor is allocation choice. The env still drives news + per-quarter
    # narration during /api/step; it just doesn't drive the headline charts.
    plan_snapshot = copy.deepcopy(env._plan) if env._plan is not None else None
    model_projection = run_etf_trajectory(action)
    counterfactuals: dict[str, dict[str, Any]] = {
        strat["id"]: {
            "label": strat["label"],
            "color": strat["color"],
            "blurb": strat["blurb"],
            "data": run_etf_trajectory(strat["action"]),
        }
        for strat in COUNTERFACTUAL_STRATEGIES
    }
    model_carbon_attribution = model_projection["carbon_by_asset"]

    sid = uuid4().hex
    session = {
        "id": sid,
        "env": env,
        "obs": obs,
        "news": news,
        "action": action,
        "completion": completion,
        "timeline": [],
        "cumulative_reward": 0.0,
        "seed": seed,
        "phase": phase,
        "created_at": time.time(),
        "plan_snapshot": plan_snapshot,
        "model_projection": model_projection,
        "model_carbon_attribution": model_carbon_attribution,
        "counterfactuals": counterfactuals,
    }
    SESSIONS[sid] = session
    return session


def step_session(session: dict[str, Any]) -> dict[str, Any]:
    env: PortfolioEnv = session["env"]
    obs = session["obs"]
    if obs.done:
        return session
    q = env.state.quarter + 1
    news_seen = session["news"] if env.state.quarter == 0 else obs.news
    next_obs = env.step(session["action"], completion=session["completion"])
    session["cumulative_reward"] += float(next_obs.reward or 0.0)
    row = obs_row(q, news_seen, next_obs, env)
    row["cumulative_reward"] = round(session["cumulative_reward"], 5)
    session["timeline"].append(row)
    session["obs"] = next_obs
    return session


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if STATIC_INDEX.exists():
        return STATIC_INDEX.read_text()
    return HTML


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "device": DEVICE, "model": SUBFOLDER}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    env = PortfolioEnv(phase=3, seed=100)
    meta = env.get_metadata()
    return {"name": meta.name, "version": meta.version, "description": meta.description}


_RECENT_NEWS_IDX: list[int] = []  # in-memory ring of last shown news indices


@app.get("/api/random-news")
def random_news() -> dict[str, Any]:
    """Returns one random news headline from the 24-item pool, avoiding the
    last few shown to keep variety across consecutive runs."""
    import random as _r
    avoid = set(_RECENT_NEWS_IDX[-6:])
    candidates = [i for i in range(len(NEWS_POOL)) if i not in avoid] or list(range(len(NEWS_POOL)))
    idx = _r.choice(candidates)
    _RECENT_NEWS_IDX.append(idx)
    if len(_RECENT_NEWS_IDX) > 100:
        del _RECENT_NEWS_IDX[:50]
    return {"index": idx, "news": NEWS_POOL[idx], "pool_size": len(NEWS_POOL)}


@app.get("/api/random-news-batch")
def random_news_batch(n: int = 12) -> dict[str, Any]:
    """Returns N unique random news headlines from the 24-item pool.
    Used at page boot to seed each of the 12 quarter slots with distinct news."""
    import random as _r
    n = max(1, min(n, len(NEWS_POOL)))
    indices = _r.sample(range(len(NEWS_POOL)), n)
    return {
        "items": [{"index": i, "news": NEWS_POOL[i]} for i in indices],
        "pool_size": len(NEWS_POOL),
    }


@app.get("/examples")
def examples() -> dict[str, str]:
    """Back-compat — returns first 5 news from the pool keyed numerically."""
    return {f"news_{i+1}": v for i, v in enumerate(NEWS_POOL[:5])}


@app.get("/api/baselines")
def baselines() -> dict[str, Any]:
    """Pre-computed counterfactual ETF backtests for the empty/Q0 page state.

    The 3 counterfactual strategies depend only on the ETF returns matrix —
    no model, no env. Frontend calls this on page load so charts are populated
    before the user generates a CarbonAlpha allocation.
    """
    cfs = {
        strat["id"]: {
            "label": strat["label"],
            "color": strat["color"],
            "blurb": strat["blurb"],
            "data": run_etf_trajectory(strat["action"]),
        }
        for strat in COUNTERFACTUAL_STRATEGIES
    }
    pareto = [
        {
            "id": cf_id,
            "label": cf["label"],
            "color": cf["color"],
            "real_return_pct": cf["data"]["real_return_pct"],
            "final_carbon": cf["data"]["final_carbon"],
            "carbon_over_cap": cf["data"]["carbon_over_cap"],
        }
        for cf_id, cf in cfs.items()
    ]
    # Benchmark is the natural "doing nothing" preview for Carbon Attribution
    # at Q0 — it shows what the chart will look like for any locked allocation.
    preview = cfs.get("benchmark", {}).get("data", {})
    return {
        "counterfactuals": {
            cf_id: {
                "label": cf["label"],
                "color": cf["color"],
                "blurb": cf["blurb"],
                "quarters": list(range(13)),
                "nav_real": cf["data"]["nav_real"],
                "carbon": cf["data"]["carbon"],
                "final_nav_real": cf["data"]["final_nav_real"],
                "final_carbon": cf["data"]["final_carbon"],
                "real_return_pct": cf["data"]["real_return_pct"],
                "carbon_over_cap": cf["data"]["carbon_over_cap"],
            }
            for cf_id, cf in cfs.items()
        },
        "pareto": pareto,
        "etf_tickers": ETF_TICKERS,
        "carbon_cap": CARBON_CAP,
        "backtest_window": "Q1 2022 – Q4 2024",
        "preview_attribution": {
            "label": "Equal-weight preview",
            "cumulative_by_asset": preview.get("carbon_by_asset", {}),
            "final_carbon": preview.get("final_carbon", 0.0),
        },
        # Exposed so the frontend can compute combined per-quarter trajectories
        # when the user re-plans for a specific quarter (allocation from quarter
        # N forward changes; quarters 1..N-1 keep their previous allocation).
        "assets": list(ASSETS),
        "quarterly_returns_pct": QUARTERLY_RETURNS_PCT,
        "carbon_intensity": CARBON_INTENSITY,
    }


@app.post("/api/plan-stream")
def plan_stream(req: RunRequest):
    """Server-Sent Events stream of the model's planning.

    Phases (each emitted as a separate `data:` line):
      - trained_start { news }                          — about to stream CarbonAlpha
      - trained_token { token }                         — incremental token chunk
      - trained_done  { completion, reasoning, action } — locked allocation parsed
      - base_start    {}                                — about to stream base Qwen 2.5
      - base_token    { token }                         — incremental token chunk
      - base_done     { completion }                    — base completion finished
      - error         { error }                         — anything went wrong
    """
    news = (req.news or "").strip() or NEWS_POOL[0]
    max_new_tokens = int(req.max_new_tokens or 420)

    def event_gen():
        try:
            inputs = _build_inputs(news)
            common_gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            # ── 1. CarbonAlpha — GRPO adapter (RL-tuned) ──
            yield f"data: {json.dumps({'phase': 'trained_start', 'news': news, 'variant': 'grpo'})}\n\n"
            streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
            kw = dict(common_gen_kwargs, **inputs, streamer=streamer)
            trained_full = ""
            for chunk, _full in _stream_with_streamer(streamer, kw, mode="grpo"):
                trained_full += chunk
                yield f"data: {json.dumps({'phase': 'trained_token', 'token': chunk})}\n\n"

            reasoning, payload = split_completion(trained_full)
            action = normalize_action(payload)
            yield f"data: {json.dumps({'phase': 'trained_done', 'completion': trained_full, 'reasoning': reasoning, 'action': action.model_dump(), 'raw_action': payload, 'news': news})}\n\n"

            # ── 2. SFT-only adapter (warm-start, pre-RL) ──
            sft_full = ""
            if SFT_LOADED:
                yield f"data: {json.dumps({'phase': 'sft_start', 'variant': 'sft'})}\n\n"
                streamer_sft = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
                kw_sft = dict(common_gen_kwargs, **inputs, streamer=streamer_sft)
                for chunk, _full in _stream_with_streamer(streamer_sft, kw_sft, mode="sft"):
                    sft_full += chunk
                    yield f"data: {json.dumps({'phase': 'sft_token', 'token': chunk})}\n\n"
                yield f"data: {json.dumps({'phase': 'sft_done', 'completion': sft_full})}\n\n"
            else:
                yield f"data: {json.dumps({'phase': 'sft_unavailable'})}\n\n"

            # ── 3. Base Qwen 2.5 7B Instruct (LoRA disabled) ──
            yield f"data: {json.dumps({'phase': 'base_start'})}\n\n"
            streamer2 = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
            kw2 = dict(common_gen_kwargs, **inputs, streamer=streamer2)
            base_full = ""
            for chunk, _full in _stream_with_streamer(streamer2, kw2, mode="base"):
                base_full += chunk
                yield f"data: {json.dumps({'phase': 'base_token', 'token': chunk})}\n\n"
            yield f"data: {json.dumps({'phase': 'base_done', 'completion': base_full})}\n\n"

        except Exception as exc:
            print(traceback.format_exc(), flush=True)
            yield f"data: {json.dumps({'phase': 'error', 'error': str(exc), 'type': type(exc).__name__})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/start")
def start(req: RunRequest) -> JSONResponse:
    start = time.time()
    try:
        news = (req.news or "").strip() or NEWS_POOL[0]
        completion = generate_completion(news, req.max_new_tokens)
        reasoning, payload = split_completion(completion)
        action = normalize_action(payload)
        session = start_session(news, req.example, action, completion, req.seed, req.phase)
        action_payload = action.model_dump()
        return JSONResponse({
            "news": news,
            "completion": completion,
            "reasoning": reasoning,
            "raw_action": payload,
            "action": action_payload,
            "assets": ASSETS,
            "episode": episode_payload(session),
            "latency_s": round(time.time() - start, 2),
        })
    except Exception as exc:
        print(traceback.format_exc(), flush=True)
        return JSONResponse({
            "error": str(exc),
            "type": type(exc).__name__,
        }, status_code=500)


@app.post("/api/step")
def step(req: StepRequest) -> JSONResponse:
    session = SESSIONS.get(req.session_id)
    if session is None:
        return JSONResponse({"error": "session not found"}, status_code=404)
    step_session(session)
    completion = session["completion"]
    reasoning, payload = split_completion(completion)
    return JSONResponse({
        "news": session["news"],
        "completion": completion,
        "reasoning": reasoning,
        "raw_action": payload,
        "action": session["action"].model_dump(),
        "assets": ASSETS,
        "episode": episode_payload(session),
        "latency_s": 0.0,
    })


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CarbonAlpha Walkthrough</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0f13;
      --panel: #131a20;
      --panel2: #18212a;
      --line: #263642;
      --text: #eef4f2;
      --muted: #9fb2ad;
      --accent: #62d29f;
      --accent2: #72a7ff;
      --warn: #ffcf6a;
      --bad: #ff7d7d;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
    header { padding: 28px clamp(18px, 4vw, 48px) 18px; border-bottom: 1px solid var(--line); background: #0f151a; }
    h1 { margin: 0 0 8px; font-size: clamp(28px, 4vw, 48px); letter-spacing: 0; }
    .sub { color: var(--muted); max-width: 980px; line-height: 1.5; }
    main { padding: 22px clamp(18px, 4vw, 48px) 40px; display: grid; gap: 18px; }
    .grid { display: grid; gap: 16px; }
    .cols { grid-template-columns: minmax(320px, 0.95fr) minmax(360px, 1.35fr); align-items: start; }
    .charts { grid-template-columns: repeat(2, minmax(300px, 1fr)); }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    label { display: block; font-weight: 650; margin-bottom: 8px; }
    textarea, select, input { width: 100%; background: #0b1116; color: var(--text); border: 1px solid #30424f; border-radius: 6px; padding: 11px 12px; font: inherit; }
    textarea { min-height: 154px; resize: vertical; line-height: 1.45; }
    button { background: var(--accent); border: 0; color: #04120d; font-weight: 750; border-radius: 6px; padding: 12px 14px; cursor: pointer; width: 100%; font: inherit; }
    button:disabled { opacity: 0.6; cursor: wait; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; }
    .metric { background: var(--panel2); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
    .metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .metric strong { font-size: 22px; }
    .reason { white-space: pre-wrap; line-height: 1.5; color: #dce8e3; }
    pre { background: #0b1116; border: 1px solid #30424f; border-radius: 8px; padding: 12px; overflow: auto; color: #dce8e3; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; border-bottom: 1px solid var(--line); padding: 8px; vertical-align: top; }
    th { color: var(--muted); font-weight: 700; }
    .pill { display: inline-block; padding: 4px 8px; border-radius: 999px; background: #0d2a20; color: #9ff2c7; border: 1px solid #245f49; font-size: 12px; }
    .status { color: var(--muted); min-height: 24px; }
    canvas { width: 100% !important; height: 280px !important; }
    .hidden { display: none; }
    @media (max-width: 980px) { .cols, .charts, .row { grid-template-columns: 1fr; } .metrics { grid-template-columns: repeat(2, 1fr); } }
  </style>
</head>
<body>
  <header>
    <h1>CarbonAlpha Walkthrough</h1>
    <div class="sub">A custom OpenEnv demo: choose a macro shock, let the Qwen2.5 SFT model reason into one locked allocation, then advance the 12-quarter environment yourself.</div>
  </header>
  <main>
    <section class="grid cols">
      <div class="panel">
        <label for="example">Example macro event</label>
        <select id="example"></select>
        <div style="height:12px"></div>
        <label for="news">Macro news</label>
        <textarea id="news"></textarea>
        <div class="row" style="margin-top:12px">
          <div><label for="seed">OpenEnv seed</label><input id="seed" type="number" min="0" max="999999" value="100" /></div>
          <div><label for="phase">Difficulty phase</label><select id="phase"><option value="1">Phase 1</option><option value="2">Phase 2</option><option value="3" selected>Phase 3</option></select></div>
        </div>
        <div style="height:14px"></div>
        <button id="run">Generate allocation</button>
        <div style="height:10px"></div>
        <button id="next" disabled>Next quarter</button>
        <p class="status" id="status"></p>
      </div>
      <div class="panel">
        <div class="metrics">
          <div class="metric"><span>Valid action</span><strong id="valid">-</strong></div>
          <div class="metric"><span>Current NAV</span><strong id="nav">-</strong></div>
          <div class="metric"><span>Regret reward</span><strong id="regret">-</strong></div>
          <div class="metric"><span>Carbon used</span><strong id="carbon">-</strong></div>
        </div>
        <div style="height:14px"></div>
        <span class="pill" id="meta">OpenEnv</span>
        <h2>Model Reasoning</h2>
        <div class="reason" id="reasoning">Run a walkthrough to see the model's causal reasoning.</div>
      </div>
    </section>

    <section class="grid charts">
      <div class="panel"><h2>Allocation</h2><canvas id="allocationChart"></canvas></div>
      <div class="panel"><h2>NAV vs Baseline</h2><canvas id="navChart"></canvas></div>
      <div class="panel"><h2>Carbon Timeline</h2><canvas id="carbonChart"></canvas></div>
      <div class="panel"><h2>Rewards</h2><canvas id="rewardChart"></canvas></div>
    </section>

    <section class="grid cols">
      <div class="panel"><h2>Allocation JSON</h2><pre id="actionJson">{}</pre></div>
      <div class="panel"><h2>Reward Components</h2><pre id="gradeJson">{}</pre></div>
    </section>

    <section class="panel">
      <h2>OpenEnv Episode Timeline</h2>
      <div style="overflow:auto"><table id="timeline"><thead><tr><th>Q</th><th>Shock</th><th>Regime</th><th>NAV</th><th>Baseline</th><th>Carbon</th><th>Step Reward</th><th>Narration</th></tr></thead><tbody></tbody></table></div>
    </section>
  </main>
<script>
const charts = {};
let sessionId = null;
const $ = id => document.getElementById(id);
function fmt(x, d=3){ return Number.isFinite(Number(x)) ? Number(x).toFixed(d) : '-'; }
function destroy(id){ if(charts[id]) { charts[id].destroy(); delete charts[id]; } }
function chart(id, config){ destroy(id); charts[id] = new Chart($(id), config); }
async function loadExamples(){
  const res = await fetch('/examples');
  const examples = await res.json();
  const sel = $('example');
  sel.innerHTML = '';
  Object.entries(examples).forEach(([k,v]) => {
    const opt = document.createElement('option'); opt.value=k; opt.textContent=k.replaceAll('_',' '); opt.dataset.news=v; sel.appendChild(opt);
  });
  $('news').value = sel.options[0].dataset.news;
  sel.onchange = () => $('news').value = sel.options[sel.selectedIndex].dataset.news;
}
function render(data){
  const ep = data.episode;
  const grade = ep.grade;
  const action = data.action;
  sessionId = ep.session_id;
  const done = ep.state.done === true;
  $('valid').textContent = data.raw_action ? 'yes' : 'fallback';
  $('nav').textContent = fmt(grade.final_nav_real, 3);
  $('regret').textContent = fmt(grade.r_regret, 4);
  const last = ep.timeline[ep.timeline.length-1] || {};
  $('carbon').textContent = `${fmt(last.carbon ?? 0, 2)} kg`;
  $('meta').textContent = `${ep.metadata.name} v${ep.metadata.version} | phase ${ep.metadata.phase} | seed ${ep.metadata.seed} | Q${ep.state.quarter}/12 | ${done ? 'complete' : 'active'}`;
  $('reasoning').textContent = data.reasoning || data.completion;
  $('actionJson').textContent = JSON.stringify(action, null, 2);
  $('gradeJson').textContent = JSON.stringify(grade, null, 2);
  $('next').disabled = done;

  chart('allocationChart', {type:'bar', data:{labels:data.assets, datasets:[{label:'weight', data:action.weights, backgroundColor:['#72a7ff','#ffcf6a','#62d29f','#d69cff','#b8c2cc']}]}, options:{plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true, max:1}}}});
  chart('navChart', {type:'line', data:{labels:ep.series.quarters, datasets:[{label:'agent NAV', data:ep.series.nav_real, borderColor:'#62d29f', tension:.25},{label:'baseline NAV', data:ep.series.baseline_nav_real, borderColor:'#72a7ff', tension:.25}]}, options:{responsive:true}});
  chart('carbonChart', {type:'line', data:{labels:ep.series.quarters, datasets:[{label:'carbon kg', data:ep.series.carbon, borderColor:'#ffcf6a', tension:.25},{label:'cap', data:ep.series.carbon_cap, borderColor:'#ff7d7d', borderDash:[6,6]}]}, options:{responsive:true}});
  chart('rewardChart', {type:'bar', data:{labels:ep.timeline.map(r=>'Q'+r.quarter), datasets:[{type:'bar', label:'step reward', data:ep.series.step_reward, backgroundColor:'#72a7ff99'},{type:'line', label:'cumulative', data:ep.series.cumulative_reward, borderColor:'#62d29f'}]}, options:{responsive:true}});

  const body = $('timeline').querySelector('tbody'); body.innerHTML='';
  ep.timeline.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${r.quarter}</td><td>${r.shock}</td><td>${r.regime}</td><td>${fmt(r.nav_real,3)}</td><td>${fmt(r.baseline_nav_real,3)}</td><td>${fmt(r.carbon,2)}</td><td>${fmt(r.step_reward,4)}</td><td>${r.narration}</td>`;
    body.appendChild(tr);
  });
}
async function run(){
  $('run').disabled = true; $('next').disabled = true; $('status').textContent = 'Thinking and locking one allocation...';
  try {
    const res = await fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({news:$('news').value, example:$('example').value, seed:Number($('seed').value), phase:Number($('phase').value)})});
    if(!res.ok) throw new Error(await res.text());
    render(await res.json());
    $('status').textContent = 'Allocation locked. Advance to the next quarter.';
  } catch(e) { $('status').textContent = 'Error: ' + e.message; }
  finally { $('run').disabled = false; }
}
async function nextQuarter(){
  if(!sessionId) return;
  $('next').disabled = true; $('status').textContent = 'Advancing one quarter through OpenEnv...';
  try {
    const res = await fetch('/api/step', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({session_id:sessionId})});
    if(!res.ok) throw new Error(await res.text());
    const data = await res.json();
    render(data);
    $('status').textContent = data.episode.state.done ? 'Episode complete.' : `Quarter ${data.episode.state.quarter} complete.`;
  } catch(e) { $('status').textContent = 'Error: ' + e.message; $('next').disabled = false; }
}
$('run').onclick = run;
$('next').onclick = nextQuarter;
loadExamples();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
