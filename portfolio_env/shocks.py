"""17-shock pool across 3 difficulty tiers.

Brother's TODO: concrete shocks below are v0.6 drafts. Review realism
(impact magnitudes, directional calls, news headline plausibility) and
fill in the six placeholder slots marked `PLACEHOLDER`.

Each shock:
- has a tier: easy | ambiguous | hard
- fires at a specific quarter (chosen at reset time)
- news string is shown to LLM at START of that quarter
- impacts apply as MULTIPLICATIVE return adjustments at END of quarter
- regime_shift (optional) switches inflation regime starting next quarter
- tags classify for tech_bet probability tilts ('transition_risk', 'physical_risk',
  'supply_chain', 'inflation', 'deflation')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .inflation import Regime

Tier = Literal['easy', 'ambiguous', 'hard']


@dataclass
class Shock:
    id: str
    tier: Tier
    news: str
    impacts: dict[str, float]                # asset → additive return adjustment
    regime_shift: Regime | None = None
    tags: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# EASY TIER (6 shocks) — 1-2 assets move obvious direction, rest stable
# ══════════════════════════════════════════════════════════════════════

EASY_SHOCKS: list[Shock] = [
    Shock(
        id='easy_tech_earnings',
        tier='easy',
        news='Routine earnings season. Tech majors beat estimates by 2.8% on average. '
             'Bond yields steady. No macro surprises.',
        impacts={'TECH': +0.04, 'OIL': 0.0, 'GREEN': 0.0, 'REAL_ESTATE': +0.01, 'BONDS': 0.0},
        tags=[],
    ),
    Shock(
        id='easy_oil_opec_cut',
        tier='easy',
        news='OPEC+ announces modest production cut of 500k bpd. Spot crude +4%. '
             'No broader market reaction.',
        impacts={'TECH': 0.0, 'OIL': +0.05, 'GREEN': -0.01, 'REAL_ESTATE': 0.0, 'BONDS': 0.0},
        tags=[],
    ),
    Shock(
        id='easy_green_subsidy',
        tier='easy',
        news='Germany expands solar subsidy program by €4B. European renewable '
             'manufacturers rally on policy tailwind.',
        impacts={'TECH': 0.0, 'OIL': 0.0, 'GREEN': +0.06, 'REAL_ESTATE': 0.0, 'BONDS': 0.0},
        tags=['transition_risk'],  # green-friendly
    ),
    Shock(
        id='easy_housing_cooling',
        tier='easy',
        news='US existing home sales fall 3.8% MoM on mortgage rate resistance. '
             'Housing market cooling but no dislocation.',
        impacts={'TECH': 0.0, 'OIL': 0.0, 'GREEN': 0.0, 'REAL_ESTATE': -0.03, 'BONDS': +0.01},
        tags=[],
    ),
    Shock(
        id='easy_ev_penetration',
        tier='easy',
        news='EV registrations cross 20% of European new-car sales in Q3. '
             'Traditional auto OEMs accelerate fleet electrification capex. '
             'Charging network operators report record utilization.',
        impacts={'TECH': +0.02, 'OIL': -0.03, 'GREEN': +0.07, 'REAL_ESTATE': 0.0, 'BONDS': 0.0},
        tags=['transition_risk'],
    ),
    # PLACEHOLDER — brother to fill one more easy-tier shock
    Shock(
        id='easy_PLACEHOLDER_6',
        tier='easy',
        news='PLACEHOLDER — brother fills in one more easy-tier shock.',
        impacts={'TECH': 0.0, 'OIL': 0.0, 'GREEN': 0.0, 'REAL_ESTATE': 0.0, 'BONDS': 0.0},
    ),
]


# ══════════════════════════════════════════════════════════════════════
# AMBIGUOUS TIER (7 shocks) — signals within headline conflict
# ══════════════════════════════════════════════════════════════════════

AMBIGUOUS_SHOCKS: list[Shock] = [
    Shock(
        id='ambig_hurricane_gulf',
        tier='ambiguous',
        news='Category 5 hurricane forecast for US Gulf Coast. Insurers downgrade '
             'REIT exposure. Gulf refineries at risk. FEMA preparing $80B '
             'reconstruction package. Fed hints at emergency rate cut.',
        impacts={
            'TECH':        -0.03,  # mild risk-off
            'OIL':         +0.08,  # refinery supply cut > demand hit (COUNTERINTUITIVE)
            'GREEN':       +0.12,  # reconstruction → new grid/renewables (3rd-order)
            'REAL_ESTATE': -0.25,  # direct physical damage
            'BONDS':       +0.08,  # rate-cut expectation + flight to safety
        },
        tags=['physical_risk'],
    ),
    Shock(
        id='ambig_stagflation_trigger',
        tier='ambiguous',
        news='Fed minutes leaked: PCE core unexpectedly at 5.8%. Committee signals '
             'sustained tightening into 2027. 10-year yields climb 80bp. Dollar '
             'rallies against EM. Oil services announce capacity expansion.',
        impacts={
            'TECH':        -0.10,
            'OIL':         +0.11,
            'GREEN':       -0.08,
            'REAL_ESTATE': +0.02,
            'BONDS':       -0.09,
        },
        regime_shift='stagflationary',
        tags=['inflation'],
    ),
    Shock(
        id='ambig_tech_breakthrough',
        tier='ambiguous',
        news='Major semiconductor firm demos 3nm chip with 40% power reduction. '
             'Data center operators announce capex cuts on efficiency gains. '
             'Power utility stocks hit on reduced demand forecasts.',
        impacts={
            'TECH':        +0.10,
            'OIL':          0.0,
            'GREEN':       -0.05,  # less power demand hurts renewable buildout economics
            'REAL_ESTATE': -0.04,  # data-center REIT exposure
            'BONDS':        0.0,
        },
        tags=[],
    ),
    Shock(
        id='ambig_insurance_retreat',
        tier='ambiguous',
        news='Three top-10 US insurers announce exit from Florida and California '
             'property markets citing climate-loss economics. State regulators '
             'hint at taxpayer-backed reinsurance pool. 10-year Treasury yields '
             'fall 30bp on flight-to-safety; municipal bond market freezes.',
        impacts={
            'TECH':        -0.03,   # mild risk-off
            'OIL':         -0.01,
            'GREEN':       +0.04,   # climate-adaptation capex narrative strengthens
            'REAL_ESTATE': -0.18,   # direct (obvious)
            'BONDS':       +0.09,   # flight to quality + rate-cut pricing (non-obvious wins)
        },
        tags=['physical_risk'],
    ),
    Shock(
        id='ambig_ai_efficiency',
        tier='ambiguous',
        news='Major lab demos 10× inference-efficiency gain on next-gen reasoning '
             'model; hyperscalers announce deferred GPU orders. Data-center capex '
             'forecasts revised down 40% for 2028. Power utility stocks sell off '
             'on lowered electricity-demand projections.',
        impacts={
            'TECH':        +0.09,   # software efficiency wins (1st-order obvious)
            'OIL':          0.0,
            'GREEN':       -0.06,   # renewable buildout economics hurt (data-center demand softens)
            'REAL_ESTATE': -0.08,   # data-center REIT exposure (non-obvious)
            'BONDS':       +0.02,
        },
        tags=[],
    ),
    # PLACEHOLDER — brother to fill 2 more ambiguous-tier shocks (see BROTHER_BRIEF.md §Task 1)
    Shock(
        id='ambig_PLACEHOLDER_6',
        tier='ambiguous',
        news='PLACEHOLDER — brother to fill an ambiguous-tier shock.',
        impacts={'TECH': 0.0, 'OIL': 0.0, 'GREEN': 0.0, 'REAL_ESTATE': 0.0, 'BONDS': 0.0},
    ),
    Shock(
        id='ambig_PLACEHOLDER_7',
        tier='ambiguous',
        news='PLACEHOLDER — brother to fill an ambiguous-tier shock.',
        impacts={'TECH': 0.0, 'OIL': 0.0, 'GREEN': 0.0, 'REAL_ESTATE': 0.0, 'BONDS': 0.0},
    ),
]


# ══════════════════════════════════════════════════════════════════════
# HARD TIER (4 shocks) — 2nd/3rd-order effects dominate; naive reading loses
# ══════════════════════════════════════════════════════════════════════

HARD_SHOCKS: list[Shock] = [
    Shock(
        id='hard_rare_earth_rotation',
        tier='hard',
        news='China announces 80% reduction in rare-earth exports over 18 months '
             'citing domestic demand. US semiconductor export controls tighten. '
             'Renewable manufacturers warn of 3-quarter supply chain disruption. '
             'Oil majors announce record buybacks on sector rotation inflows.',
        impacts={
            'TECH':        -0.18,
            'OIL':         +0.14,  # sector rotation wins
            'GREEN':       -0.22,  # rare-earth dependency collapses the "safe green" thesis
            'REAL_ESTATE': -0.02,
            'BONDS':       +0.05,
        },
        tags=['supply_chain', 'fragmentation', 'transition_risk'],
    ),
    Shock(
        id='hard_deflation_pulse',
        tier='hard',
        news='China manufacturing PMI crashes to 41; export prices fall 12% YoY. '
             'Global supply gluts detected across semiconductors, oil, real estate. '
             'Treasury yields plunge on safe-haven bid. Bank of Japan intervenes.',
        impacts={
            'TECH':        -0.12,
            'OIL':         -0.14,
            'GREEN':       -0.05,
            'REAL_ESTATE': -0.08,
            'BONDS':       +0.06,  # ONLY regime where bonds is the right call
        },
        regime_shift='deflationary',
        tags=['deflation'],
    ),
    Shock(
        id='hard_taiwan_water_chip',
        tier='hard',
        news='Taiwan water reservoirs hit 15% capacity; TSMC announces 30% output '
             'cut across advanced nodes over next 3 quarters. Solar panel '
             'manufacturers warn of polysilicon bottleneck. Pentagon accelerates '
             'CHIPS Act drawdown. Oil majors announce record buybacks on '
             'sector-rotation inflows.',
        impacts={
            'TECH':        -0.15,   # supply shock direct hit (1st-order)
            'OIL':         +0.11,   # rotation flows into commodities (3rd-order — naive reads this as "crisis = sell oil")
            'GREEN':       -0.13,   # solar supply chain (2nd-order non-obvious)
            'REAL_ESTATE': -0.04,
            'BONDS':       +0.04,
        },
        tags=['supply_chain', 'fragmentation', 'physical_risk'],
    ),
    Shock(
        id='hard_carbon_offset_fraud',
        tier='hard',
        news='Two leading ratings agencies publish analysis finding 40% of voluntary '
             'carbon offsets invalid (double-counting, phantom sequestration). '
             'EU CBAM enforcement agency announces retroactive audit. Offset market '
             'prices plunge 60%. Compliance-market credit prices spike 3×.',
        impacts={
            'TECH':        -0.02,
            'OIL':         -0.14,   # compliance cost just went up massively (3rd-order; naive reads "carbon stuff down = oil free")
            'GREEN':       +0.16,   # actual abatement required; renewable+CCS projects re-rated (2nd-order win)
            'REAL_ESTATE': -0.05,   # construction carbon costs rise
            'BONDS':       +0.02,
        },
        tags=['transition_risk', 'carbon_priced'],
    ),
]


ALL_SHOCKS: list[Shock] = EASY_SHOCKS + AMBIGUOUS_SHOCKS + HARD_SHOCKS

SHOCKS_BY_ID: dict[str, Shock] = {s.id: s for s in ALL_SHOCKS}
SHOCKS_BY_TIER: dict[Tier, list[Shock]] = {
    'easy':      EASY_SHOCKS,
    'ambiguous': AMBIGUOUS_SHOCKS,
    'hard':      HARD_SHOCKS,
}


def shocks_available(phase: int) -> list[Shock]:
    """Return the shock pool for a curriculum phase.
    1 = easy only, 2 = easy + ambiguous, 3 = all.
    """
    if phase == 1:
        return EASY_SHOCKS
    if phase == 2:
        return EASY_SHOCKS + AMBIGUOUS_SHOCKS
    return ALL_SHOCKS
