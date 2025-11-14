## controls.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

__all__ = [
    "ControlSet",
    "ControlCosts",
    "CostTCO",
    "annualized_cost",
    "prob_multiplier",
    "severity_multiplier",
    "control_effects",
    "total_cost",
]

# ---------------------------
# Data structures
# ---------------------------

@dataclass(frozen=True)
class ControlSet:
    server: bool = False
    media: bool = False
    error: bool = False
    external: bool = False

# --- NEW: rich TCO structure (optional) ---
@dataclass(frozen=True)
class CostTCO:
    # One-time (amortized)
    impl: float = 0.0           # implementation / PS
    hw: float = 0.0             # hardware / appliance
    training: float = 0.0
    impl_years: int = 3
    hw_years: int = 4
    training_years: int = 2
    discount_rate: float = 0.08

    # Recurring (opex)
    license_annual: float = 0.0
    cloud_annual: float = 0.0
    staff_fte: float = 0.0
    loaded_rate: float = 180_000.0  # fully loaded per FTE per year
    siem_gb_day: float = 0.0
    siem_cost_per_gb_day: float = 0.0
    other_annual: float = 0.0

    # Offsets (subtract)
    retired_annual: float = 0.0

    # Shared/platform allocation (portion of a shared platform $/yr)
    platform_alloc_annual: float = 0.0

def _annuity(pv: float, r: float, n: int) -> float:
    n = max(1, int(n))
    if r <= 0:
        return pv / n
    f = (1.0 + r) ** n
    return pv * (r * f) / (f - 1.0)

def annualized_cost(c: Union[float, CostTCO]) -> float:
    """Return annualized $ whether `c` is a plain annual number or a TCO bundle."""
    if isinstance(c, (int, float)):
        return float(c)
    if not isinstance(c, CostTCO):
        return 0.0

    one_time = (
        _annuity(c.impl,     c.discount_rate, c.impl_years) +
        _annuity(c.hw,       c.discount_rate, c.hw_years) +
        _annuity(c.training, c.discount_rate, c.training_years)
    )
    recurring = (
        c.license_annual +
        c.cloud_annual +
        c.staff_fte * c.loaded_rate +
        c.siem_gb_day * c.siem_cost_per_gb_day * 365.0 +
        c.other_annual +
        c.platform_alloc_annual
    )
    return max(0.0, one_time + recurring - c.retired_annual)

@dataclass(frozen=True)
class ControlCosts:
    server:   Union[float, CostTCO] = 0.0
    media:    Union[float, CostTCO] = 0.0
    error:    Union[float, CostTCO] = 0.0
    external: Union[float, CostTCO] = 0.0

    # Optional: apply control flags if passed; else sum all annualized
    def total(self, ctrl: ControlSet | None = None) -> float:
        if ctrl is None:
            return sum(annualized_cost(v) for v in (self.server, self.media, self.error, self.external))
        total = 0.0
        if ctrl.server:   total += annualized_cost(self.server)
        if ctrl.media:    total += annualized_cost(self.media)
        if ctrl.error:    total += annualized_cost(self.error)
        if ctrl.external: total += annualized_cost(self.external)
        return total

# ---------------------------
# Tunable effect constants
# ---------------------------

_LIKELIHOOD_MULT: Dict[str, float] = {
    "external": 0.75,
    "server":   0.85,
    "error":    0.90,
}

_SEVERITY_MULT: Dict[str, float] = {
    "media": 0.80,
}

_P_ANY_MULT: Dict[str, float] = {
    "external": 0.85,
    "error":    0.95,
}

_GPD_SCALE_MULT: Dict[str, float] = {
    "media": 0.70,
}

# ---------------------------
# Backward-compatible helpers
# ---------------------------

def prob_multiplier(ctrl: ControlSet) -> float:
    mult = 1.0
    if ctrl.external:
        mult *= _LIKELIHOOD_MULT["external"]
    if ctrl.server:
        mult *= _LIKELIHOOD_MULT["server"]
    if ctrl.error:
        mult *= _LIKELIHOOD_MULT["error"]
    return mult

def severity_multiplier(ctrl: ControlSet) -> float:
    mult = 1.0
    if ctrl.media:
        mult *= _SEVERITY_MULT["media"]
    return mult

# ---------------------------
# Causal effects for spliced model
# ---------------------------
from engine import ControlEffects  # keep import here to avoid circulars during type checking

def control_effects(ctrl: ControlSet) -> ControlEffects:
    lam_mult = prob_multiplier(ctrl)

    p_any_mult = 1.0
    if ctrl.external:
        p_any_mult *= _P_ANY_MULT["external"]
    if ctrl.error:
        p_any_mult *= _P_ANY_MULT["error"]

    gpd_scale_mult = 1.0
    if ctrl.media:
        gpd_scale_mult *= _GPD_SCALE_MULT["media"]

    return ControlEffects(
        lam_mult=lam_mult,
        p_any_mult=p_any_mult,
        gpd_scale_mult=gpd_scale_mult,
    )

def total_cost(ctrl: ControlSet, costs: ControlCosts) -> float:
    total = 0.0
    if ctrl.server:
        total += annualized_cost(costs.server)
    if ctrl.media:
        total += annualized_cost(costs.media)
    if ctrl.error:
        total += annualized_cost(costs.error)
    if ctrl.external:
        total += annualized_cost(costs.external)
    return total
def effects_from_shares_improved(ctrl: ControlSet, action_shares: Dict[str, float], pattern_shares: Dict[str, float]) -> ControlEffects:
    def _norm(d):
        s = sum(d.values()) or 1.0
        return {k: v/s for k, v in d.items()}
    a = _norm(action_shares)
    p = _norm(pattern_shares)

    lam_mult = p_any_mult = gpd_mult = 1.0
    hack = a.get("hacking",0)+p.get("Web Applications",0)+p.get("Crimeware",0)
    misuse = a.get("misuse",0)+p.get("Privilege Misuse",0)
    err = a.get("error",0)+p.get("Miscellaneous Errors",0)
    phys = a.get("physical",0)+p.get("Lost and Stolen Assets",0)

    if ctrl.server:
        lam_mult *= (1 - 0.35*hack)
        p_any_mult *= (1 - 0.20*hack)
        gpd_mult *= (1 - 0.15*hack)
    if ctrl.media:
        lam_mult *= (1 - 0.25*phys)
        p_any_mult *= (1 - 0.25*phys)
    if ctrl.error:
        lam_mult *= (1 - 0.20*err)
        p_any_mult *= (1 - 0.25*err)
    if ctrl.external:
        lam_mult *= (1 - 0.30*(hack+misuse))
        gpd_mult *= (1 - 0.20*(hack+misuse))

    import numpy as np
    lam_mult = float(np.clip(lam_mult, 0.2, 1.0))
    p_any_mult = float(np.clip(p_any_mult, 0.2, 1.0))
    gpd_mult = float(np.clip(gpd_mult, 0.2, 1.0))
    return ControlEffects(lam_mult, p_any_mult, gpd_mult)
