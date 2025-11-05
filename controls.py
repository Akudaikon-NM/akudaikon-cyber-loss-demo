# controls.py
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
    """
    Annualized costs for each control family.
    You may pass either a plain annual number (float) or a CostTCO.
    """
    server:   Union[float, CostTCO] = 0.0
    media:    Union[float, CostTCO] = 0.0
    error:    Union[float, CostTCO] = 0.0
    external: Union[float, CostTCO] = 0.0

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
