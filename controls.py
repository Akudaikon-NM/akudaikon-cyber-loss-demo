# controls.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# Public API surface exported by this module
__all__ = [
    "ControlSet",
    "ControlCosts",
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
    """
    Toggle which control families are active in the portfolio.

    server   : hardening/patching, config baselines
    media    : encryption/DLP/handling that primarily bends the tail
    error    : change/config discipline (reduces blunders/misconfigs)
    external : identity/MFA and perimeter-facing reductions
    """
    server: bool = False
    media: bool = False
    error: bool = False
    external: bool = False


@dataclass(frozen=True)
class ControlCosts:
    """
    Annualized costs for each control family (USD/year).
    """
    server: float = 0.0
    media: float = 0.0
    error: float = 0.0
    external: float = 0.0


# ---------------------------
# Tunable effect constants
# ---------------------------
# Likelihood multipliers applied to lambda (and Poisson-like frequency).
_LIKELIHOOD_MULT: Dict[str, float] = {
    "external": 0.75,  # MFA / external attack surface
    "server":   0.85,  # patching / hardening
    "error":    0.90,  # change control / guardrails
    # "media" is omitted here (severity-focused)
}

# Scalar multiplier applied to severity (non-parametric fallback).
_SEVERITY_MULT: Dict[str, float] = {
    "media": 0.80,  # encryption/DLP bends the tail
}

# Spliced/causal model parameter multipliers:
# - p_any controls the chance of any loss mass in a year
# - gpd_scale bends the tail in the extreme-value component
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
    """
    Backward-compatible: collapse active controls to a single lambda multiplier.
    Media is excluded since it primarily affects severity tails.
    """
    mult = 1.0
    if ctrl.external:
        mult *= _LIKELIHOOD_MULT["external"]
    if ctrl.server:
        mult *= _LIKELIHOOD_MULT["server"]
    if ctrl.error:
        mult *= _LIKELIHOOD_MULT["error"]
    return mult


def severity_multiplier(ctrl: ControlSet) -> float:
    """
    Backward-compatible: collapse active controls to a single scalar
    applied to severity draws.
    """
    mult = 1.0
    if ctrl.media:
        mult *= _SEVERITY_MULT["media"]
    return mult


# ---------------------------
# Causal effects for spliced model
# ---------------------------
from engine import ControlEffects  # keep import here to avoid circulars during type checking


def control_effects(ctrl: ControlSet) -> ControlEffects:
    """
    Map a ControlSet to parameter-level causal effects used by the spliced model.
    Returns an engine.ControlEffects with:
      - lam_mult      : frequency (lambda) multiplier
      - p_any_mult    : probability of any-loss-in-year multiplier
      - gpd_scale_mult: tail scale multiplier (extreme-value severity)
    """
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
    """
    Sum annualized costs for the active control portfolio.
    """
    total = 0.0
    if ctrl.server:
        total += costs.server
    if ctrl.media:
        total += costs.media
    if ctrl.error:
        total += costs.error
    if ctrl.external:
        total += costs.external
    return total
