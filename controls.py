# controls.py
from dataclasses import dataclass

@dataclass
class ControlSet:
    server: bool = False
    media: bool = False
    error: bool = False
    external: bool = False

@dataclass
class ControlCosts:
    server: float = 0.0
    media: float = 0.0
    error: float = 0.0
    external: float = 0.0

def prob_multiplier(ctrl: ControlSet) -> float:
    """
    Backward-compatible: flatten to a single λ multiplier.
    """
    mult = 1.0
    if ctrl.external: mult *= 0.75   # MFA/external surface
    if ctrl.server:   mult *= 0.85   # patching/hardening
    if ctrl.error:    mult *= 0.90   # change control
    # media primarily affects *severity tail*; leave prob here
    return mult

def severity_multiplier(ctrl: ControlSet) -> float:
    """
    Backward-compatible: flatten to scalar severity multiplier.
    """
    mult = 1.0
    if ctrl.media:    mult *= 0.80   # encryption/DLP bends tail
    return mult

# New: causal effects used by spliced model
from engine import ControlEffects
def control_effects(ctrl: ControlSet) -> ControlEffects:
    lam_mult = prob_multiplier(ctrl)
    # P(any) moves with external and error (less phish/misconfig seasons)
    p_any_mult = 1.0
    if ctrl.external: p_any_mult *= 0.85
    if ctrl.error:    p_any_mult *= 0.95
    # Tail severity scale responds to media
    gpd_scale_mult = 1.0
    if ctrl.media:    gpd_scale_mult *= 0.70
    return ControlEffects(lam_mult=lam_mult, p_any_mult=p_any_mult, gpd_scale_mult=gpd_scale_mult)

def total_cost(ctrl: ControlSet, costs: ControlCosts) -> float:
    tot = 0.0
    if ctrl.server:   tot += costs.server
    if ctrl.media:    tot += costs.media
    if ctrl.error:    tot += costs.error
    if ctrl.external: tot += costs.external
    return tot
