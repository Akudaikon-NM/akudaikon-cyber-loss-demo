from dataclasses import dataclass

@dataclass
class ControlSet:
    server: bool = False
    media: bool = False
    error: bool = False
    external: bool = False

@dataclass
class ControlCosts:
    server: float = 50000.0
    media: float = 30000.0
    error: float = 40000.0
    external: float = 100000.0

def prob_multiplier(ctrl: ControlSet) -> float:
    m = 1.0
    if ctrl.server:   m *= 0.85
    if ctrl.media:    m *= 0.95
    if ctrl.error:    m *= 0.85
    if ctrl.external: m *= 0.70
    return m

def severity_multiplier(ctrl: ControlSet) -> float:
    m = 1.0
    if ctrl.server:   m *= 0.95
    if ctrl.media:    m *= 0.90
    if ctrl.error:    m *= 0.90
    if ctrl.external: m *= 0.95
    return m

def total_cost(ctrl: ControlSet, costs: ControlCosts) -> float:
    c = 0.0
    if ctrl.server:   c += costs.server
    if ctrl.media:    c += costs.media
    if ctrl.error:    c += costs.error
    if ctrl.external: c += costs.external
    return c
