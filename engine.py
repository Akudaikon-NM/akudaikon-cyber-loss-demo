from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

def draw_severity(num_customers: int,
                  cost_per_customer: float,
                  severity_scale: float = 1.0,
                  rng: Optional[np.random.Generator] = None) -> float:
    rng = rng or np.random.default_rng()
    frac = rng.beta(0.7, 5.0)
    affected = int(min(num_customers, max(0, frac * num_customers)))
    return severity_scale * affected * cost_per_customer

def simulate_annual_losses(trials: int,
                           lam: float,
                           num_customers: int,
                           cost_per_customer: float,
                           severity_scale: float = 1.0,
                           seed: Optional[int] = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=lam, size=trials)
    losses = np.zeros(trials, dtype=float)
    for i, n in enumerate(counts):
        if n == 0:
            continue
        fracs = rng.beta(0.7, 5.0, size=n)
        affected = np.minimum(num_customers, np.maximum(0, (fracs * num_customers).astype(int)))
        losses[i] = severity_scale * affected.sum() * cost_per_customer
    return losses

def compute_metrics(losses: np.ndarray, net_worth: float) -> dict:
    losses = np.asarray(losses, dtype=float)
    eal  = float(losses.mean())
    var95 = float(np.percentile(losses, 95))
    var99 = float(np.percentile(losses, 99))
    return {
        "EAL": eal,
        "VaR95": var95,
        "VaR99": var99,
        "VaR95_to_NetWorth": (var95 / net_worth) if net_worth > 0 else np.nan,
        "VaR99_to_NetWorth": (var99 / net_worth) if net_worth > 0 else np.nan,
    }

def lec(losses: np.ndarray, num_points: int = 200) -> pd.DataFrame:
    losses = np.asarray(losses, dtype=float)
    x = np.quantile(losses, np.linspace(0, 0.999, num_points))
    exceed = 1.0 - np.searchsorted(np.sort(losses), x, side="right") / losses.size
    return pd.DataFrame({"loss": x, "exceed_prob": exceed})
