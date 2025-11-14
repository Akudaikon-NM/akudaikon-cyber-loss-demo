# engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "SplicedParams",
    "FreqParams",
    "ModelConfig",
    "ControlEffects",
    "posterior_lambda",
    "build_spliced_from_priors",
    "sample_spliced",
    "sample_frequency",
    "simulate_annual_losses",
    "compute_metrics",
    "lec",
    "lec_bands",
]

# ---------------------------------------------------------------------
# Severity: Spliced Lognormal (body) + GPD (tail)
# ---------------------------------------------------------------------

@dataclass
class SplicedParams:
    """Parameters for a simple spliced loss model."""
    thr_q: float = 0.95       # quantile threshold that splits body vs tail
    thr: float = 0.0          # absolute threshold (computed from thr_q)
    mu: float = 10.0          # lognormal mu (log scale)
    sigma: float = 1.0        # lognormal sigma
    gpd_c: float = 0.3        # GPD shape (xi)
    gpd_scale: float = 2e5    # GPD scale (beta)


def _fit_spliced_from_losses(losses: np.ndarray, q: float = 0.95) -> SplicedParams:
    """
    Fit a spliced body/tail model from synthetic losses.
    Body: Lognormal fit to values <= threshold.
    Tail:  GPD fit (excesses over threshold).
    """
    losses = np.asarray(losses, dtype=float)
    thr = float(np.quantile(losses, q))

    body = losses[losses <= thr]
    tail = losses[losses > thr] - thr

    # Guardrails for pathological samples
    body = body[body > 0.0]
    if body.size < 20:
        body = np.maximum(losses, 1.0)

    # Lognormal on body (fit normal to logs)
    mu, sigma = stats.norm.fit(np.log(body))

    # GPD on tail (fallback if few tail points)
    if tail.size < 10:
        gpd_c, gpd_scale = 0.25, max(thr, 1.0) * 0.5
    else:
        gpd_c, _loc, gpd_scale = stats.genpareto.fit(tail, floc=0.0)

    return SplicedParams(thr_q=q, thr=thr, mu=mu, sigma=sigma, gpd_c=gpd_c, gpd_scale=gpd_scale)


def sample_spliced(n: int, p: SplicedParams, seed: Optional[int] = None) -> np.ndarray:
    """Draw severities from the spliced distribution."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    out = np.empty(n, dtype=float)

    mask_body = u <= p.thr_q
    # Body ~ Lognormal
    out[mask_body] = np.exp(rng.normal(p.mu, p.sigma, mask_body.sum()))
    # Tail ~ thr + GPD
    ut = rng.random((~mask_body).sum())
    out[~mask_body] = p.thr + stats.genpareto.ppf(ut, p.gpd_c, 0.0, p.gpd_scale)
    return out


# ---------------------------------------------------------------------
# Frequency: hurdle Bernoulli(any) x (Poisson or NegBin) count
# ---------------------------------------------------------------------

@dataclass
class FreqParams:
    """Frequency model for annual incident counts."""
    lam: float                 # mean incidents/year
    p_any: float = 0.85        # hurdle: probability there is any incident this year
    negbin: bool = False       # if True, use NegBin for overdispersion
    r: float = 1.5             # NegBin dispersion (larger r => closer to Poisson)


def sample_frequency(n_years: int, fp: FreqParams, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # draw raw incident counts
    if fp.negbin:
        p = fp.r / (fp.r + fp.lam) if (fp.r + fp.lam) > 0 else 1.0
        counts = stats.nbinom.rvs(fp.r, p, size=n_years, random_state=rng)
    else:
        counts = rng.poisson(fp.lam, size=n_years)
    # per-incident “paid loss” thinning
    if 0.0 <= fp.p_any < 1.0:
        paid = rng.binomial(counts, fp.p_any)
    else:
        paid = counts
    return paid

def posterior_lambda(
    alpha0: float,
    beta0: float,
    k: int,
    T: float,
    draws: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Gamma(alpha0, beta0) prior for lambda; observe k incidents in T years.
    Returns samples from posterior lambda ~ Gamma(alpha0 + k, beta0 + T).
    """
    rng = np.random.default_rng(seed)
    alpha_post = alpha0 + max(k, 0)
    beta_post = beta0 + max(T, 0.0)
    return rng.gamma(alpha_post, 1.0 / beta_post, size=draws)


# ---------------------------------------------------------------------
# Model config and control effects
# ---------------------------------------------------------------------

@dataclass
class ModelConfig:
    trials: int
    net_worth: float
    seed: int = 42
    record_cap: int = 250_000
    cost_per_record: float = 175.0


@dataclass
class ControlEffects:
    """
    Multiplicative effects applied to frequency and tail severity.
    1.0 means no change.
    """
    lam_mult: float = 1.0
    p_any_mult: float = 1.0
    gpd_scale_mult: float = 1.0


def build_spliced_from_priors(cfg: ModelConfig) -> SplicedParams:
    """
    Lightweight prior for severity:
    Draw synthetic losses from Beta fraction of records * cost_per_record,
    then fit the spliced (lognormal + GPD) model. Replace with a real fit
    when you wire in your empirical dataset.
    """
    rng = np.random.default_rng(cfg.seed + 9)
    frac = rng.beta(0.8, 8.0, size=50_000)  # many small, few large
    base_losses = np.minimum(frac * cfg.record_cap, cfg.record_cap) * cfg.cost_per_record
    return _fit_spliced_from_losses(base_losses, q=0.95)


def simulate_annual_losses(
    cfg: ModelConfig,
    fp: FreqParams,
    sp: SplicedParams,
    ce: ControlEffects = ControlEffects()
) -> np.ndarray:
    """
    Simulate a distribution of annual losses with causal control effects:
      - Frequency: scale lambda and p_any
      - Severity tail: scale GPD scale parameter
    """
    # Apply control effects to frequency
    lam = max(fp.lam * ce.lam_mult, 0.0)
    p_any = float(np.clip(fp.p_any * ce.p_any_mult, 0.0, 1.0))

    # Apply control effects to severity tail
    sp2 = SplicedParams(**sp.__dict__)
    sp2.gpd_scale = max(sp.gpd_scale * ce.gpd_scale_mult, 1.0)

    rng = np.random.default_rng(cfg.seed)
    counts = sample_frequency(
        cfg.trials,
        FreqParams(lam=lam, p_any=p_any, negbin=fp.negbin, r=fp.r),
        seed=cfg.seed + 1,
    )

    # Pre-sample a pool of severities for speed (expand if needed)
    max_events = int(np.clip(counts.max(), 0, 10_000))
    sev_pool = sample_spliced(max(1, max_events * 4), sp2, seed=cfg.seed + 2)

    annual = np.zeros(cfg.trials, dtype=float)
    cursor = 0
    for i, k in enumerate(counts):
        if k == 0:
            continue
        need = int(k)
        if cursor + need > sev_pool.size:
            # Refill the pool if exhausted
            refill = sample_spliced(max(need * 2, 1000), sp2, seed=int(rng.integers(1_000_000_000)))
            sev_pool = np.concatenate([sev_pool, refill])
        annual[i] = sev_pool[cursor: cursor + need].sum()
        cursor += need

    return annual


# ---------------------------------------------------------------------
# Metrics and curves
# ---------------------------------------------------------------------

def compute_metrics(losses: np.ndarray, net_worth: float) -> Dict[str, float]:
    x = np.asarray(losses, float)
    eal  = float(np.mean(x)) if x.size else 0.0
    v95  = float(np.percentile(x, 95)) if x.size else 0.0
    v99  = float(np.percentile(x, 99)) if x.size else 0.0
    cvar = float(x[x >= v95].mean())   if x.size else 0.0
    mx   = float(np.max(x))            if x.size else 0.0
    pr   = float(np.mean(x >= net_worth)) if (x.size and net_worth > 0) else 0.0
    return dict(EAL=eal, VaR95=v95, VaR99=v99, CVaR95=cvar, Max=mx, P(Ruin)=pr)


def lec(losses: np.ndarray, n: int = 200) -> pd.DataFrame:
    x = np.asarray(losses, float)
    grid = np.quantile(x, np.linspace(0.01, 0.999, n))
    ex = (x[:, None] >= grid[None, :]).mean(axis=0)
    return pd.DataFrame({"Loss": grid, "Exceedance_Prob": ex})

def lec_bands(samples: np.ndarray, n: int = 200, level: float = 0.90) -> pd.DataFrame:
    """
    Credible bands for the LEC.
    samples: array of shape (S, T) where S is posterior draws and T is trials per draw.
    Returns a grid of losses with (lo, median, hi) exceedance probabilities.
    """
    samples = np.asarray(samples, dtype=float)
    pooled = samples.reshape(-1)
    grid = np.quantile(pooled, np.linspace(0.01, 0.999, n))

    # Exceedance per posterior draw
    ex = np.empty((samples.shape[0], grid.size), dtype=float)
    for i, s in enumerate(samples):
        ex[i] = (s[:, None] >= grid[None, :]).mean(axis=0)

    alpha = (1.0 - level) / 2.0
    lo = np.quantile(ex, alpha, axis=0)
    md = np.quantile(ex, 0.5, axis=0)
    hi = np.quantile(ex, 1.0 - alpha, axis=0)

    return pd.DataFrame({"loss": grid, "lo": lo, "median": md, "hi": hi})
