# engine.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy import stats

# ---------- Severity: Spliced Lognormal (body) + GPD (tail) ----------
@dataclass
class SplicedParams:
    thr_q: float = 0.95         # quantile threshold used to separate body/tail
    thr: float = 0.0            # absolute threshold (computed)
    mu: float = 10.0            # lognormal mu (on log scale)
    sigma: float = 1.0          # lognormal sigma
    gpd_c: float = 0.3          # GPD shape (xi)
    gpd_scale: float = 2e5      # GPD scale (beta)

def _fit_spliced_from_losses(losses: np.ndarray, q: float=0.95) -> SplicedParams:
    thr = np.quantile(losses, q)
    body = losses[losses <= thr]
    tail = losses[losses > thr] - thr
    # guardrails
    body = body[body > 0]
    if len(body) < 20:  # fallback if insufficient data
        body = np.maximum(losses, 1.0)

    # lognormal fit on body
    mu, sigma = stats.norm.fit(np.log(body))
    # GPD fit on tail
    if len(tail) < 10:
        c, scale = 0.25, np.maximum(thr, 1.0) * 0.5
    else:
        c, loc, scale = stats.genpareto.fit(tail, floc=0.0)
    return SplicedParams(thr_q=q, thr=thr, mu=mu, sigma=sigma, gpd_c=c, gpd_scale=scale)

def sample_spliced(n: int, p: SplicedParams, seed: Optional[int]=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    out = np.empty(n)
    mask_body = u <= p.thr_q
    # body ~ Lognormal
    out[mask_body] = np.exp(rng.normal(p.mu, p.sigma, mask_body.sum()))
    # tail ~ thr + GPD
    ut = rng.random((~mask_body).sum())
    out[~mask_body] = p.thr + stats.genpareto.ppf(ut, p.gpd_c, 0, p.gpd_scale)
    return out

# ---------- Frequency: Hurdle Poisson / NegBin with optional Bayesian λ ----------
@dataclass
class FreqParams:
    lam: float                 # mean incidents/year
    p_any: float = 0.85        # hurdle: prob(any incident this year)
    negbin: bool = False       # use NegBin for overdispersion
    r: float = 1.5             # NegBin dispersion

def sample_frequency(n_years: int, fp: FreqParams, seed: Optional[int]=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    any_breach = rng.binomial(1, fp.p_any, n_years)
    if fp.negbin:
        # NegBin with mean=lam and dispersion r  -> p = r/(r+lam)
        p = fp.r / (fp.r + fp.lam)
        counts = stats.nbinom.rvs(fp.r, p, size=n_years, random_state=rng)
    else:
        counts = rng.poisson(fp.lam, n_years)
    # hurdle: zero or at least 1
    return np.where(any_breach == 0, 0, np.maximum(counts, 1))

def posterior_lambda(alpha0: float, beta0: float, k: int, T: float, draws: int, seed: Optional[int]=None) -> np.ndarray:
    """Gamma(alpha0, beta0) prior; observe k incidents over T years; returns lambda samples."""
    rng = np.random.default_rng(seed)
    alpha_post = alpha0 + k
    beta_post = beta0 + T
    return rng.gamma(alpha_post, 1.0/beta_post, size=draws)

# ---------- Loss model glue ----------
@dataclass
class ModelConfig:
    trials: int
    net_worth: float
    seed: int = 42
    record_cap: int = 250_000
    cost_per_record: float = 175.0

@dataclass
class ControlEffects:
    # Multipliers applied to frequency & tail; keep 1.0 = no change
    lam_mult: float = 1.0
    p_any_mult: float = 1.0
    gpd_scale_mult: float = 1.0

def build_spliced_from_priors(cfg: ModelConfig) -> SplicedParams:
    """
    Simple prior that maps 'records × $/record' into a starting loss sample,
    then fits a spliced distribution. Replace with real fitting when your dataset is wired.
    """
    rng = np.random.default_rng(cfg.seed + 9)
    # Draw synthetic losses from Beta fraction of records × cost_per_record
    frac = rng.beta(0.8, 8.0, size=50_000)  # skewed toward smaller events; long tail via splice
    base_losses = np.minimum(frac * cfg.record_cap, cfg.record_cap) * cfg.cost_per_record
    return _fit_spliced_from_losses(base_losses, q=0.95)

def simulate_annual_losses(
    cfg: ModelConfig,
    fp: FreqParams,
    sp: SplicedParams,
    ce: ControlEffects = ControlEffects()
) -> np.ndarray:
    """
    Simulate distribution of annual loss with controls applied causally:
    - frequency: lam and p_any scaling
    - severity tail: GPD scale scaling
    """
    # apply control effects
    lam = max(fp.lam * ce.lam_mult, 0.0)
    p_any = np.clip(fp.p_any * ce.p_any_mult, 0.0, 1.0)

    sp2 = SplicedParams(**{**sp.__dict__})
    sp2.gpd_scale = max(sp.gpd_scale * ce.gpd_scale_mult, 1.0)

    rng = np.random.default_rng(cfg.seed)
    counts = sample_frequency(cfg.trials, FreqParams(lam=lam, p_any=p_any, negbin=fp.negbin, r=fp.r), seed=cfg.seed+1)
    # sample all severities up front (upper bound) for speed
    max_events = int(np.clip(counts.max(), 0, 10_000))
    sev_pool = sample_spliced(max(1, max_events * 4), sp2, seed=cfg.seed+2)

    annual = np.zeros(cfg.trials)
    cursor = 0
    for i, k in enumerate(counts):
        if k == 0:
            continue
        need = int(k)
        if cursor + need > len(sev_pool):
            # refill
            sev_pool = np.concatenate([sev_pool, sample_spliced(max(need * 2, 1000), sp2, seed=rng.integers(1e9))])
        annual[i] = sev_pool[cursor:cursor+need].sum()
        cursor += need

    return annual

# ---------- Metrics & curves ----------
def compute_metrics(losses: np.ndarray, net_worth: float) -> Dict[str, float]:
    losses = np.asarray(losses)
    eal = float(np.mean(losses))
    v95 = float(np.quantile(losses, 0.95))
    v99 = float(np.quantile(losses, 0.99))
    return dict(
        EAL=eal,
        VaR95=v95,
        VaR99=v99,
        VaR95_to_NetWorth=v95 / net_worth if net_worth > 0 else np.nan,
        VaR99_to_NetWorth=v99 / net_worth if net_worth > 0 else np.nan
    )

def lec(losses: np.ndarray, n: int=200) -> pd.DataFrame:
    """Loss Exceedance Curve: P(Loss ≥ x)."""
    x = np.quantile(losses, np.linspace(0.01, 0.999, n))
    probs = (losses[:, None] >= x[None, :]).mean(axis=0)
    return pd.DataFrame({"loss": x, "exceed_prob": probs})

def lec_bands(samples: np.ndarray, n: int=200, level: float=0.90) -> pd.DataFrame:
    """
    Credible bands for LEC: samples is (S, T) array of annual losses from S posterior draws.
    Returns loss grid & (lower, median, upper) exceed prob.
    """
    # build a common grid from pooled losses
    pooled = samples.reshape(-1)
    grid = np.quantile(pooled, np.linspace(0.01, 0.999, n))
    # compute exceed probs per posterior sample
    ex = []
    for s in samples:
        ex.append((s[:, None] >= grid[None, :]).mean(axis=0))
    ex = np.stack(ex, axis=0)
    lo = np.quantile(ex, (1-level)/2, axis=0)
    md = np.quantile(ex, 0.5, axis=0)
    hi = np.quantile(ex, 1-(1-level)/2, axis=0)
    return pd.DataFrame({"loss": grid, "lo": lo, "median": md, "hi": hi})
