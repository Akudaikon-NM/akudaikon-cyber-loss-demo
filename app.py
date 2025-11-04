# --- Imports from the model/engine modules ---
from engine import (
    ModelConfig, FreqParams, SplicedParams,
    build_spliced_from_priors, simulate_annual_losses,
    compute_metrics, lec, lec_bands, posterior_lambda
)
from controls import ControlSet, ControlCosts, control_effects, total_cost

# --- Stdlib / 3rd party ---
import io
import os
import json
import tempfile
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="Akudaikon | Cyber-Loss Demo", layout="wide")
st.title("Akudaikon | Cyber-Loss Demo")
st.caption("Monte Carlo loss model with control ROI and optional Bayesian frequency.")

# ------------------------------------------------------------------------------------
# Configuration (no more magic numbers)
# ------------------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    MIN_LAMBDA_MULTIPLIER: float = 0.30  # Max 70% reduction
    MIN_PANY_MULTIPLIER: float = 0.40    # Max 60% reduction
    MIN_GPD_SCALE_MULTIPLIER: float = 0.50  # Max 50% reduction

    CREDIBLE_INTERVAL_LEVEL: float = 0.90
    POSTERIOR_DRAWS: int = 200
    LEC_POINTS: int = 200

    CONVERGENCE_CV_THRESHOLD: float = 0.05
    BATCH_SIZE_FRACTION: float = 0.10

CFG = SimulationConfig()

# ------------------------------------------------------------------------------------
# Utility: CSV security + uploads validation (OWASP-ish)
# ------------------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB cap

def _escape_csv_injection(val):
    if isinstance(val, str) and val and val[0] in ("=", "+", "-", "@"):
        return "'" + val
    return val

def _safe_to_csv(df: pd.DataFrame) -> str:
    df_safe = df.copy()
    obj_cols = df_safe.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        df_safe[obj_cols] = df_safe[obj_cols].applymap(_escape_csv_injection)
    buf = io.StringIO(newline="")
    df_safe.to_csv(buf, index=False)
    return buf.getvalue()

def _validate_upload(file, label: str):
    if file is None:
        return
    if getattr(file, "size", 0) > MAX_UPLOAD_BYTES:
        st.error(f"{label}: file is too large (> {MAX_UPLOAD_BYTES // (1024*1024)}MB).")
        st.stop()
    ctype = getattr(file, "type", "") or getattr(file, "content_type", "")
    if ctype and "csv" not in ctype.lower() and "text" not in ctype.lower():
        st.error(f"{label}: expected a CSV (got {ctype}).")
        st.stop()

# ------------------------------------------------------------------------------------
# Core helpers: validation, diagnostics, bootstrap CIs, convergence, multi-year
# ------------------------------------------------------------------------------------
def validate_losses(losses: np.ndarray, name: str) -> np.ndarray:
    if losses is None or len(losses) == 0:
        raise ValueError(f"{name}: Empty loss array")
    if np.any(np.isnan(losses)):
        raise ValueError(f"{name}: Contains NaN values")
    if np.any(np.isinf(losses)):
        raise ValueError(f"{name}: Contains infinite values")
    if np.any(losses < 0):
        raise ValueError(f"{name}: Contains negative losses")
    return losses

def check_convergence(losses: np.ndarray, metric: str = "EAL") -> dict:
    n = len(losses)
    if n < 1000:
        return {"converged": False, "cv": np.inf, "recommendation": "increase trials"}
    batch_size = max(1, n // 10)
    # make exact 10 batches
    n10 = batch_size * 10
    x = losses[:n10].reshape(10, batch_size).mean(axis=1)
    cv = float(np.std(x) / np.mean(x)) if np.mean(x) > 0 else np.inf
    return {
        "converged": cv < CFG.CONVERGENCE_CV_THRESHOLD,
        "cv": cv,
        "recommendation": "increase trials" if cv >= CFG.CONVERGENCE_CV_THRESHOLD else "adequate"
    }

def var_confidence_interval(losses: np.ndarray, alpha: float = 0.95, n_bootstrap: int = 1000, seed: int = 42) -> dict:
    n = len(losses)
    var_point = float(np.percentile(losses, alpha * 100))
    rng = np.random.default_rng(seed)
    boot_vars = []
    for _ in range(n_bootstrap):
        sample = rng.choice(losses, size=n, replace=True)
        boot_vars.append(np.percentile(sample, alpha * 100))
    boot_vars = np.asarray(boot_vars)
    return {
        "point": var_point,
        "ci_lower": float(np.percentile(boot_vars, 2.5)),
        "ci_upper": float(np.percentile(boot_vars, 97.5)),
        "stderr": float(np.std(boot_vars)),
    }

def multi_year_simulation(cfg: ModelConfig, fp: FreqParams, sp: SplicedParams, years: int = 5) -> np.ndarray:
    annual_losses = np.zeros((cfg.trials, years))
    for year in range(years):
        annual_losses[:, year] = simulate_annual_losses(cfg, fp, sp)
    return annual_losses.cumsum(axis=1)

def monte_carlo_diagnostics(losses: np.ndarray) -> dict:
    n = len(losses)
    def _autocorr(x, lag):
        if lag >= len(x):
            return 0.0
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    autocorr_1 = _autocorr(losses, 1)
    ess = n / (1 + 2 * max(0.0, autocorr_1)) if n > 1 else 1
    tail_5pct = losses[losses > np.percentile(losses, 95)]
    tail_ratio = len(tail_5pct) / n if n > 0 else 0.0
    zero_ratio = float((losses == 0).sum()) / n if n > 0 else 0.0
    mc_se_eal = float(np.std(losses) / np.sqrt(n)) if n > 0 else np.inf
    mean_losses = float(np.mean(losses)) if n > 0 else 0.0
    rel_err = (mc_se_eal / mean_losses) if mean_losses > 0 else np.inf
    return {
        "ess": float(ess),
        "ess_ratio": float(ess / n) if n > 0 else 0.0,
        "tail_ratio": float(tail_ratio),
        "zero_ratio": float(zero_ratio),
        "mc_se_eal": float(mc_se_eal),
        "rel_error_eal": float(rel_err),
    }

# ------------------------------------------------------------------------------------
# Control effects: shares + diminishing returns
# ------------------------------------------------------------------------------------
def _normalize_shares(raw: Mapping[str, float]) -> dict:
    pairs = [(k.strip(), float(v)) for k, v in (raw or {}).items() if k and v is not None]
    total = sum(max(0.0, v) for _, v in pairs)
    if total <= 0:
        return {}
    return {k: (max(0.0, v) / total) for k, v in pairs if v > 0}

DEFAULT_ACTION_SHARES = _normalize_shares({
    "Error": 0.35, "Hacking": 0.25, "Misuse": 0.25, "Social": 0.10, "Physical": 0.05
})
DEFAULT_PATTERN_SHARES = _normalize_shares({
    "Privilege Misuse": 0.40, "Basic Web App Attacks": 0.30, "Misc Errors": 0.30
})

ACTION_IMPACT_BY_CONTROL = {
    "external": {"Hacking": 0.30, "Social": 0.25},
    "error": {"Error": 0.25},
    "server": {"Hacking": 0.15, "Misuse": 0.05},
    "media": {},
}
PATTERN_TAIL_IMPACT_BY_CONTROL = {
    "media": {"Misc Errors": 0.35, "Privilege Misuse": 0.15},
    "server": {"Basic Web App Attacks": 0.10},
    "external": {},
    "error": {},
}

def effects_from_shares_improved(
    ctrl: ControlSet,
    action_shares: Optional[Mapping[str, float]] = None,
    pattern_shares: Optional[Mapping[str, float]] = None,
    interaction_discount: float = 0.15
):
    from engine import ControlEffects
    a_sh = _normalize_shares(action_shares or DEFAULT_ACTION_SHARES)
    p_sh = _normalize_shares(pattern_shares or DEFAULT_PATTERN_SHARES)

    lam_mult = 1.0
    p_any_mult = 1.0
    gpd_scale_mult = 1.0

    def apply_action(control_key: str):
        nonlocal lam_mult, p_any_mult
        impact = ACTION_IMPACT_BY_CONTROL.get(control_key, {})
        if not impact:
            return
        for act, strength in impact.items():
            share = a_sh.get(act, 0.0)
            factor = max(0.0, 1.0 - share * strength)
            lam_mult *= factor
            p_any_mult *= factor

    def apply_pattern(control_key: str):
        nonlocal gpd_scale_mult
        impact = PATTERN_TAIL_IMPACT_BY_CONTROL.get(control_key, {})
        if not impact:
            return
        for patt, strength in impact.items():
            share = p_sh.get(patt, 0.0)
            factor = max(0.0, 1.0 - share * strength)
            gpd_scale_mult *= factor

    if getattr(ctrl, "external", False):
        apply_action("external"); apply_pattern("external")
    if getattr(ctrl, "error", False):
        apply_action("error");    apply_pattern("error")
    if getattr(ctrl, "server", False):
        apply_action("server");   apply_pattern("server")
    if getattr(ctrl, "media", False):
        apply_action("media");    apply_pattern("media")

    # Floors (caps on max reduction)
    lam_mult = max(CFG.MIN_LAMBDA_MULTIPLIER, lam_mult)
    p_any_mult = max(CFG.MIN_PANY_MULTIPLIER, p_any_mult)
    gpd_scale_mult = max(CFG.MIN_GPD_SCALE_MULTIPLIER, gpd_scale_mult)

    # Diminishing returns across multiple controls
    n_active = sum([getattr(ctrl, c, False) for c in ["server", "media", "error", "external"]])
    if n_active > 1:
        interaction_factor = 1.0 + (n_active - 1) * interaction_discount
        lam_mult = 1.0 - (1.0 - lam_mult) / interaction_factor
        p_any_mult = 1.0 - (1.0 - p_any_mult) / interaction_factor
        gpd_scale_mult = 1.0 - (1.0 - gpd_scale_mult) / interaction_factor

    return ControlEffects(lam_mult=lam_mult, p_any_mult=p_any_mult, gpd_scale_mult=gpd_scale_mult)

# ------------------------------------------------------------------------------------
# Bayesian: propagate parameter uncertainty end-to-end
# ------------------------------------------------------------------------------------
def full_bayesian_simulation(
    cfg: ModelConfig, fp_prior: FreqParams, sp: SplicedParams,
    alpha0: float, beta0: float, k_obs: int, T_obs: float,
    ce=None, n_samples: int = 100
):
    lam_samples = posterior_lambda(alpha0, beta0, k_obs, T_obs, draws=n_samples, seed=cfg.seed)
    baseline = []
    controlled = []
    for lam_i in lam_samples:
        fp_i = FreqParams(lam=float(lam_i), p_any=fp_prior.p_any, negbin=fp_prior.negbin, r=fp_prior.r)
        loss_b = simulate_annual_losses(cfg, fp_i, sp)
        baseline.append(loss_b)
        if ce is not None:
            loss_c = simulate_annual_losses(cfg, fp_i, sp, ce)
        else:
            loss_c = loss_b.copy()
        controlled.append(loss_c)

    baseline = np.stack(baseline, axis=0)     # (n_samples, trials)
    controlled = np.stack(controlled, axis=0) # (n_samples, trials)

    mean_eals_b = baseline.mean(axis=1)
    mean_eals_c = controlled.mean(axis=1)

    return {
        "lam_samples": lam_samples,
        "baseline_samples": baseline,
        "controlled_samples": controlled,
        "baseline_eal_mean": float(mean_eals_b.mean()),
        "baseline_eal_ci": (
            float(np.percentile(mean_eals_b, 2.5)),
            float(np.percentile(mean_eals_b, 97.5)),
        ),
        "controlled_eal_mean": float(mean_eals_c.mean()),
        "controlled_eal_ci": (
            float(np.percentile(mean_eals_c, 2.5)),
            float(np.percentile(mean_eals_c, 97.5)),
        ),
    }

# ------------------------------------------------------------------------------------
# Distributions & scenarios
# ------------------------------------------------------------------------------------
def plot_loss_distributions(base_losses, ctrl_losses):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=base_losses[base_losses > 0], name="Baseline", opacity=0.6, nbinsx=50))
    fig.add_trace(go.Histogram(x=ctrl_losses[ctrl_losses > 0], name="Controlled", opacity=0.6, nbinsx=50))
    fig.update_xaxes(type="log", title="Annual Loss (USD)")
    fig.update_yaxes(title="Frequency")
    fig.update_layout(barmode="overlay", title="Loss Distribution Comparison (log scale)")
    return fig

def compare_control_scenarios(cfg, fp, sp, costs, action_shares, pattern_shares):
    scenarios = []
    control_names = ["server", "media", "error", "external"]
    for i in range(16):  # 2^4
        ctrl = ControlSet(
            server=(i & 1) > 0,
            media=(i & 2) > 0,
            error=(i & 4) > 0,
            external=(i & 8) > 0
        )
        try:
            ce = effects_from_shares_improved(ctrl, action_shares, pattern_shares)
        except Exception:
            ce = control_effects(ctrl)
        losses = simulate_annual_losses(cfg, fp, sp, ce)
        metrics = compute_metrics(losses, cfg.net_worth)
        cost = total_cost(ctrl, costs)
        # ROSI vs baseline EAL is better, but we lack baseline here; use EAL-cost over cost as heuristic.
        rosi = ((metrics["EAL"] - cost) / cost * 100.0) if cost > 0 else np.nan
        active = [name for j, name in enumerate(control_names) if (i & (1 << j)) > 0]
        scenarios.append({
            "scenario": " + ".join(active) if active else "None",
            "n_controls": len(active),
            "EAL": metrics["EAL"],
            "VaR95": metrics["VaR95"],
            "cost": cost,
            "ROSI": rosi
        })
    return pd.DataFrame(scenarios).sort_values("ROSI", ascending=False)

# ------------------------------------------------------------------------------------
# App Toggle (risk layer)
# ------------------------------------------------------------------------------------
mode = st.sidebar.radio("Risk mode", ("Cyber Breach (records-based)", "AI Incidents (monetary)"), index=0)

# ------------------------------------------------------------------------------------
# Data-driven control shares (sidebar)
# ------------------------------------------------------------------------------------
with st.sidebar.expander("Data-driven control effects (shares)", expanded=False):
    st.caption("Use NAICS-52 demo shares or upload CSVs to weight control effects by ACTION/PATTERN.")
    shares_mode = st.radio("Shares source", ["Built-in NAICS-52 (demo)", "Upload CSVs"], index=0, key="shares_mode")
    action_shares = DEFAULT_ACTION_SHARES
    pattern_shares = DEFAULT_PATTERN_SHARES
    if shares_mode == "Upload CSVs":
        up_actions = st.file_uploader("Upload action shares CSV (columns: category, share)", type=["csv"], key="up_actions")
        up_patterns = st.file_uploader("Upload pattern shares CSV (columns: category, share)", type=["csv"], key="up_patterns")
        _validate_upload(up_actions, "Action shares CSV")
        _validate_upload(up_patterns, "Pattern shares CSV")

        def _read_shares(file) -> dict:
            try:
                df_u = pd.read_csv(file)
                cat_col = next((c for c in df_u.columns if c.lower() in ["category", "action", "pattern", "name"]), None)
                share_col = next((c for c in df_u.columns if "share" in c.lower() or "weight" in c.lower()), None)
                if not cat_col or not share_col:
                    st.warning("CSV must have columns like [category, share]. Using defaults.")
                    return {}
                return _normalize_shares(dict(zip(df_u[cat_col], df_u[share_col])))
            except Exception as e:
                st.warning(f"Could not parse shares CSV: {e}. Using defaults.")
                return {}

        if up_actions is not None:
            tmp = _read_shares(up_actions)
            if tmp: action_shares = tmp
        if up_patterns is not None:
            tmp = _read_shares(up_patterns)
            if tmp: pattern_shares = tmp

    if action_shares:
        st.write("**Action shares**")
        st.dataframe(pd.DataFrame({"action": list(action_shares.keys()), "share": list(action_shares.values())}))
    if pattern_shares:
        st.write("**Pattern shares**")
        st.dataframe(pd.DataFrame({"pattern": list(pattern_shares.keys()), "share": list(pattern_shares.values())}))

st.session_state["_action_shares"] = action_shares
st.session_state["_pattern_shares"] = pattern_shares

# ------------------------------------------------------------------------------------
# Advanced frequency (sidebar, persistent)
# ------------------------------------------------------------------------------------
with st.sidebar.expander("Advanced frequency", expanded=False):
    use_bayes   = st.checkbox("Bayesian lambda (Gamma prior + your data)", value=False, key="adv_use_bayes")
    alpha0      = st.number_input("lambda prior alpha", min_value=0.01, max_value=50.0, value=2.0, step=0.1, key="adv_alpha0")
    beta0       = st.number_input("lambda prior beta",  min_value=0.01, max_value=50.0, value=8.0, step=0.1, key="adv_beta0")
    k_obs       = st.number_input("Incidents observed (k)", min_value=0, max_value=100000, value=0, step=1, key="adv_k_obs")
    T_obs       = st.number_input("Observation years (T)",  min_value=0.0, max_value=200.0, value=0.0, step=0.5, key="adv_T_obs")
    use_negbin  = st.checkbox("Use Negative Binomial (overdispersion)", value=False, key="adv_use_negbin")
    disp_r      = st.number_input("NegBin dispersion r", min_value=0.5, max_value=10.0, value=1.5, step=0.1, key="adv_disp_r")

st.markdown("**Calibration (from dataset slice)**")
ALPHA_MIN = 0.01
BETA_MIN  = 0.01
STEP      = 0.1
PREC      = 4

_defaults = {"adv_use_bayes": False, "adv_alpha0": 2.0, "adv_beta0": 8.0, "adv_pseudo_w": 2.0}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

def _round_to_step(x: float, step: float = STEP, prec: int = PREC) -> float:
    return round(step * round(x / step), prec)

def seed_prior_cb():
    k = float(st.session_state.get("adv_k_obs", 0))
    T = float(st.session_state.get("adv_T_obs", 0.0))
    w = float(st.session_state.get("adv_pseudo_w", 2.0))
    lam_hat = (k / T) if T > 0 else 0.0
    alpha_suggest = _round_to_step(max(ALPHA_MIN, lam_hat * w))
    beta_suggest  = _round_to_step(max(BETA_MIN,  w))
    st.session_state.update({
        "adv_alpha0": float(max(ALPHA_MIN, alpha_suggest)),
        "adv_beta0":  float(max(BETA_MIN,  beta_suggest)),
        "adv_use_bayes": True,
    })

if T_obs and T_obs > 0:
    lam_hat = float(k_obs) / float(T_obs)
    st.caption(f"λ̂ (k/T) = {lam_hat:.4f} incidents/year")
    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.number_input("Pseudo-years (weight for prior)",
                        min_value=0.1, max_value=50.0,
                        value=float(st.session_state["adv_pseudo_w"]),
                        step=0.1, key="adv_pseudo_w")
    _alpha_preview = _round_to_step(max(ALPHA_MIN, lam_hat * float(st.session_state["adv_pseudo_w"])))
    _beta_preview  = _round_to_step(max(BETA_MIN,  float(st.session_state["adv_pseudo_w"])))
    with cols[1]:
        st.write(f"Suggested α₀ = max({ALPHA_MIN}, λ̂·w) → **{_alpha_preview:.4f}**")
        st.write(f"Suggested β₀ = max({BETA_MIN}, w) → **{_beta_preview:.4f}**")
    with cols[2]:
        disabled = (lam_hat <= 0.0)
        tip = "Need k>0 (or T>0) for informative seeding." if disabled else "Apply and enable Bayesian frequency."
        st.button("Apply prior α₀=λ̂·w, β₀=w", use_container_width=True,
                  disabled=disabled, help=tip, on_click=seed_prior_cb)
else:
    st.caption("Provide k and T to compute λ̂ (and optionally seed a weak prior).")

# ------------------------------------------------------------------------------------
# NAICS presets (Finance & Insurance)
# ------------------------------------------------------------------------------------
NAICS_FINANCE_PRESETS = {
    "521110 — Monetary Authorities (Central Bank)": {"lambda": 0.35, "records_cap": 1_000_000, "cost_per_record": 185.0, "net_worth": 5_000_000_000.0},
    "522110 — Commercial Banking": {"lambda": 0.60, "records_cap": 5_000_000, "cost_per_record": 185.0, "net_worth": 2_000_000_000.0},
    "522120 — Savings Institutions": {"lambda": 0.45, "records_cap": 1_500_000, "cost_per_record": 185.0, "net_worth": 800_000_000.0},
    "522130 — Credit Unions": {"lambda": 0.35, "records_cap": 250_000, "cost_per_record": 185.0, "net_worth": 100_000_000.0},
    "522190 — Other Depository Credit Intermediation": {"lambda": 0.45, "records_cap": 1_000_000, "cost_per_record": 185.0, "net_worth": 500_000_000.0},
    "522210 — Credit Card Issuing": {"lambda": 0.55, "records_cap": 3_000_000, "cost_per_record": 185.0, "net_worth": 1_000_000_000.0},
    "522220 — Sales Financing": {"lambda": 0.40, "records_cap": 1_000_000, "cost_per_record": 175.0, "net_worth": 400_000_000.0},
    "522291 — Consumer Lending": {"lambda": 0.45, "records_cap": 1_500_000, "cost_per_record": 185.0, "net_worth": 600_000_000.0},
    "522292 — Real Estate Credit (incl. Mortgage Lending)": {"lambda": 0.40, "records_cap": 2_000_000, "cost_per_record": 185.0, "net_worth": 800_000_000.0},
    "522293 — International Trade Financing": {"lambda": 0.35, "records_cap": 500_000, "cost_per_record": 175.0, "net_worth": 700_000_000.0},
    "522294 — Secondary Market Financing": {"lambda": 0.35, "records_cap": 3_000_000, "cost_per_record": 175.0, "net_worth": 1_500_000_000.0},
    "522298 — All Other Nondepository Credit Intermediation": {"lambda": 0.35, "records_cap": 800_000, "cost_per_record": 175.0, "net_worth": 300_000_000.0},
    "522310 — Mortgage & Nonmortgage Loan Brokers": {"lambda": 0.30, "records_cap": 600_000, "cost_per_record": 175.0, "net_worth": 150_000_000.0},
    "522320 — Financial Transactions Processing / Reserve / Clearinghouse": {"lambda": 0.65, "records_cap": 8_000_000, "cost_per_record": 200.0, "net_worth": 1_500_000_000.0},
    "522390 — Other Activities Related to Credit Intermediation": {"lambda": 0.30, "records_cap": 500_000, "cost_per_record": 175.0, "net_worth": 200_000_000.0},
    "523110 — Investment Banking & Securities Dealing": {"lambda": 0.45, "records_cap": 1_500_000, "cost_per_record": 185.0, "net_worth": 2_000_000_000.0},
    "523120 — Securities Brokerage": {"lambda": 0.45, "records_cap": 2_500_000, "cost_per_record": 185.0, "net_worth": 1_200_000_000.0},
    "523130 — Commodity Contracts Dealing": {"lambda": 0.35, "records_cap": 500_000, "cost_per_record": 175.0, "net_worth": 500_000_000.0},
    "523140 — Commodity Contracts Brokerage": {"lambda": 0.35, "records_cap": 800_000, "cost_per_record": 175.0, "net_worth": 600_000_000.0},
    "523210 — Securities & Commodity Exchanges": {"lambda": 0.40, "records_cap": 1_000_000, "cost_per_record": 185.0, "net_worth": 2_500_000_000.0},
    "523910 — Miscellaneous Intermediation": {"lambda": 0.35, "records_cap": 600_000, "cost_per_record": 175.0, "net_worth": 250_000_000.0},
    "523920 — Portfolio Management": {"lambda": 0.35, "records_cap": 1_200_000, "cost_per_record": 175.0, "net_worth": 900_000_000.0},
    "523930 — Investment Advice": {"lambda": 0.30, "records_cap": 400_000, "cost_per_record": 175.0, "net_worth": 150_000_000.0},
    "523991 — Trust, Fiduciary & Custody Activities": {"lambda": 0.35, "records_cap": 1_000_000, "cost_per_record": 185.0, "net_worth": 700_000_000.0},
    "523999 — Miscellaneous Financial Investment Activities": {"lambda": 0.30, "records_cap": 500_000, "cost_per_record": 175.0, "net_worth": 200_000_000.0},
    "524113 — Direct Life Insurance Carriers": {"lambda": 0.50, "records_cap": 3_000_000, "cost_per_record": 210.0, "net_worth": 1_500_000_000.0},
    "524114 — Direct Health & Medical Insurance Carriers": {"lambda": 0.55, "records_cap": 4_000_000, "cost_per_record": 250.0, "net_worth": 1_800_000_000.0},
    "524126 — Direct Property & Casualty Insurance Carriers": {"lambda": 0.45, "records_cap": 2_000_000, "cost_per_record": 200.0, "net_worth": 1_500_000_000.0},
    "524127 — Direct Title Insurance Carriers": {"lambda": 0.35, "records_cap": 1_000_000, "cost_per_record": 185.0, "net_worth": 600_000_000.0},
    "524128 — Other Direct Insurance Carriers": {"lambda": 0.40, "records_cap": 1_500_000, "cost_per_record": 200.0, "net_worth": 900_000_000.0},
    "524210 — Insurance Agencies & Brokerages": {"lambda": 0.30, "records_cap": 600_000, "cost_per_record": 185.0, "net_worth": 150_000_000.0},
    "524291 — Claims Adjusting": {"lambda": 0.30, "records_cap": 500_000, "cost_per_record": 185.0, "net_worth": 120_000_000.0},
    "524292 — Third-Party Administration of Insurance & Pension Funds": {"lambda": 0.40, "records_cap": 1_500_000, "cost_per_record": 200.0, "net_worth": 400_000_000.0},
    "524298 — All Other Insurance Related Activities": {"lambda": 0.30, "records_cap": 500_000, "cost_per_record": 185.0, "net_worth": 120_000_000.0},
    "525110 — Pension Funds": {"lambda": 0.35, "records_cap": 2_000_000, "cost_per_record": 200.0, "net_worth": 2_000_000_000.0},
    "525120 — Health & Welfare Funds": {"lambda": 0.40, "records_cap": 2_500_000, "cost_per_record": 230.0, "net_worth": 1_200_000_000.0},
    "525190 — Other Insurance Funds": {"lambda": 0.35, "records_cap": 1_500_000, "cost_per_record": 210.0, "net_worth": 900_000_000.0},
    "525910 — Open-End Investment Funds": {"lambda": 0.35, "records_cap": 1_500_000, "cost_per_record": 175.0, "net_worth": 1_500_000_000.0},
    "525920 — Trusts, Estates & Agency Accounts": {"lambda": 0.30, "records_cap": 800_000, "cost_per_record": 185.0, "net_worth": 700_000_000.0},
    "525990 — Other Financial Vehicles": {"lambda": 0.30, "records_cap": 1_000_000, "cost_per_record": 175.0, "net_worth": 1_000_000_000.0},
}

# ------------------------------------------------------------------------------------
# BRANCH 1: CYBER BREACH (records-based)
# ------------------------------------------------------------------------------------
if mode == "Cyber Breach (records-based)":

    # NAICS presets
    with st.sidebar.expander("Finance NAICS presets", expanded=False):
        use_naics = st.checkbox("Use preset", value=False, key="naics_enable")
        _keys = list(NAICS_FINANCE_PRESETS.keys())
        _default_label = "522130 — Credit Unions"
        _default_index = _keys.index(_default_label) if _default_label in _keys else 0
        choice = st.selectbox("Select NAICS (Finance)", _keys, index=_default_index,
                              disabled=not use_naics, key="naics_choice")
        if use_naics:
            p = NAICS_FINANCE_PRESETS[choice]
            st.session_state["in_lambda"]      = p["lambda"]
            st.session_state["in_records_cap"] = p["records_cap"]
            st.session_state["in_cpr"]         = p["cost_per_record"]
            st.session_state["in_networth"]    = p["net_worth"]
            st.caption(f"Preset applied: {choice}")

    # Scenario + Controls form
    with st.sidebar.form("scenario_form"):
        st.header("Scenario")
        trials            = st.number_input("Simulation trials", min_value=1_000, max_value=500_000, value=50_000, step=5_000, key="in_trials")
        net_worth         = st.number_input("Net worth (USD)", min_value=0.0, value=1_000_000.0, step=100_000.0, format="%.0f", key="in_networth")
        seed              = st.number_input("Random seed", min_value=0, value=42, step=1, key="in_seed")
        num_customers     = st.number_input("Records / customers cap", min_value=1, value=1_000_000, step=10_000, key="in_records_cap")
        cost_per_customer = st.number_input("Cost per record (USD)", min_value=1.0, value=150.0, step=10.0, format="%.2f", key="in_cpr")
        lam               = st.number_input("Annual incident rate (lambda)", min_value=0.0, value=0.40, step=0.05, format="%.2f", key="in_lambda")

        st.markdown("---")
        st.subheader("Controls")
        ctrl = ControlSet(
            server   = st.checkbox("Server hardening / patching", value=False, key="ctl_server"),
            media    = st.checkbox("Media protection / encryption/DLP", value=False, key="ctl_media"),
            error    = st.checkbox("Change control / error-proofing", value=False, key="ctl_error"),
            external = st.checkbox("External / MFA & perimeter", value=False, key="ctl_external"),
        )

        with st.expander("Control costs (USD/yr)", expanded=False):
            costs = ControlCosts(
                server   = st.number_input("Server cost",   min_value=0.0, value=80_000.0,  step=1_000.0, format="%.0f", key="cost_server"),
                media    = st.number_input("Media cost",    min_value=0.0, value=90_000.0,  step=1_000.0, format="%.0f", key="cost_media"),
                error    = st.number_input("Error cost",    min_value=0.0, value=60_000.0,  step=1_000.0, format="%.0f", key="cost_error"),
                external = st.number_input("External cost", min_value=0.0, value=100_000.0, step=1_000.0, format="%.0f", key="cost_external"),
            )

        st.caption(f"Selected controls annual cost: ${total_cost(ctrl, costs):,.0f}")
        submitted = st.form_submit_button("Run simulation", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Simulating…"):
            # Config
            cfg = ModelConfig(
                trials=int(trials),
                net_worth=float(net_worth),
                seed=int(seed),
                record_cap=int(num_customers),
                cost_per_record=float(cost_per_customer),
            )

            # Frequency params (point prior)
            fp = FreqParams(lam=float(lam), p_any=0.85, negbin=bool(use_negbin), r=float(disp_r))
            sp: SplicedParams = build_spliced_from_priors(cfg)

            # Effects with data-driven shares + diminishing returns
            ash = st.session_state.get("_action_shares", DEFAULT_ACTION_SHARES)
            psh = st.session_state.get("_pattern_shares", DEFAULT_PATTERN_SHARES)
            try:
                ce = effects_from_shares_improved(ctrl, ash, psh)
            except Exception:
                ce = control_effects(ctrl)

            # --- FULL BAYES vs fallback ---
            if use_bayes and T_obs > 0:
                st.info("Running full Bayesian simulation (parameter uncertainty propagated)…")
                bayes = full_bayesian_simulation(
                    cfg, fp, sp, float(alpha0), float(beta0), int(k_obs), float(T_obs),
                    ce=ce, n_samples=min(100, CFG.POSTERIOR_DRAWS)
                )
                base_losses = bayes["baseline_samples"].reshape(-1)     # posterior-mixture
                ctrl_losses = bayes["controlled_samples"].reshape(-1)
                lam_draws = bayes["lam_samples"]

                # Posterior EAL summaries
               st.markdown("---")
st.subheader("Sensitivity analysis")

with st.expander("Run sensitivity analysis", expanded=False):
    st.caption("Vary one frequency parameter ±50% and observe EAL & VaR95.")
    sens_param = st.selectbox(
        "Parameter",
        ["lam", "p_any", "r"],  # incident rate, prob(any loss), NegBin dispersion
        index=0
    )
    include_controls = st.checkbox(
        "Include current control effects",
        value=True,
        help="If checked, sensitivity uses the same control effects as the main run."
    )

    if st.button("Run sensitivity"):
        try:
            sens_df = run_sensitivity_analysis(
                cfg, fp, sp, ctrl, costs, sens_param,
                ce if include_controls else None
            )

            fig_s = go.Figure()
            fig_s.add_scatter(
                x=sens_df["param_value"], y=sens_df["EAL"],
                mode="lines+markers", name="EAL"
            )
            fig_s.add_scatter(
                x=sens_df["param_value"], y=sens_df["VaR95"],
                mode="lines+markers", name="VaR95"
            )
            fig_s.update_layout(
                title=f"Sensitivity of EAL & VaR95 to {sens_param}",
                xaxis_title=f"{sens_param} value",
                yaxis_title="USD"
            )
            st.plotly_chart(fig_s, use_container_width=True)

            st.dataframe(
                sens_df.rename(columns={"param_value": f"{sens_param}_value"})
                      .style.format({
                          "multiplier": "{:.2f}",
                          f"{sens_param}_value": "{:,.4f}",
                          "EAL": "${:,.0f}",
                          "VaR95": "${:,.0f}",
                      }),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Sensitivity run failed: {e}")

            # KPI tiles
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("EAL (Baseline)",   f"${base_m['EAL']:,.0f}")
            c2.metric("EAL (Controlled)", f"${ctrl_m['EAL']:,.0f}", delta=f"-${delta_eal:,.0f}")
            c3.metric("VaR95 (Base→Ctrl)", f"${base_m['VaR95']:,.0f}", delta=f"-${(base_m['VaR95'] - ctrl_m['VaR95']):,.0f}")
            c4.metric("VaR99 (Base→Ctrl)", f"${base_m['VaR99']:,.0f}", delta=f"-${(base_m['VaR99'] - ctrl_m['VaR99']):,.0f}")

            d1, d2, d3 = st.columns(3)
            d1.metric("VaR95 / Net Worth (Base)", f"{base_m['VaR95_to_NetWorth']*100:,.2f}%")
            d2.metric("VaR95 / Net Worth (Ctrl)", f"{ctrl_m['VaR95_to_NetWorth']*100:,.2f}%")
            d3.metric("ROSI (annualized)", "—" if np.isnan(rosi) else f"{rosi:,.1f}%")

            # Convergence check
            conv = check_convergence(base_losses)
            if not conv["converged"]:
                st.warning(f"⚠️ Simulation may not be converged (CV={conv['cv']:.3f}). Consider increasing trials.")

            st.markdown("---")

            # LEC (with optional credible bands)
            lec_b = lec(base_losses, n=CFG.LEC_POINTS).assign(scenario="Baseline")
            lec_c = lec(ctrl_losses, n=CFG.LEC_POINTS).assign(scenario="Controlled")
            fig = go.Figure()
            fig.add_scatter(x=lec_b["loss"], y=lec_b["exceed_prob"], mode="lines", name="Baseline")
            fig.add_scatter(x=lec_c["loss"], y=lec_c["exceed_prob"], mode="lines", name="Controlled")

            if use_bayes and T_obs > 0 and lam_draws is not None:
                S = min(80, len(lam_draws))
                # Baseline bands
                samples = []
                for i in range(S):
                    fp_i = FreqParams(lam=float(lam_draws[i]), p_any=fp.p_any, negbin=fp.negbin, r=fp.r)
                    samples.append(simulate_annual_losses(cfg, fp_i, sp))
                samples = np.stack(samples, axis=0)
                band_b = lec_bands(samples, n=CFG.LEC_POINTS, level=CFG.CREDIBLE_INTERVAL_LEVEL)
                fig.add_scatter(x=band_b["loss"], y=band_b["hi"], mode="lines", name="Baseline 90% hi",
                                line=dict(width=0.5), showlegend=False)
                fig.add_scatter(x=band_b["loss"], y=band_b["lo"], mode="lines", name="Baseline 90% lo",
                                line=dict(width=0.5), fill="tonexty", fillcolor="rgba(0,0,0,0.08)", showlegend=False)
                # Controlled bands
                samples_c = []
                for i in range(S):
                    fp_i = FreqParams(lam=float(lam_draws[i]), p_any=fp.p_any, negbin=fp.negbin, r=fp.r)
                    samples_c.append(simulate_annual_losses(cfg, fp_i, sp, ce))
                samples_c = np.stack(samples_c, axis=0)
                band_c = lec_bands(samples_c, n=CFG.LEC_POINTS, level=CFG.CREDIBLE_INTERVAL_LEVEL)
                fig.add_scatter(x=band_c["loss"], y=band_c["hi"], mode="lines", name="Controlled 90% hi",
                                line=dict(width=0.5), showlegend=False)
                fig.add_scatter(x=band_c["loss"], y=band_c["lo"], mode="lines", name="Controlled 90% lo",
                                line=dict(width=0.5), fill="tonexty", fillcolor="rgba(0,0,0,0.08)", showlegend=False)

            fig.update_layout(title="Loss Exceedance Curve (LEC) with Optional Credible Bands",
                              xaxis_title="Annual Loss (USD)", yaxis_title="P(Loss >= x)")
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log", range=[-2.5, 0])
            st.plotly_chart(fig, use_container_width=True)

            # Bootstrap CI for VaR95 (sample-size aware)
            var95_ci = var_confidence_interval(base_losses, 0.95, n_bootstrap=800, seed=seed)
            st.metric("VaR95 (95% CI)",
                      f"${var95_ci['point']:,.0f}",
                      delta=f"±${(var95_ci['ci_upper'] - var95_ci['ci_lower'])/2:,.0f}")

            # Summary table
            st.subheader("Summary")
            summary_df = pd.DataFrame({
                "Metric": ["EAL", "VaR95", "VaR99", "VaR95/NetWorth", "VaR99/NetWorth", "Control Cost", "Delta EAL", "ROSI %"],
                "Baseline":  [base_m["EAL"], base_m["VaR95"], base_m["VaR99"], base_m["VaR95_to_NetWorth"], base_m["VaR99_to_NetWorth"], np.nan, np.nan, np.nan],
                "Controlled":[ctrl_m["EAL"], ctrl_m["VaR95"], ctrl_m["VaR99"], ctrl_m["VaR95_to_NetWorth"], ctrl_m["VaR99_to_NetWorth"], total_cost(ctrl, costs), delta_eal, rosi],
            })
            st.dataframe(summary_df.style.format({"Baseline": "{:,.2f}", "Controlled": "{:,.2f}"}), use_container_width=True)

            # Distributions
            st.plotly_chart(plot_loss_distributions(base_losses, ctrl_losses), use_container_width=True)

            # Sensitivity analysis (lambda)
            if st.checkbox("Run sensitivity analysis"):
                with st.expander("Sensitivity to Lambda"):
                    base_value = fp.lam
                    variations = [0.5, 0.75, 1.0, 1.25, 1.5]
                    rows = []
                    for mult in variations:
                        fp_test = FreqParams(lam=base_value * mult, p_any=fp.p_any, negbin=fp.negbin, r=fp.r)
                        losses = simulate_annual_losses(cfg, fp_test, sp)
                        m = compute_metrics(losses, cfg.net_worth)
                        rows.append({"multiplier": mult, "param_value": base_value * mult, "EAL": m["EAL"], "VaR95": m["VaR95"]})
                    sens_df = pd.DataFrame(rows)
                    fig_s = px.line(sens_df, x="param_value", y="EAL", title="EAL Sensitivity to Lambda")
                    st.plotly_chart(fig_s, use_container_width=True)
                    st.dataframe(sens_df)

            # Multi-year horizon
            horizon = st.slider("Risk horizon (years)", 1, 10, 1)
            if horizon > 1:
                multi_losses = multi_year_simulation(cfg, fp, sp, horizon)
                final_year = multi_losses[:, -1]
                st.metric(f"{horizon}-Year Cumulative VaR95", f"${np.percentile(final_year, 95):,.0f}")

            # Compare all control combos (16)
            if st.button("Compare all control combinations"):
                with st.spinner("Running 16 scenarios..."):
                    comparison = compare_control_scenarios(cfg, fp, sp, costs, ash, psh)
                    st.dataframe(comparison, use_container_width=True)
                    best = comparison.iloc[0]
                    st.success(f"🏆 Best ROSI: {best['scenario']} ({best['ROSI']:.1f}%)")

            # Diagnostics (sidebar)
            with st.sidebar.expander("📊 Simulation Diagnostics"):
                diag = monte_carlo_diagnostics(base_losses)
                st.metric("Effective Sample Size", f"{diag['ess']:.0f}")
                st.metric("Relative Error (EAL)", f"{diag['rel_error_eal']*100:.2f}%")
                if diag['rel_error_eal'] > 0.05:
                    st.warning("High relative error - increase trials")

            # Download: annual samples + full JSON report
            out_df = pd.DataFrame({"annual_loss_baseline": base_losses, "annual_loss_controlled": ctrl_losses})
            csv_text = _safe_to_csv(out_df)
            st.download_button("Download annual losses (CSV)", csv_text, "cyber_annual_losses.csv", "text/csv")

            def generate_full_report(cfg, fp, sp, ctrl, costs, base_losses, ctrl_losses):
                from datetime import datetime
                report = {
                    "metadata": {"timestamp": datetime.now().isoformat(), "trials": cfg.trials, "seed": cfg.seed},
                    "configuration": {
                        "net_worth": cfg.net_worth,
                        "frequency": {"lambda": fp.lam, "p_any": fp.p_any, "negbin": fp.negbin, "r": fp.r},
                        "controls": {"active": [k for k, v in ctrl.__dict__.items() if v], "total_cost": total_cost(ctrl, costs)}
                    },
                    "results": {"baseline": compute_metrics(base_losses, cfg.net_worth), "controlled": compute_metrics(ctrl_losses, cfg.net_worth)},
                    "diagnostics": {"baseline": monte_carlo_diagnostics(base_losses), "controlled": monte_carlo_diagnostics(ctrl_losses)},
                    "raw_data": {"baseline_losses": base_losses.tolist(), "controlled_losses": ctrl_losses.tolist()},
                }
                return report

            report = generate_full_report(cfg, fp, sp, ctrl, costs, base_losses, ctrl_losses)
            st.download_button("📄 Download Full Report (JSON)", json.dumps(report, indent=2), "cyber_risk_report.json", "application/json")

# ------------------------------------------------------------------------------------
# BRANCH 2: AI INCIDENTS (monetary)
# ------------------------------------------------------------------------------------
else:
    st.header("AI Incidents | Monetary Risk")
    st.caption("AIID incidents enriched with policy context → EAL, VaR95/99, LEC, and ROI.")

    # Inputs
    c1, c2 = st.columns(2)
    enriched_up = c1.file_uploader("Enriched incidents CSV", type=["csv"], accept_multiple_files=False)
    hai62_up    = c2.file_uploader("HAI 6.2 join-pack CSV", type=["csv"], accept_multiple_files=False)
    _validate_upload(enriched_up, "Enriched incidents CSV")
    _validate_upload(hai62_up, "HAI 6.2 join-pack CSV")

    c3, c4, c5 = st.columns(3)
    min_conf = c3.slider("Min loss confidence (for training $ severity)", 0.0, 1.0, 0.70, 0.05)
    trials   = int(c4.selectbox("Monte Carlo trials", [2000, 5000, 10000, 20000], index=2))
    seed     = int(c5.number_input("Random seed", value=42, step=1))

    # Decide data source
    from pathlib import Path
    DATA_DIR   = Path(__file__).resolve().parent / "data"
    DEF_ENRICH = DATA_DIR / "incidents.csv"
    DEF_HAI62  = DATA_DIR / "joinpack_hai_6_2.csv"

    if (enriched_up is not None and hai62_up is not None):
        source = "uploads"
    elif DEF_ENRICH.exists() and DEF_HAI62.exists():
        source = "repo"
    else:
        source = "demo"

    def _lec_dataframe(losses: np.ndarray, n: int = 200) -> pd.DataFrame:
        lo = max(1.0, float(np.percentile(losses, 1)))
        hi = float(np.percentile(losses, 99.9))
        if hi <= lo:
            hi = lo * 10.0
        xs = np.logspace(np.log10(lo), np.log10(hi), n)
        probs = [(losses >= x).mean() for x in xs]
        return pd.DataFrame({"loss": xs, "prob_exceed": probs})

    def _simulate_ai_demo(trials: int, seed: int, lam: float = 0.45, sev_mu: float = 11.5, sev_sigma: float = 1.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        k = rng.poisson(lam=lam, size=trials)
        m = int(k.max()) if trials > 0 else 0
        if m == 0:
            return np.zeros(trials)
        sev = rng.lognormal(mean=sev_mu, sigma=sev_sigma, size=(trials, m))
        mask = np.arange(m)[None, :] < k[:, None]
        return (sev * mask).sum(axis=1)

    def _metrics_from_losses(losses: np.ndarray) -> tuple[float, float, float]:
        return (float(losses.mean()), float(np.percentile(losses, 95)), float(np.percentile(losses, 99)))

    # Normalizer for AIID/HAI CSVs
    def normalize_aiid_csvs(enriched_src, hai62_src):
        inc = pd.read_csv(enriched_src)
        hai = pd.read_csv(hai62_src)
        if "incident_id" not in hai.columns and "id" in hai.columns:
            hai = hai.rename(columns={"id": "incident_id"})
        if "year" not in hai.columns:
            year_source = None
            if {"incident_id","year"}.issubset(inc.columns):
                year_source = inc[["incident_id","year"]].drop_duplicates()
            else:
                for date_col in ["published", "date", "incident_date", "event_date", "report_date"]:
                    if date_col in inc.columns:
                        yy = pd.to_datetime(inc[date_col], errors="coerce").dt.year
                        if yy.notna().any():
                            inc = inc.assign(year=yy)
                            year_source = inc[["incident_id","year"]].drop_duplicates()
                            break
            if year_source is not None:
                hai = hai.merge(year_source, on="incident_id", how="left")
        for c in ["loss_usd", "loss_confidence", "severity_proxy", "life_deployment"]:
            if c in hai.columns:
                hai[c] = pd.to_numeric(hai[c], errors="coerce")
        bin_cols = [c for c in hai.columns if c.startswith(("domain_", "mod_", "fig_6_2_"))]
        for c in bin_cols:
            hai[c] = (pd.to_numeric(hai[c], errors="coerce").fillna(0) > 0).astype(int)
        if "loss_confidence" in hai.columns:
            hai["loss_confidence"] = hai["loss_confidence"].clip(0, 1)
        ti = tempfile.NamedTemporaryFile(delete=False, suffix=".csv"); ti.close()
        th = tempfile.NamedTemporaryFile(delete=False, suffix=".csv"); th.close()
        inc.to_csv(ti.name, index=False)
        hai.to_csv(th.name, index=False)
        return ti.name, th.name

    # AI data validation
    def validate_ai_data(df_ai: pd.DataFrame) -> dict:
        issues = []
        n = len(df_ai)
        if "loss_usd" in df_ai.columns:
            has_loss = df_ai["loss_usd"].notna() & (df_ai["loss_usd"] > 0)
            n_losses = int(has_loss.sum())
            if n_losses < 30:
                issues.append(f"Only {n_losses} loss observations - results unreliable")
        if "loss_confidence" in df_ai.columns:
            low_conf = (df_ai["loss_confidence"] < 0.5).sum()
            if n > 0 and (low_conf / n) > 0.7:
                issues.append("70%+ of losses have confidence < 0.5")
        if "year" in df_ai.columns and df_ai["year"].notna().any():
            years = df_ai["year"].dropna()
            if not years.empty:
                year_span = int(years.max() - years.min())
                if year_span < 2:
                    issues.append(f"Only {year_span} years of data - frequency unreliable")
        return {"n_records": n, "issues": issues, "quality_score": max(0, 100 - len(issues) * 20)}

    # PATH A: Uploads or repo defaults → real pipeline
    if source in ("uploads", "repo"):
        try:
            from ai_monetary import (
                load_ai_table, fit_severity, fit_frequency,
                scenario_vector, simulate_eal_var, lec_dataframe
            )
        except Exception:
            st.error("AI Incidents mode needs scikit-learn. Add 'scikit-learn' to requirements.txt and redeploy.")
            st.stop()

        if source == "uploads":
            enriched_src = enriched_up
            hai62_src    = hai62_up
            st.success("Using uploaded CSVs.")
        else:
            enriched_src = str(DEF_ENRICH)
            hai62_src    = str(DEF_HAI62)
            st.success("Loaded repo defaults: data/incidents.csv and data/joinpack_hai_6_2.csv")

        norm_inc, norm_hai = normalize_aiid_csvs(enriched_src, hai62_src)
        try:
            df_ai = load_ai_table(norm_inc, norm_hai)
        finally:
            for p in (norm_inc, norm_hai):
                try: os.unlink(p)
                except Exception: pass

        # Data quality warnings
        data_quality = validate_ai_data(df_ai)
        st.metric("Data Quality Score", f"{data_quality['quality_score']}/100")
        for issue in data_quality["issues"]:
            st.warning(f"⚠️ {issue}")

        countries = ["(all)"] + (sorted(df_ai["country_group"].dropna().unique().tolist())
                                 if "country_group" in df_ai else [])
        country   = st.selectbox("Country", countries or ["(all)"])
        domains   = st.multiselect(
            "Domains",
            ["finance","healthcare","transport","social_media","hiring_hr","law_enforcement","education"],
            default=["finance"]
        )
        mods      = st.multiselect("Modalities", ["vision","nlp","recommender","generative","autonomous"], default=[])

        sev_model, sigma = fit_severity(df_ai, min_conf=min_conf)
        freq_model       = fit_frequency(df_ai)
        x_row            = scenario_vector(df_ai, None if country=="(all)" else country, domains, mods)

        eal, var95, var99, losses = simulate_eal_var(freq_model, sev_model, sigma, x_row,
                                                     trials=trials, seed=seed)

        k1, k2, k3 = st.columns(3)
        k1.metric("EAL",    f"${eal:,.0f}")
        k2.metric("VaR 95", f"${var95:,.0f}")
        k3.metric("VaR 99", f"${var99:,.0f}")

        lec_ai = lec_dataframe(losses)
        fig = px.line(lec_ai, x="loss", y="prob_exceed",
                      title="AI Incidents — Loss Exceedance Curve",
                      labels={"loss": "Loss ($)", "prob_exceed": "P(Loss ≥ x)"})
        fig.update_xaxes(type="log"); fig.update_yaxes(type="log", range=[-2.5, 0])
        st.plotly_chart(fig, use_container_width=True)

    # PATH B: Demo
    else:
        st.info("No CSVs found — running synthetic **demo dataset** (Poisson frequency + LogNormal severity).")
        losses = _simulate_ai_demo(trials=trials, seed=seed, lam=0.45, sev_mu=11.5, sev_sigma=1.0)
        eal, var95, var99 = _metrics_from_losses(losses)

        k1, k2, k3 = st.columns(3)
        k1.metric("EAL",    f"${eal:,.0f}")
        k2.metric("VaR 95", f"${var95:,.0f}")
        k3.metric("VaR 99", f"${var99:,.0f}")

        lec_ai = _lec_dataframe(losses)
        fig = px.line(lec_ai, x="loss", y="prob_exceed", title="AI Incidents — Loss Exceedance Curve (DEMO)",
                      labels={"loss": "Loss ($)", "prob_exceed": "P(Loss ≥ x)"})
        fig.update_xaxes(type="log"); fig.update_yaxes(type="log", range=[-2.5, 0])
        st.plotly_chart(fig, use_container_width=True)

        buf = io.StringIO()
        pd.DataFrame({"annual_loss_demo": losses}).to_csv(buf, index=False)
        st.download_button("Download demo losses (CSV)", buf.getvalue(), "ai_demo_annual_losses.csv", "text/csv")


