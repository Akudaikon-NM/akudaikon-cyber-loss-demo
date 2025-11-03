# --- Imports from the model/engine modules ---
from engine import (
    ModelConfig, FreqParams, SplicedParams,
    build_spliced_from_priors, simulate_annual_losses,
    compute_metrics, lec, lec_bands, posterior_lambda
)
from controls import ControlSet, ControlCosts, control_effects, total_cost

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from ai_monetary import (
    load_ai_table, fit_severity, fit_frequency,
    scenario_vector, simulate_eal_var, lec_dataframe
)
import plotly.express as px
from typing import Mapping, Optional

st.set_page_config(page_title="Akudaikon | Cyber-Loss Demo", layout="wide")
st.title("Akudaikon | Cyber-Loss Demo")
st.caption("Monte Carlo loss model with control ROI and optional Bayesian frequency.")

# --- choose which risk layer to run ---
mode = st.sidebar.radio(
    "Risk mode",
    ("Cyber Breach (records-based)", "AI Incidents (monetary)"),
    index=0
)

# ---------- Data-driven control effects (actions & patterns) ----------
def _normalize_shares(raw: Mapping[str, float]) -> dict:
    pairs = [(k.strip(), float(v)) for k, v in (raw or {}).items() if k and v is not None]
    total = sum(max(0.0, v) for _, v in pairs)
    if total <= 0:
        return {}
    return {k: (max(0.0, v) / total) for k, v in pairs if v > 0}

# Built-in NAICS 52 (Finance/Insurance) demo shares (replace with your VCDB slice)
DEFAULT_ACTION_SHARES = _normalize_shares({
    "Error": 0.35, "Hacking": 0.25, "Misuse": 0.25, "Social": 0.10, "Physical": 0.05
})
DEFAULT_PATTERN_SHARES = _normalize_shares({
    "Privilege Misuse": 0.40, "Basic Web App Attacks": 0.30, "Misc Errors": 0.30
})

# How each control influences frequency (λ, p_any) per ACTION share
ACTION_IMPACT_BY_CONTROL = {
    "external": {"Hacking": 0.30, "Social": 0.25},
    "error": {"Error": 0.25},
    "server": {"Hacking": 0.15, "Misuse": 0.05},
    "media": {},
}

# How each control influences TAIL severity scale per PATTERN share (GPD scale)
PATTERN_TAIL_IMPACT_BY_CONTROL = {
    "media": {"Misc Errors": 0.35, "Privilege Misuse": 0.15},
    "server": {"Basic Web App Attacks": 0.10},
    "external": {},
    "error": {},
}

def effects_from_shares(
    ctrl: "ControlSet",
    action_shares: Optional[Mapping[str, float]] = None,
    pattern_shares: Optional[Mapping[str, float]] = None,
    min_lam_mult: float = 0.50,
    min_pany_mult: float = 0.50,
    min_gpd_scale_mult: float = 0.50,
) -> "ControlEffects":
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

    lam_mult = max(min_lam_mult, lam_mult)
    p_any_mult = max(min_pany_mult, p_any_mult)
    gpd_scale_mult = max(min_gpd_scale_mult, gpd_scale_mult)

    return ControlEffects(lam_mult=lam_mult, p_any_mult=p_any_mult, gpd_scale_mult=gpd_scale_mult)

# ---------------------------------------------------------------------
# Advanced frequency (outside the form so it doesn't reset on submit)
# ---------------------------------------------------------------------
with st.sidebar.expander("Advanced frequency", expanded=False):
    use_bayes   = st.checkbox("Bayesian lambda (Gamma prior + your data)", value=False, key="adv_use_bayes")
    alpha0      = st.number_input("lambda prior alpha", min_value=0.01, max_value=50.0, value=2.0, step=0.1, key="adv_alpha0")
    beta0       = st.number_input("lambda prior beta",  min_value=0.01, max_value=50.0, value=8.0, step=0.1, key="adv_beta0")
    k_obs       = st.number_input("Incidents observed (k)", min_value=0, max_value=100000, value=0, step=1, key="adv_k_obs")
    T_obs       = st.number_input("Observation years (T)",  min_value=0.0, max_value=200.0, value=0.0, step=0.5, key="adv_T_obs")
    use_negbin  = st.checkbox("Use Negative Binomial (overdispersion)", value=False, key="adv_use_negbin")
    disp_r      = st.number_input("NegBin dispersion r", min_value=0.5, max_value=10.0, value=1.5, step=0.1, key="adv_disp_r")

# --- Calibration helper (k,T → λ̂ ; optional prior seeding) ---
st.markdown("**Calibration (from dataset slice)**")
ALPHA_MIN = 0.01
BETA_MIN  = 0.01
STEP      = 0.1
PREC      = 4

_defaults = {
    "adv_use_bayes": False,
    "adv_alpha0": 2.0,
    "adv_beta0": 8.0,
    "adv_pseudo_w": 2.0,
}
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

# -----------------------------------------------
# NAICS 52 (Finance & Insurance) presets
# -----------------------------------------------
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
    "524114 — Direct Health & Medical Insurance Carriers": {"lambda": 0.55, "records_cap": 4_000_000, "cost_per_record": 250.0, "net_worth": 1_800_000_000.

