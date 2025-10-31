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

st.set_page_config(page_title="Akudaikon | Cyber-Loss Demo", layout="wide")
st.title("Akudaikon | Cyber-Loss Demo")
st.caption("Monte Carlo loss model with control ROI and optional Bayesian frequency.")
# ---------- Data-driven control effects (actions & patterns) ----------
from typing import Mapping, Optional

def _normalize_shares(raw: Mapping[str, float]) -> dict:
    """Ensure shares sum to 1.0 and strip empties; robust to floats/strings."""
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
# Numbers are fractional reductions applied *proportionally to that action's share*
ACTION_IMPACT_BY_CONTROL = {
    "external": {  # MFA/perimeter: cut external vectors (hacking/social)
        "Hacking": 0.30, "Social": 0.25
    },
    "error": {     # Change control: cut error-caused incidents
        "Error": 0.25
    },
    "server": {    # Patching/hardening: cut infra-driven compromises
        "Hacking": 0.15, "Misuse": 0.05
    },
    "media": {     # Media protection has little effect on frequency (mostly tail)
        # keep empty or tiny effects if desired
    },
}

# How each control influences TAIL severity scale per PATTERN share (GPD scale)
PATTERN_TAIL_IMPACT_BY_CONTROL = {
    "media": {  # encryption/DLP, better handling => bend the tail
        "Misc Errors": 0.35, "Privilege Misuse": 0.15
    },
    "server": {  # safer servers slightly reduce extreme loss potential
        "Basic Web App Attacks": 0.10
    },
    "external": {},  # primarily frequency
    "error":   {},   # primarily frequency
}

def effects_from_shares(
    ctrl: "ControlSet",
    action_shares: Optional[Mapping[str, float]] = None,
    pattern_shares: Optional[Mapping[str, float]] = None,
    min_lam_mult: float = 0.50,
    min_pany_mult: float = 0.50,
    min_gpd_scale_mult: float = 0.50,
) -> "ControlEffects":
    """
    Blend toggled controls with action/pattern shares into multipliers:
      - lam_mult, p_any_mult: down-weighted by ACTION shares for active controls
      - gpd_scale_mult: tail scale down-weighted by PATTERN shares for active controls
    Clamped so multipliers never drop below the provided minima.
    """
    from engine import ControlEffects  # local import to avoid circulars in some setups

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
        # Each action reduces λ and p_any proportionally to its share and control strength
        for act, strength in impact.items():
            share = a_sh.get(act, 0.0)
            # Blend multiplicatively: (1 - share*strength)
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

    # Apply in any order; effects compound multiplicatively
    if getattr(ctrl, "external", False):
        apply_action("external"); apply_pattern("external")
    if getattr(ctrl, "error", False):
        apply_action("error");    apply_pattern("error")
    if getattr(ctrl, "server", False):
        apply_action("server");   apply_pattern("server")
    if getattr(ctrl, "media", False):
        apply_action("media");    apply_pattern("media")

    # Clamp so we never overshrink unrealistically
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

# Keep these consistent with your number_input minima/steps above
ALPHA_MIN = 0.01
BETA_MIN  = 0.01
STEP      = 0.1     # your number_inputs for alpha/beta use step=0.1
PREC      = 4       # rounding precision for safety vs STEP

# Ensure keys exist once (types matter for Streamlit widgets)
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
    # aligns value to the widget step to avoid precision conflicts
    return round(step * round(x / step), prec)

def seed_prior_cb():
    # read current k, T, w directly from session_state (always present)
    k = float(st.session_state.get("adv_k_obs", 0))
    T = float(st.session_state.get("adv_T_obs", 0.0))
    w = float(st.session_state.get("adv_pseudo_w", 2.0))

    lam_hat = (k / T) if T > 0 else 0.0

    alpha_suggest = max(ALPHA_MIN, lam_hat * w)
    beta_suggest  = max(BETA_MIN,  w)

    # align to widget steps to avoid Streamlit complaining
    alpha_suggest = _round_to_step(float(alpha_suggest))
    beta_suggest  = _round_to_step(float(beta_suggest))

    # final guard-rails (respect number_input bounds)
    alpha_suggest = max(ALPHA_MIN, alpha_suggest)
    beta_suggest  = max(BETA_MIN,  beta_suggest)

    st.session_state.update({
        "adv_alpha0": float(alpha_suggest),
        "adv_beta0":  float(beta_suggest),
        "adv_use_bayes": True,
    })
    st.rerun()

if T_obs and T_obs > 0:
    lam_hat = float(k_obs) / float(T_obs)
    st.caption(f"λ̂ (k/T) = {lam_hat:.4f} incidents/year")

    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.number_input(
            "Pseudo-years (weight for prior)",
            min_value=0.1, max_value=50.0,
            value=float(st.session_state["adv_pseudo_w"]),
            step=0.1, key="adv_pseudo_w"
        )

    # preview suggested values (clamped + rounded to step)
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
    "521110 — Monetary Authorities (Central Bank)": {
        "lambda": 0.35, "records_cap": 1_000_000, "cost_per_record": 185.0, "net_worth": 5_000_000_000.0,
    },
    "522110 — Commercial Banking": {
        "lambda": 0.60, "records_cap": 5_000_000, "cost_per_record": 185.0, "net_worth": 2_000_000_000.0,
    },
    "522120 — Savings Institutions": {
        "lambda": 0.45, "records_cap": 1_500_000, "cost_per_record": 185.0, "net_worth": 800_000_000.0,
    },
    "522130 — Credit Unions": {
        "lambda": 0.35, "records_cap": 250_000, "cost_per_record": 185.0, "net_worth": 100_000_000.0,
    },
    "522190 — Other Depository Credit Intermediation": {
        "lambda": 0.45, "records_cap": 1_000_000, "cost_per_record": 185.0, "net_worth": 500_000_000.0,
    },
    "522210 — Credit Card Issuing": {
        "lambda": 0.55, "records_cap": 3_000_000, "cost_per_record": 185.0, "net_worth": 1_000_000_000.0,
    },
    "522220 — Sales Financing": {
        "lambda": 0.40, "records_cap": 1_000_000, "cost_per_record": 175.0, "net_worth": 400_000_000.0,
    },
    "522291 — Consumer Lending": {
        "lambda": 0.45, "records_cap": 1_500_000, "cost_per_record": 185.0, "net_worth": 600_000_000.0,
    },
    "522292 — Real Estate Credit (incl. Mortgage Lending)": {
        "lambda": 0.40, "records_cap": 2_000_000, "cost_per_record": 185.0, "net_worth": 800_000_000.0,
    },
    "522293 — International Trade Financing": {
        "lambda": 0.35, "records_cap": 500_000, "cost_per_record": 175.0, "net_worth": 700_000_000.0,
    },
    "522294 — Secondary Market Financing": {
        "lambda": 0.35, "records_cap": 3_000_000, "cost_per_record": 175.0, "net_worth": 1_500_000_000.0,
    },
    "522298 — All Other Nondepository Credit Intermediation": {
        "lambda": 0.35, "records_cap": 800_000, "cost_per_record": 175.0, "net_worth": 300_000_000.0,
    },
    "522310 — Mortgage & Nonmortgage Loan Brokers": {
        "lambda": 0.30, "records_cap": 600_000, "cost_per_record": 175.0, "net_worth": 150_000_000.0,
    },
    "522320 — Financial Transactions Processing / Reserve / Clearinghouse": {
        "lambda": 0.65, "records_cap": 8_000_000, "cost_per_record": 200.0, "net_worth": 1_500_000_000.0,
    },
    "522390 — Other Activities Related to Credit Intermediation": {
        "lambda": 0.30, "records_cap": 500_000, "cost_per_record": 175.0, "net_worth": 200_000_000.0,
    },
    "523110 — Investment Banking & Securities Dealing": {
        "lambda": 0.45, "records_cap": 1_500_000, "cost_per_record": 185.0, "net_worth": 2_000_000_000.0,
    },
    "523120 — Securities Brokerage": {
        "lambda": 0.45, "records_cap": 2_500_000, "cost_per_record": 185.0, "net_worth": 1_200_000_000.0,
    },
    "523130 — Commodity Contracts Dealing": {
        "lambda": 0.35, "records_cap": 500_000, "cost_per_record": 175.0, "net_worth": 500_000_000.0,
    },
    "523140 — Commodity Contracts Brokerage": {
        "lambda": 0.35, "records_cap": 800_000, "cost_per_record": 175.0, "net_worth": 600_000_000.0,
    },
    "523210 — Securities & Commodity Exchanges": {
        "lambda": 0.40, "records_cap": 1_000_000, "cost_per_record": 185.0, "net_worth": 2_500_000_000.0,
    },
    "523910 — Miscellaneous Intermediation": {
        "lambda": 0.35, "records_cap": 600_000, "cost_per_record": 175.0, "net_worth": 250_000_000.0,
    },
    "523920 — Portfolio Management": {
        "lambda": 0.35, "records_cap": 1_200_000, "cost_per_record": 175.0, "net_worth": 900_000_000.0,
    },
    "523930 — Investment Advice": {
        "lambda": 0.30, "records_cap": 400_000, "cost_per_record": 175.0, "net_worth": 150_000_000.0,
    },
    "523991 — Trust, Fiduciary & Custody Activities": {
        "lambda": 0.35, "records_cap": 1_000_000, "cost_per_record": 185.0, "net_worth": 700_000_000.0,
    },
    "523999 — Miscellaneous Financial Investment Activities": {
        "lambda": 0.30, "records_cap": 500_000, "cost_per_record": 175.0, "net_worth": 200_000_000.0,
    },
    "524113 — Direct Life Insurance Carriers": {
        "lambda": 0.50, "records_cap": 3_000_000, "cost_per_record": 210.0, "net_worth": 1_500_000_000.0,
    },
    "524114 — Direct Health & Medical Insurance Carriers": {
        "lambda": 0.55, "records_cap": 4_000_000, "cost_per_record": 250.0, "net_worth": 1_800_000_000.0,
    },
    "524126 — Direct Property & Casualty Insurance Carriers": {
        "lambda": 0.45, "records_cap": 2_000_000, "cost_per_record": 200.0, "net_worth": 1_500_000_000.0,
    },
    "524127 — Direct Title Insurance Carriers": {
        "lambda": 0.35, "records_cap": 1_000_000, "cost_per_record": 185.0, "net_worth": 600_000_000.0,
    },
    "524128 — Other Direct Insurance Carriers": {
        "lambda": 0.40, "records_cap": 1_500_000, "cost_per_record": 200.0, "net_worth": 900_000_000.0,
    },
    "524210 — Insurance Agencies & Brokerages": {
        "lambda": 0.30, "records_cap": 600_000, "cost_per_record": 185.0, "net_worth": 150_000_000.0,
    },
    "524291 — Claims Adjusting": {
        "lambda": 0.30, "records_cap": 500_000, "cost_per_record": 185.0, "net_worth": 120_000_000.0,
    },
    "524292 — Third-Party Administration of Insurance & Pension Funds": {
        "lambda": 0.40, "records_cap": 1_500_000, "cost_per_record": 200.0, "net_worth": 400_000_000.0,
    },
    "524298 — All Other Insurance Related Activities": {
        "lambda": 0.30, "records_cap": 500_000, "cost_per_record": 185.0, "net_worth": 120_000_000.0,
    },
    "525110 — Pension Funds": {
        "lambda": 0.35, "records_cap": 2_000_000, "cost_per_record": 200.0, "net_worth": 2_000_000_000.0,
    },
    "525120 — Health & Welfare Funds": {
        "lambda": 0.40, "records_cap": 2_500_000, "cost_per_record": 230.0, "net_worth": 1_200_000_000.0,
    },
    "525190 — Other Insurance Funds": {
        "lambda": 0.35, "records_cap": 1_500_000, "cost_per_record": 210.0, "net_worth": 900_000_000.0,
    },
    "525910 — Open-End Investment Funds": {
        "lambda": 0.35, "records_cap": 1_500_000, "cost_per_record": 175.0, "net_worth": 1_500_000_000.0,
    },
    "525920 — Trusts, Estates & Agency Accounts": {
        "lambda": 0.30, "records_cap": 800_000, "cost_per_record": 185.0, "net_worth": 700_000_000.0,
    },
    "525990 — Other Financial Vehicles": {
        "lambda": 0.30, "records_cap": 1_000_000, "cost_per_record": 175.0, "net_worth": 1_000_000_000.0,
    },
}

with st.sidebar.expander("Finance NAICS presets", expanded=False):
    use_naics = st.checkbox("Use preset", value=False, key="naics_enable")

    # ensure Credit Unions is the actual default, regardless of dict order
    _keys = list(NAICS_FINANCE_PRESETS.keys())
    _default_label = "522130 — Credit Unions"
    _default_index = _keys.index(_default_label) if _default_label in _keys else 0

    choice = st.selectbox(
        "Select NAICS (Finance)",
        _keys,
        index=_default_index,
        disabled=not use_naics,
        key="naics_choice",
    )

    if use_naics:
        p = NAICS_FINANCE_PRESETS[choice]
        # Seed scenario inputs via session_state so the form reflects the preset
        st.session_state["in_lambda"]      = p["lambda"]
        st.session_state["in_records_cap"] = p["records_cap"]
        st.session_state["in_cpr"]         = p["cost_per_record"]
        st.session_state["in_networth"]    = p["net_worth"]
        st.caption(f"Preset applied: {choice}")
# -----------------------------------------------
# Data-driven control effects: shares source
# -----------------------------------------------
with st.sidebar.expander("Data-driven control effects (shares)", expanded=False):
    st.caption("Use NAICS-52 demo shares or upload CSVs to weight control effects by ACTION/PATTERN.")
    shares_mode = st.radio(
        "Shares source",
        ["Built-in NAICS-52 (demo)", "Upload CSVs"],
        index=0,
        key="shares_mode"
    )

    action_shares = DEFAULT_ACTION_SHARES
    pattern_shares = DEFAULT_PATTERN_SHARES

    if shares_mode == "Upload CSVs":
        up_actions = st.file_uploader("Upload action shares CSV (columns: category, share)", type=["csv"], key="up_actions")
        up_patterns = st.file_uploader("Upload pattern shares CSV (columns: category, share)", type=["csv"], key="up_patterns")

        def _read_shares(file) -> dict:
            try:
                df_u = pd.read_csv(file)
                # support common column names
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

    # Small preview
    if action_shares:
        st.write("**Action shares**")
        st.dataframe(pd.DataFrame({"action": list(action_shares.keys()), "share": list(action_shares.values())}))
    if pattern_shares:
        st.write("**Pattern shares**")
        st.dataframe(pd.DataFrame({"pattern": list(pattern_shares.keys()), "share": list(pattern_shares.values())}))

# store in session so the run block can see them
st.session_state["_action_shares"] = action_shares
st.session_state["_pattern_shares"] = pattern_shares

# ---------------------------------------------------------------------
# Scenario + Controls (grouped in ONE form)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Run the simulation once the form is submitted
# ---------------------------------------------------------------------
if submitted:
    with st.spinner("Simulating..."):
        # -------------------- Config --------------------
        cfg = ModelConfig(
            trials=int(trials),
            net_worth=float(net_worth),
            seed=int(seed),
            record_cap=int(num_customers),
            cost_per_record=float(cost_per_customer),
        )

        # ---------------- Frequency (Bayesian optional) ----------------
        lam_base = float(lam)
        lam_draws = None
        if use_bayes and T_obs > 0:
            lam_draws = posterior_lambda(
                float(alpha0), float(beta0),
                int(k_obs), float(T_obs),
                draws=200, seed=int(seed) + 100
            )
            lam_base = float(np.median(lam_draws))

        fp = FreqParams(
            lam=lam_base,
            p_any=0.85,
            negbin=bool(use_negbin),
            r=float(disp_r)
        )

        # ---------------- Severity prior (spliced) ----------------
        sp: SplicedParams = build_spliced_from_priors(cfg)

        # ---------------- Baseline ----------------
        base_losses = simulate_annual_losses(cfg, fp, sp)
        base_m = compute_metrics(base_losses, cfg.net_worth)

        # ---------------- Controlled (data-driven effects) ----------------
        # Pull the latest shares chosen/uploaded in the sidebar
        ash = st.session_state.get("_action_shares", DEFAULT_ACTION_SHARES)
        psh = st.session_state.get("_pattern_shares", DEFAULT_PATTERN_SHARES)

        # Map controls -> parameter multipliers using the shares
        try:
            ce = effects_from_shares(ctrl, ash, psh)
        except Exception:
            # Safe fallback to static control mapping
            ce = control_effects(ctrl)

        # Run the controlled simulation and metrics
        ctrl_losses = simulate_annual_losses(cfg, fp, sp, ce)
        ctrl_m = compute_metrics(ctrl_losses, cfg.net_worth)

        # ----------------------------- ROI --------------------------------
        ctrl_cost = total_cost(ctrl, costs)
        delta_eal = base_m["EAL"] - ctrl_m["EAL"]
        rosi = ((delta_eal - ctrl_cost) / ctrl_cost * 100.0) if ctrl_cost > 0 else np.nan

        # ----------------------------- KPI tiles --------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EAL (Baseline)",   f"${base_m['EAL']:,.0f}")
        c2.metric("EAL (Controlled)", f"${ctrl_m['EAL']:,.0f}", delta=f"-${delta_eal:,.0f}")
        c3.metric("VaR95 (Base→Ctrl)", f"${base_m['VaR95']:,.0f}",
                  delta=f"-${(base_m['VaR95'] - ctrl_m['VaR95']):,.0f}")
        c4.metric("VaR99 (Base→Ctrl)", f"${base_m['VaR99']:,.0f}",
                  delta=f"-${(base_m['VaR99'] - ctrl_m['VaR99']):,.0f}")

        d1, d2, d3 = st.columns(3)
        d1.metric("VaR95 / Net Worth (Base)", f"{base_m['VaR95_to_NetWorth']*100:,.2f}%")
        d2.metric("VaR95 / Net Worth (Ctrl)", f"{ctrl_m['VaR95_to_NetWorth']*100:,.2f}%")
        d3.metric("ROSI (annualized)", "—" if np.isnan(rosi) else f"{rosi:,.1f}%")

        st.markdown("---")

        # ---------------- LEC (with optional credible bands) --------------
        lec_b = lec(base_losses, n=200).assign(scenario="Baseline")
        lec_c = lec(ctrl_losses, n=200).assign(scenario="Controlled")

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
            band_b = lec_bands(samples, n=200, level=0.90)
            fig.add_scatter(x=band_b["loss"], y=band_b["hi"], mode="lines",
                            name="Baseline 90% hi", line=dict(width=0.5), showlegend=False)
            fig.add_scatter(x=band_b["loss"], y=band_b["lo"], mode="lines",
                            name="Baseline 90% lo", line=dict(width=0.5),
                            fill="tonexty", fillcolor="rgba(0,0,0,0.08)", showlegend=False)

            # Controlled bands
            samples_c = []
            for i in range(S):
                fp_i = FreqParams(lam=float(lam_draws[i]), p_any=fp.p_any, negbin=fp.negbin, r=fp.r)
                samples_c.append(simulate_annual_losses(cfg, fp_i, sp, ce))
            samples_c = np.stack(samples_c, axis=0)
            band_c = lec_bands(samples_c, n=200, level=0.90)
            fig.add_scatter(x=band_c["loss"], y=band_c["hi"], mode="lines",
                            name="Controlled 90% hi", line=dict(width=0.5), showlegend=False)
            fig.add_scatter(x=band_c["loss"], y=band_c["lo"], mode="lines",
                            name="Controlled 90% lo", line=dict(width=0.5),
                            fill="tonexty", fillcolor="rgba(0,0,0,0.08)", showlegend=False)

        fig.update_layout(title="Loss Exceedance Curve (LEC) with Optional Credible Bands",
                          xaxis_title="Annual Loss (USD)", yaxis_title="P(Loss >= x)")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log", range=[-2.5, 0])
        st.plotly_chart(fig, use_container_width=True)

        # ---------------- Summary table -----------------------------------
        st.subheader("Summary")
        summary_df = pd.DataFrame({
            "Metric": ["EAL", "VaR95", "VaR99", "VaR95/NetWorth", "VaR99/NetWorth",
                       "Control Cost", "Delta EAL", "ROSI %"],
            "Baseline":  [base_m["EAL"], base_m["VaR95"], base_m["VaR99"],
                          base_m["VaR95_to_NetWorth"], base_m["VaR99_to_NetWorth"],
                          np.nan, np.nan, np.nan],
            "Controlled":[ctrl_m["EAL"], ctrl_m["VaR95"], ctrl_m["VaR99"],
                          ctrl_m["VaR95_to_NetWorth"], ctrl_m["VaR99_to_NetWorth"],
                          ctrl_cost, delta_eal, rosi],
        })
        st.dataframe(
            summary_df.style.format({"Baseline": "{:,.2f}", "Controlled": "{:,.2f}"}),
            use_container_width=True
        )

        # ---------------- Download CSV ------------------------------------
        buf = io.StringIO()
        pd.DataFrame({
            "annual_loss_baseline": base_losses,
            "annual_loss_controlled": ctrl_losses
        }).to_csv(buf, index=False)
        st.download_button(
            "Download annual losses (CSV)",
            buf.getvalue(),
            "cyber_annual_losses.csv",
            "text/csv"
        )

