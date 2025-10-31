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
    if T_obs and T_obs > 0:
        lam_hat = float(k_obs) / float(T_obs)
        st.caption(f"λ̂ (k/T) = {lam_hat:.4f} incidents/year")
        with st.popover("Seed prior from λ̂"):
            w = st.number_input("Pseudo-years (weight for prior)", min_value=0.1, max_value=50.0, value=2.0, step=0.1, key="adv_pseudo_w")
            if st.button("Apply prior α₀=λ̂·w, β₀=w", key="btn_seed_prior"):
                st.session_state["adv_alpha0"] = lam_hat * w
                st.session_state["adv_beta0"]  = w
                st.session_state["adv_use_bayes"] = True
                st.success("Prior seeded from λ̂.")
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
        cfg = ModelConfig(
            trials=int(trials),
            net_worth=float(net_worth),
            seed=int(seed),
            record_cap=int(num_customers),
            cost_per_record=float(cost_per_customer),
        )

        # Frequency (with optional Bayesian update)
        lam_base = float(lam)
        lam_draws = None
        if use_bayes and T_obs > 0:
            lam_draws = posterior_lambda(
                float(alpha0), float(beta0),
                int(k_obs), float(T_obs),
                draws=200, seed=int(seed)+100
            )
            lam_base = float(np.median(lam_draws))

        fp = FreqParams(lam=lam_base, p_any=0.85, negbin=bool(use_negbin), r=float(disp_r))

        # Severity prior (spliced)
        sp: SplicedParams = build_spliced_from_priors(cfg)

        # Baseline
        base_losses = simulate_annual_losses(cfg, fp, sp)
        base_m = compute_metrics(base_losses, cfg.net_worth)

        # Controlled
        ce = control_effects(ctrl)
        ctrl_losses = simulate_annual_losses(cfg, fp, sp, ce)
        ctrl_m = compute_metrics(ctrl_losses, cfg.net_worth)

        # ROI
        ctrl_cost = total_cost(ctrl, costs)
        delta_eal = base_m["EAL"] - ctrl_m["EAL"]
        rosi = ((delta_eal - ctrl_cost) / ctrl_cost * 100.0) if ctrl_cost > 0 else np.nan

        # KPI tiles
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EAL (Baseline)",   f"${base_m['EAL']:,.0f}")
        c2.metric("EAL (Controlled)", f"${ctrl_m['EAL']:,.0f}", delta=f"-${delta_eal:,.0f}")
        c3.metric("VaR95 (Base→Ctrl)", f"${base_m['VaR95']:,.0f}", delta=f"-${(base_m['VaR95']-ctrl_m['VaR95']):,.0f}")
        c4.metric("VaR99 (Base→Ctrl)", f"${base_m['VaR99']:,.0f}", delta=f"-${(base_m['VaR99']-ctrl_m['VaR99']):,.0f}")

        d1, d2, d3 = st.columns(3)
        d1.metric("VaR95 / Net Worth (Base)", f"{base_m['VaR95_to_NetWorth']*100:,.2f}%")
        d2.metric("VaR95 / Net Worth (Ctrl)", f"{ctrl_m['VaR95_to_NetWorth']*100:,.2f}%")
        d3.metric("ROSI (annualized)", "—" if np.isnan(rosi) else f"{rosi:,.1f}%")

        st.markdown("---")

        # LEC (with optional credible bands)
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
            band_c = lec_bands(samples_c, n=200, level=0.90)
            fig.add_scatter(x=band_c["loss"], y=band_c["hi"], mode="lines", name="Controlled 90% hi",
                            line=dict(width=0.5), showlegend=False)
            fig.add_scatter(x=band_c["loss"], y=band_c["lo"], mode="lines", name="Controlled 90% lo",
                            line=dict(width=0.5), fill="tonexty", fillcolor="rgba(0,0,0,0.08)", showlegend=False)

        fig.update_layout(title="Loss Exceedance Curve (LEC) with Optional Credible Bands",
                          xaxis_title="Annual Loss (USD)", yaxis_title="P(Loss >= x)")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log", range=[-2.5, 0])
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
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
        st.dataframe(summary_df.style.format({"Baseline": "{:,.2f}", "Controlled": "{:,.2f}"}), use_container_width=True)

        # Download CSV of annual losses
        buf = io.StringIO()
        pd.DataFrame({"annual_loss_baseline": base_losses, "annual_loss_controlled": ctrl_losses}).to_csv(buf, index=False)
        st.download_button("Download annual losses (CSV)", buf.getvalue(), "cyber_annual_losses.csv", "text/csv")
