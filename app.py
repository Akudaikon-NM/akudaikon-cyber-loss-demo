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

# ---------------------------------------------------------------------
# Advanced frequency options (Bayesian lambda and Negative Binomial)
# ---------------------------------------------------------------------
with st.sidebar.expander("Advanced frequency", expanded=False):
    use_bayes = st.checkbox("Bayesian lambda (Gamma prior + your data)", value=False)
    alpha0 = st.number_input("lambda prior alpha", min_value=0.01, max_value=50.0, value=2.0, step=0.1)
    beta0  = st.number_input("lambda prior beta",  min_value=0.01, max_value=50.0, value=8.0, step=0.1)
    k_obs  = st.number_input("Incidents observed", min_value=0, max_value=100, value=0, step=1)
    T_obs  = st.number_input("Observation years",  min_value=0.0, max_value=50.0, value=0.0, step=0.5)

    use_negbin = st.checkbox("Use Negative Binomial (overdispersion)", value=False)
    disp_r = st.number_input("NegBin dispersion r", min_value=0.5, max_value=10.0, value=1.5, step=0.1)

# Provide a clear run trigger if you do not already have one upstream.
run = st.sidebar.button("Run simulation", type="primary")

if run:
    with st.spinner("Simulating..."):
        # -----------------------------------------------------------------
        # Config (assumes these sidebar inputs already exist upstream)
        # trials, net_worth, seed, num_customers, cost_per_customer, lam
        # ctrl (ControlSet or list) and costs (ControlCosts) are also assumed.
        # -----------------------------------------------------------------
        cfg = ModelConfig(
            trials=int(trials),
            net_worth=float(net_worth),
            seed=int(seed),
            record_cap=int(num_customers),
            cost_per_record=float(cost_per_customer)
        )

        # ------------------ Frequency parameters -------------------------
        lam_base = float(lam)

        lam_draws = None
        if use_bayes and T_obs > 0:
            lam_draws = posterior_lambda(
                float(alpha0), float(beta0), int(k_obs), float(T_obs),
                draws=200, seed=int(seed) + 100
            )
            # Use the posterior median as the point estimate for the main run
            lam_base = float(np.median(lam_draws))

        fp = FreqParams(
            lam=lam_base,
            p_any=0.85,                 # keep or wire to your UI
            negbin=bool(use_negbin),
            r=float(disp_r)
        )

        # ------------------ Severity prior (spliced) ---------------------
        # build_spliced_from_priors may consult cfg for exposure and $/record.
        sp: SplicedParams = build_spliced_from_priors(cfg)

        # ------------------ Baseline simulation --------------------------
        base_losses = simulate_annual_losses(cfg, fp, sp)
        base_m = compute_metrics(base_losses, cfg.net_worth)

        # ------------------ Controlled simulation ------------------------
        ce = control_effects(ctrl)  # ctrl: your chosen control set from UI
        ctrl_losses = simulate_annual_losses(cfg, fp, sp, ce)
        ctrl_m = compute_metrics(ctrl_losses, cfg.net_worth)

        # ------------------ ROI ------------------------------------------
        ctrl_cost = total_cost(ctrl, costs)  # costs: ControlCosts mapping
        delta_eal = base_m["EAL"] - ctrl_m["EAL"]
        rosi = ((delta_eal - ctrl_cost) / ctrl_cost * 100.0) if ctrl_cost > 0 else np.nan

        # ------------------ KPI tiles ------------------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("EAL (Baseline)",   f"${base_m['EAL']:,.0f}")
        c2.metric("EAL (Controlled)", f"${ctrl_m['EAL']:,.0f}", delta=f"-${delta_eal:,.0f}")
        c3.metric("VaR95 (Base->Ctrl)", f"${base_m['VaR95']:,.0f}",
                  delta=f"-${(base_m['VaR95'] - ctrl_m['VaR95']):,.0f}")
        c4.metric("VaR99 (Base->Ctrl)", f"${base_m['VaR99']:,.0f}",
                  delta=f"-${(base_m['VaR99'] - ctrl_m['VaR99']):,.0f}")

        d1, d2, d3 = st.columns(3)
        d1.metric("VaR95 / Net Worth (Base)", f"{base_m['VaR95_to_NetWorth']*100:,.2f}%")
        d2.metric("VaR95 / Net Worth (Ctrl)", f"{ctrl_m['VaR95_to_NetWorth']*100:,.2f}%")
        d3.metric("ROSI (annualized)", "—" if np.isnan(rosi) else f"{rosi:,.1f}%")

        st.markdown("---")

        # ------------------ LEC curves (with optional credible bands) ----
        lec_b = lec(base_losses, n=200).assign(scenario="Baseline")
        lec_c = lec(ctrl_losses, n=200).assign(scenario="Controlled")

        fig = go.Figure()
        fig.add_scatter(x=lec_b["loss"], y=lec_b["exceed_prob"],
                        mode="lines", name="Baseline")
        fig.add_scatter(x=lec_c["loss"], y=lec_c["exceed_prob"],
                        mode="lines", name="Controlled")

        if use_bayes and T_obs > 0 and lam_draws is not None:
            S = min(80, len(lam_draws))

            # Baseline bands
            samples = []
            for i in range(S):
                fp_i = FreqParams(lam=float(lam_draws[i]),
                                  p_any=fp.p_any, negbin=fp.negbin, r=fp.r)
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
                fp_i = FreqParams(lam=float(lam_draws[i]),
                                  p_any=fp.p_any, negbin=fp.negbin, r=fp.r)
                samples_c.append(simulate_annual_losses(cfg, fp_i, sp, ce))
            samples_c = np.stack(samples_c, axis=0)
            band_c = lec_bands(samples_c, n=200, level=0.90)
            fig.add_scatter(x=band_c["loss"], y=band_c["hi"], mode="lines",
                            name="Controlled 90% hi", line=dict(width=0.5), showlegend=False)
            fig.add_scatter(x=band_c["loss"], y=band_c["lo"], mode="lines",
                            name="Controlled 90% lo", line=dict(width=0.5),
                            fill="tonexty", fillcolor="rgba(0,0,0,0.08)", showlegend=False)

        fig.update_layout(
            title="Loss Exceedance Curve (LEC) with Optional Credible Bands",
            xaxis_title="Annual Loss (USD)",
            yaxis_title="P(Loss >= x)"
        )
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log", range=[-2.5, 0])
        st.plotly_chart(fig, use_container_width=True)

        # ------------------ Summary table --------------------------------
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

        # ------------------ Download (CSV) -------------------------------
        buf = io.StringIO()
        pd.DataFrame({
            "annual_loss_baseline": base_losses,
            "annual_loss_controlled": ctrl_losses
        }).to_csv(buf, index=False)
        st.download_button(
            label="Download annual losses (CSV)",
            data=buf.getvalue(),
            file_name="cyber_annual_losses.csv",
            mime="text/csv"
        )
