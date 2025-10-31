# inside your app.py, keep imports but add:
from engine import (
    ModelConfig, FreqParams, SplicedParams,
    build_spliced_from_priors, simulate_annual_losses,
    compute_metrics, lec, lec_bands, posterior_lambda
)
from controls import ControlSet, ControlCosts, control_effects, total_cost

# (Optionally add an Advanced expander for Bayesian λ and NegBin)
with st.sidebar.expander("Advanced frequency", expanded=False):
    use_bayes = st.checkbox("Bayesian λ (Gamma prior + your data)", value=False)
    alpha0 = st.number_input("λ prior α", 0.01, 50.0, 2.0, step=0.1)
    beta0  = st.number_input("λ prior β", 0.01, 50.0, 8.0, step=0.1)
    k_obs  = st.number_input("Incidents observed", 0, 100, 0, step=1)
    T_obs  = st.number_input("Observation years", 0.0, 50.0, 0.0, step=0.5)
    use_negbin = st.checkbox("Use Negative Binomial (overdispersion)", value=False)
    disp_r = st.number_input("NegBin dispersion r", 0.5, 10.0, 1.5, step=0.1)

if run:
    with st.spinner("Simulating..."):
        # Config
        cfg = ModelConfig(
            trials=int(trials),
            net_worth=float(net_worth),
            seed=int(seed),
            record_cap=int(num_customers),
            cost_per_record=float(cost_per_customer)
        )

        # Frequency params
        lam_base = float(lam)
        if use_bayes and T_obs > 0:
            lam_draws = posterior_lambda(alpha0, beta0, int(k_obs), float(T_obs), draws=200, seed=seed+100)
            lam_base = float(np.median(lam_draws))  # use median λ for point run; also keep lam_draws for bands
        fp = FreqParams(lam=lam_base, p_any=0.85, negbin=bool(use_negbin), r=float(disp_r))

        # Severity prior (spliced); replace with real fit when wired
        sp = build_spliced_from_priors(cfg)

        # ----- Baseline -----
        base_losses = simulate_annual_losses(cfg, fp, sp)
        base_m = compute_metrics(base_losses, cfg.net_worth)

        # ----- Controlled (causal effects) -----
        ce = control_effects(ctrl)
        ctrl_losses = simulate_annual_losses(cfg, fp, sp, ce)
        ctrl_m = compute_metrics(ctrl_losses, cfg.net_worth)

        # ----- ROI -----
        ctrl_cost = total_cost(ctrl, costs)
        delta_eal = base_m["EAL"] - ctrl_m["EAL"]
        rosi = ((delta_eal - ctrl_cost)/ctrl_cost*100.0) if ctrl_cost>0 else np.nan

        # ----- Metrics UI (unchanged) -----
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("EAL (Baseline)", f"${base_m['EAL']:,.0f}")
        c2.metric("EAL (Controlled)", f"${ctrl_m['EAL']:,.0f}", delta=f"-${delta_eal:,.0f}")
        c3.metric("VaR95 (Base→Ctrl)", f"${base_m['VaR95']:,.0f}", delta=f"-${(base_m['VaR95']-ctrl_m['VaR95']):,.0f}")
        c4.metric("VaR99 (Base→Ctrl)", f"${base_m['VaR99']:,.0f}", delta=f"-${(base_m['VaR99']-ctrl_m['VaR99']):,.0f}")

        d1,d2,d3 = st.columns(3)
        d1.metric("VaR95 / Net Worth (Base)", f"{base_m['VaR95_to_NetWorth']*100:,.2f}%")
        d2.metric("VaR95 / Net Worth (Ctrl)", f"{ctrl_m['VaR95_to_NetWorth']*100:,.2f}%")
        d3.metric("ROSI (annualized)", "—" if np.isnan(rosi) else f"{rosi:,.1f}%")

        st.markdown("---")

        # ----- LEC curves + (optional) credible bands from λ posterior -----
        import plotly.graph_objects as go
        lec_b = lec(base_losses, 200).assign(scenario="Baseline")
        lec_c = lec(ctrl_losses, 200).assign(scenario="Controlled")

        fig = go.Figure()
        fig.add_scatter(x=lec_b["loss"], y=lec_b["exceed_prob"], mode="lines", name="Baseline")
        fig.add_scatter(x=lec_c["loss"], y=lec_c["exceed_prob"], mode="lines", name="Controlled")

        # credible bands if Bayesian used
        if use_bayes and T_obs > 0:
            S = min(80, len(lam_draws))
            # baseline bands
            samples = []
            for i in range(S):
                fp_i = FreqParams(lam=float(lam_draws[i]), p_any=fp.p_any, negbin=fp.negbin, r=fp.r)
                samples.append(simulate_annual_losses(cfg, fp_i, sp))
            samples = np.stack(samples, axis=0)
            band_b = lec_bands(samples, n=200, level=0.90)
            fig.add_scatter(x=band_b["loss"], y=band_b["hi"], mode="lines", name="Baseline 90% hi", line=dict(width=0.5), showlegend=False)
            fig.add_scatter(x=band_b["loss"], y=band_b["lo"], mode="lines", name="Baseline 90% lo", line=dict(width=0.5), fill="tonexty", fillcolor="rgba(0,0,0,0.08)", showlegend=False)

            # controlled bands (apply control effects)
            samples_c = []
            for i in range(S):
                fp_i = FreqParams(lam=float(lam_draws[i]), p_any=fp.p_any, negbin=fp.negbin, r=fp.r)
                samples_c.append(simulate_annual_losses(cfg, fp_i, sp, ce))
            samples_c = np.stack(samples_c, axis=0)
            band_c = lec_bands(samples_c, n=200, level=0.90)
            fig.add_scatter(x=band_c["loss"], y=band_c["hi"], mode="lines", name="Controlled 90% hi", line=dict(width=0.5), showlegend=False)
            fig.add_scatter(x=band_c["loss"], y=band_c["lo"], mode="lines", name="Controlled 90% lo", line=dict(width=0.5), fill="tonexty", fillcolor="rgba(0,0,0,0.08)", showlegend=False)

        fig.update_layout(title="Loss Exceedance Curve (LEC) with Optional Credible Bands",
                          xaxis_title="Annual Loss ($)", yaxis_title="P(Loss ≥ x)")
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log", range=[-2.5,0])
        st.plotly_chart(fig, use_container_width=True)

        # ----- Summary table (unchanged) -----
        st.subheader("Summary")
        tbl = pd.DataFrame({
            "Metric": ["EAL","VaR95","VaR99","VaR95/NetWorth","VaR99/NetWorth","Control Cost","ΔEAL","ROSI %"],
            "Baseline":[base_m["EAL"],base_m["VaR95"],base_m["VaR99"],base_m["VaR95_to_NetWorth"],base_m["VaR99_to_NetWorth"],np.nan,np.nan,np.nan],
            "Controlled":[ctrl_m["EAL"],ctrl_m["VaR95"],ctrl_m["VaR99"],ctrl_m["VaR95_to_NetWorth"],ctrl_m["VaR99_to_NetWorth"],ctrl_cost,delta_eal,rosi],
        })
        st.dataframe(tbl.style.format({"Baseline":"{:,.2f}","Controlled":"{:,.2f}"}), use_container_width=True)

        # ----- Download -----
        buf = io.StringIO()
        pd.DataFrame({"annual_loss_baseline":base_losses,"annual_loss_controlled":ctrl_losses}).to_csv(buf,index=False)
        st.download_button("Download annual losses (CSV)", buf.getvalue(), "cyber_annual_losses.csv", "text/csv")
