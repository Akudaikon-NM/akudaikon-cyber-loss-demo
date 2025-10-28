import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from engine import simulate_annual_losses, compute_metrics, lec
from controls import ControlSet, ControlCosts, prob_multiplier, severity_multiplier, total_cost

st.set_page_config(page_title="Akudaikon | Cyber Loss Demo", layout="wide")
st.title("Akudaikon | Cyber Loss Demo")
st.caption("Monte Carlo Simulation for Cyber Risk Analysis (Demo)")

with st.expander("How this demo works", expanded=True):
    st.write("""
Synthetic prototype: Frequency ~ Poisson(λ); severity is heavy-tailed (Beta fraction of customers × $/customer).
Controls apply multipliers to likelihood and/or severity. Metrics include EAL, VaR95/99, and LEC. Replace placeholders with your trained models when ready.
    """)

st.sidebar.header("Scenario Inputs")
sector = st.sidebar.selectbox(
    "NAICS Sector",
    options=[("522110 - Commercial Banking","bank"),("522130 - Credit Unions","cu"),("52 - Finance & Insurance","fin")],
    index=1
)
sector_defaults = {
    "bank": dict(lam=0.35, cost_per_cust=185.0),
    "cu":   dict(lam=0.28, cost_per_cust=169.0),
    "fin":  dict(lam=0.22, cost_per_cust=175.0),
}
lam_default = sector_defaults[sector[1]]["lam"]
cpc_default = sector_defaults[sector[1]]["cost_per_cust"]

trials = st.sidebar.number_input("Simulation trials", 5000, 200000, 10000, step=5000)
num_customers = st.sidebar.number_input("Customers / records (cap)", 1, 50_000_000, 250_000, step=10_000)
net_worth = st.sidebar.number_input("Net Worth ($)", 1, 10_000_000_000, 100_000_000, step=1_000_000)
lam = st.sidebar.number_input("Annual incident rate λ", 0.00, 5.00, float(lam_default), step=0.01, format="%.2f")
cost_per_customer = st.sidebar.number_input("Cost per customer ($)", 1.0, 5000.0, float(cpc_default), step=1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Controls")
ctrl = ControlSet(
    server   = st.sidebar.checkbox("Server Hardening / Patching", value=False),
    media    = st.sidebar.checkbox("Media Handling / DLP", value=False),
    error    = st.sidebar.checkbox("Error-Proofing / Config Mgmt", value=False),
    external = st.sidebar.checkbox("External Surface / MFA", value=False),
)
st.sidebar.markdown("**Control cost overrides (optional)**")
costs = ControlCosts(
    server   = st.sidebar.number_input("Server cost ($/yr)", 0.0, 5_000_000.0, 50_000.0, step=5_000.0),
    media    = st.sidebar.number_input("Media cost ($/yr)",  0.0, 5_000_000.0, 30_000.0, step=5_000.0),
    error    = st.sidebar.number_input("Error cost ($/yr)",  0.0, 5_000_000.0, 40_000.0, step=5_000.0),
    external = st.sidebar.number_input("External cost ($/yr)",0.0, 5_000_000.0,100_000.0, step=5_000.0),
)

seed = st.sidebar.number_input("Random seed", 0, 10_000_000, 42, step=1)
run = st.sidebar.button("Run Simulation", type="primary")

if run:
    with st.spinner("Simulating..."):
        base_losses = simulate_annual_losses(trials, lam, num_customers, cost_per_customer, 1.0, seed)
        base = compute_metrics(base_losses, net_worth)

        lam_mult = prob_multiplier(ctrl)
        sev_mult = severity_multiplier(ctrl)
        ctrl_losses = simulate_annual_losses(trials, lam*lam_mult, num_customers, cost_per_customer, sev_mult, seed+1)
        ctrl_m = compute_metrics(ctrl_losses, net_worth)

        ctrl_cost = total_cost(ctrl, costs)
        delta_eal = base["EAL"] - ctrl_m["EAL"]
        rosi = ((delta_eal - ctrl_cost)/ctrl_cost*100.0) if ctrl_cost>0 else np.nan

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("EAL (Baseline)", f"${base['EAL']:,.0f}")
        c2.metric("EAL (Controlled)", f"${ctrl_m['EAL']:,.0f}", delta=f"-${delta_eal:,.0f}")
        c3.metric("VaR95 (Base→Ctrl)", f"${base['VaR95']:,.0f}", delta=f"-${(base['VaR95']-ctrl_m['VaR95']):,.0f}")
        c4.metric("VaR99 (Base→Ctrl)", f"${base['VaR99']:,.0f}", delta=f"-${(base['VaR99']-ctrl_m['VaR99']):,.0f}")

        d1,d2,d3 = st.columns(3)
        d1.metric("VaR95 / Net Worth (Base)", f"{base['VaR95_to_NetWorth']*100:,.2f}%")
        d2.metric("VaR95 / Net Worth (Ctrl)", f"{ctrl_m['VaR95_to_NetWorth']*100:,.2f}%")
        d3.metric("ROSI (annualized)", "—" if np.isnan(rosi) else f"{rosi:,.1f}%")

        st.markdown("---")
        lec_b = lec(base_losses, 200).assign(scenario="Baseline")
        lec_c = lec(ctrl_losses, 200).assign(scenario="Controlled")
        lec_df = pd.concat([lec_b, lec_c], ignore_index=True)
        fig = px.line(lec_df, x="loss", y="exceed_prob", color="scenario", title="Loss Exceedance Curve (LEC)",
                      labels={"loss":"Annual Loss ($)", "exceed_prob":"P(Loss ≥ x)"})
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log", range=[-2.5,0])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary")
        tbl = pd.DataFrame({
            "Metric": ["EAL","VaR95","VaR99","VaR95/NetWorth","VaR99/NetWorth","Control Cost","ΔEAL","ROSI %"],
            "Baseline":[base["EAL"],base["VaR95"],base["VaR99"],base["VaR95_to_NetWorth"],base["VaR99_to_NetWorth"],np.nan,np.nan,np.nan],
            "Controlled":[ctrl_m["EAL"],ctrl_m["VaR95"],ctrl_m["VaR99"],ctrl_m["VaR95_to_NetWorth"],ctrl_m["VaR99_to_NetWorth"],ctrl_cost,delta_eal,rosi],
        })
        st.dataframe(tbl.style.format({"Baseline":"{:,.2f}","Controlled":"{:,.2f}"}), use_container_width=True)

        buf = io.StringIO()
        pd.DataFrame({"annual_loss_baseline":base_losses,"annual_loss_controlled":ctrl_losses}).to_csv(buf,index=False)
        st.download_button("Download annual losses (CSV)", buf.getvalue(), "cyber_annual_losses.csv", "text/csv")
else:
    st.info("Set inputs on the left and click **Run Simulation**.")
