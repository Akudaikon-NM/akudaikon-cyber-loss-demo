import streamlit as st
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Optional
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import lognorm

# KEEP this one at the very top (already there)
st.set_page_config(page_title="Akudaikon | Cyber-Loss Demo", layout="wide")

# Title and caption (removed duplicate set_page_config)
st.title("Akudaikon | Cyber-Loss Demo")
st.caption("Monte Carlo loss model with control ROI, diagnostics, and optional Bayesian frequency.")

# ============================================================================
# SANITY CHECKS / EXPECTED BEHAVIORS
# ============================================================================
# What to look for in outputs:
# 
# 1. With all controls off, turning only 'external' on should drop EAL primarily 
#    through λ reduction, with a modest tail effect (gpd_mult *= 0.85).
#
# 2. Increasing σ from 1.5 → 2.5 should push VaR95/VaR99 up noticeably, even if 
#    EAL moves less—lognormal fattening is mostly a tail story.
#
# 3. For NegBin: decreasing r from 3 → 1 (holding λ) should lift VaR more than 
#    EAL (over-dispersion increases tail probability).
#
# 4. Control isolation analysis should show which controls provide best ROSI. 
#    Typically external monitoring and server hardening have strong λ effects.
#
# 5. GPD shape ξ ≥ 1.0 yields infinite mean tail—results will be unstable. 
#    Keep ξ < 0.5 for realistic cyber losses.
# ============================================================================

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ModelConfig:
    trials: int = 10000
    net_worth: float = 100e6
    seed: int = 42
    record_cap: int = 0
    cost_per_record: float = 0.0

@dataclass
class FreqParams:
    lam: float = 2.0
    p_any: float = 0.7
    negbin: bool = False
    r: float = 1.0

@dataclass
class SevParams:
    mu: float = 12.0
    sigma: float = 2.0
    gpd_thresh_q: float = 0.95
    gpd_scale: float = 1e6
    gpd_shape: float = 0.3

@dataclass
class ControlSet:
    server: bool = False
    media: bool = False
    error: bool = False
    external: bool = False

@dataclass
class ControlEffects:
    lam_mult: float = 1.0
    p_any_mult: float = 1.0
    gpd_scale_mult: float = 1.0

@dataclass
class ControlCosts:
    server: float = 0.0
    media: float = 0.0
    error: float = 0.0
    external: float = 0.0
    
    def total(self) -> float:
        return self.server + self.media + self.error + self.external

# ============================================================================
# DEFAULT PARAMETERS (from VCDB analysis)
# ============================================================================

DEFAULT_ACTION_SHARES = {
    "hacking": 0.35,
    "malware": 0.25,
    "social": 0.15,
    "misuse": 0.12,
    "physical": 0.08,
    "error": 0.05
}

DEFAULT_PATTERN_SHARES = {
    "Web Applications": 0.20,
    "Privilege Misuse": 0.18,
    "Lost and Stolen Assets": 0.15,
    "Crimeware": 0.15,
    "Payment Card Skimmers": 0.10,
    "Denial of Service": 0.08,
    "Cyber-Espionage": 0.07,
    "Point of Sale": 0.05,
    "Miscellaneous Errors": 0.02
}

ACTION_LOSS_PROPENSITY = {
    "hacking": 0.72,
    "malware": 0.68,
    "social": 0.65,
    "misuse": 0.58,
    "physical": 0.45,
    "error": 0.55
}

PATTERN_TAIL_MULTIPLIERS = {
    "Web Applications": 1.2,
    "Privilege Misuse": 1.5,
    "Crimeware": 1.3,
    "Payment Card Skimmers": 0.9
}

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def simulate_annual_losses(cfg: ModelConfig, fp: FreqParams, sp: SevParams, 
                          ce: Optional[ControlEffects] = None) -> np.ndarray:
    """Simulate annual cyber losses with optional controls."""
    np.random.seed(cfg.seed)
    
    # Apply control effects
    lam_eff = fp.lam * (ce.lam_mult if ce else 1.0)
    p_any_eff = fp.p_any * (ce.p_any_mult if ce else 1.0)
    gpd_scale_eff = sp.gpd_scale * (ce.gpd_scale_mult if ce else 1.0)
    
    # Precompute dollar threshold once per call (outside the loops)
    # Better (and simpler) using scipy.stats.lognorm
    body_thresh_val = float(lognorm(s=sp.sigma, scale=np.exp(sp.mu)).ppf(sp.gpd_thresh_q))
    
    annual_losses = np.zeros(cfg.trials)
    
    for i in range(cfg.trials):
        # Generate incident count
        if fp.negbin:
            # Gamma-Poisson mixture: K ~ Poisson(L), L ~ Gamma(shape=r, scale=lam_eff/r)
            L = np.random.gamma(shape=fp.r, scale=lam_eff / fp.r)
            n_incidents = np.random.poisson(L)
        else:
            n_incidents = np.random.poisson(lam_eff)
        
        if n_incidents == 0:
            continue
        
        # Generate losses for each incident
        for _ in range(n_incidents):
            # Determine if loss occurs
            if np.random.random() > p_any_eff:
                continue
            
            # Generate loss amount (lognormal body + GPD tail)
            u = np.random.random()
            if u < sp.gpd_thresh_q:
                # Body (lognormal)
                loss = np.exp(np.random.normal(sp.mu, sp.sigma))
            else:
                # Tail (GPD on excess over threshold)
                u_tail = np.random.random()  # fresh uniform for the tail
                xi = sp.gpd_shape
                beta = gpd_scale_eff
                if xi == 0.0:
                    # Exponential tail on excess
                    excess = np.random.exponential(beta)
                else:
                    # Inverse CDF of GPD
                    excess = beta * (u_tail**(-xi) - 1.0) / xi
                loss = body_thresh_val + max(0.0, excess)
            
            annual_losses[i] += loss
    
    return annual_losses

def compute_metrics(losses: np.ndarray, net_worth: float) -> dict:
    """Compute risk metrics from loss distribution."""
    return {
        "EAL": np.mean(losses),
        "VaR95": np.percentile(losses, 95),
        "VaR99": np.percentile(losses, 99),
        "CVaR95": np.mean(losses[losses >= np.percentile(losses, 95)]),
        "Max": np.max(losses),
        "P(Ruin)": np.mean(losses >= net_worth)
    }

def var_confidence_interval(losses: np.ndarray, alpha: float, n_boot: int = 1000) -> dict:
    """Bootstrap confidence interval for VaR."""
    var_point = np.percentile(losses, alpha * 100)
    boot_vars = []
    
    for _ in range(n_boot):
        sample = np.random.choice(losses, size=len(losses), replace=True)
        boot_vars.append(np.percentile(sample, alpha * 100))
    
    boot_vars = np.array(boot_vars)
    return {
        "point": var_point,
        "ci_lower": np.percentile(boot_vars, 2.5),
        "ci_upper": np.percentile(boot_vars, 97.5)
    }

@st.cache_data(show_spinner=False)
def cached_simulate(cfg_dict, fp_dict, sp_dict, ce_dict=None):
    """Cached simulation wrapper."""
    cfg = ModelConfig(**cfg_dict)
    fp = FreqParams(**fp_dict)
    sp = SevParams(**sp_dict)
    ce = ControlEffects(**ce_dict) if ce_dict else None
    return simulate_annual_losses(cfg, fp, sp, ce)

def lec(losses: np.ndarray, n: int = 200) -> pd.DataFrame:
    """Loss Exceedance Curve with log-spaced grid."""
    losses = np.asarray(losses, float)
    losses = losses[np.isfinite(losses)]
    
    # Handle all-zero losses
    if losses.size == 0 or np.all(losses == 0):
        return pd.DataFrame({"Loss": [1.0], "Exceedance_Prob": [0.0]})
    
    losses.sort()
    
    # Choose log-spaced grid from a small positive to max
    lo = max(1.0, np.percentile(losses[losses > 0], 1) if np.any(losses > 0) else 1.0)
    hi = float(losses[-1])
    if hi <= lo:
        return pd.DataFrame({"Loss": [1.0], "Exceedance_Prob": [0.0]})
    
    grid = np.logspace(np.log10(lo), np.log10(hi), n)
    
    # For each grid x, find first index >= x via binary search
    idx = np.searchsorted(losses, grid, side='left')
    ex_prob = 1.0 - idx / float(len(losses))
    
    return pd.DataFrame({"Loss": grid, "Exceedance_Prob": ex_prob})

@st.cache_data(show_spinner=False)
def cached_lec(losses, n):
    """Cached LEC calculation."""
    return lec(losses, n=n)

def effects_from_shares_improved(ctrl: ControlSet, action_shares: dict, pattern_shares: dict) -> ControlEffects:
    """Compute control effects from action/pattern shares."""
    # Simple mapping for demo - would be more sophisticated in production
    lam_mult = 1.0
    p_any_mult = 1.0
    gpd_mult = 1.0
    
    if ctrl.server:
        lam_mult *= 0.7
        p_any_mult *= 0.8
        gpd_mult *= 0.9
    if ctrl.media:
        lam_mult *= 0.8
        p_any_mult *= 0.85
    if ctrl.error:
        lam_mult *= 0.9
        p_any_mult *= 0.9
    if ctrl.external:
        lam_mult *= 0.85
        gpd_mult *= 0.85
    
    return ControlEffects(lam_mult=lam_mult, p_any_mult=p_any_mult, gpd_scale_mult=gpd_mult)

def log_hist_figure(losses, title):
    """Create histogram with log-scaled x-axis."""
    x = np.asarray(losses, float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="No positive losses", x=0.5, y=0.5, showarrow=False)
        return fig
    
    x_log = np.log10(x)
    fig = go.Figure([go.Histogram(x=x_log, nbinsx=60)])
    lo, hi = np.floor(x_log.min()), np.ceil(x_log.max())
    ticks = np.arange(lo, hi+1)
    fig.update_layout(title=title, xaxis_title="Annual Loss (log10 $)", yaxis_title="Frequency")
    fig.update_xaxes(tickvals=ticks, ticktext=[f"${10**int(t):,.0f}" for t in ticks])
    return fig

def _normalize_shares(shares: dict) -> dict:
    """Normalize shares to sum to 1."""
    total = sum(shares.values())
    return {k: v/total for k, v in shares.items()} if total > 0 else shares

# ============================================================================
# SIDEBAR - PARAMETERS
# ============================================================================

st.sidebar.header("⚙️ Model Parameters")

# Load parameters from JSON
with st.sidebar.expander("📁 Load parameter JSON", expanded=False):
    pj = st.file_uploader("Upload parameters.json", type=["json"])
    if pj:
        params = json.load(pj)
        # Wire in action/pattern shares & tail
        _a = params.get("DEFAULT_ACTION_SHARES") or params.get("defaults", {}).get("action_shares")
        _p = params.get("DEFAULT_PATTERN_SHARES") or params.get("defaults", {}).get("pattern_shares")
        if _a:
            action_shares = _normalize_shares(_a)
            st.session_state["_action_shares"] = action_shares
            st.success("✓ Action shares loaded")
        if _p:
            pattern_shares = _normalize_shares(_p)
            st.session_state["_pattern_shares"] = pattern_shares
            st.success("✓ Pattern shares loaded")

# Use loaded shares or defaults
action_shares = st.session_state.get("_action_shares", DEFAULT_ACTION_SHARES)
pattern_shares = st.session_state.get("_pattern_shares", DEFAULT_PATTERN_SHARES)

# Display current shares in sidebar
st.sidebar.caption("Action shares in use")
st.sidebar.dataframe(
    pd.DataFrame({
        "action": list(action_shares.keys()),
        "share": list(action_shares.values())
    }).style.format({"share": "{:.1%}"}),
    hide_index=True
)

st.sidebar.caption("Pattern shares in use")
st.sidebar.dataframe(
    pd.DataFrame({
        "pattern": list(pattern_shares.keys()),
        "share": list(pattern_shares.values())
    }).style.format({"share": "{:.1%}"}),
    hide_index=True
)

# Model configuration
with st.sidebar.expander("🎲 Simulation Config", expanded=True):
    trials = st.number_input("Monte Carlo Trials", 1000, 100000, 10000, 1000)
    net_worth = st.number_input("Net Worth ($M)", 1.0, 10000.0, 100.0, 10.0) * 1e6
    seed = st.number_input("Random Seed", 0, 9999, 42)

cfg = ModelConfig(trials=trials, net_worth=net_worth, seed=seed)

# Frequency parameters
with st.sidebar.expander("📊 Frequency Parameters", expanded=True):
    lam = st.number_input("λ (mean incidents/year)", 0.1, 20.0, 2.0, 0.1)
    p_any = st.slider("P(any loss | incident)", 0.1, 0.95, 0.7, 0.05)
    negbin = st.checkbox("Use Negative Binomial", value=False)
    r = st.number_input("NegBin dispersion (r)", 0.5, 10.0, 1.0, 0.5) if negbin else 1.0
    
    # Validation warnings
    if p_any < 0.1 or p_any > 0.95:
        st.warning(f"⚠️ p(any loss)={p_any:.2f} is extreme; results may be unstable.")

fp = FreqParams(lam=lam, p_any=p_any, negbin=negbin, r=r)

# Clamp frequency params
fp.p_any = float(np.clip(fp.p_any, 0.0, 1.0))
if fp.negbin:
    fp.r = float(max(1e-6, fp.r))

# Optional Bayesian update
with st.sidebar.expander("🔬 Bayesian Frequency Update", expanded=False):
    use_bayes = st.checkbox("Enable Bayesian update")
    if use_bayes:
        T_obs = st.number_input("Observation period (years)", 1, 20, 5)
        k_obs = st.number_input("Observed incidents", 0, 100, 10)
        
        if T_obs and k_obs:
            lam_hat = float(k_obs) / float(T_obs)
            if (lam_hat > 0) and (max(lam_hat, fp.lam) / max(1e-9, min(lam_hat, fp.lam)) >= 3):
                st.info(f"ℹ️ λ differs from k/T by ≥3× (λ={fp.lam:.3f}, k/T={lam_hat:.3f}). "
                       f"Consider Bayes mode or align values.")
            
            # Simple Bayesian update (Gamma prior)
            alpha_prior = 2.0
            beta_prior = alpha_prior / lam
            alpha_post = alpha_prior + k_obs
            beta_post = beta_prior + T_obs
            lam_post = alpha_post / beta_post
            
            st.metric("Updated λ (posterior mean)", f"{lam_post:.2f}")
            fp.lam = lam_post

# Severity parameters
with st.sidebar.expander("💰 Severity Parameters", expanded=True):
    mu = st.number_input("Lognormal μ", 8.0, 16.0, 12.0, 0.5)
    sigma = st.number_input("Lognormal σ", 0.5, 4.0, 2.0, 0.1)
    gpd_thresh_q = st.slider("GPD threshold quantile", 0.85, 0.99, 0.95, 0.01)
    gpd_scale = st.number_input("GPD scale ($K)", 100.0, 10000.0, 1000.0, 100.0) * 1000
    gpd_shape = st.number_input("GPD shape (ξ)", 0.0, 1.0, 0.3, 0.05)

sp = SevParams(mu=mu, sigma=sigma, gpd_thresh_q=gpd_thresh_q, 
               gpd_scale=gpd_scale, gpd_shape=gpd_shape)

# Clamp severity params
sp.gpd_scale = float(max(1.0, sp.gpd_scale))

# Severity parameter warnings
if sp.gpd_shape >= 1.0:
    st.sidebar.warning("⚠️ ξ (GPD shape) ≥ 1 yields infinite mean tail. Results may be unstable.")
if sp.sigma > 2.5:
    st.sidebar.warning("⚠️ Lognormal σ is quite high; body may dominate and inflate EAL/VaR.")

# Controls
with st.sidebar.expander("🛡️ Control Selection", expanded=True):
    server = st.checkbox("Server hardening", value=False)
    media = st.checkbox("Media encryption", value=False)
    error_ctrl = st.checkbox("Error reduction", value=False)
    external = st.checkbox("External monitoring", value=False)

ctrl = ControlSet(server=server, media=media, error=error_ctrl, external=external)

# Control costs
with st.sidebar.expander("💵 Control Costs", expanded=True):
    cost_server = st.number_input("Server hardening ($K)", 0.0, 1000.0, 50.0, 10.0) * 1000
    cost_media = st.number_input("Media encryption ($K)", 0.0, 1000.0, 30.0, 10.0) * 1000
    cost_error = st.number_input("Error reduction ($K)", 0.0, 1000.0, 20.0, 10.0) * 1000
    cost_external = st.number_input("External monitoring ($K)", 0.0, 1000.0, 40.0, 10.0) * 1000

costs = ControlCosts(server=cost_server, media=cost_media, 
                     error=cost_error, external=cost_external)

# Validate costs
for k in ["server", "media", "error", "external"]:
    v = getattr(costs, k)
    if v < 0:
        setattr(costs, k, 0.0)
        st.sidebar.warning(f"⚠️ {k.title()} cost < 0 corrected to 0.")

# ============================================================================
# MAIN SIMULATION
# ============================================================================

st.header("🎯 Simulation Results")

# Assumption Summary Box
with st.expander("📋 Assumption Summary", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Frequency Parameters**")
        st.markdown(f"- λ (mean incidents/year): `{fp.lam:.3f}`")
        st.markdown(f"- P(any loss | incident): `{fp.p_any:.3f}`")
        st.markdown(f"- Distribution: `{'Negative Binomial' if fp.negbin else 'Poisson'}`")
        if fp.negbin:
            st.markdown(f"- NegBin dispersion (r): `{fp.r:.3f}`")
        
        st.markdown("**Control Effects**")
        st.markdown(f"- λ multiplier: `{ce.lam_mult:.3f}`")
        st.markdown(f"- P(any) multiplier: `{ce.p_any_mult:.3f}`")
        st.markdown(f"- GPD scale multiplier: `{ce.gpd_scale_mult:.3f}`")
    
    with col2:
        st.markdown("**Severity Parameters**")
        st.markdown(f"- Lognormal μ: `{sp.mu:.3f}`")
        st.markdown(f"- Lognormal σ: `{sp.sigma:.3f}`")
        st.markdown(f"- GPD threshold quantile: `{sp.gpd_thresh_q:.3f}`")
        st.markdown(f"- GPD scale (β): `${sp.gpd_scale:,.0f}`")
        st.markdown(f"- GPD shape (ξ): `{sp.gpd_shape:.3f}`")
        
        st.markdown("**Simulation Config**")
        st.markdown(f"- Monte Carlo trials: `{cfg.trials:,}`")
        st.markdown(f"- Net worth: `${cfg.net_worth:,.0f}`")
        st.markdown(f"- Random seed: `{cfg.seed}`")

# Generate control effects
ce = effects_from_shares_improved(ctrl, action_shares, pattern_shares)

# Run simulations with caching
base_losses = cached_simulate(asdict(cfg), asdict(fp), asdict(sp))
ctrl_losses = cached_simulate(asdict(cfg), asdict(fp), asdict(sp), asdict(ce))

# Compute metrics
base_metrics = compute_metrics(base_losses, cfg.net_worth)
ctrl_metrics = compute_metrics(ctrl_losses, cfg.net_worth)

# Display control multipliers
st.caption(f"📊 Applied control multipliers → λ×{ce.lam_mult:.2f}, "
          f"P(any)×{ce.p_any_mult:.2f}, tail-scale×{ce.gpd_scale_mult:.2f}")

# VaR confidence intervals
var95_base = var_confidence_interval(base_losses, 0.95)
var95_ctrl = var_confidence_interval(ctrl_losses, 0.95)
st.caption(f"📈 VaR95: Base ${var95_base['point']:,.0f} "
          f"(±{(var95_base['ci_upper']-var95_base['ci_lower'])/2:,.0f}) | "
          f"Ctrl ${var95_ctrl['point']:,.0f} "
          f"(±{(var95_ctrl['ci_upper']-var95_ctrl['ci_lower'])/2:,.0f})")

# Summary table
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("EAL Baseline", f"${base_metrics['EAL']:,.0f}")
    st.metric("VaR95 Baseline", f"${base_metrics['VaR95']:,.0f}")

with col2:
    st.metric("EAL Controlled", f"${ctrl_metrics['EAL']:,.0f}", 
             delta=f"-${base_metrics['EAL'] - ctrl_metrics['EAL']:,.0f}")
    st.metric("VaR95 Controlled", f"${ctrl_metrics['VaR95']:,.0f}",
             delta=f"-${base_metrics['VaR95'] - ctrl_metrics['VaR95']:,.0f}")

with col3:
    control_cost = costs.total()
    delta_eal = base_metrics['EAL'] - ctrl_metrics['EAL']
    rosi = ((delta_eal - control_cost) / control_cost * 100) if control_cost > 0 else 0
    
    st.metric("Control Cost", f"${control_cost:,.0f}")
    st.metric("ROSI", f"{rosi:.1f}%", delta=f"${delta_eal - control_cost:,.0f} net benefit")

# Detailed metrics table
summary_data = {
    "Metric": ["EAL", "VaR95", "VaR99", "CVaR95", "Max Loss", "P(Ruin)"],
    "Baseline": [base_metrics['EAL'], base_metrics['VaR95'], base_metrics['VaR99'],
                 base_metrics['CVaR95'], base_metrics['Max'], base_metrics['P(Ruin)']],
    "Controlled": [ctrl_metrics['EAL'], ctrl_metrics['VaR95'], ctrl_metrics['VaR99'],
                   ctrl_metrics['CVaR95'], ctrl_metrics['Max'], ctrl_metrics['P(Ruin)']]
}

summary_df = pd.DataFrame(summary_data)

format_map = {
    "Baseline": "${:,.2f}",
    "Controlled": "${:,.2f}",
}

st.dataframe(
    summary_df.style.format(format_map, na_rep="—"),
    use_container_width=True
)

# ============================================================================
# CONTROL ISOLATION ANALYSIS
# ============================================================================

st.header("🔬 Control Isolation Analysis")
st.caption("Individual control effectiveness (all others off)")

baseline_eal = base_metrics['EAL']

iso = []
for name in ["server", "media", "error", "external"]:
    ctrl_iso = ControlSet(**{k: (k == name) for k in ["server", "media", "error", "external"]})
    ce_iso = effects_from_shares_improved(ctrl_iso, action_shares, pattern_shares)
    
    # Use incremented seed for each isolation test
    cfg_iso = ModelConfig(trials=cfg.trials, net_worth=cfg.net_worth, 
                         seed=cfg.seed + hash(name) % 1000)
    losses_iso = cached_simulate(asdict(cfg_iso), asdict(fp), asdict(sp), asdict(ce_iso))
    met_iso = compute_metrics(losses_iso, cfg.net_worth)
    
    dEAL = baseline_eal - met_iso["EAL"]
    cost = getattr(costs, name)
    
    iso.append({
        "Control": name.title(),
        "ΔEAL": dEAL,
        "Cost": cost,
        "ΔEAL per $": (dEAL / cost) if cost > 0 else np.nan,
        "ROSI %": ((dEAL - cost) / cost * 100) if cost > 0 else np.nan
    })

iso_df = pd.DataFrame(iso).sort_values("ΔEAL per $", ascending=False)

st.dataframe(
    iso_df.style.format({
        "ΔEAL": "${:,.0f}",
        "Cost": "${:,.0f}",
        "ΔEAL per $": "{:,.2f}",
        "ROSI %": "{:.1f}%"
    }),
    use_container_width=True
)

# Find best ROSI
best_idx = iso_df['ROSI %'].idxmax()
best = iso_df.loc[best_idx]
st.success(f"🏆 Best ROSI: {best['Control']} ({best['ROSI %']:.1f}%)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

st.header("📊 Loss Distributions")

# Loss exceedance curves
lec_points = 100
lec_b = cached_lec(base_losses, lec_points).assign(scenario="Baseline")
lec_c = cached_lec(ctrl_losses, lec_points).assign(scenario="Controlled")
lec_combined = pd.concat([lec_b, lec_c])

fig_lec = px.line(lec_combined, x="Loss", y="Exceedance_Prob", color="scenario",
                  title="Loss Exceedance Curve",
                  labels={"Loss": "Loss Amount ($)", "Exceedance_Prob": "P(Loss ≥ x)"})
fig_lec.update_xaxes(type="log")
fig_lec.update_yaxes(type="log", range=[-2.5, 0])
st.plotly_chart(fig_lec, use_container_width=True)

# Histograms
col1, col2 = st.columns(2)

with col1:
    fig_hist_base = log_hist_figure(base_losses, "Baseline Loss Distribution")
    st.plotly_chart(fig_hist_base, use_container_width=True)

with col2:
    fig_hist_ctrl = log_hist_figure(ctrl_losses, "Controlled Loss Distribution")
    st.plotly_chart(fig_hist_ctrl, use_container_width=True)

# ============================================================================
# PORTFOLIO BATCH ANALYSIS
# ============================================================================

with st.expander("📁 Portfolio batch (CSV)", expanded=False):
    st.markdown("Upload a CSV with columns: `account_id`, `net_worth`, `lam`, `p_any`, etc.")
    up = st.file_uploader("Accounts CSV", type=["csv"])
    
    if up:
        df = pd.read_csv(up)
        st.write(f"Loaded {len(df)} accounts")
        
        if st.button("Run Portfolio Analysis"):
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in df.iterrows():
                # Extract parameters from CSV with robust type conversion
                account_id = row.get('account_id', f'Account_{idx}')
                
                account_net_worth = pd.to_numeric(row.get('net_worth', 100e6), errors='coerce')
                account_lam = pd.to_numeric(row.get('lam', 2.0), errors='coerce')
                account_p_any = pd.to_numeric(row.get('p_any', 0.7), errors='coerce')
                
                # Validate and clamp to sensible ranges
                account_net_worth = float(account_net_worth if np.isfinite(account_net_worth) else 100e6)
                account_lam = float(account_lam if np.isfinite(account_lam) else 2.0)
                account_p_any = float(np.clip(account_p_any if np.isfinite(account_p_any) else 0.7, 0.0, 1.0))
                
                # Run simulation for this account
                cfg_account = ModelConfig(trials=cfg.trials, net_worth=account_net_worth, 
                                         seed=cfg.seed + idx)
                fp_account = FreqParams(lam=account_lam, p_any=account_p_any, 
                                       negbin=fp.negbin, r=fp.r)
                
                losses_account = cached_simulate(asdict(cfg_account), asdict(fp_account), 
                                                asdict(sp))
                metrics_account = compute_metrics(losses_account, account_net_worth)
                
                results.append({
                    'account_id': account_id,
                    'EAL': metrics_account['EAL'],
                    'VaR95': metrics_account['VaR95'],
                    'VaR99': metrics_account['VaR99'],
                    'P(Ruin)': metrics_account['P(Ruin)']
                })
                
                progress_bar.progress((idx + 1) / len(df))
            
            results_df = pd.DataFrame(results)
            st.success("✓ Portfolio analysis complete!")
            st.dataframe(results_df, use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="portfolio_results.csv",
                mime="text/csv"
            )

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

st.sidebar.markdown("---")
if st.sidebar.button("💾 Export Configuration"):
    export_config = {
        "schema_version": "1.0.0",
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "model": asdict(cfg),
        "frequency": asdict(fp),
        "severity": asdict(sp),
        "controls": asdict(ctrl),
        "costs": asdict(costs),
        "action_shares": action_shares,
        "pattern_shares": pattern_shares
    }
    
    st.sidebar.download_button(
        label="Download config.json",
        data=json.dumps(export_config, indent=2),
        file_name="cyber_loss_config.json",
        mime="application/json"
    )
