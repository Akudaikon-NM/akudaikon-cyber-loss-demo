import streamlit as st
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Optional
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import lognorm
from dataclasses import is_dataclass  # add
import os

def _to_dict(x):
    """Return a plain dict whether x is a dataclass or already a dict."""
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return x
    # last-ditch: fall back to __dict__ if present
    return dict(x.__dict__) if hasattr(x, "__dict__") else x

# KEEP this one at the very top (already there)
st.set_page_config(page_title="Akudaikon | Cyber-Loss Demo", layout="wide")

# Title and caption (removed duplicate set_page_config)
st.title("Akudaikon | Cyber-Loss Demo")
st.caption("Monte Carlo loss model with control ROI, diagnostics, and optional Bayesian frequency.")
# >>> BEGIN: Help & How-To
with st.expander("❓ Help & How to Use This App", expanded=False):
    st.markdown("""
### What this app does
This is a **Monte Carlo cyber-loss model** that estimates **Expected Annual Loss (EAL)**, **VaR95/99**, **CVaR95**, **P(Ruin)**, and an **LEC**.  
It also quantifies **control ROI** by comparing **ΔEAL** to your **annual control cost** (ROSI).

---

### Inputs (left sidebar)

**🎲 Simulation Config**
- **Monte Carlo Trials**: number of simulated years. More ⇒ smoother estimates.
- **Net Worth**: used to compute **P(Ruin)** (loss ≥ net worth).
- **Random Seed**: reproducibility.
- **Cost per record / Record cap**: used when you switch to the **Records-based** model.

**📊 Frequency Parameters**
- **λ**: average number of incidents per year.
- **P(any loss | incident)**: chance an incident creates a dollar loss.
- **Use Negative Binomial + r**: over-dispersion of incident counts (lower **r** ⇒ fatter frequency tails).

**🔬 Bayesian Frequency Update (optional)**
- Enter **Observed incidents (k)** over **Observation period (T)** to compute an updated **λ** via a Gamma-Poisson posterior.

**💰 Severity Parameters**
- **Monetary model (GPD/Lognormal)**: body (**μ, σ**), tail threshold (quantile), tail **scale β** and **shape ξ**.  
  _Keep **ξ < 0.5** for realistic tails; **ξ ≥ 1** ⇒ infinite mean tail._
- **Records-based**: records ~ lognormal(**μ, σ**), multiplied by **$ / record** (typical 100–300), optional **record cap**.

**🛡️ Controls & 💵 Costs**
- Toggle **Server hardening**, **Media encryption**, **Error reduction**, **External monitoring**.  
- Enter annual costs to compute **ROSI**.

**📁 Load parameter JSON**
- Load action/pattern shares and other defaults from a JSON file.

---

### Reading the results

**🎯 Metrics**
- **EAL**: expected annual loss (mean).
- **VaR95/99**: thresholds not exceeded 95%/99% of the time.
- **CVaR95**: mean loss **above** VaR95.
- **Max**: worst simulated year.
- **P(Ruin)**: probability annual loss ≥ net worth.

**📈 Confidence Intervals**
- Bootstrap CIs for **VaR95** and **EAL** show sampling uncertainty.

**🔬 Control Isolation**
- Each control **by itself** (others off): ΔEAL, Benefit per $, and ROSI%.

**🧩 Marginal ROI**
- From your **current** bundle, shows the next control’s **incremental** ΔEAL and marginal ROSI%.

**📊 Distributions**
- **LEC** (log–log) for Baseline vs Controlled.  
- Log-scaled histograms of annual loss.

**📁 Portfolio batch**
- Upload accounts (`account_id, net_worth, lam, p_any, ...`) to batch EAL/VaR/P(Ruin).

---

### Quick sanity checks
- Turning **External monitoring** on lowers **λ** and slightly shrinks tails.
- Higher **σ** pushes **VaR95/99** up more than EAL.
- With **NegBin**, lower **r** ⇒ more tail risk than mean shift.
- **Server hardening** is strongest when hacking/web patterns dominate.
- **Media encryption** helps when physical loss of assets matters.
- Avoid **ξ ≥ 1** (infinite mean tail).
    """)
# >>> END: Help & How-To

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
    use_records: bool = False
    records_mu: float = 10.0
    records_sigma: float = 2.0
    cost_per_record: float = 150.0

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
    if not sp.use_records:
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
            
            if sp.use_records:
                # Records-based loss model
                n_records = np.exp(np.random.normal(sp.records_mu, sp.records_sigma))
                if cfg.record_cap > 0:
                    n_records = min(n_records, cfg.record_cap)
                loss = n_records * sp.cost_per_record
            else:
                # Monetary severity model (lognormal body + GPD tail)
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
    ex_prob = np.clip(ex_prob, 1.0/len(losses), 1.0)  # keep strictly positive
    
    return pd.DataFrame({"Loss": grid, "Exceedance_Prob": ex_prob})

@st.cache_data(show_spinner=False)
def cached_lec(losses, n):
    """Cached LEC calculation."""
    return lec(losses, n=n)


# ============================================================================
# CIS MAPPING (VERIS -> CIS)
# ============================================================================

def load_cis_mapping():
    """
    Loads a CSV with mappings from VERIS 'action'/'pattern' to CIS controls.
    Supported column names (case-insensitive):
        - For VERIS keys: 'action', 'pattern', 'veris', 'veris_key'
        - For CIS id:     'cis', 'cis_control', 'cis_id', 'cis_v8_id'
        - For CIS title:  'cis_title', 'cis_name', 'title', 'name'
    Returns: dict with {"loaded": bool, "action": {k:[...ids]}, "pattern": {k:[...ids]}}
    """
    paths = ["data/veris_to_cis_lookup.csv", "veris_to_cis_lookup.csv"]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                cols = {c.lower(): c for c in df.columns}
                # identify columns
                veris_cols = [c for c in ["action", "pattern", "veris", "veris_key"] if c in cols]
                cis_id_col = next((cols[c] for c in ["cis", "cis_control", "cis_id", "cis_v8_id"] if c in cols), None)
                cis_title_col = next((cols[c] for c in ["cis_title", "cis_name", "title", "name"] if c in cols), None)

                amap, pmap = {}, {}
                for _, row in df.iterrows():
                    # pick the first VERIS-like field that is not null
                    veris_val = None
                    for vc in veris_cols:
                        v = str(row.get(cols[vc])).strip() if pd.notna(row.get(cols[vc])) else None
                        if v:
                            veris_val = v
                            break
                    if not veris_val:
                        continue

                    cis_id = str(row.get(cis_id_col)).strip() if cis_id_col and pd.notna(row.get(cis_id_col)) else ""
                    cis_title = str(row.get(cis_title_col)).strip() if cis_title_col and pd.notna(row.get(cis_title_col)) else ""
                    cis_display = (f"{cis_id} – {cis_title}".strip(" –")) if cis_id or cis_title else ""

                    # place into action or pattern dict depending on exact key match if possible
                    key_lower = veris_val.lower()
                    # try to bucket to action vs pattern by name heuristics
                    # (works with your defaults; still captures generic 'veris')
                    if key_lower in [k.lower() for k in DEFAULT_ACTION_SHARES.keys()]:
                        amap.setdefault(veris_val, set()).add(cis_display or cis_id)
                    elif key_lower in [k.lower() for k in DEFAULT_PATTERN_SHARES.keys()]:
                        pmap.setdefault(veris_val, set()).add(cis_display or cis_id)
                    else:
                        # unknown; put into both buckets so it still shows up
                        amap.setdefault(veris_val, set()).add(cis_display or cis_id)
                        pmap.setdefault(veris_val, set()).add(cis_display or cis_id)

                # cast sets to sorted lists
                amap = {k: sorted([x for x in v if x]) for k, v in amap.items()}
                pmap = {k: sorted([x for x in v if x]) for k, v in pmap.items()}
                return {"loaded": True, "action": amap, "pattern": pmap, "path": p}
            except Exception as e:
                st.sidebar.warning(f"Failed to read CIS mapping CSV: {e}")
                break
    return {"loaded": False, "action": {}, "pattern": {}, "path": None}


def cis_for_profile(action_shares, pattern_shares, cis_map, action_thresh=0.10, pattern_thresh=0.10, top_n=15):
    """Aggregate CIS controls for the current action/pattern mix (threshold = share cutoff)."""
    if not cis_map.get("loaded"):
        return []
    out = set()
    for a, s in action_shares.items():
        if s >= action_thresh and a in cis_map["action"]:
            out.update(cis_map["action"][a])
    for p, s in pattern_shares.items():
        if s >= pattern_thresh and p in cis_map["pattern"]:
            out.update(cis_map["pattern"][p])
    return sorted(list(out))[:top_n]

def effects_from_shares_improved(ctrl: ControlSet, action_shares: dict, pattern_shares: dict) -> ControlEffects:
    """Compute control effects from action/pattern shares with data-driven heuristics."""
    a = _normalize_shares(action_shares)
    p = _normalize_shares(pattern_shares)

    lam_mult   = 1.0
    p_any_mult = 1.0
    gpd_mult   = 1.0

    hack_intensity   = a.get("hacking", 0) + p.get("Web Applications", 0) + p.get("Crimeware", 0)
    misuse_intensity = a.get("misuse", 0)  + p.get("Privilege Misuse", 0)
    error_intensity  = a.get("error", 0)   + p.get("Miscellaneous Errors", 0)
    physical_int     = a.get("physical", 0)+ p.get("Lost and Stolen Assets", 0)

    if ctrl.server:
        lam_mult   *= (1 - 0.35 * hack_intensity)
        p_any_mult *= (1 - 0.20 * hack_intensity)
        gpd_mult   *= (1 - 0.15 * hack_intensity)
    if ctrl.media:
        lam_mult   *= (1 - 0.25 * physical_int)
        p_any_mult *= (1 - 0.25 * physical_int)
    if ctrl.error:
        lam_mult   *= (1 - 0.20 * error_intensity)
        p_any_mult *= (1 - 0.25 * error_intensity)
    if ctrl.external:
        lam_mult   *= (1 - 0.30 * (hack_intensity + misuse_intensity))
        gpd_mult   *= (1 - 0.20 * (hack_intensity + misuse_intensity))

    lam_mult   = float(np.clip(lam_mult,   0.2, 1.0))
    p_any_mult = float(np.clip(p_any_mult, 0.2, 1.0))
    gpd_mult   = float(np.clip(gpd_mult,   0.2, 1.0))

    return ControlEffects(lam_mult=lam_mult, p_any_mult=p_any_mult, gpd_scale_mult=gpd_mult)



def eal_ci(losses: np.ndarray, n_boot: int = 1000, alpha: float = 0.95):
    """Bootstrap confidence interval for EAL."""
    m = len(losses)
    boots = np.mean(
        np.random.choice(losses, size=(n_boot, m), replace=True),
        axis=1
    )
    lo, hi = np.percentile(boots, [(1-alpha)/2*100, (1+alpha)/2*100])
    return float(np.mean(losses)), float(lo), float(hi)

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

def _stable_seed_from(s: str, base: int = 0):
    """Generate consistent seed from string ID."""
    # consistent tiny hash → [0, 9999]
    return base + (abs(hash(str(s))) % 10000)

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
# CIS mapping panel
with st.sidebar.expander("📚 CIS Mapping (VERIS → CIS)", expanded=False):
    cis_map = load_cis_mapping()
    if cis_map["loaded"]:
        st.success(f"✓ CIS mapping loaded from {cis_map['path']}")
        st.caption("CIS recommendations will appear with actions/patterns and in ROI tables.")
    else:
        st.info("Add **data/veris_to_cis_lookup.csv** (or root **veris_to_cis_lookup.csv**) to enable CIS recommendations.")


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

# Reset button for shares
if st.sidebar.button("🔄 Reset action/pattern shares to defaults"):
    st.session_state.pop("_action_shares", None)
    st.session_state.pop("_pattern_shares", None)
    st.rerun()

# Model configuration
with st.sidebar.expander("🎲 Simulation Config", expanded=True):
    trials = st.number_input("Monte Carlo Trials", 1000, 100000, 10000, 1000)
    net_worth = st.number_input("Net Worth ($M)", 1.0, 10000.0, 100.0, 10.0) * 1e6
    seed = st.number_input("Random Seed", 0, 9999, 42)
    
    # Record-related parameters
    st.markdown("**Records Parameters**")
    cost_per_record_input = st.number_input(
        "Cost per record ($)", 
        1.0, 10000.0, 150.0, 10.0,
        help="Cost per exposed/lost record (used in records-based mode)"
    )
    record_cap = st.number_input(
        "Record cap (0 = unlimited)", 
        0, 1000000000, 0, 1000000, 
        help="Maximum records per incident (0 for no cap)"
    )

cfg = ModelConfig(
    trials=trials, 
    net_worth=net_worth, 
    seed=seed, 
    record_cap=record_cap,
    cost_per_record=cost_per_record_input
)

# Frequency parameters
with st.sidebar.expander("📊 Frequency Parameters", expanded=True):
    lam = st.number_input("λ (mean incidents/year)", 0.1, 20.0, 2.0, 0.1)
    p_any = st.slider("P(any loss | incident)", 0.1, 0.95, 0.7, 0.05)
    negbin = st.checkbox("Use Negative Binomial", value=False)
    r = st.number_input("NegBin dispersion (r)", 0.5, 10.0, 1.0, 0.5) if negbin else 1.0
    
    if negbin:
        st.caption("ℹ️ Lower r ⇒ more over-dispersion (fatter frequency tails).")
    
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
    use_records = st.checkbox("Use records-based loss model", value=False,
                              help="Toggle between monetary severity (GPD/lognormal) and records × $/record")
    
    if use_records:
        st.markdown("**Records Model**")
        records_mu = st.number_input("Records lognormal μ", 6.0, 20.0, 10.0, 0.5,
                                     help="Mean log(records) — e.g., μ=10 → median ~22k records")
        records_sigma = st.number_input("Records lognormal σ", 0.5, 4.0, 2.0, 0.1,
                                        help="Std dev of log(records)")
        
        # Use cost_per_record from cfg (set in Simulation Config)
        sp = SevParams(
            use_records=True,
            records_mu=records_mu,
            records_sigma=records_sigma,
            cost_per_record=cfg.cost_per_record,  # Pull from config
            # Dummy values for monetary params (not used)
            mu=12.0, sigma=2.0, gpd_thresh_q=0.95, gpd_scale=1e6, gpd_shape=0.3
        )
        
        # Show expected records distribution
        median_records = int(np.exp(records_mu))
        mean_records = int(np.exp(records_mu + records_sigma**2 / 2))
        st.caption(f"📊 Expected records: median={median_records:,}, mean={mean_records:,}")
        st.caption(f"💵 Implied median loss: ${median_records * cfg.cost_per_record:,.0f}")
        st.caption(f"💵 Implied mean loss: ${mean_records * cfg.cost_per_record:,.0f}")
        
    else:
        st.markdown("**Monetary Model (GPD/Lognormal)**")
        mu = st.number_input("Lognormal μ", 8.0, 16.0, 12.0, 0.5)
        sigma = st.number_input("Lognormal σ", 0.5, 4.0, 2.0, 0.1)
        gpd_thresh_q = st.slider("GPD threshold quantile", 0.85, 0.99, 0.95, 0.01)
        gpd_scale = st.number_input("GPD scale ($K)", 100.0, 10000.0, 1000.0, 100.0) * 1000
        gpd_shape = st.number_input("GPD shape (ξ)", 0.0, 1.0, 0.3, 0.05)
        
        sp = SevParams(
            use_records=False,
            mu=mu, sigma=sigma, 
            gpd_thresh_q=gpd_thresh_q, 
            gpd_scale=gpd_scale, 
            gpd_shape=gpd_shape,
            # Dummy values for records params (not used in monetary mode)
            records_mu=10.0, records_sigma=2.0, cost_per_record=cfg.cost_per_record
        )
        
        # Clamp severity params
        sp.gpd_scale = float(max(1.0, sp.gpd_scale))
        
        # Severity parameter warnings
        if sp.gpd_shape >= 1.0:
            st.sidebar.warning("⚠️ ξ (GPD shape) ≥ 1 yields infinite mean tail. Results may be unstable.")
        if sp.sigma > 2.5:
            st.sidebar.warning("⚠️ Lognormal σ is quite high; body may dominate and inflate EAL/VaR.")

# Advanced confidence interval settings
with st.sidebar.expander("📐 Confidence Intervals (advanced)", expanded=False):
    n_boot = st.slider("Bootstrap reps for VaR/EAL CI", 100, 5000, 1000, 100)

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

# Generate control effects (must be before Assumption Summary)
try:
    ce = effects_from_shares_improved(ctrl, action_shares, pattern_shares)
except Exception as _e:
    # If anything odd happens, fall back to neutral multipliers
    ce = ControlEffects()
# absolutely ensure it's never None
ce = _ensure_control_effects(ce)

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
        
                # Use a safe alias so formatting never crashes
        _ce = _ensure_control_effects(ce)
st.markdown("**Control Effects**")
st.markdown(f"- λ multiplier: `{_ce.lam_mult:.3f}`")
st.markdown(f"- P(any) multiplier: `{_ce.p_any_mult:.3f}`")
st.markdown(f"- GPD scale multiplier: `{_ce.gpd_scale_mult:.3f}`")

    a = _normalize_shares(action_shares)
    p = _normalize_shares(pattern_shares)

    lam_mult  = 1.0
    p_any_mult= 1.0
    gpd_mult  = 1.0

    hack_intensity   = a.get("hacking", 0) + p.get("Web Applications", 0) + p.get("Crimeware", 0)
    misuse_intensity = a.get("misuse", 0)  + p.get("Privilege Misuse", 0)
    error_intensity  = a.get("error", 0)   + p.get("Miscellaneous Errors", 0)
    physical_int     = a.get("physical",0) + p.get("Lost and Stolen Assets", 0)

    if ctrl.server:
        lam_mult  *= (1 - 0.35 * hack_intensity)
        p_any_mult*= (1 - 0.20 * hack_intensity)
        gpd_mult  *= (1 - 0.15 * hack_intensity)
    if ctrl.media:
        lam_mult  *= (1 - 0.25 * physical_int)
        p_any_mult*= (1 - 0.25 * physical_int)
    if ctrl.error:
        lam_mult  *= (1 - 0.20 * error_intensity)
        p_any_mult*= (1 - 0.25 * error_intensity)
    if ctrl.external:
        lam_mult  *= (1 - 0.30 * (hack_intensity + misuse_intensity))
        gpd_mult  *= (1 - 0.20 * (hack_intensity + misuse_intensity))

    lam_mult   = float(np.clip(lam_mult,   0.2, 1.0))
    p_any_mult = float(np.clip(p_any_mult, 0.2, 1.0))
    gpd_mult   = float(np.clip(gpd_mult,   0.2, 1.0))

    return ControlEffects(lam_mult=lam_mult, p_any_mult=p_any_mult, gpd_scale_mult=gpd_mult)
        # Use a safe alias so formatting never crashes
        _ce = _ensure_control_effects(ce)
        st.markdown("**Control Effects**")
        st.markdown(f"- λ multiplier: `{_ce.lam_mult:.3f}`")
        st.markdown(f"- P(any) multiplier: `{_ce.p_any_mult:.3f}`")
        st.markdown(f"- GPD scale multiplier: `{_ce.gpd_scale_mult:.3f}`")


    
    with col2:
        st.markdown("**Severity Parameters**")
        if sp.use_records:
            st.markdown("**Mode:** Records-based")
            st.markdown(f"- Records lognormal μ: `{sp.records_mu:.3f}`")
            st.markdown(f"- Records lognormal σ: `{sp.records_sigma:.3f}`")
            st.markdown(f"- Cost per record: `${sp.cost_per_record:.2f}`")
            if cfg.record_cap > 0:
                st.markdown(f"- Record cap: `{cfg.record_cap:,}`")
            
            # Show expected losses
            median_records = int(np.exp(sp.records_mu))
            st.markdown(f"- Implied median loss: `${median_records * sp.cost_per_record:,.0f}`")
        else:
            st.markdown("**Mode:** Monetary (GPD/Lognormal)")
            st.markdown(f"- Lognormal μ: `{sp.mu:.3f}`")
            st.markdown(f"- Lognormal σ: `{sp.sigma:.3f}`")
            st.markdown(f"- GPD threshold quantile: `{sp.gpd_thresh_q:.3f}`")
            st.markdown(f"- GPD scale (β): `${sp.gpd_scale:,.0f}`")
            st.markdown(f"- GPD shape (ξ): `{sp.gpd_shape:.3f}`")
        
        st.markdown("**Simulation Config**")
        st.markdown(f"- Monte Carlo trials: `{cfg.trials:,}`")
        st.markdown(f"- Net worth: `${cfg.net_worth:,.0f}`")
        st.markdown(f"- Random seed: `{cfg.seed}`")
# CIS recommendations for the current action/pattern mix
if cis_map.get("loaded"):
    st.subheader("🔗 CIS Recommendations for Current Mix")
    cis_list = cis_for_profile(action_shares, pattern_shares, cis_map, action_thresh=0.10, pattern_thresh=0.10, top_n=20)
    if cis_list:
        st.markdown(
            "- " + "\n- ".join(cis_list)
        )
    else:
        st.caption("No CIS recommendations triggered at current thresholds (raise dominant actions/patterns or lower thresholds).")

# Run simulations with caching
base_losses = cached_simulate(_to_dict(cfg), _to_dict(fp), _to_dict(sp))
ctrl_losses = cached_simulate(_to_dict(cfg), _to_dict(fp), _to_dict(sp), _to_dict(ce))



# Compute metrics
base_metrics = compute_metrics(base_losses, cfg.net_worth)
ctrl_metrics = compute_metrics(ctrl_losses, cfg.net_worth)

# Display control multipliers
st.caption(f"📊 Applied control multipliers → λ×{ce.lam_mult:.2f}, "
          f"P(any)×{ce.p_any_mult:.2f}, tail-scale×{ce.gpd_scale_mult:.2f}")

# VaR confidence intervals
var95_base = var_confidence_interval(base_losses, 0.95, n_boot=n_boot)
var95_ctrl = var_confidence_interval(ctrl_losses, 0.95, n_boot=n_boot)
st.caption(f"📈 VaR95: Base ${var95_base['point']:,.0f} "
          f"(±{(var95_base['ci_upper']-var95_base['ci_lower'])/2:,.0f}) | "
          f"Ctrl ${var95_ctrl['point']:,.0f} "
          f"(±{(var95_ctrl['ci_upper']-var95_ctrl['ci_lower'])/2:,.0f})")

# EAL confidence intervals
eal_b, eal_lo_b, eal_hi_b = eal_ci(base_losses, n_boot=n_boot)
eal_c, eal_lo_c, eal_hi_c = eal_ci(ctrl_losses, n_boot=n_boot)
st.caption(f"📊 EAL CI: Base ${eal_b:,.0f} [{eal_lo_b:,.0f}, {eal_hi_b:,.0f}] | "
           f"Ctrl ${eal_c:,.0f} [{eal_lo_c:,.0f}, {eal_hi_c:,.0f}]")

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
    net_benefit = delta_eal - control_cost
    rosi = (net_benefit / control_cost * 100) if control_cost > 0 else 0
    
    st.metric("Control Cost", f"${control_cost:,.0f}")
    st.metric("ΔEAL (Risk ↓)", f"${delta_eal:,.0f}")
    st.metric("ROSI", f"{rosi:.1f}%", delta=f"${net_benefit:,.0f} net benefit")

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

# Fixed seed bumps for reproducibility
_iso_seed_bumps = {"server": 101, "media": 202, "error": 303, "external": 404}

iso = []
for name in ["server", "media", "error", "external"]:
    ctrl_iso = ControlSet(**{k: (k == name) for k in ["server", "media", "error", "external"]})
    ce_iso = effects_from_shares_improved(ctrl_iso, action_shares, pattern_shares)
    
    # Use fixed seed bump for reproducibility
    cfg_iso = ModelConfig(
        trials=cfg.trials, 
        net_worth=cfg.net_worth, 
        seed=cfg.seed + _iso_seed_bumps[name]
    )
    losses_iso  = cached_simulate(_to_dict(cfg_iso), _to_dict(fp), _to_dict(sp), _to_dict(ce_iso))
    met_iso = compute_metrics(losses_iso, cfg.net_worth)
    
    dEAL = baseline_eal - met_iso["EAL"]
    cost = getattr(costs, name)
    
    iso.append({
        "Control": name.title(),
        "ΔEAL ($/yr)": dEAL,
        "Cost ($/yr)": cost,
        "Benefit per $": (dEAL / cost) if cost > 0 else np.nan,
        "ROSI %": ((dEAL - cost) / cost * 100) if cost > 0 else np.nan,
        "CIS (suggested)": cis_for_control(name, cis_map)
    })

def rank_cis_controls(cis_map, action_shares, pattern_shares, top_n=10):
    """
    Rank CIS controls by how strongly they are implied by the current
    action/pattern mix. Actions get weight 1.0, patterns 0.8 (tweak as you like).
    Returns a DataFrame with columns: CIS Control | Score | Why.
    """
    # Safe empty return if mapping not available
    if not cis_map or not cis_map.get("loaded"):
        return pd.DataFrame(columns=["CIS Control", "Score", "Why"])

    # Normalize shares to sum to 1
    a = _normalize_shares(action_shares)
    p = _normalize_shares(pattern_shares)

    scores = {}
    reasons = {}

    # Weight contributions from actions
    for a_key, w in a.items():
        for cis in cis_map["action"].get(a_key, []):
            scores[cis] = scores.get(cis, 0.0) + 1.0 * w
            reasons.setdefault(cis, []).append(f"action:{a_key} ({w:.0%})")

    # Weight contributions from patterns
    for p_key, w in p.items():
        for cis in cis_map["pattern"].get(p_key, []):
            scores[cis] = scores.get(cis, 0.0) + 0.8 * w   # pattern weight
            reasons.setdefault(cis, []).append(f"pattern:{p_key} ({w:.0%})")

    if not scores:
        return pd.DataFrame(columns=["CIS Control", "Score", "Why"])

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    rows = [
        {"CIS Control": cis, "Score": score, "Why": "; ".join(reasons.get(cis, []))}
        for cis, score in ranked
    ]
    return pd.DataFrame(rows)

iso_df = pd.DataFrame(iso).sort_values("Benefit per $", ascending=False)

st.dataframe(
    iso_df.style.format({
        "ΔEAL ($/yr)": "${:,.0f}",
        "Cost ($/yr)": "${:,.0f}",
        "Benefit per $": "{:,.2f}",
        "ROSI %": "{:.1f}%"
    }),
    use_container_width=True
)

# Download isolation results
iso_csv = iso_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Isolation ROI (CSV)", iso_csv, "isolation_roi.csv", "text/csv")

# Find best ROSI (safe for all-NaN)
if iso_df['ROSI %'].notna().any():
    best_idx = iso_df['ROSI %'].idxmax()
    best = iso_df.loc[best_idx]
    st.success(f"🏆 Best ROSI: {best['Control']} ({best['ROSI %']:.1f}%)")
else:
    st.info("Set non-zero control costs to compute ROSI (currently all costs are $0).")

# ============================================================================
# MARGINAL ROI ANALYSIS
# ============================================================================

st.header("🧩 Marginal ROI (from current bundle)")
st.caption("Cost-effectiveness of adding one more control to your selected bundle")

current_losses = ctrl_losses
current_eal = ctrl_metrics['EAL']

marg = []
for name in ["server", "media", "error", "external"]:
    if getattr(ctrl, name):  # already in bundle
        continue
    
    # bundle + this control
    ctrl_plus = ControlSet(**{
        k: (getattr(ctrl, k) or (k == name)) 
        for k in ["server", "media", "error", "external"]
    })
    ce_plus = effects_from_shares_improved(ctrl_plus, action_shares, pattern_shares)
    cfg_plus = ModelConfig(trials=cfg.trials, net_worth=cfg.net_worth, seed=cfg.seed + 777)
    losses_plus = cached_simulate(_to_dict(cfg_plus), _to_dict(fp), _to_dict(sp), _to_dict(ce_plus))

    eal_plus = float(np.mean(losses_plus))
    
    dEAL = current_eal - eal_plus
    cost = getattr(costs, name)
    
    marg.append({
        "Add": name.title(),
        "ΔEAL from bundle ($/yr)": dEAL,
        "Incremental Cost ($/yr)": cost,
        "Marginal ROSI %": ((dEAL - cost) / cost * 100) if cost > 0 else np.nan,
        "CIS (suggested)": cis_for_control(name, cis_map)
    })


if marg:
    marg_df = pd.DataFrame(marg).sort_values("Marginal ROSI %", ascending=False)
    st.dataframe(
        marg_df.style.format({
            "ΔEAL from bundle ($/yr)": "${:,.0f}",
            "Incremental Cost ($/yr)": "${:,.0f}",
            "Marginal ROSI %": "{:.1f}%"
        }),
        use_container_width=True
    )

    marg_csv = marg_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Marginal ROI (CSV)", marg_csv, "marginal_roi.csv", "text/csv")
else:
    st.info("All controls are already selected; no marginal adds to evaluate.")

# >>> BEGIN: CIS recommendation table
st.subheader("🧭 CIS Control Recommendations (ranked by relevance)")
cis_rank_df = rank_cis_controls(cis_map, action_shares, pattern_shares, top_n=10)

if cis_rank_df.empty:
    st.info("Add data/veris_to_cis_lookup.csv (or root veris_to_cis_lookup.csv) to enable CIS recommendations.")
else:
    st.dataframe(
        cis_rank_df.style.format({"Score": "{:.2f}"}),
        use_container_width=True
    )


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
fig_lec.add_hline(y=0.01, line_dash="dot", opacity=0.2)
fig_lec.add_hline(y=0.001, line_dash="dot", opacity=0.2)
st.plotly_chart(fig_lec, use_container_width=True)

# Download LEC data
lec_export = lec_combined.rename(columns={"scenario": "Scenario"})
lec_csv = lec_export.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download LEC Points (CSV)", lec_csv, "lec_points.csv", "text/csv")

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
                account_lam       = pd.to_numeric(row.get('lam', 2.0), errors='coerce')
                account_p_any     = pd.to_numeric(row.get('p_any', 0.7), errors='coerce')
                
                # Validate and clamp to sensible ranges
                account_net_worth = float(account_net_worth if np.isfinite(account_net_worth) else 100e6)
                account_lam       = float(account_lam if np.isfinite(account_lam) else 2.0)
                account_p_any     = float(np.clip(account_p_any if np.isfinite(account_p_any) else 0.7, 0.0, 1.0))
                
                # Per-account config and frequency
                cfg_account = ModelConfig(
                    trials=cfg.trials, 
                    net_worth=account_net_worth, 
                    seed=_stable_seed_from(account_id, base=cfg.seed),
                    record_cap=cfg.record_cap,
                    cost_per_record=cfg.cost_per_record
                )
                fp_account = FreqParams(
                    lam=account_lam,
                    p_any=account_p_any,
                    negbin=fp.negbin,
                    r=fp.r
                )
                
                # Use the same severity params 'sp' selected in the sidebar
                losses_account  = cached_simulate(_to_dict(cfg_account), _to_dict(fp_account), _to_dict(sp))

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
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="portfolio_results.csv",
                mime="text/csv"
            )

with st.expander("🧪 Sanity check guide (what to expect)", expanded=False):
    st.markdown("""
**Expected behaviors (monetary model):**
- Turning **External monitoring** on should drop **λ** and modestly shrink tails.
- Increasing **σ** raises **VaR95/99** more than **EAL** (fatter body/tail).
- With **NegBin**, lowering **r** raises tail risk more than mean frequency.
- **ξ ≥ 1** ⇒ infinite mean tail; keep **ξ < 0.5** for realistic cyber losses.
- **Server hardening** strongest when hacking/web app patterns dominate.
- **Media encryption** primarily helps with physical asset loss patterns.
- **Control isolation** shows standalone value; **marginal ROI** shows incremental value from current bundle.

**Records-based model:**
- Higher **records μ** shifts the whole distribution right (more records per incident).
- Higher **records σ** increases variability—mega-breaches more likely.
- **Cost per record** scales linearly with loss (typical $100–$300 for PII/PHI).
- **Record cap** truncates the tail; VaR can drop materially when capped.
""")

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
