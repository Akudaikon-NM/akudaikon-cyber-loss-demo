import streamlit as st
import pandas as pd
import numpy as np
import json
from dataclasses import dataclass, asdict, is_dataclass
from typing import Optional
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import lognorm
import os

# Configure Streamlit page (title + full-width). Needs to be at the top.
st.set_page_config(page_title="Akudaikon | Cyber-Loss Demo", layout="wide")

# App header
st.title("Akudaikon | Cyber-Loss Demo")
st.caption("Monte Carlo loss model with control ROI, diagnostics, and optional Bayesian frequency.")

# ===========================
# HELP / HOW-TO (collapsible)
# ===========================
with st.expander("❓ Help & How to Use This App", expanded=False):
    st.markdown("""
## Overview
This app is a **Monte Carlo cyber-loss simulator** that helps you:
- Model annual cyber-loss distributions using frequency and severity parameters
- Evaluate control effectiveness and ROI
- Understand loss exceedance probabilities and tail risk
- Make data-driven cybersecurity investment decisions

---

## 📊 Core Concepts

### Frequency-Severity Framework
The app models losses using a **compound distribution**:
1. **Frequency**: How many incidents occur per year (Poisson or Negative Binomial)
2. **Severity**: How much each incident costs (Lognormal + GPD tail, or records-based)
3. **Annual Loss**: Sum of all incident losses in a simulated year

### Key Metrics
- **EAL (Expected Annual Loss)**: Average loss per year across all simulations
- **VaR (Value at Risk)**: Loss threshold at a percentile (e.g., VaR95 = 95th percentile loss)
- **CVaR (Conditional VaR)**: Average loss when losses exceed VaR threshold
- **P(Ruin)**: Probability that annual loss exceeds your net worth
- **ROSI (Return on Security Investment)**: `(Risk Reduction - Control Cost) / Control Cost × 100%`

---

## ⚙️ Input Parameters

### 🎲 Simulation Config
- **Monte Carlo Trials**: Number of simulated years (more = smoother distributions, 10,000 recommended)
- **Net Worth**: Total assets at risk; used to calculate P(Ruin)
- **Random Seed**: Set for reproducible results across runs
- **Cost per Record**: Dollar amount per exposed/lost record (used in records-based mode)
- **Record Cap**: Maximum records per incident (0 = unlimited)

### 📊 Frequency Parameters
Controls how often incidents occur:
- **λ (lambda)**: Mean incidents per year
  - Example: λ=2.0 means ~2 incidents/year on average
  - Typical range: 0.5–5.0 for most organizations
- **P(any loss | incident)**: Probability an incident produces a dollar loss
  - Not all incidents result in quantifiable losses
  - Typical range: 0.5–0.9
- **Distribution Type**:
  - **Poisson**: Standard, assumes independent events
  - **Negative Binomial**: Allows over-dispersion (clustering of incidents)
- **NegBin Dispersion (r)**: Lower r = fatter tails, more variance
  - r=1.0 is moderate over-dispersion
  - r<0.5 is high over-dispersion (use cautiously)

### 💰 Severity Parameters (Two Modes)

#### **Mode 1: Monetary Model** (default)
Models loss amounts directly in dollars using:
- **Lognormal Body**:
  - **μ (mu)**: Mean of log-losses (controls median loss)
    - μ=12 → median loss ≈ $163k
    - μ=14 → median loss ≈ $1.2M
  - **σ (sigma)**: Standard deviation of log-losses (controls spread)
    - σ=2.0 is typical (moderate variability)
    - Higher σ = more extreme small/large losses
- **GPD Tail** (Generalized Pareto Distribution):
  - **Threshold Quantile**: Where tail begins (e.g., 0.95 = top 5% of losses)
  - **Scale (β)**: Controls tail severity (higher = more extreme)
  - **Shape (ξ)**: Controls tail weight (0.2–0.4 typical, ≥1.0 unstable)

#### **Mode 2: Records-Based Model**
Models losses as: `Number of Records × Cost per Record`
- **Records μ**: Mean of log(records)
  - μ=10 → median ≈ 22,000 records
  - μ=12 → median ≈ 163,000 records
- **Records σ**: Variability in record counts
- Useful for breach scenarios with clear record exposure

### 🛡️ Control Selection
Four example controls (customize for your needs):
- **Server Hardening**: Reduces hacking/web app incidents
- **Media Encryption**: Reduces physical/lost-asset incidents
- **Error Reduction**: Reduces human error incidents
- **External Monitoring**: Reduces hacking/misuse incidents

Each control applies **multipliers** to λ, P(any loss), and tail severity based on your VERIS action/pattern mix.

### 💵 Control Costs
Annual operational cost per control (in $K):
- Enter realistic yearly costs (licenses, staffing, maintenance)
- Used to calculate ROSI and net benefit

---

## 📈 Outputs & Visualizations

### Main Results Dashboard
- **Metric Cards**: Compare baseline vs. controlled scenarios
  - Shows absolute values and deltas (improvements)
- **ROSI Calculation**: Shows return on security investment
  - \>100% = controls generate net benefit
  - <0% = controls cost more than risk reduction

### 🔬 Control Isolation Analysis
Tests each control **individually** (all others off):
- **ΔEAL**: Risk reduction from this control alone
- **Benefit per $**: How much risk reduction per dollar spent
- **ROSI %**: Return on investment for this control
- Use to identify most cost-effective controls

### 🧩 Marginal ROI
Shows value of adding **one more control** to your current bundle:
- Helps prioritize next investment
- Accounts for diminishing returns as you add controls

### 📊 Visualizations
- **Loss Exceedance Curve (LEC)**: P(Loss ≥ x) on log-log scale
  - Shows tail risk and extreme event probabilities
  - Compare baseline vs. controlled curves
- **Loss Histograms**: Distribution of annual losses (log scale)
  - See shape of loss distribution
  - Identify skewness and tail behavior

### 🔗 CIS Recommendations
If you upload `veris_to_cis_lookup.csv`:
- See which **CIS Controls v8** map to your VERIS profile
- Get ranked recommendations based on your action/pattern mix
- Links controls to industry-standard framework

---

## 🔬 Advanced Features

### Bayesian Frequency Update
If you have **observed incident data**:
1. Enable "Bayesian Frequency Update"
2. Enter observation period (years) and incident count
3. App updates λ using Bayesian posterior (Gamma-Poisson conjugate prior)
4. Blends your prior belief with actual data

### Confidence Intervals
- Bootstrap resampling provides **95% CIs** for VaR and EAL
- Shows statistical uncertainty in estimates
- More bootstrap reps = smoother CIs (but slower)

### Portfolio Batch Analysis
Upload CSV with multiple accounts/business units:
- Required columns: `account_id`, `net_worth`, `lam`, `p_any`
- Runs parallel simulations with per-account parameters
- Exports aggregated results for portfolio risk view

---

## 🎯 Typical Workflow

1. **Set baseline parameters** (frequency, severity) based on:
   - Historical incident data
   - Industry benchmarks (VCDB, IRIS20, etc.)
   - Expert judgment
   
2. **Review baseline metrics**:
   - Is EAL reasonable for your organization?
   - Do tail risks (VaR99, P(Ruin)) align with expectations?
   
3. **Load VERIS action/pattern shares** (optional):
   - Upload  with your threat profile
   - Or use defaults (VCDB-informed)
   
4. **Select controls** and set costs:
   - Check controls you're evaluating
   - Enter realistic annual costs
   
5. **Compare scenarios**:
   - Review metric deltas and ROSI
   - Check isolation analysis for best individual controls
   - Review marginal ROI for next investment
   
6. **Validate results**:
   - Use  (at bottom)
   - Export LEC points and review tail behavior
   - Adjust parameters if needed
   
7. **Export configuration**:
   - Save  for reproducibility
   - Share with stakeholders
   - Use for periodic updates

---

## 💡 Tips for Best Results

### Parameter Selection
- **Start conservative**: Use moderate λ and σ values, increase gradually
- **Validate with data**: If you have claims data, calibrate to match
- **Test sensitivity**: Vary parameters ±20% to see impact on metrics
- **Watch warnings**: App flags extreme/unstable parameter combinations

### Interpreting ROSI
- **Positive ROSI**: Control generates net benefit (good!)
- **High ROSI (>200%)**: Control is highly cost-effective
- **Negative ROSI**: Control costs exceed risk reduction (may still be worth it for compliance/reputation)
- **Compare marginal vs. isolated**: Marginal ROI shows diminishing returns

### Common Pitfalls
- **Over-fitting**: Don't tune parameters to match single data point
- **Ignoring tail**: VaR95 can be deceptive; check VaR99 and CVaR95 too
- **Zero costs**: Set realistic costs to get meaningful ROSI
- **Extreme severity**: σ>3 or ξ>0.5 can produce unstable results

### Using CIS Recommendations
- Upload `veris_to_cis_lookup.csv` to map VERIS threat patterns to CIS Controls v8
- The app analyzes your action/pattern profile and recommends relevant controls
- Use as starting point for control selection
- Customize based on your environment

**CSV Format:**
The lookup CSV should have two columns:
- `veris_field`: VERIS taxonomy path (e.g., `action.hacking.variety.Exploit vuln`)
- `cis_control`: CIS Control number (e.g., `7.0` for Continuous Vulnerability Management)

**Example CSV structure:**
```
veris_field,cis_control
action.hacking.variety.Exploit vuln,7.0
action.hacking.variety.Use of stolen creds,5.0
action.malware.variety.Ransomware,11.0
action.social.variety.Phishing,14.0
action.hacking.vector.Web application,16.0
```

**Supported VERIS fields:**
- `action.[type].variety.[specific]` - Attack varieties (e.g., hacking, malware, social)
- `action.[type].vector.[method]` - Attack vectors (e.g., web application, email)
- `action.[type].result.[outcome]` - Attack results (e.g., exfiltrate)

**CIS Controls included in this app:**
- Control 2: Inventory and Control of Software Assets
- Control 3: Data Protection
- Control 4: Secure Configuration
- Control 5: Account Management
- Control 6: Access Control Management
- Control 7: Continuous Vulnerability Management
- Control 9: Email and Web Browser Protections
- Control 10: Malware Defenses
- Control 11: Data Recovery (Backups)
- Control 12: Network Infrastructure Management
- Control 13: Network Monitoring and Defense
- Control 14: Security Awareness and Skills Training
- Control 16: Application Software Security
- Control 18: Penetration Testing

The app automatically aggregates detailed VERIS fields to high-level actions (hacking, malware, social, etc.) 
and patterns (Web Applications, Crimeware, etc.) for easier analysis.

---

## 📚 Additional Resources

### Understanding the Math
- **Compound distribution**: [Wikipedia: Compound Probability Distribution](https://en.wikipedia.org/wiki/Compound_probability_distribution)
- **GPD for tail modeling**: Klugman, Panjer & Willmot, *Loss Models*
- **VERIS Framework**: [VERIS Community Database](http://veriscommunity.net/)

### Cyber Risk Quantification
- FAIR Institute: [Factor Analysis of Information Risk](https://www.fairinstitute.org/)
- NIST Cybersecurity Framework
- ISO 31000 Risk Management

### Need Help?
- Check parameter warnings (yellow/red alerts)
- Review sanity check guide (bottom of page)
- Start with defaults, adjust incrementally
- Export config  to share with team

### 🏛️ Policy Layer (Insurance) — How it works

The policy layer lets you see **insurer-net** losses after applying annual terms. It models **aggregate (per-year)** terms, not per-occurrence deductibles.

**Inputs**
- **Retention / SIR** – Amount the insured pays first each year.
- **Annual Aggregate Limit** – Max the insurer will pay in a year (0 = unlimited).
- **Coinsurance** – Insurer’s share of covered losses above the retention (e.g., 0.90).

**Computation (per simulated year)**
1. Let **L** be the gross annual loss.
2. **Excess over retention:** `Excess = max(L − Retention, 0)`.
3. **Apply coinsurance:** `Covered = Coinsurance × Excess`.
4. **Apply annual limit:** `Insurer Net = min(Covered, Limit)` (if Limit = 0, no cap).
5. **Insured Net** (not displayed by default) would be:
   - If `L ≤ Retention`: insured pays **L**; insurer pays **0**.
   - If `L > Retention`:
     - Insurer pays `Insurer Net` (from step 4).
     - Insured pays the rest: `L − Insurer Net`.

The app reports **insurer-centric** metrics (EAL, VaR, P(Ruin)) on the **net** series and can optionally overlay the **Net LEC** on the chart.

**Notes & Edge Cases**
- This is an **annual aggregate** retention—not per-loss deductibles. Multiple incidents in a year accumulate against the same retention and limit.
- If `Coinsurance = 1.0`, the insurer covers all excess (subject to Limit).
- If `Limit = 0`, coverage is unlimited (subject to coinsurance).
- If `Retention = 0`, coverage begins immediately at the first dollar of loss.
- Very large tails can quickly exhaust the annual limit; check the Net LEC to see how often that happens.

**Quick Example**
- Retention = \$1,000,000; Limit = \$10,000,000; Coinsurance = 0.9  
- Yearly loss **L = \$5,000,000**  
  - Excess = \$4,000,000  
  - Covered = 0.9 × 4,000,000 = **\$3,600,000**  
  - Limit not hit → **Insurer Net = \$3.6M**, Insured pays \$1.4M  
- Yearly loss **L = \$20,000,000**  
  - Excess = \$19,000,000 → Covered = \$17,100,000 → Limit caps at **\$10,000,000**  
  - **Insurer Net = \$10M**, Insured pays \$10M (retention + over-limit remainder)

---

## ⚠️ Limitations & Disclaimers

This is a **demonstration tool** for educational purposes:
- Results depend heavily on input parameters
- Models are simplifications of complex cyber risk
- Not a substitute for professional risk assessment
- Validate outputs against your organization's data
- Use for prioritization and relative comparisons, not absolute predictions

**Always**: Combine quantitative models with qualitative judgment, threat intelligence, and compliance requirements.
    """)
# ===========================
# DATA CLASSES (typed config)
# ===========================
@dataclass
class ModelConfig:
    trials: int = 10000              # Number of simulated years
    net_worth: float = 100e6         # Used for P(Ruin)
    seed: int = 42                   # RNG seed for reproducibility
    record_cap: int = 0              # Max records per incident (0 = no cap)
    cost_per_record: float = 150.0   # $ used in records-based model

@dataclass
class FreqParams:
    lam: float = 2.0                 # Mean incidents/year
    p_any: float = 0.7               # P(incident produces a dollar loss)
    negbin: bool = False             # Use NegBin if True; else Poisson
    r: float = 1.0                   # NegBin dispersion (smaller r ⇒ fatter tails)

@dataclass
class SevParams:
    # Lognormal body
    mu: float = 12.0
    sigma: float = 2.0
    # GPD tail
    gpd_thresh_q: float = 0.95
    gpd_scale: float = 1e6
    gpd_shape: float = 0.3
    # Records-based option
    use_records: bool = False
    records_mu: float = 10.0
    records_sigma: float = 2.0
    cost_per_record: float = 150.0

@dataclass
class ControlSet:
    # Four demo controls (checkboxes)
    server: bool = False
    media: bool = False
    error: bool = False
    external: bool = False

@dataclass
class ControlEffects:
    # Multipliers applied to base params when controls are ON
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

@dataclass
class PolicyTerms:
    retention: float = 0.0    # deductible/self-insured retention per year
    limit: float = 0.0        # annual aggregate limit (0 = unlimited)
    coinsurance: float = 1.0  # insurer pays this fraction above retention (e.g., 0.9)

def apply_policy_terms(annual_losses: np.ndarray, terms: PolicyTerms) -> np.ndarray:
    """
    Apply annual aggregate terms to each simulated year:
    - First pay retention by insured
    - Above retention, insurer pays coinsurance * min(excess, remaining limit)
    - Return the insurer-paid loss series (net to insurer), and you can derive net-to-insured too.
    """
    L = annual_losses.astype(float).copy()
    # Excess over retention
    excess = np.clip(L - terms.retention, 0.0, None)
    # Apply coinsurance
    covered = excess * terms.coinsurance
    # Apply annual limit
    if terms.limit and terms.limit > 0:
        covered = np.minimum(covered, terms.limit)
    return covered
    
# ======================================
# DEFAULTS (approx. VCDB-informed priors)
# ======================================
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

# ==================
# UTILITY FUNCTIONS
# ==================
def _to_dict(x):
    """Dataclass/dict → plain dict for caching/serialization."""
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return x
    return dict(x.__dict__) if hasattr(x, "__dict__") else x

def _ensure_control_effects(ce):
    """Always return a valid ControlEffects object."""
    if ce is None or not hasattr(ce, 'lam_mult'):
        return ControlEffects()
    return ce

def _normalize_shares(shares: dict) -> dict:
    """Normalize weights to sum to 1 (no-op on zero-sum)."""
    total = sum(shares.values())
    return {k: v/total for k, v in shares.items()} if total > 0 else shares

def _stable_seed_from(s: str, base: int = 0):
    """Deterministic small integer derived from a string; add base."""
    return base + (abs(hash(str(s))) % 10000)

# =============================
# VERIS → CIS MAPPING UTILITIES
# =============================
def load_cis_mapping():
    """
    Loads CSV mapping VERIS action/pattern to CIS controls.
    Tries two locations; returns dict with 'loaded' flag and maps.
    """
    paths = ["data/veris_to_cis_lookup.csv", "veris_to_cis_lookup.csv"]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                cols = {c.lower(): c for c in df.columns}
                
                # Identify potential column names (case-insensitive)
                veris_cols = [c for c in ["action", "pattern", "veris", "veris_key"] if c in cols]
                cis_id_col = next((cols[c] for c in ["cis", "cis_control", "cis_id", "cis_v8_id"] if c in cols), None)
                cis_title_col = next((cols[c] for c in ["cis_title", "cis_name", "title", "name"] if c in cols), None)

                amap, pmap = {}, {}
                for _, row in df.iterrows():
                    # Select first non-null VERIS-like field
                    veris_val = None
                    for vc in veris_cols:
                        v = str(row.get(cols[vc])).strip() if pd.notna(row.get(cols[vc])) else None
                        if v:
                            veris_val = v
                            break
                    if not veris_val:
                        continue

                    # Compose readable CIS label
                    cis_id = str(row.get(cis_id_col)).strip() if cis_id_col and pd.notna(row.get(cis_id_col)) else ""
                    cis_title = str(row.get(cis_title_col)).strip() if cis_title_col and pd.notna(row.get(cis_title_col)) else ""
                    cis_display = (f"{cis_id} – {cis_title}".strip(" –")) if cis_id or cis_title else ""

                    # Heuristically bucket into action vs pattern maps
                    key_lower = veris_val.lower()
                    if key_lower in [k.lower() for k in DEFAULT_ACTION_SHARES.keys()]:
                        amap.setdefault(veris_val, set()).add(cis_display or cis_id)
                    elif key_lower in [k.lower() for k in DEFAULT_PATTERN_SHARES.keys()]:
                        pmap.setdefault(veris_val, set()).add(cis_display or cis_id)
                    else:
                        # Unknown key: include in both so it still surfaces
                        amap.setdefault(veris_val, set()).add(cis_display or cis_id)
                        pmap.setdefault(veris_val, set()).add(cis_display or cis_id)

                # Convert sets → sorted lists
                amap = {k: sorted([x for x in v if x]) for k, v in amap.items()}
                pmap = {k: sorted([x for x in v if x]) for k, v in pmap.items()}
                return {"loaded": True, "action": amap, "pattern": pmap, "path": p}
            except Exception as e:
                st.sidebar.warning(f"Failed to read CIS mapping CSV: {e}")
                break
    # Fallback if not found or error
    return {"loaded": False, "action": {}, "pattern": {}, "path": None}

def cis_for_profile(action_shares, pattern_shares, cis_map, action_thresh=0.10, pattern_thresh=0.10, top_n=15):
    """Return list of CIS controls triggered by the current mix at cutoffs."""
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

def cis_for_control(control_name, cis_map):
    """Suggest CIS items per demo control, using a small VERIS keyword map."""
    if not cis_map.get("loaded"):
        return ""
    mapping = {
        "server": ["hacking", "Web Applications"],
        "media": ["physical", "Lost and Stolen Assets"],
        "error": ["error", "Miscellaneous Errors"],
        "external": ["hacking", "misuse", "Privilege Misuse"]
    }
    suggestions = set()
    for key in mapping.get(control_name, []):
        suggestions.update(cis_map["action"].get(key, []))
        suggestions.update(cis_map["pattern"].get(key, []))
    return ", ".join(sorted(list(suggestions))[:3]) if suggestions else "—"

def rank_cis_controls(cis_map, action_shares, pattern_shares, top_n=10):
    """Rank CIS controls by weighted relevance (actions 1.0, patterns 0.8)."""
    if not cis_map or not cis_map.get("loaded"):
        return pd.DataFrame(columns=["CIS Control", "Score", "Why"])

    a = _normalize_shares(action_shares)
    p = _normalize_shares(pattern_shares)

    scores, reasons = {}, {}
    # Score from actions
    for a_key, w in a.items():
        for cis in cis_map["action"].get(a_key, []):
            scores[cis] = scores.get(cis, 0.0) + 1.0 * w
            reasons.setdefault(cis, []).append(f"action:{a_key} ({w:.0%})")
    # Score from patterns
    for p_key, w in p.items():
        for cis in cis_map["pattern"].get(p_key, []):
            scores[cis] = scores.get(cis, 0.0) + 0.8 * w
            reasons.setdefault(cis, []).append(f"pattern:{p_key} ({w:.0%})")

    if not scores:
        return pd.DataFrame(columns=["CIS Control", "Score", "Why"])

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    rows = [{"CIS Control": cis, "Score": score, "Why": "; ".join(reasons.get(cis, []))}
            for cis, score in ranked]
    return pd.DataFrame(rows)

# ======================
# SIMULATION CORE LOGIC
# ======================
def simulate_annual_losses(cfg: ModelConfig, fp: FreqParams, sp: SevParams, 
                           ce: Optional[ControlEffects] = None) -> np.ndarray:
    """Simulate annual loss totals under frequency+severity, optionally with controls."""
    np.random.seed(cfg.seed)  # reproducibility

    # Apply control multipliers (or pass-through 1.0 if no controls)
    lam_eff = fp.lam * (ce.lam_mult if ce else 1.0)
    p_any_eff = fp.p_any * (ce.p_any_mult if ce else 1.0)
    gpd_scale_eff = sp.gpd_scale * (ce.gpd_scale_mult if ce else 1.0)

    # Precompute body threshold for monetary model
    if not sp.use_records:
        body_thresh_val = float(lognorm(s=sp.sigma, scale=np.exp(sp.mu)).ppf(sp.gpd_thresh_q))

    annual_losses = np.zeros(cfg.trials)

    for i in range(cfg.trials):
        # Frequency
        if fp.negbin:
            L = np.random.gamma(shape=fp.r, scale=lam_eff / fp.r)
            n_incidents = np.random.poisson(L)
        else:
            n_incidents = np.random.poisson(lam_eff)
        if n_incidents == 0:
            continue

        # Severity per incident
        for _ in range(n_incidents):
            if np.random.random() > p_any_eff:
                continue

            if sp.use_records:
                # Records-based severity (lognormal records × $/record), with optional cap
                n_records = np.exp(np.random.normal(sp.records_mu, sp.records_sigma))
                if cfg.record_cap > 0:
                    n_records = min(n_records, cfg.record_cap)
                loss = n_records * cfg.cost_per_record  # cfg is the authority
            else:
                # Monetary model: lognormal body + GPD tail on excess
                u = np.random.random()
                if u < sp.gpd_thresh_q:
                    loss = np.exp(np.random.normal(sp.mu, sp.sigma))
                else:
                    u_tail = np.random.random()
                    xi = sp.gpd_shape
                    beta = gpd_scale_eff
                    if xi == 0.0:
                        excess = np.random.exponential(beta)
                    else:
                        excess = beta * (u_tail**(-xi) - 1.0) / xi
                    loss = body_thresh_val + max(0.0, excess)

            annual_losses[i] += loss

    return annual_losses


def compute_metrics(losses: np.ndarray, net_worth: float) -> dict:
    """Compute EAL, VaR95/99, CVaR95, Max, and P(Ruin) from simulated losses."""
    return {
        "EAL": np.mean(losses),
        "VaR95": np.percentile(losses, 95),
        "VaR99": np.percentile(losses, 99),
        "CVaR95": np.mean(losses[losses >= np.percentile(losses, 95)]),
        "Max": np.max(losses),
        "P(Ruin)": np.mean(losses >= net_worth)
    }

def var_confidence_interval(losses: np.ndarray, alpha: float, n_boot: int = 1000) -> dict:
    """Bootstrap VaR(alpha) CI by resampling the annual loss vector."""
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

def eal_ci(losses: np.ndarray, n_boot: int = 1000, alpha: float = 0.95):
    """Bootstrap CI for the mean (EAL)."""
    m = len(losses)
    boots = np.mean(
        np.random.choice(losses, size=(n_boot, m), replace=True),
        axis=1
    )
    lo, hi = np.percentile(boots, [(1-alpha)/2*100, (1+alpha)/2*100])
    return float(np.mean(losses)), float(lo), float(hi)

@st.cache_data(show_spinner=False)
def cached_simulate(cfg_dict, fp_dict, sp_dict, ce_dict=None):
    """Rebuild dataclasses from dicts and run simulate_annual_losses (memoized)."""
    cfg = ModelConfig(**cfg_dict)
    fp = FreqParams(**fp_dict)
    sp = SevParams(**sp_dict)
    ce = ControlEffects(**ce_dict) if ce_dict else None
    return simulate_annual_losses(cfg, fp, sp, ce)

def lec(losses: np.ndarray, n: int = 200) -> pd.DataFrame:
    """Compute loss exceedance curve points on a log-spaced grid."""
    losses = np.asarray(losses, float)
    losses = losses[np.isfinite(losses)]
    
    # Edge-cases: empty or all zeros
    if losses.size == 0 or np.all(losses == 0):
        return pd.DataFrame({"Loss": [1.0], "Exceedance_Prob": [0.0]})
    
    losses.sort()
    
    # Grid from small positive to max loss
    lo = max(1.0, np.percentile(losses[losses > 0], 1) if np.any(losses > 0) else 1.0)
    hi = float(losses[-1])
    if hi <= lo:
        return pd.DataFrame({"Loss": [1.0], "Exceedance_Prob": [0.0]})
    
    grid = np.logspace(np.log10(lo), np.log10(hi), n)
    # For each x, find how many losses ≥ x (via searchsorted index)
    idx = np.searchsorted(losses, grid, side='left')
    ex_prob = 1.0 - idx / float(len(losses))
    ex_prob = np.clip(ex_prob, 1.0/len(losses), 1.0)  # keep strictly positive
    
    return pd.DataFrame({"Loss": grid, "Exceedance_Prob": ex_prob})

@st.cache_data(show_spinner=False)
def cached_lec(losses, n):
    """Memoized LEC wrapper."""
    return lec(losses, n=n)

def effects_from_shares_improved(ctrl: ControlSet, action_shares: dict, pattern_shares: dict) -> ControlEffects:
    """Heuristic mapping: VERIS share mix → control multipliers (λ, p(any), tail-scale)."""
    a = _normalize_shares(action_shares)
    p = _normalize_shares(pattern_shares)

    # Initialize multipliers at 1.0 (no change)
    lam_mult = 1.0
    p_any_mult = 1.0
    gpd_mult = 1.0

    # Simple channel intensities from action+pattern weights
    hack_intensity = a.get("hacking", 0) + p.get("Web Applications", 0) + p.get("Crimeware", 0)
    misuse_intensity = a.get("misuse", 0) + p.get("Privilege Misuse", 0)
    error_intensity = a.get("error", 0) + p.get("Miscellaneous Errors", 0)
    physical_int = a.get("physical", 0) + p.get("Lost and Stolen Assets", 0)

    # Apply reductions only for toggled controls
    if ctrl.server:
        lam_mult *= (1 - 0.35 * hack_intensity)
        p_any_mult *= (1 - 0.20 * hack_intensity)
        gpd_mult *= (1 - 0.15 * hack_intensity)
    if ctrl.media:
        lam_mult *= (1 - 0.25 * physical_int)
        p_any_mult *= (1 - 0.25 * physical_int)
    if ctrl.error:
        lam_mult *= (1 - 0.20 * error_intensity)
        p_any_mult *= (1 - 0.25 * error_intensity)
    if ctrl.external:
        lam_mult *= (1 - 0.30 * (hack_intensity + misuse_intensity))
        gpd_mult *= (1 - 0.20 * (hack_intensity + misuse_intensity))

    # Clip to safe demo bounds
    lam_mult = float(np.clip(lam_mult, 0.2, 1.0))
    p_any_mult = float(np.clip(p_any_mult, 0.2, 1.0))
    gpd_mult = float(np.clip(gpd_mult, 0.2, 1.0))

    return ControlEffects(lam_mult=lam_mult, p_any_mult=p_any_mult, gpd_scale_mult=gpd_mult)

def log_hist_figure(losses, title):
    """Plotly histogram on log10-transformed losses with pretty x ticks."""
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

# =========================
# SIDEBAR: USER PARAMETERS
# =========================
st.sidebar.header("⚙️ Model Parameters")

# Load parameter JSON (optional) — can override action/pattern shares
with st.sidebar.expander("📁 Load parameter JSON", expanded=False):
    pj = st.file_uploader("Upload parameters.json", type=["json"], key="params_json")
    if pj:
        params = json.load(pj)
        cfg_block  = params.get("model")
fp_block   = params.get("frequency")
sp_block   = params.get("severity")
ctrl_block = params.get("controls")
costs_blk  = params.get("costs")

if cfg_block:  st.session_state["_cfg_loaded"]   = cfg_block
if fp_block:   st.session_state["_fp_loaded"]    = fp_block
if sp_block:   st.session_state["_sp_loaded"]    = sp_block
if ctrl_block: st.session_state["_ctrl_loaded"]  = ctrl_block
if costs_blk:  st.session_state["_costs_loaded"] = costs_blk

st.success("✓ Full scenario loaded"); st.rerun()


# CIS mapping status
with st.sidebar.expander("📚 CIS Mapping (VERIS → CIS)", expanded=False):
    cis_map = load_cis_mapping()
    if cis_map["loaded"]:
        st.success(f"✓ CIS mapping loaded from {cis_map['path']}")
        st.caption("CIS recommendations will appear with actions/patterns and in ROI tables.")
    else:
        st.info("Add **data/veris_to_cis_lookup.csv** (or root **veris_to_cis_lookup.csv**) to enable CIS recommendations.")

# Use session-loaded shares or defaults
action_shares = st.session_state.get("_action_shares", DEFAULT_ACTION_SHARES)
pattern_shares = st.session_state.get("_pattern_shares", DEFAULT_PATTERN_SHARES)

# Preview current shares
st.sidebar.caption("Action shares in use")
st.sidebar.dataframe(
    pd.DataFrame({"action": list(action_shares.keys()), "share": list(action_shares.values())}).style.format({"share": "{:.1%}"}),
    hide_index=True
)
st.sidebar.caption("Pattern shares in use")
st.sidebar.dataframe(
    pd.DataFrame({"pattern": list(pattern_shares.keys()), "share": list(pattern_shares.values())}).style.format({"share": "{:.1%}"}),
    hide_index=True
)

# Reset shares to built-ins
if st.sidebar.button("🔄 Reset action/pattern shares to defaults"):
    st.session_state.pop("_action_shares", None)
    st.session_state.pop("_pattern_shares", None)
    st.rerun()

# --- Simulation Config inputs ---
with st.sidebar.expander("🎲 Simulation Config", expanded=True):
    trials = st.number_input("Monte Carlo Trials", 1000, 100000, 10000, 1000)
    net_worth = st.number_input("Net Worth ($M)", 1.0, 10000.0, 100.0, 10.0) * 1e6
    seed = st.number_input("Random Seed", 0, 9999, 42)
    # Records parameters used by records-based severity (even if not chosen)
    st.markdown("**Records Parameters**")
    cost_per_record_input = st.number_input("Cost per record ($)", 1.0, 10000.0, 150.0, 10.0,
                                            help="Cost per exposed/lost record (used in records-based mode)")
    record_cap = st.number_input("Record cap (0 = unlimited)", 0, 1000000000, 0, 1000000,
                                 help="Maximum records per incident (0 for no cap)")

# Build ModelConfig from UI inputs
cfg = ModelConfig(trials=trials, net_worth=net_worth, seed=seed,
                  record_cap=record_cap, cost_per_record=cost_per_record_input)

# --- Frequency inputs ---
with st.sidebar.expander("📊 Frequency Parameters", expanded=True):
    lam = st.number_input("λ (mean incidents/year)", 0.1, 20.0, 2.0, 0.1)
    p_any = st.slider("P(any loss | incident)", 0.1, 0.95, 0.7, 0.05)
    negbin = st.checkbox("Use Negative Binomial", value=False)
    r = st.number_input("NegBin dispersion (r)", 0.5, 10.0, 1.0, 0.5) if negbin else 1.0
    if negbin:
        st.caption("ℹ️ Lower r ⇒ more over-dispersion (fatter frequency tails).")
    if p_any < 0.1 or p_any > 0.95:
        st.warning(f"⚠️ p(any loss)={p_any:.2f} is extreme; results may be unstable.")

# Build and clamp frequency params
fp = FreqParams(lam=lam, p_any=p_any, negbin=negbin, r=r)
fp.p_any = float(np.clip(fp.p_any, 0.0, 1.0))
if fp.negbin:
    fp.r = float(max(1e-6, fp.r))

# --- Optional Bayesian update of λ ---
with st.sidebar.expander("🔬 Bayesian Frequency Update", expanded=False):
    use_bayes = st.checkbox("Enable Bayesian update")
    if use_bayes:
        T_obs = st.number_input("Observation period (years)", 1, 20, 5)
        k_obs = st.number_input("Observed incidents", 0, 100, 10)
        if T_obs and k_obs:
            lam_hat = float(k_obs) / float(T_obs)
            # Noticeable mismatch nudge
            if (lam_hat > 0) and (max(lam_hat, fp.lam) / max(1e-9, min(lam_hat, fp.lam)) >= 3):
                st.info(f"ℹ️ λ differs from k/T by ≥3× (λ={fp.lam:.3f}, k/T={lam_hat:.3f}). Consider Bayes mode or align values.")
            # Conjugate prior update: Gamma(α, β) prior with α=2 fixed; β so that prior mean≈lam
            alpha_prior = 2.0
            beta_prior = alpha_prior / lam
            alpha_post = alpha_prior + k_obs
            beta_post = beta_prior + T_obs
            lam_post = alpha_post / beta_post
            st.metric("Updated λ (posterior mean)", f"{lam_post:.2f}")
            fp.lam = lam_post

# --- Severity inputs (two modes) ---
with st.sidebar.expander("💰 Severity Parameters", expanded=True):
    use_records = st.checkbox("Use records-based loss model", value=False,
                              help="Toggle between monetary severity (GPD/lognormal) and records × $/record")
    if use_records:
        # Records-based severity paramization
        st.markdown("**Records Model**")
        records_mu = st.number_input("Records lognormal μ", 6.0, 20.0, 10.0, 0.5,
                                     help="Mean log(records) — e.g., μ=10 → median ~22k records")
        records_sigma = st.number_input("Records lognormal σ", 0.5, 4.0, 2.0, 0.1,
                                        help="Std dev of log(records)")
        sp = SevParams(use_records=True, records_mu=records_mu, records_sigma=records_sigma,
                       cost_per_record=cfg.cost_per_record,
                       mu=12.0, sigma=2.0, gpd_thresh_q=0.95, gpd_scale=1e6, gpd_shape=0.3)
        # Quick derived stats for intuition
        median_records = int(np.exp(records_mu))
        mean_records = int(np.exp(records_mu + records_sigma**2 / 2))
        st.caption(f"📊 Expected records: median={median_records:,}, mean={mean_records:,}")
        st.caption(f"💵 Implied median loss: ${median_records * cfg.cost_per_record:,.0f}")
        st.caption(f"💵 Implied mean loss: ${mean_records * cfg.cost_per_record:,.0f}")
    else:
        # Monetary severity paramization (lognormal body + GPD tail)
        st.markdown("**Monetary Model (GPD/Lognormal)**")
        mu = st.number_input("Lognormal μ", 8.0, 16.0, 12.0, 0.5)
        sigma = st.number_input("Lognormal σ", 0.5, 4.0, 2.0, 0.1)
        gpd_thresh_q = st.slider("GPD threshold quantile", 0.85, 0.99, 0.95, 0.01)
        gpd_scale = st.number_input("GPD scale ($K)", 100.0, 10000.0, 1000.0, 100.0) * 1000
        gpd_shape = st.number_input("GPD shape (ξ)", 0.0, 1.0, 0.3, 0.05)
        sp = SevParams(use_records=False, mu=mu, sigma=sigma, 
                       gpd_thresh_q=gpd_thresh_q, gpd_scale=gpd_scale, gpd_shape=gpd_shape,
                       records_mu=10.0, records_sigma=2.0, cost_per_record=cfg.cost_per_record)
        sp.gpd_scale = float(max(1.0, sp.gpd_scale))  # avoid zero/negative scale
        if sp.gpd_shape >= 1.0:
            st.sidebar.warning("⚠️ ξ (GPD shape) ≥ 1 yields infinite mean tail. Results may be unstable.")
        if sp.sigma > 2.5:
            st.sidebar.warning("⚠️ Lognormal σ is quite high; body may dominate and inflate EAL/VaR.")

# Bootstrap controls (for CIs)
with st.sidebar.expander("📐 Confidence Intervals (advanced)", expanded=False):
    n_boot = st.slider("Bootstrap reps for VaR/EAL CI", 100, 5000, 1000, 100)
    if cfg.trials < 2000 and n_boot > 1000:
        st.caption("⚠️ Consider reducing reps or increasing trials for stable CIs.")

# Control selections
with st.sidebar.expander("🛡️ Control Selection", expanded=True):
    server = st.checkbox("Server hardening", value=False)
    media = st.checkbox("Media encryption", value=False)
    error_ctrl = st.checkbox("Error reduction", value=False)
    external = st.checkbox("External monitoring", value=False)
ctrl = ControlSet(server=server, media=media, error=error_ctrl, external=external)

# Control costs ($K inputs → $)
with st.sidebar.expander("💵 Control Costs", expanded=True):
    cost_server = st.number_input("Server hardening ($K)", 0.0, 1000.0, 50.0, 10.0) * 1000
    cost_media = st.number_input("Media encryption ($K)", 0.0, 1000.0, 30.0, 10.0) * 1000
    cost_error = st.number_input("Error reduction ($K)", 0.0, 1000.0, 20.0, 10.0) * 1000
    cost_external = st.number_input("External monitoring ($K)", 0.0, 1000.0, 40.0, 10.0) * 1000
costs = ControlCosts(server=cost_server, media=cost_media, error=cost_error, external=cost_external)

# Clamp negative costs to zero (and warn)
for k in ["server", "media", "error", "external"]:
    v = getattr(costs, k)
    if v < 0:
        setattr(costs, k, 0.0)
        st.sidebar.warning(f"⚠️ {k.title()} cost < 0 corrected to 0.")
with st.sidebar.expander("🏛️ Policy Layer (annual terms)", expanded=False):
    retention = st.number_input("Retention / SIR ($)", 0.0, 1e9, 1_000_000.0, 100_000.0)
    limit = st.number_input("Annual Aggregate Limit ($; 0 = unlimited)", 0.0, 1e10, 10_000_000.0, 1_000_000.0)
    coins = st.slider("Coinsurance (insurer share above retention)", 0.0, 1.0, 0.9, 0.05)
    terms = PolicyTerms(retention=retention, limit=limit, coinsurance=coins)

# ======================
# MAIN SIMULATION BLOCK
# ======================
# Compute control effects safely (fallback → neutral effects)
try:
    ce = effects_from_shares_improved(ctrl, action_shares, pattern_shares)
except Exception:
    ce = ControlEffects()
ce = _ensure_control_effects(ce)

st.header("🎯 Simulation Results")

# Summary of assumptions (frequency, controls, severity, config)
with st.expander("📋 Assumption Summary", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Frequency Parameters**")
        st.markdown(f"- λ (mean incidents/year): `{fp.lam:.3f}`")
        st.markdown(f"- P(any loss | incident): `{fp.p_any:.3f}`")
        st.markdown(f"- Distribution: `{'Negative Binomial' if fp.negbin else 'Poisson'}`")
        if fp.negbin:
            st.markdown(f"- NegBin dispersion (r): `{fp.r:.3f}`")
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

# Show CIS suggestions for the overall mix
if cis_map.get("loaded"):
    st.subheader("🔗 CIS Recommendations for Current Mix")
    cis_list = cis_for_profile(action_shares, pattern_shares, cis_map, action_thresh=0.10, pattern_thresh=0.10, top_n=20)
    if cis_list:
        st.markdown("- " + "\n- ".join(cis_list))
    else:
        st.caption("No CIS recommendations triggered at current thresholds.")

# Run baseline vs controlled simulations (cached)
base_losses = cached_simulate(_to_dict(cfg), _to_dict(fp), _to_dict(sp))
ctrl_losses = cached_simulate(_to_dict(cfg), _to_dict(fp), _to_dict(sp), _to_dict(ce))

# Compute metrics for both scenarios
base_metrics = compute_metrics(base_losses, cfg.net_worth)
ctrl_metrics = compute_metrics(ctrl_losses, cfg.net_worth)

# Quick recap of multipliers applied
st.caption(f"📊 Applied control multipliers → λ×{ce.lam_mult:.2f}, P(any)×{ce.p_any_mult:.2f}, tail-scale×{ce.gpd_scale_mult:.2f}")

# VaR/EAL confidence intervals via bootstrap
var95_base = var_confidence_interval(base_losses, 0.95, n_boot=n_boot)
var95_ctrl = var_confidence_interval(ctrl_losses, 0.95, n_boot=n_boot)
st.caption(f"📈 VaR95: Base ${var95_base['point']:,.0f} (±{(var95_base['ci_upper']-var95_base['ci_lower'])/2:,.0f}) | "
           f"Ctrl ${var95_ctrl['point']:,.0f} (±{(var95_ctrl['ci_upper']-var95_ctrl['ci_lower'])/2:,.0f})")

eal_b, eal_lo_b, eal_hi_b = eal_ci(base_losses, n_boot=n_boot)
eal_c, eal_lo_c, eal_hi_c = eal_ci(ctrl_losses, n_boot=n_boot)
st.caption(f"📊 EAL CI: Base ${eal_b:,.0f} [{eal_lo_b:,.0f}, {eal_hi_b:,.0f}] | "
           f"Ctrl ${eal_c:,.0f} [{eal_lo_c:,.0f}, {eal_hi_c:,.0f}]")
# Insurer-net annual losses after applying terms
base_net = apply_policy_terms(base_losses, terms)
ctrl_net = apply_policy_terms(ctrl_losses, terms)

# Insurer-centric metrics
base_metrics_ins = compute_metrics(base_net, cfg.net_worth)   # net of terms
ctrl_metrics_ins = compute_metrics(ctrl_net, cfg.net_worth)

st.caption("📜 Policy Layer active: metrics below include annual retention/limit/coinsurance (insurer net).")
# --- Insured Net metrics + CSV export (NEW) ---

# Insured pays whatever the insurer doesn't (gross − insurer_net); never negative.
base_insured = np.maximum(base_losses - base_net, 0.0)
ctrl_insured = np.maximum(ctrl_losses - ctrl_net, 0.0)

base_metrics_insured = compute_metrics(base_insured, cfg.net_worth)
ctrl_metrics_insured = compute_metrics(ctrl_insured, cfg.net_worth)

# Toggle to show side-by-side Net metrics
show_insured = st.checkbox("Show Insured Net metrics (side-by-side)", value=True)

if show_insured:
    st.subheader("🧾 Net Metrics — Insurer vs Insured (after annual terms)")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Insurer Net**")
        ins_df = pd.DataFrame({
            "Metric": ["EAL", "VaR95", "VaR99", "CVaR95", "Max Loss", "P(Ruin)"],
            "Baseline": [
                base_metrics_ins["EAL"], base_metrics_ins["VaR95"], base_metrics_ins["VaR99"],
                base_metrics_ins["CVaR95"], base_metrics_ins["Max"], base_metrics_ins["P(Ruin)"]
            ],
            "Controlled": [
                ctrl_metrics_ins["EAL"], ctrl_metrics_ins["VaR95"], ctrl_metrics_ins["VaR99"],
                ctrl_metrics_ins["CVaR95"], ctrl_metrics_ins["Max"], ctrl_metrics_ins["P(Ruin)"]
            ],
        })
        st.dataframe(
            ins_df.style.format({"Baseline": "${:,.2f}", "Controlled": "${:,.2f}"}).hide(axis="index"),
            use_container_width=True
        )

    with c2:
        st.markdown("**Insured Net**")
        insured_df = pd.DataFrame({
            "Metric": ["EAL", "VaR95", "VaR99", "CVaR95", "Max Loss", "P(Ruin)"],
            "Baseline": [
                base_metrics_insured["EAL"], base_metrics_insured["VaR95"], base_metrics_insured["VaR99"],
                base_metrics_insured["CVaR95"], base_metrics_insured["Max"], base_metrics_insured["P(Ruin)"]
            ],
            "Controlled": [
                ctrl_metrics_insured["EAL"], ctrl_metrics_insured["VaR95"], ctrl_metrics_insured["VaR99"],
                ctrl_metrics_insured["CVaR95"], ctrl_metrics_insured["Max"], ctrl_metrics_insured["P(Ruin)"]
            ],
        })
        st.dataframe(
            insured_df.style.format({"Baseline": "${:,.2f}", "Controlled": "${:,.2f}"}).hide(axis="index"),
            use_container_width=True
        )

# Build one CSV covering Gross, Insurer Net, and Insured Net (Baseline & Controlled)
def _metrics_rows(view_name: str, base_m: dict, ctrl_m: dict):
    return [
        {"View": view_name, "Scenario": "Baseline",  "Metric": "EAL",     "Value": base_m["EAL"]},
        {"View": view_name, "Scenario": "Baseline",  "Metric": "VaR95",   "Value": base_m["VaR95"]},
        {"View": view_name, "Scenario": "Baseline",  "Metric": "VaR99",   "Value": base_m["VaR99"]},
        {"View": view_name, "Scenario": "Baseline",  "Metric": "CVaR95",  "Value": base_m["CVaR95"]},
        {"View": view_name, "Scenario": "Baseline",  "Metric": "Max",     "Value": base_m["Max"]},
        {"View": view_name, "Scenario": "Baseline",  "Metric": "P(Ruin)", "Value": base_m["P(Ruin)"]},
        {"View": view_name, "Scenario": "Controlled","Metric": "EAL",     "Value": ctrl_m["EAL"]},
        {"View": view_name, "Scenario": "Controlled","Metric": "VaR95",   "Value": ctrl_m["VaR95"]},
        {"View": view_name, "Scenario": "Controlled","Metric": "VaR99",   "Value": ctrl_m["VaR99"]},
        {"View": view_name, "Scenario": "Controlled","Metric": "CVaR95",  "Value": ctrl_m["CVaR95"]},
        {"View": view_name, "Scenario": "Controlled","Metric": "Max",     "Value": ctrl_m["Max"]},
        {"View": view_name, "Scenario": "Controlled","Metric": "P(Ruin)", "Value": ctrl_m["P(Ruin)"]},
    ]

gross_rows   = _metrics_rows("Gross",        base_metrics,        ctrl_metrics)
ins_rows     = _metrics_rows("Insurer Net",  base_metrics_ins,    ctrl_metrics_ins)
insured_rows = _metrics_rows("Insured Net",  base_metrics_insured,ctrl_metrics_insured)

net_export_df = pd.DataFrame(gross_rows + ins_rows + insured_rows)
# Pretty formatting for display if you want a quick peek:
st.dataframe(
    net_export_df.copy().assign(
        Value_fmt=lambda d: np.where(d["Metric"]=="P(Ruin)", d["Value"].map(lambda x: f"{x:.4%}"),
                                     d["Value"].map(lambda x: f"${x:,.2f}"))
    )[["View","Scenario","Metric","Value_fmt"]],
    use_container_width=True
)

net_csv = net_export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Gross/Insurer/Insured Metrics (CSV)",
    net_csv,
    file_name="net_metrics_all_views.csv",
    mime="text/csv"
)

# Metric cards (baseline vs controlled and ROI)
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
    delta_eal = base_metrics['EAL'] - ctrl_metrics['EAL']   # risk reduction (positive = good)
    net_benefit = delta_eal - control_cost                  # ROSI numerator
    rosi = (net_benefit / control_cost * 100) if control_cost > 0 else 0
    st.metric("Control Cost", f"${control_cost:,.0f}")
    st.metric("ΔEAL (Risk ↓)", f"${delta_eal:,.0f}")
    st.metric("ROSI", f"{rosi:.1f}%", delta=f"${net_benefit:,.0f} net benefit")

# Table of detailed metrics
summary_data = {
    "Metric": ["EAL", "VaR95", "VaR99", "CVaR95", "Max Loss", "P(Ruin)"],
    "Baseline": [base_metrics['EAL'], base_metrics['VaR95'], base_metrics['VaR99'],
                 base_metrics['CVaR95'], base_metrics['Max'], base_metrics['P(Ruin)']],
    "Controlled": [ctrl_metrics['EAL'], ctrl_metrics['VaR95'], ctrl_metrics['VaR99'],
                   ctrl_metrics['CVaR95'], ctrl_metrics['Max'], ctrl_metrics['P(Ruin)']]
}
summary_df = pd.DataFrame(summary_data)
format_map = {"Baseline": "${:,.2f}", "Controlled": "${:,.2f}"}
st.dataframe(summary_df.style.format(format_map, na_rep="—"), use_container_width=True)
st.subheader("📑 Insurer Net Metrics (after terms)")
ins_cols = st.columns(3)
with ins_cols[0]:
    st.metric("EAL (Net)", f"${base_metrics_ins['EAL']:,.0f} → ${ctrl_metrics_ins['EAL']:,.0f}",
              delta=f"-${base_metrics_ins['EAL'] - ctrl_metrics_ins['EAL']:,.0f}")
with ins_cols[1]:
    st.metric("VaR95 (Net)", f"${base_metrics_ins['VaR95']:,.0f} → ${ctrl_metrics_ins['VaR95']:,.0f}",
              delta=f"-${base_metrics_ins['VaR95'] - ctrl_metrics_ins['VaR95']:,.0f}")
with ins_cols[2]:
    st.metric("P(Ruin) (Net)", f"{base_metrics_ins['P(Ruin)']:.2%} → {ctrl_metrics_ins['P(Ruin)']:.2%}")

# ============================
# CONTROL ISOLATION (one-by-one)
# ============================
st.header("🔬 Control Isolation Analysis")
st.caption("Individual control effectiveness (all others off)")

baseline_eal = base_metrics['EAL']
_iso_seed_bumps = {"server": 101, "media": 202, "error": 303, "external": 404}  # stable per-control RNG

iso = []
for name in ["server", "media", "error", "external"]:
    # Only this control ON
    ctrl_iso = ControlSet(**{k: (k == name) for k in ["server", "media", "error", "external"]})
    ce_iso = effects_from_shares_improved(ctrl_iso, action_shares, pattern_shares)
    # Bump seed so cache keeps each isolation run distinct but stable
    cfg_iso = ModelConfig(trials=cfg.trials, net_worth=cfg.net_worth, seed=cfg.seed + _iso_seed_bumps[name],
                          record_cap=cfg.record_cap, cost_per_record=cfg.cost_per_record)
    losses_iso = cached_simulate(_to_dict(cfg_iso), _to_dict(fp), _to_dict(sp), _to_dict(ce_iso))
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

iso_df = pd.DataFrame(iso).sort_values("Benefit per $", ascending=False)
st.dataframe(iso_df.style.format({
    "ΔEAL ($/yr)": "${:,.0f}",
    "Cost ($/yr)": "${:,.0f}",
    "Benefit per $": "{:,.2f}",
    "ROSI %": "{:.1f}%"
}), use_container_width=True)

# Download isolation CSV
iso_csv = iso_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Isolation ROI (CSV)", iso_csv, "isolation_roi.csv", "text/csv")

# Highlight best ROSI if available (cost>0)
if iso_df['ROSI %'].notna().any():
    best_idx = iso_df['ROSI %'].idxmax()
    best = iso_df.loc[best_idx]
    st.success(f"🏆 Best ROSI: {best['Control']} ({best['ROSI %']:.1f}%)")
else:
    st.info("Set non-zero control costs to compute ROSI (currently all costs are $0).")

# ============================
# MARGINAL ROI FROM CURRENT BUNDLE
# ============================
st.header("🧩 Marginal ROI (from current bundle)")
st.caption("Cost-effectiveness of adding one more control to your selected bundle")

current_losses = ctrl_losses                 # baseline for marginal adds is current bundle
current_eal = ctrl_metrics['EAL']

marg = []
for name in ["server", "media", "error", "external"]:
    if getattr(ctrl, name):  # skip already-selected controls
        continue
    # Add this one control to current bundle
    ctrl_plus = ControlSet(**{k: (getattr(ctrl, k) or (k == name)) for k in ["server", "media", "error", "external"]})
    ce_plus = effects_from_shares_improved(ctrl_plus, action_shares, pattern_shares)
    cfg_plus = ModelConfig(trials=cfg.trials, net_worth=cfg.net_worth, seed=cfg.seed + 777,
                           record_cap=cfg.record_cap, cost_per_record=cfg.cost_per_record)
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
    st.dataframe(marg_df.style.format({
        "ΔEAL from bundle ($/yr)": "${:,.0f}",
        "Incremental Cost ($/yr)": "${:,.0f}",
        "Marginal ROSI %": "{:.1f}%"
    }), use_container_width=True)
    marg_csv = marg_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Marginal ROI (CSV)", marg_csv, "marginal_roi.csv", "text/csv")
else:
    st.info("All controls are already selected; no marginal adds to evaluate.")

# ============================
# CIS RANKED RECOMMENDATIONS
# ============================
st.subheader("🧭 CIS Control Recommendations (ranked by relevance)")
cis_rank_df = rank_cis_controls(cis_map, action_shares, pattern_shares, top_n=10)
if cis_rank_df.empty:
    st.info("Add data/veris_to_cis_lookup.csv to enable CIS recommendations.")
else:
    st.dataframe(cis_rank_df.style.format({"Score": "{:.2f}"}), use_container_width=True)

# ==============
# VISUALIZATIONS
# ==============
st.header("📊 Loss Distributions")

# LEC (Baseline vs Controlled)
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

# Log-histograms of annual loss (two columns)
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(log_hist_figure(base_losses, "Baseline Loss Distribution"), use_container_width=True)
with col2:
    st.plotly_chart(log_hist_figure(ctrl_losses, "Controlled Loss Distribution"), use_container_width=True)
show_net_lec = st.checkbox("Overlay Policy-Layer (Net) on LEC", value=False)
if show_net_lec:
    lec_bn = cached_lec(base_net, lec_points).assign(scenario="Baseline (Net)")
    lec_cn = cached_lec(ctrl_net, lec_points).assign(scenario="Controlled (Net)")
    lec_combined = pd.concat([lec_combined, lec_bn, lec_cn])
    fig_lec = px.line(lec_combined, x="Loss", y="Exceedance_Prob", color="scenario",
                      title="Loss Exceedance Curve (Gross & Net)",
                      labels={"Loss": "Loss Amount ($)", "Exceedance_Prob": "P(Loss ≥ x)"})
    fig_lec.update_xaxes(type="log")
    fig_lec.update_yaxes(type="log", range=[-2.5, 0])
    fig_lec.add_hline(y=0.01, line_dash="dot", opacity=0.2)
    fig_lec.add_hline(y=0.001, line_dash="dot", opacity=0.2)
    st.plotly_chart(fig_lec, use_container_width=True)

# ============================
# PORTFOLIO BATCH ANALYSIS (UNIFIED)
# ============================
with st.expander("📁 Portfolio batch (CSV)", expanded=False):
    st.markdown("Upload a CSV with columns: `account_id`, `net_worth`, `lam`, `p_any`, etc.")
    up = st.file_uploader("Accounts CSV", type=["csv"], key="accounts_csv")  # <-- unique key
    rho = st.slider("Correlation (common shock on frequency)", 0.0, 0.9, 0.0, 0.1,
                    help="0 = independent; higher = more shared shock across accounts",
                    key="portfolio_rho")

    if up is not None and st.button("Run Portfolio Analysis", key="run_portfolio"):
        df = pd.read_csv(up)
        st.write(f"Loaded {len(df)} accounts")

        results = []
        progress_bar = st.progress(0)

        # Pre-draw common factor across simulation years (if rho>0)
        rng = np.random.default_rng(cfg.seed)
        common = (rng.lognormal(mean=0.0, sigma=rho, size=cfg.trials) if rho > 0
                  else np.ones(cfg.trials))

        def simulate_with_common(cfg_account: ModelConfig, fp_account: FreqParams, sp: SevParams):
            """Poisson/Gamma–Poisson frequency with multiplicative common factor."""
            np.random.seed(cfg_account.seed)
            annual = np.zeros(cfg_account.trials)
            for t in range(cfg_account.trials):
                lam_t = fp_account.lam * common[t]
                if fp_account.negbin:
                    L = np.random.gamma(shape=fp_account.r, scale=lam_t / fp_account.r)
                    n = np.random.poisson(L)
                else:
                    n = np.random.poisson(lam_t)

                if n == 0:
                    continue
                for _ in range(n):
                    if np.random.random() > fp_account.p_any:
                        continue
                    if sp.use_records:
                        nrec = np.exp(np.random.normal(sp.records_mu, sp.records_sigma))
                        if cfg_account.record_cap > 0:
                            nrec = min(nrec, cfg_account.record_cap)
                        loss = nrec * cfg_account.cost_per_record
                    else:
                        u = np.random.random()
                        if u < sp.gpd_thresh_q:
                            loss = np.exp(np.random.normal(sp.mu, sp.sigma))
                        else:
                            u_tail = np.random.random()
                            xi, beta = sp.gpd_shape, sp.gpd_scale
                            excess = (beta * (u_tail**(-xi) - 1.0) / xi) if xi != 0 else np.random.exponential(beta)
                            thresh = float(lognorm(s=sp.sigma, scale=np.exp(sp.mu)).ppf(sp.gpd_thresh_q))
                            loss = thresh + max(0.0, excess)
                    annual[t] += loss
            return annual

        for idx, row in df.iterrows():
            account_id = row.get('account_id', f'Account_{idx}')
            account_net_worth = pd.to_numeric(row.get('net_worth', 100e6), errors='coerce')
            account_lam = pd.to_numeric(row.get('lam', 2.0), errors='coerce')
            account_p_any = pd.to_numeric(row.get('p_any', 0.7), errors='coerce')

            account_net_worth = float(account_net_worth if np.isfinite(account_net_worth) else 100e6)
            account_lam = float(account_lam if np.isfinite(account_lam) else 2.0)
            account_p_any = float(np.clip(account_p_any if np.isfinite(account_p_any) else 0.7, 0.0, 1.0))

            cfg_account = ModelConfig(
                trials=cfg.trials,
                net_worth=account_net_worth,
                seed=_stable_seed_from(account_id, base=cfg.seed),
                record_cap=cfg.record_cap,
                cost_per_record=cfg.cost_per_record
            )
            fp_account = FreqParams(lam=account_lam, p_any=account_p_any, negbin=fp.negbin, r=fp.r)

            if rho > 0:
                losses_account = simulate_with_common(cfg_account, fp_account, sp)
            else:
                losses_account = cached_simulate(_to_dict(cfg_account), _to_dict(fp_account), _to_dict(sp))

            metrics_account = compute_metrics(losses_account, account_net_worth)
            results.append({
                'account_id': account_id,
                'EAL': metrics_account['EAL'],
                'VaR95': metrics_account['VaR95'],
                'VaR99': metrics_account['VaR99'],
                'P(Ruin)': metrics_account['P(Ruin)']
            })
            progress_bar.progress((idx + 1) / max(1, len(df)))

        results_df = pd.DataFrame(results)
        st.success("✓ Portfolio analysis complete!")
        st.dataframe(results_df, use_container_width=True)
        st.download_button(
            label="📥 Download Results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="portfolio_results.csv",
            mime="text/csv"
        )
# ============================
# SANITY CHECK GUIDE (expander)
# ============================
with st.expander("🧪 Sanity check guide (what to expect)", expanded=False):
    st.markdown("""
- **Frequency sanity:** If λ≈2 and p(any)≈0.7, expect ~1–2 paid-loss incidents per year.
- **Scale sanity:** With lognormal μ=12, σ=2, median single-incident loss ≈ \$160k; tails rise fast.
- **Tail sanity:** Increasing GPD shape ξ or scale β should push **LEC** right and lift **VaR99**.
- **Controls sanity:** Turning on controls should lower EAL/VaR; ΔEAL should be in the same ballpark as your multipliers suggest.
- **Policy layer sanity:** Higher retention reduces **Insurer Net EAL** but raises **Insured Net EAL**; a tight annual limit clips the **Insurer Net Max**.
- **Bootstrap sanity:** Wider CIs when trials are low or σ/ξ are high; narrow when trials increase.
If any of these don’t hold, re-check parameters for extreme values or typos.
    """)

# ============================
# EXPORT CURRENT CONFIG (JSON)
# ============================
st.sidebar.markdown("---")
export_config = {
    "schema_version": "1.0.1",
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
    label="💾 Download config.json",
    data=json.dumps(export_config, indent=2),
    file_name="cyber_loss_config.json",
    mime="application/json"
)
# Handy maintenance control
if st.sidebar.button("🧹 Clear cached results"):
    st.cache_data.clear()
    st.rerun()


