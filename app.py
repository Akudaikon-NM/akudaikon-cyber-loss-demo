# app.py — Akudaikon | Cyber-Loss Demo (complete, corrected)
# ------------------------------------------------
# Requires: engine.py, controls.py
# Optional: ai_monetary.py (for AI incidents real pipeline)
# ------------------------------------------------

# --- Imports from the model/engine modules ---
from engine import (
    ModelConfig, FreqParams, SplicedParams,
    build_spliced_from_priors, simulate_annual_losses,
    compute_metrics, lec, lec_bands, posterior_lambda
)
from controls import ControlSet, ControlCosts, control_effects, total_cost

# --- Standard libs & typing ---
import io, os, json, tempfile
from dataclasses import dataclass
from typing import Mapping, Optional

# --- 3P libs ---
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# --- In-app Help Panel ---------------------------------------------------------
def render_help_panel():
    with st.expander("❓ Help & User Guide", expanded=False):
        st.markdown("""
### What this app does
Runs a Monte Carlo cyber/AI-loss model to quantify **Expected Annual Loss (EAL)**, **tail risk (VaR95/99)**, **Loss Exceedance Curves (LEC)**, and **Return on Security Investment (ROSI)** under different control combinations. Optionally propagates **Bayesian** uncertainty for incident frequency (λ).

---

### Quick start
1. Pick **Risk mode** (sidebar): *Cyber Breach (records-based)* or *AI Incidents (monetary)*.  
2. (Cyber) Optionally apply a **NAICS preset** to seed λ, record cap, $/record, and net worth.  
3. Set **Scenario** inputs (trials, seed, net worth, records cap, cost/record, λ).  
4. Toggle **Controls** and set **Control costs**.  
5. (Optional) In **Advanced frequency**, provide (k, T) and enable **Bayesian lambda**; or enable **Negative Binomial** for overdispersion.  
6. Click **Run simulation** and review KPIs, charts, sensitivity, and downloads.

---

### Inputs (sidebar)
**Risk mode**
- *Cyber Breach (records-based)*: severity ≈ (fraction of customers) × ($/record), capped by records; controls can alter frequency, breach-any probability, and tail severity.
- *AI Incidents (monetary)*: trains frequency/severity from AI incident data (uploaded or demo); outputs monetary losses directly.

**Finance NAICS presets (Cyber)**
- Pre-fills: **λ**, **records cap**, **$ per record**, **net worth** for representative finance sub-sectors. You can still edit values after applying.

**Scenario**
- **Simulation trials**: number of Monte Carlo runs (e.g., 50k). More trials → more stable tail estimates; higher compute.
- **Random seed**: reproducibility.
- **Net worth (USD)**: used for VaR/Net Worth ratios.
- **Records / customers cap**: upper bound on exposed records per year.
- **Cost per record (USD)**: unit severity cost when records are breached.
- **Annual incident rate (λ)**: expected number of incidents per year.

**Controls + Control costs**
- Toggles: **Server hardening/patching: CIS 4, 7, 12, 10, 8**, **Media/DLP: CIS 3, 4, 6, 11, 15, 8**, **Change control/error-proofing: CIS 4, 16, 7, 8, 12/13, 5, 17**, **External/MFA & perimeter: CIS 6, 5, 12, 13, 9, 10, 16, 15, 8**.
- Costs: annual spend per control; used in **ROSI**.
- Effects are **data-driven** using ACTION/PATTERN shares with **diminishing returns** and **floors** to avoid unrealistic zero-risk.

**Data-driven control effects (shares)**
- Use built-in demo shares (NAICS-52-like) or upload 2 CSVs of shares by **ACTION** and **PATTERN**.
- These shares weight how each control reduces **λ / P(any loss)** (ACTION) and **tail severity** (PATTERN).

**Advanced frequency**
- **Bayesian lambda (Gamma prior + your data)**:
  - Prior: **α₀, β₀** (Gamma); Data: **k incidents** over **T years**.
  - Button suggests α₀ = λ̂·w, β₀ = w from **λ̂ = k/T** and pseudo-years **w**.
  - When enabled, the app **draws λ** from the posterior and re-runs the model many times, producing EAL/LEC with credible variation.
- **Negative Binomial (overdispersion)**:
  - Enable `Use Negative Binomial`; set dispersion **r** (smaller r → heavier variance than Poisson).

---

### Outputs (main area)
**KPIs**
- **EAL (Baseline / Controlled)**: mean annual loss with and without controls; delta shows reduction.
- **VaR95 / VaR99 (Base → Ctrl)**: 95th/99th percentile annual loss; deltas show tail risk reduction.
- **VaR/Net Worth**: risk as a % of capital.
- **ROSI**: \\( \\, \\frac{\\Delta EAL - \\text{Control Cost}}{\\text{Control Cost}} \\times 100\\% \\, \\)** (annualized heuristic).

**Convergence & Diagnostics**
- Convergence check via **batch CV**; if high, increase trials.
- Diagnostics: **ESS**, **tail mass**, **zero-loss ratio**, **MC SE(EAL)**, **relative error**.

**Loss Exceedance Curve (LEC)**
- Plots \\( P(\\text{Loss} \\ge x) \\) vs **x** (log–log).  
- If Bayesian enabled: shows **credible bands** around LECs from posterior λ draws.

**Distribution comparison**
- Overlaid histograms (log-x) for Baseline vs Controlled annual loss distributions.

**Sensitivity analysis (one at a time)**
- Varies **λ**, **P(any)**, or **r** by ±50% in 25% steps; plots impact on **EAL** and **VaR95**.

**Multi-year horizon**
- Simulates cumulative losses over 1–10 years and reports final-year **VaR95**.

**Scenario comparer (16 combos)**
- Evaluates all control combinations; shows **EAL**, **VaR95**, **cost**, and a ROSI ranking.

**Downloads**
- **Annual losses (CSV)** for Baseline and Controlled.
- **Full JSON report** with configuration, metrics, diagnostics, and loss samples.

---

### How inputs relate to outputs
- **λ (frequency)** ↑ ⇒ more events per year ⇒ **EAL**, **VaR95/99**, and LEC tail ↑.
- **P(any loss)** ↑ ⇒ higher chance an event produces loss ⇒ **EAL** ↑.
- **Records cap** / **$ per record** ↑ ⇒ larger severities ⇒ **EAL** and **VaR95/99** ↑.
- **Controls**:
  - **Server / External** mostly reduce **λ** and **P(any)** (via ACTION shares).
  - **Media / Server** can reduce **tail severity** (via PATTERN shares).
  - **Diminishing returns** and **floors** enforce realistic, non-zero residual risk.
- **Negative Binomial r** ↓ ⇒ more overdispersion ⇒ fatter tail ⇒ **VaR95/99** ↑.
- **Bayesian λ**: uncertainty in **λ** propagates to **EAL/LEC**; you’ll see credible spread, not just point estimates.

---

### Tips
- For board demos, start with **NAICS preset**, 50k trials, then toggle controls to show **delta EAL** and **VaR** shifts.
- If **Relative Error (EAL) > 5%** or convergence CV is high, increase trials.
- Use **Sensitivity** to show which parameter most drives tail risk for your audience.
- **ROSI** is a heuristic here; you can replace its formula to match your CFO’s standard.

*Questions or feature requests? Add them to the issue tracker or email Akudaikon.*
""")
# After:
# st.title("Akudaikon | Cyber-Loss Demo")
# st.caption("Monte Carlo loss model with control ROI, diagnostics, and optional Bayesian frequency.")
render_help_panel()

# ---------------------------------------------------------------------
# App Identity
# ---------------------------------------------------------------------
st.set_page_config(page_title="Akudaikon | Cyber-Loss Demo", layout="wide")
st.title("Akudaikon | Cyber-Loss Demo")
st.caption("Monte Carlo loss model with control ROI, diagnostics, and optional Bayesian frequency.")


# ---------------------------------------------------------------------
# Centralized configuration (no more scattered magic numbers)
# ---------------------------------------------------------------------
@dataclass
class SimulationConfig:
    MIN_LAMBDA_MULTIPLIER: float = 0.30     # Max 70% reduction from controls
    MIN_PANY_MULTIPLIER: float = 0.40       # Max 60% reduction
    MIN_GPD_SCALE_MULTIPLIER: float = 0.50  # Max 50% reduction

    CREDIBLE_INTERVAL_LEVEL: float = 0.90   # For LEC credible bands
    POSTERIOR_DRAWS: int = 200
    LEC_POINTS: int = 200

    CONVERGENCE_CV_THRESHOLD: float = 0.05
    BATCH_SIZE_FRACTION: float = 0.10       # For convergence batching

    MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024
    REPORT_MAX_LOSSES: int = 200_000
    REPORT_SCHEMA_VER: str = "1.2.0"

CFG = SimulationConfig()


# ---------------------------------------------------------------------
# Security & CSV helpers (OWASP-ish hardening for uploads/exports)
# ---------------------------------------------------------------------
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
    if getattr(file, "size", 0) > CFG.MAX_UPLOAD_BYTES:
        st.error(f"{label}: file is too large (> {CFG.MAX_UPLOAD_BYTES // (1024*1024)}MB).")
        st.stop()
    ctype = getattr(file, "type", "") or getattr(file, "content_type", "")
    if ctype and ("csv" not in ctype.lower() and "text" not in ctype.lower()):
        st.error(f"{label}: expected a CSV (got {ctype}).")
        st.stop()


# ---------------------------------------------------------------------
# Shares normalization & default action/pattern weights (demo)
# ---------------------------------------------------------------------
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

# Action-level control impacts (λ & p_any)
ACTION_IMPACT_BY_CONTROL = {
    "external": {"Hacking": 0.30, "Social": 0.25},
    "error": {"Error": 0.25},
    "server": {"Hacking": 0.15, "Misuse": 0.05},
    "media": {},
}

# Pattern-level control impacts (tail severity scale)
PATTERN_TAIL_IMPACT_BY_CONTROL = {
    "media": {"Misc Errors": 0.35, "Privilege Misuse": 0.15},
    "server": {"Basic Web App Attacks": 0.10},
    "external": {},
    "error": {},
}


# ---------------------------------------------------------------------
# Improved control-effects model (diminishing returns & interaction)
# ---------------------------------------------------------------------
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
        if not impact: return
        for act, strength in impact.items():
            share = a_sh.get(act, 0.0)
            factor = max(0.0, 1.0 - share * strength)
            lam_mult *= factor
            p_any_mult *= factor

    def apply_pattern(control_key: str):
        nonlocal gpd_scale_mult
        impact = PATTERN_TAIL_IMPACT_BY_CONTROL.get(control_key, {})
        if not impact: return
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

    # Apply floors (regulatory realism)
    lam_mult      = max(CFG.MIN_LAMBDA_MULTIPLIER, lam_mult)
    p_any_mult    = max(CFG.MIN_PANY_MULTIPLIER, p_any_mult)
    gpd_scale_mult= max(CFG.MIN_GPD_SCALE_MULTIPLIER, gpd_scale_mult)

    # Diminishing returns across multiple controls
    n_active = sum([getattr(ctrl, c, False) for c in ['server','media','error','external']])
    if n_active > 1:
        interaction_factor = 1.0 + (n_active - 1) * interaction_discount
        lam_mult       = 1.0 - (1.0 - lam_mult) / interaction_factor
        p_any_mult     = 1.0 - (1.0 - p_any_mult) / interaction_factor
        gpd_scale_mult = 1.0 - (1.0 - gpd_scale_mult) / interaction_factor

    return ControlEffects(lam_mult=lam_mult, p_any_mult=p_any_mult, gpd_scale_mult=gpd_scale_mult)


# ---------------------------------------------------------------------
# Simulation validation & diagnostics
# ---------------------------------------------------------------------
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
    batch_size = max(1, int(n * CFG.BATCH_SIZE_FRACTION))
    # avoid last short batch bias
    edges = list(range(0, n - batch_size + 1, batch_size))
    if not edges:  # very small n
        return {"converged": True, "cv": 0.0, "recommendation": "adequate"}

    batch_means = [losses[i:i+batch_size].mean() for i in edges]
    cv = (np.std(batch_means) / np.mean(batch_means)) if np.mean(batch_means) > 0 else np.inf
    return {
        "converged": cv < CFG.CONVERGENCE_CV_THRESHOLD,
        "cv": float(cv),
        "recommendation": "increase trials" if cv >= CFG.CONVERGENCE_CV_THRESHOLD else "adequate"
    }

def var_confidence_interval(losses: np.ndarray, alpha: float = 0.95, n_bootstrap: int = 1000) -> dict:
    n = len(losses)
    var_point = float(np.percentile(losses, alpha * 100))
    rng = np.random.default_rng(42)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(losses, size=n, replace=True)
        boot[i] = np.percentile(sample, alpha * 100)
    return {
        "point": var_point,
        "ci_lower": float(np.percentile(boot, 2.5)),
        "ci_upper": float(np.percentile(boot, 97.5)),
        "stderr": float(np.std(boot))
    }

def multi_year_simulation(cfg, fp, sp, years: int = 5) -> np.ndarray:
    annual_losses = np.zeros((cfg.trials, years))
    for year in range(years):
        annual_losses[:, year] = simulate_annual_losses(cfg, fp, sp)
    return annual_losses.cumsum(axis=1)

def monte_carlo_diagnostics(losses: np.ndarray) -> dict:
    n = len(losses)
    def autocorr(x, lag):
        if lag >= len(x): return 0.0
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
    autocorr_1 = autocorr(losses, 1)
    ess = n / (1 + 2 * max(0.0, autocorr_1))
    tail_5pct = losses[losses >= np.percentile(losses, 95)]
    tail_ratio = len(tail_5pct) / n
    zero_ratio = float((losses == 0).sum()) / n
    mc_se_eal = float(np.std(losses) / np.sqrt(n)) if n > 1 else float("inf")
    mean_loss = float(np.mean(losses))
    rel_err = (mc_se_eal / mean_loss) if mean_loss > 0 else float("inf")
    return {
        "ess": float(ess),
        "ess_ratio": float(ess / n),
        "tail_ratio": float(tail_ratio),
        "zero_ratio": float(zero_ratio),
        "mc_se_eal": mc_se_eal,
        "rel_error_eal": float(rel_err),
    }


# ---------------------------------------------------------------------
# Bayesian: propagate parameter uncertainty end-to-end
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Sensitivity analysis helper
# ---------------------------------------------------------------------
def run_sensitivity_analysis(cfg, fp, sp, param: str, ce=None):
    """
    Sensitivity to a single frequency parameter by ±50% in 25% steps.
    Returns a tidy DataFrame with EAL and VaR95.
    """
    base_val = getattr(fp, param, None)
    if base_val is None:
        raise ValueError(f"FreqParams has no attribute '{param}'")

    def clamp_param(name, value):
        if name == "p_any":
            return float(np.clip(value, 0.0, 1.0))
        if name == "r":
            return float(max(1e-6, value))
        return float(value)

    steps = [0.50, 0.75, 1.00, 1.25, 1.50]
    rows = []
    for mult in steps:
        test_val = clamp_param(param, base_val * mult)
        fp_test = FreqParams(
            lam=test_val if param == "lam" else fp.lam,
            p_any=test_val if param == "p_any" else fp.p_any,
            negbin=fp.negbin,
            r=test_val if param == "r" else fp.r
        )
        losses = simulate_annual_losses(cfg, fp_test, sp, ce)
        losses = np.asarray(losses, dtype=float)
        losses = losses[np.isfinite(losses)]
        losses = losses[losses >= 0]

        if losses.size == 0:
            eal = np.nan; v95 = np.nan
        else:
            mets = compute_metrics(losses, cfg.net_worth)
            eal = mets["EAL"]; v95 = mets["VaR95"]

        rows.append({
            "multiplier": mult,
            "param_value": test_val,
            "EAL": eal,
            "VaR95": v95
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Distributions & scenarios
# ---------------------------------------------------------------------
def plot_loss_distributions(base_losses, ctrl_losses):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=base_losses[base_losses > 0], name="Baseline", opacity=0.6, nbinsx=50
    ))
    fig.add_trace(go.Histogram(
        x=ctrl_losses[ctrl_losses > 0], name="Controlled", opacity=0.6, nbinsx=50
    ))
    fig.update_xaxes(type="log", title="Annual Loss (USD)")
    fig.update_yaxes(title="Frequency")
    fig.update_layout(barmode='overlay', title="Loss Distribution Comparison (log scale)")
    return fig

def compare_control_scenarios(cfg, fp, sp, costs, action_shares, pattern_shares):
    scenarios = []
    control_names = ['server','media','error','external']
    for i in range(16):
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
        losses = validate_losses(losses, "Scenario")
        metrics = compute_metrics(losses, cfg.net_worth)
        cost = total_cost(ctrl, costs)
        active = [name for j, name in enumerate(control_names) if (i & (1 << j)) > 0]
        # ROSI here is a simple EAL delta vs cost proxy; you can swap for your preferred ROSI
        scenarios.append({
            "scenario": " + ".join(active) if active else "None",
            "n_controls": len(active),
            "EAL": metrics["EAL"],
            "VaR95": metrics["VaR95"],
            "cost": cost,
            "ROSI": ((metrics["EAL"] - cost) / cost * 100.0) if cost > 0 else np.nan
        })
    return pd.DataFrame(scenarios).sort_values("ROSI", ascending=False)


# ---------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------
def generate_full_report(cfg, fp, sp, ctrl, costs, base_losses, ctrl_losses):
    from datetime import datetime
    baseline_metrics  = compute_metrics(base_losses, cfg.net_worth)
    controlled_metrics= compute_metrics(ctrl_losses, cfg.net_worth)
    diag_base = monte_carlo_diagnostics(base_losses)
    diag_ctrl = monte_carlo_diagnostics(ctrl_losses)

    report = {
        "schema_version": CFG.REPORT_SCHEMA_VER,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "app": "Akudaikon Cyber-Loss Demo",
        },
        "provenance": {
            "trials": int(cfg.trials),
            "seed": int(cfg.seed),
            "engine": {
                "frequency": {
                    "lambda": float(fp.lam),
                    "p_any": float(fp.p_any),
                    "negbin": bool(fp.negbin),
                    "r": float(fp.r),
                },
                "severity": {"type": "spliced"},
            },
        },
        "configuration": {
            "net_worth": float(cfg.net_worth),
            "record_cap": int(getattr(cfg, "record_cap", 0)),
            "cost_per_record": float(getattr(cfg, "cost_per_record", 0.0)),
            "controls": {
                "active": [k for k, v in ctrl.__dict__.items() if v],
                "annual_cost": float(total_cost(ctrl, costs)),
                "costs": {k: getattr(costs, k) for k in costs.__dict__.keys()},
            },
        },
        "results": {
            "baseline": baseline_metrics,
            "controlled": controlled_metrics,
            "deltas": {
                "EAL": baseline_metrics["EAL"] - controlled_metrics["EAL"],
                "VaR95": baseline_metrics["VaR95"] - controlled_metrics["VaR95"],
                "VaR99": baseline_metrics["VaR99"] - controlled_metrics["VaR99"],
            },
        },
        "diagnostics": {"baseline": diag_base, "controlled": diag_ctrl},
        "raw_data": {
            "baseline_losses_sample": base_losses[:min(10000, len(base_losses))].tolist(),
            "controlled_losses_sample": ctrl_losses[:min(10000, len(ctrl_losses))].tolist(),
        }
    }
    return report


# ---------------------------------------------------------------------
# AIID/HAI CSV normalizer & validator (for AI Incidents mode)
# ---------------------------------------------------------------------
def normalize_aiid_csvs(enriched_src, hai62_src):
    inc = pd.read_csv(enriched_src)
    hai = pd.read_csv(hai62_src)

    if "incident_id" not in hai.columns and "id" in hai.columns:
        hai = hai.rename(columns={"id": "incident_id"})

    if "year" not in hai.columns:
        year_source = None
        if {"incident_id", "year"}.issubset(inc.columns):
            year_source = inc[["incident_id", "year"]].drop_duplicates()
        else:
            for date_col in ["published","date","incident_date","event_date","report_date"]:
                if date_col in inc.columns:
                    yy = pd.to_datetime(inc[date_col], errors="coerce").dt.year
                    if yy.notna().any():
                        inc = inc.assign(year=yy)
                        year_source = inc[["incident_id", "year"]].drop_duplicates()
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

def validate_ai_data(df_ai: pd.DataFrame) -> dict:
    issues = []
    if "loss_usd" in df_ai.columns:
        has_loss = df_ai["loss_usd"].notna() & (df_ai["loss_usd"] > 0)
        n_losses = int(has_loss.sum())
        if n_losses < 30:
            issues.append(f"Only {n_losses} loss observations - results may be unreliable")
    else:
        issues.append("Missing 'loss_usd' column")

    if "loss_confidence" in df_ai.columns:
        low_conf = int((df_ai["loss_confidence"] < 0.5).sum())
        if len(df_ai) > 0 and (low_conf / len(df_ai)) > 0.7:
            issues.append("70%+ of losses have confidence < 0.5")

    if "year" in df_ai.columns and df_ai["year"].notna().any():
        year_span = int(df_ai["year"].max() - df_ai["year"].min())
        if year_span < 2:
            issues.append(f"Only {year_span} years of data - frequency may be unreliable")
    else:
        issues.append("Missing usable 'year' column")

    score = max(0, 100 - len(issues) * 20)
    return {
        "n_records": int(len(df_ai)),
        "n_with_loss": int(df_ai["loss_usd"].gt(0).sum()) if "loss_usd" in df_ai.columns else 0,
        "issues": issues,
        "quality_score": score
    }


# =====================================================================
# UI: Risk mode selector
# =====================================================================
mode = st.sidebar.radio(
    "Risk mode",
    ("Cyber Breach (records-based)", "AI Incidents (monetary)"),
    index=0
)


# ---------------------------------------------------------------------
# Advanced frequency panel (Bayes & NegBin)
# ---------------------------------------------------------------------
with st.sidebar.expander("Advanced frequency", expanded=False):
    use_bayes   = st.checkbox("Bayesian lambda (Gamma prior + your data)", value=False, key="adv_use_bayes")
    alpha0      = st.number_input("lambda prior alpha", min_value=0.01, max_value=50.0, value=2.0, step=0.1, key="adv_alpha0")
    beta0       = st.number_input("lambda prior beta",  min_value=0.01, max_value=50.0, value=8.0, step=0.1, key="adv_beta0")
    k_obs       = st.number_input("Incidents observed (k)", min_value=0, max_value=100000, value=0, step=1, key="adv_k_obs")
    T_obs       = st.number_input("Observation years (T)",  min_value=0.0, max_value=200.0, value=0.0, step=0.5, key="adv_T_obs")
    use_negbin  = st.checkbox("Use Negative Binomial (overdispersion)", value=False, key="adv_use_negbin")
    disp_r      = st.number_input("NegBin dispersion r", min_value=0.5, max_value=10.0, value=1.5, step=0.1, key="adv_disp_r")

st.markdown("**Calibration (from dataset slice)**")
ALPHA_MIN = 0.01; BETA_MIN = 0.01; STEP = 0.1; PREC = 4
_defaults = {"adv_use_bayes": False, "adv_alpha0": 2.0, "adv_beta0": 8.0, "adv_pseudo_w": 2.0}
for _k, _v in _defaults.items():
    if _k not in st.session_state: st.session_state[_k] = _v

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
    cols = st.columns([1,1,1])
    with cols[0]:
        st.number_input("Pseudo-years (weight for prior)", min_value=0.1, max_value=50.0,
                        value=float(st.session_state["adv_pseudo_w"]), step=0.1, key="adv_pseudo_w")
    _alpha_preview = _round_to_step(max(ALPHA_MIN, lam_hat * float(st.session_state["adv_pseudo_w"])))
    _beta_preview  = _round_to_step(max(BETA_MIN,  float(st.session_state["adv_pseudo_w"])))
    with cols[1]:
        st.write(f"Suggested α₀ = max({ALPHA_MIN}, λ̂·w) → **{_alpha_preview:.4f}**")
        st.write(f"Suggested β₀ = max({BETA_MIN}, w) → **{_beta_preview:.4f}**")
    with cols[2]:
        disabled = (lam_hat <= 0.0)
        tip = "Need k>0 (or T>0) for informative seeding." if disabled else "Apply and enable Bayesian frequency."
        st.button("Apply prior α₀=λ̂·w, β₀=w", use_container_width=True, disabled=disabled, help=tip, on_click=seed_prior_cb)
else:
    st.caption("Provide k and T to compute λ̂ (and optionally seed a weak prior).")


# ---------------------------------------------------------------------
# Data-driven control effects (via ACTION/PATTERN shares)
# ---------------------------------------------------------------------
with st.sidebar.expander("Data-driven control effects (shares)", expanded=False):
    st.caption("Use NAICS-52 demo shares or upload CSVs to weight control effects by ACTION/PATTERN.")
    shares_mode = st.radio("Shares source", ["Built-in NAICS-52 (demo)", "Upload CSVs"], index=0, key="shares_mode")
    action_shares = DEFAULT_ACTION_SHARES
    pattern_shares = DEFAULT_PATTERN_SHARES

    if shares_mode == "Upload CSVs":
        up_actions = st.file_uploader("Upload action shares CSV (columns: category, share)", type=["csv"], key="up_actions")
        up_patterns= st.file_uploader("Upload pattern shares CSV (columns: category, share)", type=["csv"], key="up_patterns")
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


# ---------------------------------------------------------------------
# NAICS Presets (Finance & Insurance)
# ---------------------------------------------------------------------
NAICS_FINANCE_PRESETS = {
    "522130 — Credit Unions": {"lambda": 0.35, "records_cap": 250_000, "cost_per_record": 185.0, "net_worth": 100_000_000.0},
    "522110 — Commercial Banking": {"lambda": 0.60, "records_cap": 5_000_000, "cost_per_record": 185.0, "net_worth": 2_000_000_000.0},
    "522320 — Financial Transactions Processing": {"lambda": 0.65, "records_cap": 8_000_000, "cost_per_record": 200.0, "net_worth": 1_500_000_000.0},
}


# =====================================================================
# BRANCH 1: CYBER BREACH (records-based)
# =====================================================================
if mode == "Cyber Breach (records-based)":

    with st.sidebar.expander("Finance NAICS presets", expanded=False):
        use_naics = st.checkbox("Use preset", value=False, key="naics_enable")
        _keys = list(NAICS_FINANCE_PRESETS.keys())
        _default_label = "522130 — Credit Unions"
        _default_index = _keys.index(_default_label) if _default_label in _keys else 0
        choice = st.selectbox("Select NAICS (Finance)", _keys, index=_default_index, disabled=not use_naics, key="naics_choice")
        if use_naics:
            p = NAICS_FINANCE_PRESETS[choice]
            st.session_state["in_lambda"]      = p["lambda"]
            st.session_state["in_records_cap"] = p["records_cap"]
            st.session_state["in_cpr"]         = p["cost_per_record"]
            st.session_state["in_networth"]    = p["net_worth"]
            st.caption(f"Preset applied: {choice}")

    # Scenario + Controls
    with st.sidebar.form("scenario_form"):
        st.header("Scenario")
        trials            = st.number_input("Simulation trials", min_value=1_000, max_value=500_000, value=50_000, step=5_000, key="in_trials")
        net_worth         = st.number_input("Net worth (USD)", min_value=0.0, value=float(st.session_state.get("in_networth", 1_000_000.0)), step=100_000.0, format="%.0f", key="in_networth")
        seed              = st.number_input("Random seed", min_value=0, value=42, step=1, key="in_seed")
        num_customers     = st.number_input("Records / customers cap", min_value=1, value=int(st.session_state.get("in_records_cap", 1_000_000)), step=10_000, key="in_records_cap")
        cost_per_customer = st.number_input("Cost per record (USD)", min_value=1.0, value=float(st.session_state.get("in_cpr", 150.0)), step=10.0, format="%.2f", key="in_cpr")
        lam               = st.number_input("Annual incident rate (lambda)", min_value=0.0, value=float(st.session_state.get("in_lambda", 0.40)), step=0.05, format="%.2f", key="in_lambda")

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

    # ---- Run the simulation once the form is submitted ----
    if submitted:
        with st.spinner("Simulating..."):
            # Config
            cfg = ModelConfig(
                trials=int(trials),
                net_worth=float(net_worth),
                seed=int(seed),
                record_cap=int(num_customers),
                cost_per_record=float(cost_per_customer),
            )

            # Frequency (Bayesian optional)
            lam_base = float(lam)
            lam_draws = None
            if use_bayes and T_obs > 0:
                lam_draws = posterior_lambda(
                    float(alpha0), float(beta0), int(k_obs), float(T_obs),
                    draws=CFG.POSTERIOR_DRAWS, seed=int(seed) + 100
                )
                lam_base = float(np.median(lam_draws))

            fp = FreqParams(lam=lam_base, p_any=0.85, negbin=bool(use_negbin), r=float(disp_r))

            # Severity prior (spliced)
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

                # Show Posterior EAL summaries
                st.success(f"Bayesian EAL (baseline): ${bayes['baseline_eal_mean']:,.0f} "
                          f"[95% CI: ${bayes['baseline_eal_ci'][0]:,.0f} – ${bayes['baseline_eal_ci'][1]:,.0f}]")
                st.success(f"Bayesian EAL (controlled): ${bayes['controlled_eal_mean']:,.0f} "
                          f"[95% CI: ${bayes['controlled_eal_ci'][0]:,.0f} – ${bayes['controlled_eal_ci'][1]:,.0f}]")
            else:
                base_losses = simulate_annual_losses(cfg, fp, sp)
                ctrl_losses = simulate_annual_losses(cfg, fp, sp, ce)

            # Validate
            base_losses = validate_losses(base_losses, "Baseline")
            ctrl_losses = validate_losses(ctrl_losses, "Controlled")

            # Metrics & KPIs
            base_m = compute_metrics(base_losses, cfg.net_worth)
            ctrl_m = compute_metrics(ctrl_losses, cfg.net_worth)
            control_cost = total_cost(ctrl, costs)
            delta_eal = base_m["EAL"] - ctrl_m["EAL"]
            rosi = ((delta_eal - control_cost) / control_cost * 100.0) if control_cost > 0 else np.nan

            # KPI tiles
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

            # Convergence
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

            # VaR CIs
            var95_result = var_confidence_interval(base_losses, 0.95)
            var99_result = var_confidence_interval(base_losses, 0.99)
            e1, e2 = st.columns(2)
            e1.metric("VaR95 (± 95% CI half-width)", f"${var95_result['point']:,.0f}",
                      delta=f"±${((var95_result['ci_upper']-var95_result['ci_lower'])/2):,.0f}")
            e2.metric("VaR99 (± 95% CI half-width)", f"${var99_result['point']:,.0f}",
                      delta=f"±${((var99_result['ci_upper']-var99_result['ci_lower'])/2):,.0f}")

            # Distribution comparison
            st.plotly_chart(plot_loss_distributions(base_losses, ctrl_losses), use_container_width=True)

            # Summary table
            st.subheader("Summary")
            summary_df = pd.DataFrame({
                "Metric": ["EAL", "VaR95", "VaR99", "VaR95/NetWorth", "VaR99/NetWorth", "Control Cost", "Delta EAL", "ROSI %"],
                "Baseline":  [base_m["EAL"], base_m["VaR95"], base_m["VaR99"],
                              base_m["VaR95_to_NetWorth"], base_m["VaR99_to_NetWorth"],
                              np.nan, np.nan, np.nan],
                "Controlled":[ctrl_m["EAL"], ctrl_m["VaR95"], ctrl_m["VaR99"],
                              ctrl_m["VaR95_to_NetWorth"], ctrl_m["VaR99_to_NetWorth"],
                              control_cost, delta_eal, rosi],
            })
            st.dataframe(summary_df.style.format({"Baseline": "{:,.2f}", "Controlled": "{:,.2f}"}), use_container_width=True)

            # ---------- SENSITIVITY + HORIZON + COMBINATIONS (UI) ----------
            st.markdown("---")
            st.subheader("Advanced Analysis")

            # Sensitivity analysis
            with st.expander("Run sensitivity analysis", expanded=False):
                st.caption("Vary one frequency parameter ±50% and observe EAL & VaR95.")
                sens_param = st.selectbox(
                    "Parameter",
                    ["lam", "p_any", "r"],
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
                            cfg, fp, sp, sens_param,
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

            # Downloads
            out_df = pd.DataFrame({"annual_loss_baseline": base_losses, "annual_loss_controlled": ctrl_losses})
            st.download_button("Download annual losses (CSV)", _safe_to_csv(out_df),
                               "cyber_annual_losses.csv", "text/csv")

            # Full report (JSON)
            report = generate_full_report(cfg, fp, sp, ctrl, costs, base_losses, ctrl_losses)
            st.download_button("📄 Download Full Report (JSON)", json.dumps(report, indent=2),
                               "cyber_risk_report.json", "application/json")


# =====================================================================
# BRANCH 2: AI INCIDENTS (monetary)
# =====================================================================
elif mode == "AI Incidents (monetary)":

    st.header("AI Incidents | Monetary Risk")
    st.caption("AIID incidents enriched with policy context → EAL, VaR95/99, LEC.")

    # ---- Inputs
    c1, c2 = st.columns(2)
    enriched_up = c1.file_uploader("Enriched incidents CSV", type=["csv"], accept_multiple_files=False)
    hai62_up    = c2.file_uploader("HAI 6.2 join-pack CSV", type=["csv"], accept_multiple_files=False)
    _validate_upload(enriched_up, "Enriched incidents CSV")
    _validate_upload(hai62_up, "HAI 6.2 join-pack CSV")

    c3, c4, c5 = st.columns(3)
    min_conf = c3.slider("Min loss confidence (for training $ severity)", 0.0, 1.0, 0.70, 0.05)
    trials   = int(c4.selectbox("Monte Carlo trials", [2000, 5000, 10000, 20000], index=2))
    seed     = int(c5.number_input("Random seed", value=42, step=1))

    # Decide source: uploads → repo defaults → synthetic demo
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

    # Helper: generic LEC from losses
    def _lec_dataframe(losses: np.ndarray, n: int = CFG.LEC_POINTS) -> pd.DataFrame:
        lo = max(1.0, float(np.percentile(losses, 1)))
        hi = float(np.percentile(losses, 99.9))
        if hi <= lo: hi = lo * 10.0
        xs = np.logspace(np.log10(lo), np.log10(hi), n)
        probs = [(losses >= x).mean() for x in xs]
        return pd.DataFrame({"loss": xs, "prob_exceed": probs})

    # Synthetic demo fallback
    def _simulate_ai_demo(trials: int, seed: int, lam: float = 0.45, sev_mu: float = 11.5, sev_sigma: float = 1.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        k = rng.poisson(lam=lam, size=trials)
        m = int(k.max()) if trials > 0 else 0
        if m == 0: return np.zeros(trials)
        sev = rng.lognormal(mean=sev_mu, sigma=sev_sigma, size=(trials, m))
        mask = np.arange(m)[None, :] < k[:, None]
        return (sev * mask).sum(axis=1)

    def _metrics_from_losses(losses: np.ndarray) -> tuple[float, float, float]:
        return (float(losses.mean()), float(np.percentile(losses, 95)), float(np.percentile(losses, 99)))

    # PATH A: Uploads or repo defaults → real pipeline
    if source in ("uploads", "repo"):
        try:
            from ai_monetary import (
                load_ai_table, fit_severity, fit_frequency,
                scenario_vector, simulate_eal_var, lec_dataframe
            )
        except Exception:
            st.error("AI Incidents mode needs scikit-learn & ai_monetary.py. Add dependencies and redeploy.")
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

        # Data quality checks
        data_quality = validate_ai_data(df_ai)
        st.metric("Data Quality Score", f"{data_quality['quality_score']}/100")
        for issue in data_quality['issues']:
            st.warning(f"⚠️ {issue}")

        countries = ["(all)"] + (sorted(df_ai["country_group"].dropna().unique().tolist()) if "country_group" in df_ai else [])
        country = st.selectbox("Country", countries or ["(all)"])
        domains = st.multiselect("Domains", ["finance","healthcare","transport","social_media","hiring_hr","law_enforcement","education"], default=["finance"])
        mods    = st.multiselect("Modalities", ["vision","nlp","recommender","generative","autonomous"], default=[])

        sev_model, sigma = fit_severity(df_ai, min_conf=min_conf)
        freq_model       = fit_frequency(df_ai)
        x_row            = scenario_vector(df_ai, None if country=="(all)" else country, domains, mods)

        eal, var95, var99, losses = simulate_eal_var(freq_model, sev_model, sigma, x_row, trials=trials, seed=seed)

        k1, k2, k3 = st.columns(3)
        k1.metric("EAL",    f"${eal:,.0f}")
        k2.metric("VaR 95", f"${var95:,.0f}")
        k3.metric("VaR 99", f"${var99:,.0f}")

        lec_ai = lec_dataframe(losses)
        fig = px.line(lec_ai, x="loss", y="prob_exceed", title="AI Incidents — Loss Exceedance Curve",
                      labels={"loss": "Loss ($)", "prob_exceed": "P(Loss ≥ x)"})
        fig.update_xaxes(type="log"); fig.update_yaxes(type="log", range=[-2.5, 0])
        st.plotly_chart(fig, use_container_width=True)

        # Optional: download the annual loss series
        st.download_button("Download scenario losses (CSV)", _safe_to_csv(pd.DataFrame({"annual_loss": losses})),
                           "ai_scenario_annual_losses.csv", "text/csv")

    # PATH B: Nothing available → synthetic demo
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
        st.download_button("Download demo losses (CSV)", buf.getvalue(),
                           "ai_demo_annual_losses.csv", "text/csv")

