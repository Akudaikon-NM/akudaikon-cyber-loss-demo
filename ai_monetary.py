# ai_monetary.py
"""
AI Incident Risk Modeling
-------------------------
Fits frequency (Poisson) and severity (Ridge on log-losses) models from AIID data,
then simulates annual losses via Monte Carlo.

Highlights:
- Robust column name handling (tries multiple alternatives)
- Data validation (minimum observations, positive losses)
- Safety clipping on model predictions (λ, μ, σ)
- Better LEC tail resolution
- Strong model diagnostics (Poisson deviance, McFadden pseudo-R², AUC/Brier, RMSE/MAE on log and $)
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.metrics import (
    mean_poisson_deviance,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    brier_score_loss,
)

AI_DEFAULT_ENRICHED = "/mnt/data/akudaikon_incidents_enriched.csv"
AI_DEFAULT_HAI62    = "/mnt/data/akudaikon_joinpack_hai_6_2.csv"

# ---------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------
def _feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return list of feature column names present in the dataframe.
    Includes core features (domains, modalities, regulatory, etc.) and HAI 6.2 columns.
    """
    core = [
        "severity_proxy", "regulatory_action", "country_group",
        "dom_finance", "dom_healthcare", "dom_transport", "dom_social_media",
        "dom_hiring_hr", "dom_law_enforcement", "dom_education",
        "mod_vision", "mod_nlp", "mod_recommender", "mod_generative", "mod_autonomous",
        "life_development", "life_deployment", "year"
    ]
    hai = [c for c in df.columns if c.startswith("fig_6_2_")]
    # keep only those that exist and are not all-null
    cols = [c for c in core if c in df.columns]
    cols += [c for c in hai if c in df.columns]
    return cols


def _prep_X(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Extract feature columns, coerce to numeric, fill NaN with 0.
    If no feature columns exist, returns an empty one-row DF of zeros (intercept-only feel).
    """
    if not cols:
        # return a dummy single column to avoid zero-feature errors downstream
        return pd.DataFrame({"_bias": np.ones(len(df), dtype=float)})

    X = df[list(cols)].copy()
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0.0)


# ---------------------------------------------------------------------
# Data loading with robust column handling
# ---------------------------------------------------------------------
def load_ai_table(
    enriched_csv: str = AI_DEFAULT_ENRICHED,
    hai62_csv: Optional[str] = AI_DEFAULT_HAI62
) -> pd.DataFrame:
    """
    Load and merge enriched incidents CSV with HAI 6.2 join-pack CSV.
    Handles common column name variations and missing columns gracefully.

    Returns:
        DataFrame with incident_id (if available), year, and feature columns ready for modeling.
    """
    df = pd.read_csv(enriched_csv)

    # Extract year if needed
    if "year" not in df.columns:
        for date_col in ["date", "published", "incident_date", "event_date", "report_date"]:
            if date_col in df.columns:
                df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
                if df["year"].notna().any():
                    break
        if "year" not in df.columns:
            df["year"] = 2020  # neutral fallback

    # Merge HAI 6.2 if available
    if hai62_csv:
        try:
            hai = pd.read_csv(hai62_csv)
            if "incident_id" not in hai.columns and "id" in hai.columns:
                hai = hai.rename(columns={"id": "incident_id"})
            if "incident_id" in df.columns and "incident_id" in hai.columns:
                use_cols = ["incident_id"] + [c for c in hai.columns if c != "incident_id"]
                df = df.merge(hai[use_cols], on="incident_id", how="left", suffixes=("", "_hai"))
            else:
                print("⚠️ Warning: Cannot merge HAI data - 'incident_id' column missing")
        except Exception as e:
            print(f"⚠️ Warning: Could not load HAI data: {e}")

    return df


# ---------------------------------------------------------------------
# Severity model (Ridge regression on log-losses)
# ---------------------------------------------------------------------
def fit_severity(df: pd.DataFrame, min_conf: float = 0.7) -> Tuple[Ridge, float]:
    """
    Fit a Ridge regression model to predict log1p(monetary loss).

    Args:
        df: Incident dataframe.
        min_conf: Minimum loss_confidence threshold (0-1) to include observations.

    Returns:
        (trained Ridge model, residual std dev in log space)
    """
    use = df.copy()

    # Find a usable monetary loss column
    loss_col = None
    for col in [
        "loss_estimate_combined_usd", "loss_usd", "loss_estimate_usd",
        "loss_amount", "monetary_loss", "severity_proxy"
    ]:
        if col in use.columns:
            loss_col = col
            break
    if loss_col is None:
        raise ValueError("No loss column found. Tried several common names.")

    if loss_col != "loss_estimate_combined_usd":
        use["loss_estimate_combined_usd"] = use[loss_col]

    # Confidence filter
    if "loss_confidence" in use.columns:
        use = use[use["loss_confidence"].fillna(0) >= float(min_conf)]

    # Positive losses only
    use = use[use["loss_estimate_combined_usd"].fillna(0) > 0]

    if len(use) < 10:
        raise ValueError(
            f"Insufficient data: only {len(use)} valid loss observations after filtering (min 10)."
        )

    y = np.log1p(use["loss_estimate_combined_usd"].clip(lower=0))
    X = _prep_X(use, _feature_cols(use))
    if len(X) != len(y):
        raise ValueError(f"Feature/target size mismatch: {len(X)} vs {len(y)}")

    model = Ridge(alpha=1.0, random_state=42).fit(X, y)

    resid = y - model.predict(X)
    sigma = float(np.std(resid)) if len(resid) > 1 else 1.0
    # keep sigma within sane MC bounds
    sigma = float(np.clip(sigma, 0.1, 5.0))
    return model, sigma


# ---------------------------------------------------------------------
# Frequency model (Poisson regression)
# ---------------------------------------------------------------------
def fit_frequency(df: pd.DataFrame) -> PoissonRegressor:
    """
    Fit a Poisson regression model to predict incident frequency (harm occurrence proxy).

    Returns:
        Trained PoissonRegressor.
    """
    harm_col = None
    for col in ["harm_occurred_final", "harm_occurred", "incident_occurred", "is_incident", "harm_binary"]:
        if col in df.columns:
            harm_col = col
            break
    if harm_col is None:
        raise ValueError(
            "No harm/incident indicator column found. "
            "Tried: harm_occurred_final, harm_occurred, incident_occurred, is_incident, harm_binary"
        )

    y = df[harm_col].fillna(0).astype(int).clip(0, 1)
    X = _prep_X(df, _feature_cols(df))
    if len(X) != len(y):
        raise ValueError(f"Feature/target size mismatch: {len(X)} vs {len(y)}")

    n_pos = int(y.sum())
    if n_pos < 5:
        raise ValueError(
            f"Insufficient positive harm incidents: {n_pos} (min 5). "
            "Check data labeling/quality."
        )

    model = PoissonRegressor(alpha=1.0, max_iter=1000).fit(X, y)
    return model


# ---------------------------------------------------------------------
# Scenario vector construction
# ---------------------------------------------------------------------
def scenario_vector(
    df: pd.DataFrame,
    country: Optional[str] = None,
    domains: Optional[Sequence[str]] = None,
    modalities: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """
    Build a single-row feature vector for a scenario by filtering and averaging.
    Falls back to whole dataset if the filter is empty.
    """
    mask = pd.Series(True, index=df.index)

    if country and "country_group" in df.columns:
        mask &= (df["country_group"].fillna("Other/Unknown") == country)

    for d in (domains or []):
        col = f"dom_{d}"
        if col in df.columns:
            mask &= (df[col] == 1)

    for m in (modalities or []):
        col = f"mod_{m}"
        if col in df.columns:
            mask &= (df[col] == 1)

    cohort = df[mask] if bool(mask.any()) else df
    X = _prep_X(cohort, _feature_cols(df))
    # mean of cohort → one row
    return X.mean(numeric_only=True).to_frame().T


# ---------------------------------------------------------------------
# Monte Carlo simulation with safety bounds
# ---------------------------------------------------------------------
def simulate_eal_var(
    freq_model: PoissonRegressor,
    sev_model: Ridge,
    sigma_log: float,
    x_row: pd.DataFrame,
    trials: int = 10_000,
    seed: int = 42
) -> Tuple[float, float, float, np.ndarray]:
    """
    Simulate annual losses via Monte Carlo using fitted frequency (Poisson) and severity (lognormal) models.

    Returns:
        (EAL, VaR95, VaR99, annual_losses_array)
    """
    rng = np.random.default_rng(int(seed))

    # Predict λ (rate per year) with safety bounds
    lam_raw = float(freq_model.predict(x_row)[0])
    lam = float(np.clip(lam_raw, 1e-3, 100.0))  # clip to reasonable annual rate

    # Predict log-mean severity μ with safety bounds
    mu_raw = float(sev_model.predict(x_row)[0])
    mu = float(np.clip(mu_raw, 0.0, 30.0))  # exp(30) ~ 1e13

    # Clip sigma to prevent extreme tails
    sigma = float(np.clip(sigma_log, 0.1, 5.0))

    # Draw counts and severities
    k = rng.poisson(lam, size=trials)  # number of events per trial
    total = np.zeros(trials, dtype=float)

    # Efficient sampling: handle only trials with k>0 in a vectorized-ish loop
    nonzero_idx = np.where(k > 0)[0]
    for i in nonzero_idx:
        total[i] = rng.lognormal(mean=mu, sigma=sigma, size=int(k[i])).sum()

    # Clean
    total = total[np.isfinite(total)]
    if total.size == 0:
        raise ValueError(
            f"All simulated losses invalid (nan/inf). Check λ={lam:.4f}, μ={mu:.4f}, σ={sigma:.4f}."
        )

    eal   = float(np.mean(total))
    var95 = float(np.percentile(total, 95))
    var99 = float(np.percentile(total, 99))
    return eal, var95, var99, total


# ---------------------------------------------------------------------
# Loss Exceedance Curve (LEC) with better tail resolution
# ---------------------------------------------------------------------
def lec_dataframe(losses: Iterable[float], points: int = 200) -> pd.DataFrame:
    """
    Generate Loss Exceedance Curve (LEC) with denser tail sampling (90th–99.9th percentiles).
    """
    losses = np.asarray(list(losses), dtype=float)
    losses = losses[np.isfinite(losses) & (losses >= 0)]
    if losses.size == 0:
        return pd.DataFrame({"loss": [0.0], "prob_exceed": [1.0]})

    pts_body = max(2, points // 2)
    pts_tail = max(2, points - pts_body)
    qs = np.concatenate([
        np.linspace(0.0, 0.90, pts_body, endpoint=False),
        np.linspace(0.90, 0.999, pts_tail)
    ])
    loss_levels = np.percentile(losses, qs * 100.0)
    exceed_probs = 1.0 - qs
    return pd.DataFrame({"loss": loss_levels, "prob_exceed": exceed_probs})


# ---------------------------------------------------------------------
# Model diagnostics (robust)
# ---------------------------------------------------------------------
def _is_binary(y: np.ndarray) -> bool:
    return np.all(np.isin(y, [0, 1]))


def _loglik_poisson(y: np.ndarray, mu: np.ndarray, eps: float = 1e-12) -> float:
    """Poisson log-likelihood up to additive constant: sum(y*log(mu) - mu - log(y!))."""
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), eps, np.inf)
    return float(np.sum(y * np.log(mu) - mu - np.vectorize(math.lgamma)(y + 1.0)))


def model_diagnostics(
    freq_model: PoissonRegressor,
    sev_model: Ridge,
    df_test: pd.DataFrame,
    min_conf: float = 0.7
) -> Dict[str, float]:
    """
    Diagnostics for frequency (Poisson/binary) and severity (log1p-trained) models.

    Returns a dict with:
      - Frequency: MAE, RMSE, mean Poisson deviance, McFadden pseudo-R²,
                   and (if binary) ROC AUC, Brier score.
      - Severity:  RMSE/MAE on log1p scale and on $ scale, plus MAPE (safe).
    """
    out: Dict[str, float] = {}

    # --- Frequency diagnostics
    harm_col = None
    for col in ["harm_occurred_final", "harm_occurred", "incident_occurred", "is_incident", "harm_binary"]:
        if col in df_test.columns:
            harm_col = col
            break

    if harm_col:
        y_freq = df_test[harm_col].fillna(0).astype(int).clip(0, 1).to_numpy()
        X_freq = _prep_X(df_test, _feature_cols(df_test))
        mu_hat = np.clip(freq_model.predict(X_freq), 1e-12, 1e6)
        out["n_test_freq"] = int(len(y_freq))

        # basic errors on λ
        out["freq_mae"]  = float(mean_absolute_error(y_freq, mu_hat))
        out["freq_rmse"] = float(np.sqrt(mean_squared_error(y_freq, mu_hat)))

        # Poisson deviance
        try:
            out["freq_poisson_deviance"] = float(mean_poisson_deviance(y_freq, mu_hat))
        except Exception:
            out["freq_poisson_deviance"] = float("nan")

        # McFadden pseudo-R² (Poisson)
        try:
            ll_full = _loglik_poisson(y_freq, mu_hat)
            mu_null = np.full_like(y_freq, max(y_freq.mean(), 1e-12), dtype=float)
            ll_null = _loglik_poisson(y_freq, mu_null)
            pseudo_r2 = 1.0 - (ll_full / ll_null) if ll_null != 0 else np.nan
            out["freq_pseudoR2_mcfadden"] = float(np.clip(pseudo_r2, -1.0, 1.0))
        except Exception:
            out["freq_pseudoR2_mcfadden"] = float("nan")

        # If labels are binary, add AUC/Brier using p ≈ 1 - exp(-λ)
        if _is_binary(y_freq):
            p_hat = 1.0 - np.exp(-np.clip(mu_hat, 0, 50))
            try:
                out["freq_auc"] = float(roc_auc_score(y_freq.astype(int), p_hat))
            except Exception:
                out["freq_auc"] = float("nan")
            try:
                out["freq_brier"] = float(brier_score_loss(y_freq.astype(int), np.clip(p_hat, 1e-6, 1-1e-6)))
            except Exception:
                out["freq_brier"] = float("nan")
    else:
        out["n_test_freq"] = 0

    # --- Severity diagnostics
    loss_col = None
    for col in ["loss_estimate_combined_usd", "loss_usd", "loss_estimate_usd", "loss_amount", "monetary_loss", "severity_proxy"]:
        if col in df_test.columns:
            loss_col = col
            break

    if loss_col:
        df_sev = df_test.copy()
        if "loss_confidence" in df_sev.columns:
            df_sev = df_sev[df_sev["loss_confidence"].fillna(0) >= float(min_conf)]
        df_sev = df_sev[df_sev[loss_col].fillna(0) > 0]

        if len(df_sev) > 5:
            y_usd = df_sev[loss_col].astype(float).clip(lower=0).to_numpy()
            y_log = np.log1p(y_usd)
            X_sev = _prep_X(df_sev, _feature_cols(df_sev))
            yhat_log = np.asarray(sev_model.predict(X_sev), dtype=float)

            n = min(len(y_log), len(yhat_log))
            y_log = y_log[:n]; yhat_log = yhat_log[:n]
            y_usd = y_usd[:n]
            yhat_usd = np.expm1(yhat_log)

            out["n_test_sev"]   = int(n)
            out["sev_rmse_log"] = float(np.sqrt(mean_squared_error(y_log, yhat_log)))
            out["sev_mae_log"]  = float(mean_absolute_error(y_log, yhat_log))
            out["sev_rmse_usd"] = float(np.sqrt(mean_squared_error(y_usd, yhat_usd)))
            out["sev_mae_usd"]  = float(mean_absolute_error(y_usd, yhat_usd))

            mask_pos = y_usd > 0
            out["sev_mape"] = float(
                np.mean(np.abs((yhat_usd[mask_pos] - y_usd[mask_pos]) / y_usd[mask_pos])) * 100.0
            ) if np.any(mask_pos) else float("nan")

            # Optional: for continuity with earlier versions
            try:
                out["sev_r2_log"] = float(r2_score(y_log, yhat_log))
            except Exception:
                out["sev_r2_log"] = float("nan")
        else:
            out["n_test_sev"] = 0
    else:
        out["n_test_sev"] = 0

    return out
