# ai_monetary.py
"""
AI Incident Risk Modeling
-------------------------
Fits frequency (Poisson) and severity (Ridge on log-losses) models from AIID data,
then simulates annual losses via Monte Carlo.

Improvements:
- Robust column name handling (tries multiple alternatives)
- Data validation (minimum observations, positive losses)
- Safety clipping on model predictions (λ, μ, σ)
- Better LEC tail resolution
- Model diagnostics (R², RMSE)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.metrics import r2_score, mean_squared_error

AI_DEFAULT_ENRICHED = "/mnt/data/akudaikon_incidents_enriched.csv"
AI_DEFAULT_HAI62    = "/mnt/data/akudaikon_joinpack_hai_6_2.csv"


# -----------------------------------------------------------------------------
# Feature engineering helpers
# -----------------------------------------------------------------------------
def _feature_cols(df: pd.DataFrame) -> list:
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
    return [c for c in core if c in df.columns] + hai


def _prep_X(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Extract feature columns, coerce to numeric, and fill NaN with 0.
    """
    X = df[cols].copy()
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0.0)


# -----------------------------------------------------------------------------
# Data loading with robust column handling
# -----------------------------------------------------------------------------
def load_ai_table(enriched_csv=AI_DEFAULT_ENRICHED, hai62_csv=AI_DEFAULT_HAI62) -> pd.DataFrame:
    """
    Load and merge enriched incidents CSV with HAI 6.2 join-pack CSV.
    Handles common column name variations and missing columns gracefully.
    
    Returns:
        pd.DataFrame: Merged dataset with incident_id, year, and feature columns.
    """
    df = pd.read_csv(enriched_csv)
    
    # Extract year from date columns if not present
    if "year" not in df.columns:
        for date_col in ["date", "published", "incident_date", "event_date", "report_date"]:
            if date_col in df.columns:
                df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
                if df["year"].notna().any():
                    break
        
        # If still no year, create a dummy column
        if "year" not in df.columns:
            df["year"] = 2020  # Default fallback
    
    # Merge with HAI 6.2 data if provided
    if hai62_csv:
        try:
            hai = pd.read_csv(hai62_csv)
            
            # Handle incident_id vs id column name variation
            if "incident_id" not in hai.columns and "id" in hai.columns:
                hai = hai.rename(columns={"id": "incident_id"})
            
            # Ensure both dataframes have incident_id for merging
            if "incident_id" in df.columns and "incident_id" in hai.columns:
                use_cols = ["incident_id"] + [c for c in hai.columns if c not in ("incident_id",)]
                df = df.merge(hai[use_cols], on="incident_id", how="left", suffixes=("", "_hai"))
            else:
                print("⚠️ Warning: Cannot merge HAI data - 'incident_id' column missing")
        except Exception as e:
            print(f"⚠️ Warning: Could not load HAI data: {e}")
    
    return df


# -----------------------------------------------------------------------------
# Severity model (Ridge regression on log-losses)
# -----------------------------------------------------------------------------
def fit_severity(df: pd.DataFrame, min_conf=0.7):
    """
    Fit a Ridge regression model to predict log-transformed monetary losses.
    
    Args:
        df: DataFrame with incident features and loss columns
        min_conf: Minimum loss_confidence threshold (0-1) to include observations
    
    Returns:
        tuple: (trained_model, residual_std_dev)
        
    Raises:
        ValueError: If insufficient valid loss observations (<10)
    """
    use = df.copy()
    
    # Find loss column (try multiple common names)
    loss_col = None
    for col in ["loss_estimate_combined_usd", "loss_usd", "loss_estimate_usd", 
                "loss_amount", "monetary_loss", "severity_proxy"]:
        if col in use.columns:
            loss_col = col
            break
    
    if loss_col is None:
        raise ValueError(
            "No loss column found. Tried: loss_estimate_combined_usd, loss_usd, "
            "loss_estimate_usd, loss_amount, monetary_loss, severity_proxy"
        )
    
    # Rename to standard column name
    if loss_col != "loss_estimate_combined_usd":
        use["loss_estimate_combined_usd"] = use[loss_col]
    
    # Filter by confidence threshold
    if "loss_confidence" in use.columns:
        use = use[use["loss_confidence"].fillna(0) >= float(min_conf)]
    
    # Filter to positive losses only
    use = use[use["loss_estimate_combined_usd"].fillna(0) > 0]
    
    # Validate sufficient data
    if len(use) < 10:
        raise ValueError(
            f"Insufficient data: only {len(use)} valid loss observations after filtering "
            f"(minimum 10 required). Try lowering min_conf or checking data quality."
        )
    
    # Prepare features and target
    y = np.log1p(use["loss_estimate_combined_usd"].clip(lower=0))
    X = _prep_X(use, _feature_cols(use))
    
    # Sanity check alignment
    if len(X) != len(y):
        raise ValueError(f"Feature matrix ({len(X)}) and target ({len(y)}) size mismatch")
    
    # Fit Ridge model
    model = Ridge(alpha=1.0, random_state=42).fit(X, y)
    
    # Compute residual standard deviation for Monte Carlo
    resid = y - model.predict(X)
    sigma = float(np.std(resid)) if len(resid) > 1 else 1.0
    
    return model, sigma


# -----------------------------------------------------------------------------
# Frequency model (Poisson regression)
# -----------------------------------------------------------------------------
def fit_frequency(df: pd.DataFrame):
    """
    Fit a Poisson regression model to predict incident frequency (harm occurrence).
    
    Args:
        df: DataFrame with incident features and harm indicator column
    
    Returns:
        PoissonRegressor: Trained model
        
    Raises:
        ValueError: If no harm indicator column found or insufficient positive cases (<5)
    """
    # Find harm indicator column (try multiple common names)
    harm_col = None
    for col in ["harm_occurred_final", "harm_occurred", "incident_occurred", 
                "is_incident", "harm_binary"]:
        if col in df.columns:
            harm_col = col
            break
    
    if harm_col is None:
        raise ValueError(
            "No harm/incident indicator column found. Tried: harm_occurred_final, "
            "harm_occurred, incident_occurred, is_incident, harm_binary"
        )
    
    # Prepare binary target (0/1)
    y = df[harm_col].fillna(0).astype(int).clip(0, 1)
    X = _prep_X(df, _feature_cols(df))
    
    # Validate alignment
    if len(X) != len(y):
        raise ValueError(f"Feature/target size mismatch: {len(X)} vs {len(y)}")
    
    # Check for sufficient positive cases
    n_positive = int(y.sum())
    if n_positive < 5:
        raise ValueError(
            f"Insufficient positive harm incidents: {n_positive} (minimum 5 required). "
            f"Check data quality or harm indicator column."
        )
    
    # Fit Poisson model
    model = PoissonRegressor(alpha=1.0, max_iter=1000).fit(X, y)
    return model


# -----------------------------------------------------------------------------
# Scenario vector construction
# -----------------------------------------------------------------------------
def scenario_vector(df: pd.DataFrame, country=None, domains=None, modalities=None):
    """
    Build a feature vector for a specific scenario by filtering and averaging.
    
    Args:
        df: Full incident dataset
        country: Country group to filter (e.g., "USA", "EU")
        domains: List of domain tags (e.g., ["finance", "healthcare"])
        modalities: List of modality tags (e.g., ["vision", "nlp"])
    
    Returns:
        pd.DataFrame: Single-row feature vector (mean of matching incidents)
    """
    mask = pd.Series(True, index=df.index)
    
    # Filter by country
    if country and "country_group" in df.columns:
        mask &= (df["country_group"].fillna("Other/Unknown") == country)
    
    # Filter by domains
    for d in (domains or []):
        col = f"dom_{d}"
        if col in df.columns:
            mask &= (df[col] == 1)
    
    # Filter by modalities
    for m in (modalities or []):
        col = f"mod_{m}"
        if col in df.columns:
            mask &= (df[col] == 1)
    
    # Fallback to full dataset if no matches
    cohort = df[mask] if mask.any() else df
    
    # Return mean feature vector as single-row DataFrame
    X = _prep_X(cohort, _feature_cols(df))
    return X.mean(numeric_only=True).to_frame().T


# -----------------------------------------------------------------------------
# Monte Carlo simulation with safety bounds
# -----------------------------------------------------------------------------
def simulate_eal_var(freq_model, sev_model, sigma_log, x_row, trials=10000, seed=42):
    """
    Simulate annual losses via Monte Carlo using fitted frequency and severity models.
    
    Args:
        freq_model: Trained PoissonRegressor for incident frequency
        sev_model: Trained Ridge model for log-severity
        sigma_log: Residual std dev from severity fit (for lognormal sampling)
        x_row: Single-row DataFrame with scenario features
        trials: Number of Monte Carlo runs
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (EAL, VaR95, VaR99, annual_losses_array)
        
    Raises:
        ValueError: If all simulated losses are invalid (inf/nan)
    """
    rng = np.random.default_rng(int(seed))
    
    # Predict frequency (lambda) with safety bounds
    lam_raw = freq_model.predict(x_row)[0]
    lam = float(np.clip(lam_raw, 0.001, 100.0))  # Reasonable annual incident rate
    
    # Predict log-severity (mu) with safety bounds
    mu_raw = sev_model.predict(x_row)[0]
    mu = float(np.clip(mu_raw, 0.0, 30.0))  # ~$1 to $10^13 when exponentiated
    
    # Clip sigma to prevent extreme variance
    sigma_log = float(np.clip(sigma_log, 0.1, 5.0))
    
    # Simulate annual losses
    n_events = rng.poisson(lam, size=trials)
    total = np.zeros(trials, dtype=float)
    
    for i, k in enumerate(n_events):
        if k > 0:
            # Generate k severity samples and sum for annual loss
            severities = rng.lognormal(mean=mu, sigma=sigma_log, size=int(k))
            total[i] = float(severities.sum())
    
    # Remove invalid values
    total = total[np.isfinite(total)]
    if len(total) == 0:
        raise ValueError(
            "All simulated losses are invalid (inf/nan). Check model predictions: "
            f"lambda={lam:.4f}, mu={mu:.4f}, sigma={sigma_log:.4f}"
        )
    
    # Compute risk metrics
    eal = float(np.mean(total))
    var95 = float(np.percentile(total, 95))
    var99 = float(np.percentile(total, 99))
    
    return eal, var95, var99, total


# -----------------------------------------------------------------------------
# Loss Exceedance Curve with better tail resolution
# -----------------------------------------------------------------------------
def lec_dataframe(losses, points=200):
    """
    Generate Loss Exceedance Curve (LEC) with denser sampling in the tail.
    
    Args:
        losses: Array of annual loss values
        points: Total number of points on the curve
    
    Returns:
        pd.DataFrame: Two columns (loss, prob_exceed) for plotting
    """
    losses = np.asarray(losses, dtype=float)
    losses = losses[np.isfinite(losses) & (losses >= 0)]
    
    if len(losses) == 0:
        return pd.DataFrame({"loss": [0.0], "prob_exceed": [1.0]})
    
    # Use log-spaced quantiles to get better tail resolution
    qs = np.concatenate([
        np.linspace(0, 0.90, points // 2),      # Linear spacing in body (0-90th percentile)
        np.linspace(0.90, 0.999, points // 2)   # Denser spacing in tail (90-99.9th percentile)
    ])
    
    loss_levels = np.percentile(losses, qs * 100)
    exceed_probs = 1.0 - qs
    
    return pd.DataFrame({
        "loss": loss_levels,
        "prob_exceed": exceed_probs
    })


# -----------------------------------------------------------------------------
# Model diagnostics for validation
# -----------------------------------------------------------------------------
def model_diagnostics(freq_model, sev_model, df_test: pd.DataFrame, min_conf=0.7):
    """
    Compute R² and RMSE for trained models on a test set.
    
    Args:
        freq_model: Trained PoissonRegressor
        sev_model: Trained Ridge model
        df_test: Test dataframe with true labels
        min_conf: Minimum confidence threshold for severity test set
    
    Returns:
        dict: Diagnostic metrics (freq_r2, sev_r2, sev_rmse_log, n_test_freq, n_test_sev)
    """
    # Frequency diagnostics
    harm_col = None
    for col in ["harm_occurred_final", "harm_occurred", "incident_occurred"]:
        if col in df_test.columns:
            harm_col = col
            break
    
    if harm_col:
        y_freq_test = df_test[harm_col].fillna(0).astype(int).clip(0, 1)
        X_freq_test = _prep_X(df_test, _feature_cols(df_test))
        freq_pred = freq_model.predict(X_freq_test)
        freq_r2 = r2_score(y_freq_test, freq_pred)
        n_test_freq = len(y_freq_test)
    else:
        freq_r2 = np.nan
        n_test_freq = 0
    
    # Severity diagnostics
    loss_col = None
    for col in ["loss_estimate_combined_usd", "loss_usd", "loss_estimate_usd"]:
        if col in df_test.columns:
            loss_col = col
            break
    
    if loss_col:
        df_sev = df_test.copy()
        if "loss_confidence" in df_sev.columns:
            df_sev = df_sev[df_sev["loss_confidence"].fillna(0) >= min_conf]
        df_sev = df_sev[df_sev[loss_col].fillna(0) > 0]
        
        if len(df_sev) > 5:
            y_sev_test = np.log1p(df_sev[loss_col].clip(lower=0))
            X_sev_test = _prep_X(df_sev, _feature_cols(df_sev))
            sev_pred = sev_model.predict(X_sev_test)
            sev_r2 = r2_score(y_sev_test, sev_pred)
            sev_rmse = np.sqrt(mean_squared_error(y_sev_test, sev_pred))
            n_test_sev = len(y_sev_test)
        else:
            sev_r2 = np.nan
            sev_rmse = np.nan
            n_test_sev = 0
    else:
        sev_r2 = np.nan
        sev_rmse = np.nan
        n_test_sev = 0
    
    return {
        "freq_r2": float(freq_r2) if not np.isnan(freq_r2) else None,
        "sev_r2": float(sev_r2) if not np.isnan(sev_r2) else None,
        "sev_rmse_log": float(sev_rmse) if not np.isnan(sev_rmse) else None,
        "n_test_freq": int(n_test_freq),
        "n_test_sev": int(n_test_sev)
    }
