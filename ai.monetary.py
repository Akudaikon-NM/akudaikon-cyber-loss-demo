# ai_monetary.py
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, PoissonRegressor

AI_DEFAULT_ENRICHED = "/mnt/data/akudaikon_incidents_enriched.csv"
AI_DEFAULT_HAI62    = "/mnt/data/akudaikon_joinpack_hai_6_2.csv"

def _feature_cols(df: pd.DataFrame):
    core = [
        "severity_proxy","regulatory_action","country_group",
        "dom_finance","dom_healthcare","dom_transport","dom_social_media",
        "dom_hiring_hr","dom_law_enforcement","dom_education",
        "mod_vision","mod_nlp","mod_recommender","mod_generative","mod_autonomous",
        "life_development","life_deployment","year"
    ]
    hai = [c for c in df.columns if c.startswith("fig_6_2_")]
    return [c for c in core if c in df.columns] + hai

def _prep_X(df: pd.DataFrame, cols):
    X = df[cols].copy()
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0.0)

def load_ai_table(enriched_csv=AI_DEFAULT_ENRICHED, hai62_csv=AI_DEFAULT_HAI62) -> pd.DataFrame:
    df = pd.read_csv(enriched_csv)
    if "year" not in df and "date" in df:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    if hai62_csv:
        hai = pd.read_csv(hai62_csv)
        use_cols = ["incident_id","year"] + [c for c in hai.columns if c not in ("incident_id","year")]
        df = df.merge(hai[use_cols], on=["incident_id","year"], how="left")
    return df

def fit_severity(df: pd.DataFrame, min_conf=0.7):
    use = df.copy()
    if "loss_confidence" in use.columns:
        use = use[use["loss_confidence"].fillna(0) >= float(min_conf)]
    y = np.log1p(use["loss_estimate_combined_usd"].clip(lower=0))
    X = _prep_X(use, _feature_cols(use))
    model = Ridge(alpha=1.0, random_state=42).fit(X, y)
    resid = y - model.predict(X)
    sigma = float(np.std(resid)) if len(resid) else 1.0  # for Monte Carlo
    return model, sigma

def fit_frequency(df: pd.DataFrame):
    y = df["harm_occurred_final"].fillna(0).astype(int)
    X = _prep_X(df, _feature_cols(df))
    model = PoissonRegressor(alpha=1.0, max_iter=1000).fit(X, y)
    return model

def scenario_vector(df: pd.DataFrame, country=None, domains=None, modalities=None):
    m = pd.Series(True, index=df.index)
    if country and "country_group" in df:
        m &= (df["country_group"].fillna("Other/Unknown") == country)
    for d in (domains or []):
        c = f"dom_{d}"
        if c in df: m &= (df[c] == 1)
    for mo in (modalities or []):
        c = f"mod_{mo}"
        if c in df: m &= (df[c] == 1)
    cohort = df[m] if m.any() else df
    X = _prep_X(cohort, _feature_cols(df))
    return X.mean(numeric_only=True).to_frame().T

def simulate_eal_var(freq_model, sev_model, sigma_log, x_row, trials=10000, seed=42):
    rng = np.random.default_rng(int(seed))
    lam = float(np.clip(freq_model.predict(x_row)[0], 1e-8, None))
    mu  = float(sev_model.predict(x_row)[0])  # log-$
    n_events = rng.poisson(lam, size=trials)
    total = np.zeros(trials)
    for i, k in enumerate(n_events):
        if k:
            total[i] = rng.lognormal(mean=mu, sigma=sigma_log, size=int(k)).sum()
    eal   = float(total.mean())
    var95 = float(np.quantile(total, 0.95))
    var99 = float(np.quantile(total, 0.99))
    return eal, var95, var99, total

def lec_dataframe(losses, points=120):
    qs = np.linspace(0, 0.999, points)
    return pd.DataFrame({"loss": np.quantile(losses, qs), "prob_exceed": 1.0-qs})
