# 🛡️ Akudaikon Cyber-Loss Demo

**Production-grade Monte Carlo cyber-loss model with Bayesian inference, control optimization, and AI incident integration.**

Full-featured Streamlit application for quantitative cyber risk analysis with CFO-native metrics (EAL, VaR, ROSI).

---

## 🎯 Goal

**Treat security like capital allocation.** Each control has a cost and reduces expected loss (ΔEAL) and tail risk (ΔVaR). Rank, bundle, and justify spend in CFO-native terms with statistical rigor.

---

## ✨ Features

### Core Capabilities
- ✅ **Monte Carlo engine**: Poisson/Negative Binomial frequency + Spliced Lognormal-GPD severity
- ✅ **Bayesian frequency modeling**: Gamma priors with posterior propagation and credible intervals
- ✅ **Data-driven control effects**: ACTION/PATTERN shares from VERIS with diminishing returns
- ✅ **NAICS presets**: Pre-configured parameters for Finance sub-sectors (Credit Unions, Commercial Banking, FinTech)
- ✅ **Dual risk modes**: Cyber Breach (records-based) and AI Incidents (monetary, AIID integration)

### Analytics & Visualization
- ✅ **Metrics**: EAL, VaR95/99, VaR-to-Net-Worth ratios
- ✅ **Loss Exceedance Curves**: Log-log plots with credible bands
- ✅ **Convergence diagnostics**: Batch CV, ESS, MC standard error, relative error
- ✅ **Sensitivity analysis**: One-at-a-time parameter sweeps (±50%)
- ✅ **Multi-year horizon**: Cumulative risk projections (1-10 years)
- ✅ **Scenario comparison**: All 16 control combinations with ROSI ranking

### Exports & Reporting
- ✅ **CSV downloads**: Annual losses (baseline & controlled)
- ✅ **JSON reports**: Full provenance, configuration, metrics, diagnostics, raw data
- ✅ **OWASP-hardened exports**: CSV injection protection

---

## 🔧 Architecture

```
akudaikon-cyber-loss/
├── app.py                              # Streamlit UI & orchestration
├── engine.py                           # Monte Carlo simulation engine
├── controls.py                         # Security control definitions & effects
├── ai_monetary.py                      # AI incident modeling (optional)
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
├── README.md                           # This file
├── data/                               # Data directory (gitignored)
│   ├── incidents.csv                   # Optional: AIID enriched data
│   └── joinpack_hai_6_2.csv            # Optional: HAI 6.2 taxonomy
└── .streamlit/
    └── secrets.toml                    # Streamlit secrets (gitignored)
```

### Key Modules

**`engine.py`** - Monte Carlo simulation engine
- Frequency models: Poisson, Negative Binomial
- Severity model: Spliced Lognormal-GPD (body + heavy tail)
- Bayesian inference: Gamma prior → posterior for λ
- LEC generation with credible bands

**`controls.py`** - Security control framework
- 4 control categories mapped to CIS Controls & NIST CSF
- Baseline effects with diminishing returns
- Cost tracking and ROSI calculation

**`ai_monetary.py`** - AI incident risk modeling
- Fits Ridge regression (severity) and Poisson regression (frequency) from AIID data
- Scenario builder with domain/modality/geography filters
- Strong model diagnostics (R², AUC, Brier, RMSE/MAE)

---

## 📊 How It Works

### Cyber Breach Mode (Records-Based)

1. **Frequency**: Annual incidents ~ Poisson(λ) or NegBin(λ, r)
2. **Loss probability**: Each incident produces loss with probability p(any)
3. **Severity**: Spliced distribution
   - **Body** (90%): Lognormal (typical breaches: 5-20% of records)
   - **Tail** (10%): Generalized Pareto (catastrophic: >50% of records)
4. **Controls**: Multiply λ, p(any), and tail scale by control effects
5. **Outputs**: EAL, VaR95/99, LEC, ROSI

### AI Incidents Mode (Monetary)

1. **Train models** on AIID data:
   - Frequency: Poisson regression on harm occurrence
   - Severity: Ridge regression on log(loss_usd)
2. **Build scenario** vector from domain/modality/geography filters
3. **Simulate** losses using trained models
4. **Output** EAL, VaR95/99, LEC for AI-specific risks

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd akudaikon-cyber-loss

# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 📖 User Guide

### Quick Demo (30 seconds)

1. **Select risk mode**: "Cyber Breach (records-based)"
2. **Apply NAICS preset**: "522130 — Credit Unions"
3. **Toggle controls**: Enable "External / MFA & perimeter"
4. **Click "Run simulation"**
5. **Review outputs**: EAL delta, VaR reduction, ROSI

### Advanced Usage

#### Bayesian Frequency Modeling

**Use case**: You have historical incident data and want to propagate parameter uncertainty.

1. Open **"Advanced frequency"** in sidebar
2. Check **"Bayesian lambda (Gamma prior + your data)"**
3. Enter your data:
   - **k incidents observed**: e.g., 5
   - **T observation years**: e.g., 2.5
4. Click **"Apply prior α₀=λ̂·w, β₀=w"** to seed weak prior
5. Run simulation → outputs include credible intervals

**Interpretation**: Instead of point estimate λ=2.0, you'll see EAL with 95% CI reflecting uncertainty in λ.

#### Data-Driven Control Effects

**Use case**: Customize control effectiveness using your VERIS data.

1. Open **"Data-driven control effects (shares)"**
2. Choose **"Upload CSVs"**
3. Upload 2 CSVs:
   - **Action shares**: columns `[category, share]`, e.g., `Hacking, 0.40`
   - **Pattern shares**: columns `[category, share]`, e.g., `Privilege Misuse, 0.35`
4. Controls now weighted by your ACTION/PATTERN distribution

#### Sensitivity Analysis

**Use case**: Show board which parameter drives tail risk.

1. Run baseline simulation
2. Open **"Run sensitivity analysis"**
3. Select parameter: `lam`, `p_any`, or `r`
4. Click **"Run sensitivity"**
5. Review chart showing EAL/VaR95 vs parameter value

#### Scenario Comparison

**Use case**: Find optimal control bundle for your budget.

1. Run baseline simulation
2. Click **"Compare all control combinations"**
3. Review 16 scenarios ranked by ROSI
4. Identify highest ROSI within budget constraints

---

## 🎛️ Controls & Mappings

### Control Categories

| Control | CIS Controls | NIST CSF | Typical Cost | Effect |
|---------|--------------|----------|--------------|--------|
| **Server** | 4, 7, 8, 10, 12 | PR.PT, DE.CM | $80k/yr | -25% λ, -15% tail |
| **Media** | 3, 4, 6, 8, 11, 15 | PR.DS, RC.RP | $90k/yr | -40% tail |
| **Error** | 4, 5, 7, 8, 12, 13, 16, 17 | ID.GV, PR.AT, RS.RP | $60k/yr | -20% λ, -10% p(any) |
| **External** | 5, 6, 8, 9, 10, 12, 13, 15, 16 | PR.AC, DE.AE, RS.AN | $100k/yr | -35% λ, -15% p(any) |

### Diminishing Returns

Multiple controls have **interaction discount** (15% per additional control):
- 1 control: Full effect
- 2 controls: 85% of additive effect
- 3 controls: 75% of additive effect
- 4 controls: 67% of additive effect

**Floors** (regulatory realism):
- λ multiplier ≥ 0.30 (max 70% reduction)
- p(any) multiplier ≥ 0.40 (max 60% reduction)
- Tail scale multiplier ≥ 0.50 (max 50% reduction)

---

## 📈 Understanding Outputs

### Expected Annual Loss (EAL)
Mean loss across all trials. **This is your expected cost.**
- **Baseline**: No controls
- **Controlled**: With selected controls
- **Delta**: Risk reduction from controls

### Value at Risk (VaR)
- **VaR95**: 95th percentile loss. 5% chance losses exceed this.
- **VaR99**: 99th percentile loss. 1% chance losses exceed this.
- Used for capital planning, board reporting, and regulatory compliance.

### VaR to Net Worth Ratio
Board/regulator metric showing tail risk as % of net worth.
- **NCUA**: Monitors for credit unions
- **OCC**: Monitors for national banks
- **Typical threshold**: Keep VaR99 < 5% of net worth

### ROSI (Return on Security Investment)
```
ROSI = (Delta_EAL - Control_Cost) / Control_Cost × 100%
```

- **Positive ROSI**: Control saves more than it costs (strong business case)
- **Negative ROSI**: Control costs more than saves (may still be justified for compliance/tolerance)
- **Isolated ROSI**: Control tested alone
- **Marginal ROSI**: Control added to current bundle (shows diminishing returns)

### Loss Exceedance Curve (LEC)
Log-log plot of P(Loss ≥ x) vs x.
- **Steeper slope**: More concentrated risk
- **Longer tail**: More extreme events
- **Credible bands** (if Bayesian): Show parameter uncertainty

---

## 🔬 Advanced Features

### Negative Binomial Overdispersion

**Use case**: Your incident counts have variance > mean (overdispersion).

Enable **"Use Negative Binomial"** in Advanced frequency.
- **r = ∞**: Poisson (variance = mean)
- **r = 1.5**: Moderate overdispersion (variance = mean + mean²/1.5)
- **r = 0.5**: Heavy overdispersion (variance = mean + 2×mean²)

### Multi-Year Horizon

Simulate cumulative losses over 1-10 years.
- Each year draws new incidents independently
- Useful for multi-year capital planning
- Reports final-year VaR95 (cumulative risk)

### Convergence Diagnostics

App checks simulation quality via:
- **Batch CV**: Coefficient of variation across trial batches (target < 5%)
- **ESS**: Effective sample size accounting for autocorrelation
- **MC SE**: Monte Carlo standard error of EAL
- **Relative error**: MC SE / EAL (target < 5%)

**If convergence fails**: Increase trials from 50k → 100k → 250k.

---

## 💡 Production Deployment Tips

### 1. Replace Synthetic Models

**Cyber Breach mode**: Calibrate frequency/severity from your data
```python
# Fit λ from your incident response metrics
lambda_rate = your_incidents_per_year

# Fit cost_per_record from actual breach costs
cost_per_record = (forensics + legal + notification + ...) / records_affected
```

**AI Incidents mode**: Train on your AIID data
```bash
# Place CSVs in data/ directory
cp your_incidents.csv data/incidents.csv
cp your_hai_taxonomy.csv data/joinpack_hai_6_2.csv

# App automatically loads and trains models
```

### 2. Calibrate Control Effects

Replace baseline effects in `controls.py` with vendor studies or historical data:
```python
# Example: Your EDR reduced incidents by 40% after deployment
if ctrl.server:
    lam_mult *= 0.60  # Updated from 0.75 based on your data
```

### 3. Configure for Your Sector

Add your NAICS preset in `app.py`:
```python
NAICS_FINANCE_PRESETS = {
    "Your Industry": {
        "lambda": 0.XX,
        "records_cap": XXXXX,
        "cost_per_record": XXX.X,
        "net_worth": XXXXXXXXX.X
    }
}
```

### 4. Set Up Data Pipeline

For AI Incidents mode, automate data refresh:
```bash
# Cron job to refresh AIID data weekly
0 0 * * 0 python scripts/fetch_aiid.py
```

### 5. Security Hardening

- Store secrets in `.streamlit/secrets.toml` (gitignored)
- Enable authentication if deploying publicly
- Use HTTPS for production deployment
- Review OWASP Top 10 for web apps

---

## 🔒 Security & Privacy

- ✅ **Local execution**: All simulations run on your infrastructure
- ✅ **No external calls**: No data sent to third parties
- ✅ **CSV injection protection**: All exports sanitized (OWASP)
- ✅ **Input validation**: File upload size limits and type checks
- ✅ **Secrets management**: Use `.streamlit/secrets.toml` for sensitive configs

---

## 📊 Sample Outputs

### Baseline vs Controlled (Credit Union)

| Metric | Baseline | With External Control | Delta |
|--------|----------|---------------------|-------|
| **EAL** | $2.1M | $1.4M | -$700k |
| **VaR95** | $8.5M | $5.8M | -$2.7M |
| **VaR99** | $15.2M | $10.3M | -$4.9M |
| **Control Cost** | $0 | $100k | +$100k |
| **ROSI** | N/A | 600% | +600% |

**Interpretation**: External control (MFA + perimeter) costs $100k but reduces EAL by $700k → 600% annualized ROSI.

---

## 🛠️ Troubleshooting

### "Simulation may not be converged"
**Solution**: Increase trials from 50k → 100k in sidebar.

### "Only X loss observations - results may be unreliable"
**Solution** (AI mode): Add more AIID data or lower `min_conf` threshold.

### "High relative error"
**Solution**: Increase trials or reduce tail heaviness (lower GPD shape parameter).

### "Import error: ai_monetary not found"
**Solution**: Install scikit-learn: `pip install scikit-learn>=1.3.0`

---

## 📝 Citation

If you use this tool in research or publications:

```bibtex
@software{akudaikon2024,
  title={Akudaikon Cyber-Loss Demo: Monte Carlo Risk Quantification},
  author={Akudaikon},
  year={2024},
  url={https://github.com/your-org/akudaikon-cyber-loss}
}
```

---

## 📧 Support

- **GitHub Issues**: [your-repo/issues]
- **Documentation**: [your-docs-url]
- **Email**: [support@akudaikon.com]

---

## 🙏 Acknowledgments

- **VERIS Community**: For ACTION/PATTERN taxonomy
- **AI Incident Database (AIID)**: For incident data and enrichment
- **CIS Controls**: For control framework mapping
- **NIST CSF**: For risk management framework
- **Streamlit**: For the amazing app framework

---

## 📜 License

[Your License - e.g., MIT, Apache 2.0, Proprietary]

---

**Built with ❤️ for security-minded CFOs, CISOs, and risk practitioners**

*Disclaimer: This tool provides quantitative risk estimates for planning purposes. Actual losses may vary. Consult with legal, accounting, and cybersecurity professionals before making business decisions.*
