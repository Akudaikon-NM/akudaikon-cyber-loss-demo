# akudaikon-cyber-loss-demo
Minimal Streamlit demo: Monte Carlo cyber-loss model with control toggles, EAL/VaR, and LEC. Defaults are synthetic; replace multipliers and severity with your trained models later.
# akudaikon-cyber-loss-demo

Minimal Streamlit demo: Monte Carlo cyber-loss model with control toggles, EAL/VaR, and an LEC.  
Defaults are synthetic; replace multipliers and severity with your trained models later.

## How it works (short)
- **Frequency** ~ Poisson(λ)
- **Severity** ~ heavy-tailed proxy = (Beta fraction of customers) × ($/customer), capped by customers
- **Controls** apply likelihood/severity multipliers (and optional annual cost)
- **Outputs**: Expected Annual Loss (EAL), VaR95/99, Loss Exceedance Curve (LEC), CSV export

## Inputs (left panel)
| Field | What it means | Tips |
|---|---|---|
| **NAICS sector** | Sets starter priors (breach probability & exposure records) | Swap these with your sector priors when ready |
| **Simulation trials** | Monte Carlo runs (e.g., 10k) | 10k is fine for demos; increase for smoother tails |
| **Customers / records (cap)** | Population at risk (ceiling for records exposed) | Pull from 5300 Call Report or internal systems |
| **Net Worth ($)** | For impact ratio (VaR / Net Worth) | Useful for board reporting |
| **Annual incident rate λ** | Events/year (Poisson mean) | Use your IRP metrics or industry baselines |
| **Cost per customer ($)** | Per-record severity scalar | Replace with your loss model coefficients |
| **Controls (checkboxes)** | Applies multipliers and optional cost | Replace with calibrated control effect sizes |
| **Control cost overrides** | Annual cost of each control | Enables ROI comparisons |
| **Random seed** | Reproducibility | Fix for repeatable charts/exports |

## Outputs
- **EAL (baseline vs controlled)**: expected annualized loss in $
- **VaR95/99**: tail loss percentiles
- **LEC**: `P(Loss ≥ x)` vs `x` (log-log)
- **CSV exports**: baseline and controlled sampled losses

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
