🛡️ Akudaikon Cyber-Loss Demo

Monte Carlo cyber-loss model with control ROI, policy (insurance) terms, and CIS-mapped recommendations.
Full Streamlit app with CFO-native metrics: EAL, VaR95/99, CVaR, P(Ruin), ROSI.

🎯 Goal

Treat security like capital allocation. Each control has a cost and changes loss distribution. Quantify ΔEAL and ΔVaR, compare to control cost, and rank by ROSI (return on security investment).

✨ Features
Modeling

Frequency: Poisson or Negative Binomial (over-dispersion). Optional Bayesian update of λ (Gamma–Poisson).

Severity: Spliced Lognormal (body) + GPD (tail); optional records-based mode.

Controls: Four toggles (Server, Media, Error, External) that apply multipliers to λ, p(any), and tail scale; effects can be weighted by your VERIS action/pattern mix.

Policy Layer: Annual aggregate retention / limit / coinsurance to view Insurer Net vs Insured Net losses.

Analytics & Visualization

Metrics: EAL, VaR95/99, CVaR95, Max, P(Ruin), VaR-to-Net-Worth.

Confidence Intervals: Bootstrap CIs for EAL and VaR95.

LEC: Loss Exceedance Curve (log–log), optionally overlay net results after policy terms.

ROI Tools:

Control Isolation (each control on alone) → ΔEAL, Benefit/$, ROSI%.

Marginal ROI (add one control to current bundle).

CIS Recommendations: Optional VERIS→CIS CSV mapping for control suggestions.

Portfolio Batch: Simulate many accounts from CSV; optional frequency correlation.

Exports

CSV: LEC points, isolation ROI, marginal ROI, and Gross/Insurer/Insured metric tables.

JSON: One-click config export for reproducibility.

Cache control: Clear cached sims from the sidebar.

akudaikon-cyber-loss/
├── app.py                 # Streamlit UI
├── engine.py              # Simulation engine (spliced severity, freq models, LEC)
├── controls.py            # Control definitions, costs, multipliers
├── requirements.txt
├── README.md
└── data/
    └── veris_to_cis_lookup.csv   # optional (enables CIS suggestions)
git clone <your-repo-url>
cd akudaikon-cyber-loss

python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

pip install -r requirements.txt
streamlit>=1.37
numpy>=1.26
pandas>=2.0
plotly>=5.22
scipy>=1.11
streamlit run app.py

https://akudaikon-cyber-loss-demo-a3juepz2atljcqbzs8iqqe.streamlit.app/

🧭 Using the App (fast path)

Set trials, net worth, and frequency (λ, p(any)).

Choose severity mode: Monetary (Lognormal+GPD) or Records × $/record.

Pick controls and enter annual costs.

(Optional) Set policy terms (retention/limit/coinsurance).

Click through Results, LEC, Isolation ROI, Marginal ROI.

Download CSVs or config.json for your run.
veris_field,cis_control,cis_title
action.hacking.variety.Exploit vuln,7.0,Continuous Vulnerability Management
action.social.variety.Phishing,14.0,Security Awareness and Skills Training
...
ROSI % = (ΔEAL − Control_Cost) / Control_Cost × 100%
Excess   = max(L − Retention, 0)
Covered  = Coinsurance × Excess
Insurer  = min(Covered, Limit)      # if Limit=0 → unlimited
Insured  = L − Insurer
account_id, net_worth, lam, p_any
🧪 Sanity Checks

With λ≈2 and p(any)≈0.7, expect ~1–2 paid-loss incidents/year.

Raising GPD shape/scale thickens tails → higher VaR99 and a flatter LEC.

Turning on controls should reduce EAL/VaR; ΔEAL should align with multipliers.

Higher retention lowers Insurer Net but raises Insured Net; tight limit caps Insurer Max.

🛠️ Troubleshooting

SyntaxError: 'return' outside function
You likely pasted the incident-severity block at top-level. Ensure the severity code is indented inside simulate_annual_losses(...) and that the final return annual_losses is aligned with the function’s left margin (not outside its scope).

Very wide CIs or unstable tails
Lower GPD shape (ξ) or increase trials and bootstrap reps.

No CIS suggestions appear
Add data/veris_to_cis_lookup.csv (see format above).

🔒 Security & Privacy

All computation runs locally in your session.

No outbound calls from the app.

CSV exports are sanitized against spreadsheet-formula injection.
@software{akudaikon2024,
  title  = {Akudaikon Cyber-Loss Demo: Monte Carlo Risk Quantification},
  author = {Akudaikon},
  year   = {2024},
  url    = {https://github.com/your-org/akudaikon-cyber-loss}
}
Built for security-minded CFOs, CISOs, and risk teams who want defensible, decision-ready numbers.
