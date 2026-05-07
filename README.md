# NGFS Scenario Explorer

A climate finance tool that overlays NGFS Phase V scenarios on investment portfolios and reveals how damage function specification choice shifts estimated drawdowns by 30-50%.

## Key Insight

Climate scenario analysis typically focuses on *which pathway* the world follows (Net Zero vs. Current Policies). But the choice of *damage function* — how temperature translates to economic loss — is equally consequential and far less discussed. This tool makes that visible.

Toggle between three well-known damage functions at any NGFS scenario and watch portfolio drawdown estimates shift by 30-50%:

| Damage Function | At 3°C Anomaly | Character |
|---|---|---|
| Kalkuhl-Wenz (2020) | ~1.8% GDP | Conservative — panel regression on levels |
| Howard-Sterner (2017) | ~10.3% GDP | Meta-analytic — includes catastrophic damages |
| Burke-Hsiang-Miguel (2015) | ~14%+ GDP | Growth-rate channel — compounds over time |

## NGFS Phase V Scenarios

Six pathways across three IAMs (REMIND-MAgPIE, GCAM 6.0, MESSAGEix-GLOBIOM):

- **Net Zero 2050** — orderly, 1.5°C aligned
- **Below 2°C** — orderly, <2°C
- **Divergent Net Zero** — disorderly, regional divergence
- **Delayed Transition** — disorderly, sudden post-2030 tightening
- **NDCs** — hot house, ~2.5-3°C
- **Current Policies** — hot house, ~3+°C

## Setup

```bash
cd ngfs-scenario-explorer
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

### Streamlit App

```bash
streamlit run app/app.py
```

Upload a portfolio CSV or use the included sample. Select scenarios, toggle damage functions, and explore the results.

### Portfolio CSV Format

```csv
ticker,name,sector,weight,market_value
XOM,Exxon Mobil Corp,Energy,0.08,80000
AAPL,Apple Inc,Technology,0.12,120000
```

Required columns: `ticker`, `name`, `sector`, `weight`
Optional: `market_value`

Sector tags are fuzzy-matched to GICS sectors (e.g., "Tech" → Information Technology, "Oil & Gas" → Energy).

### Python API

```python
from ngfs.portfolio import parse_portfolio
from ngfs.iiasa_client import IIASAClient
from ngfs.scenario_engine import compute_all_damage_functions

portfolio = parse_portfolio("data/sample/sample_portfolio.csv")
client = IIASAClient()
trajectories = client.get_temperature_trajectories()

matrix = compute_all_damage_functions(portfolio, trajectories, years=[2050])
comparison = matrix.damage_function_comparison(year=2050)
print(comparison)
```

### Tests

```bash
pytest
```

## Architecture

```
src/ngfs/
  damage_functions.py   # Three damage functions with full implementations
  iiasa_client.py       # IIASA API client (stub with synthetic data)
  portfolio.py          # CSV parsing, GICS sector mapping
  scenario_engine.py    # Core: apply damage fn to scenario → sector drawdowns
  visualization.py      # Plotly charts for Streamlit

app/
  app.py                # Streamlit interface

data/
  cache/                # Parquet-cached IIASA data
  sample/               # Sample portfolio CSV
```

## References

- Kalkuhl, M. & Wenz, L. (2020). *J. Environ. Econ. Manag.*, 103, 102360.
- Burke, M., Hsiang, S. M., & Miguel, E. (2015). *Nature*, 527, 235-239.
- Howard, P. H. & Sterner, T. (2017). *Environ. Resource Econ.*, 68, 197-225.
- NGFS Phase V Scenarios: https://www.ngfs.net/ngfs-scenarios-portal/
