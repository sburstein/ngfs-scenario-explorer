"""
NGFS Scenario Explorer — Streamlit application.

Upload a portfolio CSV, select NGFS scenarios and IAM models, toggle
between damage functions, and see how estimated drawdowns shift by
30-50% depending on specification choice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ngfs.damage_functions import DamageFunctionName, get_damage_function, compare_at_temperature
from ngfs.iiasa_client import NGFS_SCENARIOS, NGFS_MODELS
from ngfs.portfolio import parse_portfolio, Portfolio
from ngfs.scenario_engine import (
    NGFS_TEMPERATURE_PATHWAYS,
    build_temperature_trajectories,
    compute_drawdowns,
    compute_all_damage_functions,
)
from ngfs.visualization import (
    drawdown_heatmap,
    sensitivity_bands,
    damage_function_comparison,
    temperature_trajectory_chart,
    portfolio_summary_chart,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NGFS Scenario Explorer",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("NGFS Scenario Explorer")
st.markdown(
    "Overlay NGFS Phase V climate scenarios on your portfolio and toggle "
    "between damage functions to see how specification choice shifts "
    "estimated drawdowns by **30-50%**."
)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    # Portfolio upload
    st.subheader("Portfolio")
    upload_mode = st.radio(
        "Portfolio source",
        ["Sample portfolio", "Upload CSV"],
        index=0,
    )

    portfolio: Portfolio | None = None
    sample_path = Path(__file__).parent.parent / "data" / "sample" / "sample_portfolio.csv"

    if upload_mode == "Sample portfolio":
        if sample_path.exists():
            portfolio = parse_portfolio(sample_path)
            st.success(f"Loaded sample: {len(portfolio.positions)} positions")
        else:
            st.error("Sample portfolio not found")
    else:
        uploaded_file = st.file_uploader(
            "Upload portfolio CSV",
            type=["csv"],
            help="Required columns: ticker, name, sector, weight. Optional: market_value",
        )
        if uploaded_file is not None:
            # Save temp file and parse
            tmp_path = Path("/tmp/uploaded_portfolio.csv")
            tmp_path.write_bytes(uploaded_file.getvalue())
            try:
                portfolio = parse_portfolio(tmp_path)
                st.success(f"Parsed {len(portfolio.positions)} positions")
            except Exception as e:
                st.error(f"Error parsing portfolio: {e}")

    st.divider()

    # Scenario selection
    st.subheader("Scenarios")
    selected_scenarios = st.multiselect(
        "NGFS scenarios",
        options=NGFS_SCENARIOS,
        default=NGFS_SCENARIOS,
    )

    selected_model = st.selectbox(
        "IAM model",
        options=NGFS_MODELS,
        index=0,
    )

    st.divider()

    # Damage function
    st.subheader("Damage Function")
    damage_fn_options = {
        "Kalkuhl-Wenz (2020)": DamageFunctionName.KALKUHL_WENZ,
        "Burke-Hsiang-Miguel (2015)": DamageFunctionName.BURKE_HSIANG_MIGUEL,
        "Howard-Sterner (2017)": DamageFunctionName.HOWARD_STERNER,
    }
    selected_dmg_fn_label = st.selectbox(
        "Primary damage function",
        options=list(damage_fn_options.keys()),
        index=2,
        help="Select the damage function for the heatmap view. "
             "The sensitivity analysis always shows all three.",
    )
    selected_dmg_fn = damage_fn_options[selected_dmg_fn_label]

    compare_all = st.checkbox(
        "Compare all damage functions",
        value=True,
        help="Show side-by-side comparison across all three specifications.",
    )

    st.divider()

    # Year selection
    st.subheader("Time Horizon")
    analysis_year = st.slider(
        "Analysis year",
        min_value=2025,
        max_value=2100,
        value=2050,
        step=5,
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

if portfolio is None:
    st.info("Select or upload a portfolio to begin analysis.")
    st.stop()


# Build temperature trajectories from built-in NGFS pathway data
@st.cache_data
def load_temperature_trajectories(model: str, scenarios: list[str]) -> pd.DataFrame:
    return build_temperature_trajectories(model=model, scenarios=scenarios)


temp_trajectories = load_temperature_trajectories(selected_model, selected_scenarios)


# --- Portfolio summary ---
st.header("Portfolio Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Positions", len(portfolio.positions))
with col2:
    st.metric("Sectors", portfolio.sector_count)
with col3:
    st.metric("Total Weight", f"{portfolio.total_weight:.1%}")

# Sector weight breakdown
sector_df = pd.DataFrame(
    [{"Sector": k.value, "Weight": v} for k, v in portfolio.sector_weights.items()]
).sort_values("Weight", ascending=False)

st.dataframe(
    sector_df.style.format({"Weight": "{:.1%}"}),
    use_container_width=True,
    hide_index=True,
)

# Portfolio allocation treemap
st.plotly_chart(portfolio_summary_chart(portfolio), use_container_width=True)


# --- Damage function comparison chart ---
st.header("Damage Functions")
st.markdown(
    f"Comparing three damage functions at the {analysis_year} temperature for each scenario. "
    "The spread between functions demonstrates the 30-50% specification variance."
)
st.plotly_chart(damage_function_comparison(), use_container_width=True)


# --- Temperature trajectories ---
st.header("Temperature Trajectories")
st.plotly_chart(
    temperature_trajectory_chart(temp_trajectories, model=selected_model),
    use_container_width=True,
)


# --- Drawdown analysis ---
st.header("Drawdown Analysis")

# Round analysis year to nearest available
available_years = sorted(temp_trajectories["year"].unique())
target_year = min(available_years, key=lambda y: abs(y - analysis_year))

if compare_all:
    matrix = compute_all_damage_functions(
        portfolio=portfolio,
        temperature_trajectory=temp_trajectories,
        years=[target_year],
    )
else:
    matrix = compute_drawdowns(
        portfolio=portfolio,
        temperature_trajectory=temp_trajectories,
        damage_function=selected_dmg_fn,
        years=[target_year],
    )

tab1, tab2, tab3 = st.tabs(["Sector Heatmap", "Sensitivity Bands", "Raw Data"])

with tab1:
    st.subheader(f"Sector Drawdowns — {selected_dmg_fn_label} ({target_year})")
    fig_hm = drawdown_heatmap(
        matrix,
        year=target_year,
        model=selected_model,
        damage_function=selected_dmg_fn_label,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

with tab2:
    if compare_all:
        st.subheader(f"Damage Function Sensitivity ({target_year})")
        fig_bands = sensitivity_bands(matrix, year=target_year, model=selected_model)
        st.plotly_chart(fig_bands, use_container_width=True)

        # Show the comparison table
        comparison = matrix.damage_function_comparison(year=target_year)
        if not comparison.empty:
            st.subheader("Drawdown Comparison Table")
            st.dataframe(
                comparison.style.format(
                    {col: "{:.2%}" for col in comparison.columns
                     if col not in ["scenario", "model", "year", "spread_pct"]},
                    na_rep="-",
                ),
                use_container_width=True,
            )
    else:
        st.info("Enable 'Compare all damage functions' in the sidebar to see sensitivity bands.")

with tab3:
    results_df = matrix.to_dataframe()
    if not results_df.empty:
        st.dataframe(
            results_df.style.format(
                {
                    "temperature_C": "{:.2f}",
                    "portfolio_weight": "{:.1%}",
                    "macro_damage": "{:.4f}",
                    "sector_drawdown": "{:.4f}",
                    "weighted_contribution": "{:.4f}",
                    "portfolio_drawdown": "{:.4f}",
                }
            ),
            use_container_width=True,
        )

        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name=f"ngfs_drawdowns_{target_year}.csv",
            mime="text/csv",
        )

# --- Quick reference at temperature ---
st.header("Quick Reference: Damages at Temperature")
ref_temp = st.slider(
    "Temperature anomaly (°C)",
    min_value=0.0,
    max_value=6.0,
    value=2.5,
    step=0.1,
    key="ref_temp",
)

comparison = compare_at_temperature(ref_temp)
ref_cols = st.columns(len(comparison))
for col, (name, damage) in zip(ref_cols, comparison.items()):
    with col:
        st.metric(name, f"{damage:.2%} GDP loss")
