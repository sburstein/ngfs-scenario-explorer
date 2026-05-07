"""
Visualization module for NGFS scenario analysis.

Produces three main chart types:
  1. Drawdown heatmap: scenario x sector matrix showing severity
  2. Sensitivity bands: how drawdowns shift across damage functions
  3. Damage function comparison: direct overlay of the three functions

All charts are built with Plotly for interactive Streamlit embedding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ngfs.damage_functions import DamageFunctionName, get_damage_function
from ngfs.portfolio import Portfolio
from ngfs.scenario_engine import DrawdownMatrix


def drawdown_heatmap(
    matrix: DrawdownMatrix,
    year: int = 2050,
    model: str | None = None,
    damage_function: str | None = None,
) -> go.Figure:
    """
    Create a heatmap of sector drawdowns by scenario.

    Rows: NGFS scenarios
    Columns: GICS sectors
    Values: Sector-level drawdown percentage

    Args:
        matrix: Computed drawdown matrix.
        year: Target year for the snapshot.
        model: Filter to a specific IAM model (uses first if None).
        damage_function: Filter to a specific damage function (uses first if None).

    Returns:
        Plotly Figure with annotated heatmap.
    """
    df = matrix.to_dataframe()

    # Apply filters
    df = df[df["year"] == year]
    if model:
        df = df[df["model"] == model]
    elif "model" in df.columns:
        df = df[df["model"] == df["model"].iloc[0]]
    if damage_function:
        df = df[df["damage_function"] == damage_function]
    elif "damage_function" in df.columns:
        df = df[df["damage_function"] == df["damage_function"].iloc[0]]

    # Pivot for heatmap
    pivot = df.pivot_table(
        values="sector_drawdown",
        index="scenario",
        columns="sector",
        aggfunc="first",
    )

    # Convert to percentage
    pivot_pct = pivot * 100

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_pct.values,
            x=pivot_pct.columns.tolist(),
            y=pivot_pct.index.tolist(),
            colorscale="RdYlGn_r",
            text=pivot_pct.values.round(1),
            texttemplate="%{text:.1f}%",
            textfont={"size": 11},
            hovertemplate=(
                "Scenario: %{y}<br>"
                "Sector: %{x}<br>"
                "Drawdown: %{z:.2f}%<extra></extra>"
            ),
            colorbar={"title": "Drawdown (%)"},
        )
    )

    dmg_label = damage_function or "selected"
    fig.update_layout(
        title=f"Sector Drawdowns by Scenario ({year}, {dmg_label})",
        xaxis_title="GICS Sector",
        yaxis_title="NGFS Scenario",
        height=500,
        margin={"l": 200, "r": 50, "t": 80, "b": 120},
        xaxis={"tickangle": -45},
    )

    return fig


def sensitivity_bands(
    matrix: DrawdownMatrix,
    year: int = 2050,
    model: str | None = None,
) -> go.Figure:
    """
    Show how portfolio drawdowns vary across damage functions for each scenario.

    Displays min/max range as bands with the individual function values as markers.
    This is the key visualization demonstrating the 30-50% specification variance.

    Args:
        matrix: Drawdown matrix computed with all three damage functions.
        year: Target year.
        model: Filter to a specific model.

    Returns:
        Plotly Figure with error-band chart.
    """
    df = matrix.summary_by_scenario()
    df = df[df["year"] == year]
    if model:
        df = df[df["model"] == model]
    elif "model" in df.columns:
        df = df[df["model"] == df["model"].iloc[0]]

    # Compute range per scenario
    agg = df.groupby("scenario").agg(
        min_drawdown=("portfolio_drawdown", "min"),
        max_drawdown=("portfolio_drawdown", "max"),
        mean_drawdown=("portfolio_drawdown", "mean"),
    ).reset_index()

    agg["spread"] = agg["max_drawdown"] - agg["min_drawdown"]
    agg["spread_pct"] = (agg["spread"] / agg["min_drawdown"].clip(lower=1e-6) * 100).round(1)

    # Sort by mean drawdown
    agg = agg.sort_values("mean_drawdown", ascending=True)

    fig = go.Figure()

    # Add range bars
    fig.add_trace(
        go.Bar(
            x=agg["scenario"],
            y=agg["min_drawdown"] * 100,
            name="Min (damage fn)",
            marker_color="rgba(55, 128, 191, 0.7)",
        )
    )

    fig.add_trace(
        go.Bar(
            x=agg["scenario"],
            y=agg["spread"] * 100,
            name="Spread across damage functions",
            marker_color="rgba(219, 64, 82, 0.5)",
            base=agg["min_drawdown"].values * 100,
        )
    )

    # Add individual damage function points
    colors = {
        "Kalkuhl-Wenz (2020)": "#1f77b4",
        "Burke-Hsiang-Miguel (2015)": "#d62728",
        "Howard-Sterner (2017)": "#ff7f0e",
    }

    for dmg_fn in df["damage_function"].unique():
        fn_df = df[df["damage_function"] == dmg_fn]
        fn_ordered = fn_df.set_index("scenario").reindex(agg["scenario"])

        fig.add_trace(
            go.Scatter(
                x=fn_ordered.index,
                y=fn_ordered["portfolio_drawdown"].values * 100,
                mode="markers",
                name=dmg_fn,
                marker={
                    "size": 12,
                    "symbol": "diamond",
                    "color": colors.get(dmg_fn, "black"),
                    "line": {"width": 1, "color": "white"},
                },
            )
        )

    # Add spread annotations
    for _, row in agg.iterrows():
        if row["min_drawdown"] > 0:
            fig.add_annotation(
                x=row["scenario"],
                y=(row["max_drawdown"]) * 100 + 0.5,
                text=f"{row['spread_pct']:.0f}% spread",
                showarrow=False,
                font={"size": 10, "color": "red"},
            )

    fig.update_layout(
        title=f"Portfolio Drawdown Sensitivity to Damage Function ({year})",
        xaxis_title="NGFS Scenario",
        yaxis_title="Portfolio Drawdown (%)",
        barmode="overlay",
        height=500,
        showlegend=True,
        legend={"orientation": "h", "y": -0.3},
        margin={"b": 150},
    )

    return fig


def damage_function_comparison() -> go.Figure:
    """
    Plot all three damage functions side by side across temperature range.

    Shows D(T) from 0C to 6C for direct comparison of functional forms.

    Returns:
        Plotly Figure with three overlaid curves.
    """
    temps = np.linspace(0, 6, 200)

    fig = go.Figure()

    styles = {
        DamageFunctionName.KALKUHL_WENZ: {
            "color": "#1f77b4",
            "dash": "solid",
        },
        DamageFunctionName.BURKE_HSIANG_MIGUEL: {
            "color": "#d62728",
            "dash": "dash",
        },
        DamageFunctionName.HOWARD_STERNER: {
            "color": "#ff7f0e",
            "dash": "dot",
        },
    }

    for fn_name in DamageFunctionName:
        fn = get_damage_function(fn_name)
        damages = fn(temps)

        style = styles[fn_name]
        fig.add_trace(
            go.Scatter(
                x=temps,
                y=np.asarray(damages) * 100,
                name=fn.name,
                line={"color": style["color"], "dash": style["dash"], "width": 3},
                hovertemplate=(
                    f"{fn.name}<br>"
                    "T = %{x:.1f}°C<br>"
                    "GDP Loss = %{y:.2f}%<extra></extra>"
                ),
            )
        )

    # Add reference lines for key temperature targets
    for temp, label, color in [
        (1.5, "1.5°C Paris", "green"),
        (2.0, "2.0°C Paris", "orange"),
        (3.0, "3.0°C Current Policies", "red"),
    ]:
        fig.add_vline(x=temp, line_dash="dot", line_color=color, opacity=0.5)
        fig.add_annotation(
            x=temp,
            y=0,
            yref="paper",
            text=label,
            showarrow=False,
            textangle=-90,
            font={"size": 9, "color": color},
            xshift=10,
        )

    fig.update_layout(
        title="Damage Functions: GDP Loss vs. Temperature Anomaly",
        xaxis_title="Temperature Anomaly (°C above pre-industrial)",
        yaxis_title="GDP Loss (%)",
        height=500,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        hovermode="x unified",
    )

    return fig


def temperature_trajectory_chart(
    temperature_df: pd.DataFrame,
    model: str | None = None,
) -> go.Figure:
    """
    Plot temperature trajectories for all scenarios.

    Args:
        temperature_df: DataFrame with model, scenario, year, temperature_anomaly_C.
        model: Filter to a specific model.

    Returns:
        Plotly Figure with one line per scenario.
    """
    df = temperature_df.copy()
    if model:
        df = df[df["model"] == model]
    elif "model" in df.columns:
        df = df[df["model"] == df["model"].iloc[0]]

    fig = px.line(
        df,
        x="year",
        y="temperature_anomaly_C",
        color="scenario",
        title="NGFS Temperature Trajectories",
        labels={
            "year": "Year",
            "temperature_anomaly_C": "Temperature Anomaly (°C)",
            "scenario": "Scenario",
        },
    )

    # Add Paris Agreement targets
    fig.add_hline(y=1.5, line_dash="dot", line_color="green", opacity=0.5,
                  annotation_text="1.5°C target")
    fig.add_hline(y=2.0, line_dash="dot", line_color="orange", opacity=0.5,
                  annotation_text="2.0°C target")

    fig.update_layout(height=450)

    return fig


def portfolio_summary_chart(portfolio: Portfolio) -> go.Figure:
    """
    Create a sunburst or treemap showing portfolio allocation by sector.

    Displays portfolio weights grouped by GICS sector, with individual
    holdings shown as sub-segments. Color intensity reflects the sector's
    climate risk multiplier.

    Args:
        portfolio: Validated portfolio.

    Returns:
        Plotly Figure with treemap visualization.
    """
    from ngfs.portfolio import SECTOR_CLIMATE_MULTIPLIERS

    records = []
    for pos in portfolio.positions:
        records.append({
            "ticker": pos.ticker,
            "name": pos.name,
            "sector": pos.gics_sector.value,
            "weight": pos.weight,
            "weight_pct": pos.weight * 100,
            "climate_multiplier": SECTOR_CLIMATE_MULTIPLIERS.get(pos.gics_sector, 1.0),
        })

    df = pd.DataFrame(records)

    fig = px.treemap(
        df,
        path=["sector", "ticker"],
        values="weight_pct",
        color="climate_multiplier",
        color_continuous_scale="RdYlGn_r",
        title="Portfolio Allocation by Sector (color = climate risk multiplier)",
        hover_data={"name": True, "weight_pct": ":.1f", "climate_multiplier": ":.1f"},
    )

    fig.update_layout(
        height=500,
        coloraxis_colorbar={"title": "Climate<br>Risk<br>Multiplier"},
    )

    return fig
