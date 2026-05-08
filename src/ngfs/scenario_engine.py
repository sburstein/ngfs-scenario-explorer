"""
Scenario engine: apply damage functions to NGFS pathways for portfolio drawdown analysis.

This is the core analytical module. It takes:
  1. A portfolio with GICS sector mappings and weights
  2. An NGFS scenario temperature trajectory
  3. A selected damage function

And produces sector-level drawdown estimates and scenario-conditional returns.

The key insight this engine demonstrates: for any given scenario pathway,
switching the damage function specification changes the estimated portfolio
drawdown by 30-50%. This means the choice of damage function is as
consequential as the choice of scenario.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ngfs.damage_functions import (
    DamageFunction,
    DamageFunctionName,
    get_damage_function,
)
from ngfs.portfolio import (
    SECTOR_CLIMATE_MULTIPLIERS,
    GICSSector,
    Portfolio,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in NGFS temperature pathways (NGFS Phase V published data)
#
# Each scenario is defined by its peak/2100 temperature anomaly and a
# trajectory shape. These are used when the IIASA API is unavailable.
# ---------------------------------------------------------------------------

NGFS_TEMPERATURE_PATHWAYS: dict[str, dict] = {
    "Net Zero 2050": {
        "description": "Orderly transition, 1.5C aligned",
        "peak_temp_C": 1.5,
        "temp_2050_C": 1.5,
        "temp_2100_C": 1.4,  # slight decline after peak due to net-negative emissions
        "category": "orderly",
    },
    "Below 2C": {
        "description": "Orderly transition, well below 2C",
        "peak_temp_C": 1.7,
        "temp_2050_C": 1.6,
        "temp_2100_C": 1.7,
        "category": "orderly",
    },
    "Divergent Net Zero": {
        "description": "Disorderly, 1.5C with regional divergence",
        "peak_temp_C": 1.6,
        "temp_2050_C": 1.6,
        "temp_2100_C": 1.5,
        "category": "disorderly",
    },
    "Delayed Transition": {
        "description": "Disorderly, sudden policy tightening post-2030",
        "peak_temp_C": 1.8,
        "temp_2050_C": 1.8,
        "temp_2100_C": 1.7,
        "category": "disorderly",
    },
    "Nationally Determined Contributions": {
        "description": "Hot house world, NDC pledges only",
        "peak_temp_C": 2.5,
        "temp_2050_C": 1.9,
        "temp_2100_C": 2.5,
        "category": "hot_house",
    },
    "Current Policies": {
        "description": "Hot house world, no additional policy action",
        "peak_temp_C": 3.0,
        "temp_2050_C": 2.0,
        "temp_2100_C": 3.0,
        "category": "hot_house",
    },
}


def build_temperature_trajectories(
    model: str = "REMIND-MAgPIE 3.3-4.8",
    scenarios: list[str] | None = None,
    start_year: int = 2025,
    end_year: int = 2100,
    step: int = 5,
) -> pd.DataFrame:
    """
    Build temperature trajectory DataFrame from the built-in NGFS pathway data.

    Generates smooth trajectories between the current ~1.2C anomaly and
    the scenario-specific endpoints, using a power-law interpolation that
    captures the expected trajectory shapes.

    Args:
        model: IAM model label to tag the data with.
        scenarios: List of scenario names (defaults to all six).
        start_year: First year in the trajectory.
        end_year: Last year.
        step: Year increment.

    Returns:
        DataFrame with columns: model, scenario, year, temperature_anomaly_C
    """
    if scenarios is None:
        scenarios = list(NGFS_TEMPERATURE_PATHWAYS.keys())

    years = list(range(start_year, end_year + 1, step))
    current_temp = 1.2  # approximate 2025 anomaly above pre-industrial
    records = []

    for scenario in scenarios:
        pathway = NGFS_TEMPERATURE_PATHWAYS.get(scenario)
        if pathway is None:
            logger.warning("Unknown scenario '%s', skipping", scenario)
            continue

        temp_2050 = pathway["temp_2050_C"]
        temp_2100 = pathway["temp_2100_C"]

        for year in years:
            if year <= 2025:
                temp = current_temp
            elif year <= 2050:
                # Interpolate from current to 2050 value
                frac = (year - 2025) / 25.0
                temp = current_temp + (temp_2050 - current_temp) * frac ** 0.8
            else:
                # Interpolate from 2050 to 2100 value
                frac = (year - 2050) / 50.0
                temp = temp_2050 + (temp_2100 - temp_2050) * frac ** 0.9
            records.append({
                "model": model,
                "scenario": scenario,
                "year": year,
                "temperature_anomaly_C": round(temp, 3),
            })

    return pd.DataFrame(records)


@dataclass
class DrawdownResult:
    """Results for a single sector under a specific scenario + damage function."""

    sector: GICSSector
    portfolio_weight: float
    damage_fraction: float  # GDP-level damage
    sector_multiplier: float  # Sector-specific scaling factor
    sector_drawdown: float  # damage_fraction * sector_multiplier
    weighted_contribution: float  # sector_drawdown * portfolio_weight


@dataclass
class ScenarioResult:
    """Full results for one scenario-pathway + damage function combination."""

    scenario: str
    model: str
    damage_function_name: str
    year: int
    temperature_anomaly: float
    macro_damage_fraction: float
    sector_drawdowns: list[DrawdownResult]
    portfolio_drawdown: float  # Weighted sum of sector drawdowns


@dataclass
class DrawdownMatrix:
    """
    Complete matrix of drawdowns across scenarios, damage functions, and sectors.

    This is the primary output structure. It contains results for every
    combination of (scenario, model, damage_function, year).
    """

    results: list[ScenarioResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten results into a DataFrame for visualization."""
        records = []
        for r in self.results:
            for sd in r.sector_drawdowns:
                records.append(
                    {
                        "scenario": r.scenario,
                        "model": r.model,
                        "damage_function": r.damage_function_name,
                        "year": r.year,
                        "temperature_C": r.temperature_anomaly,
                        "sector": sd.sector.value,
                        "portfolio_weight": sd.portfolio_weight,
                        "macro_damage": r.macro_damage_fraction,
                        "sector_multiplier": sd.sector_multiplier,
                        "sector_drawdown": sd.sector_drawdown,
                        "weighted_contribution": sd.weighted_contribution,
                        "portfolio_drawdown": r.portfolio_drawdown,
                    }
                )
        return pd.DataFrame(records)

    def summary_by_scenario(self) -> pd.DataFrame:
        """Aggregate portfolio-level drawdown by scenario and damage function."""
        df = self.to_dataframe()
        if df.empty:
            return df

        return (
            df.groupby(["scenario", "model", "damage_function", "year"])
            .agg(
                temperature_C=("temperature_C", "first"),
                portfolio_drawdown=("portfolio_drawdown", "first"),
            )
            .reset_index()
        )

    def damage_function_comparison(self, year: int | None = None) -> pd.DataFrame:
        """
        Compare portfolio drawdowns across damage functions for each scenario.

        This is the key analytical output. It shows how toggling the
        damage function shifts estimated drawdowns by 30-50%.
        """
        df = self.summary_by_scenario()
        if df.empty:
            return df

        if year is not None:
            df = df[df["year"] == year]

        pivot = df.pivot_table(
            values="portfolio_drawdown",
            index=["scenario", "model", "year"],
            columns="damage_function",
            aggfunc="first",
        )

        # Calculate spread across damage functions
        if len(pivot.columns) >= 2:
            pivot["spread"] = pivot.max(axis=1) - pivot.min(axis=1)
            pivot["spread_pct"] = (pivot["spread"] / pivot.min(axis=1) * 100).round(1)

        return pivot.reset_index()


def compute_sector_drawdown(
    macro_damage: float,
    sector: GICSSector,
) -> float:
    """
    Apply sector-specific climate risk multiplier to a macro damage estimate.

    Args:
        macro_damage: GDP-level damage fraction from a damage function.
        sector: GICS sector to apply the multiplier for.

    Returns:
        Sector-level drawdown fraction, clamped to [0, 1].
    """
    multiplier = SECTOR_CLIMATE_MULTIPLIERS.get(sector, 1.0)
    return min(macro_damage * multiplier, 1.0)


def compute_drawdowns(
    portfolio: Portfolio,
    temperature_trajectory: pd.DataFrame,
    damage_function: DamageFunction | DamageFunctionName | str,
    years: list[int] | None = None,
) -> DrawdownMatrix:
    """
    Compute sector-level drawdowns for a portfolio under given scenarios.

    This is the main analytical entry point. It processes each scenario's
    temperature trajectory through the selected damage function, applies
    sector-specific multipliers, and produces portfolio-weighted drawdowns.

    Args:
        portfolio: Validated portfolio with GICS sector mappings.
        temperature_trajectory: DataFrame with columns:
            model, scenario, year, temperature_anomaly_C
        damage_function: Damage function instance, enum, or string name.
        years: Optional list of years to evaluate (defaults to all available).

    Returns:
        DrawdownMatrix containing results for all scenario-year combinations.
    """
    # Resolve damage function if needed
    if isinstance(damage_function, (str, DamageFunctionName)):
        dmg_fn = get_damage_function(damage_function)
    else:
        dmg_fn = damage_function

    sector_weights = portfolio.sector_weights

    if years is not None:
        temperature_trajectory = temperature_trajectory[
            temperature_trajectory["year"].isin(years)
        ]

    matrix = DrawdownMatrix()

    # Process each model-scenario-year combination
    grouped = temperature_trajectory.groupby(["model", "scenario", "year"])

    for (model, scenario, year), group in grouped:
        temp = group["temperature_anomaly_C"].values[0]

        # Compute macro-level damage
        macro_damage = float(dmg_fn(temp))

        # Compute per-sector drawdowns
        sector_drawdowns = []
        portfolio_drawdown = 0.0

        for sector, weight in sector_weights.items():
            s_drawdown = compute_sector_drawdown(macro_damage, sector)
            weighted = s_drawdown * weight

            sector_drawdowns.append(
                DrawdownResult(
                    sector=sector,
                    portfolio_weight=weight,
                    damage_fraction=macro_damage,
                    sector_multiplier=SECTOR_CLIMATE_MULTIPLIERS.get(sector, 1.0),
                    sector_drawdown=s_drawdown,
                    weighted_contribution=weighted,
                )
            )
            portfolio_drawdown += weighted

        result = ScenarioResult(
            scenario=scenario,
            model=model,
            damage_function_name=dmg_fn.name,
            year=year,
            temperature_anomaly=temp,
            macro_damage_fraction=macro_damage,
            sector_drawdowns=sector_drawdowns,
            portfolio_drawdown=portfolio_drawdown,
        )
        matrix.results.append(result)

    logger.info(
        "Computed %d scenario-year drawdowns using %s",
        len(matrix.results),
        dmg_fn.name,
    )

    return matrix


def compute_all_damage_functions(
    portfolio: Portfolio,
    temperature_trajectory: pd.DataFrame,
    years: list[int] | None = None,
) -> DrawdownMatrix:
    """
    Compute drawdowns using ALL three damage functions for comparison.

    This produces the full comparison matrix that demonstrates the
    30-50% variance from damage function specification.

    Args:
        portfolio: Validated portfolio.
        temperature_trajectory: Temperature trajectory data.
        years: Optional year filter.

    Returns:
        Combined DrawdownMatrix with results from all three damage functions.
    """
    combined = DrawdownMatrix()

    for fn_name in DamageFunctionName:
        result = compute_drawdowns(
            portfolio=portfolio,
            temperature_trajectory=temperature_trajectory,
            damage_function=fn_name,
            years=years,
        )
        combined.results.extend(result.results)

    logger.info(
        "Computed full comparison: %d total results across %d damage functions",
        len(combined.results),
        len(DamageFunctionName),
    )

    return combined
