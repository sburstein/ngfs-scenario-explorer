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

        This is the key analytical output — it shows how toggling the
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
