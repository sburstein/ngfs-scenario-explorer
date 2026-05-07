"""Tests for the scenario engine."""

import numpy as np
import pandas as pd
import pytest

from ngfs.damage_functions import DamageFunctionName, get_damage_function
from ngfs.portfolio import GICSSector, Portfolio, Position
from ngfs.scenario_engine import (
    DrawdownMatrix,
    compute_all_damage_functions,
    compute_drawdowns,
    compute_sector_drawdown,
)


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """A simple test portfolio with three sectors."""
    return Portfolio(
        positions=[
            Position(
                ticker="XOM", name="Exxon", sector="Energy",
                gics_sector=GICSSector.ENERGY, weight=0.30,
            ),
            Position(
                ticker="AAPL", name="Apple", sector="Tech",
                gics_sector=GICSSector.INFORMATION_TECHNOLOGY, weight=0.40,
            ),
            Position(
                ticker="JPM", name="JPMorgan", sector="Banks",
                gics_sector=GICSSector.FINANCIALS, weight=0.30,
            ),
        ],
        name="Test Portfolio",
    )


@pytest.fixture
def temperature_trajectory() -> pd.DataFrame:
    """Simple temperature trajectory for testing."""
    return pd.DataFrame(
        [
            {"model": "TestModel", "scenario": "Net Zero 2050", "year": 2050, "temperature_anomaly_C": 1.5},
            {"model": "TestModel", "scenario": "Current Policies", "year": 2050, "temperature_anomaly_C": 3.0},
        ]
    )


class TestSectorDrawdown:
    """Tests for sector-level drawdown calculation."""

    def test_energy_amplified(self):
        """Energy sector should have higher drawdown than macro due to multiplier > 1."""
        macro = 0.05
        sector_dd = compute_sector_drawdown(macro, GICSSector.ENERGY)
        assert sector_dd > macro

    def test_tech_dampened(self):
        """Tech sector should have lower drawdown than macro due to multiplier < 1."""
        macro = 0.05
        sector_dd = compute_sector_drawdown(macro, GICSSector.INFORMATION_TECHNOLOGY)
        assert sector_dd < macro

    def test_clamped_to_one(self):
        """Sector drawdown should never exceed 100%."""
        assert compute_sector_drawdown(0.9, GICSSector.ENERGY) <= 1.0

    def test_zero_damage(self):
        assert compute_sector_drawdown(0.0, GICSSector.ENERGY) == 0.0


class TestComputeDrawdowns:
    """Tests for the main drawdown computation."""

    def test_basic_computation(self, sample_portfolio, temperature_trajectory):
        matrix = compute_drawdowns(
            portfolio=sample_portfolio,
            temperature_trajectory=temperature_trajectory,
            damage_function=DamageFunctionName.HOWARD_STERNER,
        )
        assert len(matrix.results) == 2  # 2 scenarios
        assert all(r.portfolio_drawdown >= 0 for r in matrix.results)

    def test_higher_temp_higher_drawdown(self, sample_portfolio, temperature_trajectory):
        matrix = compute_drawdowns(
            portfolio=sample_portfolio,
            temperature_trajectory=temperature_trajectory,
            damage_function=DamageFunctionName.HOWARD_STERNER,
        )
        nz = [r for r in matrix.results if r.scenario == "Net Zero 2050"][0]
        cp = [r for r in matrix.results if r.scenario == "Current Policies"][0]
        assert cp.portfolio_drawdown > nz.portfolio_drawdown

    def test_damage_function_by_string(self, sample_portfolio, temperature_trajectory):
        matrix = compute_drawdowns(
            portfolio=sample_portfolio,
            temperature_trajectory=temperature_trajectory,
            damage_function="howard_sterner",
        )
        assert len(matrix.results) > 0

    def test_year_filtering(self, sample_portfolio, temperature_trajectory):
        matrix = compute_drawdowns(
            portfolio=sample_portfolio,
            temperature_trajectory=temperature_trajectory,
            damage_function=DamageFunctionName.KALKUHL_WENZ,
            years=[2050],
        )
        assert all(r.year == 2050 for r in matrix.results)

    def test_sector_drawdowns_present(self, sample_portfolio, temperature_trajectory):
        matrix = compute_drawdowns(
            portfolio=sample_portfolio,
            temperature_trajectory=temperature_trajectory,
            damage_function=DamageFunctionName.HOWARD_STERNER,
        )
        for result in matrix.results:
            assert len(result.sector_drawdowns) == 3  # 3 sectors


class TestDrawdownMatrix:
    """Tests for DrawdownMatrix output methods."""

    def test_to_dataframe(self, sample_portfolio, temperature_trajectory):
        matrix = compute_drawdowns(
            portfolio=sample_portfolio,
            temperature_trajectory=temperature_trajectory,
            damage_function=DamageFunctionName.HOWARD_STERNER,
        )
        df = matrix.to_dataframe()
        assert len(df) == 6  # 2 scenarios * 3 sectors
        assert "sector" in df.columns
        assert "portfolio_drawdown" in df.columns

    def test_summary_by_scenario(self, sample_portfolio, temperature_trajectory):
        matrix = compute_drawdowns(
            portfolio=sample_portfolio,
            temperature_trajectory=temperature_trajectory,
            damage_function=DamageFunctionName.HOWARD_STERNER,
        )
        summary = matrix.summary_by_scenario()
        assert len(summary) == 2  # 2 scenarios


class TestComputeAllDamageFunctions:
    """Tests for multi-damage-function comparison."""

    def test_all_three_present(self, sample_portfolio, temperature_trajectory):
        matrix = compute_all_damage_functions(
            portfolio=sample_portfolio,
            temperature_trajectory=temperature_trajectory,
        )
        fn_names = set(r.damage_function_name for r in matrix.results)
        assert len(fn_names) == 3

    def test_total_results(self, sample_portfolio, temperature_trajectory):
        matrix = compute_all_damage_functions(
            portfolio=sample_portfolio,
            temperature_trajectory=temperature_trajectory,
        )
        # 2 scenarios * 3 damage functions = 6 results
        assert len(matrix.results) == 6

    def test_comparison_table(self, sample_portfolio, temperature_trajectory):
        matrix = compute_all_damage_functions(
            portfolio=sample_portfolio,
            temperature_trajectory=temperature_trajectory,
            years=[2050],
        )
        comparison = matrix.damage_function_comparison(year=2050)
        assert not comparison.empty
        # Should have spread column
        assert "spread" in comparison.columns

    def test_specification_variance_in_results(self, sample_portfolio):
        """
        Core test: verify that the 30-50% specification variance shows up
        in portfolio drawdowns at higher warming levels.
        """
        # Use a trajectory with meaningful warming
        traj = pd.DataFrame(
            [
                {"model": "Test", "scenario": "Hot", "year": 2050, "temperature_anomaly_C": 3.5},
            ]
        )
        matrix = compute_all_damage_functions(
            portfolio=sample_portfolio,
            temperature_trajectory=traj,
            years=[2050],
        )

        drawdowns = [r.portfolio_drawdown for r in matrix.results]
        min_dd = min(drawdowns)
        max_dd = max(drawdowns)

        if min_dd > 0:
            spread_pct = (max_dd - min_dd) / min_dd * 100
            assert spread_pct > 25, (
                f"Expected significant spread in portfolio drawdowns across "
                f"damage functions, got {spread_pct:.1f}%"
            )
