"""Tests for climate damage functions."""

import numpy as np
import pytest

from ngfs.damage_functions import (
    BurkeHsiangMiguel,
    DamageFunctionName,
    HowardSterner,
    KalkuhlWenz,
    compare_at_temperature,
    get_damage_function,
)


class TestKalkuhlWenz:
    """Tests for the Kalkuhl-Wenz (2020) damage function."""

    def setup_method(self):
        self.fn = KalkuhlWenz()

    def test_zero_warming_zero_damage(self):
        assert self.fn(0.0) == 0.0

    def test_near_zero_at_2c(self):
        """KW is very conservative — damage is ~0 at 2C (parabola vertex near 2.08C)."""
        damage = self.fn(2.0)
        assert damage >= 0.0
        assert damage < 0.01  # essentially zero at this temperature

    def test_positive_at_3c(self):
        damage = self.fn(3.0)
        assert damage > 0.0
        # At 3C, KW gives modest damage
        assert damage < 0.05

    def test_increases_with_temperature(self):
        d2 = self.fn(2.0)
        d4 = self.fn(4.0)
        d6 = self.fn(6.0)
        assert d4 > d2
        assert d6 > d4

    def test_clamped_to_unit_interval(self):
        # Even at extreme temperatures
        assert 0.0 <= self.fn(10.0) <= 1.0
        assert 0.0 <= self.fn(0.0) <= 1.0

    def test_vectorized(self):
        temps = np.array([1.0, 2.0, 3.0, 4.0])
        damages = self.fn(temps)
        assert len(damages) == 4
        assert np.all(damages >= 0)
        assert np.all(damages <= 1)
        # Should be monotonically non-decreasing (clamped at 0 for low temps)
        assert np.all(np.diff(damages) >= 0)
        # Above the vertex (~2.08C), should be strictly increasing
        high_temps = np.array([3.0, 4.0, 5.0, 6.0])
        high_damages = self.fn(high_temps)
        assert np.all(np.diff(high_damages) > 0)

    def test_marginal_damage(self):
        md = self.fn.marginal_damage(3.0)
        assert isinstance(md, float)

    def test_name_and_citation(self):
        assert "Kalkuhl" in self.fn.name
        assert "2020" in self.fn.citation


class TestBurkeHsiangMiguel:
    """Tests for the Burke-Hsiang-Miguel (2015) damage function."""

    def setup_method(self):
        self.fn = BurkeHsiangMiguel()

    def test_zero_warming_zero_damage(self):
        damage = self.fn(0.0)
        # BHM has a growth-rate channel; at 0 anomaly, damage should be ~0
        assert abs(damage) < 0.01

    def test_larger_than_kw_at_high_temps(self):
        """BHM should produce larger damages than KW due to growth compounding."""
        bhm = self.fn(4.0)
        kw = KalkuhlWenz()(4.0)
        assert bhm > kw

    def test_increases_with_temperature(self):
        d2 = self.fn(2.0)
        d4 = self.fn(4.0)
        assert d4 > d2

    def test_growth_rate_compounding(self):
        """Longer horizons should give larger cumulative damage."""
        fn_short = BurkeHsiangMiguel(horizon_years=10)
        fn_long = BurkeHsiangMiguel(horizon_years=50)
        assert fn_long(3.0) > fn_short(3.0)

    def test_clamped_to_unit_interval(self):
        assert 0.0 <= self.fn(6.0) <= 1.0

    def test_vectorized(self):
        temps = np.array([1.0, 2.0, 3.0])
        damages = self.fn(temps)
        assert len(damages) == 3
        assert np.all(damages >= 0)

    def test_name_and_citation(self):
        assert "Burke" in self.fn.name
        assert "2015" in self.fn.citation


class TestHowardSterner:
    """Tests for the Howard-Sterner (2017) damage function."""

    def setup_method(self):
        self.fn = HowardSterner()

    def test_zero_warming_zero_damage(self):
        assert self.fn(0.0) == 0.0

    def test_higher_than_nordhaus(self):
        """HS central estimate should be ~2-3x higher than DICE's ~0.00236*T^2."""
        nordhaus_at_3c = 0.00236 * 9  # ~2.1%
        hs_at_3c = self.fn(3.0)
        assert hs_at_3c > nordhaus_at_3c * 1.5

    def test_purely_quadratic(self):
        """With alpha1=0, damage should scale as T^2."""
        d2 = self.fn(2.0)
        d4 = self.fn(4.0)
        # d4 should be ~4x d2 (since (4/2)^2 = 4)
        ratio = d4 / d2
        assert abs(ratio - 4.0) < 0.01

    def test_at_2c(self):
        damage = self.fn(2.0)
        # HS at 2C: 0.01145 * 4 = ~4.6%
        assert abs(damage - 0.0458) < 0.005

    def test_clamped_to_unit_interval(self):
        assert 0.0 <= self.fn(10.0) <= 1.0

    def test_vectorized(self):
        temps = np.linspace(0, 5, 50)
        damages = self.fn(temps)
        assert len(damages) == 50
        assert np.all(damages >= 0)
        assert np.all(damages <= 1)

    def test_name_and_citation(self):
        assert "Howard" in self.fn.name
        assert "2017" in self.fn.citation


class TestDamageFunctionFactory:
    """Tests for the get_damage_function factory and comparison."""

    def test_get_by_enum(self):
        fn = get_damage_function(DamageFunctionName.HOWARD_STERNER)
        assert isinstance(fn, HowardSterner)

    def test_get_by_string(self):
        fn = get_damage_function("kalkuhl_wenz")
        assert isinstance(fn, KalkuhlWenz)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError):
            get_damage_function("nonexistent")

    def test_compare_at_temperature(self):
        results = compare_at_temperature(3.0)
        assert len(results) == 3
        assert all(v >= 0 for v in results.values())

    def test_specification_variance(self):
        """
        Core project thesis: damage functions at the same temperature
        should vary by at least 30%.
        """
        results = compare_at_temperature(3.0)
        values = list(results.values())
        min_val = min(values)
        max_val = max(values)

        if min_val > 0:
            spread_pct = (max_val - min_val) / min_val * 100
            assert spread_pct > 30, (
                f"Expected >30% spread between damage functions at 3C, "
                f"got {spread_pct:.1f}%"
            )
