"""Tests for portfolio parsing and GICS sector mapping."""

import tempfile
from pathlib import Path

import pytest

from ngfs.portfolio import (
    GICSSector,
    Portfolio,
    Position,
    parse_portfolio,
    resolve_gics_sector,
)


class TestGICSSectorMapping:
    """Tests for sector tag resolution."""

    def test_direct_match(self):
        assert resolve_gics_sector("Energy") == GICSSector.ENERGY
        assert resolve_gics_sector("Financials") == GICSSector.FINANCIALS

    def test_case_insensitive(self):
        assert resolve_gics_sector("ENERGY") == GICSSector.ENERGY
        assert resolve_gics_sector("energy") == GICSSector.ENERGY

    def test_alias_match(self):
        assert resolve_gics_sector("Technology") == GICSSector.INFORMATION_TECHNOLOGY
        assert resolve_gics_sector("Tech") == GICSSector.INFORMATION_TECHNOLOGY
        assert resolve_gics_sector("Oil & Gas") == GICSSector.ENERGY
        assert resolve_gics_sector("Banks") == GICSSector.FINANCIALS

    def test_unknown_sector_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            resolve_gics_sector("Quantum Computing")

    def test_whitespace_handling(self):
        assert resolve_gics_sector("  Energy  ") == GICSSector.ENERGY


class TestPosition:
    """Tests for Position model validation."""

    def test_valid_position(self):
        pos = Position(
            ticker="AAPL",
            name="Apple Inc",
            sector="Technology",
            gics_sector=GICSSector.INFORMATION_TECHNOLOGY,
            weight=0.10,
        )
        assert pos.ticker == "AAPL"
        assert pos.weight == 0.10

    def test_ticker_uppercased(self):
        pos = Position(
            ticker="aapl",
            name="Apple",
            sector="Tech",
            gics_sector=GICSSector.INFORMATION_TECHNOLOGY,
            weight=0.05,
        )
        assert pos.ticker == "AAPL"

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError):
            Position(
                ticker="AAPL",
                name="Apple",
                sector="Tech",
                gics_sector=GICSSector.INFORMATION_TECHNOLOGY,
                weight=-0.05,
            )

    def test_weight_over_one_rejected(self):
        with pytest.raises(ValueError):
            Position(
                ticker="AAPL",
                name="Apple",
                sector="Tech",
                gics_sector=GICSSector.INFORMATION_TECHNOLOGY,
                weight=1.5,
            )

    def test_empty_ticker_rejected(self):
        with pytest.raises(ValueError):
            Position(
                ticker="  ",
                name="Apple",
                sector="Tech",
                gics_sector=GICSSector.INFORMATION_TECHNOLOGY,
                weight=0.05,
            )


class TestPortfolio:
    """Tests for Portfolio aggregate properties."""

    def setup_method(self):
        self.portfolio = Portfolio(
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
            ]
        )

    def test_total_weight(self):
        assert abs(self.portfolio.total_weight - 1.0) < 0.001

    def test_sector_count(self):
        assert self.portfolio.sector_count == 3

    def test_sector_weights(self):
        sw = self.portfolio.sector_weights
        assert sw[GICSSector.ENERGY] == 0.30
        assert sw[GICSSector.INFORMATION_TECHNOLOGY] == 0.40
        assert sw[GICSSector.FINANCIALS] == 0.30

    def test_to_dataframe(self):
        df = self.portfolio.to_dataframe()
        assert len(df) == 3
        assert "ticker" in df.columns
        assert "weight" in df.columns


class TestParsePortfolio:
    """Tests for CSV parsing."""

    def _write_csv(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp.write(content)
        tmp.close()
        return Path(tmp.name)

    def test_valid_csv(self):
        path = self._write_csv(
            "ticker,name,sector,weight\n"
            "XOM,Exxon Mobil,Energy,0.50\n"
            "AAPL,Apple Inc,Technology,0.50\n"
        )
        portfolio = parse_portfolio(path)
        assert len(portfolio.positions) == 2
        assert abs(portfolio.total_weight - 1.0) < 0.001

    def test_with_market_value(self):
        path = self._write_csv(
            "ticker,name,sector,weight,market_value\n"
            "XOM,Exxon,Energy,0.50,50000\n"
            "AAPL,Apple,Tech,0.50,50000\n"
        )
        portfolio = parse_portfolio(path)
        assert portfolio.positions[0].market_value == 50000

    def test_missing_columns_raises(self):
        path = self._write_csv("ticker,name\nXOM,Exxon\n")
        with pytest.raises(ValueError, match="missing required columns"):
            parse_portfolio(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_portfolio("/tmp/nonexistent_portfolio.csv")

    def test_sample_portfolio(self):
        sample = Path(__file__).parent.parent / "data" / "sample" / "sample_portfolio.csv"
        if sample.exists():
            portfolio = parse_portfolio(sample)
            assert len(portfolio.positions) == 13
            assert abs(portfolio.total_weight - 1.0) < 0.01
