"""
Portfolio parsing, validation, and GICS sector mapping.

Reads a CSV of portfolio holdings, validates required fields, and maps
each holding to a GICS (Global Industry Classification Standard) sector.
This mapping is used downstream to apply differentiated climate damage
multipliers by sector.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class GICSSector(str, Enum):
    """GICS Level 1 sectors relevant for climate risk analysis."""

    ENERGY = "Energy"
    MATERIALS = "Materials"
    INDUSTRIALS = "Industrials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    HEALTH_CARE = "Health Care"
    FINANCIALS = "Financials"
    INFORMATION_TECHNOLOGY = "Information Technology"
    COMMUNICATION_SERVICES = "Communication Services"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"


# Sector-level climate risk multipliers.
# These scale the macro-level GDP damage to approximate sector-specific impacts.
# Values > 1.0 indicate the sector is more exposed than the broad economy;
# < 1.0 indicates relative insulation.
#
# Calibration sources (these are stylized estimates suitable for portfolio
# stress-testing demonstration, not production risk modeling):
#   - NGFS Phase III/IV sectoral GVA pathways (REMIND-MAgPIE), which show
#     energy/utilities/materials taking outsized hits under transition scenarios
#   - Battiston et al. (2017) "A climate stress-test of the financial system"
#     Nature Climate Change, for transition-risk sector classifications
#   - 2DII Paris Agreement Capital Transition Assessment for stranded-asset
#     exposure by sector
#   - IPCC AR6 WG2 Chapter 16 for physical-risk sector vulnerability
#
# To customize: override these values via the constructor of any function
# that calls compute_sector_drawdown(), or fork this dict and pass it in.
# Production use should derive multipliers from firm-level emissions
# intensity, asset location, and revenue-segment exposure rather than
# sector-level averages.
SECTOR_CLIMATE_MULTIPLIERS: dict[GICSSector, float] = {
    GICSSector.ENERGY: 2.5,  # Stranded fossil-fuel assets, transition risk
    GICSSector.MATERIALS: 1.8,  # Cement/steel: high carbon intensity, physical risk
    GICSSector.INDUSTRIALS: 1.4,  # Supply chain disruption, capex repricing
    GICSSector.CONSUMER_DISCRETIONARY: 1.1,  # Demand shifts (autos, travel)
    GICSSector.CONSUMER_STAPLES: 0.9,  # Agriculture exposure offset by defensive demand
    GICSSector.HEALTH_CARE: 0.7,  # Relatively insulated; some heat-stress upside
    GICSSector.FINANCIALS: 1.3,  # Loan book exposure, underwriting risk
    GICSSector.INFORMATION_TECHNOLOGY: 0.6,  # Low direct operational exposure
    GICSSector.COMMUNICATION_SERVICES: 0.5,  # Low direct operational exposure
    GICSSector.UTILITIES: 2.0,  # Transition-heavy, stranded thermal generation
    GICSSector.REAL_ESTATE: 1.6,  # Physical risk (flooding, storms, heat)
}

# Mapping from common sector name variants to canonical GICS sectors
SECTOR_ALIASES: dict[str, GICSSector] = {
    # Direct matches
    "energy": GICSSector.ENERGY,
    "materials": GICSSector.MATERIALS,
    "industrials": GICSSector.INDUSTRIALS,
    "consumer discretionary": GICSSector.CONSUMER_DISCRETIONARY,
    "consumer staples": GICSSector.CONSUMER_STAPLES,
    "health care": GICSSector.HEALTH_CARE,
    "healthcare": GICSSector.HEALTH_CARE,
    "financials": GICSSector.FINANCIALS,
    "financial": GICSSector.FINANCIALS,
    "information technology": GICSSector.INFORMATION_TECHNOLOGY,
    "technology": GICSSector.INFORMATION_TECHNOLOGY,
    "tech": GICSSector.INFORMATION_TECHNOLOGY,
    "it": GICSSector.INFORMATION_TECHNOLOGY,
    "communication services": GICSSector.COMMUNICATION_SERVICES,
    "communications": GICSSector.COMMUNICATION_SERVICES,
    "telecom": GICSSector.COMMUNICATION_SERVICES,
    "utilities": GICSSector.UTILITIES,
    "real estate": GICSSector.REAL_ESTATE,
    # Common sub-sector aliases
    "oil & gas": GICSSector.ENERGY,
    "oil and gas": GICSSector.ENERGY,
    "mining": GICSSector.MATERIALS,
    "chemicals": GICSSector.MATERIALS,
    "banks": GICSSector.FINANCIALS,
    "insurance": GICSSector.FINANCIALS,
    "pharma": GICSSector.HEALTH_CARE,
    "biotech": GICSSector.HEALTH_CARE,
    "semiconductors": GICSSector.INFORMATION_TECHNOLOGY,
    "software": GICSSector.INFORMATION_TECHNOLOGY,
    "media": GICSSector.COMMUNICATION_SERVICES,
    "reits": GICSSector.REAL_ESTATE,
    "electric utilities": GICSSector.UTILITIES,
    "power": GICSSector.UTILITIES,
    "aerospace": GICSSector.INDUSTRIALS,
    "defense": GICSSector.INDUSTRIALS,
    "automotive": GICSSector.CONSUMER_DISCRETIONARY,
    "retail": GICSSector.CONSUMER_DISCRETIONARY,
    "food & beverage": GICSSector.CONSUMER_STAPLES,
    "food": GICSSector.CONSUMER_STAPLES,
}


class Position(BaseModel):
    """A single portfolio holding."""

    ticker: str
    name: str
    sector: str
    gics_sector: GICSSector
    weight: float  # portfolio weight as decimal (0.10 = 10%)
    market_value: float | None = None  # optional, in USD

    @field_validator("weight")
    @classmethod
    def weight_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Weight must be positive, got {v}")
        if v > 1.0:
            raise ValueError(f"Weight must be <= 1.0 (decimal), got {v}")
        return v

    @field_validator("ticker")
    @classmethod
    def ticker_must_be_nonempty(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("Ticker must not be empty")
        return v


@dataclass
class Portfolio:
    """A validated collection of portfolio positions."""

    positions: list[Position]
    name: str = "Unnamed Portfolio"

    @property
    def total_weight(self) -> float:
        return sum(p.weight for p in self.positions)

    @property
    def sector_weights(self) -> dict[GICSSector, float]:
        """Aggregate portfolio weight by GICS sector."""
        weights: dict[GICSSector, float] = {}
        for p in self.positions:
            weights[p.gics_sector] = weights.get(p.gics_sector, 0.0) + p.weight
        return weights

    @property
    def sector_count(self) -> int:
        return len(set(p.gics_sector for p in self.positions))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert portfolio to a DataFrame for analysis."""
        records = [p.model_dump() for p in self.positions]
        return pd.DataFrame(records)


def resolve_gics_sector(sector_tag: str) -> GICSSector:
    """
    Map a sector tag string to a canonical GICS sector.

    Args:
        sector_tag: Free-form sector label from the portfolio CSV.

    Returns:
        Matching GICSSector enum member.

    Raises:
        ValueError: If the tag cannot be resolved to any known sector.
    """
    normalized = sector_tag.strip().lower()

    # Try alias lookup first
    if normalized in SECTOR_ALIASES:
        return SECTOR_ALIASES[normalized]

    # Try matching against enum values (case-insensitive)
    for member in GICSSector:
        if normalized == member.value.lower():
            return member

    # Try substring matching as a fallback
    for alias, sector in SECTOR_ALIASES.items():
        if alias in normalized or normalized in alias:
            logger.warning(
                "Fuzzy sector match: '%s' -> '%s' (via alias '%s')",
                sector_tag,
                sector.value,
                alias,
            )
            return sector

    raise ValueError(
        f"Cannot resolve sector tag '{sector_tag}' to a GICS sector. "
        f"Known sectors: {[s.value for s in GICSSector]}"
    )


def parse_portfolio(csv_path: str | Path) -> Portfolio:
    """
    Parse a portfolio CSV file into a validated Portfolio object.

    Expected CSV columns:
        - ticker: Stock ticker symbol (required)
        - name: Company name (required)
        - sector: Sector label, will be mapped to GICS (required)
        - weight: Portfolio weight as decimal, e.g. 0.10 for 10% (required)
        - market_value: Market value in USD (optional)

    Args:
        csv_path: Path to the portfolio CSV file.

    Returns:
        Validated Portfolio with GICS sector mappings.

    Raises:
        FileNotFoundError: If the CSV doesn't exist.
        ValueError: If required columns are missing or data is invalid.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Portfolio CSV not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Validate required columns
    required = {"ticker", "name", "sector", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Portfolio CSV missing required columns: {missing}")

    positions = []
    errors = []

    for idx, row in df.iterrows():
        try:
            gics = resolve_gics_sector(row["sector"])
            pos = Position(
                ticker=row["ticker"],
                name=row["name"],
                sector=row["sector"],
                gics_sector=gics,
                weight=float(row["weight"]),
                market_value=float(row["market_value"]) if "market_value" in row and pd.notna(row.get("market_value")) else None,
            )
            positions.append(pos)
        except (ValueError, KeyError) as e:
            errors.append(f"Row {idx + 1}: {e}")

    if errors:
        logger.warning("Portfolio parsing had %d errors:\n%s", len(errors), "\n".join(errors))

    if not positions:
        raise ValueError(f"No valid positions parsed from {path}. Errors: {errors}")

    portfolio = Portfolio(positions=positions, name=path.stem)

    # Warn if weights don't sum to ~1.0
    total = portfolio.total_weight
    if abs(total - 1.0) > 0.05:
        logger.warning(
            "Portfolio weights sum to %.4f (expected ~1.0). "
            "Results may need normalization.",
            total,
        )

    logger.info(
        "Parsed portfolio '%s': %d positions across %d sectors, total weight=%.4f",
        portfolio.name,
        len(positions),
        portfolio.sector_count,
        total,
    )

    return portfolio
