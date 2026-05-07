"""
Client for the IIASA Scenario Explorer API (NGFS Phase V scenarios).

Fetches scenario data from the IIASA database, which hosts the NGFS
climate scenario pathways. The six NGFS scenarios span three models:
REMIND-MAgPIE, GCAM 6.0, and MESSAGEix-GLOBIOM.

NGFS Phase V scenarios:
  1. Net Zero 2050          (orderly, 1.5C aligned)
  2. Below 2C               (orderly, <2C)
  3. Divergent Net Zero     (disorderly, 1.5C but with regional divergence)
  4. Delayed Transition     (disorderly, sudden policy tightening post-2030)
  5. Nationally Determined Contributions (hot house, ~2.5-3C)
  6. Current Policies       (hot house, ~3+C)

Data is cached locally as parquet for offline use and fast reload.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# IIASA Scenario Explorer API base
IIASA_API_BASE = "https://data.ece.iiasa.ac.at/ngfs-phase-5/api/v1"

# Default cache directory
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"

# NGFS Phase V scenario names
NGFS_SCENARIOS = [
    "Net Zero 2050",
    "Below 2°C",
    "Divergent Net Zero",
    "Delayed Transition",
    "Nationally Determined Contributions",
    "Current Policies",
]

# IAM models in NGFS Phase V
NGFS_MODELS = [
    "REMIND-MAgPIE 3.3-4.8",
    "GCAM 6.0 NGFS",
    "MESSAGEix-GLOBIOM 2.0-M-R12",
]

# Variables most relevant for sector-level climate risk analysis
KEY_VARIABLES = [
    "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile",
    "Emissions|CO2",
    "Emissions|CO2|Energy",
    "Carbon Sequestration|CCS",
    "Primary Energy",
    "Primary Energy|Coal",
    "Primary Energy|Gas",
    "Primary Energy|Oil",
    "Primary Energy|Nuclear",
    "Primary Energy|Solar",
    "Primary Energy|Wind",
    "Secondary Energy|Electricity",
    "GDP|MER",
    "GDP|PPP",
    "Price|Carbon",
]


@dataclass
class ScenarioData:
    """Container for fetched scenario data with metadata."""

    df: pd.DataFrame
    model: str
    scenario: str
    variables: list[str]
    last_fetched: str = ""


@dataclass
class IIASAClient:
    """
    Client for fetching NGFS scenario data from IIASA Scenario Explorer.

    Usage:
        client = IIASAClient()
        data = client.fetch_scenarios()
        client.cache_to_parquet(data)

        # Later, load from cache:
        data = client.load_from_cache()
    """

    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    api_base: str = IIASA_API_BASE

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, model: str, scenario: str) -> Path:
        """Generate a deterministic cache filename for a model-scenario pair."""
        key = f"{model}::{scenario}"
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        safe_name = f"ngfs_{h}.parquet"
        return self.cache_dir / safe_name

    def _master_cache_path(self) -> Path:
        """Path for the combined all-scenarios cache file."""
        return self.cache_dir / "ngfs_phase5_all.parquet"

    def fetch_scenarios(
        self,
        models: list[str] | None = None,
        scenarios: list[str] | None = None,
        variables: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch scenario data from the IIASA API.

        This is a stub implementation. The real implementation would use
        the IIASA REST API or the `pyam` package to query the database.

        For now, generates synthetic but structurally correct data for
        development and testing purposes.

        Args:
            models: List of IAM models (defaults to all three NGFS models).
            scenarios: List of scenario names (defaults to all six NGFS scenarios).
            variables: List of variable names to fetch (defaults to KEY_VARIABLES).

        Returns:
            DataFrame in IAMC long format: model, scenario, variable, region,
            unit, and year columns with value.
        """
        models = models or NGFS_MODELS
        scenarios = scenarios or NGFS_SCENARIOS
        variables = variables or KEY_VARIABLES

        logger.info(
            "Fetching NGFS scenarios: %d models x %d scenarios x %d variables",
            len(models),
            len(scenarios),
            len(variables),
        )

        # TODO: Replace with actual API call:
        #   import requests
        #   resp = requests.get(f"{self.api_base}/datapoints", params={...})
        # OR:
        #   import pyam
        #   conn = pyam.read_iiasa("ngfs-phase-5")
        #   df = conn.filter(model=models, scenario=scenarios, variable=variables)

        # Generate synthetic placeholder data for development
        df = self._generate_synthetic_data(models, scenarios, variables)
        logger.info("Fetched %d data points (synthetic placeholder)", len(df))
        return df

    def _generate_synthetic_data(
        self,
        models: list[str],
        scenarios: list[str],
        variables: list[str],
    ) -> pd.DataFrame:
        """
        Generate structurally valid synthetic NGFS data.

        Temperature trajectories are calibrated to roughly match the
        expected 2100 endpoints for each scenario pathway.
        """
        import numpy as np

        years = list(range(2020, 2101, 5))

        # Approximate 2100 temperature anomalies by scenario
        temp_targets = {
            "Net Zero 2050": 1.5,
            "Below 2°C": 1.8,
            "Divergent Net Zero": 1.6,
            "Delayed Transition": 2.0,
            "Nationally Determined Contributions": 2.7,
            "Current Policies": 3.3,
        }

        records = []
        for model in models:
            for scenario in scenarios:
                target_temp = temp_targets.get(scenario, 2.5)
                for variable in variables:
                    for year in years:
                        t_frac = (year - 2020) / 80.0  # fraction of 2020-2100

                        if "Temperature" in variable:
                            # Smooth temperature trajectory from ~1.1C (2020) to target
                            value = 1.1 + (target_temp - 1.1) * t_frac**0.7
                        elif "Price|Carbon" in variable:
                            # Carbon price ramps faster in ambitious scenarios
                            if "Net Zero" in scenario or "Below" in scenario:
                                value = 50 + 400 * t_frac**1.5
                            else:
                                value = 10 + 50 * t_frac
                        elif "Emissions|CO2" == variable:
                            # Emissions decline in orderly, flat in hot-house
                            base = 40.0  # GtCO2/yr in 2020
                            if "Net Zero" in scenario:
                                value = base * (1 - t_frac * 1.2)
                                value = max(value, -5.0)  # net negative
                            elif "Below" in scenario:
                                value = base * (1 - t_frac * 0.9)
                            else:
                                value = base * (1 - t_frac * 0.2)
                        elif "GDP" in variable:
                            # GDP grows with climate drag
                            base_growth = 1.0 + 0.025 * (year - 2020)
                            value = 100 * base_growth  # index, 2020=100
                        else:
                            # Generic placeholder: slow growth
                            value = 100 * (1 + 0.01 * (year - 2020))

                        # Add small model-specific perturbation
                        model_noise = hash(model) % 100 / 10000.0
                        value *= 1.0 + model_noise

                        records.append(
                            {
                                "model": model,
                                "scenario": scenario,
                                "variable": variable,
                                "region": "World",
                                "unit": _unit_for_variable(variable),
                                "year": year,
                                "value": round(value, 4),
                            }
                        )

        return pd.DataFrame(records)

    def cache_to_parquet(self, df: pd.DataFrame) -> Path:
        """
        Cache fetched scenario data as a parquet file.

        Args:
            df: DataFrame in IAMC format.

        Returns:
            Path to the saved parquet file.
        """
        path = self._master_cache_path()
        df.to_parquet(path, engine="pyarrow", index=False)
        logger.info("Cached %d rows to %s", len(df), path)
        return path

    def load_from_cache(self) -> pd.DataFrame | None:
        """
        Load previously cached scenario data.

        Returns:
            DataFrame if cache exists, None otherwise.
        """
        path = self._master_cache_path()
        if path.exists():
            df = pd.read_parquet(path, engine="pyarrow")
            logger.info("Loaded %d rows from cache: %s", len(df), path)
            return df
        logger.warning("No cache found at %s", path)
        return None

    def get_temperature_trajectories(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Extract temperature anomaly trajectories for all model-scenario pairs.

        Args:
            df: Full scenario DataFrame (loads from cache if not provided).

        Returns:
            DataFrame with columns: model, scenario, year, temperature_anomaly_C
        """
        if df is None:
            df = self.load_from_cache()
        if df is None:
            df = self.fetch_scenarios()

        temp_var = [v for v in df["variable"].unique() if "Temperature" in v]
        if not temp_var:
            raise ValueError("No temperature variable found in scenario data")

        temp_df = df[df["variable"].isin(temp_var)].copy()
        temp_df = temp_df.rename(columns={"value": "temperature_anomaly_C"})
        return temp_df[["model", "scenario", "year", "temperature_anomaly_C"]]


def _unit_for_variable(variable: str) -> str:
    """Return a plausible unit string for a variable name."""
    if "Temperature" in variable:
        return "°C"
    if "Emissions" in variable:
        return "Mt CO2/yr"
    if "Price|Carbon" in variable:
        return "US$2010/t CO2"
    if "GDP" in variable:
        return "billion US$2010/yr"
    if "Energy" in variable:
        return "EJ/yr"
    return "various"
