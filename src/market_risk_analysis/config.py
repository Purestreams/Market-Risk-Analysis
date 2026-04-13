from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_START_TIMESTAMP = "2021-01-01T00:00:00+00:00"
DEFAULT_CONFIDENCE = 0.99
RECENT_WINDOW_HOURS = 24 * 90
BACKTEST_WINDOW_HOURS = 24 * 90
BACKTEST_DAYS = 365
HORIZON_HOURS = (1, 24)
MONTE_CARLO_PATHS = 10_000
LSTM_SEQUENCE_LENGTH = 48
VAE_WINDOW = 24
RANDOM_SEED = 42


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT
    output_dir: Path = PROJECT_ROOT / "outputs"
    data_dir: Path = PROJECT_ROOT / "outputs" / "data"
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    tables_dir: Path = PROJECT_ROOT / "outputs" / "tables"
    models_dir: Path = PROJECT_ROOT / "outputs" / "models"
    raw_data_path: Path = PROJECT_ROOT / "outputs" / "data" / "btc_usd_hourly_raw.csv"
    cleaned_data_path: Path = PROJECT_ROOT / "outputs" / "data" / "btc_usd_hourly_features.csv"
    results_path: Path = PROJECT_ROOT / "outputs" / "results.json"
    report_path: Path = PROJECT_ROOT / "outputs" / "report.md"

    def ensure(self) -> None:
        for directory in (
            self.output_dir,
            self.data_dir,
            self.figures_dir,
            self.tables_dir,
            self.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)