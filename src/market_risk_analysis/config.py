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
TUNING_MONTE_CARLO_PATHS = 2_000
TUNING_VALIDATION_DAYS = 30
TUNING_EVAL_STRIDE_HOURS = 6
STUDENT_T_WINDOW_CANDIDATES = (24 * 30, 24 * 60, 24 * 90, 24 * 120)
JUMP_DIFFUSION_WINDOW_CANDIDATES = (24 * 30, 24 * 60, 24 * 90)
JUMP_DIFFUSION_THRESHOLD_CANDIDATES = (2.5, 3.0, 3.5)
LSTM_SEQUENCE_LENGTH = 48
LSTM_LOOKBACK_CANDIDATES = (24, 48, 72)
LSTM_HIDDEN_DIM = 48
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.10
LSTM_LEARNING_RATE = 1e-3
LSTM_BATCH_SIZE = 256
LSTM_TUNING_EPOCHS = 4
LSTM_FINAL_EPOCHS = 8
VAE_WINDOW = 24
VAE_LATENT_DIM_CANDIDATES = (4, 6, 8)
VAE_LEARNING_RATE = 1e-3
VAE_BATCH_SIZE = 512
VAE_TUNING_EPOCHS = 12
VAE_FINAL_EPOCHS = 20
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

    def ensure(self) -> None:
        for directory in (
            self.output_dir,
            self.data_dir,
            self.figures_dir,
            self.tables_dir,
            self.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)