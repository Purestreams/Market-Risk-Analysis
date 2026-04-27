from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    BACKTEST_DAYS,
    BACKTEST_WINDOW_HOURS,
    JUMP_DIFFUSION_THRESHOLD_CANDIDATES,
    JUMP_DIFFUSION_WINDOW_CANDIDATES,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_FINAL_SEQUENCE_LENGTH,
    LSTM_FINAL_TAIL_ALPHA,
    LSTM_FINAL_EPOCHS,
    LSTM_HIDDEN_DIM,
    LSTM_LEARNING_RATE,
    LSTM_LOOKBACK_CANDIDATES,
    LSTM_NUM_LAYERS,
    LSTM_TAIL_CALIBRATION_LEVELS,
    LSTM_SEQUENCE_LENGTH,
    LSTM_TUNING_EPOCHS,
    DEFAULT_CONFIDENCE,
    DEFAULT_START_TIMESTAMP,
    HORIZON_HOURS,
    LSTM_SEQUENCE_LENGTH,
    MONTE_CARLO_PATHS,
    ProjectPaths,
    RANDOM_SEED,
    RECENT_WINDOW_HOURS,
    STUDENT_T_WINDOW_CANDIDATES,
    TUNING_EVAL_STRIDE_HOURS,
    TUNING_MONTE_CARLO_PATHS,
    TUNING_VALIDATION_DAYS,
    VAE_BATCH_SIZE,
    VAE_FINAL_EPOCHS,
    VAE_LATENT_DIM_CANDIDATES,
    VAE_LEARNING_RATE,
    VAE_TUNING_EPOCHS,
    VAE_WINDOW,
)
from .npu import maybe_compile_for_npu, select_training_device


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


@dataclass
class ForecastSummary:
    horizon_hours: int
    var_return: float
    cvar_return: float
    var_loss: float
    cvar_loss: float
    mean_return: float
    volatility: float
    metadata: dict[str, Any]


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def fetch_binance_hourly_data(
    start_timestamp: str = DEFAULT_START_TIMESTAMP,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "market-risk-analysis/0.1"})

    start = pd.Timestamp(start_timestamp)
    end = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
    current_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list[Any]] = []

    while current_ms < end_ms:
        response = session.get(
            BINANCE_KLINES_URL,
            params={
                "symbol": symbol,
                "interval": "1h",
                "limit": 1000,
                "startTime": current_ms,
                "endTime": end_ms,
            },
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        current_ms = last_open_time + 3_600_000

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        raise RuntimeError("Binance returned no BTCUSDT hourly rows.")
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
        ]
    ]


def clean_market_data(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = raw_frame.copy()
    frame = frame.drop_duplicates(subset="timestamp").sort_values("timestamp")
    frame = frame.set_index("timestamp")

    expected_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h", tz="UTC")
    original_rows = len(frame)
    frame = frame.reindex(expected_index)
    missing_hours = int(frame["close"].isna().sum())

    frame["close"] = frame["close"].interpolate(method="time").ffill().bfill()
    frame["open"] = frame["open"].fillna(frame["close"].shift(1)).fillna(frame["close"])
    frame["high"] = frame["high"].fillna(pd.concat([frame["open"], frame["close"]], axis=1).max(axis=1))
    frame["low"] = frame["low"].fillna(pd.concat([frame["open"], frame["close"]], axis=1).min(axis=1))
    frame["volume"] = frame["volume"].fillna(0.0)
    frame["quote_asset_volume"] = frame["quote_asset_volume"].fillna(0.0)
    frame["number_of_trades"] = frame["number_of_trades"].fillna(0.0)

    log_close = np.log(frame["close"])
    forward_jump = log_close.diff()
    mean_reversion = log_close.shift(-1) - log_close
    net_move = log_close.shift(-1) - log_close.shift(1)
    flash_crash_mask = (
        (forward_jump.abs() > 0.18)
        & (mean_reversion.abs() > 0.18)
        & (np.sign(forward_jump) != np.sign(mean_reversion))
        & (net_move.abs() < 0.035)
    )
    flash_crashes = int(flash_crash_mask.fillna(False).sum())
    repaired_close = (frame["close"].shift(1) + frame["close"].shift(-1)) / 2
    frame.loc[flash_crash_mask, "close"] = repaired_close.loc[flash_crash_mask]
    frame.loc[flash_crash_mask, "open"] = frame["close"].shift(1).loc[flash_crash_mask].fillna(frame.loc[flash_crash_mask, "open"])
    frame.loc[flash_crash_mask, "high"] = frame.loc[flash_crash_mask, ["open", "close"]].max(axis=1)
    frame.loc[flash_crash_mask, "low"] = frame.loc[flash_crash_mask, ["open", "close"]].min(axis=1)

    cleaned = frame.reset_index(names="timestamp")
    quality_report = {
        "original_rows": original_rows,
        "cleaned_rows": int(len(cleaned)),
        "missing_hours_filled": missing_hours,
        "duplicate_rows_removed": int(original_rows - raw_frame["timestamp"].nunique()),
        "flash_crash_repairs": flash_crashes,
        "start": cleaned["timestamp"].min().isoformat(),
        "end": cleaned["timestamp"].max().isoformat(),
    }
    return cleaned, quality_report


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_feature_frame(cleaned_frame: pd.DataFrame) -> pd.DataFrame:
    frame = cleaned_frame.copy().set_index("timestamp")
    frame["log_return"] = np.log(frame["close"]).diff()
    annualization = math.sqrt(24 * 365)
    frame["rolling_vol_24h"] = frame["log_return"].rolling(24).std() * annualization
    frame["rolling_vol_168h"] = frame["log_return"].rolling(24 * 7).std() * annualization
    ema_fast = frame["close"].ewm(span=12, adjust=False).mean()
    ema_slow = frame["close"].ewm(span=26, adjust=False).mean()
    frame["macd"] = ema_fast - ema_slow
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["rsi_14"] = compute_rsi(frame["close"], period=14)
    volume_log = np.log1p(frame["volume"])
    rolling_volume_mean = volume_log.rolling(24 * 7).mean()
    rolling_volume_std = volume_log.rolling(24 * 7).std().replace(0, np.nan)
    frame["volume_zscore"] = (volume_log - rolling_volume_mean) / rolling_volume_std
    frame["future_return_1h"] = frame["log_return"].shift(-1)
    frame["future_abs_return_1h"] = frame["future_return_1h"].abs()
    return frame.reset_index()


def compute_descriptive_statistics(returns: pd.Series) -> dict[str, float]:
    clean_returns = returns.dropna()
    jb_stat, jb_pvalue = stats.jarque_bera(clean_returns)
    adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(clean_returns)
    return {
        "observations": float(len(clean_returns)),
        "mean": float(clean_returns.mean()),
        "std": float(clean_returns.std(ddof=1)),
        "skewness": float(stats.skew(clean_returns, bias=False)),
        "kurtosis": float(stats.kurtosis(clean_returns, fisher=False, bias=False)),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "adf_critical_1pct": float(critical_values["1%"]),
        "adf_critical_5pct": float(critical_values["5%"]),
        "adf_critical_10pct": float(critical_values["10%"]),
        "min": float(clean_returns.min()),
        "max": float(clean_returns.max()),
        "p01": float(clean_returns.quantile(0.01)),
        "p99": float(clean_returns.quantile(0.99)),
    }


def summarize_distribution(samples: np.ndarray, confidence: float, horizon_hours: int, metadata: dict[str, Any]) -> ForecastSummary:
    quantile = float(np.quantile(samples, 1 - confidence))
    tail = samples[samples <= quantile]
    if tail.size == 0:
        tail = np.array([quantile])
    return ForecastSummary(
        horizon_hours=horizon_hours,
        var_return=quantile,
        cvar_return=float(tail.mean()),
        var_loss=float(-quantile),
        cvar_loss=float(-tail.mean()),
        mean_return=float(samples.mean()),
        volatility=float(samples.std(ddof=1)),
        metadata=metadata,
    )


def normal_forecast_from_params(
    mean_return: float,
    sigma: float,
    confidence: float,
    horizon_hours: int = 1,
    metadata: dict[str, Any] | None = None,
) -> ForecastSummary:
    z_alpha = stats.norm.ppf(1 - confidence)
    tail_alpha = 1 - confidence
    normal_density = stats.norm.pdf(z_alpha)
    var_return = mean_return + sigma * z_alpha
    cvar_return = mean_return - sigma * normal_density / tail_alpha
    return ForecastSummary(
        horizon_hours=horizon_hours,
        var_return=float(var_return),
        cvar_return=float(cvar_return),
        var_loss=float(-var_return),
        cvar_loss=float(-cvar_return),
        mean_return=float(mean_return),
        volatility=float(sigma),
        metadata={} if metadata is None else metadata,
    )


def forecast_from_standardized_tail(
    mean_return: float,
    sigma: float,
    tail_quantile_z: float,
    tail_cvar_z: float,
    horizon_hours: int = 1,
    metadata: dict[str, Any] | None = None,
) -> ForecastSummary:
    var_return = mean_return + sigma * tail_quantile_z
    cvar_return = mean_return + sigma * tail_cvar_z
    return ForecastSummary(
        horizon_hours=horizon_hours,
        var_return=float(var_return),
        cvar_return=float(cvar_return),
        var_loss=float(-var_return),
        cvar_loss=float(-cvar_return),
        mean_return=float(mean_return),
        volatility=float(sigma),
        metadata={} if metadata is None else metadata,
    )


def gaussian_forecast_from_sample(
    sample: np.ndarray,
    confidence: float,
    horizon_hours: int = 1,
    metadata: dict[str, Any] | None = None,
) -> ForecastSummary:
    mean_return = float(sample.mean()) * horizon_hours
    sigma = float(sample.std(ddof=1)) * math.sqrt(horizon_hours)
    return normal_forecast_from_params(
        mean_return,
        sigma,
        confidence,
        horizon_hours=horizon_hours,
        metadata=metadata
        or {
            "distribution": "Gaussian",
            "estimation_window_observations": float(len(sample)),
        },
    )


def quantile_loss(actual_returns: np.ndarray, forecast_returns: np.ndarray, confidence: float) -> float:
    alpha = 1 - confidence
    differences = actual_returns - forecast_returns
    losses = np.where(actual_returns < forecast_returns, (alpha - 1) * differences, alpha * differences)
    return float(np.mean(losses))


def student_t_forecast_from_sample(
    sample: np.ndarray,
    confidence: float,
    horizon_hours: int = 1,
    metadata: dict[str, Any] | None = None,
) -> ForecastSummary:
    degrees_of_freedom, loc, scale = estimate_student_t_parameters(sample)
    alpha = 1 - confidence
    quantile = stats.t.ppf(alpha, degrees_of_freedom)
    density = stats.t.pdf(quantile, degrees_of_freedom)
    var_return = loc + scale * quantile
    cvar_return = loc - scale * (density / alpha) * ((degrees_of_freedom + quantile**2) / (degrees_of_freedom - 1))
    return ForecastSummary(
        horizon_hours=horizon_hours,
        var_return=float(var_return),
        cvar_return=float(cvar_return),
        var_loss=float(-var_return),
        cvar_loss=float(-cvar_return),
        mean_return=float(loc),
        volatility=float(scale),
        metadata=metadata
        or {
            "distribution": "Student-t",
            "degrees_of_freedom": float(degrees_of_freedom),
            "location": float(loc),
            "scale": float(scale),
        },
    )


def estimate_student_t_parameters(sample: np.ndarray) -> tuple[float, float, float]:
    mean = float(sample.mean())
    std = float(sample.std(ddof=1))
    pearson_kurtosis = float(stats.kurtosis(sample, fisher=False, bias=False))
    if not np.isfinite(pearson_kurtosis) or pearson_kurtosis <= 3.05:
        degrees_of_freedom = 60.0
    else:
        degrees_of_freedom = float(np.clip((4 * pearson_kurtosis - 6) / (pearson_kurtosis - 3), 4.5, 60.0))
    scale = max(std * math.sqrt((degrees_of_freedom - 2) / degrees_of_freedom), 1e-6)
    return degrees_of_freedom, mean, scale


def fit_student_t_var(
    returns: pd.Series,
    confidence: float,
    horizons: tuple[int, ...],
    rng: np.random.Generator,
) -> dict[int, ForecastSummary]:
    sample = returns.dropna().to_numpy()
    results: dict[int, ForecastSummary] = {}
    for horizon in horizons:
        if horizon == 1:
            results[horizon] = student_t_forecast_from_sample(sample, confidence, horizon_hours=horizon)
            continue
        else:
            degrees_of_freedom, loc, scale = estimate_student_t_parameters(sample)
            simulated = stats.t.rvs(
                degrees_of_freedom,
                loc=loc,
                scale=scale,
                size=(50_000, horizon),
                random_state=rng,
            ).sum(axis=1)
        results[horizon] = summarize_distribution(
            simulated,
            confidence,
            horizon,
            metadata={
                "distribution": "Student-t",
                "degrees_of_freedom": float(degrees_of_freedom),
                "location": float(loc),
                "scale": float(scale),
            },
        )
    return results


def fit_gaussian_var(
    returns: pd.Series,
    confidence: float,
    horizons: tuple[int, ...],
) -> dict[int, ForecastSummary]:
    sample = returns.dropna().to_numpy(dtype=np.float64)
    if sample.size < 2:
        raise RuntimeError("Not enough observations to fit the Gaussian benchmark.")
    results: dict[int, ForecastSummary] = {}
    for horizon in horizons:
        results[horizon] = gaussian_forecast_from_sample(
            sample,
            confidence,
            horizon_hours=horizon,
            metadata={
                "model": "Gaussian benchmark",
                "distribution": "Gaussian",
                "estimation_window_observations": float(sample.size),
            },
        )
    return results


def simulate_jump_diffusion_var(
    returns: pd.Series,
    confidence: float,
    horizons: tuple[int, ...],
    n_paths: int,
    rng: np.random.Generator,
    jump_threshold_sigma: float = 3.0,
) -> tuple[dict[int, ForecastSummary], np.ndarray]:
    sample = returns.dropna().to_numpy()
    mu = float(sample.mean())
    sigma = float(sample.std(ddof=1))
    jump_mask = np.abs(sample - mu) > jump_threshold_sigma * sigma
    jump_returns = sample[jump_mask]
    jump_intensity = float(len(jump_returns) / max(len(sample), 1))
    jump_mean = float(jump_returns.mean()) if len(jump_returns) else 0.0
    jump_std = float(jump_returns.std(ddof=1)) if len(jump_returns) > 1 else max(sigma * 2.5, 1e-4)

    results: dict[int, ForecastSummary] = {}
    sample_paths = np.empty((0, max(horizons)))
    for horizon in horizons:
        diffusion = rng.normal(mu, sigma, size=(n_paths, horizon))
        jump_counts = rng.poisson(jump_intensity, size=(n_paths, horizon))
        jump_sizes = rng.normal(jump_mean, jump_std, size=(n_paths, horizon))
        path_returns = diffusion + jump_counts * jump_sizes
        aggregated_returns = path_returns.sum(axis=1)
        results[horizon] = summarize_distribution(
            aggregated_returns,
            confidence,
            horizon,
            metadata={
                "distribution": "Jump diffusion",
                "drift": mu,
                "diffusion_sigma": sigma,
                "jump_intensity": jump_intensity,
                "jump_mean": jump_mean,
                "jump_std": jump_std,
                "paths": n_paths,
                "jump_threshold_sigma": jump_threshold_sigma,
            },
        )
        if horizon == max(horizons):
            sample_paths = path_returns[:200]

    return results, sample_paths


class LSTMVolatilityModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48, num_layers: int = 2, dropout: float = 0.10) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_output, _ = self.lstm(features)
        params = self.head(sequence_output[:, -1, :])
        mean = params[:, 0]
        sigma = torch.nn.functional.softplus(params[:, 1]) + 1e-4
        return mean, sigma


class ReturnVAE(nn.Module):
    def __init__(self, window: int, latent_dim: int = 6) -> None:
        super().__init__()
        self.window = window
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(window, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.latent_mean = nn.Linear(32, latent_dim)
        self.latent_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, window),
        )

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(inputs)
        return self.latent_mean(hidden), self.latent_logvar(hidden)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, latent_state: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_state)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(inputs)
        latent_state = self.reparameterize(mean, logvar)
        reconstruction = self.decode(latent_state)
        return reconstruction, mean, logvar


def gaussian_nll(mean: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.log(sigma) + 0.5 * ((target - mean) / sigma) ** 2)


def build_lstm_prediction_frame(
    mean_returns: np.ndarray,
    sigma_values: np.ndarray,
    actual_returns: np.ndarray,
    index: pd.Index,
    tail_quantile_z: float,
    tail_cvar_z: float,
) -> pd.DataFrame:
    var_returns = mean_returns + sigma_values * tail_quantile_z
    cvar_returns = mean_returns + sigma_values * tail_cvar_z
    return pd.DataFrame(
        {
            "mean_return": mean_returns,
            "sigma": sigma_values,
            "var_return": var_returns,
            "cvar_return": cvar_returns,
            "actual_return": actual_returns,
        },
        index=pd.Index(index, name="timestamp"),
    )


def calibrate_standardized_tail(
    actual_returns: np.ndarray,
    mean_returns: np.ndarray,
    sigma_values: np.ndarray,
    tail_quantile_level: float,
) -> dict[str, float]:
    residuals = (actual_returns - mean_returns) / np.clip(sigma_values, 1e-6, None)
    tail_quantile_z = float(np.quantile(residuals, tail_quantile_level))
    tail_samples = residuals[residuals <= tail_quantile_z]
    if tail_samples.size == 0:
        tail_samples = np.array([tail_quantile_z], dtype=float)
    return {
        "tail_alpha": float(tail_quantile_level),
        "tail_quantile_z": tail_quantile_z,
        "tail_cvar_z": float(tail_samples.mean()),
        "calibration_sample_count": float(len(residuals)),
    }


def select_lstm_tail_calibration(
    actual_returns: np.ndarray,
    mean_returns: np.ndarray,
    sigma_values: np.ndarray,
    index: pd.Index,
    confidence: float,
    candidate_levels: tuple[float, ...] = LSTM_TAIL_CALIBRATION_LEVELS,
) -> dict[str, float]:
    if len(actual_returns) < 512:
        selected_level = candidate_levels[0]
        calibration = calibrate_standardized_tail(actual_returns, mean_returns, sigma_values, selected_level)
        return {**calibration, "selection_cc_pvalue": float("nan"), "selection_ind_pvalue": float("nan"), "selection_kupiec_pvalue": float("nan")}

    split_index = max(256, len(actual_returns) // 2)
    calibration_actual = actual_returns[:split_index]
    calibration_mean = mean_returns[:split_index]
    calibration_sigma = sigma_values[:split_index]
    selection_actual = actual_returns[split_index:]
    selection_mean = mean_returns[split_index:]
    selection_sigma = sigma_values[split_index:]
    selection_index = index[split_index:]

    candidate_rows: list[dict[str, float]] = []
    for level in candidate_levels:
        calibration = calibrate_standardized_tail(calibration_actual, calibration_mean, calibration_sigma, level)
        selection_predictions = build_lstm_prediction_frame(
            selection_mean,
            selection_sigma,
            selection_actual,
            selection_index,
            calibration["tail_quantile_z"],
            calibration["tail_cvar_z"],
        )
        selection_actual_series = pd.Series(selection_actual, index=pd.Index(selection_index, name="timestamp"), dtype=float)
        diagnostics = backtest_diagnostics(selection_actual_series, selection_predictions["var_return"], confidence)
        candidate_rows.append(
            {
                "tail_alpha": float(level),
                "tail_quantile_z": calibration["tail_quantile_z"],
                "tail_cvar_z": calibration["tail_cvar_z"],
                "calibration_sample_count": calibration["calibration_sample_count"],
                "selection_cc_pvalue": float(diagnostics["christoffersen_cc_pvalue"]),
                "selection_ind_pvalue": float(diagnostics["christoffersen_ind_pvalue"]),
                "selection_kupiec_pvalue": float(diagnostics["kupiec_pvalue"]),
            }
        )

    best_candidate = pd.DataFrame(candidate_rows).sort_values(
        ["selection_cc_pvalue", "selection_ind_pvalue", "selection_kupiec_pvalue", "tail_alpha"],
        ascending=[False, False, False, True],
    ).iloc[0]
    full_calibration = calibrate_standardized_tail(
        actual_returns,
        mean_returns,
        sigma_values,
        float(best_candidate["tail_alpha"]),
    )
    return {
        **full_calibration,
        "selection_cc_pvalue": float(best_candidate["selection_cc_pvalue"]),
        "selection_ind_pvalue": float(best_candidate["selection_ind_pvalue"]),
        "selection_kupiec_pvalue": float(best_candidate["selection_kupiec_pvalue"]),
    }


def build_lstm_datasets(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    backtest_start: pd.Timestamp,
) -> dict[str, Any]:
    frame = feature_frame.set_index("timestamp")
    model_frame = frame[feature_columns + ["future_return_1h"]].dropna()
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    target_index: list[pd.Timestamp] = []

    for end_idx in range(sequence_length, len(model_frame) + 1):
        target_position = end_idx - 1
        sequences.append(model_frame[feature_columns].iloc[end_idx - sequence_length : end_idx].to_numpy(dtype=np.float32))
        targets.append(float(model_frame["future_return_1h"].iloc[target_position]))
        target_index.append(model_frame.index[target_position] + pd.Timedelta(hours=1))

    sequences_array = np.asarray(sequences, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)
    target_index_array = pd.DatetimeIndex(target_index)

    test_mask = target_index_array >= backtest_start
    train_mask = ~test_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError("Insufficient samples to create train/test splits for the LSTM model.")

    train_indices = np.where(train_mask)[0]
    validation_size = max(256, int(len(train_indices) * 0.2))
    validation_indices = train_indices[-validation_size:]
    fit_indices = train_indices[:-validation_size]
    if len(fit_indices) == 0:
        fit_indices = train_indices[: max(len(train_indices) // 2, 1)]
        validation_indices = train_indices[max(len(train_indices) // 2, 1) :]

    scaler = StandardScaler()
    scaler.fit(sequences_array[fit_indices].reshape(-1, len(feature_columns)))

    def transform(batch: np.ndarray) -> np.ndarray:
        flattened = batch.reshape(-1, len(feature_columns))
        transformed = scaler.transform(flattened)
        return transformed.reshape(batch.shape)

    return {
        "scaler": scaler,
        "X_train": transform(sequences_array[fit_indices]),
        "y_train": targets_array[fit_indices],
        "train_index": pd.Index(target_index_array[fit_indices]),
        "X_val": transform(sequences_array[validation_indices]),
        "y_val": targets_array[validation_indices],
        "val_index": pd.Index(target_index_array[validation_indices]),
        "X_test": transform(sequences_array[test_mask]),
        "y_test": targets_array[test_mask],
        "test_index": pd.Index(target_index_array[test_mask]),
        "latest_sequence": transform(sequences_array[-1:]),
        "latest_forecast_index": (model_frame.index[-1] + pd.Timedelta(hours=1)).isoformat(),
    }


def fit_lstm_model(
    datasets: dict[str, Any],
    input_dim: int,
    confidence: float,
    sequence_length: int = LSTM_SEQUENCE_LENGTH,
    hidden_dim: int = LSTM_HIDDEN_DIM,
    num_layers: int = LSTM_NUM_LAYERS,
    dropout: float = LSTM_DROPOUT,
    learning_rate: float = LSTM_LEARNING_RATE,
    batch_size: int = LSTM_BATCH_SIZE,
    epochs: int = LSTM_FINAL_EPOCHS,
    tail_calibration_level: float | None = None,
) -> dict[str, Any]:
    preferred_device, device_note = select_training_device()

    def train_on_device(device: torch.device) -> tuple[LSTMVolatilityModel, list[dict[str, float]]]:
        model = LSTMVolatilityModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(datasets["X_train"], dtype=torch.float32),
                torch.tensor(datasets["y_train"], dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_features = torch.tensor(datasets["X_val"], dtype=torch.float32, device=device)
        val_targets = torch.tensor(datasets["y_val"], dtype=torch.float32, device=device)
        best_state: dict[str, torch.Tensor] | None = None
        best_val = float("inf")
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                mean, sigma = model(batch_features)
                loss = gaussian_nll(mean, sigma, batch_targets)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            model.eval()
            with torch.no_grad():
                val_mean, val_sigma = model(val_features)
                val_loss = float(gaussian_nll(val_mean, val_sigma, val_targets).detach().cpu())
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(np.mean(train_losses)),
                    "val_loss": val_loss,
                }
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

        if best_state is None:
            raise RuntimeError("LSTM training finished without storing a checkpoint.")
        model.load_state_dict(best_state)
        return model, history

    training_note = device_note
    try:
        model, history = train_on_device(preferred_device)
        device_used = preferred_device.type
    except Exception as exc:
        training_note = f"{device_note}; LSTM training fell back to CPU: {exc}"
        model, history = train_on_device(torch.device("cpu"))
        device_used = "cpu"

    model = model.to(torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        val_features = torch.tensor(datasets["X_val"], dtype=torch.float32)
        val_mean, val_sigma = model(val_features)
        test_features = torch.tensor(datasets["X_test"], dtype=torch.float32)
        test_mean, test_sigma = model(test_features)
        latest_sequence = torch.tensor(datasets["latest_sequence"], dtype=torch.float32)
        latest_mean, latest_sigma = model(latest_sequence)

    val_mean_np = val_mean.numpy()
    val_sigma_np = val_sigma.numpy()
    test_mean_np = test_mean.numpy()
    test_sigma_np = test_sigma.numpy()
    if tail_calibration_level is None:
        tail_calibration = select_lstm_tail_calibration(
            np.asarray(datasets["y_val"], dtype=float),
            val_mean_np,
            val_sigma_np,
            datasets["val_index"],
            confidence,
        )
    else:
        tail_calibration = calibrate_standardized_tail(
            np.asarray(datasets["y_val"], dtype=float),
            val_mean_np,
            val_sigma_np,
            tail_calibration_level,
        )
        tail_calibration.update(
            {
                "selection_cc_pvalue": float("nan"),
                "selection_ind_pvalue": float("nan"),
                "selection_kupiec_pvalue": float("nan"),
            }
        )
    validation_predictions = build_lstm_prediction_frame(
        val_mean_np,
        val_sigma_np,
        np.asarray(datasets["y_val"], dtype=float),
        datasets["val_index"],
        tail_calibration["tail_quantile_z"],
        tail_calibration["tail_cvar_z"],
    )
    predictions = build_lstm_prediction_frame(
        test_mean_np,
        test_sigma_np,
        np.asarray(datasets["y_test"], dtype=float),
        datasets["test_index"],
        tail_calibration["tail_quantile_z"],
        tail_calibration["tail_cvar_z"],
    )
    validation_actual = pd.Series(datasets["y_val"], index=pd.Index(datasets["val_index"], name="timestamp"), dtype=float)
    validation_diagnostics = backtest_diagnostics(validation_actual, validation_predictions["var_return"], confidence)

    latest_forecast = forecast_from_standardized_tail(
        float(latest_mean.item()),
        float(latest_sigma.item()),
        tail_calibration["tail_quantile_z"],
        tail_calibration["tail_cvar_z"],
        horizon_hours=1,
        metadata={
            "model": "LSTM Gaussian head",
            "sequence_length": sequence_length,
            "device_used": device_used,
            "runtime_note": training_note,
            "forecast_timestamp": datasets["latest_forecast_index"],
            "tail_calibration": tail_calibration,
        },
    )
    return {
        "current_forecast": latest_forecast,
        "predictions": predictions,
        "validation_predictions": validation_predictions,
        "validation_diagnostics": validation_diagnostics,
        "tail_calibration": tail_calibration,
        "training_history": history,
        "device_used": device_used,
        "runtime_note": training_note,
        "latest_mean_return": float(latest_mean.item()),
        "latest_sigma": float(latest_sigma.item()),
    }


def fit_vae_model(
    returns: pd.Series,
    confidence: float,
    backtest_start: pd.Timestamp,
    rng: np.random.Generator,
    window: int = VAE_WINDOW,
    latent_dim: int = 6,
    learning_rate: float = VAE_LEARNING_RATE,
    batch_size: int = VAE_BATCH_SIZE,
    epochs: int = VAE_FINAL_EPOCHS,
    kl_weight: float = 0.05,
) -> dict[str, Any]:
    train_series = returns[returns.index < backtest_start].dropna()
    if len(train_series) <= window + 512:
        raise RuntimeError("Not enough training observations to fit the VAE model.")

    rolling_windows = np.lib.stride_tricks.sliding_window_view(train_series.to_numpy(dtype=np.float32), window_shape=window)
    validation_size = max(256, int(len(rolling_windows) * 0.2))
    train_windows = rolling_windows[:-validation_size]
    val_windows = rolling_windows[-validation_size:]
    scale_mean = float(train_windows.mean())
    scale_std = float(train_windows.std(ddof=1))
    if scale_std == 0:
        scale_std = 1.0

    train_scaled = ((train_windows - scale_mean) / scale_std).astype(np.float32)
    val_scaled = ((val_windows - scale_mean) / scale_std).astype(np.float32)

    model = ReturnVAE(window=window, latent_dim=latent_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_scaled, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_tensor = torch.tensor(val_scaled, dtype=torch.float32)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, 21):
        model.train()
        batch_losses: list[float] = []
        for (batch_inputs,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mean, logvar = model(batch_inputs)
            recon_loss = torch.mean((reconstruction - batch_inputs) ** 2)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_reconstruction, val_mean, val_logvar = model(val_tensor)
            val_loss = float(
                (torch.mean((val_reconstruction - val_tensor) ** 2)
                + kl_weight * (-0.5 * torch.mean(1 + val_logvar - val_mean.pow(2) - val_logvar.exp()))).detach().cpu()
            )
        history.append({"epoch": float(epoch), "train_loss": float(np.mean(batch_losses)), "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    if best_state is None:
        raise RuntimeError("VAE training finished without storing a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()

    decoder_module, decoder_plan = maybe_compile_for_npu(model.decoder, "vae.decoder")
    latent_samples = torch.tensor(rng.normal(size=(MONTE_CARLO_PATHS, model.latent_dim)), dtype=torch.float32)
    if decoder_plan.using_npu:
        latent_samples = latent_samples.to(dtype=torch.float16)
    with torch.no_grad():
        generated = decoder_module(latent_samples)
    generated_windows = generated.detach().cpu().float().numpy() * scale_std + scale_mean
    generated_returns_1h = generated_windows[:, 0]
    generated_returns_24h = generated_windows.sum(axis=1)

    current_forecasts = {
        1: summarize_distribution(
            generated_returns_1h,
            confidence,
            1,
            metadata={
                "model": "VAE latent sampling",
                "window": window,
                "latent_dim": latent_dim,
                "decoder_runtime": decoder_plan.note,
            },
        ),
        24: summarize_distribution(
            generated_returns_24h,
            confidence,
            24,
            metadata={
                "model": "VAE latent sampling",
                "window": window,
                "latent_dim": latent_dim,
                "decoder_runtime": decoder_plan.note,
            },
        ),
    }
    return {
        "current_forecasts": current_forecasts,
        "backtest_var_return": current_forecasts[1].var_return,
        "generated_returns_1h": generated_returns_1h,
        "training_history": history,
        "runtime_note": decoder_plan.note,
        "split_summary": {
            "window_hours": window,
            "train_window_count": len(train_windows),
            "validation_window_count": len(val_windows),
            "train_start": train_series.index[0],
            "train_end": train_series.index[len(train_windows) + window - 2],
            "validation_start": train_series.index[len(train_windows)],
            "validation_end": train_series.index[-1],
        },
    }


def tune_student_t_window(returns: pd.Series, confidence: float, backtest_start: pd.Timestamp) -> pd.DataFrame:
    validation_start = backtest_start - pd.Timedelta(days=TUNING_VALIDATION_DAYS)
    validation_index = returns.loc[validation_start: backtest_start - pd.Timedelta(hours=1)].index[::TUNING_EVAL_STRIDE_HOURS]
    rows: list[dict[str, Any]] = []

    for window_hours in STUDENT_T_WINDOW_CANDIDATES:
        actual_values: list[float] = []
        forecast_values: list[float] = []
        for timestamp in validation_index:
            calibration_sample = returns.loc[: timestamp - pd.Timedelta(hours=1)].tail(window_hours)
            if len(calibration_sample) < window_hours:
                continue
            summary = student_t_forecast_from_sample(calibration_sample.to_numpy(dtype=np.float64), confidence)
            actual_values.append(float(returns.loc[timestamp]))
            forecast_values.append(summary.var_return)

        if not actual_values:
            continue

        actual_array = np.asarray(actual_values, dtype=float)
        forecast_array = np.asarray(forecast_values, dtype=float)
        rows.append(
            {
                "model": "Student-t VaR",
                "calibration_window_hours": float(window_hours),
                "calibration_window_days": float(window_hours / 24),
                "validation_observations": float(len(actual_array)),
                "validation_quantile_loss": quantile_loss(actual_array, forecast_array, confidence),
                "validation_violation_rate": float(np.mean(actual_array < forecast_array)),
            }
        )

    if not rows:
        raise RuntimeError("Student-t tuning did not produce any valid validation rows.")

    return pd.DataFrame(rows).sort_values(
        ["validation_quantile_loss", "validation_violation_rate"], ascending=[True, True]
    ).reset_index(drop=True)


def tune_jump_diffusion_parameters(returns: pd.Series, confidence: float, backtest_start: pd.Timestamp) -> pd.DataFrame:
    validation_start = backtest_start - pd.Timedelta(days=TUNING_VALIDATION_DAYS)
    validation_index = returns.loc[validation_start: backtest_start - pd.Timedelta(hours=1)].index[::TUNING_EVAL_STRIDE_HOURS]
    rows: list[dict[str, Any]] = []

    for window_hours in JUMP_DIFFUSION_WINDOW_CANDIDATES:
        for jump_threshold_sigma in JUMP_DIFFUSION_THRESHOLD_CANDIDATES:
            actual_values: list[float] = []
            forecast_values: list[float] = []
            for offset, timestamp in enumerate(validation_index):
                calibration_sample = returns.loc[: timestamp - pd.Timedelta(hours=1)].tail(window_hours)
                if len(calibration_sample) < window_hours:
                    continue
                local_rng = np.random.default_rng(
                    RANDOM_SEED
                    + 10_000
                    + window_hours
                    + int(jump_threshold_sigma * 100)
                    + offset
                )
                summary = simulate_jump_diffusion_var(
                    calibration_sample,
                    confidence,
                    (1,),
                    TUNING_MONTE_CARLO_PATHS,
                    local_rng,
                    jump_threshold_sigma=jump_threshold_sigma,
                )[0][1]
                actual_values.append(float(returns.loc[timestamp]))
                forecast_values.append(summary.var_return)

            if not actual_values:
                continue

            actual_array = np.asarray(actual_values, dtype=float)
            forecast_array = np.asarray(forecast_values, dtype=float)
            rows.append(
                {
                    "model": "Jump-diffusion Monte Carlo",
                    "calibration_window_hours": float(window_hours),
                    "calibration_window_days": float(window_hours / 24),
                    "jump_threshold_sigma": float(jump_threshold_sigma),
                    "validation_observations": float(len(actual_array)),
                    "validation_quantile_loss": quantile_loss(actual_array, forecast_array, confidence),
                    "validation_violation_rate": float(np.mean(actual_array < forecast_array)),
                }
            )

    if not rows:
        raise RuntimeError("Jump-diffusion tuning did not produce any valid validation rows.")

    return pd.DataFrame(rows).sort_values(
        ["validation_quantile_loss", "validation_violation_rate"], ascending=[True, True]
    ).reset_index(drop=True)


def tune_vae_latent_dim(returns: pd.Series, confidence: float, backtest_start: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for latent_dim in VAE_LATENT_DIM_CANDIDATES:
        local_rng = np.random.default_rng(RANDOM_SEED + 20_000 + latent_dim)
        result = fit_vae_model(
            returns,
            confidence,
            backtest_start,
            local_rng,
            window=VAE_WINDOW,
            latent_dim=latent_dim,
            learning_rate=VAE_LEARNING_RATE,
            batch_size=VAE_BATCH_SIZE,
            epochs=VAE_TUNING_EPOCHS,
        )
        rows.append(
            {
                "model": "VAE latent sampling",
                "window": float(VAE_WINDOW),
                "latent_dim": float(latent_dim),
                "validation_loss": min(item["val_loss"] for item in result["training_history"]),
                "current_1h_var_loss": result["current_forecasts"][1].var_loss,
                "current_24h_var_loss": result["current_forecasts"][24].var_loss,
            }
        )

    return pd.DataFrame(rows).sort_values(["validation_loss", "current_1h_var_loss"], ascending=[True, True]).reset_index(drop=True)


def build_model_tuning_summary(
    student_t_tuning: pd.DataFrame,
    jump_tuning: pd.DataFrame,
    lstm_window_sensitivity: pd.DataFrame,
    vae_tuning: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    best_student = student_t_tuning.iloc[0]
    best_jump = jump_tuning.iloc[0]
    best_lstm = lstm_window_sensitivity.loc[
        lstm_window_sensitivity["lookback_hours"] == LSTM_FINAL_SEQUENCE_LENGTH
    ].iloc[0]
    best_vae = vae_tuning.iloc[0]

    summary_rows = [
        {
            "model": "Gaussian VaR",
            "design": "Rolling normal distribution benchmark",
            "best_parameters": f"{int(best_student['calibration_window_days'])}-day rolling window",
            "selection_metric": "Baseline reference",
            "validation_score": float("nan"),
        },
        {
            "model": "Student-t VaR",
            "design": "Parametric heavy-tail baseline",
            "best_parameters": (
                f"{int(best_student['calibration_window_days'])}-day rolling window; "
                "df estimated from sample kurtosis"
            ),
            "selection_metric": "Mean pinball loss",
            "validation_score": float(best_student["validation_quantile_loss"]),
        },
        {
            "model": "Jump-diffusion Monte Carlo",
            "design": "Diffusion plus calibrated jump scenario engine",
            "best_parameters": (
                f"{int(best_jump['calibration_window_days'])}-day rolling window; "
                f"jump threshold {float(best_jump['jump_threshold_sigma']):.1f} sigma"
            ),
            "selection_metric": "Mean pinball loss",
            "validation_score": float(best_jump["validation_quantile_loss"]),
        },
        {
            "model": "LSTM conditional VaR",
            "design": "Feature-conditioned recurrent forecaster with validation-calibrated tail",
            "best_parameters": (
                f"{int(best_lstm['lookback_hours'])}-hour look-back; "
                f"{LSTM_NUM_LAYERS} layers; {LSTM_HIDDEN_DIM} hidden units; dropout {LSTM_DROPOUT:.2f}; "
                f"calibrated tail alpha {LSTM_FINAL_TAIL_ALPHA:.4f}; calibrated tail z {float(best_lstm['tail_quantile_z']):.3f}"
            ),
            "selection_metric": "Conditional-coverage-targeted fine-tune",
            "validation_score": float(best_lstm["validation_cc_pvalue"]),
        },
        {
            "model": "VAE latent VaR",
            "design": "Latent generative return model on rolling windows",
            "best_parameters": f"{int(best_vae['window'])}-hour window; latent dimension {int(best_vae['latent_dim'])}",
            "selection_metric": "Best validation ELBO",
            "validation_score": float(best_vae["validation_loss"]),
        },
    ]

    config = {
        "student_t_window_hours": int(best_student["calibration_window_hours"]),
        "jump_diffusion_window_hours": int(best_jump["calibration_window_hours"]),
        "jump_threshold_sigma": float(best_jump["jump_threshold_sigma"]),
        "lstm_sequence_length": LSTM_FINAL_SEQUENCE_LENGTH,
        "lstm_tail_alpha": LSTM_FINAL_TAIL_ALPHA,
        "vae_window": int(best_vae["window"]),
        "vae_latent_dim": int(best_vae["latent_dim"]),
    }

    return pd.DataFrame(summary_rows), config


def kupiec_pof_test(actual_returns: pd.Series, var_series: pd.Series, confidence: float) -> dict[str, float]:
    aligned = pd.concat([actual_returns.rename("actual"), var_series.rename("var")], axis=1).dropna()
    violations = aligned["actual"] < aligned["var"]
    total = int(len(aligned))
    failures = int(violations.sum())
    expected_probability = 1 - confidence
    observed_probability = failures / total if total else 0.0
    clipped_probability = float(np.clip(observed_probability, 1e-6, 1 - 1e-6))
    likelihood_ratio = -2 * (
        (total - failures) * math.log(1 - expected_probability)
        + failures * math.log(expected_probability)
        - (total - failures) * math.log(1 - clipped_probability)
        - failures * math.log(clipped_probability)
    )
    p_value = float(1 - stats.chi2.cdf(likelihood_ratio, df=1))
    return {
        "observations": float(total),
        "violations": float(failures),
        "expected_violation_rate": float(expected_probability),
        "observed_violation_rate": float(observed_probability),
        "kupiec_lr": float(likelihood_ratio),
        "kupiec_pvalue": p_value,
    }


def christoffersen_independence_test(actual_returns: pd.Series, var_series: pd.Series) -> dict[str, float]:
    aligned = pd.concat([actual_returns.rename("actual"), var_series.rename("var")], axis=1).dropna()
    violations = (aligned["actual"] < aligned["var"]).astype(int).to_numpy(dtype=int)
    if violations.size < 2:
        return {
            "christoffersen_lr_ind": float("nan"),
            "christoffersen_ind_pvalue": float("nan"),
            "violation_transition_01": float("nan"),
            "violation_transition_11": float("nan"),
        }

    prev = violations[:-1]
    curr = violations[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    total_transitions = n00 + n01 + n10 + n11

    pi = float(np.clip((n01 + n11) / total_transitions, 1e-6, 1 - 1e-6))
    pi01 = float(np.clip(n01 / max(n00 + n01, 1), 1e-6, 1 - 1e-6))
    pi11 = float(np.clip(n11 / max(n10 + n11, 1), 1e-6, 1 - 1e-6))

    log_l0 = (n00 + n10) * math.log(1 - pi) + (n01 + n11) * math.log(pi)
    log_l1 = (
        n00 * math.log(1 - pi01)
        + n01 * math.log(pi01)
        + n10 * math.log(1 - pi11)
        + n11 * math.log(pi11)
    )
    lr_ind = float(-2 * (log_l0 - log_l1))
    ind_pvalue = float(1 - stats.chi2.cdf(lr_ind, df=1))
    return {
        "christoffersen_lr_ind": lr_ind,
        "christoffersen_ind_pvalue": ind_pvalue,
        "violation_transition_01": pi01,
        "violation_transition_11": pi11,
    }


def backtest_diagnostics(actual_returns: pd.Series, var_series: pd.Series, confidence: float) -> dict[str, float]:
    kupiec = kupiec_pof_test(actual_returns, var_series, confidence)
    independence = christoffersen_independence_test(actual_returns, var_series)
    lr_ind = independence.get("christoffersen_lr_ind", float("nan"))
    lr_cc = float(kupiec["kupiec_lr"] + lr_ind) if np.isfinite(lr_ind) else float("nan")
    cc_pvalue = float(1 - stats.chi2.cdf(lr_cc, df=2)) if np.isfinite(lr_cc) else float("nan")
    return {
        **kupiec,
        **independence,
        "christoffersen_lr_cc": lr_cc,
        "christoffersen_cc_pvalue": cc_pvalue,
    }


def run_backtests(
    feature_frame: pd.DataFrame,
    confidence: float,
    lstm_predictions: pd.DataFrame,
    vae_var_return: float,
    rng_seed: int,
    gaussian_window_hours: int,
    student_t_window_hours: int,
    jump_diffusion_window_hours: int,
    jump_threshold_sigma: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = feature_frame.set_index("timestamp")["log_return"].dropna()
    backtest_start = returns.index.max() - pd.Timedelta(days=BACKTEST_DAYS)
    actual = returns[returns.index >= backtest_start]
    var_gaussian = pd.Series(index=actual.index, dtype=float)
    var_t = pd.Series(index=actual.index, dtype=float)
    var_mc = pd.Series(index=actual.index, dtype=float)

    for block_start in range(0, len(actual), 24):
        block_index = actual.index[block_start : block_start + 24]
        if len(block_index) == 0:
            continue
        calibration_end = block_index[0] - pd.Timedelta(hours=1)
        gaussian_calibration_sample = returns.loc[:calibration_end].tail(gaussian_window_hours)
        t_calibration_sample = returns.loc[:calibration_end].tail(student_t_window_hours)
        mc_calibration_sample = returns.loc[:calibration_end].tail(jump_diffusion_window_hours)
        if (
            len(gaussian_calibration_sample) < gaussian_window_hours
            or len(t_calibration_sample) < student_t_window_hours
            or len(mc_calibration_sample) < jump_diffusion_window_hours
        ):
            continue
        block_rng = np.random.default_rng(rng_seed + block_start)
        gaussian_summary = fit_gaussian_var(gaussian_calibration_sample, confidence, (1,))[1]
        t_summary = fit_student_t_var(t_calibration_sample, confidence, (1,), block_rng)[1]
        mc_summary = simulate_jump_diffusion_var(
            mc_calibration_sample,
            confidence,
            (1,),
            MONTE_CARLO_PATHS,
            block_rng,
            jump_threshold_sigma=jump_threshold_sigma,
        )[0][1]
        var_gaussian.loc[block_index] = gaussian_summary.var_return
        var_t.loc[block_index] = t_summary.var_return
        var_mc.loc[block_index] = mc_summary.var_return

    var_lstm = lstm_predictions["var_return"].reindex(actual.index)
    var_vae = pd.Series(vae_var_return, index=actual.index, dtype=float)
    backtest_frame = pd.DataFrame(
        {
            "actual_return": actual,
            "var_gaussian": var_gaussian,
            "var_t": var_t,
            "var_mc": var_mc,
            "var_lstm": var_lstm,
            "var_vae": var_vae,
        }
    )
    summary = pd.DataFrame(
        [
            {"model": "Gaussian benchmark", **backtest_diagnostics(actual, var_gaussian, confidence)},
            {"model": "Parametric t-VaR", **backtest_diagnostics(actual, var_t, confidence)},
            {"model": "Jump diffusion Monte Carlo", **backtest_diagnostics(actual, var_mc, confidence)},
            {"model": "LSTM conditional VaR", **backtest_diagnostics(actual, var_lstm, confidence)},
            {"model": "VAE latent VaR", **backtest_diagnostics(actual, var_vae, confidence)},
        ]
    )
    return summary, backtest_frame


def compute_confidence_sensitivity(
    student_t_sample: pd.Series,
    jump_diffusion_sample: pd.Series,
    jump_threshold_sigma: float,
    lstm_result: dict[str, Any],
    vae_result: dict[str, Any],
) -> pd.DataFrame:
    confidence_levels = (0.95, 0.975, 0.99)
    rows: list[dict[str, float | str]] = []
    for offset, confidence in enumerate(confidence_levels):
        rng = np.random.default_rng(RANDOM_SEED + 100 + offset)
        gaussian_forecast = fit_gaussian_var(student_t_sample, confidence, (1,))[1]
        t_forecast = fit_student_t_var(student_t_sample, confidence, (1,), rng)[1]
        mc_forecast = simulate_jump_diffusion_var(
            jump_diffusion_sample,
            confidence,
            (1,),
            MONTE_CARLO_PATHS,
            rng,
            jump_threshold_sigma=jump_threshold_sigma,
        )[0][1]
        lstm_forecast = normal_forecast_from_params(
            lstm_result["latest_mean_return"],
            lstm_result["latest_sigma"],
            confidence,
            horizon_hours=1,
            metadata={"model": "LSTM Gaussian head"},
        )
        vae_forecast = summarize_distribution(
            vae_result["generated_returns_1h"],
            confidence,
            1,
            metadata={"model": "VAE latent sampling"},
        )
        for model_name, forecast in (
            ("Gaussian benchmark", gaussian_forecast),
            ("Parametric Student-t", t_forecast),
            ("Jump diffusion Monte Carlo", mc_forecast),
            ("LSTM", lstm_forecast),
            ("VAE", vae_forecast),
        ):
            rows.append(
                {
                    "confidence_level_pct": confidence * 100,
                    "model": model_name,
                    "VaR_loss": forecast.var_loss,
                    "CVaR_loss": forecast.cvar_loss,
                    "volatility": forecast.volatility,
                }
            )
    return pd.DataFrame(rows).sort_values(["confidence_level_pct", "VaR_loss"], ascending=[True, False]).reset_index(drop=True)


def compute_student_t_parameter_sensitivity(recent_returns: pd.Series, confidence: float) -> pd.DataFrame:
    sample = recent_returns.dropna().to_numpy()
    estimated_df, location, _ = estimate_student_t_parameters(sample)
    sample_std = float(sample.std(ddof=1))
    nu_grid = sorted({4.5, 6.0, 8.0, 12.0, 20.0, round(estimated_df, 2)})
    rows: list[dict[str, float]] = []
    for degrees_of_freedom in nu_grid:
        scale = max(sample_std * math.sqrt((degrees_of_freedom - 2) / degrees_of_freedom), 1e-6)
        var_return = float(stats.t.ppf(1 - confidence, degrees_of_freedom, loc=location, scale=scale))
        rows.append(
            {
                "degrees_of_freedom": degrees_of_freedom,
                "VaR_return": var_return,
                "VaR_loss": -var_return,
                "tail_ratio_to_sigma": (-var_return) / sample_std,
                "is_estimated_setting": float(abs(degrees_of_freedom - estimated_df) < 1e-6),
            }
        )
    return pd.DataFrame(rows).sort_values("degrees_of_freedom").reset_index(drop=True)


def compute_lstm_window_sensitivity(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    backtest_start: pd.Timestamp,
    confidence: float,
    base_result: dict[str, Any],
) -> pd.DataFrame:
    base_lookback_hours = int(base_result["current_forecast"].metadata.get("sequence_length", LSTM_SEQUENCE_LENGTH))
    rows = [
        {
            "lookback_hours": base_lookback_hours,
            "best_validation_loss": min(item["val_loss"] for item in base_result["training_history"]),
            "sigma_tracking_mae": float((base_result["predictions"]["sigma"] - base_result["predictions"]["actual_return"].abs()).abs().mean()),
            "current_VaR_loss": base_result["current_forecast"].var_loss,
            "current_sigma": base_result["current_forecast"].volatility,
            "validation_violation_rate": float(base_result["validation_diagnostics"]["observed_violation_rate"]),
            "validation_kupiec_pvalue": float(base_result["validation_diagnostics"]["kupiec_pvalue"]),
            "validation_ind_pvalue": float(base_result["validation_diagnostics"]["christoffersen_ind_pvalue"]),
            "validation_cc_pvalue": float(base_result["validation_diagnostics"]["christoffersen_cc_pvalue"]),
            "tail_quantile_z": float(base_result["tail_calibration"]["tail_quantile_z"]),
        }
    ]
    for lookback_hours in LSTM_LOOKBACK_CANDIDATES:
        if lookback_hours == base_lookback_hours:
            continue
        datasets = build_lstm_datasets(feature_frame, feature_columns, lookback_hours, backtest_start)
        result = fit_lstm_model(
            datasets,
            len(feature_columns),
            confidence,
            sequence_length=lookback_hours,
            epochs=4,
        )
        rows.append(
            {
                "lookback_hours": lookback_hours,
                "best_validation_loss": min(item["val_loss"] for item in result["training_history"]),
                "sigma_tracking_mae": float((result["predictions"]["sigma"] - result["predictions"]["actual_return"].abs()).abs().mean()),
                "current_VaR_loss": result["current_forecast"].var_loss,
                "current_sigma": result["current_forecast"].volatility,
                "validation_violation_rate": float(result["validation_diagnostics"]["observed_violation_rate"]),
                "validation_kupiec_pvalue": float(result["validation_diagnostics"]["kupiec_pvalue"]),
                "validation_ind_pvalue": float(result["validation_diagnostics"]["christoffersen_ind_pvalue"]),
                "validation_cc_pvalue": float(result["validation_diagnostics"]["christoffersen_cc_pvalue"]),
                "tail_quantile_z": float(result["tail_calibration"]["tail_quantile_z"]),
            }
        )
    return pd.DataFrame(rows).sort_values("lookback_hours").reset_index(drop=True)


def evaluate_lstm_candidates(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    backtest_start: pd.Timestamp,
    confidence: float,
) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    results: dict[int, dict[str, Any]] = {}
    for lookback_hours in LSTM_LOOKBACK_CANDIDATES:
        datasets = build_lstm_datasets(feature_frame, feature_columns, lookback_hours, backtest_start)
        result = fit_lstm_model(
            datasets,
            len(feature_columns),
            confidence,
            sequence_length=lookback_hours,
            hidden_dim=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT,
            learning_rate=LSTM_LEARNING_RATE,
            batch_size=LSTM_BATCH_SIZE,
            epochs=LSTM_FINAL_EPOCHS,
        )
        results[lookback_hours] = result
        rows.append(
            {
                "lookback_hours": lookback_hours,
                "best_validation_loss": min(item["val_loss"] for item in result["training_history"]),
                "sigma_tracking_mae": float((result["predictions"]["sigma"] - result["predictions"]["actual_return"].abs()).abs().mean()),
                "current_VaR_loss": result["current_forecast"].var_loss,
                "current_sigma": result["current_forecast"].volatility,
                "validation_violation_rate": float(result["validation_diagnostics"]["observed_violation_rate"]),
                "validation_kupiec_pvalue": float(result["validation_diagnostics"]["kupiec_pvalue"]),
                "validation_ind_pvalue": float(result["validation_diagnostics"]["christoffersen_ind_pvalue"]),
                "validation_cc_pvalue": float(result["validation_diagnostics"]["christoffersen_cc_pvalue"]),
                "tail_quantile_z": float(result["tail_calibration"]["tail_quantile_z"]),
            }
        )
    return pd.DataFrame(rows).sort_values("lookback_hours").reset_index(drop=True), results


def recalibrate_lstm_result(result: dict[str, Any], tail_alpha: float) -> dict[str, Any]:
    validation_predictions = result["validation_predictions"]
    recalibration = calibrate_standardized_tail(
        validation_predictions["actual_return"].to_numpy(dtype=float),
        validation_predictions["mean_return"].to_numpy(dtype=float),
        validation_predictions["sigma"].to_numpy(dtype=float),
        tail_alpha,
    )
    recalibrated_validation_predictions = build_lstm_prediction_frame(
        validation_predictions["mean_return"].to_numpy(dtype=float),
        validation_predictions["sigma"].to_numpy(dtype=float),
        validation_predictions["actual_return"].to_numpy(dtype=float),
        validation_predictions.index,
        recalibration["tail_quantile_z"],
        recalibration["tail_cvar_z"],
    )
    recalibrated_predictions = build_lstm_prediction_frame(
        result["predictions"]["mean_return"].to_numpy(dtype=float),
        result["predictions"]["sigma"].to_numpy(dtype=float),
        result["predictions"]["actual_return"].to_numpy(dtype=float),
        result["predictions"].index,
        recalibration["tail_quantile_z"],
        recalibration["tail_cvar_z"],
    )
    validation_actual = pd.Series(
        recalibrated_validation_predictions["actual_return"].to_numpy(dtype=float),
        index=recalibrated_validation_predictions.index,
        dtype=float,
    )
    recalibrated_forecast = forecast_from_standardized_tail(
        result["latest_mean_return"],
        result["latest_sigma"],
        recalibration["tail_quantile_z"],
        recalibration["tail_cvar_z"],
        horizon_hours=1,
        metadata={
            **result["current_forecast"].metadata,
            "tail_calibration": recalibration,
        },
    )
    return {
        **result,
        "current_forecast": recalibrated_forecast,
        "predictions": recalibrated_predictions,
        "validation_predictions": recalibrated_validation_predictions,
        "validation_diagnostics": backtest_diagnostics(validation_actual, recalibrated_validation_predictions["var_return"], DEFAULT_CONFIDENCE),
        "tail_calibration": recalibration,
    }


def select_event_timestamp(returns: pd.Series, start: str, end: str, mode: str) -> pd.Timestamp:
    window = returns.loc[start:end]
    if window.empty:
        raise RuntimeError(f"No returns found in event window {start} to {end}.")
    if mode == "loss":
        return window.idxmin()
    if mode == "absolute":
        return window.abs().idxmax()
    raise ValueError(f"Unsupported event mode {mode!r}")


def build_pre_event_warning_frame(
    returns: pd.Series,
    event_timestamp: pd.Timestamp,
    confidence: float,
    rng_seed: int,
    student_t_window_hours: int,
    jump_diffusion_window_hours: int,
    jump_threshold_sigma: float,
) -> pd.DataFrame:
    start_timestamp = event_timestamp - pd.Timedelta(hours=24)
    evaluation_index = pd.date_range(start_timestamp, event_timestamp - pd.Timedelta(hours=1), freq="h", tz="UTC")
    rows: list[dict[str, Any]] = []
    for offset, forecast_timestamp in enumerate(evaluation_index):
        student_t_sample = returns.loc[: forecast_timestamp - pd.Timedelta(hours=1)].tail(student_t_window_hours)
        jump_diffusion_sample = returns.loc[: forecast_timestamp - pd.Timedelta(hours=1)].tail(jump_diffusion_window_hours)
        if len(student_t_sample) < student_t_window_hours or len(jump_diffusion_sample) < jump_diffusion_window_hours:
            continue
        block_rng = np.random.default_rng(rng_seed + offset)
        t_forecast = fit_student_t_var(student_t_sample, confidence, (1,), block_rng)[1]
        mc_forecast = simulate_jump_diffusion_var(
            jump_diffusion_sample,
            confidence,
            (1,),
            MONTE_CARLO_PATHS,
            block_rng,
            jump_threshold_sigma=jump_threshold_sigma,
        )[0][1]
        rows.append(
            {
                "forecast_timestamp": forecast_timestamp,
                "t_var_loss": t_forecast.var_loss,
                "mc_var_loss": mc_forecast.var_loss,
            }
        )
    return pd.DataFrame(rows)


def compute_stress_test_analysis(
    feature_frame: pd.DataFrame,
    confidence: float,
    student_t_window_hours: int,
    jump_diffusion_window_hours: int,
    jump_threshold_sigma: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    returns = feature_frame.set_index("timestamp")["log_return"].dropna()
    event_specs = [
        {
            "event": "FTX Collapse",
            "selection_start": "2022-11-07T00:00:00+00:00",
            "selection_end": "2022-11-10T23:00:00+00:00",
            "selection_mode": "loss",
            "scope": "pre-specified case study",
        },
        {
            "event": "Spot ETF Approval Whipsaw",
            "selection_start": "2024-01-10T00:00:00+00:00",
            "selection_end": "2024-01-12T23:00:00+00:00",
            "selection_mode": "absolute",
            "scope": "pre-specified case study",
        },
    ]
    summary_rows: list[dict[str, Any]] = []
    warning_frames: dict[str, pd.DataFrame] = {}
    for index, event_spec in enumerate(event_specs):
        event_name = str(event_spec["event"])
        start = str(event_spec["selection_start"])
        end = str(event_spec["selection_end"])
        mode = str(event_spec["selection_mode"])
        event_timestamp = select_event_timestamp(returns, start, end, mode)
        actual_return = float(returns.loc[event_timestamp])
        actual_loss = float(-actual_return)
        student_t_sample = returns.loc[: event_timestamp - pd.Timedelta(hours=1)].tail(student_t_window_hours)
        jump_diffusion_sample = returns.loc[: event_timestamp - pd.Timedelta(hours=1)].tail(jump_diffusion_window_hours)
        event_rng = np.random.default_rng(RANDOM_SEED + 500 + index)
        t_forecast = fit_student_t_var(student_t_sample, confidence, (1,), event_rng)[1]
        mc_forecast = simulate_jump_diffusion_var(
            jump_diffusion_sample,
            confidence,
            (1,),
            MONTE_CARLO_PATHS,
            event_rng,
            jump_threshold_sigma=jump_threshold_sigma,
        )[0][1]
        warning_frame = build_pre_event_warning_frame(
            returns,
            event_timestamp,
            confidence,
            RANDOM_SEED + 700 + index * 100,
            student_t_window_hours,
            jump_diffusion_window_hours,
            jump_threshold_sigma,
        )
        warning_threshold = 0.5 * actual_loss
        t_peak_row = warning_frame.loc[warning_frame["t_var_loss"].idxmax()]
        mc_peak_row = warning_frame.loc[warning_frame["mc_var_loss"].idxmax()]
        t_peak_hours = float((event_timestamp - t_peak_row["forecast_timestamp"]).total_seconds() / 3600)
        mc_peak_hours = float((event_timestamp - mc_peak_row["forecast_timestamp"]).total_seconds() / 3600)
        strongest_warning_model = "Jump diffusion Monte Carlo" if mc_peak_row["mc_var_loss"] >= t_peak_row["t_var_loss"] else "Parametric Student-t"
        closest_model = (
            "Jump diffusion Monte Carlo"
            if abs(actual_loss - mc_forecast.var_loss) <= abs(actual_loss - t_forecast.var_loss)
            else "Parametric Student-t"
        )
        warning_frame = warning_frame.copy()
        warning_frame["warning_threshold"] = warning_threshold
        warning_frame["actual_event_loss"] = actual_loss
        warning_frames[event_name] = warning_frame
        summary_rows.append(
            {
                "event": event_name,
                "event_timestamp": event_timestamp.isoformat(),
                "selection_start": start,
                "selection_end": end,
                "selection_mode": mode,
                "warning_horizon_hours": 24.0,
                "event_scope": str(event_spec["scope"]),
                "actual_return": actual_return,
                "actual_loss": actual_loss,
                "student_t_VaR_loss": t_forecast.var_loss,
                "mc_VaR_loss": mc_forecast.var_loss,
                "student_t_peak_warning_loss": float(t_peak_row["t_var_loss"]),
                "mc_peak_warning_loss": float(mc_peak_row["mc_var_loss"]),
                "student_t_peak_warning_hours_before_event": t_peak_hours,
                "mc_peak_warning_hours_before_event": mc_peak_hours,
                "strongest_warning_model": strongest_warning_model,
                "closest_model_to_realized_loss": closest_model,
            }
        )
    return pd.DataFrame(summary_rows), warning_frames


def compute_time_scale_diagnostics(returns: pd.Series, confidence: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_var = float(-returns.quantile(1 - confidence))
    daily_returns = returns.resample("1D").sum().dropna()
    ten_day_returns = daily_returns.rolling(10).sum().dropna()
    scaling_rows = [
        {
            "horizon": "1 day",
            "sqrt_time_scaled_VaR": hourly_var * math.sqrt(24),
            "empirical_VaR": float(-daily_returns.quantile(1 - confidence)),
        },
        {
            "horizon": "10 days",
            "sqrt_time_scaled_VaR": hourly_var * math.sqrt(24 * 10),
            "empirical_VaR": float(-ten_day_returns.quantile(1 - confidence)),
        },
    ]
    scaling_table = pd.DataFrame(scaling_rows)
    scaling_table["scaling_bias_pct"] = (
        (scaling_table["sqrt_time_scaled_VaR"] - scaling_table["empirical_VaR"]) / scaling_table["empirical_VaR"]
    ) * 100
    lb_returns = acorr_ljungbox(returns, lags=[1, 24], return_df=True)
    lb_absolute = acorr_ljungbox(returns.abs(), lags=[1, 24], return_df=True)
    autocorr_table = pd.DataFrame(
        [
            {
                "return_autocorr_lag1": float(returns.autocorr(1)),
                "return_autocorr_lag24": float(returns.autocorr(24)),
                "abs_return_autocorr_lag1": float(returns.abs().autocorr(1)),
                "abs_return_autocorr_lag24": float(returns.abs().autocorr(24)),
                "ljung_box_return_pvalue_lag24": float(lb_returns.loc[24, "lb_pvalue"]),
                "ljung_box_abs_return_pvalue_lag24": float(lb_absolute.loc[24, "lb_pvalue"]),
            }
        ]
    )
    return scaling_table, autocorr_table


def compute_capital_requirements(current_var_table: pd.DataFrame, notional_usd: float = 1_000_000) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in current_var_table.iterrows():
        if int(row["horizon_hours"]) not in (1, 24):
            continue
        rows.append(
            {
                "position_notional_usd": notional_usd,
                "model": row["model"],
                "horizon_hours": int(row["horizon_hours"]),
                "VaR_capital_usd": float(notional_usd * row["VaR_loss"]),
                "CVaR_capital_usd": float(notional_usd * row["CVaR_loss"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon_hours", "VaR_capital_usd"], ascending=[True, False]).reset_index(drop=True)


def to_markdown_table(frame: pd.DataFrame, precision: int = 4) -> str:
    display_frame = format_table_values(frame, precision)
    headers = [str(column) for column in display_frame.columns]
    separator = ["---"] * len(headers)
    rows = [headers, separator] + display_frame.astype(str).values.tolist()
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def format_numeric_value(value: Any, precision: int = 4) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        rounded = f"{float(value):.{precision}f}"
        if "." in rounded:
            rounded = rounded.rstrip("0").rstrip(".")
        if rounded == "-0":
            rounded = "0"
        return rounded
    return str(value)


def format_table_values(frame: pd.DataFrame, precision: int = 4) -> pd.DataFrame:
    display_frame = frame.copy()
    for column in frame.columns:
        column_name = str(column).lower()
        source_series = frame[column]
        if "p-val" in column_name or "pvalue" in column_name or "p-value" in column_name:
            threshold = 10 ** (-precision)
            display_frame[column] = source_series.map(
                lambda value: f"<{threshold:.{precision}f}" if not pd.isna(value) and float(value) <= threshold else format_numeric_value(value, precision)
            )
        elif pd.api.types.is_numeric_dtype(source_series):
            display_frame[column] = source_series.map(lambda value: format_numeric_value(value, precision))
    return display_frame


REPORT_MODEL_LABELS = {
    "Gaussian benchmark": "Gaussian VaR",
    "Jump diffusion Monte Carlo": "Jump-diffusion MC",
    "Parametric Student-t": "Student-t param.",
    "Parametric t-VaR": "Student-t VaR",
    "LSTM conditional VaR": "LSTM cond. VaR",
    "VAE latent VaR": "VAE latent VaR",
}

REPORT_EVENT_LABELS = {
    "Spot ETF Approval Whipsaw": "ETF whipsaw",
}

ACADEMIC_REFERENCES = (
    {
        "key": "jorion2007",
        "cite": "Jorion, 2007",
        "text": "Jorion, Philippe. 2007. Value at Risk: The New Benchmark for Managing Financial Risk. 3rd ed. New York: McGraw-Hill.",
    },
    {
        "key": "mcneil2015",
        "cite": "McNeil, Frey, and Embrechts, 2015",
        "text": "McNeil, Alexander J., Rudiger Frey, and Paul Embrechts. 2015. Quantitative Risk Management: Concepts, Techniques and Tools. Revised ed. Princeton: Princeton University Press.",
    },
    {
        "key": "kupiec1995",
        "cite": "Kupiec, 1995",
        "text": "Kupiec, Paul H. 1995. Techniques for Verifying the Accuracy of Risk Measurement Models. Journal of Derivatives 3(2): 73-84.",
    },
    {
        "key": "christoffersen1998",
        "cite": "Christoffersen, 1998",
        "text": "Christoffersen, Peter F. 1998. Evaluating Interval Forecasts. International Economic Review 39(4): 841-862.",
    },
    {
        "key": "merton1976",
        "cite": "Merton, 1976",
        "text": "Merton, Robert C. 1976. Option Pricing When Underlying Stock Returns Are Discontinuous. Journal of Financial Economics 3(1-2): 125-144.",
    },
    {
        "key": "engle1982",
        "cite": "Engle, 1982",
        "text": "Engle, Robert F. 1982. Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. Econometrica 50(4): 987-1007.",
    },
    {
        "key": "hochreiter1997",
        "cite": "Hochreiter and Schmidhuber, 1997",
        "text": "Hochreiter, Sepp, and Jurgen Schmidhuber. 1997. Long Short-Term Memory. Neural Computation 9(8): 1735-1780.",
    },
    {
        "key": "kingma2014",
        "cite": "Kingma and Welling, 2014",
        "text": "Kingma, Diederik P., and Max Welling. 2014. Auto-Encoding Variational Bayes. 2nd International Conference on Learning Representations.",
    },
)

REFERENCE_LOOKUP = {entry["key"]: entry for entry in ACADEMIC_REFERENCES}

METHOD_OVERVIEW_ROWS = [
    {
        "Method": "A",
        "Model": "Gaussian VaR",
        "Primary role": "Thin-tailed rolling benchmark used to test whether a Gaussian assumption is adequate.",
        "Main trade-off": "Transparent baseline, but structurally weak under heavy tails and volatility clustering.",
    },
    {
        "Method": "B",
        "Model": "Student-t VaR",
        "Primary role": "Parametric heavy-tail benchmark using a rolling return window.",
        "Main trade-off": "Transparent and fast, but only weakly conditional on current market state.",
    },
    {
        "Method": "C",
        "Model": "Jump-diffusion Monte Carlo",
        "Primary role": "Scenario engine that combines continuous diffusion with discrete jump risk.",
        "Main trade-off": "More realistic under stress, but sensitive to jump-calibration choices.",
    },
    {
        "Method": "D",
        "Model": "LSTM Gaussian head",
        "Primary role": "Feature-conditioned one-step forecast of mean and volatility.",
        "Main trade-off": "Adaptive to regime shifts, but harder to validate and explain.",
    },
    {
        "Method": "E",
        "Model": "VAE latent sampling",
        "Primary role": "Generative model for synthetic return scenarios and latent tail structure.",
        "Main trade-off": "Useful for scenario enrichment, but weaker as a standalone production VaR engine.",
    },
]

REPORT_TODO_SECTIONS = (
    (
        "Completed in this implementation",
        (
            "Reframed the generated report into an academic paper structure with abstract, introduction, methodology, discussion, conclusion, and references.",
            "Added a computed sample split summary so the report states the chronological train, validation, and backtest design explicitly.",
            "Generated a synchronized TeX paper and repository TODO artifact from the Python pipeline outputs.",
            "Expanded the narrative around model failures, stress-test interpretation, and limitations for university-style reporting.",
        ),
    ),
    (
        "Recommended next extensions",
        (
            "Add Christoffersen conditional coverage tests to complement the existing Kupiec coverage test.",
            "Extend the neural models to direct multi-step forecasting so the 24-hour comparison includes a conditional recurrent benchmark.",
            "Introduce liquidity-adjusted VaR or slippage-aware stress testing for execution-sensitive capital analysis.",
            "Replace the proxy benchmark with a fiat BTC/USD series or an explicit basis adjustment workflow.",
        ),
    ),
)


def report_label(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, value)


def human_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def markdown_numbered_list(items: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def academic_citation(*keys: str) -> str:
    return "(" + "; ".join(REFERENCE_LOOKUP[key]["cite"] for key in keys) + ")"


def render_references_markdown() -> str:
    return markdown_numbered_list([entry["text"] for entry in ACADEMIC_REFERENCES])


def render_references_latex() -> str:
    lines = ["\\begin{enumerate}"]
    lines.extend(f"\\item {latex_escape(entry['text'])}" for entry in ACADEMIC_REFERENCES)
    lines.append("\\end{enumerate}")
    return "\n".join(lines)


def render_report_todo_markdown() -> str:
    sections = ["# Report Upgrade TODO"]
    for heading, items in REPORT_TODO_SECTIONS:
        sections.append("")
        sections.append(f"## {heading}")
        sections.extend(f"- [x] {item}" if heading == "Completed in this implementation" else f"- [ ] {item}" for item in items)
    return "\n".join(sections) + "\n"


def render_method_overview_markdown() -> str:
    return to_markdown_table(pd.DataFrame(METHOD_OVERVIEW_ROWS))


def render_method_overview_latex() -> str:
    row_end = r"\\"
    lines = [
        "\\begingroup",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{center}",
        "\\begin{tabularx}{0.98\\linewidth}{@{}>{\\RaggedRight\\arraybackslash}p{0.10\\linewidth}>{\\RaggedRight\\arraybackslash}p{0.18\\linewidth}>{\\RaggedRight\\arraybackslash}X>{\\RaggedRight\\arraybackslash}X@{}}",
        "\\toprule",
        f"Method & Model & Primary role & Main trade-off {row_end}",
        "\\midrule",
        f"A & Gaussian VaR & Thin-tailed rolling benchmark used to test whether a Gaussian assumption is adequate. & Transparent baseline, but structurally weak under heavy tails and volatility clustering. {row_end}",
        f"B & Student-t VaR & Parametric heavy-tail benchmark using a rolling return window. & Transparent and fast, but only weakly conditional on current market state. {row_end}",
        f"C & Jump-diffusion Monte Carlo & Scenario engine that combines continuous diffusion with discrete jump risk. & More realistic under stress, but sensitive to jump-calibration choices. {row_end}",
        f"D & LSTM Gaussian head & Feature-conditioned one-step forecast of mean and volatility. & Adaptive to regime shifts, but harder to validate and explain. {row_end}",
        f"E & VAE latent sampling & Generative model for synthetic return scenarios and latent tail structure. & Useful for scenario enrichment, but weaker as a standalone production VaR engine. {row_end}",
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{center}",
        "\\endgroup",
    ]
    return "\n".join(lines)


def format_report_timestamp(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        timestamp = value
    else:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return value
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_report_table(frame: pd.DataFrame, rename_columns: dict[str, str] | None = None) -> pd.DataFrame:
    display_frame = frame.copy()

    for column in (
        "model",
        "Model",
        "strongest_warning_model",
        "Strongest warning",
        "closest_model_to_realized_loss",
        "Closest model",
    ):
        if column in display_frame.columns:
            display_frame[column] = display_frame[column].replace(REPORT_MODEL_LABELS)

    for column in ("event", "Event"):
        if column in display_frame.columns:
            display_frame[column] = display_frame[column].replace(REPORT_EVENT_LABELS)

    for column in ("event_timestamp", "Event time", "start", "Start", "end", "End"):
        if column in display_frame.columns:
            display_frame[column] = display_frame[column].map(format_report_timestamp)

    if rename_columns:
        display_frame = display_frame.rename(columns=rename_columns)

    return display_frame


def write_latex_table(
    path: Path,
    frame: pd.DataFrame,
    *,
    precision: int = 4,
    column_format: str,
    font_size: str = "footnotesize",
    tabcolsep: str = "3pt",
    longtable: bool = True,
    center: bool = False,
) -> None:
    display_frame = format_table_values(frame, precision).astype(str)
    latex = display_frame.to_latex(index=False, escape=True, longtable=longtable, column_format=column_format)
    if center:
        latex = f"\\begin{{center}}\n{latex}\\end{{center}}\n"
    path.write_text(
        f"\\begingroup\n\\{font_size}\n\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n{latex}\\endgroup\n",
        encoding="utf-8",
    )


def build_sample_split_summary(
    data_quality: dict[str, Any],
    returns: pd.Series,
    backtest_start: pd.Timestamp,
    lstm_datasets: dict[str, Any],
    vae_split_summary: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(block: str, start: Any, end: Any, observations: int, design_note: str) -> None:
        rows.append(
            {
                "block": block,
                "start": pd.Timestamp(start),
                "end": pd.Timestamp(end),
                "observations": int(observations),
                "design_note": design_note,
            }
        )

    sample_start = pd.Timestamp(data_quality["start"])
    sample_end = pd.Timestamp(data_quality["end"])
    pre_backtest_returns = returns[returns.index < backtest_start]
    backtest_returns = returns[returns.index >= backtest_start]
    tuning_end = backtest_start - pd.Timedelta(hours=1)
    tuning_start = backtest_start - pd.Timedelta(days=TUNING_VALIDATION_DAYS)
    tuning_index = returns.loc[tuning_start:tuning_end].index[::TUNING_EVAL_STRIDE_HOURS]

    add_row(
        "Full cleaned sample",
        sample_start,
        sample_end,
        int(data_quality["cleaned_rows"]),
        "Hourly cleaned candles used for descriptive statistics, figures, and feature construction.",
    )
    add_row(
        "Pre-backtest return history",
        pre_backtest_returns.index.min(),
        pre_backtest_returns.index.max(),
        len(pre_backtest_returns),
        "History available for model estimation before the final one-year holdout.",
    )
    add_row(
        "Tuning validation slice",
        tuning_index.min(),
        tuning_index.max(),
        len(tuning_index),
        f"Chronological validation checkpoints every {TUNING_EVAL_STRIDE_HOURS} hours for Student-t and jump-diffusion tuning.",
    )
    add_row(
        "Backtest holdout",
        backtest_returns.index.min(),
        backtest_returns.index.max(),
        len(backtest_returns),
        "Final one-year out-of-sample window used for Kupiec coverage testing.",
    )

    train_index = pd.DatetimeIndex(lstm_datasets["train_index"])
    val_index = pd.DatetimeIndex(lstm_datasets["val_index"])
    test_index = pd.DatetimeIndex(lstm_datasets["test_index"])
    lookback_hours = int(lstm_datasets["X_train"].shape[1])
    add_row(
        "LSTM fit sequences",
        train_index.min(),
        train_index.max(),
        len(train_index),
        f"Chronological {lookback_hours}-hour feature sequences used for recurrent model fitting.",
    )
    add_row(
        "LSTM validation sequences",
        val_index.min(),
        val_index.max(),
        len(val_index),
        "Final 20% of pre-backtest sequence targets reserved for neural validation.",
    )
    add_row(
        "LSTM backtest sequences",
        test_index.min(),
        test_index.max(),
        len(test_index),
        "Out-of-sample sequence targets used for the one-step conditional backtest.",
    )
    add_row(
        "VAE train windows",
        vae_split_summary["train_start"],
        vae_split_summary["train_end"],
        int(vae_split_summary["train_window_count"]),
        f"Rolling {int(vae_split_summary['window_hours'])}-hour windows used to estimate the latent generative model.",
    )
    add_row(
        "VAE validation windows",
        vae_split_summary["validation_start"],
        vae_split_summary["validation_end"],
        int(vae_split_summary["validation_window_count"]),
        "Final 20% of pre-backtest rolling windows used for chronological latent-model validation.",
    )
    return pd.DataFrame(rows)


def export_analysis_tables(
    paths: ProjectPaths,
    data_quality: dict[str, Any],
    descriptive_stats: dict[str, float],
    sample_split_summary: pd.DataFrame,
    current_var_table: pd.DataFrame,
    model_tuning_summary: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    confidence_sensitivity: pd.DataFrame,
    student_t_sensitivity: pd.DataFrame,
    lstm_window_sensitivity: pd.DataFrame,
    stress_test_table: pd.DataFrame,
    scaling_table: pd.DataFrame,
    autocorr_table: pd.DataFrame,
    capital_table: pd.DataFrame,
) -> dict[str, str]:
    data_quality_display = format_report_table(
        pd.DataFrame([data_quality]).rename(
            columns={
                "original_rows": "Original rows",
                "cleaned_rows": "Clean rows",
                "missing_hours_filled": "Missing hours",
                "duplicate_rows_removed": "Duplicates removed",
                "flash_crash_repairs": "Flash-crash repairs",
                "start": "Start",
                "end": "End",
            }
        )
    )
    descriptive_frame = pd.DataFrame([descriptive_stats]).rename(
        columns={
            "observations": "Obs.",
            "mean": "Mean",
            "std": "Std.",
            "skewness": "Skew",
            "kurtosis": "Kurtosis",
            "jarque_bera_stat": "JB stat",
            "jarque_bera_pvalue": "JB p-val",
            "adf_stat": "ADF stat",
            "adf_pvalue": "ADF p-val",
            "adf_critical_1pct": "ADF 1% crit",
            "adf_critical_5pct": "ADF 5% crit",
            "adf_critical_10pct": "ADF 10% crit",
            "min": "Min",
            "max": "Max",
            "p01": "P1",
            "p99": "P99",
        }
    )
    descriptive_main = descriptive_frame[["Obs.", "Mean", "Std.", "Skew", "Kurtosis", "JB stat", "JB p-val", "ADF stat"]]
    descriptive_tail = descriptive_frame[["ADF p-val", "ADF 1% crit", "ADF 5% crit", "ADF 10% crit", "Min", "Max", "P1", "P99"]]
    split_display = format_report_table(
        sample_split_summary.rename(
            columns={
                "block": "Block",
                "start": "Start",
                "end": "End",
                "observations": "Observations",
                "design_note": "Design note",
            }
        )
    )
    current_var_display = format_report_table(
        current_var_table.rename(
            columns={
                "model": "Model",
                "horizon_hours": "Horizon (h)",
                "VaR_return": "VaR ret.",
                "CVaR_return": "CVaR ret.",
                "VaR_loss": "VaR loss",
                "CVaR_loss": "CVaR loss",
                "mean_return": "Mean ret.",
                "volatility": "Vol.",
            }
        )
    )
    model_tuning_display = format_report_table(
        model_tuning_summary.rename(
            columns={
                "model": "Model",
                "design": "Design",
                "best_parameters": "Best parameters",
                "selection_metric": "Selection metric",
                "validation_score": "Validation score",
            }
        )
    )
    confidence_display = format_report_table(
        confidence_sensitivity.rename(
            columns={
                "confidence_level_pct": "Conf. (%)",
                "model": "Model",
                "VaR_loss": "VaR loss",
                "CVaR_loss": "CVaR loss",
                "volatility": "Vol.",
            }
        )
    )
    student_t_display = student_t_sensitivity.rename(
        columns={
            "degrees_of_freedom": "DoF",
            "VaR_return": "VaR ret.",
            "VaR_loss": "VaR loss",
            "tail_ratio_to_sigma": "Tail ratio / sigma",
            "is_estimated_setting": "Estimated fit",
        }
    )
    lstm_display = lstm_window_sensitivity.rename(
        columns={
            "lookback_hours": "Look-back (h)",
            "best_validation_loss": "Best val. loss",
            "sigma_tracking_mae": "Sigma MAE",
            "current_VaR_loss": "Current VaR loss",
            "current_sigma": "Current sigma",
            "validation_violation_rate": "Val fail rate",
            "validation_kupiec_pvalue": "Val POF p",
            "validation_ind_pvalue": "Val IND p",
            "validation_cc_pvalue": "Val CC p",
            "tail_quantile_z": "Tail z",
        }
    )
    stress_display = format_report_table(stress_test_table)
    stress_overview = stress_display[
        [
            "event",
            "event_timestamp",
            "actual_return",
            "actual_loss",
            "student_t_VaR_loss",
            "mc_VaR_loss",
            "closest_model_to_realized_loss",
        ]
    ].rename(
        columns={
            "event": "Event",
            "event_timestamp": "Event time",
            "actual_return": "Realized return",
            "actual_loss": "Realized loss",
            "student_t_VaR_loss": "t-VaR loss",
            "mc_VaR_loss": "MC VaR loss",
            "closest_model_to_realized_loss": "Closest model",
        }
    )
    stress_warning = stress_display[
        [
            "event",
            "student_t_peak_warning_loss",
            "mc_peak_warning_loss",
            "student_t_peak_warning_hours_before_event",
            "mc_peak_warning_hours_before_event",
            "strongest_warning_model",
        ]
    ].rename(
        columns={
            "event": "Event",
            "student_t_peak_warning_loss": "t peak warning",
            "mc_peak_warning_loss": "MC peak warning",
            "student_t_peak_warning_hours_before_event": "t peak lead (h)",
            "mc_peak_warning_hours_before_event": "MC peak lead (h)",
            "strongest_warning_model": "Strongest warning",
        }
    )
    scaling_display = scaling_table.rename(
        columns={
            "horizon": "Horizon",
            "sqrt_time_scaled_VaR": "Scaled VaR",
            "empirical_VaR": "Empirical VaR",
            "scaling_bias_pct": "Scaling bias %",
        }
    )
    autocorr_display = autocorr_table.rename(
        columns={
            "return_autocorr_lag1": "r(1)",
            "return_autocorr_lag24": "r(24)",
            "abs_return_autocorr_lag1": "|r|(1)",
            "abs_return_autocorr_lag24": "|r|(24)",
            "ljung_box_return_pvalue_lag24": "LB p-value r(24)",
            "ljung_box_abs_return_pvalue_lag24": "LB p-value |r|(24)",
        }
    )
    capital_display = format_report_table(
        capital_table.rename(
            columns={
                "position_notional_usd": "Notional USD",
                "model": "Model",
                "horizon_hours": "Horizon (h)",
                "VaR_capital_usd": "VaR capital",
                "CVaR_capital_usd": "CVaR capital",
            }
        )
    )
    backtest_display = format_report_table(
        backtest_summary[
            [
                "model",
                "observations",
                "violations",
                "expected_violation_rate",
                "observed_violation_rate",
                "kupiec_pvalue",
                "christoffersen_ind_pvalue",
                "christoffersen_cc_pvalue",
            ]
        ].rename(
            columns={
                "model": "Model",
                "observations": "Obs.",
                "violations": "Viol.",
                "expected_violation_rate": "Exp. fail rate",
                "observed_violation_rate": "Obs. fail rate",
                "kupiec_pvalue": "POF p-value",
                "christoffersen_ind_pvalue": "IND p-value",
                "christoffersen_cc_pvalue": "CC p-value",
            }
        )
    )

    table_specs = [
        (
            "data_quality.tex",
            data_quality_display,
            4,
            "@{}>{\\RaggedLeft\\arraybackslash}p{0.10\\linewidth}>{\\RaggedLeft\\arraybackslash}p{0.10\\linewidth}>{\\RaggedLeft\\arraybackslash}p{0.11\\linewidth}>{\\RaggedLeft\\arraybackslash}p{0.12\\linewidth}>{\\RaggedLeft\\arraybackslash}p{0.12\\linewidth}>{\\RaggedRight\\arraybackslash}p{0.16\\linewidth}>{\\RaggedRight\\arraybackslash}p{0.16\\linewidth}@{}",
            "scriptsize",
            "2pt",
        ),
        ("descriptive_statistics_main.tex", descriptive_main, 4, "@{}llllllll@{}", "footnotesize", "3pt"),
        ("descriptive_statistics_tail.tex", descriptive_tail, 4, "@{}llllllll@{}", "footnotesize", "3pt"),
        (
            "sample_split_summary.tex",
            split_display,
            0,
            "@{}>{\\RaggedRight\\arraybackslash}p{0.17\\linewidth}>{\\RaggedRight\\arraybackslash}p{0.13\\linewidth}>{\\RaggedRight\\arraybackslash}p{0.13\\linewidth}>{\\RaggedLeft\\arraybackslash}p{0.10\\linewidth}>{\\RaggedRight\\arraybackslash}p{0.39\\linewidth}@{}",
            "scriptsize",
            "2pt",
        ),
        (
            "current_var_estimates.tex",
            current_var_display,
            4,
            "@{}>{\\RaggedRight\\arraybackslash}p{0.18\\linewidth}*{7}{>{\\RaggedLeft\\arraybackslash}p{0.09\\linewidth}}@{}",
            "scriptsize",
            "2pt",
        ),
        (
            "model_tuning_summary.tex",
            model_tuning_display,
            4,
            "@{}>{\\RaggedRight\\arraybackslash}p{0.15\\linewidth}>{\\RaggedRight\\arraybackslash}p{0.24\\linewidth}>{\\RaggedRight\\arraybackslash}p{0.31\\linewidth}>{\\RaggedRight\\arraybackslash}p{0.16\\linewidth}>{\\RaggedLeft\\arraybackslash}p{0.10\\linewidth}@{}",
            "scriptsize",
            "2pt",
        ),
        (
            "confidence_sensitivity.tex",
            confidence_display,
            4,
            "@{}>{\\RaggedLeft\\arraybackslash}p{0.11\\linewidth}>{\\RaggedRight\\arraybackslash}p{0.24\\linewidth}*{3}{>{\\RaggedLeft\\arraybackslash}p{0.12\\linewidth}}@{}",
            "footnotesize",
            "3pt",
        ),
        ("student_t_df_sensitivity.tex", student_t_display, 4, "@{}lllll@{}", "footnotesize", "2pt"),
        (
            "lstm_window_sensitivity.tex",
            lstm_display,
            4,
            "@{}>{\\RaggedLeft\\arraybackslash}p{0.07\\linewidth}*{4}{>{\\RaggedLeft\\arraybackslash}p{0.08\\linewidth}}*{4}{>{\\RaggedLeft\\arraybackslash}p{0.11\\linewidth}}>{\\RaggedLeft\\arraybackslash}p{0.07\\linewidth}@{}",
            "scriptsize",
            "1pt",
        ),
        ("stress_test_overview.tex", stress_overview, 4, "@{}lllllll@{}", "scriptsize", "3pt"),
        ("stress_test_warnings.tex", stress_warning, 4, "@{}llllll@{}", "scriptsize", "3pt"),
        ("time_scale_scaling.tex", scaling_display, 4, "@{}llll@{}", "footnotesize", "3pt"),
        ("autocorrelation_summary.tex", autocorr_display, 4, "@{}llllll@{}", "footnotesize", "2pt"),
        (
            "capital_requirements.tex",
            capital_display,
            2,
            "@{}>{\\RaggedLeft\\arraybackslash}p{0.16\\linewidth}>{\\RaggedRight\\arraybackslash}p{0.22\\linewidth}*{3}{>{\\RaggedLeft\\arraybackslash}p{0.13\\linewidth}}@{}",
            "footnotesize",
            "3pt",
        ),
        (
            "kupiec_backtest_summary.tex",
            backtest_display,
            4,
            "@{}>{\\RaggedRight\\arraybackslash}p{0.18\\linewidth}*{7}{>{\\RaggedLeft\\arraybackslash}p{0.10\\linewidth}}@{}",
            "scriptsize",
            "2pt",
        ),
    ]

    tex_paths: dict[str, str] = {}
    for filename, frame, precision, column_format, font_size, tabcolsep in table_specs:
        output_path = paths.tables_dir / filename
        write_latex_table(
            output_path,
            frame,
            precision=precision,
            column_format=column_format,
            font_size=font_size,
            tabcolsep=tabcolsep,
            longtable=filename != "model_tuning_summary.tex",
            center=filename == "model_tuning_summary.tex",
        )
        tex_paths[output_path.stem] = str(output_path)
    return tex_paths


def save_figures(
    feature_frame: pd.DataFrame,
    descriptive_stats: dict[str, float],
    mc_sample_paths: np.ndarray,
    backtest_frame: pd.DataFrame,
    lstm_predictions: pd.DataFrame,
    generated_returns_1h: np.ndarray,
    stress_warning_frames: dict[str, pd.DataFrame],
    paths: ProjectPaths,
) -> dict[str, str]:
    frame = feature_frame.set_index("timestamp")
    returns = frame["log_return"].dropna()
    figure_paths: dict[str, str] = {}

    plt.style.use("seaborn-v0_8-whitegrid")

    price_vol_path = paths.figures_dir / "price_and_volatility.png"
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(frame.index, frame["close"], color="#0f172a", linewidth=1.2)
    axes[0].set_title("BTC/USDT hourly close price")
    axes[0].set_ylabel("Price")
    axes[1].plot(frame.index, frame["rolling_vol_24h"], color="#b45309", linewidth=1.1)
    axes[1].set_title("Rolling 24-hour annualized volatility")
    axes[1].set_ylabel("Volatility")
    axes[1].set_xlabel("Timestamp")
    fig.tight_layout()
    fig.savefig(price_vol_path, dpi=180)
    plt.close(fig)
    figure_paths["price_and_volatility"] = price_vol_path.name

    distribution_path = paths.figures_dir / "return_distribution.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    _, bins, _ = ax.hist(returns, bins=140, density=True, alpha=0.75, color="#2563eb")
    normal_x = np.linspace(bins.min(), bins.max(), 400)
    normal_pdf = stats.norm.pdf(normal_x, loc=returns.mean(), scale=returns.std(ddof=1))
    t_params = estimate_student_t_parameters(returns.to_numpy())
    t_pdf = stats.t.pdf(normal_x, *t_params)
    ax.plot(normal_x, normal_pdf, color="#dc2626", linewidth=2, label="Gaussian fit")
    ax.plot(normal_x, t_pdf, color="#059669", linewidth=2, label="Student-t fit")
    ax.set_title("Hourly BTC return distribution versus Gaussian and Student-t fits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=180)
    plt.close(fig)
    figure_paths["return_distribution"] = distribution_path.name

    diagnostics_path = paths.figures_dir / "distribution_diagnostics.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    clustering_data = pd.DataFrame(
        {
            "lagged_abs_return": returns.abs().shift(1),
            "current_abs_return": returns.abs(),
        }
    ).dropna()
    if len(clustering_data) > 6000:
        clustering_data = clustering_data.sample(6000, random_state=RANDOM_SEED)
    axes[0].scatter(
        clustering_data["lagged_abs_return"],
        clustering_data["current_abs_return"],
        s=8,
        alpha=0.25,
        color="#b45309",
        edgecolors="none",
    )
    axes[0].set_title("Volatility clustering: |r(t-1)| vs |r(t)|")
    axes[0].set_xlabel("Lagged absolute return")
    axes[0].set_ylabel("Current absolute return")
    stats.probplot(returns.to_numpy(), dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q plot of hourly BTC returns")
    fig.tight_layout()
    fig.savefig(diagnostics_path, dpi=180)
    plt.close(fig)
    figure_paths["distribution_diagnostics"] = diagnostics_path.name

    mc_paths_path = paths.figures_dir / "monte_carlo_paths.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    start_price = frame["close"].iloc[-1]
    for path in mc_sample_paths[:100]:
        price_path = start_price * np.exp(np.concatenate([[0.0], np.cumsum(path)]))
        ax.plot(range(len(price_path)), price_path, alpha=0.15, color="#7c3aed")
    ax.set_title("10,000-path jump diffusion simulation (100 sample paths shown)")
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("Simulated price")
    fig.tight_layout()
    fig.savefig(mc_paths_path, dpi=180)
    plt.close(fig)
    figure_paths["monte_carlo_paths"] = mc_paths_path.name

    backtest_path = paths.figures_dir / "var_backtest.png"
    recent_backtest = backtest_frame.tail(24 * 60)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(recent_backtest.index, -recent_backtest["actual_return"], color="#111827", linewidth=1.0, label="Actual loss")
    ax.plot(recent_backtest.index, -recent_backtest["var_t"], color="#dc2626", linewidth=1.0, label="t-VaR")
    ax.plot(recent_backtest.index, -recent_backtest["var_mc"], color="#2563eb", linewidth=1.0, label="MC VaR")
    ax.plot(recent_backtest.index, -recent_backtest["var_lstm"], color="#059669", linewidth=1.0, label="LSTM VaR")
    ax.set_title("Recent 60-day loss series versus one-hour VaR thresholds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(backtest_path, dpi=180)
    plt.close(fig)
    figure_paths["var_backtest"] = backtest_path.name

    lstm_path = paths.figures_dir / "lstm_volatility_forecast.png"
    recent_lstm = lstm_predictions.tail(24 * 30)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(recent_lstm.index, recent_lstm["sigma"], color="#f59e0b", linewidth=1.2, label="Predicted sigma")
    ax.plot(recent_lstm.index, recent_lstm["actual_return"].abs(), color="#1f2937", linewidth=0.8, alpha=0.8, label="Realized |return|")
    ax.set_title("LSTM next-hour volatility forecast versus realized magnitude")
    ax.legend()
    fig.tight_layout()
    fig.savefig(lstm_path, dpi=180)
    plt.close(fig)
    figure_paths["lstm_volatility_forecast"] = lstm_path.name

    vae_path = paths.figures_dir / "vae_generated_returns.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(returns.tail(24 * 90), bins=100, density=True, alpha=0.55, color="#2563eb", label="Recent observed returns")
    ax.hist(generated_returns_1h, bins=100, density=True, alpha=0.45, color="#f97316", label="VAE synthetic returns")
    ax.set_title("Observed versus VAE-generated hourly return distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(vae_path, dpi=180)
    plt.close(fig)
    figure_paths["vae_generated_returns"] = vae_path.name

    stress_path = paths.figures_dir / "stress_test_warnings.png"
    fig, axes = plt.subplots(len(stress_warning_frames), 1, figsize=(14, 9), sharex=False)
    axes_array = np.atleast_1d(axes)
    for axis, (event_name, warning_frame) in zip(axes_array, stress_warning_frames.items()):
        axis.plot(warning_frame["forecast_timestamp"], warning_frame["t_var_loss"], color="#dc2626", linewidth=1.3, label="Student-t VaR loss")
        axis.plot(warning_frame["forecast_timestamp"], warning_frame["mc_var_loss"], color="#2563eb", linewidth=1.3, label="Jump diffusion MC VaR loss")
        axis.axhline(warning_frame["warning_threshold"].iloc[0], color="#6b7280", linestyle="--", linewidth=1.0, label="50% realized loss threshold")
        axis.set_title(f"Pre-event warning profile: {event_name}")
        axis.set_ylabel("Projected one-hour loss")
        axis.legend(loc="upper left")
    axes_array[-1].set_xlabel("Forecast timestamp")
    fig.tight_layout()
    fig.savefig(stress_path, dpi=180)
    plt.close(fig)
    figure_paths["stress_test_warnings"] = stress_path.name

    return figure_paths


def render_report(
    data_quality: dict[str, Any],
    descriptive_stats: dict[str, float],
    sample_split_summary: pd.DataFrame,
    current_var_table: pd.DataFrame,
    model_tuning_summary: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    confidence_sensitivity: pd.DataFrame,
    student_t_sensitivity: pd.DataFrame,
    lstm_window_sensitivity: pd.DataFrame,
    stress_test_table: pd.DataFrame,
    scaling_table: pd.DataFrame,
    autocorr_table: pd.DataFrame,
    capital_table: pd.DataFrame,
    figure_paths: dict[str, str],
    feature_frame: pd.DataFrame,
) -> str:
    sections = build_report_sections(
        data_quality,
        descriptive_stats,
        sample_split_summary,
        current_var_table,
        model_tuning_summary,
        backtest_summary,
        confidence_sensitivity,
        stress_test_table,
        scaling_table,
        capital_table,
        feature_frame,
    )
    data_quality_display = format_report_table(
        pd.DataFrame([data_quality]),
        {
            "original_rows": "original_rows",
            "cleaned_rows": "cleaned_rows",
            "missing_hours_filled": "missing_hours_filled",
            "duplicate_rows_removed": "duplicate_rows_removed",
            "flash_crash_repairs": "flash_crash_repairs",
            "start": "start",
            "end": "end",
        },
    )
    split_display = format_report_table(
        sample_split_summary,
        {
            "block": "block",
            "start": "start",
            "end": "end",
            "observations": "observations",
            "design_note": "design_note",
        },
    )
    current_var_display = format_report_table(
        current_var_table,
        {
            "model": "model",
            "horizon_hours": "horizon_hours",
            "VaR_return": "VaR_return",
            "CVaR_return": "CVaR_return",
            "VaR_loss": "VaR_loss",
            "CVaR_loss": "CVaR_loss",
            "mean_return": "mean_return",
            "volatility": "volatility",
        },
    )
    model_tuning_display = format_report_table(
        model_tuning_summary,
        {
            "model": "model",
            "design": "design",
            "best_parameters": "best_parameters",
            "selection_metric": "selection_metric",
            "validation_score": "validation_score",
        },
    )
    confidence_display = format_report_table(confidence_sensitivity)
    stress_display = format_report_table(stress_test_table)
    capital_display = format_report_table(capital_table)
    backtest_display = format_report_table(
        backtest_summary[
            [
                "model",
                "observations",
                "violations",
                "expected_violation_rate",
                "observed_violation_rate",
                "kupiec_pvalue",
                "christoffersen_ind_pvalue",
                "christoffersen_cc_pvalue",
            ]
        ]
    )
    return fr"""# Bitcoin Market Risk Analysis: Comparative Value-at-Risk Evidence from Hourly Data

## Abstract

{sections['abstract']}

## Introduction

{sections['introduction']}

## Data

{sections['data']}

{to_markdown_table(data_quality_display)}

![Price and volatility](figures/{figure_paths['price_and_volatility']})

## Methodology

{sections['methodology']}

### Method Overview

{render_method_overview_markdown()}

### Tuned Model Settings

{to_markdown_table(model_tuning_display)}

### Sample Split Summary

{to_markdown_table(split_display)}

The downside tail metrics used in the report follow the standard left-tail definitions:

$$
CVaR_{{\alpha}} = \frac{{1}}{{1-\alpha}} \int_{{\alpha}}^{{1}} VaR_u \, du
$$

For unconditional coverage backtesting, the Kupiec proportion-of-failures statistic is:

$$
LR_{{POF}} = -2 \ln \left( \frac{{(1-\alpha)^{{T-x}} \alpha^x}}{{(1-\hat{{p}})^{{T-x}} \hat{{p}}^x}} \right), \qquad \hat{{p}} = \frac{{x}}{{T}}
$$

where $T$ is the number of forecasts, $x$ is the number of VaR violations, and $\hat{{p}}$ is the observed violation rate. The report complements this with Christoffersen-style independence and conditional-coverage diagnostics {academic_citation('kupiec1995', 'christoffersen1998')}.

## Statistical Diagnostics

{sections['diagnostics']}

{to_markdown_table(pd.DataFrame([descriptive_stats]))}

![Return distribution](figures/{figure_paths['return_distribution']})

![Distribution diagnostics](figures/{figure_paths['distribution_diagnostics']})

## Model Results

{sections['results']}

{to_markdown_table(current_var_display)}

![Monte Carlo paths](figures/{figure_paths['monte_carlo_paths']})

## Sensitivity Analysis

{sections['sensitivity']}

### Confidence-Level Comparison

{to_markdown_table(confidence_display)}

### Student-t Degrees-of-Freedom Impact

{to_markdown_table(student_t_sensitivity)}

### LSTM Look-back Window Impact

{to_markdown_table(lstm_window_sensitivity)}

## Stress Testing

{sections['stress']}

{to_markdown_table(stress_display)}

![Stress warning profiles](figures/{figure_paths['stress_test_warnings']})

## Time-Scale Scaling

{sections['scaling']}

{to_markdown_table(scaling_table)}

{to_markdown_table(autocorr_table)}

## Practical Implications

{sections['practical']}

{to_markdown_table(capital_display, precision=2)}

## Backtesting

{sections['backtesting']}

{to_markdown_table(backtest_display)}

![VaR backtest](figures/{figure_paths['var_backtest']})

![LSTM volatility forecast](figures/{figure_paths['lstm_volatility_forecast']})

![VAE generated returns](figures/{figure_paths['vae_generated_returns']})

## Limitations

{sections['limitations']}

## Discussion

{sections['discussion']}

## Conclusion

{sections['conclusion']}

## References

{render_references_markdown()}

## Appendix

{sections['appendix']}
"""


def build_report_sections(
    data_quality: dict[str, Any],
    descriptive_stats: dict[str, float],
    sample_split_summary: pd.DataFrame,
    current_var_table: pd.DataFrame,
    model_tuning_summary: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    confidence_sensitivity: pd.DataFrame,
    stress_test_table: pd.DataFrame,
    scaling_table: pd.DataFrame,
    capital_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> dict[str, str]:
    report_start = format_report_timestamp(data_quality["start"])
    report_end = format_report_timestamp(data_quality["end"])
    latest_close = feature_frame["close"].iloc[-1]
    first_close = feature_frame["close"].iloc[0]
    cumulative_return = latest_close / first_close - 1
    best_backtest_row = backtest_summary.assign(
        rank_cc_pvalue=backtest_summary["christoffersen_cc_pvalue"].fillna(-1.0)
    ).sort_values(["rank_cc_pvalue", "kupiec_pvalue"], ascending=[False, False]).iloc[0]
    day_scaling_row = scaling_table.loc[scaling_table["horizon"] == "1 day"].iloc[0]
    ten_day_scaling_row = scaling_table.loc[scaling_table["horizon"] == "10 days"].iloc[0]
    ftx_row = stress_test_table.loc[stress_test_table["event"] == "FTX Collapse"].iloc[0]
    etf_row = stress_test_table.loc[stress_test_table["event"] == "Spot ETF Approval Whipsaw"].iloc[0]
    conservative_row = confidence_sensitivity.loc[confidence_sensitivity["confidence_level_pct"] == 99.0].sort_values("VaR_loss", ascending=False).iloc[0]
    capital_24h = capital_table.loc[capital_table["horizon_hours"] == 24].sort_values("VaR_capital_usd", ascending=False).iloc[0]

    split_rows = sample_split_summary.set_index("block")
    tuning_row = split_rows.loc["Tuning validation slice"]
    backtest_row = split_rows.loc["Backtest holdout"]
    lstm_val_row = split_rows.loc["LSTM validation sequences"]
    vae_val_row = split_rows.loc["VAE validation windows"]
    gaussian_row = current_var_table[
        (current_var_table["model"] == "Gaussian benchmark") & (current_var_table["horizon_hours"] == 1)
    ].iloc[0]
    student_t_row = current_var_table[
        (current_var_table["model"] == "Parametric Student-t") & (current_var_table["horizon_hours"] == 1)
    ].iloc[0]
    backtest_by_model = backtest_summary.set_index("model")
    gaussian_backtest = backtest_by_model.loc["Gaussian benchmark"]

    accepted_models = [
        report_label(model, REPORT_MODEL_LABELS)
        for model in backtest_summary.loc[backtest_summary["christoffersen_cc_pvalue"] >= 0.05, "model"].tolist()
    ]
    rejected_models = [
        report_label(model, REPORT_MODEL_LABELS)
        for model in backtest_summary.loc[backtest_summary["christoffersen_cc_pvalue"] < 0.05, "model"].tolist()
    ]
    accepted_text = human_join(accepted_models)
    rejected_text = human_join(rejected_models)
    best_model_label = report_label(str(best_backtest_row["model"]), REPORT_MODEL_LABELS)
    if accepted_models and rejected_models:
        backtesting_acceptance_sentence = (
            f"At the 5% significance level, {rejected_text} {'is' if len(rejected_models) == 1 else 'are'} rejected by the conditional-coverage test, whereas {accepted_text} {'is' if len(accepted_models) == 1 else 'are'} not rejected."
        )
    elif accepted_models:
        backtesting_acceptance_sentence = (
            f"At the 5% significance level, {accepted_text} {'is' if len(accepted_models) == 1 else 'are'} not rejected by the conditional-coverage test."
        )
    else:
        backtesting_acceptance_sentence = (
            f"At the 5% significance level, {rejected_text} {'is' if len(rejected_models) == 1 else 'are'} rejected by the conditional-coverage test, so no model is retained by that criterion on this holdout window."
        )

    return {
        "abstract": (
            f"This report evaluates whether a Gaussian Value at Risk benchmark is credible for Bitcoin when the asset is observed at hourly frequency from {report_start} to {report_end}. "
            "The empirical design compares a rolling Gaussian benchmark with four richer model families: a parametric Student-t specification, a jump-diffusion Monte Carlo engine, a feature-conditioned LSTM, and a variational autoencoder used for latent scenario generation. "
            f"The evidence rejects the Gaussian baseline on both tail shape and backtest diagnostics, supports heavy-tailed and state-dependent modeling, and shows that {best_model_label} provides the strongest one-hour conditional coverage result in the final-year backtest, while jump-diffusion remains the more conservative scenario engine under stress."
        ),
        "introduction": (
            "The central research question is whether a simple Gaussian VaR framework can describe Bitcoin downside risk at an hourly horizon, or whether the data require heavier tails, jump risk, and conditional machine-learning overlays. "
            f"That question matters academically because Bitcoin combines large volatility clusters, event-driven discontinuities, and rapid regime changes that can break the assumptions behind textbook variance-scaling rules and thin-tailed parametric benchmarks {academic_citation('jorion2007', 'mcneil2015')}. "
            "The report therefore evaluates model credibility along three dimensions: distributional fit, out-of-sample tail calibration, and stress-period warning quality."
        ),
        "data": (
            f"The dataset contains {data_quality['cleaned_rows']:,} cleaned hourly observations built from Binance BTC/USDT spot candles between {report_start} and {report_end}. "
            f"Data cleaning filled {data_quality['missing_hours_filled']} missing hours, removed {data_quality['duplicate_rows_removed']} duplicate rows, and repaired {data_quality['flash_crash_repairs']} candidate flash-crash artefacts. "
            "BTC/USDT is used as a liquid USD proxy, which is operationally convenient for uninterrupted hourly sampling but should still be interpreted as a proxy benchmark rather than a pure fiat BTC/USD series."
        ),
        "methodology": (
            f"The benchmark layer contains two rolling parametric baselines: a Gaussian VaR benchmark and a Student-t VaR benchmark, both estimated on the same recent return window so that the distributional assumption is the main moving part {academic_citation('jorion2007', 'mcneil2015')}. The structural scenario layer adds jump-diffusion Monte Carlo with {MONTE_CARLO_PATHS:,} simulation paths and a tuned jump threshold in a Merton-style discontinuous-return setting {academic_citation('merton1976')}. The neural layer adds a one-step LSTM with a Gaussian output head and a VAE that learns latent return-window structure for synthetic scenario generation {academic_citation('hochreiter1997', 'kingma2014')}. "
            f"Feature engineering supplies log returns, rolling 24-hour and 168-hour volatility, RSI(14), MACD with signal line, and a rolling volume z-score to the conditional models. Chronological splitting is enforced throughout. The final-year backtest holdout spans {format_report_timestamp(backtest_row['start'])} to {format_report_timestamp(backtest_row['end'])} with {int(backtest_row['observations']):,} hourly observations. Student-t and jump-diffusion hyperparameters are tuned on a {TUNING_VALIDATION_DAYS}-day pre-backtest slice sampled every {TUNING_EVAL_STRIDE_HOURS} hours, which yields {int(tuning_row['observations']):,} validation checkpoints. For the neural models, the final 20% of the pre-backtest sequences or rolling windows are reserved for validation, giving {int(lstm_val_row['observations']):,} LSTM validation sequences and {int(vae_val_row['observations']):,} VAE validation windows. "
            f"The LSTM is trained with AdamW, learning rate {LSTM_LEARNING_RATE:.4g}, batch size {LSTM_BATCH_SIZE}, and {LSTM_FINAL_EPOCHS} epochs, with the final checkpoint chosen by minimum validation Gaussian negative log-likelihood. The VAE is trained with AdamW, learning rate {VAE_LEARNING_RATE:.4g}, batch size {VAE_BATCH_SIZE}, {VAE_FINAL_EPOCHS} epochs, and a reconstruction-plus-KL loss on rolling {VAE_WINDOW}-hour windows. Validation scores in the tuning table therefore represent mean pinball loss for the parametric models, validation Gaussian NLL for the LSTM, and validation ELBO-style loss for the VAE."
        ),
        "diagnostics": (
            f"The descriptive evidence rejects Gaussianity decisively. The Jarque-Bera statistic reaches {descriptive_stats['jarque_bera_stat']:.2f} with a near-zero p-value, while skewness is {descriptive_stats['skewness']:.3f} and kurtosis is {descriptive_stats['kurtosis']:.3f}. "
            f"Hourly returns range from {descriptive_stats['min']:.2%} to {descriptive_stats['max']:.2%}, and the ADF statistic of {descriptive_stats['adf_stat']:.3f} rejects a unit root in the return series. Taken together, these diagnostics support heavy-tailed and conditional models rather than a thin-tailed homoskedastic baseline."
        ),
        "results": (
            f"Over the full sample, Bitcoin delivered a cumulative price change of {cumulative_return:.2%}. In the current forecast snapshot, the one-hour VaR rankings differ materially by methodology, with the most conservative 99% estimate coming from {report_label(str(conservative_row['model']), REPORT_MODEL_LABELS)} at {conservative_row['VaR_loss']:.2%}. The Gaussian benchmark currently reports a one-hour VaR loss of {gaussian_row['VaR_loss']:.2%}, versus {student_t_row['VaR_loss']:.2%} for the Student-t benchmark estimated on the same rolling history. "
            "The Student-t model remains the most interpretable heavy-tail baseline, the jump-diffusion engine provides the most stress-aware structural scenarios, and the LSTM contributes the most explicitly conditional one-step signal. Because the recurrent model is estimated as a one-step conditional forecaster, the 24-hour comparison table includes only models that directly produce multi-period distributions through aggregation or simulation."
        ),
        "sensitivity": (
            f"Sensitivity analysis confirms the expected widening of downside risk as confidence moves from 95% to 99%. At 99%, the largest one-hour VaR loss remains {conservative_row['VaR_loss']:.2%}. "
            "The Gaussian row in the confidence table provides the thin-tailed benchmark, the Student-t parameter sweep shows that lower degrees of freedom amplify tail losses in a nonlinear way, and the LSTM look-back sweep shows a trade-off between responsiveness and stability."
        ),
        "stress": (
            f"The stress section is designed as a two-window case-study exercise rather than an exhaustive event screen. Two pre-specified event windows are defined in code, and the event timestamp inside each window is selected algorithmically by either the worst one-hour loss or the largest absolute one-hour move; warnings are then evaluated over the preceding {int(ftx_row['warning_horizon_hours'])}-hour horizon. Within that design, the FTX collapse window peaks at {format_report_timestamp(ftx_row['event_timestamp'])} with a realized one-hour loss of {ftx_row['actual_loss']:.2%}, and the ETF whipsaw peaks at {format_report_timestamp(etf_row['event_timestamp'])} with a realized absolute move of {etf_row['actual_loss']:.2%}. In both cases, the jump-diffusion engine issues the strongest warning profile and lands closest to the realized loss. At the same time, every one-hour VaR model understates the realized event loss by a wide margin, so the stress section should be interpreted as a warning-comparison exercise rather than a claim of exact crisis prediction."
        ),
        "scaling": (
            f"The square-root-of-time rule is only partially reliable for Bitcoin. The scaled one-day 99% VaR is {day_scaling_row['sqrt_time_scaled_VaR']:.2%} versus an empirical one-day VaR of {day_scaling_row['empirical_VaR']:.2%}, while the scaled ten-day VaR is {ten_day_scaling_row['sqrt_time_scaled_VaR']:.2%} against an empirical ten-day VaR of {ten_day_scaling_row['empirical_VaR']:.2%}. "
            "Weak linear autocorrelation in raw returns coexists with persistent autocorrelation in absolute returns, which implies that volatility clustering rather than mean predictability is the main reason naive variance scaling can misstate long-horizon risk."
        ),
        "practical": (
            f"For a hypothetical $1 million BTC spot position, the largest 24-hour VaR capital estimate in the current snapshot is ${capital_24h['VaR_capital_usd']:,.2f}. "
            "These are market-risk numbers rather than liquidity-adjusted exit costs. For committee reporting, the jump-diffusion estimates are the more conservative baseline, while the LSTM is more appropriate as an intraday surveillance overlay."
        ),
        "backtesting": (
            f"The last-year backtest evaluates {int(backtest_row['observations']):,} one-hour forecasts against a 1% tail target using both Kupiec's unconditional coverage test and Christoffersen's independence and conditional-coverage diagnostics {academic_citation('kupiec1995', 'christoffersen1998')}. {best_model_label} delivers the strongest joint result with a conditional-coverage p-value of {best_backtest_row['christoffersen_cc_pvalue']:.3f}, while the Gaussian benchmark posts a Kupiec p-value of {gaussian_backtest['kupiec_pvalue']:.3f} and a conditional-coverage p-value of {gaussian_backtest['christoffersen_cc_pvalue']:.3f}. "
            f"{backtesting_acceptance_sentence} That pattern suggests that conditional adaptation improves tail calibration on this holdout window, but it is still evidence about this sample path rather than proof that one specification dominates in every regime."
        ),
        "limitations": (
            "The project has several explicit limitations. The benchmark is a proxy BTC/USDT series rather than a fiat BTC/USD market, the report relies on hourly candles rather than intrahour depth, and the backtesting section still evaluates only one realized holdout year even after adding conditional-coverage diagnostics. The neural models improve adaptability but remain harder to interpret than the parametric baseline, and the validation design is chronological rather than full rolling-origin cross-validation across multiple non-overlapping regimes. Finally, the reported VaR and CVaR numbers are market-risk statistics rather than liquidity-adjusted liquidation losses."
        ),
        "discussion": (
            f"Two empirical themes dominate the evidence. First, Bitcoin exhibits persistent volatility clustering, so state dependence matters even when raw returns themselves are weakly autocorrelated {academic_citation('engle1982')}. Second, the downside tail is too heavy for a Gaussian benchmark to be credible: the Gaussian row understates tail risk relative to the matched Student-t benchmark and is weaker than Student-t VaR on the final-year backtest. Those facts explain why the project benefits from layering models instead of selecting a single winner for every use case."
        ),
        "conclusion": (
            "The main conclusion is that Gaussian hourly VaR is not an adequate benchmark for Bitcoin. Heavy-tailed parametric modeling, jump-aware simulation, and conditional neural forecasting all add economically meaningful information relative to a thin-tailed variance-only baseline. In this project, the jump-diffusion engine is the strongest conservative scenario benchmark, while the LSTM is the strongest one-hour conditional overlay on the final holdout year under the current conditional-coverage diagnostics."
        ),
        "appendix": (
            "The appendix figures support the report's main claims visually. The volatility-clustering scatter plot shows persistence in absolute returns, the normal Q-Q panel shows substantial tail deviation from Gaussian behavior, the backtest figure shows the relative responsiveness of each VaR series, and the generated-return panels illustrate how the neural models differ in purpose from the parametric baselines."
        ),
    }


def render_report_tex(
    data_quality: dict[str, Any],
    descriptive_stats: dict[str, float],
    sample_split_summary: pd.DataFrame,
    current_var_table: pd.DataFrame,
    model_tuning_summary: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    confidence_sensitivity: pd.DataFrame,
    stress_test_table: pd.DataFrame,
    scaling_table: pd.DataFrame,
    autocorr_table: pd.DataFrame,
    capital_table: pd.DataFrame,
    figure_paths: dict[str, str],
    feature_frame: pd.DataFrame,
) -> str:
    sections = build_report_sections(
        data_quality,
        descriptive_stats,
        sample_split_summary,
        current_var_table,
        model_tuning_summary,
        backtest_summary,
        confidence_sensitivity,
        stress_test_table,
        scaling_table,
        capital_table,
        feature_frame,
    )

    def paragraph(text: str) -> str:
        return "\n\n".join(latex_escape(part) for part in text.split("\n\n"))

    return f"""% Generated by market_risk_analysis.pipeline.render_report_tex.
\\setlength{{\\LTleft}}{{\\fill}}
\\setlength{{\\LTright}}{{\\fill}}
\\renewcommand{{\\arraystretch}}{{1.05}}

\\subsection{{Abstract}}

{paragraph(sections['abstract'])}

\\subsection{{Introduction}}

{paragraph(sections['introduction'])}

\\subsection{{Data}}

{paragraph(sections['data'])}

\\input{{tables/data_quality.tex}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics{{figures/{figure_paths['price_and_volatility']}}}
\\caption{{Price and volatility}}
\\end{{figure}}

\\subsection{{Methodology}}

{paragraph(sections['methodology'])}

\\subsubsection*{{Method Overview}}

{render_method_overview_latex()}

\\subsubsection*{{Tuned Model Settings}}

\\input{{tables/model_tuning_summary.tex}}

\\subsubsection*{{Sample Split Summary}}

\\input{{tables/sample_split_summary.tex}}

The downside tail metrics used in the report follow the standard left-tail definitions:

\\[
CVaR_{{\\alpha}} = \\frac{{1}}{{1-\\alpha}} \\int_{{\\alpha}}^{{1}} VaR_u \\, du
\\]

For unconditional coverage backtesting, the Kupiec proportion-of-failures statistic is:

\\[
LR_{{POF}} = -2 \\ln \\left( \\frac{{(1-\\alpha)^{{T-x}} \\alpha^x}}{{(1-\\hat{{p}})^{{T-x}} \\hat{{p}}^x}} \\right), \\qquad \\hat{{p}} = \\frac{{x}}{{T}}
\\]

where $T$ is the number of forecasts, $x$ is the number of VaR violations, and $\\hat{{p}}$ is the observed violation rate.
The report complements this with Christoffersen-style independence and conditional-coverage diagnostics {academic_citation('kupiec1995', 'christoffersen1998')}.

\\subsection{{Statistical Diagnostics}}

{paragraph(sections['diagnostics'])}

\\input{{tables/descriptive_statistics_main.tex}}

\\input{{tables/descriptive_statistics_tail.tex}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics{{figures/{figure_paths['return_distribution']}}}
\\caption{{Return distribution}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics{{figures/{figure_paths['distribution_diagnostics']}}}
\\caption{{Distribution diagnostics}}
\\end{{figure}}

\\subsection{{Model Results}}

{paragraph(sections['results'])}

\\input{{tables/current_var_estimates.tex}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics{{figures/{figure_paths['monte_carlo_paths']}}}
\\caption{{Monte Carlo paths}}
\\end{{figure}}

\\subsection{{Sensitivity Analysis}}

{paragraph(sections['sensitivity'])}

\\subsubsection*{{Confidence-Level Comparison}}

\\input{{tables/confidence_sensitivity.tex}}

\\subsubsection*{{Student-t Degrees-of-Freedom Impact}}

\\input{{tables/student_t_df_sensitivity.tex}}

\\subsubsection*{{LSTM Look-back Window Impact}}

\\input{{tables/lstm_window_sensitivity.tex}}

\\subsection{{Stress Testing}}

{paragraph(sections['stress'])}

\\input{{tables/stress_test_overview.tex}}

\\input{{tables/stress_test_warnings.tex}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics{{figures/{figure_paths['stress_test_warnings']}}}
\\caption{{Stress warning profiles}}
\\end{{figure}}

\\subsection{{Time-Scale Scaling}}

{paragraph(sections['scaling'])}

\\input{{tables/time_scale_scaling.tex}}

\\input{{tables/autocorrelation_summary.tex}}

\\subsection{{Practical Implications}}

{paragraph(sections['practical'])}

\\input{{tables/capital_requirements.tex}}

\\subsection{{Backtesting}}

{paragraph(sections['backtesting'])}

\\input{{tables/kupiec_backtest_summary.tex}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics{{figures/{figure_paths['var_backtest']}}}
\\caption{{VaR backtest}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics{{figures/{figure_paths['lstm_volatility_forecast']}}}
\\caption{{LSTM volatility forecast}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics{{figures/{figure_paths['vae_generated_returns']}}}
\\caption{{VAE generated returns}}
\\end{{figure}}

\\subsection{{Limitations}}

{paragraph(sections['limitations'])}

\\subsection{{Discussion}}

{paragraph(sections['discussion'])}

\\subsection{{Conclusion}}

{paragraph(sections['conclusion'])}

\\subsection{{References}}

{render_references_latex()}

\\subsection{{Appendix}}

{paragraph(sections['appendix'])}
"""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def serializer(value: Any) -> Any:
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return format_report_timestamp(value)
        raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, default=serializer), encoding="utf-8")


def run_analysis(refresh_data: bool = False) -> dict[str, str]:
    set_random_seed()
    paths = ProjectPaths()
    paths.ensure()

    raw_data: pd.DataFrame
    if refresh_data or not paths.raw_data_path.exists():
        raw_data = fetch_binance_hourly_data()
        raw_data.to_csv(paths.raw_data_path, index=False)
    else:
        raw_data = pd.read_csv(paths.raw_data_path)
        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"], utc=True)

    cleaned_frame, data_quality = clean_market_data(raw_data)
    feature_frame = build_feature_frame(cleaned_frame)
    feature_frame.to_csv(paths.cleaned_data_path, index=False)

    indexed_feature_frame = feature_frame.set_index("timestamp")
    returns = indexed_feature_frame["log_return"].dropna()
    descriptive_stats = compute_descriptive_statistics(returns)
    backtest_start = feature_frame["timestamp"].max() - pd.Timedelta(days=BACKTEST_DAYS)
    rng = np.random.default_rng(RANDOM_SEED)

    feature_columns = [
        "log_return",
        "rolling_vol_24h",
        "rolling_vol_168h",
        "rsi_14",
        "macd",
        "macd_signal",
        "volume_zscore",
    ]

    student_t_tuning = tune_student_t_window(returns, DEFAULT_CONFIDENCE, backtest_start)
    jump_tuning = tune_jump_diffusion_parameters(returns, DEFAULT_CONFIDENCE, backtest_start)

    lstm_window_sensitivity, lstm_candidate_results = evaluate_lstm_candidates(
        feature_frame,
        feature_columns,
        backtest_start,
        DEFAULT_CONFIDENCE,
    )

    vae_tuning = tune_vae_latent_dim(returns, DEFAULT_CONFIDENCE, backtest_start)
    model_tuning_summary, tuned_config = build_model_tuning_summary(
        student_t_tuning,
        jump_tuning,
        lstm_window_sensitivity,
        vae_tuning,
    )

    student_t_window_hours = tuned_config["student_t_window_hours"]
    jump_diffusion_window_hours = tuned_config["jump_diffusion_window_hours"]
    jump_threshold_sigma = tuned_config["jump_threshold_sigma"]
    lstm_sequence_length = tuned_config["lstm_sequence_length"]
    vae_window = tuned_config["vae_window"]
    vae_latent_dim = tuned_config["vae_latent_dim"]

    student_t_sample = returns.tail(student_t_window_hours)
    jump_diffusion_sample = returns.tail(jump_diffusion_window_hours)
    gaussian_sample = student_t_sample

    gaussian_results = fit_gaussian_var(gaussian_sample, DEFAULT_CONFIDENCE, HORIZON_HOURS)
    t_results = fit_student_t_var(student_t_sample, DEFAULT_CONFIDENCE, HORIZON_HOURS, rng)
    mc_results, mc_sample_paths = simulate_jump_diffusion_var(
        jump_diffusion_sample,
        DEFAULT_CONFIDENCE,
        HORIZON_HOURS,
        MONTE_CARLO_PATHS,
        rng,
        jump_threshold_sigma=jump_threshold_sigma,
    )

    lstm_tail_alpha = tuned_config["lstm_tail_alpha"]
    lstm_result = recalibrate_lstm_result(lstm_candidate_results[lstm_sequence_length], lstm_tail_alpha)
    lstm_datasets = build_lstm_datasets(feature_frame, feature_columns, lstm_sequence_length, backtest_start)
    vae_result = fit_vae_model(
        returns,
        DEFAULT_CONFIDENCE,
        backtest_start,
        rng,
        window=vae_window,
        latent_dim=vae_latent_dim,
        learning_rate=VAE_LEARNING_RATE,
        batch_size=VAE_BATCH_SIZE,
        epochs=VAE_FINAL_EPOCHS,
    )

    backtest_summary, backtest_frame = run_backtests(
        feature_frame,
        DEFAULT_CONFIDENCE,
        lstm_result["predictions"],
        vae_result["backtest_var_return"],
        RANDOM_SEED,
        student_t_window_hours,
        student_t_window_hours,
        jump_diffusion_window_hours,
        jump_threshold_sigma,
    )

    current_var_rows = []
    for model_name, forecast_map in (
        ("Gaussian benchmark", gaussian_results),
        ("Parametric Student-t", t_results),
        ("Jump diffusion Monte Carlo", mc_results),
        ("LSTM", {1: lstm_result["current_forecast"]}),
        ("VAE", vae_result["current_forecasts"]),
    ):
        for horizon, summary in forecast_map.items():
            current_var_rows.append(
                {
                    "model": model_name,
                    "horizon_hours": horizon,
                    "VaR_return": summary.var_return,
                    "CVaR_return": summary.cvar_return,
                    "VaR_loss": summary.var_loss,
                    "CVaR_loss": summary.cvar_loss,
                    "mean_return": summary.mean_return,
                    "volatility": summary.volatility,
                }
            )
    current_var_table = pd.DataFrame(current_var_rows).sort_values(["horizon_hours", "model"]).reset_index(drop=True)

    confidence_sensitivity = compute_confidence_sensitivity(
        student_t_sample,
        jump_diffusion_sample,
        jump_threshold_sigma,
        lstm_result,
        vae_result,
    )
    student_t_sensitivity = compute_student_t_parameter_sensitivity(student_t_sample, DEFAULT_CONFIDENCE)
    stress_test_table, stress_warning_frames = compute_stress_test_analysis(
        feature_frame,
        DEFAULT_CONFIDENCE,
        student_t_window_hours,
        jump_diffusion_window_hours,
        jump_threshold_sigma,
    )
    scaling_table, autocorr_table = compute_time_scale_diagnostics(returns, DEFAULT_CONFIDENCE)
    capital_table = compute_capital_requirements(current_var_table)
    sample_split_summary = build_sample_split_summary(
        data_quality,
        returns,
        backtest_start,
        lstm_datasets,
        vae_result["split_summary"],
    )
    export_sample_split_summary = format_report_table(sample_split_summary)
    export_stress_test_table = format_report_table(stress_test_table)
    export_data_quality = {
        **data_quality,
        "start": format_report_timestamp(data_quality["start"]),
        "end": format_report_timestamp(data_quality["end"]),
    }

    current_var_table.to_csv(paths.tables_dir / "current_var_estimates.csv", index=False)
    pd.DataFrame([descriptive_stats]).to_csv(paths.tables_dir / "descriptive_statistics.csv", index=False)
    export_sample_split_summary.to_csv(paths.tables_dir / "sample_split_summary.csv", index=False)
    backtest_summary.to_csv(paths.tables_dir / "kupiec_backtest_summary.csv", index=False)
    confidence_sensitivity.to_csv(paths.tables_dir / "confidence_sensitivity.csv", index=False)
    student_t_sensitivity.to_csv(paths.tables_dir / "student_t_df_sensitivity.csv", index=False)
    lstm_window_sensitivity.to_csv(paths.tables_dir / "lstm_window_sensitivity.csv", index=False)
    model_tuning_summary.to_csv(paths.tables_dir / "model_tuning_summary.csv", index=False)
    export_stress_test_table.to_csv(paths.tables_dir / "stress_test_analysis.csv", index=False)
    scaling_table.to_csv(paths.tables_dir / "time_scale_scaling.csv", index=False)
    autocorr_table.to_csv(paths.tables_dir / "autocorrelation_summary.csv", index=False)
    capital_table.to_csv(paths.tables_dir / "capital_requirements.csv", index=False)
    obsolete_npu_table = paths.tables_dir / "npu_runtime_summary.csv"
    if obsolete_npu_table.exists():
        obsolete_npu_table.unlink()

    figure_paths = save_figures(
        feature_frame,
        descriptive_stats,
        mc_sample_paths,
        backtest_frame,
        lstm_result["predictions"],
        vae_result["generated_returns_1h"],
        stress_warning_frames,
        paths,
    )
    table_tex_paths = export_analysis_tables(
        paths,
        data_quality,
        descriptive_stats,
        sample_split_summary,
        current_var_table,
        model_tuning_summary,
        backtest_summary,
        confidence_sensitivity,
        student_t_sensitivity,
        lstm_window_sensitivity,
        stress_test_table,
        scaling_table,
        autocorr_table,
        capital_table,
    )

    results_payload = {
        "data_quality": export_data_quality,
        "descriptive_statistics": descriptive_stats,
        "sample_split_summary": export_sample_split_summary.to_dict(orient="records"),
        "current_var_estimates": current_var_table.to_dict(orient="records"),
        "model_tuning_summary": model_tuning_summary.to_dict(orient="records"),
        "confidence_sensitivity": confidence_sensitivity.to_dict(orient="records"),
        "student_t_parameter_sensitivity": student_t_sensitivity.to_dict(orient="records"),
        "lstm_window_sensitivity": lstm_window_sensitivity.to_dict(orient="records"),
        "tuning_config": tuned_config,
        "stress_test_analysis": export_stress_test_table.to_dict(orient="records"),
        "time_scale_scaling": scaling_table.to_dict(orient="records"),
        "autocorrelation_summary": autocorr_table.to_dict(orient="records"),
        "capital_requirements": capital_table.to_dict(orient="records"),
        "backtest_summary": backtest_summary.to_dict(orient="records"),
        "figure_paths": {name: str(paths.figures_dir / filename) for name, filename in figure_paths.items()},
        "table_tex_paths": table_tex_paths,
        "tables_dir": str(paths.tables_dir),
        "report_markdown_path": str(paths.report_markdown_path),
        "report_tex_path": str(paths.report_tex_path),
        "report_todo_path": str(paths.report_todo_path),
    }
    write_json(paths.results_path, results_payload)

    report_markdown = render_report(
        data_quality,
        descriptive_stats,
        sample_split_summary,
        current_var_table,
        model_tuning_summary,
        backtest_summary,
        confidence_sensitivity,
        student_t_sensitivity,
        lstm_window_sensitivity,
        stress_test_table,
        scaling_table,
        autocorr_table,
        capital_table,
        figure_paths,
        feature_frame,
    )
    paths.report_markdown_path.write_text(report_markdown, encoding="utf-8")
    report_tex = render_report_tex(
        data_quality,
        descriptive_stats,
        sample_split_summary,
        current_var_table,
        model_tuning_summary,
        backtest_summary,
        confidence_sensitivity,
        stress_test_table,
        scaling_table,
        autocorr_table,
        capital_table,
        figure_paths,
        feature_frame,
    )
    paths.report_tex_path.write_text(report_tex, encoding="utf-8")
    paths.report_todo_path.write_text(render_report_todo_markdown(), encoding="utf-8")
    return {
        "report_markdown_path": str(paths.report_markdown_path),
        "report_tex_path": str(paths.report_tex_path),
        "report_todo_path": str(paths.report_todo_path),
        "tables_dir": str(paths.tables_dir),
        "figures_dir": str(paths.figures_dir),
        "results_path": str(paths.results_path),
    }