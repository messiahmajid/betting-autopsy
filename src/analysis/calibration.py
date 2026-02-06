"""Calibration analysis for prediction markets."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from scipy import stats

from src.models.schemas import CalibrationBucket, Platform
from src.models.database import Database


def brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Calculate Brier score (mean squared error between predictions and outcomes).

    Lower is better. Perfect calibration = 0, random guessing = 0.25 for binary.

    Args:
        predictions: Array of predicted probabilities (0-1)
        outcomes: Array of actual outcomes (0 or 1)

    Returns:
        Brier score
    """
    predictions = np.asarray(predictions)
    outcomes = np.asarray(outcomes)
    return float(np.mean((predictions - outcomes) ** 2))


def log_score(predictions: np.ndarray, outcomes: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate log score (negative log likelihood).

    Lower is better (less negative). Measures information content.

    Args:
        predictions: Array of predicted probabilities (0-1)
        outcomes: Array of actual outcomes (0 or 1)
        epsilon: Small value to avoid log(0)

    Returns:
        Log score (negative, closer to 0 is better)
    """
    predictions = np.asarray(predictions)
    outcomes = np.asarray(outcomes)

    # Clip to avoid log(0)
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    # Log score: sum of log(p) for correct predictions
    log_scores = outcomes * np.log(predictions) + (1 - outcomes) * np.log(1 - predictions)
    return float(np.mean(log_scores))


def calibration_curve(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> list[CalibrationBucket]:
    """
    Compute calibration curve by bucketing predictions.

    Args:
        predictions: Array of predicted probabilities (0-1)
        outcomes: Array of actual outcomes (0 or 1)
        n_bins: Number of bins
        strategy: 'uniform' for equal-width bins, 'quantile' for equal-count bins

    Returns:
        List of CalibrationBucket objects
    """
    predictions = np.asarray(predictions)
    outcomes = np.asarray(outcomes)

    if strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bins = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))
        bins[0] = 0.0
        bins[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    calibration = []
    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (predictions >= bins[i]) & (predictions <= bins[i + 1])

        count = int(mask.sum())
        if count > 0:
            predicted_mean = float(predictions[mask].mean())
            actual_freq = float(outcomes[mask].mean())
            std_error = float(np.sqrt(actual_freq * (1 - actual_freq) / count)) if count > 1 else None

            calibration.append(
                CalibrationBucket(
                    bin_start=float(bins[i]),
                    bin_end=float(bins[i + 1]),
                    predicted_mean=predicted_mean,
                    actual_frequency=actual_freq,
                    count=count,
                    std_error=std_error,
                )
            )

    return calibration


def hosmer_lemeshow_test(
    predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> tuple[float, float]:
    """
    Hosmer-Lemeshow goodness-of-fit test for calibration.

    Tests whether observed frequencies match predicted probabilities.

    Args:
        predictions: Array of predicted probabilities
        outcomes: Array of actual outcomes

    Returns:
        Tuple of (chi-squared statistic, p-value)
        High p-value (>0.05) suggests good calibration.
    """
    predictions = np.asarray(predictions)
    outcomes = np.asarray(outcomes)

    # Create bins based on predicted probabilities
    bins = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))
    bins[0] = 0.0
    bins[-1] = 1.0 + 1e-10

    chi2 = 0.0
    df = 0

    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
        n_g = mask.sum()

        if n_g == 0:
            continue

        observed = outcomes[mask].sum()
        expected = predictions[mask].sum()

        # Avoid division by zero
        if expected > 0 and expected < n_g:
            chi2 += (observed - expected) ** 2 / (expected * (1 - expected / n_g))
            df += 1

    # Chi-squared test with df-2 degrees of freedom
    df = max(1, df - 2)
    p_value = 1 - stats.chi2.cdf(chi2, df)

    return float(chi2), float(p_value)


def calibration_error(
    predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        predictions: Array of predicted probabilities
        outcomes: Array of actual outcomes
        n_bins: Number of bins

    Returns:
        Dict with 'ece' and 'mce' values
    """
    curve = calibration_curve(predictions, outcomes, n_bins)
    total_count = sum(b.count for b in curve)

    ece = 0.0
    mce = 0.0

    for bucket in curve:
        error = abs(bucket.predicted_mean - bucket.actual_frequency)
        weight = bucket.count / total_count
        ece += weight * error
        mce = max(mce, error)

    return {"ece": ece, "mce": mce}


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""

    platform: Optional[Platform]
    n_markets: int
    brier_score: float
    log_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    hl_chi2: float  # Hosmer-Lemeshow chi-squared
    hl_pvalue: float  # Hosmer-Lemeshow p-value
    calibration_curve: list[CalibrationBucket]


class CalibrationAnalyzer:
    """Analyze calibration of prediction markets."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    def analyze_platform(
        self,
        platform: Optional[Platform] = None,
        n_bins: int = 10,
    ) -> CalibrationResult:
        """
        Analyze calibration for a platform.

        Args:
            platform: Platform to analyze (None for all)
            n_bins: Number of bins for calibration curve

        Returns:
            CalibrationResult with all metrics
        """
        # Get resolved markets with final prices
        df = self.db.get_resolved_markets_with_final_price(platform)

        if df.is_empty():
            raise ValueError(f"No resolved markets found for {platform}")

        # Extract predictions and outcomes
        predictions = df["final_probability"].to_numpy()
        outcomes = df["resolution"].to_numpy()

        # Filter out any NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(outcomes))
        predictions = predictions[valid_mask]
        outcomes = outcomes[valid_mask]

        if len(predictions) == 0:
            raise ValueError("No valid prediction/outcome pairs")

        # Calculate metrics
        brier = brier_score(predictions, outcomes)
        log_s = log_score(predictions, outcomes)
        errors = calibration_error(predictions, outcomes, n_bins)
        hl_chi2, hl_pvalue = hosmer_lemeshow_test(predictions, outcomes, n_bins)
        curve = calibration_curve(predictions, outcomes, n_bins)

        return CalibrationResult(
            platform=platform,
            n_markets=len(predictions),
            brier_score=brier,
            log_score=log_s,
            ece=errors["ece"],
            mce=errors["mce"],
            hl_chi2=hl_chi2,
            hl_pvalue=hl_pvalue,
            calibration_curve=curve,
        )

    def compare_platforms(self, n_bins: int = 10) -> dict[str, CalibrationResult]:
        """
        Compare calibration across all platforms with data.

        Returns:
            Dict mapping platform name to CalibrationResult
        """
        results = {}

        # Get overall calibration
        try:
            results["all"] = self.analyze_platform(None, n_bins)
        except ValueError:
            pass

        # Get per-platform calibration
        for platform in Platform:
            try:
                results[platform.value] = self.analyze_platform(platform, n_bins)
            except ValueError:
                continue

        return results

    def favorite_longshot_bias(
        self, platform: Optional[Platform] = None, n_bins: int = 10
    ) -> pl.DataFrame:
        """
        Analyze favorite-longshot bias.

        Checks if low-probability events are overpriced (positive bias)
        or underpriced (negative bias).

        Returns:
            DataFrame with bin ranges and bias metrics
        """
        df = self.db.get_resolved_markets_with_final_price(platform)

        if df.is_empty():
            raise ValueError("No data available")

        predictions = df["final_probability"].to_numpy()
        outcomes = df["resolution"].to_numpy()

        curve = calibration_curve(predictions, outcomes, n_bins)

        records = []
        for bucket in curve:
            bias = bucket.predicted_mean - bucket.actual_frequency
            records.append(
                {
                    "bin_start": bucket.bin_start,
                    "bin_end": bucket.bin_end,
                    "predicted": bucket.predicted_mean,
                    "actual": bucket.actual_frequency,
                    "bias": bias,  # positive = overpriced
                    "count": bucket.count,
                    "is_longshot": bucket.predicted_mean < 0.2,
                    "is_favorite": bucket.predicted_mean > 0.8,
                }
            )

        return pl.DataFrame(records)
