"""Forecaster comparison and benchmarking."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from scipy import stats

from src.models.schemas import Platform
from src.models.database import Database
from src.analysis.calibration import brier_score, log_score, calibration_curve


@dataclass
class ForecasterMetrics:
    """Performance metrics for a forecaster or group."""

    forecaster_id: Optional[str]
    name: str
    n_forecasts: int
    n_resolved: int
    brier_score: float
    log_score: float
    calibration_error: float
    avg_confidence: float  # How far from 50% on average


@dataclass
class ComparisonResult:
    """Result of comparing two forecasters/methods."""

    name_a: str
    name_b: str
    brier_diff: float  # Positive means A is better
    log_diff: float
    p_value: float  # Statistical significance of difference
    n_common: int  # Number of questions both forecasted


class ForecasterComparison:
    """Compare forecasting accuracy across methods."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    def community_vs_market(
        self, platform: Platform = Platform.METACULUS
    ) -> Optional[ComparisonResult]:
        """
        Compare Metaculus community predictions vs market prices.

        Returns:
            ComparisonResult or None if insufficient data
        """
        # Get resolved markets with both community and market prices
        df = self.db.get_resolved_markets_with_final_price(platform)

        if df.is_empty():
            return None

        # Get community forecasts
        forecasts_data = []
        for row in df.iter_rows(named=True):
            market_id = row["id"]
            forecasts = self.db.get_forecasts(market_id)

            if forecasts.is_empty():
                continue

            # Get community forecast (is_community = True)
            community = forecasts.filter(pl.col("is_community") == True)
            if not community.is_empty():
                latest_community = community.sort("timestamp", descending=True).head(1)
                forecasts_data.append(
                    {
                        "market_id": market_id,
                        "market_price": row["final_probability"],
                        "community_pred": latest_community["prediction"][0],
                        "resolution": row["resolution"],
                    }
                )

        if len(forecasts_data) < 10:
            return None

        forecasts_df = pl.DataFrame(forecasts_data)

        # Calculate scores
        market_prices = forecasts_df["market_price"].to_numpy()
        community_preds = forecasts_df["community_pred"].to_numpy()
        resolutions = forecasts_df["resolution"].to_numpy()

        market_brier = brier_score(market_prices, resolutions)
        community_brier = brier_score(community_preds, resolutions)

        market_log = log_score(market_prices, resolutions)
        community_log = log_score(community_preds, resolutions)

        # Statistical test for difference
        # Use paired t-test on squared errors
        market_errors = (market_prices - resolutions) ** 2
        community_errors = (community_preds - resolutions) ** 2
        _, p_value = stats.ttest_rel(market_errors, community_errors)

        return ComparisonResult(
            name_a="market_prices",
            name_b="community_forecast",
            brier_diff=community_brier - market_brier,  # Positive means market is better
            log_diff=community_log - market_log,
            p_value=float(p_value),
            n_common=len(forecasts_df),
        )

    def compare_platforms(
        self, platform_a: Platform, platform_b: Platform
    ) -> Optional[ComparisonResult]:
        """
        Compare calibration between two platforms on similar questions.

        This is tricky because platforms have different questions.
        We use linked markets or compare aggregate calibration.
        """
        # Get calibration results for each
        df_a = self.db.get_resolved_markets_with_final_price(platform_a)
        df_b = self.db.get_resolved_markets_with_final_price(platform_b)

        if df_a.is_empty() or df_b.is_empty():
            return None

        # Calculate aggregate scores
        preds_a = df_a["final_probability"].to_numpy()
        outs_a = df_a["resolution"].to_numpy()
        brier_a = brier_score(preds_a, outs_a)
        log_a = log_score(preds_a, outs_a)

        preds_b = df_b["final_probability"].to_numpy()
        outs_b = df_b["resolution"].to_numpy()
        brier_b = brier_score(preds_b, outs_b)
        log_b = log_score(preds_b, outs_b)

        # Bootstrap confidence interval for difference
        n_bootstrap = 1000
        brier_diffs = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            idx_a = np.random.choice(len(preds_a), len(preds_a), replace=True)
            idx_b = np.random.choice(len(preds_b), len(preds_b), replace=True)

            bs_brier_a = brier_score(preds_a[idx_a], outs_a[idx_a])
            bs_brier_b = brier_score(preds_b[idx_b], outs_b[idx_b])
            brier_diffs.append(bs_brier_a - bs_brier_b)

        # Two-sided p-value: proportion of bootstrap samples with different sign
        brier_diffs = np.array(brier_diffs)
        observed_diff = brier_a - brier_b
        if observed_diff > 0:
            p_value = np.mean(brier_diffs <= 0) * 2
        else:
            p_value = np.mean(brier_diffs >= 0) * 2
        p_value = min(p_value, 1.0)

        return ComparisonResult(
            name_a=platform_a.value,
            name_b=platform_b.value,
            brier_diff=brier_b - brier_a,  # Positive means A is better
            log_diff=log_b - log_a,
            p_value=p_value,
            n_common=min(len(df_a), len(df_b)),  # Not directly comparable
        )

    def analyze_crowd_failures(
        self, platform: Optional[Platform] = None, threshold: float = 0.3
    ) -> pl.DataFrame:
        """
        Identify cases where crowds failed badly.

        Args:
            platform: Platform to analyze
            threshold: Minimum prediction error to count as "failure"

        Returns:
            DataFrame of market failures
        """
        df = self.db.get_resolved_markets_with_final_price(platform)

        if df.is_empty():
            return pl.DataFrame()

        # Calculate prediction error
        df = df.with_columns(
            (pl.col("final_probability") - pl.col("resolution")).abs().alias("error"),
            (pl.col("final_probability") - pl.col("resolution")).alias("signed_error"),
        )

        # Filter to failures
        failures = df.filter(pl.col("error") >= threshold)

        # Sort by error magnitude
        failures = failures.sort("error", descending=True)

        return failures

    def category_performance(self, platform: Optional[Platform] = None) -> pl.DataFrame:
        """
        Analyze calibration by question category.

        Returns:
            DataFrame with Brier scores by category
        """
        df = self.db.get_resolved_markets_with_final_price(platform)

        if df.is_empty():
            return pl.DataFrame()

        # Group by category
        results = []
        categories = df["category"].unique().to_list()

        for category in categories:
            if category is None:
                continue

            cat_df = df.filter(pl.col("category") == category)
            if len(cat_df) < 5:
                continue

            preds = cat_df["final_probability"].to_numpy()
            outs = cat_df["resolution"].to_numpy()

            results.append(
                {
                    "category": category,
                    "n_markets": len(cat_df),
                    "brier_score": brier_score(preds, outs),
                    "avg_probability": float(np.mean(preds)),
                    "resolution_rate": float(np.mean(outs)),
                }
            )

        if not results:
            return pl.DataFrame()

        return pl.DataFrame(results).sort("brier_score")

    def time_to_resolution_analysis(
        self, platform: Optional[Platform] = None
    ) -> pl.DataFrame:
        """
        Analyze if calibration differs based on time until resolution.

        Hypothesis: Markets might be better calibrated closer to resolution.

        Returns:
            DataFrame with calibration by time bucket
        """
        # This would require price history at different times before resolution
        # Simplified version using final price only
        df = self.db.get_resolved_markets_with_final_price(platform)

        if df.is_empty():
            return pl.DataFrame()

        # Calculate days from creation to resolution
        df = df.filter(
            pl.col("resolved_at").is_not_null()
        )

        # Would need historical prices at different time points
        # For now, return basic stats
        return df.select(
            ["id", "title", "final_probability", "resolution", "resolved_at"]
        )

    def superforecaster_analysis(self) -> Optional[dict]:
        """
        Compare superforecasters vs regular forecasters on Metaculus.

        Returns:
            Dict with comparison metrics
        """
        # Get all forecasts
        markets = self.db.get_markets(platform=Platform.METACULUS, resolved_only=True)

        if not markets:
            return None

        # Aggregate forecasts by forecaster
        forecaster_data = {}

        for market in markets:
            forecasts = self.db.get_forecasts(market.id)

            if forecasts.is_empty():
                continue

            for row in forecasts.iter_rows(named=True):
                forecaster_id = row["forecaster_id"]
                if forecaster_id is None:
                    continue

                if forecaster_id not in forecaster_data:
                    forecaster_data[forecaster_id] = {
                        "predictions": [],
                        "outcomes": [],
                    }

                forecaster_data[forecaster_id]["predictions"].append(row["prediction"])
                forecaster_data[forecaster_id]["outcomes"].append(market.resolution)

        if not forecaster_data:
            return None

        # Calculate scores for each forecaster
        forecaster_scores = []
        for fid, data in forecaster_data.items():
            if len(data["predictions"]) < 10:
                continue

            preds = np.array(data["predictions"])
            outs = np.array(data["outcomes"])

            # Filter out None outcomes
            valid = ~np.isnan(outs)
            preds = preds[valid]
            outs = outs[valid]

            if len(preds) < 5:
                continue

            forecaster_scores.append(
                {
                    "forecaster_id": fid,
                    "n_forecasts": len(preds),
                    "brier_score": brier_score(preds, outs),
                }
            )

        if not forecaster_scores:
            return None

        scores_df = pl.DataFrame(forecaster_scores).sort("brier_score")

        return {
            "n_forecasters": len(scores_df),
            "best_brier": scores_df["brier_score"].min(),
            "worst_brier": scores_df["brier_score"].max(),
            "median_brier": scores_df["brier_score"].median(),
            "top_10_pct_brier": scores_df.head(max(1, len(scores_df) // 10))["brier_score"].mean(),
        }
