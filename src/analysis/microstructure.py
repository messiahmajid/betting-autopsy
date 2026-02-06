"""Market microstructure analysis."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import polars as pl

from src.models.schemas import Platform
from src.models.database import Database


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a market."""

    market_id: str
    avg_volume: float
    total_volume: float
    avg_liquidity: float
    n_price_updates: int
    price_volatility: float
    bid_ask_spread_estimate: float  # Estimated from price movements


@dataclass
class PriceDiscoveryMetrics:
    """Metrics about price discovery speed."""

    market_id: str
    avg_price_change: float
    max_price_change: float
    mean_reversion_speed: float  # How quickly large moves revert
    autocorrelation: float  # Price change autocorrelation


class MicrostructureAnalyzer:
    """Analyze market microstructure."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    def analyze_liquidity(self, market_id: str) -> Optional[LiquidityMetrics]:
        """
        Analyze liquidity metrics for a market.

        Args:
            market_id: Market to analyze

        Returns:
            LiquidityMetrics or None if insufficient data
        """
        prices = self.db.get_prices(market_id)

        if prices.is_empty() or len(prices) < 10:
            return None

        # Calculate metrics
        avg_volume = prices["volume"].mean() if prices["volume"].null_count() < len(prices) else 0
        total_volume = prices["volume"].sum() if prices["volume"].null_count() < len(prices) else 0
        avg_liquidity = (
            prices["liquidity"].mean() if prices["liquidity"].null_count() < len(prices) else 0
        )

        # Price volatility (standard deviation of returns)
        prob_series = prices["probability"].to_numpy()
        if len(prob_series) > 1:
            returns = np.diff(prob_series)
            price_volatility = float(np.std(returns))
        else:
            price_volatility = 0.0

        # Estimate bid-ask spread from price reversals
        # Intuition: in a market with spread, prices often bounce between bid and ask
        spread_estimate = self._estimate_spread(prob_series)

        return LiquidityMetrics(
            market_id=market_id,
            avg_volume=avg_volume or 0,
            total_volume=total_volume or 0,
            avg_liquidity=avg_liquidity or 0,
            n_price_updates=len(prices),
            price_volatility=price_volatility,
            bid_ask_spread_estimate=spread_estimate,
        )

    def _estimate_spread(self, prices: np.ndarray) -> float:
        """
        Estimate bid-ask spread using Roll's measure.

        Roll (1984) showed that the spread can be estimated from
        the autocovariance of price changes.
        """
        if len(prices) < 3:
            return 0.0

        returns = np.diff(prices)
        if len(returns) < 2:
            return 0.0

        # Roll's measure: spread = 2 * sqrt(-cov(r_t, r_{t-1}))
        autocov = np.cov(returns[:-1], returns[1:])[0, 1]

        if autocov < 0:
            return 2 * np.sqrt(-autocov)
        return 0.0

    def analyze_price_discovery(self, market_id: str) -> Optional[PriceDiscoveryMetrics]:
        """
        Analyze price discovery characteristics.

        Args:
            market_id: Market to analyze

        Returns:
            PriceDiscoveryMetrics or None if insufficient data
        """
        prices = self.db.get_prices(market_id)

        if prices.is_empty() or len(prices) < 10:
            return None

        prob_series = prices["probability"].to_numpy()
        returns = np.diff(prob_series)

        if len(returns) < 2:
            return None

        # Basic statistics
        avg_change = float(np.mean(np.abs(returns)))
        max_change = float(np.max(np.abs(returns)))

        # Autocorrelation of returns
        if np.std(returns) > 0:
            autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
        else:
            autocorr = 0.0

        # Mean reversion speed (how quickly large moves revert)
        # Use partial autocorrelation or simple measure
        large_moves = np.abs(returns) > 2 * np.std(returns)
        if large_moves.sum() > 0 and len(returns) > np.where(large_moves)[0].max() + 1:
            # Look at return following large moves
            following_returns = returns[1:][large_moves[:-1]]
            if len(following_returns) > 0:
                mean_reversion = float(-np.mean(following_returns * np.sign(returns[:-1][large_moves[:-1]])))
            else:
                mean_reversion = 0.0
        else:
            mean_reversion = 0.0

        return PriceDiscoveryMetrics(
            market_id=market_id,
            avg_price_change=avg_change,
            max_price_change=max_change,
            mean_reversion_speed=mean_reversion,
            autocorrelation=autocorr,
        )

    def event_study(
        self,
        market_id: str,
        event_time: datetime,
        window_hours: int = 24,
    ) -> pl.DataFrame:
        """
        Conduct event study around a specific time.

        Useful for analyzing price movement around news events.

        Args:
            market_id: Market to analyze
            event_time: Time of the event
            window_hours: Hours before and after to include

        Returns:
            DataFrame with price data around the event
        """
        start_time = event_time - timedelta(hours=window_hours)
        end_time = event_time + timedelta(hours=window_hours)

        prices = self.db.get_prices(market_id, start_time, end_time)

        if prices.is_empty():
            return pl.DataFrame()

        # Add relative time column
        prices = prices.with_columns(
            ((pl.col("timestamp") - event_time).dt.total_seconds() / 3600).alias("hours_from_event")
        )

        return prices

    def favorite_longshot_by_liquidity(
        self, platform: Optional[Platform] = None
    ) -> pl.DataFrame:
        """
        Analyze if favorite-longshot bias varies by liquidity.

        Hypothesis: Less liquid markets might show more bias.

        Returns:
            DataFrame with bias analysis by liquidity quartile
        """
        df = self.db.get_resolved_markets_with_final_price(platform)

        if df.is_empty():
            return pl.DataFrame()

        # Get liquidity data for each market
        records = []
        for row in df.iter_rows(named=True):
            market_id = row["id"]
            prices = self.db.get_prices(market_id)

            if prices.is_empty():
                continue

            avg_liquidity = prices["liquidity"].mean()
            if avg_liquidity is None:
                avg_liquidity = 0

            records.append(
                {
                    "market_id": market_id,
                    "final_probability": row["final_probability"],
                    "resolution": row["resolution"],
                    "avg_liquidity": avg_liquidity,
                }
            )

        if not records:
            return pl.DataFrame()

        result_df = pl.DataFrame(records)

        # Add liquidity quartile
        result_df = result_df.with_columns(
            pl.col("avg_liquidity")
            .qcut(4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
            .alias("liquidity_quartile")
        )

        # Calculate bias by quartile
        # Bias = predicted - actual frequency
        # Group by probability bucket and liquidity quartile
        result_df = result_df.with_columns(
            (pl.col("final_probability") * 10).floor().cast(pl.Int32).alias("prob_bucket")
        )

        bias_analysis = (
            result_df.group_by(["liquidity_quartile", "prob_bucket"])
            .agg(
                pl.mean("final_probability").alias("avg_predicted"),
                pl.mean("resolution").alias("avg_actual"),
                pl.len().alias("count"),
            )
            .with_columns((pl.col("avg_predicted") - pl.col("avg_actual")).alias("bias"))
            .sort(["liquidity_quartile", "prob_bucket"])
        )

        return bias_analysis

    def price_impact_analysis(self, market_id: str) -> dict:
        """
        Analyze price impact of trades.

        Estimates how much prices move per unit of volume.

        Returns:
            Dict with price impact metrics
        """
        prices = self.db.get_prices(market_id)

        if prices.is_empty() or len(prices) < 10:
            return {}

        # Calculate price changes and volume
        df = prices.sort("timestamp")
        df = df.with_columns(
            pl.col("probability").diff().alias("price_change"),
            pl.col("volume").diff().alias("volume_change"),
        )

        # Filter to rows with valid data
        df = df.filter(
            pl.col("price_change").is_not_null() & pl.col("volume_change").is_not_null()
        )

        if df.is_empty():
            return {}

        # Kyle's lambda: price_change = lambda * volume
        # Simple regression
        price_changes = df["price_change"].to_numpy()
        volume_changes = df["volume_change"].to_numpy()

        if np.std(volume_changes) > 0:
            # Simple correlation-based estimate
            kyle_lambda = np.cov(price_changes, volume_changes)[0, 1] / np.var(volume_changes)
        else:
            kyle_lambda = 0

        return {
            "kyle_lambda": float(kyle_lambda),
            "avg_price_change": float(np.mean(np.abs(price_changes))),
            "avg_volume_per_update": float(np.mean(np.abs(volume_changes))),
            "price_volume_correlation": float(np.corrcoef(price_changes, volume_changes)[0, 1])
            if len(price_changes) > 1
            else 0,
        }
