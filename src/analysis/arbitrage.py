"""Arbitrage detection across prediction markets."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import polars as pl

from src.models.schemas import ArbitrageOpportunity, MarketLink, Platform
from src.models.database import Database


def find_arbitrage(prob_a: float, prob_b: float) -> Optional[dict]:
    """
    Find arbitrage between two markets pricing the same event.

    Arbitrage exists when you can bet on both sides and guarantee profit.
    For the same event on markets A and B:
    - If prob_a + (1 - prob_b) < 1, buy YES on A and NO on B
    - If (1 - prob_a) + prob_b < 1, buy NO on A and YES on B

    Args:
        prob_a: Probability on market A
        prob_b: Probability on market B

    Returns:
        Dict with arbitrage details or None if no arbitrage
    """
    if prob_a <= 0 or prob_a >= 1 or prob_b <= 0 or prob_b >= 1:
        return None

    # Strategy 1: YES on A, NO on B
    combined_1 = prob_a + (1 - prob_b)
    if combined_1 < 1:
        profit_pct = (1 - combined_1) / combined_1 * 100
        return {
            "type": "yes_a_no_b",
            "profit_pct": profit_pct,
            "stake_a": prob_a / combined_1,  # Fraction to bet on A
            "stake_b": (1 - prob_b) / combined_1,  # Fraction to bet on B
            "combined_cost": combined_1,
        }

    # Strategy 2: NO on A, YES on B
    combined_2 = (1 - prob_a) + prob_b
    if combined_2 < 1:
        profit_pct = (1 - combined_2) / combined_2 * 100
        return {
            "type": "no_a_yes_b",
            "profit_pct": profit_pct,
            "stake_a": (1 - prob_a) / combined_2,
            "stake_b": prob_b / combined_2,
            "combined_cost": combined_2,
        }

    return None


def find_complement_violation(yes_price: float, no_price: float) -> Optional[dict]:
    """
    Check if YES + NO prices don't sum to ~100% (accounting for spread).

    On a single market, if YES + NO < 1, there's an arbitrage.
    If YES + NO > 1, that's the market's spread/vig.

    Args:
        yes_price: Price of YES contract
        no_price: Price of NO contract

    Returns:
        Dict with violation details or None
    """
    total = yes_price + no_price

    if total < 0.99:  # Allow small tolerance
        # Can buy both YES and NO and guarantee profit
        profit_pct = (1 - total) / total * 100
        return {
            "type": "complement_under",
            "yes_price": yes_price,
            "no_price": no_price,
            "total": total,
            "profit_pct": profit_pct,
        }

    return None


@dataclass
class ArbitrageScanner:
    """Scan for arbitrage opportunities across markets."""

    db: Database = None

    def __post_init__(self):
        if self.db is None:
            self.db = Database()

    def scan_linked_markets(self) -> list[ArbitrageOpportunity]:
        """
        Scan all linked markets for arbitrage opportunities.

        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities = []
        links = self.db.get_market_links()

        for link in links:
            platform_ids = link.get_platform_ids()
            if len(platform_ids) < 2:
                continue

            # Get latest prices for each linked market
            prices = {}
            for platform, market_id in platform_ids.items():
                market_prices = self.db.get_prices(market_id)
                if not market_prices.is_empty():
                    latest = market_prices.sort("timestamp", descending=True).head(1)
                    prices[platform] = latest["probability"][0]

            # Check all pairs for arbitrage
            platforms = list(prices.keys())
            for i, platform_a in enumerate(platforms):
                for platform_b in platforms[i + 1 :]:
                    prob_a = prices[platform_a]
                    prob_b = prices[platform_b]

                    arb = find_arbitrage(prob_a, prob_b)
                    if arb:
                        opportunities.append(
                            ArbitrageOpportunity(
                                market_link_id=link.id,
                                timestamp=datetime.now(),
                                platform_a=platform_a,
                                platform_b=platform_b,
                                prob_a=prob_a,
                                prob_b=prob_b,
                                profit_pct=arb["profit_pct"],
                                stake_a=arb["stake_a"],
                                stake_b=arb["stake_b"],
                                arb_type=arb["type"],
                            )
                        )

        return opportunities

    def find_similar_markets(
        self, title_similarity_threshold: float = 0.8
    ) -> list[tuple[str, str, float]]:
        """
        Find potentially linked markets based on title similarity.

        This helps identify markets that might be the same event
        but aren't explicitly linked.

        Returns:
            List of (market_id_1, market_id_2, similarity_score) tuples
        """
        # Get all markets
        markets = self.db.get_markets()

        # Group by platform
        by_platform = {}
        for market in markets:
            if market.platform not in by_platform:
                by_platform[market.platform] = []
            by_platform[market.platform].append(market)

        # Compare titles across platforms
        similar = []
        platforms = list(by_platform.keys())

        for i, platform_a in enumerate(platforms):
            for platform_b in platforms[i + 1 :]:
                for market_a in by_platform[platform_a]:
                    for market_b in by_platform[platform_b]:
                        similarity = self._title_similarity(market_a.title, market_b.title)
                        if similarity >= title_similarity_threshold:
                            similar.append((market_a.id, market_b.id, similarity))

        return sorted(similar, key=lambda x: x[2], reverse=True)

    def _title_similarity(self, title_a: str, title_b: str) -> float:
        """
        Calculate similarity between two titles using Jaccard similarity.

        Simple word-based similarity. For production, use something like
        sentence-transformers for semantic similarity.
        """
        words_a = set(title_a.lower().split())
        words_b = set(title_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union)

    def historical_arbitrage_analysis(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """
        Analyze historical arbitrage opportunities.

        For each linked market pair, look at price history to find
        periods where arbitrage existed.

        Returns:
            DataFrame with historical arbitrage data
        """
        links = self.db.get_market_links()
        records = []

        for link in links:
            platform_ids = link.get_platform_ids()
            if len(platform_ids) < 2:
                continue

            # Get price histories
            price_histories = {}
            for platform, market_id in platform_ids.items():
                prices = self.db.get_prices(market_id, start_date, end_date)
                if not prices.is_empty():
                    price_histories[platform] = prices

            if len(price_histories) < 2:
                continue

            # Align price histories by timestamp (simplified - join on nearest time)
            platforms = list(price_histories.keys())
            for i, platform_a in enumerate(platforms):
                for platform_b in platforms[i + 1 :]:
                    df_a = price_histories[platform_a]
                    df_b = price_histories[platform_b]

                    # For each price in A, find closest price in B
                    for row_a in df_a.iter_rows(named=True):
                        ts_a = row_a["timestamp"]
                        prob_a = row_a["probability"]

                        # Find closest timestamp in B
                        df_b_sorted = df_b.with_columns(
                            (pl.col("timestamp") - ts_a).abs().alias("time_diff")
                        ).sort("time_diff")

                        if df_b_sorted.is_empty():
                            continue

                        closest_b = df_b_sorted.head(1)
                        prob_b = closest_b["probability"][0]
                        time_diff = closest_b["time_diff"][0]

                        # Only consider if within reasonable time window (1 hour)
                        if hasattr(time_diff, "total_seconds"):
                            if abs(time_diff.total_seconds()) > 3600:
                                continue

                        # Check for arbitrage
                        arb = find_arbitrage(prob_a, prob_b)
                        if arb:
                            records.append(
                                {
                                    "link_id": link.id,
                                    "timestamp": ts_a,
                                    "platform_a": platform_a.value,
                                    "platform_b": platform_b.value,
                                    "prob_a": prob_a,
                                    "prob_b": prob_b,
                                    "arb_type": arb["type"],
                                    "profit_pct": arb["profit_pct"],
                                }
                            )

        if not records:
            return pl.DataFrame()

        return pl.DataFrame(records)

    def arbitrage_statistics(self) -> dict:
        """
        Calculate summary statistics about arbitrage opportunities.

        Returns:
            Dict with arbitrage statistics
        """
        hist = self.historical_arbitrage_analysis()

        if hist.is_empty():
            return {
                "total_opportunities": 0,
                "avg_profit_pct": 0,
                "max_profit_pct": 0,
                "by_platform_pair": {},
            }

        return {
            "total_opportunities": len(hist),
            "avg_profit_pct": hist["profit_pct"].mean(),
            "max_profit_pct": hist["profit_pct"].max(),
            "by_platform_pair": hist.group_by(["platform_a", "platform_b"])
            .agg(pl.len().alias("count"), pl.mean("profit_pct").alias("avg_profit"))
            .to_dicts(),
        }
