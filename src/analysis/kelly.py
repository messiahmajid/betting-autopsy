"""Kelly criterion and portfolio optimization."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl

from src.models.schemas import Platform, PortfolioPosition
from src.models.database import Database


def kelly_fraction(p: float, odds: float) -> float:
    """
    Calculate optimal Kelly bet fraction.

    Args:
        p: True probability of winning (0-1)
        odds: Decimal odds (e.g., 2.0 means you double your money)
              For prediction markets: odds = 1/market_price

    Returns:
        Optimal fraction of bankroll to bet (0 to 1)
        Returns 0 if no positive edge.
    """
    if p <= 0 or p >= 1 or odds <= 1:
        return 0.0

    q = 1 - p  # Probability of losing
    b = odds - 1  # Net odds (profit per unit bet)

    # Kelly formula: f = (bp - q) / b
    f = (b * p - q) / b

    return max(0.0, f)


def fractional_kelly(p: float, odds: float, fraction: float = 0.5) -> float:
    """
    Calculate fractional Kelly bet (e.g., half-Kelly).

    Reduces variance at the cost of slightly lower expected growth.

    Args:
        p: True probability of winning
        odds: Decimal odds
        fraction: Fraction of full Kelly to use (default 0.5 = half-Kelly)

    Returns:
        Fractional Kelly bet size
    """
    return kelly_fraction(p, odds) * fraction


def multi_kelly(
    probabilities: np.ndarray,
    odds: np.ndarray,
    correlations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate Kelly fractions for multiple simultaneous bets.

    For uncorrelated bets, this is simply the sum of individual Kelly fractions,
    but the total should be capped at 1.

    For correlated bets, a more sophisticated approach is needed.

    Args:
        probabilities: Array of true probabilities for each bet
        odds: Array of decimal odds for each bet
        correlations: Optional correlation matrix between outcomes

    Returns:
        Array of bet fractions for each opportunity
    """
    n = len(probabilities)
    fractions = np.array([kelly_fraction(p, o) for p, o in zip(probabilities, odds)])

    # If no correlations provided, assume independent
    if correlations is None:
        # Simple approach: scale down if total exceeds 1
        total = fractions.sum()
        if total > 1:
            fractions = fractions / total
        return fractions

    # With correlations, use quadratic optimization (simplified version)
    # This is a rough approximation - full solution requires convex optimization
    # Reduce allocation for positively correlated bets
    for i in range(n):
        for j in range(i + 1, n):
            if correlations[i, j] > 0:
                reduction = correlations[i, j] * 0.5
                fractions[i] *= 1 - reduction
                fractions[j] *= 1 - reduction

    # Ensure we don't exceed bankroll
    total = fractions.sum()
    if total > 1:
        fractions = fractions / total

    return fractions


def expected_log_growth(p: float, odds: float, fraction: float) -> float:
    """
    Calculate expected logarithmic growth rate for a bet.

    This is what Kelly criterion maximizes.

    Args:
        p: True probability of winning
        odds: Decimal odds
        fraction: Fraction of bankroll to bet

    Returns:
        Expected log growth rate
    """
    if fraction <= 0:
        return 0.0
    if fraction >= 1:
        return float("-inf")

    q = 1 - p
    win_growth = np.log(1 + fraction * (odds - 1))
    lose_growth = np.log(1 - fraction)

    return p * win_growth + q * lose_growth


@dataclass
class BetOpportunity:
    """A betting opportunity."""

    market_id: str
    platform: Platform
    market_price: float  # Current market probability
    true_probability: float  # Our estimated probability
    direction: str  # 'yes' or 'no'

    @property
    def odds(self) -> float:
        """Decimal odds for this bet."""
        if self.direction == "yes":
            return 1 / self.market_price if self.market_price > 0 else float("inf")
        else:
            return 1 / (1 - self.market_price) if self.market_price < 1 else float("inf")

    @property
    def edge(self) -> float:
        """Expected edge on this bet."""
        if self.direction == "yes":
            return self.true_probability - self.market_price
        else:
            return (1 - self.true_probability) - (1 - self.market_price)

    @property
    def kelly(self) -> float:
        """Full Kelly fraction for this bet."""
        if self.direction == "yes":
            return kelly_fraction(self.true_probability, self.odds)
        else:
            return kelly_fraction(1 - self.true_probability, self.odds)


@dataclass
class BacktestResult:
    """Results from a portfolio backtest."""

    strategy: str
    initial_bankroll: float
    final_bankroll: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    n_bets: int
    win_rate: float
    avg_edge: float
    history: pl.DataFrame


@dataclass
class KellyPortfolio:
    """Portfolio manager using Kelly criterion."""

    bankroll: float = 10000.0
    kelly_fraction: float = 0.5  # Use half-Kelly by default
    max_position_pct: float = 0.25  # Max 25% of bankroll on single bet
    positions: list[PortfolioPosition] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    def calculate_position_size(self, opportunity: BetOpportunity) -> float:
        """
        Calculate position size for an opportunity.

        Args:
            opportunity: The betting opportunity

        Returns:
            Dollar amount to bet
        """
        if opportunity.edge <= 0:
            return 0.0

        # Calculate Kelly fraction
        kelly = opportunity.kelly * self.kelly_fraction

        # Apply maximum position constraint
        kelly = min(kelly, self.max_position_pct)

        return self.bankroll * kelly

    def place_bet(
        self, opportunity: BetOpportunity, timestamp: Optional[datetime] = None
    ) -> Optional[PortfolioPosition]:
        """
        Place a bet on an opportunity.

        Args:
            opportunity: The betting opportunity
            timestamp: Time of bet (default: now)

        Returns:
            PortfolioPosition if bet was placed, None otherwise
        """
        size = self.calculate_position_size(opportunity)
        if size <= 0:
            return None

        timestamp = timestamp or datetime.now()

        position = PortfolioPosition(
            market_id=opportunity.market_id,
            platform=opportunity.platform,
            direction=opportunity.direction,
            stake=size,
            entry_price=opportunity.market_price,
            entry_time=timestamp,
        )

        self.positions.append(position)
        self.bankroll -= size

        self.history.append(
            {
                "timestamp": timestamp,
                "action": "bet",
                "market_id": opportunity.market_id,
                "direction": opportunity.direction,
                "stake": size,
                "price": opportunity.market_price,
                "bankroll": self.bankroll,
            }
        )

        return position

    def resolve_position(
        self,
        position: PortfolioPosition,
        resolution: float,  # 1.0 = yes, 0.0 = no
        timestamp: Optional[datetime] = None,
    ) -> float:
        """
        Resolve a position and update bankroll.

        Args:
            position: The position to resolve
            resolution: Market resolution (1.0 = yes, 0.0 = no)
            timestamp: Time of resolution

        Returns:
            P&L from this position
        """
        timestamp = timestamp or datetime.now()

        # Calculate payout
        if position.direction == "yes":
            won = resolution >= 0.5
            payout = position.stake / position.entry_price if won else 0
        else:
            won = resolution < 0.5
            payout = position.stake / (1 - position.entry_price) if won else 0

        pnl = payout - position.stake
        self.bankroll += payout

        position.exit_price = resolution
        position.exit_time = timestamp
        position.pnl = pnl

        self.history.append(
            {
                "timestamp": timestamp,
                "action": "resolve",
                "market_id": position.market_id,
                "direction": position.direction,
                "pnl": pnl,
                "bankroll": self.bankroll,
            }
        )

        return pnl

    def get_metrics(self) -> dict:
        """Calculate portfolio performance metrics."""
        if not self.history:
            return {}

        history_df = pl.DataFrame(self.history)
        bankroll_series = history_df["bankroll"].to_numpy()

        # Calculate returns
        returns = np.diff(bankroll_series) / bankroll_series[:-1]

        # Max drawdown
        peak = np.maximum.accumulate(bankroll_series)
        drawdown = (peak - bankroll_series) / peak
        max_drawdown = drawdown.max()

        # Risk-adjusted returns
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            negative_returns = returns[returns < 0]
            sortino = (
                returns.mean() / negative_returns.std() * np.sqrt(252)
                if len(negative_returns) > 0
                else float("inf")
            )
        else:
            sharpe = 0.0
            sortino = 0.0

        # Win rate
        resolved = [p for p in self.positions if p.pnl is not None]
        wins = [p for p in resolved if p.pnl > 0]
        win_rate = len(wins) / len(resolved) if resolved else 0

        return {
            "total_return": (self.bankroll / 10000 - 1) * 100,  # Percentage
            "max_drawdown": max_drawdown * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "n_bets": len(self.positions),
            "n_resolved": len(resolved),
            "win_rate": win_rate * 100,
        }


class BacktestEngine:
    """Backtest Kelly strategies on historical data."""

    def __init__(self, db: Optional[Database] = None):
        self.db = db or Database()

    def run_backtest(
        self,
        edge_estimate: float = 0.05,  # Assumed edge over market
        kelly_fraction: float = 0.5,
        initial_bankroll: float = 10000.0,
        platform: Optional[Platform] = None,
    ) -> BacktestResult:
        """
        Run a simple backtest assuming constant edge.

        This is a simplified backtest that assumes you have a fixed edge
        over the market. In practice, you'd need a model to estimate true
        probabilities.

        Args:
            edge_estimate: Assumed edge (e.g., 0.05 = 5% better than market)
            kelly_fraction: Fraction of Kelly to use
            initial_bankroll: Starting capital
            platform: Platform to backtest on (None for all)

        Returns:
            BacktestResult with performance metrics
        """
        # Get resolved markets with prices
        df = self.db.get_resolved_markets_with_final_price(platform)

        if df.is_empty():
            raise ValueError("No data for backtest")

        portfolio = KellyPortfolio(
            bankroll=initial_bankroll, kelly_fraction=kelly_fraction
        )

        # Sort by resolution time
        df = df.sort("resolved_at")

        for row in df.iter_rows(named=True):
            market_price = row["final_probability"]
            resolution = row["resolution"]

            if market_price is None or resolution is None:
                continue

            # Skip extreme prices
            if market_price < 0.05 or market_price > 0.95:
                continue

            # Determine direction based on edge
            # If market underestimates, bet YES
            # If market overestimates, bet NO
            true_prob_estimate = market_price + edge_estimate

            if true_prob_estimate > market_price and true_prob_estimate < 1:
                direction = "yes"
                true_prob = true_prob_estimate
            elif true_prob_estimate < market_price and true_prob_estimate > 0:
                direction = "no"
                true_prob = 1 - true_prob_estimate
            else:
                continue

            opportunity = BetOpportunity(
                market_id=row["id"],
                platform=Platform(row["platform"]),
                market_price=market_price,
                true_probability=true_prob,
                direction=direction,
            )

            position = portfolio.place_bet(opportunity, timestamp=row["resolved_at"])

            if position:
                portfolio.resolve_position(position, resolution, row["resolved_at"])

        metrics = portfolio.get_metrics()

        return BacktestResult(
            strategy=f"kelly_{kelly_fraction}_edge_{edge_estimate}",
            initial_bankroll=initial_bankroll,
            final_bankroll=portfolio.bankroll,
            total_return=metrics.get("total_return", 0),
            annualized_return=0,  # Would need time span calculation
            max_drawdown=metrics.get("max_drawdown", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            sortino_ratio=metrics.get("sortino_ratio", 0),
            n_bets=metrics.get("n_bets", 0),
            win_rate=metrics.get("win_rate", 0),
            avg_edge=edge_estimate * 100,
            history=pl.DataFrame(portfolio.history) if portfolio.history else pl.DataFrame(),
        )
