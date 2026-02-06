"""Tests for Kelly criterion calculations."""

import numpy as np
import pytest

from src.analysis.kelly import (
    kelly_fraction,
    fractional_kelly,
    multi_kelly,
    expected_log_growth,
    BetOpportunity,
    KellyPortfolio,
)
from src.models.schemas import Platform


class TestKellyFraction:
    def test_no_edge(self):
        """No edge should mean no bet."""
        # Fair odds (50% prob, 2x odds)
        assert kelly_fraction(0.5, 2.0) == 0.0

    def test_positive_edge(self):
        """Positive edge should give positive Kelly fraction."""
        # 60% chance to win at even odds (2x)
        f = kelly_fraction(0.6, 2.0)
        assert f > 0
        assert f == pytest.approx(0.2)  # (2*0.6 - 0.4) / 2 = 0.2

    def test_negative_edge(self):
        """Negative edge should give 0 Kelly fraction."""
        # 40% chance to win at even odds
        assert kelly_fraction(0.4, 2.0) == 0.0

    def test_high_edge(self):
        """High edge should give high Kelly fraction."""
        # 90% chance to win at even odds
        f = kelly_fraction(0.9, 2.0)
        assert f == pytest.approx(0.8)  # (2*0.9 - 0.1) / 2 = 0.8

    def test_edge_cases(self):
        """Test edge cases."""
        # Zero probability
        assert kelly_fraction(0.0, 2.0) == 0.0

        # Certain win
        assert kelly_fraction(1.0, 2.0) == 0.0  # Edge case handling

        # Very low odds
        assert kelly_fraction(0.5, 1.0) == 0.0  # No profit possible


class TestFractionalKelly:
    def test_half_kelly(self):
        """Half Kelly should be half of full Kelly."""
        full = kelly_fraction(0.6, 2.0)
        half = fractional_kelly(0.6, 2.0, 0.5)
        assert half == pytest.approx(full * 0.5)

    def test_quarter_kelly(self):
        """Quarter Kelly should be quarter of full Kelly."""
        full = kelly_fraction(0.7, 2.5)
        quarter = fractional_kelly(0.7, 2.5, 0.25)
        assert quarter == pytest.approx(full * 0.25)


class TestMultiKelly:
    def test_single_bet(self):
        """Single bet should match regular Kelly."""
        probs = np.array([0.6])
        odds = np.array([2.0])

        fractions = multi_kelly(probs, odds)
        expected = kelly_fraction(0.6, 2.0)

        assert fractions[0] == pytest.approx(expected)

    def test_independent_bets(self):
        """Independent bets should be scaled if sum > 1."""
        probs = np.array([0.9, 0.9, 0.9])  # High edge bets
        odds = np.array([2.0, 2.0, 2.0])

        fractions = multi_kelly(probs, odds)

        # Should sum to at most 1
        assert fractions.sum() <= 1.0

    def test_zero_edge_excluded(self):
        """Zero edge bets should get zero allocation."""
        probs = np.array([0.6, 0.5])  # One with edge, one without
        odds = np.array([2.0, 2.0])

        fractions = multi_kelly(probs, odds)

        assert fractions[0] > 0
        assert fractions[1] == 0.0


class TestExpectedLogGrowth:
    def test_zero_bet(self):
        """Zero bet should give zero growth."""
        assert expected_log_growth(0.6, 2.0, 0.0) == 0.0

    def test_kelly_optimal(self):
        """Kelly fraction should maximize expected log growth."""
        p = 0.6
        odds = 2.0
        kelly = kelly_fraction(p, odds)

        # Kelly should give higher growth than slightly off
        growth_kelly = expected_log_growth(p, odds, kelly)
        growth_lower = expected_log_growth(p, odds, kelly * 0.8)
        growth_higher = expected_log_growth(p, odds, kelly * 1.2)

        assert growth_kelly >= growth_lower
        assert growth_kelly >= growth_higher


class TestBetOpportunity:
    def test_yes_bet_odds(self):
        """YES bet odds should be 1/market_price."""
        opp = BetOpportunity(
            market_id="test",
            platform=Platform.POLYMARKET,
            market_price=0.4,
            true_probability=0.5,
            direction="yes",
        )
        assert opp.odds == pytest.approx(2.5)  # 1/0.4

    def test_no_bet_odds(self):
        """NO bet odds should be 1/(1-market_price)."""
        opp = BetOpportunity(
            market_id="test",
            platform=Platform.POLYMARKET,
            market_price=0.4,
            true_probability=0.3,
            direction="no",
        )
        assert opp.odds == pytest.approx(1.667, rel=0.01)  # 1/0.6

    def test_edge_calculation(self):
        """Edge should be difference between true and market probability."""
        opp = BetOpportunity(
            market_id="test",
            platform=Platform.POLYMARKET,
            market_price=0.4,
            true_probability=0.5,
            direction="yes",
        )
        assert opp.edge == pytest.approx(0.1)  # 0.5 - 0.4


class TestKellyPortfolio:
    def test_position_sizing(self):
        """Test that position sizing respects constraints."""
        portfolio = KellyPortfolio(
            bankroll=10000,
            kelly_fraction=0.5,
            max_position_pct=0.25,
        )

        opp = BetOpportunity(
            market_id="test",
            platform=Platform.POLYMARKET,
            market_price=0.3,
            true_probability=0.6,
            direction="yes",
        )

        size = portfolio.calculate_position_size(opp)

        # Should be at most 25% of bankroll
        assert size <= 2500
        assert size > 0

    def test_no_bet_on_negative_edge(self):
        """Should not bet when edge is negative."""
        portfolio = KellyPortfolio(bankroll=10000)

        opp = BetOpportunity(
            market_id="test",
            platform=Platform.POLYMARKET,
            market_price=0.6,
            true_probability=0.5,
            direction="yes",
        )

        size = portfolio.calculate_position_size(opp)
        assert size == 0

    def test_place_bet_updates_bankroll(self):
        """Placing a bet should reduce bankroll."""
        portfolio = KellyPortfolio(bankroll=10000, kelly_fraction=0.5)

        opp = BetOpportunity(
            market_id="test",
            platform=Platform.POLYMARKET,
            market_price=0.4,
            true_probability=0.6,
            direction="yes",
        )

        initial_bankroll = portfolio.bankroll
        position = portfolio.place_bet(opp)

        assert position is not None
        assert portfolio.bankroll < initial_bankroll
        assert len(portfolio.positions) == 1

    def test_resolve_winning_position(self):
        """Resolving a winning position should increase bankroll."""
        portfolio = KellyPortfolio(bankroll=10000, kelly_fraction=0.5)

        opp = BetOpportunity(
            market_id="test",
            platform=Platform.POLYMARKET,
            market_price=0.4,
            true_probability=0.6,
            direction="yes",
        )

        position = portfolio.place_bet(opp)
        bankroll_after_bet = portfolio.bankroll

        # Resolve as YES (win for YES bet)
        pnl = portfolio.resolve_position(position, 1.0)

        assert pnl > 0
        assert portfolio.bankroll > bankroll_after_bet

    def test_resolve_losing_position(self):
        """Resolving a losing position should leave bankroll reduced."""
        portfolio = KellyPortfolio(bankroll=10000, kelly_fraction=0.5)

        opp = BetOpportunity(
            market_id="test",
            platform=Platform.POLYMARKET,
            market_price=0.4,
            true_probability=0.6,
            direction="yes",
        )

        position = portfolio.place_bet(opp)
        bankroll_after_bet = portfolio.bankroll

        # Resolve as NO (loss for YES bet)
        pnl = portfolio.resolve_position(position, 0.0)

        assert pnl < 0
        assert portfolio.bankroll == bankroll_after_bet  # No payout
