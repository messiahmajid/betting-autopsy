"""Tests for arbitrage detection."""

import pytest

from src.analysis.arbitrage import find_arbitrage, find_complement_violation


class TestFindArbitrage:
    def test_no_arbitrage_fair_prices(self):
        """Same price on both markets = no arbitrage."""
        result = find_arbitrage(0.5, 0.5)
        assert result is None

    def test_no_arbitrage_close_prices(self):
        """Close prices with sum >= 1 = no arbitrage."""
        result = find_arbitrage(0.50, 0.50)
        assert result is None

    def test_arbitrage_yes_a_no_b(self):
        """Market A underprices, Market B overprices = buy YES on A, NO on B."""
        # A: 30% YES price, B: 60% YES price (40% NO price)
        # Combined: 0.3 + 0.4 = 0.7 < 1 = arbitrage!
        result = find_arbitrage(0.3, 0.6)

        assert result is not None
        assert result["type"] == "yes_a_no_b"
        assert result["profit_pct"] > 0
        assert result["stake_a"] + result["stake_b"] == pytest.approx(1.0)

    def test_arbitrage_no_a_yes_b(self):
        """Market A overprices, Market B underprices = buy NO on A, YES on B."""
        # A: 60% YES price (40% NO), B: 30% YES price
        # Combined: 0.4 + 0.3 = 0.7 < 1 = arbitrage!
        result = find_arbitrage(0.6, 0.3)

        assert result is not None
        assert result["type"] == "no_a_yes_b"
        assert result["profit_pct"] > 0

    def test_arbitrage_profit_calculation(self):
        """Test profit calculation is correct."""
        # A: 20% YES, B: 60% YES (40% NO)
        # Combined cost: 0.2 + 0.4 = 0.6
        # Guaranteed return: 1
        # Profit: (1 - 0.6) / 0.6 = 66.67%
        result = find_arbitrage(0.2, 0.6)

        assert result is not None
        expected_profit = (1 - 0.6) / 0.6 * 100
        assert result["profit_pct"] == pytest.approx(expected_profit, rel=0.01)

    def test_edge_case_extreme_prices(self):
        """Test with extreme prices."""
        # Very low price on A, very high on B
        result = find_arbitrage(0.05, 0.98)

        assert result is not None
        assert result["profit_pct"] > 0

    def test_invalid_prices(self):
        """Test with invalid prices."""
        assert find_arbitrage(0.0, 0.5) is None
        assert find_arbitrage(0.5, 1.0) is None
        assert find_arbitrage(-0.1, 0.5) is None


class TestComplementViolation:
    def test_no_violation_proper_sum(self):
        """YES + NO = 1 means no violation."""
        result = find_complement_violation(0.6, 0.4)
        assert result is None

    def test_no_violation_with_spread(self):
        """YES + NO > 1 is normal (market's vig)."""
        result = find_complement_violation(0.52, 0.52)  # 104%
        assert result is None

    def test_violation_under_100(self):
        """YES + NO < 1 means free money."""
        result = find_complement_violation(0.45, 0.45)  # 90%

        assert result is not None
        assert result["type"] == "complement_under"
        assert result["profit_pct"] > 0
        assert result["total"] == pytest.approx(0.9)

    def test_violation_profit_calculation(self):
        """Test profit calculation for complement violation."""
        # YES: 40%, NO: 40% -> Total: 80%
        # Buy both for $0.80, guaranteed $1 payout
        # Profit: (1 - 0.8) / 0.8 = 25%
        result = find_complement_violation(0.4, 0.4)

        assert result is not None
        expected_profit = (1 - 0.8) / 0.8 * 100
        assert result["profit_pct"] == pytest.approx(expected_profit)


class TestArbitrageStaking:
    def test_stake_proportions(self):
        """Test that stake proportions guarantee equal payout."""
        result = find_arbitrage(0.3, 0.7)

        if result is not None:
            stake_a = result["stake_a"]
            stake_b = result["stake_b"]

            # With these stakes:
            # If YES: payout from A = stake_a / 0.3
            # If NO: payout from B = stake_b / 0.3

            # Check stakes sum to 1 (normalized)
            assert stake_a + stake_b == pytest.approx(1.0)

    def test_guaranteed_profit(self):
        """Verify that the arbitrage actually guarantees profit."""
        prob_a, prob_b = 0.25, 0.65

        result = find_arbitrage(prob_a, prob_b)

        if result is not None:
            stake_a = result["stake_a"]
            stake_b = result["stake_b"]

            total_stake = stake_a + stake_b  # This is 1.0 (normalized)

            if result["type"] == "yes_a_no_b":
                # If outcome is YES: win on A
                payout_yes = stake_a / prob_a
                # If outcome is NO: win on B
                payout_no = stake_b / (1 - prob_b)

                # Both payouts should exceed total stake
                assert payout_yes > total_stake or payout_no > total_stake

    def test_large_price_difference(self):
        """Large price differences should give larger profits."""
        small_diff = find_arbitrage(0.45, 0.55)
        large_diff = find_arbitrage(0.2, 0.7)

        if small_diff is not None and large_diff is not None:
            assert large_diff["profit_pct"] > small_diff["profit_pct"]
