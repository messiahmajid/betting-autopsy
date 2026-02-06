"""Tests for calibration analysis."""

import numpy as np
import pytest

from src.analysis.calibration import (
    brier_score,
    log_score,
    calibration_curve,
    calibration_error,
    hosmer_lemeshow_test,
)


class TestBrierScore:
    def test_perfect_predictions(self):
        """Perfect predictions should have Brier score of 0."""
        predictions = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1, 0, 1, 0])
        assert brier_score(predictions, outcomes) == 0.0

    def test_worst_predictions(self):
        """Completely wrong predictions should have Brier score of 1."""
        predictions = np.array([0.0, 1.0, 0.0, 1.0])
        outcomes = np.array([1, 0, 1, 0])
        assert brier_score(predictions, outcomes) == 1.0

    def test_random_baseline(self):
        """50% predictions should give Brier score of 0.25."""
        predictions = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1, 0, 1, 0])
        assert brier_score(predictions, outcomes) == 0.25

    def test_calibrated_predictions(self):
        """Well-calibrated predictions should give reasonable Brier score."""
        # 70% predictions with 70% actual success rate
        predictions = np.array([0.7] * 100)
        outcomes = np.array([1] * 70 + [0] * 30)
        score = brier_score(predictions, outcomes)
        # Expected: 0.7 * (1-0.7)^2 + 0.3 * (0.7)^2 = 0.7 * 0.09 + 0.3 * 0.49 = 0.21
        assert abs(score - 0.21) < 0.01


class TestLogScore:
    def test_perfect_predictions(self):
        """Perfect predictions should have log score close to 0."""
        predictions = np.array([0.999, 0.001, 0.999])
        outcomes = np.array([1, 0, 1])
        score = log_score(predictions, outcomes)
        assert score > -0.01  # Very close to 0

    def test_terrible_predictions(self):
        """Confident wrong predictions should have very negative log score."""
        predictions = np.array([0.001, 0.999])
        outcomes = np.array([1, 0])
        score = log_score(predictions, outcomes)
        assert score < -5  # Very negative

    def test_uncertain_predictions(self):
        """50% predictions should give moderate log score."""
        predictions = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1, 0, 1, 0])
        score = log_score(predictions, outcomes)
        assert abs(score - np.log(0.5)) < 0.01  # log(0.5) ≈ -0.693


class TestCalibrationCurve:
    def test_uniform_bins(self):
        """Test calibration curve with uniform bins."""
        predictions = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 0, 1, 1, 1, 1])

        curve = calibration_curve(predictions, outcomes, n_bins=5)

        assert len(curve) > 0
        for bucket in curve:
            assert 0 <= bucket.bin_start < bucket.bin_end <= 1
            assert 0 <= bucket.predicted_mean <= 1
            assert 0 <= bucket.actual_frequency <= 1
            assert bucket.count > 0

    def test_perfect_calibration(self):
        """Perfectly calibrated data should show actual ≈ predicted."""
        # Generate calibrated data
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, 1000)
        outcomes = (np.random.uniform(0, 1, 1000) < predictions).astype(int)

        curve = calibration_curve(predictions, outcomes, n_bins=10)

        # Check that predicted and actual are close
        for bucket in curve:
            if bucket.count > 10:  # Only check buckets with enough data
                assert abs(bucket.predicted_mean - bucket.actual_frequency) < 0.15


class TestCalibrationError:
    def test_ece_perfect(self):
        """ECE should be 0 for perfectly calibrated predictions."""
        # Create perfectly calibrated data
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, 1000)
        outcomes = (np.random.uniform(0, 1, 1000) < predictions).astype(int)

        errors = calibration_error(predictions, outcomes)
        assert errors["ece"] < 0.05  # Should be close to 0

    def test_ece_miscalibrated(self):
        """ECE should be high for miscalibrated predictions."""
        # All predictions are 0.8 but only 20% success
        predictions = np.array([0.8] * 100)
        outcomes = np.array([1] * 20 + [0] * 80)

        errors = calibration_error(predictions, outcomes)
        assert errors["ece"] > 0.5  # Should be high


class TestHosmerLemeshow:
    def test_well_calibrated(self):
        """Well-calibrated data should have high p-value."""
        np.random.seed(42)
        predictions = np.random.uniform(0.1, 0.9, 500)
        outcomes = (np.random.uniform(0, 1, 500) < predictions).astype(int)

        chi2, p_value = hosmer_lemeshow_test(predictions, outcomes)

        assert p_value > 0.05  # Should not reject null hypothesis

    def test_poorly_calibrated(self):
        """Poorly calibrated data should have low p-value."""
        # Predictions all 0.9 but only 10% success
        predictions = np.array([0.9] * 200)
        outcomes = np.array([1] * 20 + [0] * 180)

        chi2, p_value = hosmer_lemeshow_test(predictions, outcomes)

        assert p_value < 0.05  # Should reject null hypothesis
