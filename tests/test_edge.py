"""
Tests for edge.py - P_win and EV calculations
"""

import os
import tempfile

import pandas as pd
import pytest

from trading.edge import calculate_ev, calculate_ev_from_csv, calculate_p_win


class TestCalculatePWin:
    """Tests for calculate_p_win function."""

    def test_basic_calculation(self):
        """Test basic P_win calculation with example from README."""
        # Example: 15/16 Buy, 16 ratings, Smart Score 8, high sentiment scores
        p_win = calculate_p_win(
            buy_ratings=15,
            total_ratings=16,
            smart_score=8.0,
            net_options_sentiment=89,
            net_social_sentiment=82,
            upside_breakout=89,
        )

        # P_win should be between 0 and 1
        assert 0.0 <= p_win <= 1.0
        # With strong positive signals, P_win should be > 0.5
        assert p_win > 0.5

    def test_neutral_signals(self):
        """Test with neutral signals (should be close to 0.5)."""
        p_win = calculate_p_win(
            buy_ratings=8,
            total_ratings=16,  # 50% buy
            smart_score=5.0,  # Neutral
            net_options_sentiment=50,  # Neutral
            net_social_sentiment=50,  # Neutral
            upside_breakout=50,  # Neutral
        )

        # Should be close to 0.5 (neutral prior)
        assert abs(p_win - 0.5) < 0.1

    def test_negative_signals(self):
        """Test with negative signals (should be < 0.5)."""
        p_win = calculate_p_win(
            buy_ratings=2,
            total_ratings=16,  # Low buy proportion
            smart_score=2.0,  # Low score
            net_options_sentiment=20,  # Low sentiment
            net_social_sentiment=20,  # Low sentiment
            upside_breakout=20,  # Low breakout
        )

        # Should be less than 0.5
        assert p_win < 0.5

    def test_zero_ratings(self):
        """Test with zero total ratings."""
        p_win = calculate_p_win(
            buy_ratings=0,
            total_ratings=0,
            smart_score=5.0,
            net_options_sentiment=50,
            net_social_sentiment=50,
            upside_breakout=50,
        )

        # Should still return valid probability
        assert 0.0 <= p_win <= 1.0
        # Analysts delta should be 0, so should be close to neutral
        assert abs(p_win - 0.5) < 0.2

    def test_boundary_values(self):
        """Test with boundary values."""
        # Maximum positive signals
        p_win_max = calculate_p_win(
            buy_ratings=20,
            total_ratings=20,  # 100% buy
            smart_score=10.0,  # Max score
            net_options_sentiment=100,  # Max
            net_social_sentiment=100,  # Max
            upside_breakout=100,  # Max
        )
        assert p_win_max > 0.5
        assert p_win_max < 1.0  # Sigmoid keeps it below 1.0

        # Maximum negative signals
        p_win_min = calculate_p_win(
            buy_ratings=0,
            total_ratings=20,  # 0% buy
            smart_score=0.0,  # Min score
            net_options_sentiment=0,  # Min
            net_social_sentiment=0,  # Min
            upside_breakout=0,  # Min
        )
        assert p_win_min < 0.5
        assert p_win_min > 0.0  # Sigmoid keeps it above 0.0

    def test_custom_weights(self):
        """Test with custom weights."""
        custom_weights = {
            "analysts_ratings": 0.5,
            "smart_score": 0.1,
            "net_options_sentiment": 0.1,
            "net_social_sentiment": 0.1,
            "upside_breakout": 0.2,
        }

        p_win = calculate_p_win(
            buy_ratings=15,
            total_ratings=16,
            smart_score=8.0,
            net_options_sentiment=89,
            net_social_sentiment=82,
            upside_breakout=89,
            weights=custom_weights,
        )

        assert 0.0 <= p_win <= 1.0

    def test_delta_capping(self):
        """Test that deltas are properly capped."""
        # Test analysts delta capping (should be max ±30%)
        # With 20/20 buy and 20 ratings: (1.0 * (20/20)) * 30 = 30
        p_win = calculate_p_win(
            buy_ratings=20,
            total_ratings=20,
            smart_score=5.0,
            net_options_sentiment=50,
            net_social_sentiment=50,
            upside_breakout=50,
        )
        # Should be higher than neutral but not extreme due to sigmoid
        assert p_win > 0.5


class TestCalculateEV:
    """Tests for calculate_ev function."""

    def test_basic_calculation(self):
        """Test basic EV calculation."""
        # Example from README: P_win=0.48, win_r=2.25, loss_r=-1.05
        # EV = (0.48 × 2.25) + (0.52 × -1.05) = 1.08 - 0.546 = 0.534
        ev = calculate_ev(p_win=0.48, win_r=2.25, loss_r=-1.05)

        assert abs(ev - 0.534) < 0.01

    def test_positive_ev(self):
        """Test with positive EV."""
        ev = calculate_ev(p_win=0.6, win_r=2.0, loss_r=-1.0)
        # EV = (0.6 × 2.0) + (0.4 × -1.0) = 1.2 - 0.4 = 0.8
        assert ev > 0

    def test_negative_ev(self):
        """Test with negative EV."""
        ev = calculate_ev(p_win=0.3, win_r=1.5, loss_r=-1.0)
        # EV = (0.3 × 1.5) + (0.7 × -1.0) = 0.45 - 0.7 = -0.25
        assert ev < 0

    def test_zero_ev(self):
        """Test with zero EV."""
        # When win_r * p_win = loss_r * (1 - p_win)
        # 2.0 * 0.5 = 1.0, and -1.0 * 0.5 = -0.5, so need different values
        # Let's use: p_win=0.5, win_r=2.0, loss_r=-2.0
        # EV = (0.5 × 2.0) + (0.5 × -2.0) = 1.0 - 1.0 = 0.0
        ev = calculate_ev(p_win=0.5, win_r=2.0, loss_r=-2.0)
        assert abs(ev) < 0.001

    def test_extreme_values(self):
        """Test with extreme probabilities."""
        # Very high P_win
        ev_high = calculate_ev(p_win=0.95, win_r=2.0, loss_r=-1.0)
        assert ev_high > 0

        # Very low P_win
        ev_low = calculate_ev(p_win=0.05, win_r=2.0, loss_r=-1.0)
        assert ev_low < 0


class TestCalculateEVFromCSV:
    """Tests for calculate_ev_from_csv function."""

    def test_basic_csv_processing(self):
        """Test processing a valid CSV file."""
        # Create temporary CSV
        csv_data = """buy_ratings,total_ratings,smart_score,net_options_sentiment,net_social_sentiment,upside_breakout,win_r,loss_r
15,16,8.0,89,82,89,2.25,-1.05
12,15,7.5,75,70,80,2.0,-1.0"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            df = calculate_ev_from_csv(temp_path)

            # Check that required columns are present
            assert "p_win" in df.columns
            assert "ev" in df.columns
            assert "recommendation" in df.columns

            # Check that we have 2 rows
            assert len(df) == 2

            # Check that p_win is between 0 and 1
            assert all((df["p_win"] >= 0) & (df["p_win"] <= 1))

            # Check recommendations
            assert all(df["recommendation"].isin(["take_trade", "skip_trade"]))

        finally:
            os.unlink(temp_path)

    def test_missing_column(self):
        """Test that missing required column raises error."""
        csv_data = """buy_ratings,total_ratings,smart_score,net_options_sentiment,net_social_sentiment,upside_breakout,win_r
15,16,8.0,89,82,89,2.25"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Required column 'loss_r' not found"):
                calculate_ev_from_csv(temp_path)
        finally:
            os.unlink(temp_path)

    def test_output_file(self):
        """Test that output file is created when specified."""
        csv_data = """buy_ratings,total_ratings,smart_score,net_options_sentiment,net_social_sentiment,upside_breakout,win_r,loss_r
15,16,8.0,89,82,89,2.25,-1.05"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as input_file:
            input_file.write(csv_data)
            input_path = input_file.name

        output_path = input_path.replace(".csv", "_output.csv")

        try:
            calculate_ev_from_csv(input_path, output_path=output_path)

            # Check that output file exists
            assert os.path.exists(output_path)

            # Check that output file can be read back
            df_read = pd.read_csv(output_path)
            assert len(df_read) == 1
            assert "ev" in df_read.columns

        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_recommendation_threshold(self):
        """Test that recommendation threshold works correctly."""
        csv_data = """buy_ratings,total_ratings,smart_score,net_options_sentiment,net_social_sentiment,upside_breakout,win_r,loss_r
15,16,8.0,89,82,89,2.25,-1.05
2,16,2.0,20,20,20,1.5,-1.0"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            df = calculate_ev_from_csv(temp_path)

            # First row should have high EV (take_trade)
            assert df.iloc[0]["recommendation"] == "take_trade"
            assert df.iloc[0]["ev"] >= 0.3

            # Second row should have low EV (skip_trade)
            assert df.iloc[1]["recommendation"] == "skip_trade"
            assert df.iloc[1]["ev"] < 0.3

        finally:
            os.unlink(temp_path)

    def test_empty_csv(self):
        """Test with empty CSV (only headers)."""
        csv_data = """buy_ratings,total_ratings,smart_score,net_options_sentiment,net_social_sentiment,upside_breakout,win_r,loss_r"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            df = calculate_ev_from_csv(temp_path)
            # Should return empty dataframe with required columns
            assert len(df) == 0
            assert "p_win" in df.columns
            assert "ev" in df.columns

        finally:
            os.unlink(temp_path)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_pipeline(self):
        """Test full pipeline from signals to EV."""
        # Calculate P_win
        p_win = calculate_p_win(
            buy_ratings=15,
            total_ratings=16,
            smart_score=8.0,
            net_options_sentiment=89,
            net_social_sentiment=82,
            upside_breakout=89,
        )

        # Calculate EV
        ev = calculate_ev(p_win=p_win, win_r=2.25, loss_r=-1.05)

        # Should be positive EV
        assert ev > 0
        assert p_win > 0.5  # Strong signals should give high P_win

    def test_csv_with_realistic_data(self):
        """Test CSV processing with realistic trading data."""
        csv_data = """buy_ratings,total_ratings,smart_score,net_options_sentiment,net_social_sentiment,upside_breakout,win_r,loss_r
15,16,8.0,89,82,89,2.25,-1.05
12,15,7.5,75,70,80,2.0,-1.0
8,20,6.0,45,50,55,2.8,-1.1
18,20,9.0,95,90,95,1.7,-1.0
14,16,8.5,85,78,88,2.625,-1.05"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_path = f.name

        try:
            df = calculate_ev_from_csv(temp_path)

            # All rows should have valid calculations
            assert len(df) == 5
            assert all((df["p_win"] >= 0) & (df["p_win"] <= 1))
            assert all(df["ev"].notna())

            # All should have valid recommendations
            assert all(df["recommendation"].isin(["take_trade", "skip_trade"]))
            # All rows have positive signals, so all should be take_trade
            assert all(df["recommendation"] == "take_trade")
            assert all(df["ev"] >= 0.3)

        finally:
            os.unlink(temp_path)
