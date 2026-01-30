"""
Calculate P_win from signal inputs and expected value in R-multiples.

Inputs
- buy_ratings, total_ratings: TipRanks Analysts' Ratings
- smart_score: TipRanks Smart Score (0-10)
- net_options_sentiment: Net Options Sentiment (0-100)
- net_social_sentiment: Net Social Sentiment (0-100)
- upside_breakout: Upside Breakout (0-100)

EV Formula: EV = (p_win × win_r) + ((1 - p_win) × loss_r)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Default weights from README
DEFAULT_WEIGHTS = {
    "analysts_ratings": 0.25,
    "smart_score": 0.15,
    "net_options_sentiment": 0.20,
    "net_social_sentiment": 0.20,
    "upside_breakout": 0.20,
}


def calculate_p_win(
    buy_ratings: int,
    total_ratings: int,
    smart_score: float,
    net_options_sentiment: float,
    net_social_sentiment: float,
    upside_breakout: float,
    weights: dict | None = None,
) -> float:
    """
    Calculate P_win from signal inputs using the blended model from README.

    Args:
        buy_ratings: Number of buy ratings from analysts
        total_ratings: Total number of analyst ratings
        smart_score: TipRanks Smart Score (0-10)
        net_options_sentiment: Net Options Sentiment (0-100)
        net_social_sentiment: Net Social Sentiment (0-100)
        upside_breakout: Upside Breakout score (0-100)
        weights: Optional dict with custom weights (default uses README weights)

    Returns:
        P_win probability (0.0 to 1.0)
    """
    # Use default weights if not provided
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Calculate individual deltas
    # Analysts' Ratings: (Buy Proportion × (Total Ratings / 20)) × 30
    if total_ratings > 0:
        buy_proportion = buy_ratings / total_ratings
        analysts_delta = (buy_proportion * (total_ratings / 20)) * 30
    else:
        # No ratings available, so no adjustment from analysts' ratings
        analysts_delta = 0.0
    analysts_delta = np.clip(analysts_delta, -30, 30)  # Max ±30%

    # Smart Score: ((Score − 5) / 5) × 20
    smart_delta = ((smart_score - 5) / 5) * 20
    smart_delta = np.clip(smart_delta, -20, 20)  # Max ±20%

    # Net Options Sentiment: ((Score − 50) / 50) × 20
    options_delta = ((net_options_sentiment - 50) / 50) * 20
    options_delta = np.clip(options_delta, -20, 20)  # Max ±20%

    # Net Social Sentiment: ((Score − 50) / 50) × 20
    social_delta = ((net_social_sentiment - 50) / 50) * 20
    social_delta = np.clip(social_delta, -20, 20)  # Max ±20%

    # Upside Breakout: ((Score − 50) / 50) × 20
    breakout_delta = ((upside_breakout - 50) / 50) * 20
    breakout_delta = np.clip(breakout_delta, -20, 20)  # Max ±20%

    # Weighted total delta
    total_delta = (
        analysts_delta * weights["analysts_ratings"]
        + smart_delta * weights["smart_score"]
        + options_delta * weights["net_options_sentiment"]
        + social_delta * weights["net_social_sentiment"]
        + breakout_delta * weights["upside_breakout"]
    )

    # Final P_win using sigmoid bounding: P_win = 1 / (1 + e^(-z))
    # where z = total_delta / 100
    z = total_delta / 100
    p_win = 1 / (1 + np.exp(-z))

    return float(p_win)


def calculate_ev(p_win: float, win_r: float, loss_r: float) -> float:
    """
    Calculate expected value (EV) in R-multiples.

    Args:
        p_win: Win probability (0.0 to 1.0)
        win_r: Average R-multiple on wins
        loss_r: Average R-multiple on losses (typically negative)

    Returns:
        Expected value in R-multiples
    """
    ev = (p_win * win_r) + ((1 - p_win) * loss_r)
    return ev


def calculate_ev_from_csv(
    csv_path: str | Path, output_path: str | Path | None = None
) -> pd.DataFrame:
    """
    Calculate P_win from signal inputs, then calculate EV for each row in a CSV file.

    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to save results CSV (default: None, returns DataFrame only)

    Returns:
        DataFrame with original columns plus 'p_win', 'ev', and 'recommendation' columns
    """
    # Column names (hardcoded)
    buy_ratings_col = "buy_ratings"
    total_ratings_col = "total_ratings"
    smart_score_col = "smart_score"
    net_options_sentiment_col = "net_options_sentiment"
    net_social_sentiment_col = "net_social_sentiment"
    upside_breakout_col = "upside_breakout"
    win_r_col = "win_r"
    loss_r_col = "loss_r"

    # Read CSV
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = [
        buy_ratings_col,
        total_ratings_col,
        smart_score_col,
        net_options_sentiment_col,
        net_social_sentiment_col,
        upside_breakout_col,
        win_r_col,
        loss_r_col,
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV")

    # Handle empty DataFrame
    if len(df) == 0:
        df["p_win"] = pd.Series(dtype=float)
        df["ev"] = pd.Series(dtype=float)
        df["recommendation"] = pd.Series(dtype=str)
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        return df

    # Calculate P_win for each row
    df["p_win"] = df.apply(
        lambda row: calculate_p_win(
            buy_ratings=int(row[buy_ratings_col]),
            total_ratings=int(row[total_ratings_col]),
            smart_score=float(row[smart_score_col]),
            net_options_sentiment=float(row[net_options_sentiment_col]),
            net_social_sentiment=float(row[net_social_sentiment_col]),
            upside_breakout=float(row[upside_breakout_col]),
        ),
        axis=1,
    )

    # Calculate EV for each row
    df["ev"] = df.apply(
        lambda row: calculate_ev(
            p_win=float(row["p_win"]), win_r=float(row[win_r_col]), loss_r=float(row[loss_r_col])
        ),
        axis=1,
    )

    # Add recommendation based on EV threshold (0.3-0.5R buffer)
    df["recommendation"] = df["ev"].apply(lambda x: "take_trade" if x >= 0.3 else "skip_trade")

    # Save to file if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return df


def main():
    """
    Command-line interface for calculating expected value from CSV.
    """
    parser = argparse.ArgumentParser(
        description="Calculate P_win from signal inputs and expected value in R-multiples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input CSV file should contain the following columns.
  - buy_ratings: Number of buy ratings [required]
  - total_ratings: Total number of analyst ratings [required]
  - smart_score: TipRanks Smart Score (0-10) [required]
  - net_options_sentiment: Net Options Sentiment (0-100) [required]
  - net_social_sentiment: Net Social Sentiment (0-100) [required]
  - upside_breakout: Upside Breakout score (0-100) [required]
  - win_r: Average R-multiple on wins [required]
  - loss_r: Average R-multiple on losses (typically negative) [required]

Example:
  python ev.py input.csv -o results.csv
        """,
    )

    parser.add_argument(
        "input_csv", type=str, help="Path to input CSV file with trading signal data"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        dest="output_csv",
        help="Path to output CSV file (optional, if not provided results are only printed)",
    )

    args = parser.parse_args()

    try:
        df = calculate_ev_from_csv(args.input_csv, output_path=args.output_csv)

        print("\nExpected Value Results:")
        print("=" * 80)
        print(df.to_string(index=False))
        print("\n" + "=" * 80)
        print(f"\nTotal trades analyzed: {len(df)}")
        print(f"Trades with EV >= 0.3R: {len(df[df['ev'] >= 0.3])}")
        print(f"Average EV: {df['ev'].mean():.3f}R")
        print(f"Average P_win: {df['p_win'].mean():.3f}")

    except Exception as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
