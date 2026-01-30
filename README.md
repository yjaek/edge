# Edge

**Edge** is a probabilistic trading framework for medium-term directional trading (~6 months horizon) in stocks and long call options. It estimates win probability (**P_win**) from blended signals, computes realistic **expected value (EV / expectancy)**, sizes positions dynamically, and builds diversified portfolios with active tilt.

Core philosophy (from *Trading in the Zone* and Van K. Tharp):
- Trading is a game of large-sample probabilities — focus on process and positive expectancy, not being right on every trade.
- All estimates are forward-looking and subject to change; past signal performance does not guarantee future results.

## 1. P_win – Estimated Win Probability

**P_win** represents the estimated directional probability of a positive return (underlying stock price higher at the end of the horizon than at the start, after costs/slippage) over a medium-term horizon, approximately 6 months.

It is **not** a guarantee but a forward-looking edge estimate to inform trade decisions, position sizing, and expectancy calculations.

P_win is calculated using a blended, semi-quantitative model that synthesizes professional consensus from TipRanks and institutional/social/momentum signals from Prospero.ai, then applies conservative probabilistic bounding. The goal is to quantify conviction without over-optimism.

### Key Principles
- **Neutral prior**: Start at 50%. The 50% neutral prior assumes no directional bias; in reality, equities have a long-term positive drift (~7–10% annualized), so the true unconditional probability of a positive 6-month return is often slightly higher than 50% (~52–55% historically). The 50% starting point keeps the model conservative and focused on relative edge.
- **Deltas**: Adjustments based on signal strength, capped to avoid dominance by noisy inputs.
- **Non-linear bounding**: Logistic sigmoid function keeps P_win realistic (between 0% and 100%) with diminishing returns for extreme signals.
- **Integration with expectancy**: P_win feeds into full trade evaluation: Expectancy (in R-multiples) ≈ (P_win × Reward:Risk) − (1 − P_win). Note: P_win estimates the probability for the underlying stock. When trading options, incorporate premium costs and payoff asymmetry into the expectancy calculation rather than adjusting P_win downward (see separate guidance for options).

### Inputs
- **TipRanks Analysts' Ratings**: Consensus rating and price targets (e.g., Strong Buy with volume of ratings).
- **TipRanks Smart Score**: Derived score (0–10) incorporating analyst accuracy, hedge fund activity, etc.
- **Prospero.ai Signals** (app-based, manual pull required as no public API):
  - Net Options Sentiment (0–100): Institutional options flow/pricing dynamics.
  - Net Social Sentiment (0–100): Crowd sentiment from Reddit, X, etc.
  - Upside Breakout (0–100): Potential for significant price gains/momentum.

### Calculation
1. **Compute Individual Deltas** (adjustments to the 50% prior):

   | Signal                     | Formula                                              | Max Delta | Example (value → delta)               |
   |----------------------------|------------------------------------------------------|-----------|----------------------------------------|
   | Analysts' Ratings          | (Buy Proportion × (Total Ratings / 20)) × 30         | ±30%      | 15/16 Buy, 16 ratings → +22.5%         |
   | Smart Score                | ((Score − 5) / 5) × 20                               | ±20%      | 8/10 → +12%                            |
   | Net Options Sentiment      | ((Score − 50) / 50) × 20                             | ±20%      | 89 → +15.6%                            |
   | Net Social Sentiment       | ((Score − 50) / 50) × 20                             | ±20%      | 82 → +12.8%                            |
   | Upside Breakout            | ((Score − 50) / 50) × 20                             | ±20%      | 89 → +15.6%                            |

2. **Weighted Total Delta**
   Apply weights (sum to 100%):
   - Analysts' Ratings: 25%
   - Smart Score: 15%
   - Net Options Sentiment: 20%
   - Net Social Sentiment: 20%
   - Upside Breakout: 20%

   Total Delta = ∑ (each delta × its weight)

3. **Final P_win (Sigmoid Bounding)**
   z = total_delta / 100
   P_win = 1 / (1 + e^(-z))
   (Logistic sigmoid function; ensures 0% < P_win < 100%, with non-linear scaling near 50%.)

4. **Optional Confidence Interval**
   Estimate ~±5–7% around P_win, factoring in market risks (via simple simulation or qualitative judgment).

### Customization Notes
- Weights can be adjusted based on signal reliability (e.g., boost Prospero if it correlates well with outcomes).
- Scaling factors (e.g., ×20 cap) are conservative; increase for more aggressive models.
- Horizon: Tuned for ~6 months; shorten for day trading by emphasizing short-term signals.

## 2. EV – Expected Value (Expectancy)

EV measures the average net profit/loss per trade (in R units) over many repetitions. Positive EV = mathematical edge.

**Standard Formula** (Van Tharp – R units):
EV = (P_win × Avg R-multiple on wins) + ((1 - P_win) × Avg R-multiple on losses)

- Avg R-multiple on wins = Planned R:R × Capture Rate (60–85%)
- Avg R-multiple on losses ≈ –1.0 to –1.1

**Planning Approximation** (conservative):
EV ≈ P_win × (Planned R:R × Capture Rate) − (1 − P_win) × 1

**With Costs/Slippage**: Deduct round-trip costs from Avg Win or add to Avg Loss.

**Example**:
P_win = 0.48, Planned R:R = 3.0, Capture = 0.75 → Avg Win = 2.25R
EV = (0.48 × 2.25) + (0.52 × –1.05) = **+0.534R**

**Rules**: Only take trades where EV > 0.3–0.5R (buffers variance).

## Next Steps
- Backtest signal weights and capture rates
- Integrate into portfolio optimization (HRP with EV tilt)
- Add dynamic sizing: risk % = base (0.5–1.5%) × mild EV scaling

**Disclaimer**
This is **not financial advice**. Trading involves substantial risk of loss. All estimates are forward-looking and for educational/personal use only. Past/simulated performance does not guarantee future results. Use at your own risk.

Focus on process — let the math compound. 📈

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/edge.git

# Install dependencies
pip install -r requirements.txt

# Set up pre-commit hooks (optional but recommended)
pre-commit install
```

## Development Setup

### Pre-commit Hooks

This project uses pre-commit hooks for code formatting and linting:
- **Black** - Code formatting (100 char line length)
- **Ruff** - Fast Python linter

Hooks run automatically on `git commit`. To run manually:
```bash
pre-commit run --all-files
```

### Running Tests

```bash
pytest tests/
```

### Usage

Calculate expected value from CSV:
```bash
python trading/edge.py trading/sample_input.csv -o results.csv
```

CSV should contain columns: `buy_ratings`, `total_ratings`, `smart_score`, `net_options_sentiment`, `net_social_sentiment`, `upside_breakout`, `win_r`, `loss_r`
