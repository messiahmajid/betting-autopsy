# Betting Market Autopsy

**Do prediction markets actually predict?** This project rigorously analyzes 600+ resolved markets across Metaculus and Manifold to answer that question with data.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## The Big Question

Prediction markets claim to aggregate information efficiently. When a market prices an event at 70%, it's making a testable claim: *this event should happen roughly seven times out of ten.*

But is that actually true? This project finds out.

## Key Findings

After analyzing **271 resolved markets** with final price data:

| Platform | Brier Score | Calibrated? | Markets |
|----------|-------------|-------------|---------|
| **Manifold** | 0.123 | Yes (p=0.17) | 68 |
| **Metaculus** | 0.213 | Yes (p=0.35) | 203 |

**What this means:**
- **Brier Score**: Lower is better. Random guessing scores 0.25; these markets beat that significantly.
- **Calibration**: Both platforms pass the Hosmer-Lemeshow statistical test, meaning their probabilities are trustworthy.
- **Favorite-Longshot Bias**: Confirmed. Low-probability events ("longshots") are slightly overpriced by ~3.5%.

**The verdict:** Prediction markets are remarkably well-calibrated, but subtle biases exist. The edge is too small to exploit profitably—confirming what economists have long suspected about efficient information aggregation.

---

## What's Inside

### 1. Calibration Analysis
Measures whether stated probabilities match observed frequencies.

- **Brier Score**: Mean squared error between predictions and outcomes
- **Expected Calibration Error (ECE)**: Average gap between predicted and actual frequencies
- **Hosmer-Lemeshow Test**: Statistical test for calibration goodness-of-fit
- **Calibration Curves**: Visual comparison of predicted vs. actual probabilities

### 2. Favorite-Longshot Bias Detection
Tests whether markets systematically misprice extreme probabilities.

- Longshots (<20% probability) tend to be overpriced
- Favorites (>80% probability) are fairly priced
- Classic finding from horse racing replicated in prediction markets

### 3. Kelly Criterion Portfolio
Optimal bet sizing given edge estimates.

- Full Kelly and fractional Kelly implementations
- Backtesting framework on historical data
- Risk metrics: Sharpe ratio, max drawdown, win rate

### 4. Arbitrage Scanner
Finds mispricings across markets.

- Cross-platform price comparison
- Complement violations (when P(A) + P(not A) ≠ 100%)
- Historical arbitrage opportunity analysis

---

## Live Dashboard

The project includes an interactive web dashboard that visualizes all analyses:

- Calibration curves with confidence intervals
- Platform comparison tables
- Bias visualization by probability bucket
- Browseable prediction history with accuracy metrics

---

<img width="1176" height="760" alt="dashboard" src="https://github.com/user-attachments/assets/08e80b6d-e67c-4eb5-a585-e0c0d24f8136" />


<img width="1176" height="760" alt="predictions" src="https://github.com/user-attachments/assets/f6e30e1b-1f69-409c-b7d9-63ca2deeba06" />


<img width="1176" height="644" alt="image" src="https://github.com/user-attachments/assets/b1cf2093-4892-43b4-8927-02d9df914d91" />





## Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/betting-autopsy.git
cd betting-autopsy

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Initialize and Run

```bash
# 1. Initialize the database
python -c "from src.models.database import Database; db = Database(); db.init_schema(); db.close()"

# 2. Ingest data from prediction markets
python -m src.ingest.metaculus    # Fetches ~300 resolved questions
python -m src.ingest.manifold     # Fetches ~100 resolved markets

# 3. Start the dashboard
uvicorn src.api.main:app --reload --port 8000

# 4. Open http://localhost:8000 in your browser
```

### Using the Makefile

```bash
make init-db      # Initialize database
make ingest-all   # Ingest from all platforms
make serve        # Start the dashboard
make test         # Run test suite
```

---

## Project Structure

```
betting-autopsy/
├── src/
│   ├── ingest/              # Data ingestion from platforms
│   │   ├── polymarket.py    # Polymarket API client
│   │   ├── metaculus.py     # Metaculus API client
│   │   └── manifold.py      # Manifold API client
│   │
│   ├── analysis/            # Core analysis modules
│   │   ├── calibration.py   # Brier scores, ECE, H-L test
│   │   ├── arbitrage.py     # Cross-market arbitrage detection
│   │   └── kelly.py         # Kelly criterion, backtesting
│   │
│   ├── models/
│   │   ├── schemas.py       # Pydantic data models
│   │   └── database.py      # DuckDB interface
│   │
│   ├── viz/                 # Plotly visualizations
│   │
│   └── api/
│       ├── main.py          # FastAPI application
│       ├── routes.py        # API endpoints
│       └── templates/       # Dashboard HTML
│
├── tests/                   # Comprehensive test suite
├── data/                    # DuckDB database (gitignored)
└── notebooks/               # Marimo notebooks for exploration
```

---

## API Reference

The dashboard is powered by a REST API:

| Endpoint | Description |
|----------|-------------|
| `GET /api/stats` | Database statistics (market counts) |
| `GET /api/calibration/summary` | Overall calibration metrics |
| `GET /api/calibration/comparison` | Platform-by-platform comparison |
| `GET /api/calibration/curve-data` | Calibration curve data points |
| `GET /api/calibration/bias` | Favorite-longshot bias analysis |
| `GET /api/predictions` | Resolved markets with accuracy data |
| `GET /api/markets` | List all markets |
| `GET /api/markets/{id}` | Single market with price history |

---

## Understanding the Metrics

### Brier Score
The mean squared error between predicted probabilities and binary outcomes.

```
Brier = (1/n) × Σ(prediction - outcome)²
```

- **0.0** = Perfect predictions
- **0.25** = Random guessing (always predicting 50%)
- **< 0.20** = Good forecasting

### Expected Calibration Error (ECE)
Average absolute difference between predicted and actual frequencies across probability bins.

- **0.0** = Perfectly calibrated
- **> 0.10** = Noticeable miscalibration

### Hosmer-Lemeshow Test
Statistical test comparing observed vs. expected frequencies.

- **p > 0.05** = Calibration is statistically acceptable
- **p < 0.05** = Significant evidence of miscalibration

### Kelly Criterion
Optimal bet sizing formula:

```
f* = (bp - q) / b

where:
  f* = fraction of bankroll to bet
  b  = net odds (decimal odds - 1)
  p  = probability of winning
  q  = probability of losing (1 - p)
```

---

## Data Sources

| Platform | Type | Data Available |
|----------|------|----------------|
| **Metaculus** | Forecasting platform | Community predictions on resolved questions |
| **Manifold** | Play-money market | Final prices and resolutions |

Note: Polymarket's API doesn't retain historical price data for closed markets, so calibration analysis is limited to platforms with prediction histories.

---

## Tech Stack

- **Python 3.11+** — Modern Python with type hints
- **Polars** — Fast DataFrame library (10x faster than pandas)
- **DuckDB** — Embedded analytical database
- **FastAPI** — High-performance async web framework
- **Plotly** — Interactive visualizations
- **Pydantic** — Data validation and serialization

---

## Development

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src

# Format code
ruff format .

# Lint
ruff check .
```

---

## Why This Matters

Prediction markets are increasingly used for:
- **Decision-making**: Companies use internal markets for forecasting
- **News**: Media cite prediction market odds for elections, events
- **Research**: Academics study them as information aggregation mechanisms

Understanding their accuracy—and limitations—is crucial. This project provides:
1. **Empirical evidence** that markets are well-calibrated
2. **Quantified biases** that exist despite overall accuracy
3. **Tools** for ongoing analysis as new data comes in

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Data sourced from public APIs:
- [Metaculus](https://www.metaculus.com/)
- [Manifold Markets](https://manifold.markets/)
