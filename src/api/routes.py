"""API routes for betting market analysis."""

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from src.models.database import Database
from src.models.schemas import Platform
from src.analysis.calibration import CalibrationAnalyzer
from src.analysis.arbitrage import ArbitrageScanner
from src.analysis.kelly import BacktestEngine
from src.viz.calibration import plot_calibration_curve, plot_calibration_comparison, plot_favorite_longshot_bias
from src.viz.portfolio import plot_portfolio_growth
from src.viz.arbitrage import plot_arbitrage_opportunities

router = APIRouter()


def get_db() -> Database:
    """Get database connection."""
    return Database()


@router.get("/stats")
async def get_stats():
    """Get database statistics."""
    db = get_db()
    try:
        stats = db.get_stats()
        return stats
    finally:
        db.close()


@router.get("/calibration/summary")
async def calibration_summary(platform: Optional[str] = None):
    """Get calibration summary for a platform."""
    db = get_db()
    try:
        analyzer = CalibrationAnalyzer(db)

        platform_enum = Platform(platform) if platform else None
        result = analyzer.analyze_platform(platform_enum)

        return {
            "platform": result.platform.value if result.platform else "all",
            "n_markets": result.n_markets,
            "brier_score": round(result.brier_score, 4),
            "log_score": round(result.log_score, 4),
            "ece": round(result.ece, 4),
            "mce": round(result.mce, 4),
            "hl_pvalue": round(result.hl_pvalue, 4),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        db.close()


@router.get("/calibration/comparison")
async def calibration_comparison():
    """Compare calibration across platforms."""
    db = get_db()
    try:
        analyzer = CalibrationAnalyzer(db)
        results = analyzer.compare_platforms()

        if not results:
            return []

        return [
            {
                "name": name,
                "n_markets": result.n_markets,
                "brier_score": round(result.brier_score, 4),
                "ece": round(result.ece, 4),
                "hl_pvalue": round(result.hl_pvalue, 4),
                "calibrated": result.hl_pvalue > 0.05,
            }
            for name, result in sorted(results.items(), key=lambda x: x[1].brier_score)
        ]
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()


@router.get("/calibration/chart")
async def calibration_chart(platform: Optional[str] = None):
    """Get calibration curve chart."""
    db = get_db()
    try:
        analyzer = CalibrationAnalyzer(db)
        results = analyzer.compare_platforms()

        if not results:
            return HTMLResponse("<p>No data available</p>")

        fig = plot_calibration_comparison(results)
        chart_json = fig.to_json()

        html = f"""
        <div id="cal-plot"></div>
        <script>
            Plotly.newPlot('cal-plot', {chart_json});
        </script>
        """
        return HTMLResponse(html)
    except Exception as e:
        return HTMLResponse(f"<p>Error: {e}</p>")
    finally:
        db.close()


@router.get("/calibration/curve-data")
async def calibration_curve_data(platform: Optional[str] = None):
    """Get calibration curve data points for plotting."""
    db = get_db()
    try:
        analyzer = CalibrationAnalyzer(db)
        platform_enum = Platform(platform) if platform else None
        result = analyzer.analyze_platform(platform_enum)

        return {
            "platform": result.platform.value if result.platform else "all",
            "curve": [
                {
                    "bin_start": b.bin_start,
                    "bin_end": b.bin_end,
                    "predicted": round(b.predicted_mean, 4),
                    "actual": round(b.actual_frequency, 4),
                    "count": b.count,
                    "std_error": round(b.std_error, 4) if b.std_error else None,
                }
                for b in result.calibration_curve
            ],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        db.close()


@router.get("/calibration/bias")
async def calibration_bias(platform: Optional[str] = None):
    """Get favorite-longshot bias data."""
    db = get_db()
    try:
        analyzer = CalibrationAnalyzer(db)

        platform_enum = Platform(platform) if platform else None
        bias_df = analyzer.favorite_longshot_bias(platform_enum)

        if bias_df.is_empty():
            return {"buckets": [], "longshot_bias": 0, "favorite_bias": 0}

        buckets = bias_df.to_dicts()

        # Calculate aggregate biases
        longshots = bias_df.filter(bias_df["is_longshot"])
        favorites = bias_df.filter(bias_df["is_favorite"])

        longshot_bias = float(longshots["bias"].mean()) if not longshots.is_empty() else 0
        favorite_bias = float(favorites["bias"].mean()) if not favorites.is_empty() else 0

        return {
            "buckets": [
                {
                    "bin_start": b["bin_start"],
                    "bin_end": b["bin_end"],
                    "predicted": round(b["predicted"], 4),
                    "actual": round(b["actual"], 4),
                    "bias": round(b["bias"], 4),
                    "count": b["count"],
                    "is_longshot": b["is_longshot"],
                    "is_favorite": b["is_favorite"],
                }
                for b in buckets
            ],
            "longshot_bias": round(longshot_bias, 4),
            "favorite_bias": round(favorite_bias, 4),
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()


@router.get("/arbitrage/current")
async def current_arbitrage():
    """Get current arbitrage opportunities."""
    db = get_db()
    try:
        scanner = ArbitrageScanner(db)
        opportunities = scanner.scan_linked_markets()

        if not opportunities:
            return HTMLResponse("<p>No arbitrage opportunities found</p>")

        html = """
        <table>
            <tr>
                <th>Platforms</th>
                <th>Prob A</th>
                <th>Prob B</th>
                <th>Profit %</th>
                <th>Type</th>
            </tr>
        """

        for opp in sorted(opportunities, key=lambda x: x.profit_pct, reverse=True):
            html += f"""
            <tr>
                <td>{opp.platform_a.value} vs {opp.platform_b.value}</td>
                <td>{opp.prob_a:.1%}</td>
                <td>{opp.prob_b:.1%}</td>
                <td class="good">{opp.profit_pct:.2f}%</td>
                <td>{opp.arb_type}</td>
            </tr>
            """

        html += "</table>"
        return HTMLResponse(html)
    except Exception as e:
        return HTMLResponse(f"<p>Error: {e}</p>")
    finally:
        db.close()


@router.get("/arbitrage/stats")
async def arbitrage_stats():
    """Get historical arbitrage statistics."""
    db = get_db()
    try:
        scanner = ArbitrageScanner(db)
        stats = scanner.arbitrage_statistics()

        html = f"""
        <div class="stat-grid">
            <div class="stat">
                <div class="stat-value">{stats.get('total_opportunities', 0):,}</div>
                <div class="stat-label">Historical Opportunities</div>
            </div>
            <div class="stat">
                <div class="stat-value">{stats.get('avg_profit_pct', 0):.2f}%</div>
                <div class="stat-label">Avg Profit</div>
            </div>
            <div class="stat">
                <div class="stat-value">{stats.get('max_profit_pct', 0):.2f}%</div>
                <div class="stat-label">Max Profit</div>
            </div>
        </div>
        """

        return HTMLResponse(html)
    except Exception as e:
        return HTMLResponse(f"<p>Error: {e}</p>")
    finally:
        db.close()


@router.get("/portfolio/backtest")
async def portfolio_backtest(
    edge: float = Query(0.05, description="Assumed edge over market"),
    kelly_fraction: float = Query(0.5, description="Kelly fraction"),
):
    """Run and display portfolio backtest."""
    db = get_db()
    try:
        engine = BacktestEngine(db)
        result = engine.run_backtest(
            edge_estimate=edge,
            kelly_fraction=kelly_fraction,
        )

        fig = plot_portfolio_growth(result)
        chart_json = fig.to_json()

        html = f"""
        <div class="stat-grid" style="margin-bottom: 20px;">
            <div class="stat">
                <div class="stat-value">{result.total_return:.1f}%</div>
                <div class="stat-label">Total Return</div>
            </div>
            <div class="stat">
                <div class="stat-value">{result.sharpe_ratio:.2f}</div>
                <div class="stat-label">Sharpe Ratio</div>
            </div>
            <div class="stat">
                <div class="stat-value">{result.max_drawdown:.1f}%</div>
                <div class="stat-label">Max Drawdown</div>
            </div>
            <div class="stat">
                <div class="stat-value">{result.win_rate:.1f}%</div>
                <div class="stat-label">Win Rate</div>
            </div>
        </div>
        <div id="portfolio-plot"></div>
        <script>
            Plotly.newPlot('portfolio-plot', {chart_json});
        </script>
        """
        return HTMLResponse(html)
    except ValueError as e:
        return HTMLResponse(f"<p>No data available for backtest. Run data ingestion first. ({e})</p>")
    except Exception as e:
        return HTMLResponse(f"<p>Error: {e}</p>")
    finally:
        db.close()


@router.get("/markets")
async def list_markets(
    platform: Optional[str] = None,
    resolved: bool = False,
    limit: int = Query(100, le=1000),
):
    """List markets with optional filters."""
    db = get_db()
    try:
        platform_enum = Platform(platform) if platform else None
        markets = db.get_markets(platform=platform_enum, resolved_only=resolved, limit=limit)

        return [
            {
                "id": m.id,
                "platform": m.platform.value,
                "title": m.title,
                "category": m.category,
                "resolution": m.resolution,
                "resolved_at": m.resolved_at.isoformat() if m.resolved_at else None,
            }
            for m in markets
        ]
    finally:
        db.close()


@router.get("/predictions")
async def list_predictions(
    platform: Optional[str] = None,
    limit: int = Query(50, le=200),
    sort: str = Query("recent", description="Sort by: recent, accurate, inaccurate"),
):
    """List resolved markets with their final predictions and accuracy.

    Returns markets with:
    - final_probability: the market's final prediction before resolution
    - resolution: actual outcome (1.0 = yes, 0.0 = no)
    - error: absolute difference between prediction and outcome
    - correct: whether prediction was on the right side (>0.5 predicted YES and YES happened)
    """
    db = get_db()
    try:
        # Get resolved markets with their final price snapshot
        order_clause = "m.resolved_at DESC NULLS LAST"
        if sort == "accurate":
            order_clause = "ABS(p.probability - m.resolution) ASC NULLS LAST"
        elif sort == "inaccurate":
            order_clause = "ABS(p.probability - m.resolution) DESC NULLS LAST"

        platform_filter = ""
        if platform:
            platform_filter = f"AND m.platform = '{platform}'"

        query = f"""
            WITH latest_prices AS (
                SELECT market_id, probability,
                       ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC) as rn
                FROM prices
            )
            SELECT
                m.id,
                m.platform,
                m.title,
                m.category,
                m.resolution,
                m.resolved_at,
                p.probability as final_probability
            FROM markets m
            LEFT JOIN latest_prices p ON m.id = p.market_id AND p.rn = 1
            WHERE m.resolution IS NOT NULL
            {platform_filter}
            ORDER BY {order_clause}
            LIMIT {limit}
        """

        result = db.conn.execute(query).fetchall()

        predictions = []
        for row in result:
            market_id, plat, title, category, resolution, resolved_at, final_prob = row

            # Calculate accuracy metrics
            error = None
            correct = None
            if final_prob is not None and resolution is not None:
                error = abs(final_prob - resolution)
                # Correct if prediction > 0.5 and resolution = 1, or prediction < 0.5 and resolution = 0
                if final_prob > 0.5:
                    correct = resolution == 1.0
                elif final_prob < 0.5:
                    correct = resolution == 0.0
                else:
                    correct = True  # 50/50 is never wrong per se

            predictions.append({
                "id": market_id,
                "platform": plat,
                "title": title,
                "category": category,
                "resolution": resolution,
                "resolved_at": resolved_at.isoformat() if resolved_at else None,
                "final_probability": round(final_prob, 4) if final_prob is not None else None,
                "error": round(error, 4) if error is not None else None,
                "correct": correct,
            })

        return predictions
    finally:
        db.close()


@router.get("/markets/{market_id}")
async def get_market(market_id: str):
    """Get a specific market with price history."""
    db = get_db()
    try:
        market = db.get_market(market_id)
        if not market:
            raise HTTPException(status_code=404, detail="Market not found")

        prices = db.get_prices(market_id)

        return {
            "market": {
                "id": market.id,
                "platform": market.platform.value,
                "title": market.title,
                "description": market.description,
                "category": market.category,
                "resolution": market.resolution,
                "created_at": market.created_at.isoformat() if market.created_at else None,
                "close_at": market.close_at.isoformat() if market.close_at else None,
                "resolved_at": market.resolved_at.isoformat() if market.resolved_at else None,
            },
            "prices": prices.to_dicts() if not prices.is_empty() else [],
        }
    finally:
        db.close()
