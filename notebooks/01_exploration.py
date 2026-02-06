"""
Marimo notebook for initial data exploration.

Run with: marimo edit notebooks/01_exploration.py
"""

import marimo

__generated_with = "0.3.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go

    import sys
    sys.path.insert(0, "..")

    from src.models.database import Database
    from src.models.schemas import Platform

    mo.md("# Betting Market Autopsy - Data Exploration")
    return Database, Platform, go, mo, pl, px, sys


@app.cell
def __(Database, mo):
    # Connect to database
    db = Database()

    stats = db.get_stats()
    mo.md(f"""
    ## Database Statistics

    | Metric | Value |
    |--------|-------|
    | Total Markets | {stats.get('total_markets', 0):,} |
    | Resolved Markets | {stats.get('resolved_markets', 0):,} |
    | Price Snapshots | {stats.get('total_prices', 0):,} |
    | Forecasts | {stats.get('total_forecasts', 0):,} |
    """)
    return db, stats


@app.cell
def __(db, mo, px):
    # Markets by platform
    markets_by_platform = db.conn.execute(
        "SELECT platform, COUNT(*) as count FROM markets GROUP BY platform"
    ).pl()

    if not markets_by_platform.is_empty():
        fig_platforms = px.bar(
            markets_by_platform.to_pandas(),
            x="platform",
            y="count",
            title="Markets by Platform",
            color="platform",
        )
        mo.ui.plotly(fig_platforms)
    else:
        mo.md("*No data yet. Run data ingestion first.*")
    return fig_platforms, markets_by_platform


@app.cell
def __(db, mo, px):
    # Resolution distribution
    resolutions = db.conn.execute("""
        SELECT
            CASE
                WHEN resolution IS NULL THEN 'Unresolved'
                WHEN resolution >= 0.5 THEN 'Yes'
                ELSE 'No'
            END as outcome,
            COUNT(*) as count
        FROM markets
        GROUP BY outcome
    """).pl()

    if not resolutions.is_empty():
        fig_resolutions = px.pie(
            resolutions.to_pandas(),
            values="count",
            names="outcome",
            title="Market Resolutions",
        )
        mo.ui.plotly(fig_resolutions)
    else:
        mo.md("*No resolution data*")
    return fig_resolutions, resolutions


@app.cell
def __(db, mo, px):
    # Price distribution for resolved markets
    price_dist = db.conn.execute("""
        WITH final_prices AS (
            SELECT
                p.market_id,
                p.probability,
                ROW_NUMBER() OVER (PARTITION BY p.market_id ORDER BY p.timestamp DESC) as rn
            FROM prices p
            INNER JOIN markets m ON p.market_id = m.id
            WHERE m.resolution IS NOT NULL
        )
        SELECT probability FROM final_prices WHERE rn = 1
    """).pl()

    if not price_dist.is_empty():
        fig_prices = px.histogram(
            price_dist.to_pandas(),
            x="probability",
            nbins=20,
            title="Distribution of Final Probabilities (Resolved Markets)",
        )
        fig_prices.update_layout(xaxis_title="Probability", yaxis_title="Count")
        mo.ui.plotly(fig_prices)
    else:
        mo.md("*No price data*")
    return fig_prices, price_dist


@app.cell
def __(db, mo):
    # Sample of recent markets
    recent_markets = db.conn.execute("""
        SELECT id, platform, title, resolution, resolved_at
        FROM markets
        WHERE resolution IS NOT NULL
        ORDER BY resolved_at DESC
        LIMIT 10
    """).pl()

    mo.md("## Recent Resolved Markets")
    return recent_markets,


@app.cell
def __(mo, recent_markets):
    if not recent_markets.is_empty():
        mo.ui.table(recent_markets.to_pandas())
    else:
        mo.md("*No resolved markets*")
    return


@app.cell
def __(db, mo, px):
    # Markets over time
    markets_over_time = db.conn.execute("""
        SELECT
            DATE_TRUNC('month', created_at) as month,
            platform,
            COUNT(*) as count
        FROM markets
        WHERE created_at IS NOT NULL
        GROUP BY month, platform
        ORDER BY month
    """).pl()

    if not markets_over_time.is_empty():
        fig_time = px.line(
            markets_over_time.to_pandas(),
            x="month",
            y="count",
            color="platform",
            title="Markets Created Over Time",
        )
        mo.ui.plotly(fig_time)
    else:
        mo.md("*No time data*")
    return fig_time, markets_over_time


@app.cell
def __(db, mo):
    # Categories
    categories = db.conn.execute("""
        SELECT category, COUNT(*) as count
        FROM markets
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
        LIMIT 15
    """).pl()

    mo.md("## Top Categories")
    return categories,


@app.cell
def __(categories, mo, px):
    if not categories.is_empty():
        fig_cat = px.bar(
            categories.to_pandas(),
            x="count",
            y="category",
            orientation="h",
            title="Markets by Category",
        )
        fig_cat.update_layout(yaxis=dict(categoryorder="total ascending"))
        mo.ui.plotly(fig_cat)
    else:
        mo.md("*No category data*")
    return fig_cat,


@app.cell
def __(db):
    db.close()
    return


if __name__ == "__main__":
    app.run()
