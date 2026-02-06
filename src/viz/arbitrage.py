"""Arbitrage visualization components."""

from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from src.models.schemas import ArbitrageOpportunity


def plot_arbitrage_opportunities(
    opportunities: list[ArbitrageOpportunity],
    title: str = "Arbitrage Opportunities",
) -> go.Figure:
    """
    Plot arbitrage opportunities over time.

    Args:
        opportunities: List of ArbitrageOpportunity objects
        title: Plot title

    Returns:
        Plotly Figure
    """
    if not opportunities:
        fig = go.Figure()
        fig.add_annotation(text="No arbitrage opportunities found", x=0.5, y=0.5, showarrow=False)
        return fig

    timestamps = [o.timestamp for o in opportunities]
    profits = [o.profit_pct for o in opportunities]
    platforms = [f"{o.platform_a.value} vs {o.platform_b.value}" for o in opportunities]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=profits,
            mode="markers",
            marker=dict(
                size=10,
                color=profits,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Profit %"),
            ),
            text=platforms,
            hovertemplate="Time: %{x}<br>Profit: %{y:.2f}%<br>Platforms: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Arbitrage Profit (%)",
        width=800,
        height=400,
    )

    return fig


def plot_price_divergence(
    prices_a: pl.DataFrame,
    prices_b: pl.DataFrame,
    platform_a: str,
    platform_b: str,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot price divergence between two markets over time.

    Args:
        prices_a: Price DataFrame for platform A
        prices_b: Price DataFrame for platform B
        platform_a: Name of platform A
        platform_b: Name of platform B
        title: Optional custom title

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Prices", "Price Difference"),
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
    )

    # Price lines
    if not prices_a.is_empty():
        fig.add_trace(
            go.Scatter(
                x=prices_a["timestamp"].to_list(),
                y=prices_a["probability"].to_list(),
                mode="lines",
                name=platform_a,
                line=dict(color="steelblue"),
            ),
            row=1,
            col=1,
        )

    if not prices_b.is_empty():
        fig.add_trace(
            go.Scatter(
                x=prices_b["timestamp"].to_list(),
                y=prices_b["probability"].to_list(),
                mode="lines",
                name=platform_b,
                line=dict(color="coral"),
            ),
            row=1,
            col=1,
        )

    # Price difference (if we can align the data)
    if not prices_a.is_empty() and not prices_b.is_empty():
        # Join on nearest timestamp (simplified)
        # In practice, you'd want proper time-series alignment
        diff_data = []

        for row_a in prices_a.iter_rows(named=True):
            ts_a = row_a["timestamp"]
            prob_a = row_a["probability"]

            # Find closest in B
            closest = prices_b.with_columns(
                (pl.col("timestamp") - ts_a).abs().alias("diff")
            ).sort("diff").head(1)

            if not closest.is_empty():
                prob_b = closest["probability"][0]
                diff_data.append(
                    {"timestamp": ts_a, "difference": prob_a - prob_b}
                )

        if diff_data:
            diff_df = pl.DataFrame(diff_data)

            # Highlight arbitrage zone
            fig.add_hrect(
                y0=-0.02,
                y1=0.02,
                line_width=0,
                fillcolor="green",
                opacity=0.1,
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=diff_df["timestamp"].to_list(),
                    y=diff_df["difference"].to_list(),
                    mode="lines",
                    name="Difference",
                    line=dict(color="gray"),
                    fill="tozeroy",
                ),
                row=2,
                col=1,
            )

    default_title = f"Price Divergence: {platform_a} vs {platform_b}"
    fig.update_layout(
        title=title or default_title,
        height=600,
        width=800,
    )

    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_yaxes(title_text="Difference", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)

    return fig


def plot_arbitrage_decay(
    historical_arbs: pl.DataFrame,
    title: str = "Arbitrage Decay Analysis",
) -> go.Figure:
    """
    Analyze how quickly arbitrage opportunities close.

    Args:
        historical_arbs: DataFrame with historical arbitrage data
        title: Plot title

    Returns:
        Plotly Figure
    """
    if historical_arbs.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No historical data", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Profit Distribution", "Opportunities Over Time"),
    )

    # Profit distribution
    fig.add_trace(
        go.Histogram(
            x=historical_arbs["profit_pct"].to_list(),
            nbinsx=20,
            name="Profit %",
            marker_color="steelblue",
        ),
        row=1,
        col=1,
    )

    # Opportunities over time (grouped by day/week)
    arbs_by_time = (
        historical_arbs.with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.len().alias("count"), pl.mean("profit_pct").alias("avg_profit"))
        .sort("date")
    )

    if not arbs_by_time.is_empty():
        fig.add_trace(
            go.Bar(
                x=arbs_by_time["date"].to_list(),
                y=arbs_by_time["count"].to_list(),
                name="Count",
                marker_color="coral",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=title,
        height=400,
        width=900,
    )

    fig.update_xaxes(title_text="Profit (%)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Opportunities", row=1, col=2)

    return fig


def plot_platform_spreads(
    spreads_data: dict[str, list[float]],
    title: str = "Platform Bid-Ask Spreads",
) -> go.Figure:
    """
    Compare bid-ask spreads across platforms.

    Args:
        spreads_data: Dict mapping platform name to list of spread values
        title: Plot title

    Returns:
        Plotly Figure
    """
    if not spreads_data:
        fig = go.Figure()
        fig.add_annotation(text="No spread data", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = go.Figure()

    for platform, spreads in spreads_data.items():
        fig.add_trace(
            go.Box(
                y=spreads,
                name=platform,
                boxpoints="outliers",
            )
        )

    fig.update_layout(
        title=title,
        yaxis_title="Spread (%)",
        width=600,
        height=400,
    )

    return fig


def plot_arbitrage_heatmap(
    arb_matrix: pl.DataFrame,
    title: str = "Cross-Platform Arbitrage Frequency",
) -> go.Figure:
    """
    Heatmap showing arbitrage frequency between platform pairs.

    Args:
        arb_matrix: DataFrame with platform pairs and counts
        title: Plot title

    Returns:
        Plotly Figure
    """
    if arb_matrix.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    # Pivot to matrix form
    platforms = sorted(
        set(arb_matrix["platform_a"].to_list() + arb_matrix["platform_b"].to_list())
    )

    matrix = []
    for p1 in platforms:
        row = []
        for p2 in platforms:
            if p1 == p2:
                row.append(0)
            else:
                count = arb_matrix.filter(
                    ((pl.col("platform_a") == p1) & (pl.col("platform_b") == p2))
                    | ((pl.col("platform_a") == p2) & (pl.col("platform_b") == p1))
                )
                row.append(count["count"].sum() if not count.is_empty() else 0)
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=platforms,
            y=platforms,
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )

    fig.update_layout(
        title=title,
        width=500,
        height=500,
    )

    return fig
