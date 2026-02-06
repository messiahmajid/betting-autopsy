"""Portfolio visualization components."""

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from src.analysis.kelly import BacktestResult


def plot_portfolio_growth(
    result: BacktestResult,
    title: Optional[str] = None,
    log_scale: bool = False,
) -> go.Figure:
    """
    Plot portfolio value over time.

    Args:
        result: BacktestResult from Kelly backtest
        title: Optional custom title
        log_scale: Whether to use log scale for y-axis

    Returns:
        Plotly Figure
    """
    if result.history.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig

    history = result.history

    fig = go.Figure()

    # Portfolio value line
    fig.add_trace(
        go.Scatter(
            x=history["timestamp"].to_list(),
            y=history["bankroll"].to_list(),
            mode="lines",
            name="Portfolio Value",
            line=dict(color="steelblue", width=2),
        )
    )

    # Starting value reference
    fig.add_hline(
        y=result.initial_bankroll,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial",
    )

    default_title = f"Portfolio Growth - {result.strategy}"
    fig.update_layout(
        title=title or default_title,
        xaxis_title="Time",
        yaxis_title="Portfolio Value ($)",
        yaxis_type="log" if log_scale else "linear",
        width=800,
        height=400,
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"Return: {result.total_return:.1f}%<br>"
                f"Sharpe: {result.sharpe_ratio:.2f}<br>"
                f"Max DD: {result.max_drawdown:.1f}%",
                showarrow=False,
                font=dict(size=10),
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
                align="left",
            )
        ],
    )

    return fig


def plot_drawdown(
    result: BacktestResult,
    title: str = "Portfolio Drawdown",
) -> go.Figure:
    """
    Plot drawdown over time.

    Args:
        result: BacktestResult from Kelly backtest
        title: Plot title

    Returns:
        Plotly Figure
    """
    if result.history.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig

    history = result.history
    bankroll = history["bankroll"].to_numpy()

    # Calculate drawdown
    peak = np.maximum.accumulate(bankroll)
    drawdown = (peak - bankroll) / peak * 100  # As percentage

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["timestamp"].to_list(),
            y=drawdown.tolist(),
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="coral"),
            fillcolor="rgba(255, 127, 80, 0.3)",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Drawdown (%)",
        yaxis=dict(autorange="reversed"),  # Drawdown is negative
        width=800,
        height=300,
    )

    return fig


def plot_returns_distribution(
    result: BacktestResult,
    title: str = "Returns Distribution",
) -> go.Figure:
    """
    Plot histogram of trade returns.

    Args:
        result: BacktestResult from Kelly backtest
        title: Plot title

    Returns:
        Plotly Figure
    """
    if result.history.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig

    history = result.history

    # Filter to resolved positions (those with PnL)
    pnl_data = history.filter(pl.col("action") == "resolve")

    if pnl_data.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No resolved positions", x=0.5, y=0.5, showarrow=False)
        return fig

    pnl = pnl_data["pnl"].to_numpy()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=pnl,
            nbinsx=30,
            name="P&L Distribution",
            marker_color="steelblue",
        )
    )

    # Add vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    # Add mean line
    mean_pnl = np.mean(pnl)
    fig.add_vline(
        x=mean_pnl,
        line_color="coral",
        annotation_text=f"Mean: ${mean_pnl:.2f}",
    )

    fig.update_layout(
        title=title,
        xaxis_title="P&L ($)",
        yaxis_title="Count",
        width=600,
        height=400,
    )

    return fig


def plot_strategy_comparison(
    results: list[BacktestResult],
    title: str = "Strategy Comparison",
) -> go.Figure:
    """
    Compare multiple backtest strategies.

    Args:
        results: List of BacktestResult objects
        title: Plot title

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Portfolio Growth",
            "Drawdown",
            "Risk-Return",
            "Win Rate vs Return",
        ),
    )

    colors = ["steelblue", "coral", "green", "purple", "orange"]

    for i, result in enumerate(results):
        color = colors[i % len(colors)]

        if not result.history.is_empty():
            # Portfolio growth
            fig.add_trace(
                go.Scatter(
                    x=result.history["timestamp"].to_list(),
                    y=result.history["bankroll"].to_list(),
                    mode="lines",
                    name=result.strategy,
                    line=dict(color=color),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Drawdown
            bankroll = result.history["bankroll"].to_numpy()
            peak = np.maximum.accumulate(bankroll)
            drawdown = (peak - bankroll) / peak * 100

            fig.add_trace(
                go.Scatter(
                    x=result.history["timestamp"].to_list(),
                    y=drawdown.tolist(),
                    mode="lines",
                    name=result.strategy,
                    line=dict(color=color),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    # Risk-return scatter (Sharpe vs Return)
    fig.add_trace(
        go.Scatter(
            x=[r.sharpe_ratio for r in results],
            y=[r.total_return for r in results],
            mode="markers+text",
            text=[r.strategy for r in results],
            textposition="top center",
            marker=dict(size=15, color=colors[: len(results)]),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Win rate vs return
    fig.add_trace(
        go.Scatter(
            x=[r.win_rate for r in results],
            y=[r.total_return for r in results],
            mode="markers+text",
            text=[r.strategy for r in results],
            textposition="top center",
            marker=dict(size=15, color=colors[: len(results)]),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=title,
        height=800,
        width=1000,
    )

    fig.update_xaxes(title_text="Sharpe Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Total Return (%)", row=2, col=1)
    fig.update_xaxes(title_text="Win Rate (%)", row=2, col=2)
    fig.update_yaxes(title_text="Total Return (%)", row=2, col=2)

    return fig


def plot_kelly_sensitivity(
    fractions: list[float],
    returns: list[float],
    sharpes: list[float],
    max_drawdowns: list[float],
    title: str = "Kelly Fraction Sensitivity",
) -> go.Figure:
    """
    Plot how metrics change with different Kelly fractions.

    Args:
        fractions: List of Kelly fractions tested
        returns: Total returns for each fraction
        sharpes: Sharpe ratios for each fraction
        max_drawdowns: Maximum drawdowns for each fraction
        title: Plot title

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Total Return", "Sharpe Ratio", "Max Drawdown"),
    )

    # Returns
    fig.add_trace(
        go.Scatter(
            x=fractions,
            y=returns,
            mode="lines+markers",
            name="Return",
            line=dict(color="steelblue"),
        ),
        row=1,
        col=1,
    )

    # Sharpe
    fig.add_trace(
        go.Scatter(
            x=fractions,
            y=sharpes,
            mode="lines+markers",
            name="Sharpe",
            line=dict(color="green"),
        ),
        row=1,
        col=2,
    )

    # Max Drawdown
    fig.add_trace(
        go.Scatter(
            x=fractions,
            y=max_drawdowns,
            mode="lines+markers",
            name="Max DD",
            line=dict(color="coral"),
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=title,
        height=350,
        width=1000,
        showlegend=False,
    )

    fig.update_xaxes(title_text="Kelly Fraction", row=1, col=1)
    fig.update_xaxes(title_text="Kelly Fraction", row=1, col=2)
    fig.update_xaxes(title_text="Kelly Fraction", row=1, col=3)

    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Max Drawdown (%)", row=1, col=3)

    return fig
