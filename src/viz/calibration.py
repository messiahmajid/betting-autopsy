"""Calibration visualizations using Plotly."""

from typing import Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from src.models.schemas import CalibrationBucket
from src.analysis.calibration import CalibrationResult


def plot_calibration_curve(
    result: CalibrationResult,
    title: Optional[str] = None,
    show_confidence: bool = True,
) -> go.Figure:
    """
    Plot calibration curve with perfect calibration line.

    Args:
        result: CalibrationResult from CalibrationAnalyzer
        title: Optional custom title
        show_confidence: Whether to show confidence intervals

    Returns:
        Plotly Figure
    """
    curve = result.calibration_curve

    # Extract data
    predicted = [b.predicted_mean for b in curve]
    actual = [b.actual_frequency for b in curve]
    counts = [b.count for b in curve]
    errors = [b.std_error if b.std_error else 0 for b in curve]

    # Create figure
    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="gray", dash="dash"),
        )
    )

    # Calibration curve with error bars
    if show_confidence and any(e > 0 for e in errors):
        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=actual,
                mode="markers+lines",
                name="Observed",
                marker=dict(
                    size=[max(8, min(20, c / 10)) for c in counts],
                    color="steelblue",
                ),
                error_y=dict(
                    type="data",
                    array=[1.96 * e for e in errors],  # 95% CI
                    visible=True,
                ),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=actual,
                mode="markers+lines",
                name="Observed",
                marker=dict(
                    size=[max(8, min(20, c / 10)) for c in counts],
                    color="steelblue",
                ),
            )
        )

    # Title and labels
    platform_name = result.platform.value if result.platform else "All Platforms"
    default_title = f"Calibration Curve - {platform_name}"
    if title:
        default_title = title

    fig.update_layout(
        title=dict(text=default_title),
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        xaxis=dict(range=[0, 1], dtick=0.1),
        yaxis=dict(range=[0, 1], dtick=0.1),
        width=600,
        height=500,
        showlegend=True,
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Brier: {result.brier_score:.4f}<br>ECE: {result.ece:.4f}<br>n={result.n_markets}",
                showarrow=False,
                font=dict(size=10),
                align="left",
                bgcolor="white",
                bordercolor="gray",
                borderwidth=1,
            )
        ],
    )

    return fig


def plot_calibration_comparison(
    results: dict[str, CalibrationResult],
    title: str = "Calibration Comparison",
) -> go.Figure:
    """
    Plot calibration curves for multiple platforms/methods.

    Args:
        results: Dict mapping name to CalibrationResult
        title: Plot title

    Returns:
        Plotly Figure
    """
    colors = ["steelblue", "coral", "green", "purple", "orange"]
    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect",
            line=dict(color="gray", dash="dash"),
        )
    )

    for i, (name, result) in enumerate(results.items()):
        curve = result.calibration_curve
        predicted = [b.predicted_mean for b in curve]
        actual = [b.actual_frequency for b in curve]

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=actual,
                mode="markers+lines",
                name=f"{name} (Brier: {result.brier_score:.3f})",
                marker=dict(color=color),
                line=dict(color=color),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=700,
        height=500,
    )

    return fig


def plot_brier_over_time(
    data: pl.DataFrame,
    time_col: str = "resolved_at",
    window_size: int = 50,
    title: str = "Brier Score Over Time",
) -> go.Figure:
    """
    Plot rolling Brier score over time.

    Args:
        data: DataFrame with predictions and outcomes
        time_col: Column name for timestamp
        window_size: Rolling window size
        title: Plot title

    Returns:
        Plotly Figure
    """
    if data.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig

    # Sort by time and calculate rolling Brier
    df = data.sort(time_col)

    # Calculate squared errors
    df = df.with_columns(
        ((pl.col("final_probability") - pl.col("resolution")) ** 2).alias("squared_error")
    )

    # Rolling mean
    df = df.with_columns(
        pl.col("squared_error").rolling_mean(window_size=window_size).alias("rolling_brier")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[time_col].to_list(),
            y=df["rolling_brier"].to_list(),
            mode="lines",
            name=f"Rolling Brier (n={window_size})",
            line=dict(color="steelblue"),
        )
    )

    # Add reference line for random guessing
    fig.add_hline(
        y=0.25,
        line_dash="dash",
        line_color="gray",
        annotation_text="Random Guessing",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Brier Score",
        yaxis=dict(range=[0, 0.5]),
        width=800,
        height=400,
    )

    return fig


def plot_favorite_longshot_bias(
    data: pl.DataFrame,
    title: str = "Favorite-Longshot Bias",
) -> go.Figure:
    """
    Plot bias by probability bucket.

    Positive bias = market overestimates probability (longshots overpriced)
    Negative bias = market underestimates probability (favorites underpriced)

    Args:
        data: DataFrame from CalibrationAnalyzer.favorite_longshot_bias()
        title: Plot title

    Returns:
        Plotly Figure
    """
    if data.is_empty():
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = go.Figure()

    # Bar chart of bias by bucket
    bin_labels = [f"{row['bin_start']:.0%}-{row['bin_end']:.0%}" for row in data.iter_rows(named=True)]
    biases = data["bias"].to_list()
    counts = data["count"].to_list()

    colors = ["coral" if b > 0 else "steelblue" for b in biases]

    fig.add_trace(
        go.Bar(
            x=bin_labels,
            y=biases,
            marker_color=colors,
            text=[f"n={c}" for c in counts],
            textposition="outside",
        )
    )

    fig.add_hline(y=0, line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="Probability Bucket",
        yaxis_title="Bias (Predicted - Actual)",
        width=800,
        height=400,
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text="Red = Overpriced<br>Blue = Underpriced",
                showarrow=False,
                font=dict(size=10),
                bgcolor="white",
            )
        ],
    )

    return fig


def plot_hosmer_lemeshow(
    curve: list[CalibrationBucket],
    chi2: float,
    p_value: float,
    title: str = "Hosmer-Lemeshow Test",
) -> go.Figure:
    """
    Visualize Hosmer-Lemeshow goodness-of-fit test.

    Args:
        curve: Calibration curve buckets
        chi2: Chi-squared statistic
        p_value: P-value from test
        title: Plot title

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Expected vs Observed", "Contribution to Chi²"),
    )

    # Left plot: Expected vs Observed counts
    expected = [b.predicted_mean * b.count for b in curve]
    observed = [b.actual_frequency * b.count for b in curve]
    bin_labels = [f"{b.bin_start:.0%}-{b.bin_end:.0%}" for b in curve]

    fig.add_trace(
        go.Bar(x=bin_labels, y=expected, name="Expected", marker_color="steelblue"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=bin_labels, y=observed, name="Observed", marker_color="coral"),
        row=1,
        col=1,
    )

    # Right plot: Chi² contributions
    chi2_contrib = []
    for b in curve:
        exp = b.predicted_mean * b.count
        obs = b.actual_frequency * b.count
        if exp > 0 and exp < b.count:
            contrib = (obs - exp) ** 2 / (exp * (1 - exp / b.count))
        else:
            contrib = 0
        chi2_contrib.append(contrib)

    fig.add_trace(
        go.Bar(x=bin_labels, y=chi2_contrib, name="χ² Contribution", marker_color="green"),
        row=1,
        col=2,
    )

    # Interpretation
    interpretation = "Good fit" if p_value > 0.05 else "Poor fit"

    fig.update_layout(
        title=f"{title}<br>χ² = {chi2:.2f}, p = {p_value:.4f} ({interpretation})",
        width=1000,
        height=400,
        barmode="group",
    )

    return fig
