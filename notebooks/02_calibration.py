"""
Marimo notebook for calibration analysis.

Run with: marimo edit notebooks/02_calibration.py
"""

import marimo

__generated_with = "0.3.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import polars as pl
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import sys
    sys.path.insert(0, "..")

    from src.models.database import Database
    from src.models.schemas import Platform
    from src.analysis.calibration import (
        CalibrationAnalyzer,
        brier_score,
        calibration_curve,
    )
    from src.viz.calibration import (
        plot_calibration_curve,
        plot_calibration_comparison,
        plot_favorite_longshot_bias,
    )

    mo.md("# Calibration Analysis")
    return (
        CalibrationAnalyzer,
        Database,
        Platform,
        brier_score,
        calibration_curve,
        go,
        make_subplots,
        mo,
        np,
        pl,
        plot_calibration_comparison,
        plot_calibration_curve,
        plot_favorite_longshot_bias,
        sys,
    )


@app.cell
def __(CalibrationAnalyzer, Database, mo):
    db = Database()
    analyzer = CalibrationAnalyzer(db)

    mo.md("""
    ## Calibration Overview

    A well-calibrated forecaster's predictions match reality:
    - When they say 70%, it happens 70% of the time
    - **Brier Score**: Mean squared error (lower is better, 0.25 = random)
    - **ECE**: Expected Calibration Error (lower is better)
    - **Hosmer-Lemeshow**: Statistical test for calibration (p > 0.05 = good)
    """)
    return analyzer, db


@app.cell
def __(analyzer, mo):
    # Compare all platforms
    try:
        results = analyzer.compare_platforms(n_bins=10)

        if results:
            rows = []
            for name, result in sorted(results.items(), key=lambda x: x[1].brier_score):
                calibrated = "Yes" if result.hl_pvalue > 0.05 else "No"
                rows.append(
                    f"| {name} | {result.n_markets:,} | {result.brier_score:.4f} | "
                    f"{result.ece:.4f} | {result.hl_pvalue:.4f} | {calibrated} |"
                )

            table = """
## Platform Comparison

| Platform | Markets | Brier | ECE | H-L p-value | Calibrated? |
|----------|---------|-------|-----|-------------|-------------|
""" + "\n".join(rows)

            mo.md(table)
        else:
            mo.md("*No data available. Run data ingestion first.*")
    except Exception as e:
        mo.md(f"*Error: {e}*")
    return calibrated, name, result, results, rows, table


@app.cell
def __(mo, plot_calibration_comparison, results):
    # Plot calibration curves
    if results:
        fig = plot_calibration_comparison(results)
        mo.ui.plotly(fig)
    else:
        mo.md("*No data to plot*")
    return fig,


@app.cell
def __(analyzer, mo, plot_favorite_longshot_bias):
    # Favorite-longshot bias analysis
    mo.md("""
    ## Favorite-Longshot Bias

    Do low-probability events get overpriced (positive bias)?
    Do high-probability events get underpriced (negative bias)?
    """)
    return


@app.cell
def __(analyzer, mo, plot_favorite_longshot_bias):
    try:
        bias_df = analyzer.favorite_longshot_bias(n_bins=10)
        if not bias_df.is_empty():
            fig_bias = plot_favorite_longshot_bias(bias_df)
            mo.ui.plotly(fig_bias)
        else:
            mo.md("*No data for bias analysis*")
    except Exception as e:
        mo.md(f"*Error: {e}*")
    return bias_df, fig_bias


@app.cell
def __(bias_df, mo):
    # Analyze the bias
    if not bias_df.is_empty():
        longshots = bias_df.filter(pl.col("is_longshot"))
        favorites = bias_df.filter(pl.col("is_favorite"))

        longshot_bias = longshots["bias"].mean() if not longshots.is_empty() else 0
        favorite_bias = favorites["bias"].mean() if not favorites.is_empty() else 0

        mo.md(f"""
        ### Bias Summary

        - **Longshot bias** (prob < 20%): {longshot_bias:.3f}
          - Positive = overpriced (people overbet underdogs)
        - **Favorite bias** (prob > 80%): {favorite_bias:.3f}
          - Negative = underpriced (people underbet favorites)

        The classic favorite-longshot bias suggests longshots are overpriced.
        """)
    return favorite_bias, favorites, longshot_bias, longshots


@app.cell
def __(analyzer, go, mo, np):
    # Brier score decomposition
    mo.md("""
    ## Brier Score Decomposition

    Brier score can be decomposed into:
    - **Reliability**: How well calibrated (lower = better)
    - **Resolution**: How much predictions vary from base rate (higher = better)
    - **Uncertainty**: Base rate variance (constant for dataset)

    Good forecasters have low reliability and high resolution.
    """)
    return


@app.cell
def __(analyzer, db, go, mo, np):
    try:
        df = db.get_resolved_markets_with_final_price()

        if not df.is_empty():
            predictions = df["final_probability"].to_numpy()
            outcomes = df["resolution"].to_numpy()

            # Filter valid
            valid = ~(np.isnan(predictions) | np.isnan(outcomes))
            predictions = predictions[valid]
            outcomes = outcomes[valid]

            # Base rate
            base_rate = outcomes.mean()

            # Uncertainty (maximum possible Brier score)
            uncertainty = base_rate * (1 - base_rate)

            # Calculate components using calibration curve
            from src.analysis.calibration import calibration_curve as cal_curve
            curve = cal_curve(predictions, outcomes, n_bins=10)

            # Reliability: weighted MSE of calibration
            reliability = sum(
                b.count * (b.predicted_mean - b.actual_frequency) ** 2
                for b in curve
            ) / len(predictions)

            # Resolution: weighted variance of bin frequencies from base rate
            resolution = sum(
                b.count * (b.actual_frequency - base_rate) ** 2
                for b in curve
            ) / len(predictions)

            # Brier = Uncertainty - Resolution + Reliability
            brier = uncertainty - resolution + reliability

            mo.md(f"""
            ### Decomposition Results

            | Component | Value | Interpretation |
            |-----------|-------|----------------|
            | Uncertainty | {uncertainty:.4f} | Base rate variance |
            | Resolution | {resolution:.4f} | Prediction skill (higher = better) |
            | Reliability | {reliability:.4f} | Calibration error (lower = better) |
            | **Brier Score** | {brier:.4f} | Overall (lower = better) |

            *Note: Brier = Uncertainty - Resolution + Reliability*
            """)
    except Exception as e:
        mo.md(f"*Error: {e}*")
    return (
        base_rate,
        brier,
        cal_curve,
        curve,
        df,
        outcomes,
        predictions,
        reliability,
        resolution,
        uncertainty,
        valid,
    )


@app.cell
def __(db):
    db.close()
    return


if __name__ == "__main__":
    app.run()
