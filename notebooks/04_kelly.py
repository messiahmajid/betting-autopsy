"""
Marimo notebook for Kelly criterion and portfolio backtesting.

Run with: marimo edit notebooks/04_kelly.py
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
    from src.analysis.kelly import (
        kelly_fraction,
        fractional_kelly,
        expected_log_growth,
        BacktestEngine,
        KellyPortfolio,
        BetOpportunity,
    )
    from src.models.schemas import Platform
    from src.viz.portfolio import (
        plot_portfolio_growth,
        plot_drawdown,
        plot_kelly_sensitivity,
    )

    mo.md("# Kelly Criterion & Portfolio Backtesting")
    return (
        BacktestEngine,
        BetOpportunity,
        Database,
        KellyPortfolio,
        Platform,
        expected_log_growth,
        fractional_kelly,
        go,
        kelly_fraction,
        make_subplots,
        mo,
        np,
        pl,
        plot_drawdown,
        plot_kelly_sensitivity,
        plot_portfolio_growth,
        sys,
    )


@app.cell
def __(mo):
    mo.md("""
    ## The Kelly Criterion

    The Kelly criterion tells you the optimal bet size to maximize long-term growth:

    $$f^* = \\frac{bp - q}{b}$$

    Where:
    - $f^*$ = fraction of bankroll to bet
    - $b$ = net odds (decimal odds - 1)
    - $p$ = probability of winning
    - $q$ = probability of losing (1 - p)

    **Example**: 60% chance to win at 2:1 odds → f* = (1×0.6 - 0.4)/1 = 20%
    """)
    return


@app.cell
def __(mo):
    # Interactive Kelly calculator
    prob_slider = mo.ui.slider(0.1, 0.9, value=0.6, step=0.05, label="True Win Probability")
    odds_slider = mo.ui.slider(1.1, 5.0, value=2.0, step=0.1, label="Decimal Odds")

    mo.vstack([prob_slider, odds_slider])
    return odds_slider, prob_slider


@app.cell
def __(
    expected_log_growth,
    fractional_kelly,
    go,
    kelly_fraction,
    mo,
    np,
    odds_slider,
    prob_slider,
):
    p = prob_slider.value
    odds = odds_slider.value

    kelly = kelly_fraction(p, odds)
    half_kelly = fractional_kelly(p, odds, 0.5)
    quarter_kelly = fractional_kelly(p, odds, 0.25)

    # Calculate expected value
    ev = p * (odds - 1) - (1 - p)

    mo.md(f"""
    ### Kelly Calculation

    | Metric | Value |
    |--------|-------|
    | Edge (EV per $1) | {ev:.1%} |
    | Full Kelly | {kelly:.1%} |
    | Half Kelly | {half_kelly:.1%} |
    | Quarter Kelly | {quarter_kelly:.1%} |

    {"**Positive edge!** Bet recommended." if ev > 0 else "**Negative edge.** Don't bet."}
    """)
    return ev, half_kelly, kelly, odds, p, quarter_kelly


@app.cell
def __(expected_log_growth, go, mo, np, odds, p):
    # Plot expected log growth vs bet size
    bet_sizes = np.linspace(0, min(0.99, kelly * 2), 100) if kelly > 0 else np.linspace(0, 0.5, 100)
    growths = [expected_log_growth(p, odds, f) for f in bet_sizes]

    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(x=bet_sizes, y=growths, mode="lines", name="Expected Log Growth"))

    if kelly > 0:
        fig_growth.add_vline(x=kelly, line_dash="dash", line_color="red",
                           annotation_text=f"Kelly: {kelly:.1%}")
        fig_growth.add_vline(x=kelly/2, line_dash="dot", line_color="orange",
                           annotation_text=f"Half-Kelly: {kelly/2:.1%}")

    fig_growth.update_layout(
        title="Expected Log Growth vs Bet Size",
        xaxis_title="Bet Size (fraction of bankroll)",
        yaxis_title="Expected Log Growth",
    )

    mo.ui.plotly(fig_growth)
    return bet_sizes, fig_growth, growths


@app.cell
def __(BacktestEngine, Database, mo):
    mo.md("## Portfolio Backtesting")

    db = Database()
    engine = BacktestEngine(db)
    return db, engine


@app.cell
def __(mo):
    # Backtest parameters
    edge_input = mo.ui.slider(0.01, 0.15, value=0.05, step=0.01, label="Assumed Edge")
    kelly_frac_input = mo.ui.slider(0.1, 1.0, value=0.5, step=0.1, label="Kelly Fraction")

    mo.vstack([edge_input, kelly_frac_input])
    return edge_input, kelly_frac_input


@app.cell
def __(edge_input, engine, kelly_frac_input, mo, plot_portfolio_growth):
    try:
        result = engine.run_backtest(
            edge_estimate=edge_input.value,
            kelly_fraction=kelly_frac_input.value,
            initial_bankroll=10000.0,
        )

        mo.md(f"""
        ### Backtest Results

        | Metric | Value |
        |--------|-------|
        | Strategy | {result.strategy} |
        | Initial | ${result.initial_bankroll:,.0f} |
        | Final | ${result.final_bankroll:,.0f} |
        | Total Return | {result.total_return:.1f}% |
        | Max Drawdown | {result.max_drawdown:.1f}% |
        | Sharpe Ratio | {result.sharpe_ratio:.2f} |
        | Win Rate | {result.win_rate:.1f}% |
        | Total Bets | {result.n_bets} |
        """)
    except ValueError as e:
        mo.md(f"*No data for backtest: {e}*")
        result = None
    return result,


@app.cell
def __(mo, plot_portfolio_growth, result):
    if result and not result.history.is_empty():
        fig_portfolio = plot_portfolio_growth(result)
        mo.ui.plotly(fig_portfolio)
    else:
        mo.md("*Run backtest to see portfolio growth*")
    return fig_portfolio,


@app.cell
def __(mo):
    mo.md("## Kelly Fraction Sensitivity Analysis")
    return


@app.cell
def __(engine, go, mo, np, plot_kelly_sensitivity):
    # Run backtests with different Kelly fractions
    try:
        fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        returns = []
        sharpes = []
        drawdowns = []

        for frac in fractions:
            res = engine.run_backtest(edge_estimate=0.05, kelly_fraction=frac)
            returns.append(res.total_return)
            sharpes.append(res.sharpe_ratio)
            drawdowns.append(res.max_drawdown)

        fig_sens = plot_kelly_sensitivity(fractions, returns, sharpes, drawdowns)
        mo.ui.plotly(fig_sens)
    except Exception as e:
        mo.md(f"*Cannot run sensitivity analysis: {e}*")
    return drawdowns, fig_sens, frac, fractions, res, returns, sharpes


@app.cell
def __(mo):
    mo.md("""
    ## Key Insights

    **Why use fractional Kelly?**

    1. **Reduced variance**: Half-Kelly has ~75% of the growth but much less volatility
    2. **Estimation error**: We rarely know true probabilities exactly
    3. **Psychological comfort**: Smaller swings are easier to stick with
    4. **Tail risk**: Full Kelly can lead to large drawdowns

    **The edge matters most**:
    - Without an edge, Kelly = 0 (don't bet)
    - A 5% edge at even odds → 5% Kelly
    - Compound growth requires consistent positive edge
    """)
    return


@app.cell
def __(go, mo, np):
    # Simulate Kelly vs fixed betting
    mo.md("## Simulation: Kelly vs Fixed Betting")

    np.random.seed(42)
    n_bets = 100
    p_win = 0.55  # 55% win rate
    odds = 2.0  # Even money

    kelly_bet = kelly_fraction(p_win, odds)

    # Simulate both strategies
    outcomes = np.random.random(n_bets) < p_win

    # Kelly
    kelly_bankroll = [10000]
    for won in outcomes:
        br = kelly_bankroll[-1]
        bet = br * kelly_bet
        if won:
            kelly_bankroll.append(br + bet * (odds - 1))
        else:
            kelly_bankroll.append(br - bet)

    # Fixed 10% betting
    fixed_bankroll = [10000]
    for won in outcomes:
        br = fixed_bankroll[-1]
        bet = br * 0.10
        if won:
            fixed_bankroll.append(br + bet * (odds - 1))
        else:
            fixed_bankroll.append(br - bet)

    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(y=kelly_bankroll, name=f"Kelly ({kelly_bet:.1%})"))
    fig_sim.add_trace(go.Scatter(y=fixed_bankroll, name="Fixed (10%)"))
    fig_sim.update_layout(
        title="Kelly vs Fixed Betting Simulation",
        xaxis_title="Bet Number",
        yaxis_title="Bankroll ($)",
    )

    mo.ui.plotly(fig_sim)
    return (
        bet,
        br,
        fig_sim,
        fixed_bankroll,
        kelly_bankroll,
        kelly_bet,
        n_bets,
        odds,
        outcomes,
        p_win,
        won,
    )


@app.cell
def __(db):
    db.close()
    return


if __name__ == "__main__":
    app.run()
