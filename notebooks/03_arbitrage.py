"""
Marimo notebook for arbitrage detection and analysis.

Run with: marimo edit notebooks/03_arbitrage.py
"""

import marimo

__generated_with = "0.3.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import sys
    sys.path.insert(0, "..")

    from src.models.database import Database
    from src.analysis.arbitrage import (
        ArbitrageScanner,
        find_arbitrage,
        find_complement_violation,
    )
    from src.viz.arbitrage import (
        plot_arbitrage_opportunities,
        plot_price_divergence,
    )

    mo.md("# Arbitrage Detection")
    return (
        ArbitrageScanner,
        Database,
        find_arbitrage,
        find_complement_violation,
        go,
        make_subplots,
        mo,
        pl,
        plot_arbitrage_opportunities,
        plot_price_divergence,
        sys,
    )


@app.cell
def __(mo):
    mo.md("""
    ## How Arbitrage Works

    **Cross-market arbitrage** occurs when the same event is priced differently:

    - Market A: 30% YES → pays 3.33x if YES
    - Market B: 60% YES → 40% NO → pays 2.5x if NO

    **Strategy**: Buy YES on A, buy NO on B

    - Cost: $0.30 + $0.40 = $0.70
    - If YES: Get $1 from A → Profit $0.30
    - If NO: Get $1 from B → Profit $0.30

    **Guaranteed 42.8% profit!**
    """)
    return


@app.cell
def __(ArbitrageScanner, Database, mo):
    db = Database()
    scanner = ArbitrageScanner(db)

    mo.md("## Interactive Arbitrage Calculator")
    return db, scanner


@app.cell
def __(mo):
    # Interactive inputs
    prob_a_slider = mo.ui.slider(0.05, 0.95, value=0.3, step=0.05, label="Market A Probability")
    prob_b_slider = mo.ui.slider(0.05, 0.95, value=0.6, step=0.05, label="Market B Probability")

    mo.vstack([prob_a_slider, prob_b_slider])
    return prob_a_slider, prob_b_slider


@app.cell
def __(find_arbitrage, mo, prob_a_slider, prob_b_slider):
    prob_a = prob_a_slider.value
    prob_b = prob_b_slider.value

    arb = find_arbitrage(prob_a, prob_b)

    if arb:
        mo.md(f"""
        ### Arbitrage Found!

        | Metric | Value |
        |--------|-------|
        | Strategy | {arb['type'].replace('_', ' ').title()} |
        | Profit | **{arb['profit_pct']:.2f}%** |
        | Stake on A | {arb['stake_a']:.1%} |
        | Stake on B | {arb['stake_b']:.1%} |
        | Combined Cost | {arb['combined_cost']:.1%} |

        **Example with $1000:**
        - Bet ${arb['stake_a'] * 1000:.2f} on {'YES' if 'yes_a' in arb['type'] else 'NO'} at Market A
        - Bet ${arb['stake_b'] * 1000:.2f} on {'NO' if 'yes_a' in arb['type'] else 'YES'} at Market B
        - Guaranteed return: ${1000 / arb['combined_cost']:.2f}
        - Profit: ${1000 / arb['combined_cost'] - 1000:.2f}
        """)
    else:
        spread = abs(prob_a - prob_b)
        mo.md(f"""
        ### No Arbitrage

        Price spread: {spread:.1%}

        For arbitrage, you need: prob_A + (1 - prob_B) < 1 or vice versa.

        Current: {prob_a:.1%} + {1 - prob_b:.1%} = {prob_a + (1 - prob_b):.1%}

        Try increasing the price difference!
        """)
    return arb, prob_a, prob_b, spread


@app.cell
def __(mo, scanner):
    mo.md("## Current Arbitrage Opportunities")
    return


@app.cell
def __(mo, plot_arbitrage_opportunities, scanner):
    # Scan for arbitrage
    try:
        opportunities = scanner.scan_linked_markets()

        if opportunities:
            mo.md(f"Found **{len(opportunities)}** arbitrage opportunities!")

            # Table
            rows = []
            for opp in sorted(opportunities, key=lambda x: x.profit_pct, reverse=True)[:10]:
                rows.append(
                    f"| {opp.platform_a.value} | {opp.platform_b.value} | "
                    f"{opp.prob_a:.1%} | {opp.prob_b:.1%} | {opp.profit_pct:.2f}% |"
                )

            if rows:
                mo.md("""
| Platform A | Platform B | Prob A | Prob B | Profit |
|------------|------------|--------|--------|--------|
""" + "\n".join(rows))
        else:
            mo.md("*No linked markets found for arbitrage scanning. Link markets to enable.*")
    except Exception as e:
        mo.md(f"*Error: {e}*")
    return opportunities, opp, rows


@app.cell
def __(mo, scanner):
    mo.md("## Historical Arbitrage Analysis")
    return


@app.cell
def __(mo, scanner):
    try:
        stats = scanner.arbitrage_statistics()

        if stats.get("total_opportunities", 0) > 0:
            mo.md(f"""
            ### Historical Statistics

            | Metric | Value |
            |--------|-------|
            | Total Opportunities | {stats['total_opportunities']:,} |
            | Average Profit | {stats['avg_profit_pct']:.2f}% |
            | Maximum Profit | {stats['max_profit_pct']:.2f}% |
            """)
        else:
            mo.md("*No historical arbitrage data. Need linked markets with price history.*")
    except Exception as e:
        mo.md(f"*Error: {e}*")
    return stats,


@app.cell
def __(mo, scanner):
    mo.md("## Finding Similar Markets")
    return


@app.cell
def __(mo, scanner):
    # Find potentially linked markets
    try:
        similar = scanner.find_similar_markets(title_similarity_threshold=0.5)

        if similar:
            mo.md(f"Found **{len(similar)}** potentially similar market pairs:")

            rows = []
            for m1, m2, sim in similar[:10]:
                rows.append(f"| {m1[:30]}... | {m2[:30]}... | {sim:.1%} |")

            mo.md("""
| Market 1 | Market 2 | Similarity |
|----------|----------|------------|
""" + "\n".join(rows))
        else:
            mo.md("*No similar markets found across platforms.*")
    except Exception as e:
        mo.md(f"*Error: {e}*")
    return m1, m2, rows, sim, similar


@app.cell
def __(db):
    db.close()
    return


if __name__ == "__main__":
    app.run()
