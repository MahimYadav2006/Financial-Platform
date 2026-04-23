#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Stock Data Intelligence Platform — Task 1                   ║
║  Data Collection, Cleaning, Feature Engineering & Analysis   ║
║                                                              ║
║  Author: Mahim Yadav                                         ║
║  Tech:   Python, Pandas, NumPy, yfinance, Matplotlib         ║
╚══════════════════════════════════════════════════════════════╝

This script orchestrates the full Task 1 pipeline:

    1. DATA COLLECTION
       → Fetches 2 years of OHLCV data for 15 major NSE stocks + NIFTY 50
       → Uses yfinance API with retry logic and exponential backoff
       → Saves raw CSVs for audit trail

    2. DATA CLEANING
       → Handles missing values (forward/backward fill)
       → Fixes date formats and timezone issues
       → Validates price integrity (negative prices, High >= Low)
       → Removes duplicates
       → Generates data quality report

    3. FEATURE ENGINEERING
       Tier 1 (Required):
           • Daily Return = (Close - Open) / Open
           • 7-day Simple Moving Average
           • 52-week High / Low
       Tier 2 (Technical Indicators):
           • RSI, MACD, Bollinger Bands, ATR, OBV, EMAs
       Tier 3 (Custom / Creative):
           • Garman-Klass Volatility (OHLC-based estimator)
           • Rolling Sharpe Ratio (risk-adjusted returns)
           • Maximum Drawdown (peak-to-trough risk)
           • Composite Momentum Score (multi-timeframe)
           • Mock Sentiment Index (volume + momentum + volatility)
           • Trend Strength Index (MA alignment + consistency)
           • Gap Analysis (overnight price gaps)
           • Accumulation/Distribution Pressure (money flow)

    4. CROSS-STOCK ANALYSIS
       • Correlation matrix (pairwise return correlations)
       • Beta & Alpha vs NIFTY 50 (CAPM framework)
       • Risk-Return profiles (Sharpe, Sortino, Calmar, VaR)
       • Sector performance aggregation
       • Top Gainers / Losers identification
       • Volatility rankings

    5. VISUALIZATION
       • 10+ publication-quality charts in dark theme
       • Individual technical charts for top stocks
       • 6-panel summary dashboard

Usage:
    python main.py                    # Full pipeline (fetch + process + analyze)
    python main.py --use-cache        # Use cached data (skip API calls)
    python main.py --no-charts        # Skip chart generation
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from src.config import (
    STOCK_UNIVERSE,
    SECTOR_MAP,
    BENCHMARK_SYMBOL,
    START_DATE,
    END_DATE,
    DATA_PROCESSED_DIR,
    OUTPUT_REPORTS_DIR,
    OUTPUT_CHARTS_DIR,
)
from src.data_collector import StockDataCollector
from src.data_cleaner import StockDataCleaner
from src.feature_engineering import FeatureEngineer
from src.analysis import CrossStockAnalyzer
from src.visualizer import StockVisualizer

# ──────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

console = Console()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stock Data Intelligence Platform — Task 1 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                  # Full pipeline with data fetching
    python main.py --use-cache      # Reuse previously downloaded data
    python main.py --no-charts      # Process data without generating charts
        """,
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load data from cached CSVs instead of fetching from API",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation (faster for data-only runs)",
    )
    return parser.parse_args()


def print_header():
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   📊  STOCK DATA INTELLIGENCE PLATFORM                           ║
║       Task 1: Data Collection & Preparation                      ║
║                                                                  ║
║   Universe: 15 NSE Stocks + NIFTY 50 Benchmark                  ║
║   Period:   2 Years Historical Data                              ║
║   Metrics:  50+ Technical & Custom Indicators                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold cyan", box=box.DOUBLE))
    console.print(f"  Start Date : {START_DATE}")
    console.print(f"  End Date   : {END_DATE}")
    console.print(f"  Stocks     : {len(STOCK_UNIVERSE)} + {BENCHMARK_SYMBOL}")
    console.print(f"  Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    console.print()


def display_stock_universe():
    """Display the stock universe in a rich table."""
    table = Table(
        title="📋 Stock Universe",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="bold white")
    table.add_column("Company", style="white")
    table.add_column("Sector", style="yellow")

    for i, (symbol, name) in enumerate(STOCK_UNIVERSE.items(), 1):
        sector = SECTOR_MAP.get(symbol, "Unknown")
        table.add_row(str(i), symbol.replace(".NS", ""), name, sector)

    table.add_row("—", BENCHMARK_SYMBOL, "NIFTY 50 Index", "Benchmark", style="bold cyan")
    console.print(table)
    console.print()


def display_analysis_results(results: dict[str, pd.DataFrame]):
    """Display key analysis results in rich tables."""
    # ── Stock Summary Table ──
    summary = results.get("stock_summary")
    if summary is not None and not summary.empty:
        table = Table(
            title="📊 Stock Summary (Latest Data)",
            box=box.ROUNDED,
            border_style="green",
            header_style="bold green",
        )
        table.add_column("Symbol", style="bold white")
        table.add_column("Close (₹)", justify="right")
        table.add_column("52W High", justify="right")
        table.add_column("52W Low", justify="right")
        table.add_column("% From High", justify="right")
        table.add_column("RSI", justify="right")
        table.add_column("Sentiment", justify="right")
        table.add_column("Status")

        for _, row in summary.iterrows():
            rsi = row.get("RSI")
            rsi_str = f"{rsi:.0f}" if pd.notna(rsi) else "—"
            rsi_style = (
                "red" if pd.notna(rsi) and rsi > 70
                else "green" if pd.notna(rsi) and rsi < 30
                else "white"
            )

            sent = row.get("Sentiment_Index")
            sent_str = f"{sent:.0f}" if pd.notna(sent) else "—"

            pct_high = row.get("Pct_From_52W_High", 0)
            pct_str = f"{pct_high:.1f}%"

            status = row.get("Sentiment_Status", "—")

            table.add_row(
                str(row["Symbol"]),
                f"₹{row['Latest_Close']:,.2f}",
                f"₹{row['52W_High']:,.2f}",
                f"₹{row['52W_Low']:,.2f}",
                pct_str,
                f"[{rsi_style}]{rsi_str}[/{rsi_style}]",
                sent_str,
                str(status),
            )

        console.print(table)
        console.print()

    # ── Risk-Return Profile ──
    profile = results.get("risk_return_profile")
    if profile is not None and not profile.empty:
        table = Table(
            title="⚖️  Risk-Return Profile",
            box=box.ROUNDED,
            border_style="yellow",
            header_style="bold yellow",
        )
        table.add_column("Symbol", style="bold white")
        table.add_column("Return %", justify="right")
        table.add_column("Volatility %", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Max DD %", justify="right")
        table.add_column("VaR 95%", justify="right")

        for _, row in profile.iterrows():
            ret_style = "green" if row["Annual_Return_Pct"] > 0 else "red"
            sharpe_style = (
                "green" if row["Sharpe_Ratio"] > 0.5
                else "yellow" if row["Sharpe_Ratio"] > 0
                else "red"
            )

            table.add_row(
                str(row["Symbol"]),
                f"[{ret_style}]{row['Annual_Return_Pct']:.1f}%[/{ret_style}]",
                f"{row['Annual_Volatility_Pct']:.1f}%",
                f"[{sharpe_style}]{row['Sharpe_Ratio']:.2f}[/{sharpe_style}]",
                f"[red]{row['Max_Drawdown_Pct']:.1f}%[/red]",
                f"{row['VaR_95_Daily_Pct']:.2f}%",
            )

        console.print(table)
        console.print()

    # ── Beta Analysis ──
    beta = results.get("beta_analysis")
    if beta is not None and not beta.empty:
        table = Table(
            title="📈 Beta Analysis (vs NIFTY 50)",
            box=box.ROUNDED,
            border_style="magenta",
            header_style="bold magenta",
        )
        table.add_column("Symbol", style="bold white")
        table.add_column("Beta", justify="right")
        table.add_column("Alpha %", justify="right")
        table.add_column("R²", justify="right")
        table.add_column("Type")

        for _, row in beta.iterrows():
            beta_style = (
                "red" if row["Beta"] > 1.2
                else "green" if row["Beta"] < 0.8
                else "yellow"
            )
            alpha_style = "green" if row["Alpha_Annual"] > 0 else "red"

            table.add_row(
                str(row["Symbol"]),
                f"[{beta_style}]{row['Beta']:.3f}[/{beta_style}]",
                f"[{alpha_style}]{row['Alpha_Annual']:.1f}%[/{alpha_style}]",
                f"{row['R_Squared']:.3f}",
                str(row["Classification"]),
            )

        console.print(table)
        console.print()

    # ── Top Gainers / Losers ──
    gl = results.get("top_gainers_losers")
    if gl is not None and not gl.empty:
        console.print(Panel("🏆 TOP GAINERS (30 Days)", style="bold green"))
        top = gl.nlargest(5, "Return_30D_Pct")
        for _, row in top.iterrows():
            console.print(
                f"  [green]▲[/green] {row['Symbol']:>12}  "
                f"[green]+{row['Return_30D_Pct']:.1f}%[/green]  "
                f"₹{row['Latest_Close']:,.2f}"
            )

        console.print()
        console.print(Panel("📉 TOP LOSERS (30 Days)", style="bold red"))
        bottom = gl.nsmallest(5, "Return_30D_Pct")
        for _, row in bottom.iterrows():
            console.print(
                f"  [red]▼[/red] {row['Symbol']:>12}  "
                f"[red]{row['Return_30D_Pct']:.1f}%[/red]  "
                f"₹{row['Latest_Close']:,.2f}"
            )
        console.print()

    # ── Sector Performance ──
    sector = results.get("sector_performance")
    if sector is not None and not sector.empty:
        table = Table(
            title="🏛️  Sector Performance",
            box=box.ROUNDED,
            border_style="blue",
            header_style="bold blue",
        )
        table.add_column("Sector", style="bold white")
        table.add_column("Stocks", justify="center")
        table.add_column("Avg Return %", justify="right")
        table.add_column("Avg Volatility %", justify="right")
        table.add_column("Sentiment", justify="right")

        for _, row in sector.iterrows():
            ret_style = "green" if row["Avg_Annual_Return_Pct"] > 0 else "red"
            table.add_row(
                str(row["Sector"]),
                str(row["Num_Stocks"]),
                f"[{ret_style}]{row['Avg_Annual_Return_Pct']:.1f}%[/{ret_style}]",
                f"{row['Avg_Volatility_Pct']:.1f}%",
                f"{row['Avg_Sentiment']:.0f}",
            )

        console.print(table)
        console.print()


def display_final_summary(start_time: float, results: dict):
    """Display pipeline execution summary."""
    elapsed = time.time() - start_time

    console.print(
        Panel(
            f"""
[bold cyan]✅ PIPELINE COMPLETED SUCCESSFULLY[/bold cyan]

[bold]Execution Time:[/bold] {elapsed:.1f} seconds

[bold]Output Files:[/bold]
  📂 data/raw/          — Raw downloaded CSVs
  📂 data/processed/    — Cleaned & enriched datasets
  📂 output/reports/    — Analysis CSV reports
  📂 output/charts/     — Visualization charts (PNG)
  📄 pipeline.log       — Full execution log

[bold]Key Datasets:[/bold]
  • all_stocks_raw.csv       — Original API data
  • all_stocks_cleaned.csv   — Cleaned dataset
  • all_stocks_enriched.csv  — Full feature set (50+ metrics)

[bold]Analysis Reports:[/bold]
  • correlation_matrix.csv   — Return correlations
  • beta_analysis.csv        — CAPM beta & alpha
  • risk_return_profile.csv  — Sharpe, Sortino, VaR
  • stock_summary.csv        — Comprehensive summary
  • sector_performance.csv   — Sector aggregation
  • volatility_ranking.csv   — Risk ranking
            """,
            title="📊 Pipeline Summary",
            border_style="green",
            box=box.DOUBLE,
        )
    )


def main():
    """Main pipeline orchestrator."""
    args = parse_args()
    start_time = time.time()

    # ── Header ──
    print_header()
    display_stock_universe()

    # ══════════════════════════════════════════════════════════════
    # STEP 1: DATA COLLECTION
    # ══════════════════════════════════════════════════════════════
    console.print(Panel("📥 STEP 1: DATA COLLECTION", style="bold cyan", box=box.HEAVY))

    collector = StockDataCollector()

    if args.use_cache:
        loaded = collector.load_from_cache()
        if not loaded:
            console.print("[yellow]  No cache found. Fetching from API...[/yellow]")
            collector.fetch_all(save_raw=True)
    else:
        collector.fetch_all(save_raw=True)

    raw_data = collector.raw_data
    if not raw_data:
        console.print("[red bold]  ✗ No data collected. Exiting.[/red bold]")
        sys.exit(1)

    console.print(f"  [green]✓ Collected data for {len(raw_data)} tickers[/green]\n")

    # ══════════════════════════════════════════════════════════════
    # STEP 2: DATA CLEANING
    # ══════════════════════════════════════════════════════════════
    console.print(Panel("🧹 STEP 2: DATA CLEANING & VALIDATION", style="bold cyan", box=box.HEAVY))

    cleaner = StockDataCleaner()
    cleaned_data = cleaner.clean_all(raw_data)

    console.print(f"  [green]✓ Cleaned {len(cleaned_data)} stocks[/green]")

    # Print quality report
    issues_total = sum(
        len(r["issues_found"]) for r in cleaner.quality_report.values()
    )
    console.print(f"  [yellow]  Issues detected & fixed: {issues_total}[/yellow]\n")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════
    console.print(Panel("⚙️  STEP 3: FEATURE ENGINEERING", style="bold cyan", box=box.HEAVY))

    engineer = FeatureEngineer()
    enriched_data = engineer.transform_all(cleaned_data)

    total_features = max(
        (s["new_features"] for s in engineer.metrics_summary.values()), default=0
    )
    console.print(
        f"  [green]✓ Added ~{total_features} features per stock[/green]\n"
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 4: CROSS-STOCK ANALYSIS
    # ══════════════════════════════════════════════════════════════
    console.print(Panel("🔬 STEP 4: CROSS-STOCK ANALYSIS", style="bold cyan", box=box.HEAVY))

    analyzer = CrossStockAnalyzer(enriched_data)
    analysis_results = analyzer.run_all_analyses()

    console.print(f"  [green]✓ Completed {len(analysis_results)} analyses[/green]\n")

    # Display analysis results in rich tables
    display_analysis_results(analysis_results)

    # ══════════════════════════════════════════════════════════════
    # STEP 5: VISUALIZATION
    # ══════════════════════════════════════════════════════════════
    if not args.no_charts:
        console.print(Panel("📊 STEP 5: VISUALIZATION", style="bold cyan", box=box.HEAVY))

        visualizer = StockVisualizer(enriched_data, analysis_results)
        visualizer.generate_all_charts()

        console.print(f"  [green]✓ Charts saved to {OUTPUT_CHARTS_DIR}/[/green]\n")

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    display_final_summary(start_time, analysis_results)


if __name__ == "__main__":
    main()
