"""
Visualization Module
=====================
Generates publication-quality charts and visual reports for stock analysis.

Chart Categories:
    1. Individual Stock Charts — Price history, technical indicators
    2. Cross-Stock Charts — Correlation heatmap, risk-return scatter
    3. Sector Analysis — Performance comparison, sector rotation
    4. Summary Dashboard — Multi-panel overview of market insights

All charts are saved as high-resolution PNG files in the output/charts/ directory.
Uses matplotlib + seaborn for static charts with a consistent dark theme.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from src.config import (
    SECTOR_MAP,
    BENCHMARK_SYMBOL,
    OUTPUT_CHARTS_DIR,
    STOCK_UNIVERSE,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Global Style Configuration
# ──────────────────────────────────────────────────────────────
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
ACCENT_BLUE = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_RED = "#f85149"
ACCENT_YELLOW = "#d29922"
ACCENT_PURPLE = "#bc8cff"
ACCENT_ORANGE = "#f0883e"
ACCENT_CYAN = "#39d353"

PALETTE = [
    ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW,
    ACCENT_PURPLE, ACCENT_ORANGE, ACCENT_CYAN,
    "#79c0ff", "#7ee787", "#ffa657", "#d2a8ff",
    "#ff7b72", "#56d364", "#e3b341", "#a5d6ff",
]


def _apply_dark_style():
    """Apply a consistent dark theme to all charts."""
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": PANEL_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "legend.facecolor": PANEL_BG,
        "legend.edgecolor": GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.titlesize": 16,
    })


class StockVisualizer:
    """
    Generates all visualization charts for the stock analysis platform.

    Usage:
        viz = StockVisualizer(enriched_data, analysis_results)
        viz.generate_all_charts()
    """

    def __init__(
        self,
        enriched_data: dict[str, pd.DataFrame],
        analysis_results: dict[str, pd.DataFrame],
        output_dir: Optional[Path] = None,
    ):
        self.data = enriched_data
        self.analysis = analysis_results
        self.output_dir = output_dir or OUTPUT_CHARTS_DIR
        self.stock_symbols = [s for s in enriched_data if s != BENCHMARK_SYMBOL]
        _apply_dark_style()

    def generate_all_charts(self):
        """Generate all chart categories."""
        logger.info(
            f"\n{'='*60}\n"
            f"  VISUALIZATION\n"
            f"  Generating charts...\n"
            f"{'='*60}\n"
        )

        self._chart_correlation_heatmap()
        self._chart_risk_return_scatter()
        self._chart_sector_performance()
        self._chart_price_comparison()
        self._chart_top_gainers_losers()
        self._chart_volatility_comparison()
        self._chart_beta_analysis()
        self._chart_sentiment_overview()
        self._chart_52week_position()
        self._chart_dashboard_summary()

        # Individual stock technical charts (for top 5 by volume)
        top_stocks = sorted(
            self.stock_symbols,
            key=lambda s: self.data[s]["Volume"].mean(),
            reverse=True,
        )[:5]

        for symbol in top_stocks:
            self._chart_individual_stock(symbol)

        logger.info(f"\n  All charts saved to: {self.output_dir}/\n")

    def _save_chart(self, fig, name: str):
        """Save chart with consistent settings."""
        filepath = self.output_dir / f"{name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        logger.info(f"  ✓ Saved: {name}.png")

    # ══════════════════════════════════════════════════════════════
    #  CROSS-STOCK CHARTS
    # ══════════════════════════════════════════════════════════════

    def _chart_correlation_heatmap(self):
        """Correlation matrix heatmap of daily returns."""
        corr = self.analysis.get("correlation_matrix")
        if corr is None or corr.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark", as_cmap=True)

        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmin=-1, vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8, "color": TEXT_COLOR},
            linewidths=0.5,
            linecolor=GRID_COLOR,
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        )

        ax.set_title("Stock Return Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
        fig.tight_layout()
        self._save_chart(fig, "correlation_heatmap")

    def _chart_risk_return_scatter(self):
        """Risk-return scatter plot (efficient frontier visualization)."""
        profile = self.analysis.get("risk_return_profile")
        if profile is None or profile.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Color by sector
        sectors = profile["Sector"].unique()
        sector_colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(sectors)}

        for _, row in profile.iterrows():
            color = sector_colors[row["Sector"]]
            ax.scatter(
                row["Annual_Volatility_Pct"],
                row["Annual_Return_Pct"],
                c=color,
                s=120,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.5,
                zorder=5,
            )
            ax.annotate(
                row["Symbol"],
                (row["Annual_Volatility_Pct"], row["Annual_Return_Pct"]),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=8,
                color=TEXT_COLOR,
                fontweight="bold",
            )

        # Add quadrant lines
        ax.axhline(y=0, color=ACCENT_YELLOW, linestyle="--", alpha=0.3, linewidth=1)
        mean_vol = profile["Annual_Volatility_Pct"].mean()
        ax.axvline(x=mean_vol, color=ACCENT_YELLOW, linestyle="--", alpha=0.3, linewidth=1)

        # Quadrant labels
        ax.text(0.02, 0.98, "Low Risk\nHigh Return ★", transform=ax.transAxes,
                fontsize=9, color=ACCENT_GREEN, alpha=0.6, va="top")
        ax.text(0.98, 0.98, "High Risk\nHigh Return", transform=ax.transAxes,
                fontsize=9, color=ACCENT_YELLOW, alpha=0.6, va="top", ha="right")
        ax.text(0.02, 0.02, "Low Risk\nLow Return", transform=ax.transAxes,
                fontsize=9, color=ACCENT_BLUE, alpha=0.6)
        ax.text(0.98, 0.02, "High Risk\nLow Return ✗", transform=ax.transAxes,
                fontsize=9, color=ACCENT_RED, alpha=0.6, ha="right")

        # Legend for sectors
        for sector, color in sector_colors.items():
            ax.scatter([], [], c=color, s=60, label=sector)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), framealpha=0.3, fontsize=8)

        ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
        ax.set_ylabel("Annualized Return (%)", fontsize=12)
        ax.set_title("Risk-Return Profile", fontsize=16, fontweight="bold", pad=15)

        fig.tight_layout()
        self._save_chart(fig, "risk_return_scatter")

    def _chart_sector_performance(self):
        """Sector performance bar chart."""
        sector = self.analysis.get("sector_performance")
        if sector is None or sector.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Returns
        colors = [ACCENT_GREEN if v >= 0 else ACCENT_RED
                  for v in sector["Avg_Annual_Return_Pct"]]
        axes[0].barh(sector["Sector"], sector["Avg_Annual_Return_Pct"], color=colors, alpha=0.85)
        axes[0].set_xlabel("Avg. Annual Return (%)")
        axes[0].set_title("Sector Returns", fontweight="bold")
        axes[0].axvline(x=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.5)

        # Volatility
        axes[1].barh(sector["Sector"], sector["Avg_Volatility_Pct"],
                     color=ACCENT_BLUE, alpha=0.85)
        axes[1].set_xlabel("Avg. Annual Volatility (%)")
        axes[1].set_title("Sector Volatility", fontweight="bold")

        fig.suptitle("Sector Performance Overview", fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        self._save_chart(fig, "sector_performance")

    def _chart_price_comparison(self):
        """Normalized price comparison (rebased to 100)."""
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, symbol in enumerate(self.stock_symbols):
            df = self.data[symbol].sort_values("Date")
            # Normalize to 100 at start
            normalized = df["Close"] / df["Close"].iloc[0] * 100
            label = symbol.replace(".NS", "")
            ax.plot(
                df["Date"], normalized,
                color=PALETTE[i % len(PALETTE)],
                linewidth=1.2,
                label=label,
                alpha=0.85,
            )

        # Add benchmark
        if BENCHMARK_SYMBOL in self.data:
            bench = self.data[BENCHMARK_SYMBOL].sort_values("Date")
            norm_bench = bench["Close"] / bench["Close"].iloc[0] * 100
            ax.plot(
                bench["Date"], norm_bench,
                color="white", linewidth=2, linestyle="--",
                label="NIFTY 50", alpha=0.7, zorder=10,
            )

        ax.axhline(y=100, color=ACCENT_YELLOW, linestyle=":", alpha=0.3, linewidth=1)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price (Base = 100)")
        ax.set_title("Normalized Price Comparison", fontsize=16, fontweight="bold", pad=15)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        fig.tight_layout()
        self._save_chart(fig, "price_comparison_normalized")

    def _chart_top_gainers_losers(self):
        """Top gainers and losers bar chart."""
        gl = self.analysis.get("top_gainers_losers")
        if gl is None or gl.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Top 5 Gainers (30-day)
        top = gl.nlargest(5, "Return_30D_Pct")
        colors_top = [ACCENT_GREEN] * len(top)
        axes[0].barh(top["Symbol"], top["Return_30D_Pct"], color=colors_top, alpha=0.85)
        axes[0].set_xlabel("30-Day Return (%)")
        axes[0].set_title("Top 5 Gainers (30D)", fontweight="bold", color=ACCENT_GREEN)
        for i, (_, row) in enumerate(top.iterrows()):
            axes[0].text(
                row["Return_30D_Pct"] + 0.3, i,
                f'{row["Return_30D_Pct"]:.1f}%',
                va="center", fontsize=9, color=TEXT_COLOR,
            )

        # Top 5 Losers (30-day)
        bottom = gl.nsmallest(5, "Return_30D_Pct")
        colors_bot = [ACCENT_RED] * len(bottom)
        axes[1].barh(bottom["Symbol"], bottom["Return_30D_Pct"], color=colors_bot, alpha=0.85)
        axes[1].set_xlabel("30-Day Return (%)")
        axes[1].set_title("Top 5 Losers (30D)", fontweight="bold", color=ACCENT_RED)
        for i, (_, row) in enumerate(bottom.iterrows()):
            axes[1].text(
                row["Return_30D_Pct"] - 0.3, i,
                f'{row["Return_30D_Pct"]:.1f}%',
                va="center", fontsize=9, color=TEXT_COLOR, ha="right",
            )

        fig.suptitle("Market Movers", fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        self._save_chart(fig, "top_gainers_losers")

    def _chart_volatility_comparison(self):
        """Volatility comparison across stocks."""
        vol = self.analysis.get("volatility_ranking")
        if vol is None or vol.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        colors = []
        for _, row in vol.iterrows():
            if row["Risk_Level"] == "High":
                colors.append(ACCENT_RED)
            elif row["Risk_Level"] == "Medium":
                colors.append(ACCENT_YELLOW)
            else:
                colors.append(ACCENT_GREEN)

        ax.barh(vol["Symbol"], vol["Historical_Vol_Pct"], color=colors, alpha=0.85)
        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_title("Volatility Ranking", fontsize=16, fontweight="bold", pad=15)

        # Add risk level labels
        for i, (_, row) in enumerate(vol.iterrows()):
            ax.text(
                row["Historical_Vol_Pct"] + 0.5, i,
                row["Risk_Level"],
                va="center", fontsize=8, color=TEXT_COLOR, alpha=0.7,
            )

        fig.tight_layout()
        self._save_chart(fig, "volatility_ranking")

    def _chart_beta_analysis(self):
        """Beta analysis visualization."""
        beta = self.analysis.get("beta_analysis")
        if beta is None or beta.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        colors = []
        for _, row in beta.iterrows():
            if row["Classification"] == "Aggressive":
                colors.append(ACCENT_RED)
            elif row["Classification"] == "Moderate":
                colors.append(ACCENT_YELLOW)
            else:
                colors.append(ACCENT_GREEN)

        bars = ax.barh(beta["Symbol"], beta["Beta"], color=colors, alpha=0.85)
        ax.axvline(x=1.0, color="white", linestyle="--", alpha=0.5, linewidth=1, label="Market β=1")
        ax.set_xlabel("Beta (β)")
        ax.set_title("Stock Beta vs NIFTY 50", fontsize=16, fontweight="bold", pad=15)
        ax.legend(fontsize=9)

        for i, (_, row) in enumerate(beta.iterrows()):
            ax.text(
                row["Beta"] + 0.02, i,
                f'{row["Beta"]:.2f} ({row["Classification"]})',
                va="center", fontsize=8, color=TEXT_COLOR,
            )

        fig.tight_layout()
        self._save_chart(fig, "beta_analysis")

    def _chart_sentiment_overview(self):
        """Sentiment Index overview for all stocks."""
        fig, ax = plt.subplots(figsize=(12, 7))

        sentiments = []
        for symbol in self.stock_symbols:
            df = self.data[symbol]
            if "Sentiment_Index" in df.columns:
                latest = df["Sentiment_Index"].dropna()
                if len(latest) > 0:
                    sentiments.append({
                        "Symbol": symbol.replace(".NS", ""),
                        "Sentiment": latest.iloc[-1],
                    })

        if not sentiments:
            plt.close(fig)
            return

        sent_df = pd.DataFrame(sentiments).sort_values("Sentiment", ascending=True)

        colors = []
        for val in sent_df["Sentiment"]:
            if val >= 75:
                colors.append(ACCENT_GREEN)
            elif val >= 60:
                colors.append("#7ee787")
            elif val >= 40:
                colors.append(ACCENT_YELLOW)
            elif val >= 25:
                colors.append(ACCENT_ORANGE)
            else:
                colors.append(ACCENT_RED)

        ax.barh(sent_df["Symbol"], sent_df["Sentiment"], color=colors, alpha=0.85)
        ax.axvline(x=50, color="white", linestyle="--", alpha=0.4, linewidth=1, label="Neutral")
        ax.set_xlabel("Sentiment Index (0=Fear, 100=Greed)")
        ax.set_title("Market Sentiment Index", fontsize=16, fontweight="bold", pad=15)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=9)

        fig.tight_layout()
        self._save_chart(fig, "sentiment_overview")

    def _chart_52week_position(self):
        """52-week range position for all stocks."""
        fig, ax = plt.subplots(figsize=(12, 7))

        positions = []
        for symbol in self.stock_symbols:
            df = self.data[symbol]
            if "Position_In_52W_Range" in df.columns:
                latest = df["Position_In_52W_Range"].dropna()
                if len(latest) > 0:
                    positions.append({
                        "Symbol": symbol.replace(".NS", ""),
                        "Position": latest.iloc[-1],
                    })

        if not positions:
            plt.close(fig)
            return

        pos_df = pd.DataFrame(positions).sort_values("Position", ascending=True)

        colors = []
        for val in pos_df["Position"]:
            if val >= 80:
                colors.append(ACCENT_GREEN)
            elif val >= 50:
                colors.append(ACCENT_BLUE)
            elif val >= 30:
                colors.append(ACCENT_YELLOW)
            else:
                colors.append(ACCENT_RED)

        ax.barh(pos_df["Symbol"], pos_df["Position"], color=colors, alpha=0.85)
        ax.set_xlabel("Position in 52-Week Range (%)")
        ax.set_title("52-Week Range Position", fontsize=16, fontweight="bold", pad=15)
        ax.set_xlim(0, 100)

        # Reference lines
        ax.axvline(x=50, color="white", linestyle="--", alpha=0.3, linewidth=1)
        ax.axvline(x=80, color=ACCENT_GREEN, linestyle=":", alpha=0.3, linewidth=1, label="Near 52W High")
        ax.axvline(x=20, color=ACCENT_RED, linestyle=":", alpha=0.3, linewidth=1, label="Near 52W Low")
        ax.legend(fontsize=8)

        fig.tight_layout()
        self._save_chart(fig, "52week_range_position")

    def _chart_individual_stock(self, symbol: str):
        """
        Multi-panel technical chart for an individual stock.

        Panel 1: Price + Moving Averages + Bollinger Bands
        Panel 2: Volume
        Panel 3: RSI
        Panel 4: MACD
        """
        df = self.data[symbol].sort_values("Date").tail(252)  # Last 1 year
        name = symbol.replace(".NS", "")

        fig = plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.08)

        # ── Panel 1: Price + MAs + Bollinger ──
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df["Date"], df["Close"], color=ACCENT_BLUE, linewidth=1.5, label="Close")

        if "SMA_7" in df.columns:
            ax1.plot(df["Date"], df["SMA_7"], color=ACCENT_GREEN, linewidth=0.8,
                     alpha=0.7, label="SMA 7")
        if "SMA_50" in df.columns:
            ax1.plot(df["Date"], df["SMA_50"], color=ACCENT_YELLOW, linewidth=0.8,
                     alpha=0.7, label="SMA 50")
        if "SMA_200" in df.columns:
            ax1.plot(df["Date"], df["SMA_200"], color=ACCENT_RED, linewidth=0.8,
                     alpha=0.7, label="SMA 200")

        # Bollinger Bands
        if "BB_Upper" in df.columns:
            ax1.fill_between(
                df["Date"], df["BB_Upper"], df["BB_Lower"],
                alpha=0.08, color=ACCENT_BLUE, label="Bollinger Band",
            )

        ax1.set_ylabel("Price (₹)")
        ax1.set_title(f"{name} — Technical Analysis (1 Year)", fontsize=16,
                       fontweight="bold", pad=15)
        ax1.legend(loc="upper left", fontsize=8, framealpha=0.3)
        ax1.set_xticklabels([])

        # ── Panel 2: Volume ──
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        vol_colors = [ACCENT_GREEN if c >= o else ACCENT_RED
                      for c, o in zip(df["Close"], df["Open"])]
        ax2.bar(df["Date"], df["Volume"], color=vol_colors, alpha=0.6, width=1)
        if "Volume_SMA_20" in df.columns:
            ax2.plot(df["Date"], df["Volume_SMA_20"], color=ACCENT_YELLOW,
                     linewidth=0.8, alpha=0.7, label="Vol SMA 20")
        ax2.set_ylabel("Volume")
        ax2.legend(loc="upper left", fontsize=8, framealpha=0.3)
        ax2.set_xticklabels([])

        # ── Panel 3: RSI ──
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        if "RSI" in df.columns:
            ax3.plot(df["Date"], df["RSI"], color=ACCENT_PURPLE, linewidth=1)
            ax3.axhline(y=70, color=ACCENT_RED, linestyle="--", alpha=0.5, linewidth=0.8)
            ax3.axhline(y=30, color=ACCENT_GREEN, linestyle="--", alpha=0.5, linewidth=0.8)
            ax3.fill_between(df["Date"], 30, 70, alpha=0.04, color=ACCENT_YELLOW)
            ax3.set_ylabel("RSI")
            ax3.set_ylim(0, 100)
        ax3.set_xticklabels([])

        # ── Panel 4: MACD ──
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        if "MACD_Line" in df.columns:
            ax4.plot(df["Date"], df["MACD_Line"], color=ACCENT_BLUE, linewidth=1, label="MACD")
            ax4.plot(df["Date"], df["MACD_Signal"], color=ACCENT_ORANGE, linewidth=1, label="Signal")

            hist_colors = [ACCENT_GREEN if v >= 0 else ACCENT_RED
                          for v in df["MACD_Histogram"]]
            ax4.bar(df["Date"], df["MACD_Histogram"], color=hist_colors, alpha=0.4, width=1)
            ax4.axhline(y=0, color=TEXT_COLOR, linewidth=0.5, alpha=0.3)
            ax4.set_ylabel("MACD")
            ax4.legend(loc="upper left", fontsize=8, framealpha=0.3)

        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        fig.tight_layout()
        self._save_chart(fig, f"technical_{name}")

    def _chart_dashboard_summary(self):
        """Multi-panel summary dashboard."""
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

        # ── Panel 1: Cumulative Returns ──
        ax1 = fig.add_subplot(gs[0, 0])
        for i, symbol in enumerate(self.stock_symbols[:8]):
            df = self.data[symbol].sort_values("Date")
            if "Cumulative_Return" in df.columns:
                label = symbol.replace(".NS", "")
                ax1.plot(df["Date"], df["Cumulative_Return"],
                        color=PALETTE[i % len(PALETTE)], linewidth=1, label=label)
        ax1.set_title("Cumulative Returns (%)", fontweight="bold")
        ax1.legend(fontsize=6, framealpha=0.3, ncol=2)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, fontsize=7)

        # ── Panel 2: RSI Distribution ──
        ax2 = fig.add_subplot(gs[0, 1])
        rsi_values = []
        rsi_labels = []
        for symbol in self.stock_symbols:
            df = self.data[symbol]
            if "RSI" in df.columns:
                latest = df["RSI"].dropna()
                if len(latest) > 0:
                    rsi_values.append(latest.iloc[-1])
                    rsi_labels.append(symbol.replace(".NS", ""))

        if rsi_values:
            colors = [
                ACCENT_RED if v > 70 else ACCENT_GREEN if v < 30 else ACCENT_BLUE
                for v in rsi_values
            ]
            bars = ax2.bar(rsi_labels, rsi_values, color=colors, alpha=0.85)
            ax2.axhline(y=70, color=ACCENT_RED, linestyle="--", alpha=0.5, linewidth=0.8)
            ax2.axhline(y=30, color=ACCENT_GREEN, linestyle="--", alpha=0.5, linewidth=0.8)
            ax2.set_title("Current RSI Levels", fontweight="bold")
            ax2.set_ylim(0, 100)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, fontsize=7)

        # ── Panel 3: Sharpe Ratios ──
        ax3 = fig.add_subplot(gs[0, 2])
        profile = self.analysis.get("risk_return_profile")
        if profile is not None and not profile.empty:
            prof_sorted = profile.sort_values("Sharpe_Ratio", ascending=True)
            colors = [ACCENT_GREEN if v > 0.5 else ACCENT_YELLOW if v > 0 else ACCENT_RED
                      for v in prof_sorted["Sharpe_Ratio"]]
            ax3.barh(prof_sorted["Symbol"], prof_sorted["Sharpe_Ratio"], color=colors, alpha=0.85)
            ax3.axvline(x=0, color="white", linewidth=0.5, alpha=0.3)
            ax3.set_title("Sharpe Ratios", fontweight="bold")

        # ── Panel 4: Max Drawdowns ──
        ax4 = fig.add_subplot(gs[1, 0])
        drawdowns = []
        for symbol in self.stock_symbols:
            df = self.data[symbol]
            if "Drawdown" in df.columns:
                min_dd = df["Drawdown"].min() * 100
                drawdowns.append({
                    "Symbol": symbol.replace(".NS", ""),
                    "Max_Drawdown": min_dd,
                })
        if drawdowns:
            dd_df = pd.DataFrame(drawdowns).sort_values("Max_Drawdown", ascending=True)
            ax4.barh(dd_df["Symbol"], dd_df["Max_Drawdown"], color=ACCENT_RED, alpha=0.85)
            ax4.set_title("Maximum Drawdowns (%)", fontweight="bold")

        # ── Panel 5: Volume Heatmap (last 20 days, top 8 stocks) ──
        ax5 = fig.add_subplot(gs[1, 1])
        vol_data = {}
        for symbol in self.stock_symbols[:8]:
            df = self.data[symbol].sort_values("Date").tail(20)
            vol_data[symbol.replace(".NS", "")] = df.set_index("Date")["Volume"].values

        if vol_data:
            min_len = min(len(v) for v in vol_data.values())
            vol_matrix = pd.DataFrame(
                {k: v[:min_len] for k, v in vol_data.items()}
            ).T
            # Normalize each stock's volume
            vol_norm = vol_matrix.div(vol_matrix.max(axis=1), axis=0)
            sns.heatmap(
                vol_norm, cmap="YlOrRd", ax=ax5, cbar_kws={"shrink": 0.5},
                xticklabels=False, yticklabels=True,
            )
            ax5.set_title("Volume Heatmap (Last 20 Days)", fontweight="bold")

        # ── Panel 6: Momentum Score Distribution ──
        ax6 = fig.add_subplot(gs[1, 2])
        mom_scores = []
        for symbol in self.stock_symbols:
            df = self.data[symbol]
            if "Momentum_Score" in df.columns:
                latest = df["Momentum_Score"].dropna()
                if len(latest) > 0:
                    mom_scores.append({
                        "Symbol": symbol.replace(".NS", ""),
                        "Momentum": latest.iloc[-1],
                    })
        if mom_scores:
            mom_df = pd.DataFrame(mom_scores).sort_values("Momentum", ascending=True)
            colors = [ACCENT_GREEN if v > 0 else ACCENT_RED for v in mom_df["Momentum"]]
            ax6.barh(mom_df["Symbol"], mom_df["Momentum"], color=colors, alpha=0.85)
            ax6.axvline(x=0, color="white", linewidth=0.5, alpha=0.3)
            ax6.set_title("Momentum Scores", fontweight="bold")

        fig.suptitle(
            "📊 Stock Intelligence Dashboard",
            fontsize=20, fontweight="bold", y=1.01, color=ACCENT_BLUE,
        )
        fig.tight_layout()
        self._save_chart(fig, "dashboard_summary")
