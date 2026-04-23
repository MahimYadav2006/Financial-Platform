"""
Cross-Stock Analysis Module
============================
Performs portfolio-level and cross-stock analytics that require
data from multiple stocks simultaneously.

Key analyses:
    - Correlation matrix between all stocks
    - Sector-level performance aggregation
    - Beta calculation against NIFTY 50 benchmark
    - Top gainers / losers identification
    - Risk-return profile for each stock
    - Sector rotation analysis
    - Portfolio diversification insights

These metrics go beyond single-stock indicators and provide
portfolio-level intelligence that demonstrates financial acumen.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    SECTOR_MAP,
    BENCHMARK_SYMBOL,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    ROLLING_BETA_WINDOW,
    DATA_PROCESSED_DIR,
    OUTPUT_REPORTS_DIR,
)

logger = logging.getLogger(__name__)


class CrossStockAnalyzer:
    """
    Performs cross-stock analysis requiring data from multiple securities.

    Usage:
        analyzer = CrossStockAnalyzer(enriched_data)
        analyzer.run_all_analyses()
        report = analyzer.get_full_report()
    """

    def __init__(self, enriched_data: dict[str, pd.DataFrame]):
        """
        Args:
            enriched_data: Dictionary mapping symbol -> enriched DataFrame.
        """
        self.data = enriched_data
        self.stock_symbols = [s for s in enriched_data if s != BENCHMARK_SYMBOL]
        self.benchmark_data = enriched_data.get(BENCHMARK_SYMBOL)
        self.results: dict[str, pd.DataFrame] = {}

    def run_all_analyses(self) -> dict[str, pd.DataFrame]:
        """Execute all cross-stock analyses."""
        logger.info(
            f"\n{'='*60}\n"
            f"  CROSS-STOCK ANALYSIS\n"
            f"  Analyzing {len(self.stock_symbols)} stocks\n"
            f"{'='*60}\n"
        )

        self.results["correlation_matrix"] = self._compute_correlation_matrix()
        self.results["beta_analysis"] = self._compute_beta()
        self.results["risk_return_profile"] = self._compute_risk_return_profile()
        self.results["sector_performance"] = self._compute_sector_performance()
        self.results["top_gainers_losers"] = self._compute_gainers_losers()
        self.results["stock_summary"] = self._compute_stock_summary()
        self.results["sector_correlation"] = self._compute_sector_correlation()
        self.results["volatility_ranking"] = self._compute_volatility_ranking()

        # Save all results
        for name, df in self.results.items():
            filepath = OUTPUT_REPORTS_DIR / f"{name}.csv"
            df.to_csv(filepath)
            logger.info(f"  ✓ Saved: {name}.csv")

        return self.results

    def _build_returns_matrix(self, return_col: str = "CC_Return") -> pd.DataFrame:
        """Build a matrix of daily returns for all stocks (Date × Symbol)."""
        frames = []
        for symbol in self.stock_symbols:
            df = self.data[symbol][["Date", return_col]].copy()
            df = df.rename(columns={return_col: symbol})
            df = df.set_index("Date")
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=1).dropna()

    def _build_price_matrix(self) -> pd.DataFrame:
        """Build a matrix of closing prices for all stocks."""
        frames = []
        for symbol in self.stock_symbols:
            df = self.data[symbol][["Date", "Close"]].copy()
            df = df.rename(columns={"Close": symbol})
            df = df.set_index("Date")
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=1).dropna()

    def _compute_correlation_matrix(self) -> pd.DataFrame:
        """
        Compute pairwise Pearson correlation of daily returns.

        High correlation → stocks move together (low diversification benefit)
        Low/negative correlation → portfolio diversification opportunity
        """
        logger.info("  Computing correlation matrix...")
        returns_matrix = self._build_returns_matrix()

        if returns_matrix.empty:
            return pd.DataFrame()

        corr = returns_matrix.corr()

        # Replace ticker suffixes for readability
        clean_names = {s: s.replace(".NS", "") for s in corr.columns}
        corr = corr.rename(index=clean_names, columns=clean_names)

        logger.info(f"    Matrix size: {corr.shape[0]}×{corr.shape[1]}")

        # Find most and least correlated pairs
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], corr.iloc[i, j]))

        pairs.sort(key=lambda x: x[2], reverse=True)
        if pairs:
            top = pairs[0]
            bottom = pairs[-1]
            logger.info(f"    Most correlated:  {top[0]} ↔ {top[1]} (r={top[2]:.3f})")
            logger.info(
                f"    Least correlated: {bottom[0]} ↔ {bottom[1]} (r={bottom[2]:.3f})"
            )

        return corr

    def _compute_beta(self) -> pd.DataFrame:
        """
        Compute Beta for each stock against the NIFTY 50 benchmark.

        Beta measures systematic risk:
            β > 1: More volatile than market (aggressive)
            β = 1: Moves with market
            β < 1: Less volatile than market (defensive)
            β < 0: Moves opposite to market (rare)
        """
        logger.info("  Computing Beta (vs NIFTY 50)...")

        if self.benchmark_data is None:
            logger.warning("    Benchmark data not available. Skipping Beta.")
            return pd.DataFrame()

        benchmark_returns = self.benchmark_data.set_index("Date")["CC_Return"].dropna()
        betas = []

        for symbol in self.stock_symbols:
            stock_returns = self.data[symbol].set_index("Date")["CC_Return"].dropna()

            # Align dates
            aligned = pd.concat(
                [stock_returns, benchmark_returns], axis=1, join="inner"
            )
            aligned.columns = ["Stock", "Benchmark"]
            aligned = aligned.dropna()

            if len(aligned) < 30:
                continue

            # Beta = Cov(stock, market) / Var(market)
            covariance = aligned["Stock"].cov(aligned["Benchmark"])
            market_variance = aligned["Benchmark"].var()
            beta = covariance / market_variance if market_variance > 0 else 0

            # Alpha (Jensen's Alpha)
            stock_annual_return = aligned["Stock"].mean() * TRADING_DAYS_PER_YEAR
            market_annual_return = aligned["Benchmark"].mean() * TRADING_DAYS_PER_YEAR
            alpha = stock_annual_return - (
                RISK_FREE_RATE + beta * (market_annual_return - RISK_FREE_RATE)
            )

            # R-squared
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                aligned["Benchmark"], aligned["Stock"]
            )

            company = self.data[symbol]["Company"].iloc[0]
            sector = SECTOR_MAP.get(symbol, "Unknown")

            betas.append(
                {
                    "Symbol": symbol.replace(".NS", ""),
                    "Company": company,
                    "Sector": sector,
                    "Beta": round(beta, 4),
                    "Alpha_Annual": round(alpha * 100, 2),  # as percentage
                    "R_Squared": round(r_value ** 2, 4),
                    "Classification": (
                        "Aggressive" if beta > 1.2
                        else "Moderate" if beta > 0.8
                        else "Defensive"
                    ),
                }
            )

        beta_df = pd.DataFrame(betas).sort_values("Beta", ascending=False)
        logger.info(f"    Computed Beta for {len(betas)} stocks")
        return beta_df

    def _compute_risk_return_profile(self) -> pd.DataFrame:
        """
        Compute annualized risk-return metrics for each stock.

        Creates a comprehensive risk profile including:
            - Annualized return and volatility
            - Sharpe ratio
            - Sortino ratio (penalizes only downside volatility)
            - Maximum drawdown
            - Calmar ratio (return / max drawdown)
            - Value at Risk (VaR) at 95% confidence
        """
        logger.info("  Computing risk-return profiles...")
        profiles = []

        for symbol in self.stock_symbols:
            df = self.data[symbol]
            returns = df["CC_Return"].dropna()

            if len(returns) < 20:
                continue

            # Annualized return
            annual_return = returns.mean() * TRADING_DAYS_PER_YEAR

            # Annualized volatility
            annual_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

            # Sharpe Ratio
            sharpe = (
                (annual_return - RISK_FREE_RATE) / annual_vol
                if annual_vol > 0
                else 0
            )

            # Sortino Ratio (uses only downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            sortino = (
                (annual_return - RISK_FREE_RATE) / downside_std
                if downside_std > 0
                else 0
            )

            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()

            # Calmar Ratio
            calmar = (
                annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            )

            # Value at Risk (95% confidence, historical method)
            var_95 = returns.quantile(0.05)

            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()

            company = df["Company"].iloc[0]
            sector = SECTOR_MAP.get(symbol, "Unknown")

            profiles.append(
                {
                    "Symbol": symbol.replace(".NS", ""),
                    "Company": company,
                    "Sector": sector,
                    "Annual_Return_Pct": round(annual_return * 100, 2),
                    "Annual_Volatility_Pct": round(annual_vol * 100, 2),
                    "Sharpe_Ratio": round(sharpe, 3),
                    "Sortino_Ratio": round(sortino, 3),
                    "Max_Drawdown_Pct": round(max_drawdown * 100, 2),
                    "Calmar_Ratio": round(calmar, 3),
                    "VaR_95_Daily_Pct": round(var_95 * 100, 2),
                    "Skewness": round(skewness, 3),
                    "Kurtosis": round(kurtosis, 3),
                }
            )

        profile_df = pd.DataFrame(profiles).sort_values(
            "Sharpe_Ratio", ascending=False
        )
        logger.info(f"    Profiled {len(profiles)} stocks")
        return profile_df

    def _compute_sector_performance(self) -> pd.DataFrame:
        """Aggregate performance metrics by sector."""
        logger.info("  Computing sector performance...")
        sector_data = []

        for sector in set(SECTOR_MAP.values()):
            sector_symbols = [
                s for s, sec in SECTOR_MAP.items()
                if sec == sector and s in self.data
            ]

            if not sector_symbols:
                continue

            sector_returns = []
            sector_vols = []
            latest_sentiments = []

            for symbol in sector_symbols:
                df = self.data[symbol]
                returns = df["CC_Return"].dropna()

                if len(returns) > 0:
                    sector_returns.append(returns.mean() * TRADING_DAYS_PER_YEAR)
                    sector_vols.append(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

                if "Sentiment_Index" in df.columns:
                    latest_val = df["Sentiment_Index"].dropna()
                    if len(latest_val) > 0:
                        latest_sentiments.append(latest_val.iloc[-1])

            sector_data.append(
                {
                    "Sector": sector,
                    "Num_Stocks": len(sector_symbols),
                    "Avg_Annual_Return_Pct": round(np.mean(sector_returns) * 100, 2)
                    if sector_returns else 0,
                    "Avg_Volatility_Pct": round(np.mean(sector_vols) * 100, 2)
                    if sector_vols else 0,
                    "Avg_Sentiment": round(np.mean(latest_sentiments), 1)
                    if latest_sentiments else 50,
                }
            )

        sector_df = pd.DataFrame(sector_data).sort_values(
            "Avg_Annual_Return_Pct", ascending=False
        )
        logger.info(f"    Analyzed {len(sector_data)} sectors")
        return sector_df

    def _compute_gainers_losers(self) -> pd.DataFrame:
        """Identify top gainers and losers (last 30 days and YTD)."""
        logger.info("  Identifying top gainers and losers...")
        results = []

        for symbol in self.stock_symbols:
            df = self.data[symbol].sort_values("Date")

            if len(df) < 30:
                continue

            # Last 30 days return
            last_30 = df.tail(30)
            ret_30d = (
                (last_30["Close"].iloc[-1] - last_30["Close"].iloc[0])
                / last_30["Close"].iloc[0]
                * 100
            )

            # YTD return (from start of current year)
            current_year = df["Date"].max().year
            ytd_data = df[df["Date"].dt.year == current_year]
            if len(ytd_data) > 1:
                ret_ytd = (
                    (ytd_data["Close"].iloc[-1] - ytd_data["Close"].iloc[0])
                    / ytd_data["Close"].iloc[0]
                    * 100
                )
            else:
                ret_ytd = 0

            # Total period return
            ret_total = (
                (df["Close"].iloc[-1] - df["Close"].iloc[0])
                / df["Close"].iloc[0]
                * 100
            )

            company = df["Company"].iloc[0]

            results.append(
                {
                    "Symbol": symbol.replace(".NS", ""),
                    "Company": company,
                    "Sector": SECTOR_MAP.get(symbol, "Unknown"),
                    "Return_30D_Pct": round(ret_30d, 2),
                    "Return_YTD_Pct": round(ret_ytd, 2),
                    "Return_Total_Pct": round(ret_total, 2),
                    "Latest_Close": round(df["Close"].iloc[-1], 2),
                }
            )

        return pd.DataFrame(results).sort_values("Return_30D_Pct", ascending=False)

    def _compute_stock_summary(self) -> pd.DataFrame:
        """Generate a comprehensive summary for each stock."""
        logger.info("  Generating stock summaries...")
        summaries = []

        for symbol in self.stock_symbols:
            df = self.data[symbol].sort_values("Date")
            latest = df.iloc[-1]

            summary = {
                "Symbol": symbol.replace(".NS", ""),
                "Company": latest.get("Company", symbol),
                "Sector": SECTOR_MAP.get(symbol, "Unknown"),
                "Latest_Close": round(latest["Close"], 2),
                "52W_High": round(latest.get("High_52W", 0), 2),
                "52W_Low": round(latest.get("Low_52W", 0), 2),
                "Pct_From_52W_High": round(latest.get("Pct_From_52W_High", 0), 2),
            }

            # Add latest indicator values
            for col in [
                "RSI", "Sentiment_Index", "Momentum_Score",
                "GK_Volatility", "Rolling_Sharpe", "Trend_Strength",
                "Drawdown_Pct",
            ]:
                if col in df.columns:
                    val = latest.get(col)
                    summary[col] = round(float(val), 2) if pd.notna(val) else None

            # Signal summary
            if "RSI_Signal" in df.columns:
                summary["RSI_Status"] = str(latest.get("RSI_Signal", ""))
            if "Sentiment_Label" in df.columns:
                summary["Sentiment_Status"] = str(latest.get("Sentiment_Label", ""))
            if "AD_Trend" in df.columns:
                summary["Money_Flow"] = str(latest.get("AD_Trend", ""))

            summaries.append(summary)

        return pd.DataFrame(summaries).sort_values("Symbol")

    def _compute_sector_correlation(self) -> pd.DataFrame:
        """Compute average correlation between sectors."""
        logger.info("  Computing sector correlations...")
        returns_matrix = self._build_returns_matrix()

        if returns_matrix.empty:
            return pd.DataFrame()

        # Map symbols to sectors
        sector_returns = {}
        for symbol in returns_matrix.columns:
            sector = SECTOR_MAP.get(symbol, "Unknown")
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(returns_matrix[symbol])

        # Average returns per sector (equal-weighted sector portfolios)
        sector_avg = {}
        for sector, ret_list in sector_returns.items():
            sector_avg[sector] = pd.concat(ret_list, axis=1).mean(axis=1)

        sector_df = pd.DataFrame(sector_avg)
        return sector_df.corr().round(3)

    def _compute_volatility_ranking(self) -> pd.DataFrame:
        """Rank stocks by multiple volatility metrics."""
        logger.info("  Computing volatility rankings...")
        rankings = []

        for symbol in self.stock_symbols:
            df = self.data[symbol]
            returns = df["CC_Return"].dropna()

            if len(returns) < 20:
                continue

            # Historical volatility (annualized std dev)
            hist_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

            # Garman-Klass volatility (latest)
            gk_vol = df["GK_Volatility"].dropna()
            latest_gk = gk_vol.iloc[-1] * 100 if len(gk_vol) > 0 else 0

            # ATR percentage (latest)
            atr_pct = df["ATR_Pct"].dropna()
            latest_atr = atr_pct.iloc[-1] if len(atr_pct) > 0 else 0

            # Max 1-day move
            max_up = returns.max() * 100
            max_down = returns.min() * 100

            company = df["Company"].iloc[0]

            rankings.append(
                {
                    "Symbol": symbol.replace(".NS", ""),
                    "Company": company,
                    "Sector": SECTOR_MAP.get(symbol, "Unknown"),
                    "Historical_Vol_Pct": round(hist_vol, 2),
                    "GK_Volatility_Pct": round(latest_gk, 2),
                    "ATR_Pct": round(latest_atr, 2),
                    "Max_Daily_Gain_Pct": round(max_up, 2),
                    "Max_Daily_Loss_Pct": round(max_down, 2),
                    "Risk_Level": (
                        "High" if hist_vol > 35
                        else "Medium" if hist_vol > 20
                        else "Low"
                    ),
                }
            )

        return pd.DataFrame(rankings).sort_values(
            "Historical_Vol_Pct", ascending=False
        )

    def get_full_report(self) -> dict[str, pd.DataFrame]:
        """Return all analysis results."""
        return self.results
