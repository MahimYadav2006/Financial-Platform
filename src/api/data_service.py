"""
Data Service Layer
===================
Loads pre-processed CSV data from Task 1, indexes it in memory,
and provides high-level methods consumed by the API routes.

Design Decisions:
    - Loads data ONCE at startup (singleton pattern) → sub-ms response times
    - All heavy computation was already done in Task 1; this layer only reads & queries
    - Maintains both enriched per-stock DataFrames and pre-computed report DataFrames
    - Symbol lookup is case-insensitive and works with/without .NS suffix
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    OUTPUT_REPORTS_DIR,
    STOCK_UNIVERSE,
    SECTOR_MAP,
    BENCHMARK_SYMBOL,
    TRADING_DAYS_PER_YEAR,
    RISK_FREE_RATE,
)

logger = logging.getLogger(__name__)


class DataService:
    """
    Singleton-style service that holds all processed stock data in memory.

    Loads:
        - all_stocks_enriched.csv  → per-day data with 60+ features
        - Report CSVs from output/reports/ → pre-computed analytics
    """

    _instance: Optional["DataService"] = None

    def __new__(cls) -> "DataService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._stocks: dict[str, pd.DataFrame] = {}  # symbol → DataFrame
        self._reports: dict[str, pd.DataFrame] = {}  # report_name → DataFrame
        self._symbol_map: dict[str, str] = {}  # clean_symbol.upper() → full_symbol
        self._loaded = False

    # ──────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────
    def load(self) -> None:
        """Load all processed data into memory."""
        enriched_path = DATA_PROCESSED_DIR / "all_stocks_enriched.csv"

        if not enriched_path.exists():
            logger.warning(f"Enriched data not found at {enriched_path}")
            logger.warning("Run the Task 1 pipeline first: python main.py")
            return

        logger.info("Loading enriched dataset...")
        df = pd.read_csv(enriched_path, parse_dates=["Date"])

        # Split into per-symbol DataFrames (much faster queries)
        for symbol, group in df.groupby("Symbol"):
            group = group.sort_values("Date").reset_index(drop=True)
            self._stocks[symbol] = group

            # Build lookup map:  RELIANCE → RELIANCE.NS, etc.
            clean = self._clean_symbol(symbol)
            self._symbol_map[clean] = symbol

        # Also load pre-computed report CSVs
        report_files = [
            "stock_summary", "risk_return_profile", "beta_analysis",
            "top_gainers_losers", "sector_performance", "correlation_matrix",
            "volatility_ranking", "sector_correlation",
        ]
        for name in report_files:
            path = OUTPUT_REPORTS_DIR / f"{name}.csv"
            if path.exists():
                self._reports[name] = pd.read_csv(path)

        self._loaded = True
        logger.info(
            f"DataService ready: {len(self._stocks)} stocks, "
            f"{len(self._reports)} reports loaded."
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ──────────────────────────────────────────────────────────
    # Symbol Resolution
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        """Normalize symbol for case-insensitive lookup."""
        return symbol.upper().replace(".NS", "").replace("^", "")

    def resolve_symbol(self, user_input: str) -> Optional[str]:
        """
        Resolve a user-supplied symbol to the internal full symbol.

        Handles: RELIANCE, reliance, RELIANCE.NS, Reliance, etc.
        Returns: "RELIANCE.NS" or None
        """
        clean = self._clean_symbol(user_input)
        return self._symbol_map.get(clean)

    def get_all_symbols(self) -> list[str]:
        """Return list of available stock symbols (excluding benchmark)."""
        return [s for s in self._stocks if s != BENCHMARK_SYMBOL]

    # ──────────────────────────────────────────────────────────
    # /companies
    # ──────────────────────────────────────────────────────────
    def get_companies(self) -> list[dict]:
        """Return overview info for all companies."""
        companies = []
        for symbol in self.get_all_symbols():
            df = self._stocks[symbol]
            latest = df.iloc[-1]

            companies.append({
                "symbol": self._clean_symbol(symbol),
                "name": latest.get("Company", symbol),
                "sector": SECTOR_MAP.get(symbol, "Unknown"),
                "latest_close": round(float(latest["Close"]), 2),
                "high_52w": round(float(latest.get("High_52W", 0)), 2),
                "low_52w": round(float(latest.get("Low_52W", 0)), 2),
                "pct_from_52w_high": round(float(latest.get("Pct_From_52W_High", 0)), 2),
                "data_points": len(df),
            })

        return sorted(companies, key=lambda c: c["symbol"])

    # ──────────────────────────────────────────────────────────
    # /data/{symbol}
    # ──────────────────────────────────────────────────────────
    def get_stock_data(self, symbol: str, days: int = 30) -> Optional[dict]:
        """
        Return last N days of stock data for a symbol.

        Returns dict with metadata + list of daily records.
        """
        full_symbol = self.resolve_symbol(symbol)
        if full_symbol is None:
            return None

        df = self._stocks[full_symbol].copy()
        df = df.sort_values("Date")

        # Take last N trading days
        df_slice = df.tail(days)

        records = []
        for _, row in df_slice.iterrows():
            records.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
                "daily_return_pct": self._safe_round(row.get("Daily_Return_Pct")),
                "sma_7": self._safe_round(row.get("SMA_7")),
                "rsi": self._safe_round(row.get("RSI")),
                "macd_histogram": self._safe_round(row.get("MACD_Histogram")),
                "bollinger_pct_b": self._safe_round(row.get("BB_PctB")),
                "sentiment_index": self._safe_round(row.get("Sentiment_Index")),
            })

        latest = df.iloc[-1]
        return {
            "symbol": self._clean_symbol(full_symbol),
            "company": str(latest.get("Company", full_symbol)),
            "sector": SECTOR_MAP.get(full_symbol, "Unknown"),
            "period_start": df_slice.iloc[0]["Date"].strftime("%Y-%m-%d"),
            "period_end": df_slice.iloc[-1]["Date"].strftime("%Y-%m-%d"),
            "trading_days": len(records),
            "data": records,
        }

    # ──────────────────────────────────────────────────────────
    # /summary/{symbol}
    # ──────────────────────────────────────────────────────────
    def get_stock_summary(self, symbol: str) -> Optional[dict]:
        """
        Return comprehensive 52-week summary + technical snapshot.
        """
        full_symbol = self.resolve_symbol(symbol)
        if full_symbol is None:
            return None

        df = self._stocks[full_symbol].sort_values("Date")
        latest = df.iloc[-1]

        # 52-week average close (last 252 trading days)
        last_252 = df.tail(252)
        avg_close_52w = float(last_252["Close"].mean())

        # Returns
        return_30d = self._compute_period_return(df, 30)
        return_ytd = self._compute_ytd_return(df)

        # Risk report lookup
        risk = self._get_risk_for_symbol(full_symbol)
        beta_data = self._get_beta_for_symbol(full_symbol)

        clean_sym = self._clean_symbol(full_symbol)

        return {
            "symbol": clean_sym,
            "company": str(latest.get("Company", full_symbol)),
            "sector": SECTOR_MAP.get(full_symbol, "Unknown"),
            "latest_close": round(float(latest["Close"]), 2),
            "high_52w": round(float(latest.get("High_52W", 0)), 2),
            "low_52w": round(float(latest.get("Low_52W", 0)), 2),
            "avg_close_52w": round(avg_close_52w, 2),
            "pct_from_52w_high": round(float(latest.get("Pct_From_52W_High", 0)), 2),
            "pct_from_52w_low": round(float(latest.get("Pct_From_52W_Low", 0)), 2),
            "position_in_52w_range": round(float(latest.get("Position_In_52W_Range", 50)), 2),
            "return_30d_pct": return_30d,
            "return_ytd_pct": return_ytd,
            "technicals": {
                "rsi": self._safe_round(latest.get("RSI")),
                "rsi_signal": self._safe_str(latest.get("RSI_Signal")),
                "macd_line": self._safe_round(latest.get("MACD_Line")),
                "macd_signal": self._safe_round(latest.get("MACD_Signal")),
                "macd_histogram": self._safe_round(latest.get("MACD_Histogram")),
                "sma_7": self._safe_round(latest.get("SMA_7")),
                "sma_20": self._safe_round(latest.get("SMA_20")),
                "sma_50": self._safe_round(latest.get("SMA_50")),
                "sma_200": self._safe_round(latest.get("SMA_200")),
                "bollinger_upper": self._safe_round(latest.get("BB_Upper")),
                "bollinger_lower": self._safe_round(latest.get("BB_Lower")),
                "atr": self._safe_round(latest.get("ATR")),
                "atr_pct": self._safe_round(latest.get("ATR_Pct")),
            },
            "risk": {
                "annual_return_pct": risk.get("Annual_Return_Pct"),
                "annual_volatility_pct": risk.get("Annual_Volatility_Pct"),
                "sharpe_ratio": risk.get("Sharpe_Ratio"),
                "sortino_ratio": risk.get("Sortino_Ratio"),
                "max_drawdown_pct": risk.get("Max_Drawdown_Pct"),
                "calmar_ratio": risk.get("Calmar_Ratio"),
                "var_95_daily_pct": risk.get("VaR_95_Daily_Pct"),
                "beta": beta_data.get("Beta"),
                "alpha_annual_pct": beta_data.get("Alpha_Annual"),
            },
            "custom_metrics": {
                "garman_klass_volatility": self._safe_round(latest.get("GK_Volatility")),
                "momentum_score": self._safe_round(latest.get("Momentum_Score")),
                "sentiment_index": self._safe_round(latest.get("Sentiment_Index")),
                "sentiment_label": self._safe_str(latest.get("Sentiment_Label")),
                "trend_strength": self._safe_round(latest.get("Trend_Strength")),
                "rolling_sharpe": self._safe_round(latest.get("Rolling_Sharpe")),
                "money_flow": self._safe_str(latest.get("AD_Trend")),
            },
        }

    # ──────────────────────────────────────────────────────────
    # /compare
    # ──────────────────────────────────────────────────────────
    def compare_stocks(self, symbol1: str, symbol2: str) -> Optional[dict]:
        """
        Head-to-head comparison of two stocks.
        Returns performance stats, correlation, and an algorithmic verdict.
        """
        full1 = self.resolve_symbol(symbol1)
        full2 = self.resolve_symbol(symbol2)

        if full1 is None or full2 is None:
            return None

        df1 = self._stocks[full1].sort_values("Date")
        df2 = self._stocks[full2].sort_values("Date")

        entry1 = self._build_compare_entry(full1, df1)
        entry2 = self._build_compare_entry(full2, df2)

        # Correlation computation
        correlation = self._compute_correlation(df1, df2)

        # Verdict
        verdict = self._compute_verdict(entry1, entry2, correlation)

        # Normalized price history for charting
        price_history = self._build_normalized_prices(df1, df2, full1, full2)

        return {
            "stock1": entry1,
            "stock2": entry2,
            "correlation": correlation,
            "verdict": verdict,
            "price_history": price_history,
        }

    def _build_compare_entry(self, full_symbol: str, df: pd.DataFrame) -> dict:
        """Build a comparison entry for one stock."""
        latest = df.iloc[-1]
        last_252 = df.tail(252)
        avg_close = float(last_252["Close"].mean())

        risk = self._get_risk_for_symbol(full_symbol)
        beta_data = self._get_beta_for_symbol(full_symbol)
        gl = self._get_gainers_losers_for_symbol(full_symbol)

        return {
            "symbol": self._clean_symbol(full_symbol),
            "company": str(latest.get("Company", full_symbol)),
            "sector": SECTOR_MAP.get(full_symbol, "Unknown"),
            "latest_close": round(float(latest["Close"]), 2),
            "high_52w": round(float(latest.get("High_52W", 0)), 2),
            "low_52w": round(float(latest.get("Low_52W", 0)), 2),
            "avg_close_52w": round(avg_close, 2),
            "return_30d_pct": gl.get("Return_30D_Pct"),
            "return_ytd_pct": gl.get("Return_YTD_Pct"),
            "return_total_pct": gl.get("Return_Total_Pct"),
            "annual_return_pct": risk.get("Annual_Return_Pct"),
            "annual_volatility_pct": risk.get("Annual_Volatility_Pct"),
            "sharpe_ratio": risk.get("Sharpe_Ratio"),
            "max_drawdown_pct": risk.get("Max_Drawdown_Pct"),
            "beta": beta_data.get("Beta"),
            "rsi": self._safe_round(latest.get("RSI")),
            "sentiment_index": self._safe_round(latest.get("Sentiment_Index")),
            "momentum_score": self._safe_round(latest.get("Momentum_Score")),
            "trend_strength": self._safe_round(latest.get("Trend_Strength")),
        }

    def _compute_correlation(self, df1: pd.DataFrame, df2: pd.DataFrame) -> dict:
        """Compute return and price correlation between two stocks."""
        # Align on date
        merged = pd.merge(
            df1[["Date", "Close", "CC_Return"]],
            df2[["Date", "Close", "CC_Return"]],
            on="Date", suffixes=("_1", "_2"), how="inner",
        )

        if len(merged) < 10:
            return {
                "return_correlation": None,
                "price_correlation": None,
                "correlation_strength": "Insufficient data",
            }

        ret_corr = float(merged["CC_Return_1"].corr(merged["CC_Return_2"]))
        price_corr = float(merged["Close_1"].corr(merged["Close_2"]))

        # Human-readable interpretation
        abs_corr = abs(ret_corr)
        if abs_corr >= 0.8:
            strength = "Very Strong"
        elif abs_corr >= 0.6:
            strength = "Strong"
        elif abs_corr >= 0.4:
            strength = "Moderate"
        elif abs_corr >= 0.2:
            strength = "Weak"
        else:
            strength = "Very Weak / Uncorrelated"

        if ret_corr < 0:
            strength = f"Negative {strength}"

        return {
            "return_correlation": round(ret_corr, 4),
            "price_correlation": round(price_corr, 4),
            "correlation_strength": strength,
        }

    def _compute_verdict(self, entry1: dict, entry2: dict, corr: dict) -> dict:
        """Generate a comparison verdict with reasoning."""
        sym1, sym2 = entry1["symbol"], entry2["symbol"]

        # Point-based scoring
        scores = {sym1: 0, sym2: 0}
        reasons = []

        # Sharpe
        s1 = entry1.get("sharpe_ratio") or 0
        s2 = entry2.get("sharpe_ratio") or 0
        sharpe_winner = sym1 if s1 > s2 else sym2
        scores[sharpe_winner] += 2
        reasons.append(f"{sharpe_winner} has better risk-adjusted returns (Sharpe: {max(s1,s2):.2f} vs {min(s1,s2):.2f})")

        # Returns
        r1 = entry1.get("annual_return_pct") or 0
        r2 = entry2.get("annual_return_pct") or 0
        ret_winner = sym1 if r1 > r2 else sym2
        scores[ret_winner] += 1

        # Lower risk (lower volatility is better)
        v1 = entry1.get("annual_volatility_pct") or 999
        v2 = entry2.get("annual_volatility_pct") or 999
        risk_winner = sym1 if v1 < v2 else sym2
        scores[risk_winner] += 1
        reasons.append(f"{risk_winner} has lower volatility ({min(v1,v2):.1f}% vs {max(v1,v2):.1f}%)")

        # Momentum
        m1 = entry1.get("momentum_score") or 50
        m2 = entry2.get("momentum_score") or 50
        mom_leader = sym1 if m1 > m2 else sym2
        scores[mom_leader] += 1

        # Max drawdown (less negative is better)
        dd1 = entry1.get("max_drawdown_pct") or -100
        dd2 = entry2.get("max_drawdown_pct") or -100
        dd_winner = sym1 if dd1 > dd2 else sym2
        scores[dd_winner] += 1

        # Overall
        overall = sym1 if scores[sym1] >= scores[sym2] else sym2
        if scores[sym1] == scores[sym2]:
            overall_text = f"Both {sym1} and {sym2} are closely matched — it's a tie on our scoring model."
        else:
            overall_text = (
                f"{overall} leads with a score of {scores[overall]} vs "
                f"{scores[sym1 if overall == sym2 else sym2]}. "
                f"It offers a better balance of risk and return."
            )

        return {
            "sharpe_winner": sharpe_winner,
            "return_winner": ret_winner,
            "lower_risk": risk_winner,
            "momentum_leader": mom_leader,
            "overall_verdict": overall_text,
            "reasoning": " | ".join(reasons),
        }

    def _build_normalized_prices(
        self, df1: pd.DataFrame, df2: pd.DataFrame,
        sym1: str, sym2: str
    ) -> dict:
        """Build base-100 normalized price series for charting."""
        merged = pd.merge(
            df1[["Date", "Close"]],
            df2[["Date", "Close"]],
            on="Date", suffixes=("_1", "_2"), how="inner",
        ).sort_values("Date")

        if merged.empty:
            return {"dates": [], self._clean_symbol(sym1): [], self._clean_symbol(sym2): []}

        base1 = merged["Close_1"].iloc[0]
        base2 = merged["Close_2"].iloc[0]

        return {
            "dates": merged["Date"].dt.strftime("%Y-%m-%d").tolist(),
            self._clean_symbol(sym1): (merged["Close_1"] / base1 * 100).round(2).tolist(),
            self._clean_symbol(sym2): (merged["Close_2"] / base2 * 100).round(2).tolist(),
        }

    # ──────────────────────────────────────────────────────────
    # Extra Endpoints
    # ──────────────────────────────────────────────────────────
    def get_sectors(self) -> list[dict]:
        """Return sector performance data."""
        report = self._reports.get("sector_performance")
        if report is None:
            return []
        return report.to_dict(orient="records")

    def get_top_movers(self, n: int = 5) -> dict:
        """Return top gainers and losers (30-day returns)."""
        report = self._reports.get("top_gainers_losers")
        if report is None:
            return {"gainers": [], "losers": []}

        top = report.nlargest(n, "Return_30D_Pct")
        bottom = report.nsmallest(n, "Return_30D_Pct")

        def _to_mover(row) -> dict:
            return {
                "symbol": row["Symbol"],
                "company": row["Company"],
                "sector": row["Sector"],
                "return_30d_pct": round(row["Return_30D_Pct"], 2),
                "latest_close": round(row["Latest_Close"], 2),
            }

        return {
            "gainers": [_to_mover(row) for _, row in top.iterrows()],
            "losers": [_to_mover(row) for _, row in bottom.iterrows()],
        }

    def get_correlation_matrix(self) -> Optional[dict]:
        """Return correlation matrix."""
        report = self._reports.get("correlation_matrix")
        if report is None:
            return None

        # First column is the index (symbol names)
        symbols = report.iloc[:, 0].tolist()
        matrix = report.iloc[:, 1:].values.tolist()

        # Find most and least correlated pairs
        most_corr = {"pair": "", "value": -1}
        least_corr = {"pair": "", "value": 2}

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                val = float(report.iloc[i, j + 1])
                if val > most_corr["value"]:
                    most_corr = {"pair": f"{symbols[i]} ↔ {symbols[j]}", "value": round(val, 4)}
                if val < least_corr["value"]:
                    least_corr = {"pair": f"{symbols[i]} ↔ {symbols[j]}", "value": round(val, 4)}

        return {
            "symbols": symbols,
            "matrix": matrix,
            "most_correlated": most_corr,
            "least_correlated": least_corr,
        }

    # ──────────────────────────────────────────────────────────
    # /predict/{symbol}
    # ──────────────────────────────────────────────────────────
    def predict_stock(self, symbol: str, lookback_days: int = 30, predict_days: int = 7) -> Optional[dict]:
        """
        Uses simple linear regression (via scipy) on the closing prices of the 
        last `lookback_days` to predict the next `predict_days`.
        """
        from scipy import stats
        import datetime

        full_symbol = self.resolve_symbol(symbol)
        if full_symbol is None:
            return None

        df = self._stocks[full_symbol].sort_values("Date")
        if len(df) < lookback_days:
            lookback_days = len(df)

        last_n = df.tail(lookback_days).copy()
        
        # x is days from 0 to lookback_days - 1
        x = np.arange(lookback_days)
        y = last_n["Close"].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        last_date = pd.to_datetime(last_n["Date"].iloc[-1])
        
        predictions = []
        for i in range(1, predict_days + 1):
            pred_x = lookback_days - 1 + i
            pred_y = slope * pred_x + intercept
            pred_date = last_date + datetime.timedelta(days=i)
            # Skip weekends (simple approximation)
            while pred_date.weekday() >= 5:
                pred_date += datetime.timedelta(days=1)

            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "predicted_close": round(float(pred_y), 2)
            })

        latest = df.iloc[-1]
        company = str(latest.get("Company", full_symbol))
        
        trend = "Neutral"
        if slope > 0:
            trend = "Up"
        elif slope < 0:
            trend = "Down"

        return {
            "symbol": self._clean_symbol(full_symbol),
            "company": company,
            "model_type": "Linear Regression",
            "historical_days_used": lookback_days,
            "prediction_days": predict_days,
            "predictions": predictions,
            "r_squared": round(float(r_value ** 2), 4),
            "trend_direction": trend
        }

    # ──────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────
    def _get_risk_for_symbol(self, full_symbol: str) -> dict:
        """Lookup risk-return profile from pre-computed reports."""
        report = self._reports.get("risk_return_profile")
        if report is None:
            return {}
        clean = self._clean_symbol(full_symbol)
        match = report[report["Symbol"] == clean]
        if match.empty:
            return {}
        row = match.iloc[0]
        return {k: self._safe_round(v) for k, v in row.to_dict().items() if k != "Symbol"}

    def _get_beta_for_symbol(self, full_symbol: str) -> dict:
        """Lookup beta analysis from pre-computed reports."""
        report = self._reports.get("beta_analysis")
        if report is None:
            return {}
        clean = self._clean_symbol(full_symbol)
        match = report[report["Symbol"] == clean]
        if match.empty:
            return {}
        row = match.iloc[0]
        return {k: self._safe_round(v) for k, v in row.to_dict().items() if k != "Symbol"}

    def _get_gainers_losers_for_symbol(self, full_symbol: str) -> dict:
        """Lookup return data from pre-computed reports."""
        report = self._reports.get("top_gainers_losers")
        if report is None:
            return {}
        clean = self._clean_symbol(full_symbol)
        match = report[report["Symbol"] == clean]
        if match.empty:
            return {}
        row = match.iloc[0]
        return {k: self._safe_round(v) for k, v in row.to_dict().items() if k != "Symbol"}

    def _compute_period_return(self, df: pd.DataFrame, days: int) -> Optional[float]:
        """Compute simple return over last N trading days."""
        if len(df) < days:
            return None
        start_price = df.iloc[-days]["Close"]
        end_price = df.iloc[-1]["Close"]
        if start_price == 0:
            return None
        return round(((end_price - start_price) / start_price) * 100, 2)

    def _compute_ytd_return(self, df: pd.DataFrame) -> Optional[float]:
        """Compute year-to-date return."""
        current_year = df["Date"].max().year
        ytd = df[df["Date"].dt.year == current_year]
        if len(ytd) < 2:
            return None
        start_price = ytd.iloc[0]["Close"]
        end_price = ytd.iloc[-1]["Close"]
        if start_price == 0:
            return None
        return round(((end_price - start_price) / start_price) * 100, 2)

    @staticmethod
    def _safe_round(val, decimals: int = 2) -> Optional[float]:
        """Round numeric values, return None for NaN/missing."""
        if val is None:
            return None
        try:
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return None
            return round(f, decimals)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_str(val) -> Optional[str]:
        """Convert to string if non-null."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val)
        return s if s and s.lower() != "nan" else None
