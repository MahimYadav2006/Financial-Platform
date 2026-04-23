"""
Feature Engineering Module
===========================
Computes all technical indicators, financial metrics, and custom analytics.

This is the heart of the analysis engine. It takes cleaned OHLCV data and
enriches it with a comprehensive suite of metrics organized into three tiers:

Tier 1 — Required Metrics (Assignment Spec):
    - Daily Return
    - 7-day Moving Average
    - 52-week High/Low

Tier 2 — Standard Technical Indicators:
    - Multiple Moving Averages (7, 20, 50, 200-day)
    - Exponential Moving Averages (EMA)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - Average True Range (ATR)
    - On-Balance Volume (OBV)

Tier 3 — Custom / Creative Metrics:
    - Garman-Klass Volatility (superior to simple std dev)
    - Rolling Sharpe Ratio (risk-adjusted returns)
    - Maximum Drawdown (risk metric)
    - Price Momentum Score (composite indicator)
    - Mock Sentiment Index (volume + momentum + volatility composite)
    - Trend Strength Index (custom ADX-inspired metric)
    - Gap Analysis (overnight price gaps)
    - Accumulation/Distribution Pressure

Design Decisions:
    - All calculations are vectorized with NumPy/Pandas for performance
    - NaN values at the start of rolling windows are expected and preserved
    - Each metric function is standalone for testability
    - The pipeline applies all metrics in a single pass per stock
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    MOVING_AVG_WINDOWS,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BOLLINGER_WINDOW,
    BOLLINGER_STD,
    ATR_PERIOD,
    ROLLING_SHARPE_WINDOW,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    DATA_PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Enriches cleaned stock data with technical indicators and custom metrics.

    Usage:
        engineer = FeatureEngineer()
        enriched_data = engineer.transform_all(cleaned_data)
    """

    def __init__(self):
        self.data: dict[str, pd.DataFrame] = {}
        self.metrics_summary: dict[str, dict] = {}

    def transform_all(
        self, cleaned_data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Apply the full feature engineering pipeline to all stocks.

        Args:
            cleaned_data: Dictionary mapping symbol -> cleaned DataFrame.

        Returns:
            Dictionary mapping symbol -> enriched DataFrame.
        """
        logger.info(
            f"\n{'='*60}\n"
            f"  FEATURE ENGINEERING\n"
            f"  Computing metrics for {len(cleaned_data)} stocks\n"
            f"{'='*60}\n"
        )

        for symbol, df in cleaned_data.items():
            enriched = self._compute_all_features(symbol, df.copy())
            self.data[symbol] = enriched

            # Track metrics summary
            self.metrics_summary[symbol] = {
                "total_features": len(enriched.columns),
                "new_features": len(enriched.columns) - len(df.columns),
                "rows": len(enriched),
            }

            logger.info(
                f"  ✓ {symbol}: +{self.metrics_summary[symbol]['new_features']} "
                f"features → {self.metrics_summary[symbol]['total_features']} total"
            )

        # Save enriched combined dataset
        if self.data:
            combined = pd.concat(self.data.values(), ignore_index=True)
            combined_path = DATA_PROCESSED_DIR / "all_stocks_enriched.csv"
            combined.to_csv(combined_path, index=False)
            logger.info(f"\n  Enriched dataset saved: {combined_path}")

        self._log_feature_summary()
        return self.data

    def _compute_all_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature computations in order."""
        # Ensure sorted by date
        df = df.sort_values("Date").reset_index(drop=True)

        # ── Tier 1: Required Metrics ──────────────────────────────
        df = self._add_daily_return(df)
        df = self._add_moving_averages(df)
        df = self._add_52_week_high_low(df)

        # ── Tier 2: Standard Technical Indicators ─────────────────
        df = self._add_ema(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_atr(df)
        df = self._add_obv(df)

        # ── Tier 3: Custom / Creative Metrics ─────────────────────
        df = self._add_garman_klass_volatility(df)
        df = self._add_rolling_sharpe(df)
        df = self._add_max_drawdown(df)
        df = self._add_momentum_score(df)
        df = self._add_sentiment_index(df)
        df = self._add_trend_strength(df)
        df = self._add_gap_analysis(df)
        df = self._add_accumulation_distribution(df)
        df = self._add_cumulative_return(df)

        return df

    # ══════════════════════════════════════════════════════════════
    #  TIER 1: REQUIRED METRICS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
        """
        Daily Return = (Close - Open) / Open

        Measures intraday price movement as a percentage.
        Also adds Close-to-Close return for inter-day analysis.
        """
        df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
        df["Daily_Return_Pct"] = df["Daily_Return"] * 100  # Percentage form

        # Close-to-Close return (more commonly used in finance)
        df["CC_Return"] = df["Close"].pct_change()

        # Log return (additive property, better for statistical analysis)
        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

        return df

    @staticmethod
    def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple Moving Averages for multiple windows.
        The 7-day MA is explicitly required; we add 20, 50, 200 for richer analysis.
        """
        for window in MOVING_AVG_WINDOWS:
            df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()

        # Volume Moving Average (for volume analysis)
        df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()

        return df

    @staticmethod
    def _add_52_week_high_low(df: pd.DataFrame) -> pd.DataFrame:
        """
        52-week (252 trading days) rolling High and Low.
        Also adds distance from 52-week high/low as a percentage.
        """
        window = 252  # Trading days in a year

        df["High_52W"] = df["High"].rolling(window=window, min_periods=1).max()
        df["Low_52W"] = df["Low"].rolling(window=window, min_periods=1).min()

        # Distance from 52-week high (always <= 0, shows how far price has fallen)
        df["Pct_From_52W_High"] = (
            (df["Close"] - df["High_52W"]) / df["High_52W"] * 100
        )

        # Distance from 52-week low (always >= 0, shows how far price has risen)
        df["Pct_From_52W_Low"] = (
            (df["Close"] - df["Low_52W"]) / df["Low_52W"] * 100
        )

        # 52-week range position (0 = at low, 100 = at high)
        range_size = df["High_52W"] - df["Low_52W"]
        df["Position_In_52W_Range"] = np.where(
            range_size > 0,
            ((df["Close"] - df["Low_52W"]) / range_size * 100),
            50.0,  # If range is 0, price hasn't moved
        )

        return df

    # ══════════════════════════════════════════════════════════════
    #  TIER 2: STANDARD TECHNICAL INDICATORS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _add_ema(df: pd.DataFrame) -> pd.DataFrame:
        """Exponential Moving Averages — react faster to recent price changes."""
        for window in [12, 26, 50]:
            df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
        return df

    @staticmethod
    def _add_rsi(df: pd.DataFrame) -> pd.DataFrame:
        """
        Relative Strength Index (RSI).

        RSI measures momentum on a scale of 0-100.
        - RSI > 70 → Overbought (potential sell signal)
        - RSI < 30 → Oversold (potential buy signal)

        Uses Wilder's smoothing method (exponential moving average).
        """
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()
        avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, min_periods=RSI_PERIOD).mean()

        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # RSI classification
        df["RSI_Signal"] = pd.cut(
            df["RSI"],
            bins=[0, 30, 45, 55, 70, 100],
            labels=["Oversold", "Weak", "Neutral", "Strong", "Overbought"],
            include_lowest=True,
        )

        return df

    @staticmethod
    def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence (MACD).

        MACD is a trend-following momentum indicator showing the relationship
        between two EMAs. The MACD line crossing above the signal line is
        considered bullish, and vice versa.
        """
        ema_fast = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()

        df["MACD_Line"] = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD_Line"].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df["MACD_Histogram"] = df["MACD_Line"] - df["MACD_Signal"]

        # MACD crossover signal
        df["MACD_Crossover"] = np.where(
            (df["MACD_Line"] > df["MACD_Signal"])
            & (df["MACD_Line"].shift(1) <= df["MACD_Signal"].shift(1)),
            1,  # Bullish crossover
            np.where(
                (df["MACD_Line"] < df["MACD_Signal"])
                & (df["MACD_Line"].shift(1) >= df["MACD_Signal"].shift(1)),
                -1,  # Bearish crossover
                0,   # No crossover
            ),
        )

        return df

    @staticmethod
    def _add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """
        Bollinger Bands — volatility-based envelope around the moving average.

        Price touching the upper band suggests overbought conditions,
        while touching the lower band suggests oversold conditions.
        Band squeeze (narrowing) often precedes a breakout.
        """
        sma = df["Close"].rolling(window=BOLLINGER_WINDOW).mean()
        std = df["Close"].rolling(window=BOLLINGER_WINDOW).std()

        df["BB_Upper"] = sma + (BOLLINGER_STD * std)
        df["BB_Middle"] = sma
        df["BB_Lower"] = sma - (BOLLINGER_STD * std)

        # %B — Where price is relative to the bands (0 = lower, 1 = upper)
        band_width = df["BB_Upper"] - df["BB_Lower"]
        df["BB_PctB"] = np.where(
            band_width > 0,
            (df["Close"] - df["BB_Lower"]) / band_width,
            0.5,
        )

        # Bandwidth — Measure of volatility (narrow = squeeze)
        df["BB_Width"] = np.where(
            df["BB_Middle"] > 0, band_width / df["BB_Middle"], 0
        )

        return df

    @staticmethod
    def _add_atr(df: pd.DataFrame) -> pd.DataFrame:
        """
        Average True Range (ATR) — Volatility indicator.

        True Range is the greatest of:
            - Current High - Current Low
            - |Current High - Previous Close|
            - |Current Low - Previous Close|

        ATR is the smoothed average of True Range.
        """
        high_low = df["High"] - df["Low"]
        high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
        low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()

        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(
            axis=1
        )
        df["True_Range"] = true_range
        df["ATR"] = true_range.rolling(window=ATR_PERIOD).mean()

        # ATR as percentage of price (normalized volatility)
        df["ATR_Pct"] = (df["ATR"] / df["Close"]) * 100

        return df

    @staticmethod
    def _add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        On-Balance Volume (OBV) — Volume-based momentum indicator.

        OBV adds volume on up-days and subtracts on down-days.
        Rising OBV confirms uptrend; divergence from price signals reversal.
        """
        direction = np.sign(df["Close"].diff())
        df["OBV"] = (df["Volume"] * direction).fillna(0).cumsum()
        df["OBV_SMA_20"] = df["OBV"].rolling(window=20).mean()

        return df

    # ══════════════════════════════════════════════════════════════
    #  TIER 3: CUSTOM / CREATIVE METRICS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _add_garman_klass_volatility(df: pd.DataFrame) -> pd.DataFrame:
        """
        Garman-Klass Volatility Estimator.

        Superior to simple standard deviation because it uses OHLC data
        rather than just closing prices, capturing intraday price dynamics.

        Formula: σ² = 0.5 * ln(H/L)² - (2ln2 - 1) * ln(C/O)²

        Reference: Garman & Klass (1980)
        """
        log_hl = np.log(df["High"] / df["Low"]) ** 2
        log_co = np.log(df["Close"] / df["Open"]) ** 2

        # Daily Garman-Klass variance
        gk_daily = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

        # 20-day rolling annualized volatility
        df["GK_Volatility"] = np.sqrt(
            gk_daily.rolling(window=20).mean() * TRADING_DAYS_PER_YEAR
        )

        # Volatility percentile rank over 252 days
        df["Volatility_Rank"] = (
            df["GK_Volatility"]
            .rolling(window=252, min_periods=20)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        )

        return df

    @staticmethod
    def _add_rolling_sharpe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling Sharpe Ratio — Risk-adjusted return metric.

        Sharpe = (Mean Return - Risk-Free Rate) / Std(Return)

        Annualized using √252 scaling.
        Higher Sharpe = better risk-adjusted performance.
        """
        daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR

        rolling_mean = df["CC_Return"].rolling(window=ROLLING_SHARPE_WINDOW).mean()
        rolling_std = df["CC_Return"].rolling(window=ROLLING_SHARPE_WINDOW).std()

        df["Rolling_Sharpe"] = np.where(
            rolling_std > 0,
            ((rolling_mean - daily_rf) / rolling_std) * np.sqrt(TRADING_DAYS_PER_YEAR),
            0,
        )

        return df

    @staticmethod
    def _add_max_drawdown(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling Maximum Drawdown — Worst peak-to-trough decline.

        Essential risk metric that shows the maximum loss an investor
        could have experienced over the rolling window.
        """
        # Cumulative peak
        df["Cumulative_Peak"] = df["Close"].cummax()

        # Current drawdown from peak
        df["Drawdown"] = (df["Close"] - df["Cumulative_Peak"]) / df["Cumulative_Peak"]
        df["Drawdown_Pct"] = df["Drawdown"] * 100

        # Rolling max drawdown (worst drawdown in last 252 days)
        df["Max_Drawdown_252"] = (
            df["Drawdown"].rolling(window=252, min_periods=1).min() * 100
        )

        return df

    @staticmethod
    def _add_momentum_score(df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite Momentum Score — Multi-timeframe momentum indicator.

        Combines returns over multiple periods (5, 10, 20, 60 days)
        into a single normalized score between -100 and +100.

        This is a custom metric that provides a holistic view of
        momentum across different time horizons.
        """
        periods = [5, 10, 20, 60]
        momentum_components = []

        for period in periods:
            ret = df["Close"].pct_change(periods=period)
            # Rank the return relative to its own history (percentile)
            ranked = ret.rolling(window=252, min_periods=period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
            momentum_components.append(ranked)

        # Weighted average: shorter-term gets slightly more weight
        weights = [0.3, 0.25, 0.25, 0.2]
        momentum = sum(w * m for w, m in zip(weights, momentum_components))

        # Scale to -100 to +100
        df["Momentum_Score"] = (momentum - 0.5) * 200

        return df

    @staticmethod
    def _add_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Mock Sentiment Index — Composite of price action and volume signals.

        This creative metric simulates market sentiment using a combination of:
            1. Price momentum (30% weight) — Are prices trending up?
            2. Volume trend (20% weight) — Is volume confirming the trend?
            3. Volatility regime (20% weight) — Low vol = complacent, high = fearful
            4. RSI positioning (15% weight) — Overbought/oversold extremes
            5. Price vs Moving Average (15% weight) — Bullish/bearish positioning

        Score ranges from 0 (Extreme Fear) to 100 (Extreme Greed),
        inspired by the CNN Fear & Greed Index.
        """
        # Component 1: Price momentum (20-day return percentile)
        returns_20d = df["Close"].pct_change(20)
        momentum_signal = returns_20d.rolling(252, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # Component 2: Volume trend (is volume above or below average?)
        vol_ratio = df["Volume"] / df["Volume"].rolling(50).mean()
        # High volume on up days = bullish, high volume on down days = bearish
        vol_signal = np.where(
            df["Close"] > df["Close"].shift(1),
            vol_ratio.clip(upper=2) / 2,       # Up day: higher volume → more bullish
            1 - vol_ratio.clip(upper=2) / 2,    # Down day: higher volume → more bearish
        )

        # Component 3: Volatility regime (inverse — low vol = complacent/greedy)
        if "GK_Volatility" in df.columns:
            vol_rank = df["GK_Volatility"].rolling(252, min_periods=20).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
            volatility_signal = 1 - vol_rank  # Invert: low vol = high sentiment
        else:
            volatility_signal = pd.Series(0.5, index=df.index)

        # Component 4: RSI positioning
        if "RSI" in df.columns:
            rsi_signal = df["RSI"] / 100
        else:
            rsi_signal = pd.Series(0.5, index=df.index)

        # Component 5: Price vs 50-day SMA
        if "SMA_50" in df.columns:
            sma_distance = (df["Close"] - df["SMA_50"]) / df["SMA_50"]
            sma_signal = sma_distance.clip(-0.1, 0.1) / 0.2 + 0.5
        else:
            sma_signal = pd.Series(0.5, index=df.index)

        # Weighted composite
        sentiment = (
            0.30 * momentum_signal
            + 0.20 * pd.Series(vol_signal, index=df.index)
            + 0.20 * volatility_signal
            + 0.15 * rsi_signal
            + 0.15 * sma_signal
        )

        # Scale to 0-100
        df["Sentiment_Index"] = (sentiment * 100).clip(0, 100)

        # Classification
        df["Sentiment_Label"] = pd.cut(
            df["Sentiment_Index"],
            bins=[0, 25, 40, 60, 75, 100],
            labels=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"],
            include_lowest=True,
        )

        return df

    @staticmethod
    def _add_trend_strength(df: pd.DataFrame) -> pd.DataFrame:
        """
        Trend Strength Index — Custom metric measuring trend conviction.

        Combines moving average alignment, price consistency, and volume
        confirmation into a single 0-100 score.

        High score = strong, confirmed trend (either direction).
        Low score = choppy, trendless market.
        """
        signals = []

        # Signal 1: MA Alignment (are MAs stacked in order?)
        if all(f"SMA_{w}" in df.columns for w in [7, 20, 50, 200]):
            bullish_stack = (
                (df["SMA_7"] > df["SMA_20"]).astype(float)
                + (df["SMA_20"] > df["SMA_50"]).astype(float)
                + (df["SMA_50"] > df["SMA_200"]).astype(float)
            ) / 3
            bearish_stack = (
                (df["SMA_7"] < df["SMA_20"]).astype(float)
                + (df["SMA_20"] < df["SMA_50"]).astype(float)
                + (df["SMA_50"] < df["SMA_200"]).astype(float)
            ) / 3
            ma_alignment = pd.concat([bullish_stack, bearish_stack], axis=1).max(axis=1)
            signals.append(ma_alignment)

        # Signal 2: Directional consistency (% of last 20 days in same direction)
        if "CC_Return" in df.columns:
            positive_days = (df["CC_Return"] > 0).rolling(20).mean()
            directional_consistency = (positive_days - 0.5).abs() * 2  # 0 when 50/50
            signals.append(directional_consistency)

        # Signal 3: Price vs Bollinger Band (strong trend = near outer bands)
        if "BB_PctB" in df.columns:
            # Distance from middle band, normalized
            bb_trend = (df["BB_PctB"] - 0.5).abs() * 2
            signals.append(bb_trend)

        if signals:
            df["Trend_Strength"] = (
                sum(signals) / len(signals) * 100
            ).clip(0, 100)
        else:
            df["Trend_Strength"] = 50.0

        return df

    @staticmethod
    def _add_gap_analysis(df: pd.DataFrame) -> pd.DataFrame:
        """
        Gap Analysis — Measures overnight price gaps.

        Gaps occur when the opening price differs significantly from
        the previous close, often due to after-hours news or events.

        Gap Up = Open > Previous Close (bullish overnight sentiment)
        Gap Down = Open < Previous Close (bearish overnight sentiment)
        """
        prev_close = df["Close"].shift(1)

        df["Gap"] = (df["Open"] - prev_close) / prev_close * 100
        df["Gap_Type"] = pd.cut(
            df["Gap"],
            bins=[-float("inf"), -1, -0.25, 0.25, 1, float("inf")],
            labels=["Large Gap Down", "Gap Down", "No Gap", "Gap Up", "Large Gap Up"],
        )

        # Rolling gap tendency (average gap over 20 days)
        df["Avg_Gap_20"] = df["Gap"].rolling(20).mean()

        return df

    @staticmethod
    def _add_accumulation_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """
        Accumulation/Distribution Pressure — Money flow indicator.

        Uses the Close Location Value (CLV) to determine if a stock
        is being accumulated (bought) or distributed (sold).

        CLV = [(Close - Low) - (High - Close)] / (High - Low)
        AD = CLV × Volume

        Positive AD = Accumulation (buying pressure dominates)
        Negative AD = Distribution (selling pressure dominates)
        """
        high_low_range = df["High"] - df["Low"]

        clv = np.where(
            high_low_range > 0,
            ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low_range,
            0,
        )

        df["AD_Line"] = (pd.Series(clv, index=df.index) * df["Volume"]).cumsum()
        df["AD_SMA_20"] = df["AD_Line"].rolling(20).mean()

        # AD trend (is accumulation/distribution accelerating?)
        df["AD_Trend"] = np.where(
            df["AD_Line"] > df["AD_SMA_20"], "Accumulation", "Distribution"
        )

        return df

    @staticmethod
    def _add_cumulative_return(df: pd.DataFrame) -> pd.DataFrame:
        """Cumulative return from the start of the data period."""
        initial_price = df["Close"].iloc[0]
        df["Cumulative_Return"] = (df["Close"] / initial_price - 1) * 100

        return df

    def _log_feature_summary(self):
        """Log feature engineering summary statistics."""
        total_features = max(
            (s["new_features"] for s in self.metrics_summary.values()), default=0
        )
        logger.info(
            f"\n{'─'*60}\n"
            f"  Feature Engineering Summary:\n"
            f"    Stocks enriched: {len(self.data)}\n"
            f"    New features per stock: ~{total_features}\n"
            f"    Feature categories:\n"
            f"      • Returns & Price Action: 5 features\n"
            f"      • Moving Averages: {len(MOVING_AVG_WINDOWS) + 4} features\n"
            f"      • Momentum (RSI, MACD): 7 features\n"
            f"      • Volatility (BB, ATR, GK): 8 features\n"
            f"      • Volume (OBV, AD): 5 features\n"
            f"      • Risk (Sharpe, Drawdown): 5 features\n"
            f"      • Custom (Sentiment, Momentum, Trend): 8 features\n"
            f"      • Other (52W, Gap, Cumulative): 8+ features\n"
            f"{'─'*60}\n"
        )

    def get_combined_dataframe(self) -> pd.DataFrame:
        """Return a single DataFrame with all enriched stocks."""
        if not self.data:
            raise ValueError("No enriched data. Call transform_all() first.")
        return pd.concat(self.data.values(), ignore_index=True)
