"""
Pydantic Schemas (Response Models)
====================================
Defines strongly-typed response payloads for every API endpoint.
These schemas enforce data contracts, power Swagger documentation,
and provide automatic JSON serialization.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Meta / Health
# ──────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="ok")
    version: str = Field(..., example="2.0.0")
    data_available: bool = Field(..., description="Whether processed stock data is loaded")
    stocks_loaded: int = Field(..., description="Number of companies available")
    timestamp: str = Field(..., description="Server timestamp (ISO 8601)")


# ──────────────────────────────────────────────────────────────
# /companies
# ──────────────────────────────────────────────────────────────
class CompanyInfo(BaseModel):
    """Represents a single company overview."""
    symbol: str = Field(..., example="RELIANCE", description="NSE ticker symbol (without .NS suffix)")
    name: str = Field(..., example="Reliance Industries")
    sector: str = Field(..., example="Oil & Gas / Conglomerate")
    latest_close: float = Field(..., description="Most recent closing price (₹)")
    high_52w: float = Field(..., description="52-week high price (₹)")
    low_52w: float = Field(..., description="52-week low price (₹)")
    pct_from_52w_high: float = Field(..., description="Distance from 52-week high (%)")
    data_points: int = Field(..., description="Number of trading days of data available")


class CompaniesResponse(BaseModel):
    """Response for GET /companies."""
    count: int = Field(..., description="Total number of companies")
    companies: list[CompanyInfo]


# ──────────────────────────────────────────────────────────────
# /data/{symbol}
# ──────────────────────────────────────────────────────────────
class DailyStockData(BaseModel):
    """Single trading day's OHLCV data with derived metrics."""
    date: str = Field(..., example="2026-04-22")
    open: float
    high: float
    low: float
    close: float
    volume: int
    daily_return_pct: Optional[float] = Field(None, description="Intraday return (%)")
    sma_7: Optional[float] = Field(None, description="7-day simple moving average")
    rsi: Optional[float] = Field(None, description="Relative Strength Index (14-period)")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram value")
    bollinger_pct_b: Optional[float] = Field(None, description="Bollinger %B position")
    sentiment_index: Optional[float] = Field(None, description="Composite sentiment (0-100)")


class StockDataResponse(BaseModel):
    """Response for GET /data/{symbol}."""
    symbol: str
    company: str
    sector: str
    period_start: str
    period_end: str
    trading_days: int
    data: list[DailyStockData]


# ──────────────────────────────────────────────────────────────
# /summary/{symbol}
# ──────────────────────────────────────────────────────────────
class TechnicalSnapshot(BaseModel):
    """Current technical indicator readings."""
    rsi: Optional[float] = Field(None, description="RSI (14-period)")
    rsi_signal: Optional[str] = Field(None, description="RSI interpretation (Overbought / Oversold / Strong)")
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    sma_7: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr: Optional[float] = Field(None, description="Average True Range (14-period)")
    atr_pct: Optional[float] = Field(None, description="ATR as % of price")


class RiskMetrics(BaseModel):
    """Risk profile for the stock."""
    annual_return_pct: Optional[float] = None
    annual_volatility_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    calmar_ratio: Optional[float] = None
    var_95_daily_pct: Optional[float] = Field(None, description="Value at Risk 95% (daily %)")
    beta: Optional[float] = Field(None, description="Beta vs NIFTY 50")
    alpha_annual_pct: Optional[float] = Field(None, description="Jensen's Alpha (annual %)")


class CustomMetrics(BaseModel):
    """Custom / creative metrics computed during feature engineering."""
    garman_klass_volatility: Optional[float] = Field(None, description="OHLC-based volatility estimator")
    momentum_score: Optional[float] = Field(None, description="Multi-timeframe momentum (0-100)")
    sentiment_index: Optional[float] = Field(None, description="Composite sentiment (0-100)")
    sentiment_label: Optional[str] = Field(None, description="Extreme Fear / Fear / Neutral / Greed / Extreme Greed")
    trend_strength: Optional[float] = Field(None, description="MA alignment + consistency (0-100)")
    rolling_sharpe: Optional[float] = Field(None, description="30-day rolling Sharpe")
    money_flow: Optional[str] = Field(None, description="Accumulation / Distribution / Neutral")


class StockSummaryResponse(BaseModel):
    """Response for GET /summary/{symbol}."""
    symbol: str
    company: str
    sector: str
    latest_close: float
    high_52w: float
    low_52w: float
    avg_close_52w: float = Field(..., description="Average closing price over 52 weeks")
    pct_from_52w_high: float
    pct_from_52w_low: float
    position_in_52w_range: float = Field(
        ..., description="Position within 52-week range (0 = at low, 100 = at high)"
    )
    return_30d_pct: Optional[float] = Field(None, description="30-day return (%)")
    return_ytd_pct: Optional[float] = Field(None, description="Year-to-date return (%)")
    technicals: TechnicalSnapshot
    risk: RiskMetrics
    custom_metrics: CustomMetrics


# ──────────────────────────────────────────────────────────────
# /compare
# ──────────────────────────────────────────────────────────────
class CompareStockEntry(BaseModel):
    """Performance data for one stock in a comparison."""
    symbol: str
    company: str
    sector: str
    latest_close: float
    high_52w: float
    low_52w: float
    avg_close_52w: float
    return_30d_pct: Optional[float] = None
    return_ytd_pct: Optional[float] = None
    return_total_pct: Optional[float] = None
    annual_return_pct: Optional[float] = None
    annual_volatility_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    beta: Optional[float] = None
    rsi: Optional[float] = None
    sentiment_index: Optional[float] = None
    momentum_score: Optional[float] = None
    trend_strength: Optional[float] = None


class PriceCorrelation(BaseModel):
    """Correlation data between two stocks."""
    return_correlation: Optional[float] = Field(None, description="Pearson correlation of daily returns")
    price_correlation: Optional[float] = Field(None, description="Pearson correlation of closing prices")
    correlation_strength: Optional[str] = Field(None, description="Human-readable interpretation")


class CompareVerdict(BaseModel):
    """Algorithmic comparison verdict."""
    sharpe_winner: Optional[str] = None
    return_winner: Optional[str] = None
    lower_risk: Optional[str] = None
    momentum_leader: Optional[str] = None
    overall_verdict: str = Field(..., description="Summary conclusion")
    reasoning: str = Field(..., description="Explanation of the verdict")


class CompareResponse(BaseModel):
    """Response for GET /compare."""
    stock1: CompareStockEntry
    stock2: CompareStockEntry
    correlation: PriceCorrelation
    verdict: CompareVerdict
    price_history: dict = Field(
        ..., description="Normalized price history for both stocks (base=100)"
    )


# ──────────────────────────────────────────────────────────────
# Extras (beyond assignment scope — senior-level additions)
# ──────────────────────────────────────────────────────────────
class SectorPerformance(BaseModel):
    """Sector-level aggregate performance."""
    sector: str
    num_stocks: int
    avg_annual_return_pct: float
    avg_volatility_pct: float
    avg_sentiment: float


class SectorsResponse(BaseModel):
    """Response for GET /sectors."""
    count: int
    sectors: list[SectorPerformance]


class TopMover(BaseModel):
    """A top gainer or loser."""
    symbol: str
    company: str
    sector: str
    return_30d_pct: float
    latest_close: float


class TopMoversResponse(BaseModel):
    """Response for GET /market/movers."""
    gainers: list[TopMover]
    losers: list[TopMover]


class CorrelationMatrixResponse(BaseModel):
    """Response for GET /market/correlations."""
    symbols: list[str]
    matrix: list[list[float]]
    most_correlated: dict
    least_correlated: dict


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    status_code: int

# ──────────────────────────────────────────────────────────────
# /predict/{symbol}
# ──────────────────────────────────────────────────────────────
class PredictionPoint(BaseModel):
    date: str
    predicted_close: float

class PredictionResponse(BaseModel):
    symbol: str
    company: str
    model_type: str = Field(..., description="E.g., Linear Regression")
    historical_days_used: int
    prediction_days: int
    predictions: list[PredictionPoint]
    r_squared: float = Field(..., description="Model fit quality (0 to 1)")
    trend_direction: str = Field(..., description="Up, Down, or Neutral")
