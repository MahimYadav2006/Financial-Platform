"""
API Routes
===========
Defines all REST endpoints for the Stock Data Intelligence Platform.

Endpoints:
    GET  /                           → API info / health check
    GET  /companies                  → List all companies
    GET  /data/{symbol}              → Last N days of OHLCV + indicators
    GET  /summary/{symbol}           → 52-week high/low, avg close, technicals, risk
    GET  /compare                    → Head-to-head comparison (Bonus)
    GET  /sectors                    → Sector performance summary
    GET  /market/movers              → Top gainers and losers
    GET  /market/correlations        → Full correlation matrix

All endpoints return structured JSON with proper HTTP status codes
and are documented via OpenAPI (Swagger).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Path

from src.api.data_service import DataService
from src.api.schemas import (
    HealthResponse,
    CompaniesResponse,
    StockDataResponse,
    StockSummaryResponse,
    CompareResponse,
    SectorsResponse,
    TopMoversResponse,
    CorrelationMatrixResponse,
    PredictionResponse,
)

router = APIRouter()

# Singleton data service — loaded at app startup
data_svc = DataService()


# ──────────────────────────────────────────────────────────────
# Health / Root
# ──────────────────────────────────────────────────────────────
@router.get(
    "/",
    response_model=HealthResponse,
    tags=["Meta"],
    summary="API Health Check",
    description="Returns the current status of the API, data availability, and server timestamp.",
)
async def health_check():
    return HealthResponse(
        status="ok",
        version="2.0.0",
        data_available=data_svc.is_loaded,
        stocks_loaded=len(data_svc.get_all_symbols()),
        timestamp=datetime.now().isoformat(),
    )


# ──────────────────────────────────────────────────────────────
# GET /companies
# ──────────────────────────────────────────────────────────────
@router.get(
    "/companies",
    response_model=CompaniesResponse,
    tags=["Companies"],
    summary="List all available companies",
    description=(
        "Returns a list of all companies in the stock universe with their "
        "latest closing price, 52-week high/low, sector classification, "
        "and number of available data points."
    ),
)
async def get_companies():
    companies = data_svc.get_companies()
    return CompaniesResponse(count=len(companies), companies=companies)


# ──────────────────────────────────────────────────────────────
# GET /data/{symbol}
# ──────────────────────────────────────────────────────────────
@router.get(
    "/data/{symbol}",
    response_model=StockDataResponse,
    tags=["Stock Data"],
    summary="Get daily stock data",
    description=(
        "Returns the last N trading days of OHLCV data plus key technical indicators "
        "(SMA-7, RSI, MACD, Bollinger %B, Sentiment Index) for the specified stock. "
        "Default is 30 days."
    ),
    responses={
        404: {"description": "Symbol not found"},
    },
)
async def get_stock_data(
    symbol: str = Path(
        ...,
        description="Stock ticker symbol (e.g., RELIANCE, INFY, TCS). Case-insensitive.",
        examples=["RELIANCE", "INFY", "TCS", "HDFCBANK"],
    ),
    days: int = Query(
        30,
        ge=1,
        le=730,
        description="Number of trading days to return (1–730). Default: 30.",
    ),
):
    result = data_svc.get_stock_data(symbol, days=days)
    if result is None:
        available = [s.replace(".NS", "") for s in data_svc.get_all_symbols()]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Symbol '{symbol}' not found. "
                f"Available symbols: {', '.join(sorted(available))}"
            ),
        )
    return StockDataResponse(**result)


# ──────────────────────────────────────────────────────────────
# GET /summary/{symbol}
# ──────────────────────────────────────────────────────────────
@router.get(
    "/summary/{symbol}",
    response_model=StockSummaryResponse,
    tags=["Stock Data"],
    summary="Get stock summary with 52-week metrics",
    description=(
        "Returns a comprehensive summary for the specified stock, including:\n"
        "- 52-week high, low, and average close\n"
        "- Current technical indicators (RSI, MACD, Bollinger, SMAs)\n"
        "- Risk profile (Sharpe, Sortino, VaR, Beta, Alpha)\n"
        "- Custom metrics (Momentum Score, Sentiment Index, Trend Strength)\n"
        "- Return percentages (30-day, YTD)"
    ),
    responses={
        404: {"description": "Symbol not found"},
    },
)
async def get_stock_summary(
    symbol: str = Path(
        ...,
        description="Stock ticker symbol (e.g., RELIANCE, INFY, TCS). Case-insensitive.",
        examples=["RELIANCE", "INFY", "TCS", "HDFCBANK"],
    ),
):
    result = data_svc.get_stock_summary(symbol)
    if result is None:
        available = [s.replace(".NS", "") for s in data_svc.get_all_symbols()]
        raise HTTPException(
            status_code=404,
            detail=(
                f"Symbol '{symbol}' not found. "
                f"Available symbols: {', '.join(sorted(available))}"
            ),
        )
    return StockSummaryResponse(**result)


# ──────────────────────────────────────────────────────────────
# GET /compare (Bonus)
# ──────────────────────────────────────────────────────────────
@router.get(
    "/compare",
    response_model=CompareResponse,
    tags=["Comparison"],
    summary="Compare two stocks' performance (Bonus)",
    description=(
        "Head-to-head comparison of two stocks across multiple dimensions:\n"
        "- Price & return metrics (30D, YTD, Total, Annualized)\n"
        "- Risk metrics (Volatility, Sharpe, Max Drawdown, Beta)\n"
        "- Technical signals (RSI, Momentum, Sentiment, Trend)\n"
        "- Return correlation between the two stocks\n"
        "- Algorithmic verdict with scoring and reasoning\n"
        "- Normalized price history (base 100) for charting"
    ),
    responses={
        400: {"description": "Missing or invalid symbol parameters"},
        404: {"description": "One or both symbols not found"},
    },
)
async def compare_stocks(
    symbol1: str = Query(
        ...,
        description="First stock ticker (e.g., INFY)",
        examples=["INFY"],
    ),
    symbol2: str = Query(
        ...,
        description="Second stock ticker (e.g., TCS)",
        examples=["TCS"],
    ),
):
    if symbol1.upper() == symbol2.upper():
        raise HTTPException(
            status_code=400,
            detail="Cannot compare a stock with itself. Provide two different symbols.",
        )

    # Validate both symbols exist
    available = [s.replace(".NS", "") for s in data_svc.get_all_symbols()]
    errors = []
    if data_svc.resolve_symbol(symbol1) is None:
        errors.append(f"'{symbol1}'")
    if data_svc.resolve_symbol(symbol2) is None:
        errors.append(f"'{symbol2}'")

    if errors:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Symbol(s) {', '.join(errors)} not found. "
                f"Available: {', '.join(sorted(available))}"
            ),
        )

    result = data_svc.compare_stocks(symbol1, symbol2)
    return CompareResponse(**result)


# ──────────────────────────────────────────────────────────────
# GET /predict/{symbol} (Bonus ML Feature)
# ──────────────────────────────────────────────────────────────
@router.get(
    "/predict/{symbol}",
    response_model=PredictionResponse,
    tags=["Stock Data"],
    summary="Predict future prices (Bonus)",
    description=(
        "Uses a simple Linear Regression model built with Scipy to identify the "
        "current trend based on recent history and plot out predicted closing prices "
        "for the upcoming week."
    ),
    responses={
        404: {"description": "Symbol not found"},
    },
)
async def predict_stock(
    symbol: str = Path(
        ...,
        description="Stock ticker symbol (e.g., RELIANCE)",
    ),
    lookback: int = Query(30, description="Days of history to analyze"),
    days: int = Query(7, description="Days ahead to predict"),
):
    result = data_svc.predict_stock(symbol, lookback_days=lookback, predict_days=days)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol}' not found.",
        )
    return PredictionResponse(**result)


# ──────────────────────────────────────────────────────────────
# Extra Endpoints (Beyond Assignment — Senior-Level)
# ──────────────────────────────────────────────────────────────
@router.get(
    "/sectors",
    response_model=SectorsResponse,
    tags=["Market Intelligence"],
    summary="Sector performance overview",
    description="Returns aggregated performance metrics for each sector.",
)
async def get_sectors():
    sectors = data_svc.get_sectors()
    return SectorsResponse(
        count=len(sectors),
        sectors=[
            {
                "sector": s["Sector"],
                "num_stocks": s["Num_Stocks"],
                "avg_annual_return_pct": s["Avg_Annual_Return_Pct"],
                "avg_volatility_pct": s["Avg_Volatility_Pct"],
                "avg_sentiment": s["Avg_Sentiment"],
            }
            for s in sectors
        ],
    )


@router.get(
    "/market/movers",
    response_model=TopMoversResponse,
    tags=["Market Intelligence"],
    summary="Top gainers and losers",
    description="Returns the top N gainers and losers based on 30-day returns.",
)
async def get_top_movers(
    n: int = Query(5, ge=1, le=15, description="Number of stocks per category (1–15)."),
):
    movers = data_svc.get_top_movers(n)
    return TopMoversResponse(**movers)


@router.get(
    "/market/correlations",
    response_model=CorrelationMatrixResponse,
    tags=["Market Intelligence"],
    summary="Full correlation matrix",
    description=(
        "Returns the pairwise Pearson correlation matrix of daily returns "
        "across all stocks, plus the most and least correlated pairs."
    ),
)
async def get_correlations():
    result = data_svc.get_correlation_matrix()
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Correlation matrix not available. Run the pipeline first.",
        )
    return CorrelationMatrixResponse(**result)
