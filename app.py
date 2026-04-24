#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Stock Data Intelligence Platform — Task 2                   ║
║  Backend REST API (FastAPI + Uvicorn)                        ║
║                                                              ║
║  Author: Mahim Yadav                                         ║
║  Tech:   Python, FastAPI, Pandas, Pydantic                   ║
╚══════════════════════════════════════════════════════════════╝

This module initializes the FastAPI application, configures middleware
(CORS, logging, response timing), and registers all API routes.

Endpoints (required by assignment):
    GET  /companies            → All available companies
    GET  /data/{symbol}        → Last 30 days of stock data
    GET  /summary/{symbol}     → 52-week high, low, average close
    GET  /compare              → Compare two stocks (Bonus)

Extra endpoints (creative additions):
    GET  /sectors              → Sector performance
    GET  /market/movers        → Top gainers/losers
    GET  /market/correlations  → Correlation matrix

Auto-generated Swagger docs at: /docs
ReDoc alternative at:           /redoc

Usage:
    # From the project root, inside venv:
    python app.py                          # Starts on http://127.0.0.1:8000
    python app.py --port 8080              # Custom port
    python app.py --reload                 # Hot reload (development)

    # Or with Uvicorn directly:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import time
import logging
import argparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

from src.api.routes import router, data_svc

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stock_api")


# ──────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="📊 Stock Data Intelligence API",
    description=(
        "## Overview\n\n"
        "A comprehensive REST API for accessing, analyzing, and comparing "
        "Indian stock market data (NSE). Built on top of a robust data pipeline "
        "that collects 2 years of OHLCV data, engineers 60+ technical and custom "
        "indicators, and computes portfolio-level analytics.\n\n"
        "## Features\n\n"
        "- **15 NSE Stocks** across IT, Banking, FMCG, Pharma, Auto, and Telecom sectors\n"
        "- **60+ Indicators** per stock (RSI, MACD, Bollinger, ATR, OBV, etc.)\n"
        "- **Custom Metrics**: Momentum Score, Sentiment Index, Trend Strength, Garman-Klass Vol\n"
        "- **Risk Analytics**: Sharpe, Sortino, Calmar, VaR, Beta, Alpha (CAPM)\n"
        "- **Stock Comparison**: Head-to-head analysis with algorithmic verdict\n"
        "- **Market Intelligence**: Sector performance, gainers/losers, correlation matrix\n\n"
        "## Data Source\n\n"
        "Stock data is sourced from **Yahoo Finance** via the `yfinance` library "
        "and processed through a multi-stage pipeline (cleaning → feature engineering → analysis).\n\n"
        "## Author\n\n"
        "**Mahim Yadav** | JarNox Internship Assignment — Part 2"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Mahim Yadav",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        {
            "name": "Meta",
            "description": "API health checks and metadata.",
        },
        {
            "name": "Companies",
            "description": "Retrieve the list of all available companies and their basic information.",
        },
        {
            "name": "Stock Data",
            "description": (
                "Access individual stock data — daily OHLCV records with technical indicators, "
                "and comprehensive summaries with 52-week metrics."
            ),
        },
        {
            "name": "Comparison",
            "description": (
                "Head-to-head stock comparison with multi-dimensional scoring, "
                "correlation analysis, and algorithmic verdict."
            ),
        },
        {
            "name": "Market Intelligence",
            "description": (
                "Portfolio-level analytics: sector performance, top movers, and correlation matrices."
            ),
        },
    ],
)


# ──────────────────────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────────────────────
# CORS — allow frontend dashboards to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add X-Response-Time header to every response."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000  # ms
    response.headers["X-Response-Time"] = f"{elapsed:.2f}ms"
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)"
    )
    return response


# ──────────────────────────────────────────────────────────────
# Exception Handlers
# ──────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return clean JSON."""
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please try again.",
            "status_code": 500,
        },
    )


# ──────────────────────────────────────────────────────────────
# Startup Event — Load Data
# ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_load_data():
    """Load all processed stock data into memory on application startup."""
    logger.info("=" * 60)
    logger.info("  📊 STOCK DATA INTELLIGENCE API")
    logger.info("  Loading processed data from Task 1 pipeline...")
    logger.info("=" * 60)

    data_svc.load()

    if data_svc.is_loaded:
        symbols = data_svc.get_all_symbols()
        logger.info(f"  ✅ Data loaded: {len(symbols)} stocks ready to serve")
        logger.info(f"  📖 Swagger UI: http://127.0.0.1:8000/docs")
        logger.info(f"  📖 ReDoc:      http://127.0.0.1:8000/redoc")
    else:
        logger.warning("  ⚠️  No data loaded! Run the pipeline first: python main.py")

    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# Register Routes
# ──────────────────────────────────────────────────────────────
app.include_router(router)

# ──────────────────────────────────────────────────────────────
# Visualization Dashboard (Part 3)
# ──────────────────────────────────────────────────────────────
FRONTEND_DIR = "frontend"
import os
if not os.path.exists(FRONTEND_DIR):
    os.makedirs(FRONTEND_DIR, exist_ok=True)

# Serve /assets/* for backwards compatibility
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend_assets")


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    """Serves the frontend visualization dashboard."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Dashboard is being built...</h1>"


# === Root-level static file routes ===
# When the browser is at /dashboard and the HTML uses relative paths like
# "style.css" or "./app.js", the browser resolves them to /style.css and
# /app.js. These explicit routes handle that case, so relative paths work
# both from the FastAPI server AND when the HTML is opened directly from disk.
from fastapi.responses import FileResponse


@app.get("/style.css", include_in_schema=False)
async def serve_root_css():
    """Serve stylesheet at root for relative path resolution from /dashboard."""
    path = os.path.join(FRONTEND_DIR, "style.css")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/css")
    return HTMLResponse("/* not found */", status_code=404)


@app.get("/app.js", include_in_schema=False)
async def serve_root_js():
    """Serve JavaScript at root for relative path resolution from /dashboard."""
    path = os.path.join(FRONTEND_DIR, "app.js")
    if os.path.exists(path):
        return FileResponse(path, media_type="application/javascript")
    return HTMLResponse("// not found", status_code=404)


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────
def parse_args():
    """Parse CLI arguments for the API server."""
    parser = argparse.ArgumentParser(
        description="Stock Data Intelligence API Server (Task 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app.py                     # Start on port 8000
    python app.py --port 8080         # Custom port
    python app.py --reload            # Hot-reload for development
    python app.py --host 0.0.0.0      # Bind to all interfaces
        """,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
