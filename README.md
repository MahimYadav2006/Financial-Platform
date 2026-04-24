# 📊 Stock Data Intelligence Platform

An end-to-end financial data collection, analysis, API, and **visualization** platform built for the **JarNox Software Engineering Internship Assignment**.

## 🚀 Overview

This repository addresses all three parts of the assignment:

1. **`main.py` (Data Pipeline — Part 1)**: Fetches 2 years of historical OHLCV data for 15 major NSE stocks + NIFTY 50 via `yfinance`, cleans the data, and generates over 60 technical/custom indicators. Computes portfolio risk profiles and cross-stock analytics.
2. **`app.py` (FastAPI Server — Part 2)**: A high-performance REST API that serves the processed data. Loads all CSVs directly into memory at startup to offer sub-millisecond query responses and auto-generates interactive Swagger documentation.
3. **`frontend/` (Visualization Dashboard — Part 3)**: A premium, interactive **single-page dashboard** served directly from FastAPI. Features glassmorphic dark theme, Chart.js-powered closing price charts with SMA overlay, volume visualization, AI prediction via linear regression, head-to-head stock comparison, market heatmaps, and sector performance analytics.

## 🛠️ Technology Stack

* **Language**: Python 3.12
* **Backend Framework**: FastAPI & Uvicorn
* **Data Processing**: Pandas, NumPy
* **Data Source**: yfinance
* **Schema Validation**: Pydantic
* **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
* **Charting**: Chart.js 4.x
* **Prediction**: Scipy (Linear Regression)
* **Icons**: Phosphor Icons
* **Typography**: Inter (Google Fonts)

## 📦 Setup Instructions

1. **Clone the repository and ensure you have Python 3.10+ installed.**
2. **Create a virtual environment & install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the Data Pipeline (Part 1):**
   This fetches raw data, runs cleaning and feature engineering, and exports processed CSVs to `data/processed/`.
   ```bash
   python main.py
   ```

4. **Start the API Server (Part 2 + Part 3):**
   ```bash
   python app.py
   ```
   *The server runs on http://127.0.0.1:8000*

5. **Open the Visualization Dashboard (Part 3):**
   Navigate to **http://127.0.0.1:8000/dashboard** in your browser.

## 📖 API Documentation & Endpoints

FastAPI automatically generates interactive API docs. Once the server is running, visit:
* **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Core Endpoints Implementing Assignment Requirements:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/companies` | Returns a list of all 15 available NSE companies. |
| `GET` | `/data/{symbol}` | Returns the last 30 days of stock data for a given ticker. |
| `GET` | `/summary/{symbol}` | Returns 52-week summary + technical and risk metrics. |
| `GET` | `/compare?symbol1&symbol2` | **(Bonus)** Head-to-head comparison of two stocks with algorithmic verdict. |
| `GET` | `/predict/{symbol}` | **(Bonus)** AI-powered price prediction using Linear Regression (Scipy). |

### Extra API Features:
* `/sectors`: View aggregated performance by sector.
* `/market/movers`: Returns top 5 gainers and losers.
* `/market/correlations`: View a comprehensive return and price correlation matrix.

## 🎨 Dashboard Features (Part 3)

| Feature | Description |
| :--- | :--- |
| **Company Watchlist** | Searchable sidebar listing all 15 NSE stocks with live prices and 52W proximity indicators |
| **Closing Price Chart** | Interactive Chart.js line chart with gradient fill, SMA-7 overlay, and smooth animations |
| **Volume Chart** | Synchronized volume bar chart with green/red coloring based on price direction |
| **Time Range Filters** | Toggle between 1M, 3M, 6M, 1Y, and ALL timeframes |
| **AI Prediction** | 7-day price forecast using Linear Regression, displayed as a dashed prediction line |
| **Stock Comparison** | Head-to-head modal with normalized price chart, Sharpe/risk/momentum scoring, and algorithmic verdict |
| **Market Intelligence Panel** | Real-time metrics: Momentum, Sentiment, RSI, Sharpe, Beta, Volatility, Max Drawdown, VaR, Returns |
| **52-Week Range Gauge** | Visual progress bar showing current price position in 52-week range |
| **Market Heatmap** | Color-coded grid of all stocks based on 52-week high proximity |
| **Sector Performance** | Horizontal bar chart showing annualized sector returns |
| **Top Movers** | Header chips showing today's top gainer and loser |
| **Toast Notifications** | Elegant notification system with categorized alerts (info, success, error) |
| **Glassmorphism UI** | Premium dark theme with frosted glass panels, ambient gradients, and micro-animations |

## 🧠 Approach & Insights

* **Custom Metrics Added**: Beyond standard moving averages and RSI, this pipeline computes the *Garman-Klass Volatility* (OHLC-estimators), *Sortino & Calmar Analytics*, and creates a custom `Momentum Score` and `Sentiment Index`.
* **API Design**: The API uses a robust `DataService` singleton layer that loads once on application startup, avoiding Disk I/O or DataFrame rebuilds on every request.
* **Frontend Architecture**: Pure vanilla HTML/CSS/JS — no build step, no npm, no framework overhead. Chart.js handles all charting with gradient fills and custom tooltips. The dashboard is served directly by FastAPI as a static mount + dynamic route.
* **Typing & Schemas**: Using `Pydantic` `BaseModel` classes enforces response structures across the API, allowing FastAPI to build clean OpenAPI specifications.

## 📁 File Structure

```
Financial Platform/
├── app.py                      # FastAPI server (Part 2 + Part 3 serving)
├── main.py                     # Data pipeline orchestrator (Part 1)
├── requirements.txt            # Python dependencies
├── frontend/
│   ├── index.html              # Dashboard HTML (Part 3)
│   ├── style.css               # Premium glassmorphic dark theme
│   └── app.js                  # Dashboard logic, charting, interactivity
├── src/
│   ├── config.py               # Stock universe, sector maps, parameters
│   ├── data_collector.py       # yfinance data fetching
│   ├── data_cleaner.py         # Data cleaning & validation
│   ├── feature_engineering.py  # 60+ technical indicator computation
│   ├── analysis.py             # Cross-stock analytics
│   ├── visualizer.py           # Matplotlib chart generation
│   └── api/
│       ├── routes.py           # API route definitions
│       ├── data_service.py     # In-memory data service singleton
│       └── schemas.py          # Pydantic response models
├── data/
│   ├── raw/                    # Raw downloaded CSVs
│   └── processed/              # Cleaned & enriched datasets
└── output/
    ├── charts/                 # Generated visualization PNGs
    └── reports/                # Analysis CSV reports
```

## 🤝 Author

**Mahim Yadav** — JarNox SWE Internship Assignment
