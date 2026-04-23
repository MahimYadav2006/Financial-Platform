"""
Data Collector Module
=====================
Fetches historical stock data from Yahoo Finance using the yfinance library.

Key responsibilities:
    - Download OHLCV data for the entire stock universe + benchmark
    - Handle API failures gracefully with retries and logging
    - Save raw data to CSV for reproducibility and audit trail
    - Provide both individual and bulk download capabilities

Design Decisions:
    - We use yfinance because it provides free, reliable NSE data
    - Raw data is saved as-is before any cleaning, ensuring reproducibility
    - Each stock gets its own CSV file for modularity
    - A combined CSV is also saved for cross-stock analysis
"""

import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config import (
    STOCK_UNIVERSE,
    BENCHMARK_SYMBOL,
    BENCHMARK_NAME,
    START_DATE,
    END_DATE,
    DATA_RAW_DIR,
)

logger = logging.getLogger(__name__)


class StockDataCollector:
    """
    Handles fetching and caching of stock market data from Yahoo Finance.

    Attributes:
        symbols (dict): Mapping of ticker symbols to company names.
        start_date (str): Start date for historical data (YYYY-MM-DD).
        end_date (str): End date for historical data (YYYY-MM-DD).
        raw_data (dict): Dictionary mapping symbol -> raw DataFrame.
    """

    def __init__(
        self,
        symbols: Optional[dict] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_benchmark: bool = True,
    ):
        self.symbols = symbols or STOCK_UNIVERSE.copy()
        self.start_date = start_date or START_DATE
        self.end_date = end_date or END_DATE
        self.include_benchmark = include_benchmark
        self.raw_data: dict[str, pd.DataFrame] = {}

        # Add benchmark to symbols if requested
        if self.include_benchmark and BENCHMARK_SYMBOL not in self.symbols:
            self.symbols[BENCHMARK_SYMBOL] = BENCHMARK_NAME

    def fetch_single_stock(
        self, symbol: str, max_retries: int = 3, delay: float = 1.0
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a single stock with retry logic.

        Args:
            symbol: Yahoo Finance ticker symbol (e.g., 'RELIANCE.NS')
            max_retries: Number of retry attempts on failure
            delay: Seconds to wait between retries (with exponential backoff)

        Returns:
            DataFrame with columns [Open, High, Low, Close, Adj Close, Volume]
            or None if all retries fail.
        """
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Fetching {symbol} (attempt {attempt}/{max_retries})..."
                )
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=False,  # Keep both Close and Adj Close
                )

                if df.empty:
                    logger.warning(f"No data returned for {symbol}.")
                    return None

                # Standardize column names
                df.index.name = "Date"
                df = df.reset_index()

                # Add metadata columns
                df["Symbol"] = symbol
                df["Company"] = self.symbols.get(symbol, symbol)

                logger.info(
                    f"  ✓ {symbol}: {len(df)} trading days fetched "
                    f"({df['Date'].min().strftime('%Y-%m-%d')} → "
                    f"{df['Date'].max().strftime('%Y-%m-%d')})"
                )
                return df

            except Exception as e:
                wait_time = delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.warning(
                    f"  ✗ Error fetching {symbol}: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                if attempt < max_retries:
                    time.sleep(wait_time)

        logger.error(f"  ✗ Failed to fetch {symbol} after {max_retries} attempts.")
        return None

    def fetch_all(self, save_raw: bool = True) -> dict[str, pd.DataFrame]:
        """
        Fetch data for the entire stock universe.

        Args:
            save_raw: If True, saves each stock's raw data to CSV.

        Returns:
            Dictionary mapping symbol -> DataFrame.
        """
        logger.info(
            f"\n{'='*60}\n"
            f"  STOCK DATA COLLECTION\n"
            f"  Universe: {len(self.symbols)} tickers\n"
            f"  Period: {self.start_date} → {self.end_date}\n"
            f"{'='*60}\n"
        )

        success_count = 0
        fail_count = 0

        for symbol, name in self.symbols.items():
            df = self.fetch_single_stock(symbol)
            if df is not None:
                self.raw_data[symbol] = df
                success_count += 1

                # Save individual stock CSV
                if save_raw:
                    filepath = DATA_RAW_DIR / f"{symbol.replace('.', '_')}_raw.csv"
                    df.to_csv(filepath, index=False)
                    logger.debug(f"  Saved: {filepath}")
            else:
                fail_count += 1

            # Be respectful to the API
            time.sleep(0.5)

        # Save combined dataset
        if save_raw and self.raw_data:
            combined = pd.concat(self.raw_data.values(), ignore_index=True)
            combined_path = DATA_RAW_DIR / "all_stocks_raw.csv"
            combined.to_csv(combined_path, index=False)
            logger.info(f"\n  Combined dataset saved: {combined_path}")

        logger.info(
            f"\n{'─'*60}\n"
            f"  Collection Summary:\n"
            f"    ✓ Success: {success_count} stocks\n"
            f"    ✗ Failed:  {fail_count} stocks\n"
            f"    Total trading days: {sum(len(df) for df in self.raw_data.values())}\n"
            f"{'─'*60}\n"
        )

        return self.raw_data

    def get_combined_dataframe(self) -> pd.DataFrame:
        """Return a single DataFrame with all stocks combined."""
        if not self.raw_data:
            raise ValueError("No data loaded. Call fetch_all() first.")
        return pd.concat(self.raw_data.values(), ignore_index=True)

    def load_from_cache(self) -> bool:
        """
        Load previously downloaded raw data from CSV files.

        Returns:
            True if data was loaded successfully, False otherwise.
        """
        combined_path = DATA_RAW_DIR / "all_stocks_raw.csv"
        if combined_path.exists():
            logger.info(f"Loading cached data from {combined_path}...")
            df = pd.read_csv(combined_path, parse_dates=["Date"])
            for symbol in df["Symbol"].unique():
                self.raw_data[symbol] = df[df["Symbol"] == symbol].copy()
            logger.info(f"  ✓ Loaded {len(self.raw_data)} stocks from cache.")
            return True
        return False
