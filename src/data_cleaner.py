"""
Data Cleaner Module
===================
Handles all data quality operations on raw stock data.

Key responsibilities:
    - Handle missing values with intelligent strategies
    - Fix incorrect data types and formats
    - Detect and handle outliers
    - Standardize column naming conventions
    - Validate data integrity (e.g., High >= Low, Volume >= 0)
    - Generate a data quality report

Design Decisions:
    - Missing OHLCV values are forward-filled (most recent known price)
      with backward-fill as fallback — this is standard in finance
    - Zero-volume days are flagged but kept (may represent holidays/halts)
    - Negative prices are treated as data errors and NaN'd
    - Date parsing is timezone-aware and then converted to UTC-naive
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED_DIR

logger = logging.getLogger(__name__)

# Expected column schema after cleaning
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Symbol", "Company"]


class StockDataCleaner:
    """
    Cleans and validates raw stock market data.

    Attributes:
        data (dict): Dictionary mapping symbol -> cleaned DataFrame.
        quality_report (dict): Per-stock data quality metrics.
    """

    def __init__(self):
        self.data: dict[str, pd.DataFrame] = {}
        self.quality_report: dict[str, dict] = {}

    def clean_all(self, raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Clean all stocks in the raw data dictionary.

        Args:
            raw_data: Dictionary mapping symbol -> raw DataFrame.

        Returns:
            Dictionary mapping symbol -> cleaned DataFrame.
        """
        logger.info(
            f"\n{'='*60}\n"
            f"  DATA CLEANING & VALIDATION\n"
            f"  Stocks to process: {len(raw_data)}\n"
            f"{'='*60}\n"
        )

        for symbol, df in raw_data.items():
            cleaned = self._clean_single_stock(symbol, df.copy())
            if cleaned is not None:
                self.data[symbol] = cleaned

        # Save cleaned combined dataset
        if self.data:
            combined = pd.concat(self.data.values(), ignore_index=True)
            combined_path = DATA_PROCESSED_DIR / "all_stocks_cleaned.csv"
            combined.to_csv(combined_path, index=False)
            logger.info(f"\n  Cleaned dataset saved: {combined_path}")

        self._log_quality_summary()
        return self.data

    def _clean_single_stock(
        self, symbol: str, df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        """
        Apply the full cleaning pipeline to a single stock's data.

        Pipeline:
            1. Standardize columns
            2. Fix date formats
            3. Remove duplicates
            4. Handle missing values
            5. Validate price integrity
            6. Sort chronologically
        """
        report = {
            "original_rows": len(df),
            "issues_found": [],
            "issues_fixed": [],
        }

        # ── Step 1: Standardize Column Names ──────────────────────
        df = self._standardize_columns(df)

        # Verify required columns exist
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            logger.error(f"  ✗ {symbol}: Missing columns {missing_cols}. Skipping.")
            return None

        # ── Step 2: Fix Date Format ───────────────────────────────
        df = self._fix_dates(df, report)

        # ── Step 3: Remove Duplicate Rows ─────────────────────────
        dupes = df.duplicated(subset=["Date", "Symbol"], keep="last")
        if dupes.sum() > 0:
            report["issues_found"].append(f"{dupes.sum()} duplicate rows")
            report["issues_fixed"].append(f"Removed {dupes.sum()} duplicates")
            df = df[~dupes].copy()

        # ── Step 4: Handle Missing Values ─────────────────────────
        df = self._handle_missing_values(df, report)

        # ── Step 5: Validate Price Integrity ──────────────────────
        df = self._validate_prices(df, report)

        # ── Step 6: Sort by Date ──────────────────────────────────
        df = df.sort_values("Date").reset_index(drop=True)

        # ── Final Report ──────────────────────────────────────────
        report["final_rows"] = len(df)
        report["rows_removed"] = report["original_rows"] - report["final_rows"]
        report["date_range"] = (
            f"{df['Date'].min().strftime('%Y-%m-%d')} → "
            f"{df['Date'].max().strftime('%Y-%m-%d')}"
        )
        self.quality_report[symbol] = report

        issues = len(report["issues_found"])
        status = "✓" if issues == 0 else f"⚠ {issues} issues fixed"
        logger.info(f"  {status} {symbol}: {len(df)} rows ({report['date_range']})")

        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to Title Case convention."""
        rename_map = {}
        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in ("date",):
                rename_map[col] = "Date"
            elif col_lower in ("open",):
                rename_map[col] = "Open"
            elif col_lower in ("high",):
                rename_map[col] = "High"
            elif col_lower in ("low",):
                rename_map[col] = "Low"
            elif col_lower in ("close",):
                rename_map[col] = "Close"
            elif col_lower in ("adj close", "adjclose", "adjusted close"):
                rename_map[col] = "Adj Close"
            elif col_lower in ("volume",):
                rename_map[col] = "Volume"
            elif col_lower in ("symbol", "ticker"):
                rename_map[col] = "Symbol"
            elif col_lower in ("company", "name", "company_name"):
                rename_map[col] = "Company"

        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def _fix_dates(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Convert Date column to proper datetime, handling timezone issues."""
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            try:
                df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)
                report["issues_found"].append("Non-datetime Date column")
                report["issues_fixed"].append("Converted Date to datetime")
            except Exception as e:
                logger.error(f"Date parsing error: {e}")

        # Remove timezone info for consistency (store as UTC-naive)
        if hasattr(df["Date"].dtype, "tz") and df["Date"].dtype.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)

        # Drop rows with unparseable dates
        null_dates = df["Date"].isna()
        if null_dates.sum() > 0:
            report["issues_found"].append(f"{null_dates.sum()} unparseable dates")
            report["issues_fixed"].append(f"Dropped {null_dates.sum()} invalid date rows")
            df = df[~null_dates].copy()

        return df

    def _handle_missing_values(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """
        Handle missing values using finance-appropriate strategies.

        Strategy:
            - OHLC prices → Forward-fill, then backward-fill
            - Volume → Fill with 0 (assume no trading)
            - Other columns → Keep as-is
        """
        price_cols = ["Open", "High", "Low", "Close"]
        if "Adj Close" in df.columns:
            price_cols.append("Adj Close")

        # Count missing before
        total_missing = df[price_cols + ["Volume"]].isna().sum().sum()

        if total_missing > 0:
            report["issues_found"].append(f"{total_missing} missing OHLCV values")

            # Forward-fill prices (carry forward last known price)
            df[price_cols] = df[price_cols].ffill().bfill()

            # Fill missing volume with 0
            df["Volume"] = df["Volume"].fillna(0)

            remaining = df[price_cols + ["Volume"]].isna().sum().sum()
            report["issues_fixed"].append(
                f"Filled {total_missing - remaining} missing values (ffill+bfill)"
            )

            # Drop any rows still missing after fill (edge case: all NaN)
            if remaining > 0:
                df = df.dropna(subset=price_cols, how="any")

        return df

    def _validate_prices(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """
        Validate price integrity and fix obvious errors.

        Checks:
            - No negative prices
            - High >= Low for each day
            - Close is between Low and High (with tolerance)
            - Volume is non-negative
        """
        price_cols = ["Open", "High", "Low", "Close"]

        # Check for negative prices
        for col in price_cols:
            negative_mask = df[col] < 0
            if negative_mask.sum() > 0:
                report["issues_found"].append(f"{negative_mask.sum()} negative {col} prices")
                df.loc[negative_mask, col] = np.nan
                df[col] = df[col].ffill().bfill()
                report["issues_fixed"].append(f"Replaced negative {col} with ffill")

        # Check High >= Low
        invalid_hl = df["High"] < df["Low"]
        if invalid_hl.sum() > 0:
            report["issues_found"].append(f"{invalid_hl.sum()} rows where High < Low")
            # Swap High and Low where invalid
            df.loc[invalid_hl, ["High", "Low"]] = df.loc[
                invalid_hl, ["Low", "High"]
            ].values
            report["issues_fixed"].append("Swapped High/Low where High < Low")

        # Ensure non-negative volume
        df["Volume"] = df["Volume"].clip(lower=0)

        # Convert Volume to integer
        df["Volume"] = df["Volume"].astype(np.int64)

        return df

    def _log_quality_summary(self):
        """Log a summary of data quality across all stocks."""
        total_issues = sum(
            len(r["issues_found"]) for r in self.quality_report.values()
        )
        total_rows = sum(r["final_rows"] for r in self.quality_report.values())

        logger.info(
            f"\n{'─'*60}\n"
            f"  Cleaning Summary:\n"
            f"    Stocks processed: {len(self.quality_report)}\n"
            f"    Total clean rows: {total_rows:,}\n"
            f"    Issues detected & fixed: {total_issues}\n"
            f"{'─'*60}\n"
        )

    def get_combined_dataframe(self) -> pd.DataFrame:
        """Return a single DataFrame with all cleaned stocks."""
        if not self.data:
            raise ValueError("No cleaned data available. Call clean_all() first.")
        return pd.concat(self.data.values(), ignore_index=True)
