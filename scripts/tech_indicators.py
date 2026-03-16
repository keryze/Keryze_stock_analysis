#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from smart_data_source import SmartDataSource
from stock_cache_db import StockCache


class TechIndicatorCalculator:
    """Minimal technical indicator adapter for the web app APIs."""

    def __init__(self):
        self.ds = SmartDataSource()
        self.cache = StockCache()

    def get_stock_history(self, code: str, days: int = 60) -> Optional[pd.DataFrame]:
        df = self.ds.get_history_data(code, days=days)
        if df is None or df.empty:
            return None

        data = df.copy()
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            data = data.set_index("date")

        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        return data.dropna(subset=["open", "high", "low", "close"], how="any")

    def calculate_indicators(self, code: str, days: int = 60) -> Optional[Dict]:
        history = self.get_stock_history(code, days=days)
        if history is None or history.empty or len(history) < 20:
            return None

        close = history["close"]
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = (100 - (100 / (1 + rs))).fillna(50).iloc[-1]

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = (dif - dea).iloc[-1]

        return {
            "ma5": float(ma5) if pd.notna(ma5) else None,
            "ma10": float(ma10) if pd.notna(ma10) else None,
            "ma20": float(ma20) if pd.notna(ma20) else None,
            "rsi": float(rsi) if pd.notna(rsi) else None,
            "macd": float(macd) if pd.notna(macd) else None,
            "macd_dif": float(dif.iloc[-1]) if pd.notna(dif.iloc[-1]) else None,
            "macd_dea": float(dea.iloc[-1]) if pd.notna(dea.iloc[-1]) else None,
        }

    def close(self):
        self.ds.close()
        self.cache.close()
