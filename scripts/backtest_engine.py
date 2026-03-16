#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from smart_data_source import SmartDataSource


class BacktestEngine:
    """
    Minimal backtest engine for the web API.
    Strategy names are accepted for compatibility, but execution uses a simple
    MA5/MA20 crossover model so the endpoint returns structured results.
    """

    def __init__(self):
        self.ds = SmartDataSource()

    def _load_history(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        df = self.ds.get_history_data(symbol, days=240)
        if df is None or df.empty:
            return None

        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
        data = data[(data["date"] >= pd.to_datetime(start_date)) & (data["date"] <= pd.to_datetime(end_date))]
        if data.empty:
            return None

        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        return data.dropna(subset=["close"]).reset_index(drop=True)

    def backtest(
        self,
        symbol: str,
        strategy_name: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
    ) -> Optional[Dict]:
        history = self._load_history(symbol, start_date, end_date)
        if history is None or len(history) < 20:
            return None

        history["ma5"] = history["close"].rolling(5).mean()
        history["ma20"] = history["close"].rolling(20).mean()

        cash = float(initial_capital)
        position = 0
        trades: List[Dict] = []
        equity_curve: List[Dict] = []

        for i in range(20, len(history)):
            row = history.iloc[i]
            prev = history.iloc[i - 1]
            date = row["date"]
            price = float(row["close"])

            buy_signal = prev["ma5"] <= prev["ma20"] and row["ma5"] > row["ma20"]
            sell_signal = prev["ma5"] >= prev["ma20"] and row["ma5"] < row["ma20"]

            if position == 0 and buy_signal:
                qty = int(cash // price)
                if qty > 0:
                    amount = qty * price
                    cash -= amount
                    position = qty
                    trades.append({
                        "date": date,
                        "action": "buy",
                        "price": price,
                        "qty": qty,
                        "amount": round(amount, 2),
                    })
            elif position > 0 and sell_signal:
                amount = position * price
                cost_basis = trades[-1]["amount"] if trades else amount
                profit = amount - cost_basis
                cash += amount
                trades.append({
                    "date": date,
                    "action": "sell",
                    "price": price,
                    "qty": position,
                    "amount": round(amount, 2),
                    "profit": round(profit, 2),
                })
                position = 0

            total_value = cash + position * price
            equity_curve.append({"date": date.strftime("%Y-%m-%d"), "value": round(total_value, 2)})

        last_price = float(history.iloc[-1]["close"])
        final_value = cash + position * last_price
        total_return = ((final_value - initial_capital) / initial_capital) * 100

        return {
            "symbol": symbol,
            "strategy": strategy_name,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": round(float(initial_capital), 2),
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return, 2),
            "trade_count": len(trades),
            "position": position,
            "trades": trades,
            "equity_curve": equity_curve,
        }

    def close(self):
        self.ds.close()
