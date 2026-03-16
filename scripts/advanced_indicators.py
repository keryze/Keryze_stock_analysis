#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced technical indicators for long-term stock selection.
Provides pure pandas/numpy implementations with optional TA-Lib acceleration.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
except Exception:  # pragma: no cover
    talib = None


class AdvancedIndicators:
    """Indicator helper used by long_term_selector.py."""

    @staticmethod
    def _series(df: pd.DataFrame, name: str) -> pd.Series:
        """Get a numeric series with tolerant column-name matching."""
        candidates = [name, name.lower(), name.upper(), name.title()]
        for col in candidates:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
        raise KeyError(f"Missing required column: {name}")

    def score_trend(self, df: pd.DataFrame) -> Dict:
        """
        Score trend quality on a 0-100 scale.

        Returns:
            {
              'score': float,
              'rating': str,
              'reasons': List[str],
              'ma20': float,
              'ma60': float,
            }
        """
        close = self._series(df, "close")

        ma20 = close.rolling(20, min_periods=20).mean()
        ma60 = close.rolling(60, min_periods=60).mean()

        c_now = float(close.iloc[-1])
        ma20_now = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else c_now
        ma60_now = float(ma60.iloc[-1]) if not np.isnan(ma60.iloc[-1]) else c_now

        score = 0.0
        reasons: List[str] = []

        # 1) Price above moving averages
        if c_now > ma20_now:
            score += 25
            reasons.append("价格在20日线上方")
        if c_now > ma60_now:
            score += 20
            reasons.append("价格在60日线上方")

        # 2) MA alignment
        if ma20_now > ma60_now:
            score += 25
            reasons.append("20日线在60日线上方")

        # 3) MA20 slope
        if len(ma20.dropna()) >= 6:
            ma20_prev = float(ma20.dropna().iloc[-6])
            if ma20_now > ma20_prev:
                score += 20
                reasons.append("20日线向上")

        # 4) Recent strength (20-day return)
        if len(close) >= 21:
            ret20 = (float(close.iloc[-1]) - float(close.iloc[-21])) / float(close.iloc[-21]) * 100
            if ret20 > 0:
                score += min(10, ret20 / 2)
                reasons.append(f"20日涨幅 {ret20:.1f}%")

        score = float(max(0, min(100, score)))

        if score >= 80:
            rating = "强势上涨"
        elif score >= 60:
            rating = "稳健上涨"
        elif score >= 40:
            rating = "震荡整理"
        else:
            rating = "弱势下行"

        return {
            "score": round(score, 2),
            "rating": rating,
            "reasons": reasons,
            "ma20": round(ma20_now, 4),
            "ma60": round(ma60_now, 4),
        }

    def calc_obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        close = self._series(df, "close")
        volume = self._series(df, "volume").fillna(0)

        if talib is not None:
            obv = talib.OBV(close.values.astype(float), volume.values.astype(float))
            return pd.Series(obv, index=close.index)

        direction = np.sign(close.diff().fillna(0.0))
        obv = (direction * volume).cumsum()
        return obv

    def calc_volume_ratio(self, df: pd.DataFrame, period: int = 5) -> pd.Series:
        """Volume ratio = current volume / N-day average volume."""
        volume = self._series(df, "volume").replace(0, np.nan)
        avg_vol = volume.rolling(period, min_periods=period).mean()
        ratio = volume / avg_vol
        return ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    def calc_adx(
        self,
        df: pd.DataFrame,
        period: int = 14,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate ADX and DI lines.

        Returns:
            (adx, plus_di, minus_di)
        """
        high = self._series(df, "high")
        low = self._series(df, "low")
        close = self._series(df, "close")

        if talib is not None:
            adx = talib.ADX(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=period)
            plus_di = talib.PLUS_DI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=period)
            minus_di = talib.MINUS_DI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=period)
            return (
                pd.Series(adx, index=close.index),
                pd.Series(plus_di, index=close.index),
                pd.Series(minus_di, index=close.index),
            )

        prev_close = close.shift(1)

        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_n = pd.Series(tr, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
        plus_dm_n = pd.Series(plus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean()
        minus_dm_n = pd.Series(minus_dm, index=close.index).ewm(alpha=1 / period, adjust=False).mean()

        plus_di = 100 * (plus_dm_n / tr_n.replace(0, np.nan))
        minus_di = 100 * (minus_dm_n / tr_n.replace(0, np.nan))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

    def calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        high = self._series(df, "high")
        low = self._series(df, "low")
        close = self._series(df, "close")

        if talib is not None:
            atr = talib.ATR(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=period)
            return pd.Series(atr, index=close.index)

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        return atr.fillna(method="bfill").fillna(0)

    def calc_bias(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """BIAS(%) = (Close - MA) / MA * 100."""
        close = self._series(df, "close")
        ma = close.rolling(period, min_periods=period).mean()
        bias = (close - ma) / ma.replace(0, np.nan) * 100
        return bias.replace([np.inf, -np.inf], np.nan).fillna(0.0)
