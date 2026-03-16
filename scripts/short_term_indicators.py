#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Short-term technical indicators used by short_term_selector.py.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import talib
except Exception:  # pragma: no cover
    talib = None


class ShortTermIndicators:
    """Indicator helper for short-term stock selection."""

    @staticmethod
    def _series(df: pd.DataFrame, name: str) -> pd.Series:
        """Get a column in a tolerant way (lower/upper/title)."""
        candidates = [name, name.lower(), name.upper(), name.title()]
        for col in candidates:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
        raise KeyError(f"Missing required column: {name}")

    def calc_rsi(self, df: pd.DataFrame, period: int = 6) -> pd.Series:
        close = self._series(df, "close")
        if talib is not None:
            rsi = talib.RSI(close.values.astype(float), timeperiod=period)
            return pd.Series(rsi, index=close.index)

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calc_kdj(
        self,
        df: pd.DataFrame,
        n: int = 9,
        k_period: int = 3,
        d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        high = self._series(df, "high")
        low = self._series(df, "low")
        close = self._series(df, "close")

        low_n = low.rolling(n, min_periods=1).min()
        high_n = high.rolling(n, min_periods=1).max()
        denominator = (high_n - low_n).replace(0, np.nan)
        rsv = (close - low_n) / denominator * 100
        rsv = rsv.fillna(50)

        k = rsv.ewm(alpha=1 / k_period, adjust=False).mean()
        d = k.ewm(alpha=1 / d_period, adjust=False).mean()
        j = 3 * k - 2 * d

        return k, d, j

    def detect_kdj_cross(self, k: pd.Series, d: pd.Series, j: pd.Series) -> Dict:
        if len(k) < 2 or len(d) < 2 or len(j) < 2:
            return {
                "k": 50.0,
                "d": 50.0,
                "j": 50.0,
                "golden_cross": False,
                "death_cross": False,
                "oversold": False,
                "overbought": False,
                "score": 0,
                "signal": None,
            }

        k_now, k_prev = float(k.iloc[-1]), float(k.iloc[-2])
        d_now, d_prev = float(d.iloc[-1]), float(d.iloc[-2])
        j_now = float(j.iloc[-1])

        golden_cross = k_prev <= d_prev and k_now > d_now
        death_cross = k_prev >= d_prev and k_now < d_now
        oversold = j_now < 20
        overbought = j_now > 80

        score = 0
        signal = None

        if golden_cross and j_now < 50:
            score = 20
            signal = "KDJ金叉"
        elif oversold:
            score = 15
            signal = "KDJ超卖"
        elif death_cross and j_now > 70:
            score = -10
            signal = "KDJ死叉"
        elif overbought:
            score = -5
            signal = "KDJ超买"

        return {
            "k": k_now,
            "d": d_now,
            "j": j_now,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "oversold": oversold,
            "overbought": overbought,
            "score": score,
            "signal": signal,
        }

    def calc_macd_short(
        self,
        df: pd.DataFrame,
        fast: int = 8,
        slow: int = 17,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        close = self._series(df, "close")

        if talib is not None:
            dif, dea, hist = talib.MACD(
                close.values.astype(float),
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal,
            )
            return (
                pd.Series(dif, index=close.index),
                pd.Series(dea, index=close.index),
                pd.Series(hist, index=close.index),
            )

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = dif - dea
        return dif, dea, hist

    def detect_macd_cross(
        self,
        dif: pd.Series,
        dea: pd.Series,
        macd_hist: pd.Series,
    ) -> Dict:
        if len(dif) < 2 or len(dea) < 2 or len(macd_hist) < 2:
            return {
                "dif": 0.0,
                "dea": 0.0,
                "macd_hist": 0.0,
                "golden_cross": False,
                "death_cross": False,
                "signal": None,
            }

        dif_now, dif_prev = float(dif.iloc[-1]), float(dif.iloc[-2])
        dea_now, dea_prev = float(dea.iloc[-1]), float(dea.iloc[-2])
        hist_now, hist_prev = float(macd_hist.iloc[-1]), float(macd_hist.iloc[-2])

        golden_cross = dif_prev <= dea_prev and dif_now > dea_now
        death_cross = dif_prev >= dea_prev and dif_now < dea_now

        signal = None
        if golden_cross:
            signal = "MACD金叉"
        elif death_cross:
            signal = "MACD死叉"
        elif hist_prev <= 0 < hist_now:
            signal = "MACD翻红"
        elif hist_prev >= 0 > hist_now:
            signal = "MACD翻绿"

        return {
            "dif": dif_now,
            "dea": dea_now,
            "macd_hist": hist_now,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "signal": signal,
        }

    def calc_bollinger(
        self,
        df: pd.DataFrame,
        period: int = 10,
        std: float = 1.5,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        close = self._series(df, "close")

        if talib is not None:
            upper, middle, lower = talib.BBANDS(
                close.values.astype(float),
                timeperiod=period,
                nbdevup=std,
                nbdevdn=std,
                matype=0,
            )
            return (
                pd.Series(upper, index=close.index),
                pd.Series(middle, index=close.index),
                pd.Series(lower, index=close.index),
            )

        middle = close.rolling(period, min_periods=period).mean()
        stdv = close.rolling(period, min_periods=period).std(ddof=0)
        upper = middle + std * stdv
        lower = middle - std * stdv
        return upper, middle, lower

    def detect_bollinger_signal(
        self,
        df: pd.DataFrame,
        upper: pd.Series,
        middle: pd.Series,
        lower: pd.Series,
    ) -> Dict:
        close = self._series(df, "close")
        if len(close) < 2:
            return {
                "upper": np.nan,
                "middle": np.nan,
                "lower": np.nan,
                "bandwidth": 0.0,
                "position_pct": 50.0,
                "signal": None,
            }

        c_now, c_prev = float(close.iloc[-1]), float(close.iloc[-2])
        u_now, m_now, l_now = float(upper.iloc[-1]), float(middle.iloc[-1]), float(lower.iloc[-1])

        band = u_now - l_now
        if not np.isfinite(band) or band <= 0:
            position_pct = 50.0
            bandwidth = 0.0
        else:
            position_pct = (c_now - l_now) / band * 100
            position_pct = float(np.clip(position_pct, 0, 100))
            bandwidth = (band / m_now * 100) if m_now else 0.0

        signal = None
        if np.isfinite(l_now) and c_prev < l_now <= c_now:
            signal = "下轨反弹"
        elif np.isfinite(m_now) and c_prev <= m_now < c_now and position_pct < 60:
            signal = "中轨支撑"
        elif np.isfinite(u_now) and c_now >= u_now:
            signal = "触及上轨"
        elif np.isfinite(l_now) and c_now <= l_now:
            signal = "跌破下轨"

        return {
            "upper": u_now,
            "middle": m_now,
            "lower": l_now,
            "bandwidth": float(bandwidth),
            "position_pct": position_pct,
            "signal": signal,
        }

    def detect_volume_surge(self, df: pd.DataFrame, ratio: float = 1.5) -> Dict:
        volume = self._series(df, "volume")
        close = self._series(df, "close")

        if len(volume) < 6 or len(close) < 2:
            return {"volume_ratio": 1.0, "price_change": 0.0, "surge_type": None}

        vol_now = float(volume.iloc[-1])
        vol_ma5 = float(volume.iloc[-6:-1].mean())
        volume_ratio = vol_now / vol_ma5 if vol_ma5 > 0 else 1.0

        price_change = (float(close.iloc[-1]) - float(close.iloc[-2])) / float(close.iloc[-2]) * 100

        surge_type = None
        if volume_ratio >= ratio and price_change > 0:
            surge_type = "放量上涨"
        elif volume_ratio >= ratio and price_change < 0:
            surge_type = "放量下跌"
        elif volume_ratio < 0.8 and price_change > 0:
            surge_type = "缩量上涨"

        return {
            "volume_ratio": float(volume_ratio),
            "price_change": float(price_change),
            "surge_type": surge_type,
        }

    def calc_atr_short(self, df: pd.DataFrame, period: int = 7) -> pd.Series:
        high = self._series(df, "high")
        low = self._series(df, "low")
        close = self._series(df, "close")

        if talib is not None:
            atr = talib.ATR(
                high.values.astype(float),
                low.values.astype(float),
                close.values.astype(float),
                timeperiod=period,
            )
            return pd.Series(atr, index=close.index)

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean()
        return atr

    def calc_trade_points(
        self,
        current_price: float,
        atr: float,
        stop_multiplier: float = 2.0,
        profit_multiplier: float = 3.0,
    ) -> Dict:
        price = float(current_price) if current_price else 0.0
        atr_value = float(atr) if atr and np.isfinite(atr) else 0.0

        if price <= 0:
            return {
                "buy_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "stop_loss_pct": 0.0,
                "take_profit_pct": 0.0,
                "atr": atr_value,
                "atr_pct": 0.0,
                "risk_reward_ratio": 0.0,
            }

        if atr_value <= 0:
            stop_loss = price * 0.97
            take_profit = price * 1.05
        else:
            stop_loss = price - atr_value * stop_multiplier
            take_profit = price + atr_value * profit_multiplier

        stop_loss_pct = (stop_loss - price) / price * 100
        take_profit_pct = (take_profit - price) / price * 100

        risk = price - stop_loss
        reward = take_profit - price
        risk_reward_ratio = reward / risk if risk > 0 else 0.0

        return {
            "buy_price": round(price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "stop_loss_pct": round(stop_loss_pct, 2),
            "take_profit_pct": round(take_profit_pct, 2),
            "atr": round(atr_value, 4),
            "atr_pct": round((atr_value / price * 100) if price > 0 else 0.0, 2),
            "risk_reward_ratio": round(float(risk_reward_ratio), 2),
        }
