#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Advanced long-term indicator helpers for enhanced_long_term_selector.py."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
except Exception:  # pragma: no cover
    talib = None


class AdvancedLongTermIndicators:
    """Higher-level indicator and signal fusion utilities."""

    @staticmethod
    def _series(df: pd.DataFrame, name: str) -> pd.Series:
        candidates = [name, name.lower(), name.upper(), name.title()]
        for col in candidates:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
        raise KeyError(f"Missing required column: {name}")

    def calc_dmi(self, df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Return (+DI, -DI, ADX)."""
        high = self._series(df, "high")
        low = self._series(df, "low")
        close = self._series(df, "close")

        if talib is not None:
            plus_di = talib.PLUS_DI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=period)
            minus_di = talib.MINUS_DI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=period)
            adx = talib.ADX(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=period)
            return (
                pd.Series(plus_di, index=close.index).fillna(0),
                pd.Series(minus_di, index=close.index).fillna(0),
                pd.Series(adx, index=close.index).fillna(0),
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

        return plus_di.fillna(0), minus_di.fillna(0), adx.fillna(0)

    def analyze_dmi_signal(self, plus_di: float, minus_di: float, adx: float) -> Dict:
        """Classify DMI state into buy/sell/hold with strength annotation."""
        plus_di = float(plus_di)
        minus_di = float(minus_di)
        adx = float(adx)

        spread = plus_di - minus_di

        if adx >= 30 and spread > 5:
            signal = "strong_buy"
            strength = "strong"
            reason = "DMI强多头且趋势强"
        elif adx >= 20 and spread > 0:
            signal = "buy"
            strength = "medium"
            reason = "DMI多头占优"
        elif adx >= 25 and spread < -5:
            signal = "sell"
            strength = "strong"
            reason = "DMI空头占优且趋势强"
        elif spread < 0:
            signal = "sell"
            strength = "weak"
            reason = "DMI空头占优"
        else:
            signal = "hold"
            strength = "weak"
            reason = "DMI信号中性"

        return {
            "signal": signal,
            "strength": strength,
            "reason": reason,
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "adx": round(adx, 2),
        }

    def optimize_signal_trigger(self, signals: Dict[str, Dict]) -> Dict:
        """Fuse multiple signal blocks into a final decision."""
        buy_votes = 0
        sell_votes = 0
        reasons: List[str] = []

        for name, payload in signals.items():
            sig = str(payload.get("signal", "")).lower()
            if sig in {"strong_buy", "buy"}:
                buy_votes += 2 if sig == "strong_buy" else 1
                reasons.append(f"{name} 看多")
            elif sig in {"sell", "strong_sell"}:
                sell_votes += 2 if sig == "strong_sell" else 1
            elif payload.get("rating") in {"强势上涨", "稳健上涨"}:
                buy_votes += 1
                reasons.append(f"{name} 趋势向上")

        score = buy_votes - sell_votes

        if score >= 4:
            decision = "强烈买入"
        elif score >= 2:
            decision = "买入"
        elif score <= -3:
            decision = "卖出"
        else:
            decision = "观望"

        if not reasons:
            reasons = ["信号共振不足"]

        return {
            "decision": decision,
            "score": float(score),
            "signal_count": int(buy_votes),
            "reasons": reasons,
        }

    def calc_peg_ratio(self, pe: float, growth: float) -> Dict:
        """Calculate PEG with defensive guards."""
        pe = float(pe)
        growth = float(growth)

        if pe <= 0 or growth <= 0:
            return {"peg": None, "pe": pe, "growth": growth}

        # growth is expected to be percentage value like 20 for 20%
        peg = pe / growth
        return {"peg": round(peg, 4), "pe": pe, "growth": growth}
