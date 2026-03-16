#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fundamental data provider used by enhanced_long_term_selector.py.

Priority: Tushare -> akshare -> deterministic fallback.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import hashlib
import os

import pandas as pd
from env_settings import TUSHARE_TOKEN


class FundamentalData:
    """Fetch normalized fundamentals for a stock code."""

    def __init__(self, tushare_token: Optional[str] = None):
        self._cache: dict[str, Dict] = {}

        self._token = tushare_token or os.getenv("TUSHARE_TOKEN") or TUSHARE_TOKEN or None

        self._ts = None
        self._pro = None
        self._ak = None

        try:
            import tushare as ts  # type: ignore

            self._ts = ts
            if self._token:
                ts.set_token(self._token)
                self._pro = ts.pro_api()
        except Exception:
            self._ts = None
            self._pro = None

        try:
            import akshare as ak  # type: ignore

            self._ak = ak
        except Exception:
            self._ak = None

    @staticmethod
    def _normalize_code(code: str) -> str:
        code = str(code).strip()
        if code.isdigit():
            return code.zfill(6)
        return code

    @staticmethod
    def _to_ts_code(code: str) -> str:
        return f"{code}.SH" if code.startswith("6") else f"{code}.SZ"

    @staticmethod
    def _to_float(val, default: Optional[float] = None) -> Optional[float]:
        try:
            if val is None:
                return default
            f = float(val)
            if pd.isna(f):
                return default
            return f
        except Exception:
            return default

    @staticmethod
    def _pick(row: pd.Series, candidates: list[str], default: Optional[float] = None) -> Optional[float]:
        for col in candidates:
            if col in row.index:
                v = FundamentalData._to_float(row[col], default=None)
                if v is not None:
                    return v
        return default

    @staticmethod
    def _stable_seed(code: str) -> int:
        digest = hashlib.md5(code.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _fallback(self, code: str) -> Dict:
        seed = self._stable_seed(code)

        roe = 8.0 + (seed % 1800) / 100.0
        profit_growth = -5.0 + (seed % 4500) / 100.0
        dividend_yield = 0.2 + (seed % 580) / 100.0
        revenue_growth = -8.0 + (seed % 5200) / 100.0
        pe = 6.0 + (seed % 5400) / 100.0

        return {
            "code": code,
            "roe": round(roe, 2),
            "profit_growth": round(profit_growth, 2),
            "dividend_yield": round(dividend_yield, 2),
            "revenue_growth": round(revenue_growth, 2),
            "pe": round(pe, 2),
            "source": "fallback",
        }

    def _get_tushare_fundamental(self, code: str) -> Dict:
        if self._pro is None:
            return {}

        ts_code = self._to_ts_code(code)
        out: Dict = {}

        # 1) Valuation / dividend fields from daily_basic
        try:
            trade_date = datetime.now().strftime("%Y%m%d")
            df_basic = self._pro.daily_basic(
                ts_code=ts_code,
                trade_date=trade_date,
                fields="ts_code,trade_date,pe,pe_ttm,dv_ttm,dv_ratio",
            )
            if df_basic is None or df_basic.empty:
                df_basic = self._pro.daily_basic(
                    ts_code=ts_code,
                    fields="ts_code,trade_date,pe,pe_ttm,dv_ttm,dv_ratio",
                    limit=1,
                )
            if df_basic is not None and not df_basic.empty:
                row = df_basic.iloc[0]
                pe = self._pick(row, ["pe_ttm", "pe"])
                dv = self._pick(row, ["dv_ttm", "dv_ratio"])
                if pe is not None:
                    out["pe"] = pe
                if dv is not None:
                    out["dividend_yield"] = dv
        except Exception:
            pass

        # 2) Profitability/growth from fina_indicator
        try:
            df_fi = self._pro.fina_indicator(
                ts_code=ts_code,
                fields="ts_code,end_date,roe,q_roe,netprofit_yoy,q_netprofit_yoy,tr_yoy,q_sales_yoy,or_yoy",
                limit=1,
            )
            if df_fi is not None and not df_fi.empty:
                row = df_fi.iloc[0]
                roe = self._pick(row, ["roe", "q_roe"])
                pg = self._pick(row, ["netprofit_yoy", "q_netprofit_yoy"])
                rg = self._pick(row, ["tr_yoy", "q_sales_yoy", "or_yoy"])
                if roe is not None:
                    out["roe"] = roe
                if pg is not None:
                    out["profit_growth"] = pg
                if rg is not None:
                    out["revenue_growth"] = rg
        except Exception:
            pass

        if out:
            out["source"] = "tushare"
        return out

    def _call_first_available(self, fn_names: list[str], **kwargs):
        if self._ak is None:
            return None
        for name in fn_names:
            fn = getattr(self._ak, name, None)
            if fn is None:
                continue
            try:
                return fn(**kwargs)
            except Exception:
                continue
        return None

    def _get_akshare_fundamental(self, code: str) -> Dict:
        if self._ak is None:
            return {}

        out: Dict = {}

        # 1) Financial indicators (ROE / growth)
        try:
            df_fin = self._call_first_available(
                ["stock_financial_analysis_indicator", "stock_financial_abstract"],
                symbol=code,
            )
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty:
                row = df_fin.iloc[-1]
                roe = self._pick(row, ["净资产收益率(%)", "净资产收益率加权(%)", "ROE"])
                pg = self._pick(row, ["净利润同比增长率(%)", "扣非净利润同比增长率(%)", "净利润同比增长率"])
                rg = self._pick(row, ["营业总收入同比增长率(%)", "营业收入同比增长率(%)", "营收同比增长率(%)"])
                if roe is not None:
                    out["roe"] = roe
                if pg is not None:
                    out["profit_growth"] = pg
                if rg is not None:
                    out["revenue_growth"] = rg
        except Exception:
            pass

        # 2) PE / dividend from indicator endpoints (if available)
        try:
            df_val = self._call_first_available(
                ["stock_a_indicator_lg", "stock_a_lg_indicator", "stock_value_em"],
                symbol=code,
            )
            if isinstance(df_val, pd.DataFrame) and not df_val.empty:
                row = df_val.iloc[-1]
                pe = self._pick(row, ["pe_ttm", "pe", "市盈率(TTM)", "市盈率-动态"])
                dv = self._pick(row, ["dv_ttm", "股息率", "股息率(%)", "股息率TTM"])
                if pe is not None:
                    out["pe"] = pe
                if dv is not None:
                    out["dividend_yield"] = dv
        except Exception:
            pass

        if out:
            out["source"] = "akshare"
        return out

    def get_stock_fundamental(self, code: str) -> Dict:
        code = self._normalize_code(code)
        if code in self._cache:
            return self._cache[code]

        result: Dict = {
            "code": code,
            "roe": None,
            "profit_growth": None,
            "dividend_yield": None,
            "revenue_growth": None,
            "pe": None,
            "source": "",
        }

        used_sources: list[str] = []

        ts_data = self._get_tushare_fundamental(code)
        if ts_data:
            result.update({k: v for k, v in ts_data.items() if k in result and v is not None})
            used_sources.append("tushare")

        missing = [
            "roe",
            "profit_growth",
            "dividend_yield",
            "revenue_growth",
            "pe",
        ]
        missing = [k for k in missing if result.get(k) is None]

        if missing:
            ak_data = self._get_akshare_fundamental(code)
            if ak_data:
                for k in missing:
                    v = ak_data.get(k)
                    if v is not None:
                        result[k] = v
                used_sources.append("akshare")

        if any(result.get(k) is None for k in ["roe", "profit_growth", "dividend_yield", "revenue_growth", "pe"]):
            fb = self._fallback(code)
            for k in ["roe", "profit_growth", "dividend_yield", "revenue_growth", "pe"]:
                if result.get(k) is None:
                    result[k] = fb[k]
            used_sources.append("fallback")

        result["roe"] = round(float(result["roe"]), 2)
        result["profit_growth"] = round(float(result["profit_growth"]), 2)
        result["dividend_yield"] = round(float(result["dividend_yield"]), 2)
        result["revenue_growth"] = round(float(result["revenue_growth"]), 2)
        result["pe"] = round(float(result["pe"]), 2)
        result["source"] = "+".join(dict.fromkeys(used_sources)) if used_sources else "fallback"

        self._cache[code] = result
        return result

    def close(self):
        return None
