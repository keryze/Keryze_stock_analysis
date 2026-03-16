#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List


class MarketAnalysis:
    """Small market overview calculator for the web API."""

    def get_market_overview(self, stocks: List[Dict]) -> Dict:
        total = len(stocks)
        if total == 0:
            return {
                "total": 0,
                "gainers": 0,
                "losers": 0,
                "flat": 0,
                "avg_change_pct": 0.0,
                "top_gainer": None,
                "top_loser": None,
            }

        gainers = [s for s in stocks if float(s.get("change_pct") or 0) > 0]
        losers = [s for s in stocks if float(s.get("change_pct") or 0) < 0]
        flat = total - len(gainers) - len(losers)
        avg_change_pct = sum(float(s.get("change_pct") or 0) for s in stocks) / total

        top_gainer = max(stocks, key=lambda s: float(s.get("change_pct") or -999))
        top_loser = min(stocks, key=lambda s: float(s.get("change_pct") or 999))

        return {
            "total": total,
            "gainers": len(gainers),
            "losers": len(losers),
            "flat": flat,
            "avg_change_pct": round(avg_change_pct, 2),
            "top_gainer": {
                "code": top_gainer.get("code"),
                "name": top_gainer.get("name"),
                "change_pct": float(top_gainer.get("change_pct") or 0),
            },
            "top_loser": {
                "code": top_loser.get("code"),
                "name": top_loser.get("name"),
                "change_pct": float(top_loser.get("change_pct") or 0),
            },
        }
