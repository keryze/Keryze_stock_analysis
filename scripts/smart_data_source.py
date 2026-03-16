#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart data source facade.
Delegates to HybridDataSource and keeps backward-compatible method names.
"""

from datetime import datetime, time as dt_time
from typing import Dict, Optional

import pandas as pd

from hybrid_data_source import HybridDataSource


class SmartDataSourceV2:
    """Backward-compatible wrapper around HybridDataSource."""

    def __init__(self, tushare_token: Optional[str] = None):
        self.hybrid = HybridDataSource(tushare_token)

    def is_trading_time(self) -> bool:
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()

        if weekday >= 5:
            return False

        morning_start = dt_time(9, 15)
        morning_end = dt_time(11, 30)
        afternoon_start = dt_time(13, 0)
        afternoon_end = dt_time(15, 0)
        return (morning_start <= current_time <= morning_end) or (
            afternoon_start <= current_time <= afternoon_end
        )

    def get_realtime_price(self, code: str) -> Optional[Dict]:
        return self.hybrid.get_realtime_price(code)

    def get_realtime_quote(self, code: str) -> Optional[Dict]:
        return self.get_realtime_price(code)

    def is_trading_day(self) -> bool:
        return self.is_trading_time()

    def get_batch_realtime(self, codes: list) -> list:
        return self.hybrid.get_realtime_batch(codes)

    def get_history_data(self, code: str, days: int = 120) -> Optional[pd.DataFrame]:
        return self.hybrid.get_history_data(code, days)

    def get_fund_flow(self, code: str) -> Optional[Dict]:
        return self.hybrid.get_fund_flow(code)

    def close(self):
        self.hybrid.close()


SmartDataSource = SmartDataSourceV2


if __name__ == '__main__':
    ds = SmartDataSource()
    print('trading_time', ds.is_trading_time())
