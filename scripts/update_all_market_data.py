#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新全市场 A 股数据到缓存数据库（纯 Tushare/Tinyshare）。
默认按 trade_date 一次性全量拉取，不逐只循环。
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import tinyshare as ts

from env_settings import TUSHARE_TOKEN
from stock_cache_db import StockCache
from update_fund_flow_cache import update_all_market_fund_flow


if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
sys.modules['tushare'] = ts


def _recent_open_trade_dates(pro, limit: int = 10) -> List[str]:
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=45)).strftime('%Y%m%d')
    cal = pro.trade_cal(
        exchange='SSE',
        is_open='1',
        start_date=start_date,
        end_date=end_date,
        fields='cal_date',
    )
    if cal is None or cal.empty:
        return []
    dates = sorted([str(v) for v in cal['cal_date'].tolist() if str(v).strip()], reverse=True)
    return dates[:max(1, limit)]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float):
            if value != value:
                return default
            return value
        out = float(str(value).strip())
        return default if out != out else out
    except Exception:
        return default


def _fetch_all_market_daily(pro, trade_date: Optional[str] = None, retries: int = 3):
    specified_date = (trade_date or '').strip()
    candidate_dates = [specified_date] if specified_date else _recent_open_trade_dates(pro, limit=10)
    if not candidate_dates:
        return None, '', 'no_trade_date'

    last_error = None
    for date in candidate_dates:
        for _ in range(max(1, retries)):
            try:
                df = pro.daily(trade_date=date)
                if df is not None and not df.empty:
                    return df, date, None
            except Exception as e:
                last_error = e
    return None, '', last_error


def _load_name_map(pro) -> Dict[str, str]:
    try:
        df = pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,name',
        )
        if df is None or df.empty:
            return {}
        return {str(r['ts_code']): str(r['name']) for _, r in df.iterrows() if str(r.get('ts_code', '')).strip()}
    except Exception:
        return {}


def update_all_market_data(trade_date: Optional[str] = None) -> int:
    print(f"\n{'='*60}")
    print('开始更新全市场A股数据（Tushare）')
    print(f"{'='*60}\n")

    try:
        pro = ts.pro_api()

        print('正在一次性拉取全市场日线行情...')
        df, used_date, err = _fetch_all_market_daily(pro, trade_date=trade_date, retries=3)
        if df is None or df.empty:
            print(f'更新失败：未获取到全市场数据，error={err}')
            return 0

        print(f'成功拉取 {len(df)} 条，trade_date={used_date}')
        print('正在拉取股票名称映射...')
        name_map = _load_name_map(pro)

        stocks_data = []
        for _, row in df.iterrows():
            ts_code = str(row.get('ts_code', '')).strip()
            if not ts_code:
                continue
            code = ts_code.split('.')[0].zfill(6)
            open_price = _safe_float(row.get('open'))
            close_price = _safe_float(row.get('close'))
            pre_close = _safe_float(row.get('pre_close'))
            pct_chg = _safe_float(row.get('pct_chg'))
            if pct_chg == 0 and pre_close:
                pct_chg = ((close_price - pre_close) / pre_close) * 100

            stocks_data.append({
                'code': code,
                'name': name_map.get(ts_code, code),
                'price': close_price if close_price > 0 else open_price,
                'change_pct': pct_chg,
                'volume': _safe_float(row.get('vol')) * 100,
                'amount': _safe_float(row.get('amount')) * 1000,
            })

        if not stocks_data:
            print('更新失败：转换后无有效股票记录')
            return 0

        cache = StockCache()
        try:
            cache.save_stocks(stocks_data)
        finally:
            cache.close()

        print(f"\n{'='*60}")
        print('数据更新完成')
        print(f'总数: {len(stocks_data)}')
        print(f'trade_date: {used_date}')
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        return len(stocks_data)
    except Exception as e:
        print(f'更新失败: {e}')
        return 0


def update_market_and_fund_flow(trade_date: Optional[str] = None):
    success_count = update_all_market_data(trade_date=trade_date)
    if success_count <= 0:
        return success_count

    print('开始更新全市场资金流缓存...')
    update_all_market_fund_flow(trade_date=trade_date)
    return success_count


if __name__ == '__main__':
    arg_date = sys.argv[1].strip() if len(sys.argv) > 1 else None
    update_market_and_fund_flow(trade_date=arg_date)
