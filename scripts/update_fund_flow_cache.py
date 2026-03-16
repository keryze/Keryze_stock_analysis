#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新资金流缓存到 stock_cache.db。
默认按 trade_date 一次拉取全市场（Tushare/Tinyshare）。
可选 --watchlist 走旧的逐只更新模式。
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import random
import tinyshare as ts

from env_settings import TUSHARE_TOKEN

random.uniform = lambda a, b: 0
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
sys.modules['tushare'] = ts

from stock_async_fetcher import StockAsyncFetcher
from stock_cache_db import StockCache


SCRIPT_DIR = Path(__file__).resolve().parent
WATCHLIST_PATH = SCRIPT_DIR / 'watchlist.json'


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float):
            if value != value:
                return default
            return value
        text = str(value).strip()
        if not text:
            return default
        out = float(text)
        return default if out != out else out
    except Exception:
        return default


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


def _to_code(ts_code: str) -> str:
    return str(ts_code).split('.')[0].zfill(6)


def update_all_market_fund_flow(trade_date: Optional[str] = None, retries: int = 3) -> int:
    pro = ts.pro_api()
    specified_date = (trade_date or '').strip()
    candidate_dates = [specified_date] if specified_date else _recent_open_trade_dates(pro, limit=10)
    if not candidate_dates:
        print('未获取到可用交易日，无法全量拉取资金流')
        return 1

    df = None
    last_error = None
    date_to_fetch = ''
    for candidate in candidate_dates:
        for _ in range(max(1, retries)):
            try:
                df = pro.moneyflow(trade_date=candidate)
                if df is not None and not df.empty:
                    date_to_fetch = candidate
                    break
            except Exception as e:
                last_error = e
        if df is not None and not df.empty:
            break

    if df is None or df.empty:
        print(f'全量资金流拉取失败，尝试日期={candidate_dates[:5]} error={last_error}')
        return 1

    rows = []
    for _, row in df.iterrows():
        code = _to_code(row.get('ts_code', ''))
        if not code:
            continue

        net_mf_amount = _safe_float(row.get('net_mf_amount'))
        if net_mf_amount == 0:
            buy_lg = _safe_float(row.get('buy_lg_amount'))
            buy_elg = _safe_float(row.get('buy_elg_amount'))
            sell_lg = _safe_float(row.get('sell_lg_amount'))
            sell_elg = _safe_float(row.get('sell_elg_amount'))
            main_in_wan = buy_lg + buy_elg - sell_lg - sell_elg
        else:
            main_in_wan = net_mf_amount

        retail_in_wan = _safe_float(row.get('buy_sm_amount')) - _safe_float(row.get('sell_sm_amount'))
        main_ratio = _safe_float(row.get('net_mf_ratio'))

        rows.append({
            'code': code,
            'main_in': main_in_wan * 10000,
            'retail_in': retail_in_wan * 10000,
            'main_ratio': main_ratio,
        })

    if not rows:
        print(f'全量资金流返回空记录，trade_date={date_to_fetch}')
        return 1

    cache = StockCache()
    try:
        cache.save_fund_flows(rows)
    finally:
        cache.close()

    print(f'全量资金流更新完成: trade_date={date_to_fetch} 成功写入 {len(rows)} 条')
    return 0


def load_watchlist() -> List[str]:
    if not WATCHLIST_PATH.exists():
        return []

    with open(WATCHLIST_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    codes = []
    for item in raw:
        code = str(item).strip()
        if not code:
            continue
        if code.isdigit():
            code = code.zfill(6)
        if code.startswith('3') or code.startswith('688'):
            continue
        codes.append(code)

    return list(dict.fromkeys(codes))


def update_watchlist_fund_flow(start_index: int = 1) -> int:
    codes = load_watchlist()
    if not codes:
        print('未找到可更新的 watchlist 股票池')
        return 1

    if start_index < 1:
        start_index = 1
    if start_index > len(codes):
        print(f'起始序号超出范围: {start_index} > {len(codes)}')
        return 1

    selected_codes = codes[start_index - 1:]

    fetcher = StockAsyncFetcher()
    success_count = 0
    cached_count = 0
    fail_count = 0
    skipped_count = 0
    remote_request_count = 0

    try:
        print(f'开始更新资金流缓存，从第 {start_index} 只开始，共 {len(selected_codes)} 只（总池 {len(codes)}）')
        for offset, code in enumerate(selected_codes, 0):
            idx = start_index + offset
            stock_info = fetcher.cache.get_stock(code)
            stock_name = str((stock_info or {}).get('name') or '')
            if any(flag in stock_name for flag in ['ST', '*ST', '退', 'PT']):
                skipped_count += 1
                print(f'[{idx}/{len(codes)}] {code} {stock_name or ""} 跳过')
                continue

            fund = fetcher.fetch_fund_flow(code, use_cache=True, max_age_hours=8)
            if fund:
                source = str(fund.get('source') or 'unknown')
                if source == 'cache':
                    cached_count += 1
                    print(f'[{idx}/{len(codes)}] {code} 使用缓存 (source={source})')
                    continue

                remote_request_count += 1
                success_count += 1
                print(f'[{idx}/{len(codes)}] {code} 主力净流入 {fund["main_in"] / 10000:.2f}万 (source={source})')
            else:
                remote_request_count += 1
                fail_count += 1
                print(f'[{idx}/{len(codes)}] {code} 资金流获取失败')
    finally:
        fetcher.close()

    print(f'资金流缓存更新完成: 成功 {success_count} | 缓存 {cached_count} | 跳过 {skipped_count} | 失败 {fail_count} | 总计 {len(codes)}')
    return 0 if success_count > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description='更新资金流缓存')
    parser.add_argument('--watchlist', action='store_true', help='按 watchlist 逐只更新（兼容旧模式）')
    parser.add_argument('--start-index', type=int, default=1, help='watchlist 模式起始序号')
    parser.add_argument('--trade-date', type=str, default='', help='全量模式指定交易日，格式 YYYYMMDD')
    args = parser.parse_args()

    if args.watchlist:
        return update_watchlist_fund_flow(start_index=args.start_index)
    return update_all_market_fund_flow(trade_date=args.trade_date or None)


if __name__ == '__main__':
    raise SystemExit(main())
