#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid data source.
Realtime: Sina during trading hours, akshare as fallback.
History: Tushare first when configured, akshare fallback.
Fund flow: akshare Eastmoney individual fund flow.
get_realtime_price()：实时行情，优先交易时段走新浪，失败再回落到 akshare
get_realtime_batch()：批量实时行情
get_history_data()：历史 K 线，优先 Tushare，否则走 akshare
get_fund_flow()：个股资金流，现在走 akshare 的东方财富接口
"""

from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional

import akshare as ak
import pandas as pd
import requests
import time
import random
import tushare as ts
from stock_cache_db import StockCache
from env_settings import EASTMONEY_COOKIE, TUSHARE_TOKEN


class HybridDataSource:
    """Unified market data adapter."""

    def __init__(self, tushare_token: Optional[str] = None):
        self.tushare_token = tushare_token
        self.tushare_available = False
        self.pro = None
        self.cache = StockCache()
        self.fund_flow_market_cache: Dict[str, str] = {}
        self.eastmoney_cookie = EASTMONEY_COOKIE
        self.eastmoney_cookie_parts = self._parse_cookie_string(EASTMONEY_COOKIE)
        self.eastmoney_st_sn = int(self.eastmoney_cookie_parts.get('st_sn', '1') or '1')
        self.eastmoney_st_psi_middle = self._extract_st_psi_middle(
            self.eastmoney_cookie_parts.get('st_psi', '')
        )

        if self.tushare_token:
            try:
                ts.set_token(self.tushare_token)
                self.pro = ts.pro_api()
                self.pro.trade_cal(exchange='SSE', start_date='20260101', end_date='20260101')
                self.tushare_available = True
                print('Tushare Pro connected')#做一次连通性测试
            except Exception as e:
                print(f'Tushare Pro unavailable: {e}')
                print('Falling back to Sina plus akshare')
        else:
            try:
                self.pro = ts.pro_api()
                self.pro.trade_cal(exchange='SSE', start_date='20260101', end_date='20260101')
                self.tushare_available = True
                print('Tushare Pro connected (global token)')
            except Exception:
                if TUSHARE_TOKEN:
                    try:
                        ts.set_token(TUSHARE_TOKEN)
                        self.pro = ts.pro_api()
                        self.pro.trade_cal(exchange='SSE', start_date='20260101', end_date='20260101')
                        self.tushare_available = True
                        print('Tushare Pro connected (config token)')
                    except Exception:
                        print('Tushare token not configured, using Sina plus akshare')
                else:
                    print('Tushare token not configured, using Sina plus akshare')

    def _is_trading_time(self) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:
            return False

        current_time = now.time()
        morning_start = dt_time(9, 15)
        morning_end = dt_time(11, 30)
        afternoon_start = dt_time(13, 0)
        afternoon_end = dt_time(15, 0)
        return (morning_start <= current_time <= morning_end) or (
            afternoon_start <= current_time <= afternoon_end
        )

    def _infer_market(self, code: str) -> str:
        if code.startswith('6'):
            return 'sh'
        if code.startswith('4') or code.startswith('8'):
            return 'bj'
        return 'sz'

    def _candidate_markets_for_fund_flow(self, code: str) -> List[str]:
        """根据代码生成资金流接口的市场尝试顺序，并带进程内成功缓存。"""
        cached_market = self.fund_flow_market_cache.get(code)
        if cached_market:
            return [cached_market]

        db_market = self.cache.get_market_mapping(code)
        if db_market:
            self.fund_flow_market_cache[code] = db_market
            return [db_market]

        primary = self._infer_market(code)
        if primary == 'bj':
            return ['bj']

        fallback = 'sh' if primary == 'sz' else 'sz'
        return [primary, fallback]

    def get_realtime_price(self, code: str) -> Optional[Dict]:
        if self._is_trading_time():
            result = self._get_sina_realtime(code)
            if result:
                return result
        return self._get_akshare_realtime(code)

    def get_realtime_batch(self, codes: List[str]) -> List[Dict]:
        if self._is_trading_time():
            result = self._get_sina_batch(codes)
            if result:
                return result

        results = []
        for code in codes:
            data = self.get_realtime_price(code)
            if data:
                results.append(data)
        return results

    def get_history_data(self, code: str, days: int = 120) -> Optional[pd.DataFrame]:
        if self.tushare_available:
            result = self._get_tushare_history(code, days)
            if result is not None and not result.empty:
                return result
        return self._get_akshare_history(code, days)

    def get_fund_flow(self, code: str) -> Optional[Dict]:
        return self._get_tushare_fund_flow(code)

    def _to_ts_code(self, code: str) -> str:
        if code.startswith('6'):
            return f'{code}.SH'
        if code.startswith('4') or code.startswith('8'):
            return f'{code}.BJ'
        return f'{code}.SZ'

    def _get_tushare_fund_flow(self, code: str) -> Optional[Dict]:
        if not self.tushare_available or self.pro is None:
            return None

        try:
            ts_code = self._to_ts_code(code)
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
            df = self.pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return None

            row = df.iloc[0]
            buy_lg = pd.to_numeric(row.get('buy_lg_amount'), errors='coerce')
            buy_elg = pd.to_numeric(row.get('buy_elg_amount'), errors='coerce')
            sell_lg = pd.to_numeric(row.get('sell_lg_amount'), errors='coerce')
            sell_elg = pd.to_numeric(row.get('sell_elg_amount'), errors='coerce')
            buy_sm = pd.to_numeric(row.get('buy_sm_amount'), errors='coerce')
            sell_sm = pd.to_numeric(row.get('sell_sm_amount'), errors='coerce')
            net_mf = pd.to_numeric(row.get('net_mf_amount'), errors='coerce')

            if pd.isna(buy_lg) and pd.isna(buy_elg) and pd.isna(sell_lg) and pd.isna(sell_elg) and pd.isna(net_mf):
                return None

            if pd.isna(net_mf):
                main_in_wan = (0.0 if pd.isna(buy_lg) else float(buy_lg)) + (0.0 if pd.isna(buy_elg) else float(buy_elg)) - (0.0 if pd.isna(sell_lg) else float(sell_lg)) - (0.0 if pd.isna(sell_elg) else float(sell_elg))
            else:
                main_in_wan = float(net_mf)

            retail_in_wan = (0.0 if pd.isna(buy_sm) else float(buy_sm)) - (0.0 if pd.isna(sell_sm) else float(sell_sm))
            main_ratio = pd.to_numeric(row.get('net_mf_ratio'), errors='coerce')

            return {
                'code': code,
                'main_in': main_in_wan * 10000,
                'retail_in': retail_in_wan * 10000,
                'main_ratio': 0.0 if pd.isna(main_ratio) else float(main_ratio),
                'date': str(row.get('trade_date') or ''),
                'source': 'tushare',
                'market': self._infer_market(code),
            }
        except Exception:
            return None

    @staticmethod
    def _parse_cookie_string(cookie_string: str) -> Dict[str, str]:
        parts: Dict[str, str] = {}
        for item in cookie_string.split(';'):
            item = item.strip()
            if not item or '=' not in item:
                continue
            key, value = item.split('=', 1)
            parts[key.strip()] = value.strip()
        return parts

    @staticmethod
    def _extract_st_psi_middle(st_psi: str) -> str:
        parts = [part for part in str(st_psi).split('-') if part]
        if len(parts) >= 3:
            return parts[1]
        return '113300300815'

    def _next_dynamic_cookie_string(self) -> str:
        if not self.eastmoney_cookie_parts:
            return ''

        cookie_parts = dict(self.eastmoney_cookie_parts)
        self.eastmoney_st_sn += 1
        now_ms = int(time.time() * 1000)
        random_suffix = random.randint(1000000000, 9999999999)

        cookie_parts['st_sn'] = str(self.eastmoney_st_sn)
        cookie_parts['st_psi'] = f'{now_ms}-{self.eastmoney_st_psi_middle}-{random_suffix}'

        return '; '.join(f'{key}={value}' for key, value in cookie_parts.items())

    def _build_eastmoney_headers(self) -> Dict[str, str]:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
            'Referer': 'https://quote.eastmoney.com/',
        }
        if self.eastmoney_cookie:
            headers['Cookie'] = self._next_dynamic_cookie_string()
        return headers

    def _get_cookie_fund_flow(self, code: str, market: str) -> Optional[Dict]:
        if not self.eastmoney_cookie:
            return None

        market_map = {'sh': '1', 'sz': '0', 'bj': '0'}
        secid_prefix = market_map.get(market)
        if secid_prefix is None:
            return None

        session = requests.Session()
        session.trust_env = False
        url = 'https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get'
        params = {
            'lmt': '0',
            'klt': '101',
            'secid': f'{secid_prefix}.{code}',
            'fields1': 'f1,f2,f3,f7',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65',
            'ut': 'b2884a393a59ad64002292a3e90d46a5',
            '_': str(int(time.time() * 1000)),
        }

        try:
            response = session.get(
                url,
                params=params,
                headers=self._build_eastmoney_headers(),
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get('data') or {}
            klines = data.get('klines') or []
            if not klines:
                return None

            last_line = klines[-1].split(',')
            if len(last_line) < 13:
                return None

            self.fund_flow_market_cache[code] = market
            self.cache.save_market_mapping(code, market)
            return {
                'code': code,
                'main_in': float(last_line[1]),
                'retail_in': float(last_line[2]),
                'main_ratio': float(last_line[6]),
                'date': last_line[0],
                'source': 'eastmoney_cookie',
                'market': market,
            }
        except Exception:
            return None

    def _get_prioritized_fund_flow(self, code: str) -> Optional[Dict]:
        """资金流优先走 Cookie 直连，akshare 作为备用。"""
        candidate_markets = self._candidate_markets_for_fund_flow(code)

        for market in candidate_markets:
            cookie_result = self._get_cookie_fund_flow(code, market)
            if cookie_result:
                return cookie_result

        return self._get_akshare_fund_flow(code)

    def _get_sina_realtime(self, code: str) -> Optional[Dict]:
        try:
            symbol = f"sh{code}" if code.startswith('6') else f"sz{code}"
            url = f'http://hq.sinajs.cn/list={symbol}'
            response = requests.get(url, timeout=3)
            if response.status_code != 200:
                return None

            content = response.text
            if 'var hq_str_' not in content:
                return None

            data_str = content.split('"')[1]
            fields = data_str.split(',')
            if len(fields) < 32:
                return None

            name = fields[0]
            price = float(fields[3])
            prev_close = float(fields[2])
            change_pct = ((price - prev_close) / prev_close) * 100 if prev_close else 0.0
            return {
                'code': code,
                'name': name,
                'price': price,
                'change_pct': change_pct,
                'volume': float(fields[8]),
                'amount': float(fields[9]),
                'open': float(fields[1]),
                'high': float(fields[4]),
                'low': float(fields[5]),
                'prev_close': prev_close,
                'source': 'sina',
            }
        except Exception as e:
            print(f'Sina realtime failed {code}: {e}')
            return None

    def _get_sina_batch(self, codes: List[str]) -> Optional[List[Dict]]:
        try:
            symbols = [(f'sh{code}' if code.startswith('6') else f'sz{code}') for code in codes[:50]]
            url = f"http://hq.sinajs.cn/list={','.join(symbols)}"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return None

            results = []
            lines = response.text.strip().split('\n')
            for i, line in enumerate(lines):
                if 'var hq_str_' not in line or i >= len(codes):
                    continue
                data_str = line.split('"')[1]
                fields = data_str.split(',')
                if len(fields) < 32:
                    continue
                price = float(fields[3])
                prev_close = float(fields[2])
                change_pct = ((price - prev_close) / prev_close) * 100 if prev_close else 0.0
                results.append({
                    'code': codes[i],
                    'name': fields[0],
                    'price': price,
                    'change_pct': change_pct,
                    'volume': float(fields[8]),
                    'amount': float(fields[9]),
                    'source': 'sina_batch',
                })
            return results if results else None
        except Exception as e:
            print(f'Sina batch failed: {e}')
            return None

    def _get_akshare_realtime(self, code: str) -> Optional[Dict]:
        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return None

            stock = df[df['代码'] == code]
            if stock.empty:
                return None

            row = stock.iloc[0]
            return {
                'code': code,
                'name': row['名称'],
                'price': row['最新价'],
                'change_pct': row['涨跌幅'],
                'volume': row['成交量'],
                'amount': row['成交额'],
                'source': 'akshare',
            }
        except Exception as e:
            print(f'akshare realtime failed {code}: {e}')
            return None

    def _get_tushare_history(self, code: str, days: int) -> Optional[pd.DataFrame]:
        try:
            if self.pro is None:
                return None

            ts_code = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime('%Y%m%d')
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df.empty:
                return None

            df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
            df['volume'] = df['volume'] * 100
            df['amount'] = df['amount'] * 1000
            df = df.sort_values('date')
            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].tail(days)
        except Exception as e:
            print(f'Tushare history failed {code}: {e}')
            return None

    def _get_akshare_history(self, code: str, days: int) -> Optional[pd.DataFrame]:
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days * 2)).strftime('%Y%m%d')
            df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start_date, end_date=end_date, adjust='')
            if df is None or df.empty:
                return None

            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
            })
            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].tail(days)
        except Exception as e:
            print(f'akshare history failed {code}: {e}')
            return None

    def _get_akshare_fund_flow(self, code: str) -> Optional[Dict]:
        for market in self._candidate_markets_for_fund_flow(code):
            try:
                df = ak.stock_individual_fund_flow(stock=code, market=market)
                if df is None or df.empty:
                    cookie_result = self._get_cookie_fund_flow(code, market)
                    if cookie_result:
                        return cookie_result
                    continue

                row = df.iloc[-1]
                main_in = pd.to_numeric(row.get('主力净流入-净额'), errors='coerce')
                retail_in = pd.to_numeric(row.get('小单净流入-净额'), errors='coerce')
                main_ratio = pd.to_numeric(row.get('主力净流入-净占比'), errors='coerce')
                trade_date = row.get('日期')
                if pd.isna(main_in):
                    cookie_result = self._get_cookie_fund_flow(code, market)
                    if cookie_result:
                        return cookie_result
                    continue

                self.fund_flow_market_cache[code] = market
                self.cache.save_market_mapping(code, market)
                return {
                    'code': code,
                    'main_in': float(main_in),
                    'retail_in': 0.0 if pd.isna(retail_in) else float(retail_in),
                    'main_ratio': 0.0 if pd.isna(main_ratio) else float(main_ratio),
                    'date': str(trade_date) if trade_date is not None else '',
                    'source': 'akshare_eastmoney',
                    'market': market,
                }
            except Exception:
                cookie_result = self._get_cookie_fund_flow(code, market)
                if cookie_result:
                    return cookie_result
                continue

        return None

    def close(self):
        if self.cache:
            self.cache.close()


_instance = None


def get_hybrid_source(tushare_token: Optional[str] = None) -> HybridDataSource:
    global _instance
    if _instance is None:
        _instance = HybridDataSource(tushare_token)
    return _instance


if __name__ == '__main__':
    ds = HybridDataSource()
    print('trading_time', ds._is_trading_time())
