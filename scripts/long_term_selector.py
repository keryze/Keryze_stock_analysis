#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中长线选股引擎
每日推荐5-10只优质股票
综合多维度指标评分
"""

import pandas as pd
import numpy as np
import tinyshare as ts
from datetime import datetime
import json
from typing import List, Dict, Optional
from pathlib import Path
from stock_cache_db import StockCache
from advanced_indicators import AdvancedIndicators
from market_sentiment import calculate_market_sentiment
from env_settings import TUSHARE_TOKEN
from email_sender import send_email_with_attachment


class LongTermSelector:
    """中长线选股引擎"""
    
    def __init__(self):
        self.params = {
            'history_days_short': 120,
            'history_days_long': 300,
            'history_prefetch_buffer_days': 2,
            'min_required_history_days': 250,
            'recommend_score_core': 72.0,
            'recommend_score_watch': 62.0,
            'recommend_score_floor': 58.0,
            'trend_weight': 15.0,
            'growth_weight': 30.0,
            'moat_weight': 15.0,
            'valuation_weight': 15.0,
            'flow_weight': 10.0,
            'risk_weight': 15.0,
            'market_heat_adjustment_strong': 1.0,
            'market_heat_adjustment_warm': 0.5,
            'market_heat_adjustment_weak': -1.0,
        }
        self.cache = StockCache()
        self.indicators = AdvancedIndicators()
        if TUSHARE_TOKEN:
            ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        self.market_context = self._build_market_context()
        self.sector_market_heat_map: Optional[Dict[str, Dict]] = None
        self.stock_sector_cache: Dict[str, Dict] = {}
        self.ts_code_to_industry: Dict[str, str] = {}
        self.code_to_industry: Dict[str, str] = {}
        self.history_cache: Dict[str, pd.DataFrame] = {}
        self.fundamental_cache: Dict[str, Dict] = {}

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """安全转换数值，过滤None/NaN/无穷大。"""
        parsed = LongTermSelector._safe_float_or_none(value)
        return default if parsed is None else parsed

    @staticmethod
    def _safe_float_or_none(value) -> Optional[float]:
        """安全转换数值，缺失时返回 None，便于评分层区分未知值。"""
        try:
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return None
            return value
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_ts_code(code: str) -> str:
        code = str(code).zfill(6)
        if code.startswith('6'):
            return f'{code}.SH'
        if code.startswith('4') or code.startswith('8'):
            return f'{code}.BJ'
        return f'{code}.SZ'

    def _prepare_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = pd.to_numeric(out['close'], errors='coerce')
        out['ma20'] = close.rolling(20).mean()
        out['ma60'] = close.rolling(60).mean()
        out['ma120'] = close.rolling(120).mean()
        out['ma250'] = close.rolling(250).mean()
        out['ret_20d'] = close.pct_change(20) * 100
        out['ret_60d'] = close.pct_change(60) * 100
        out['ret_120d'] = close.pct_change(120) * 100
        out['ret_250d'] = close.pct_change(250) * 100
        out['high_52w'] = close.rolling(250).max()
        out['low_52w'] = close.rolling(250).min()
        out['drawdown_from_52w_high'] = np.where(
            out['high_52w'] > 0,
            (close / out['high_52w'] - 1.0) * 100,
            np.nan,
        )
        out['range_position_52w'] = np.where(
            (out['high_52w'] - out['low_52w']) > 0,
            (close - out['low_52w']) / (out['high_52w'] - out['low_52w']),
            np.nan,
        )
        return out

    def _get_history_data_tushare(self, code: str, days: Optional[int] = None) -> Optional[pd.DataFrame]:
        days = days or int(self.params['history_days_long'])
        code = str(code).zfill(6)
        cached = self.history_cache.get(code)
        if cached is not None and not cached.empty:
            return cached.tail(days).copy()

        ts_code = self._to_ts_code(code)
        end_date = datetime.now().strftime('%Y%m%d')
        buffer_days = int(self.params['history_prefetch_buffer_days'])
        start_date = (datetime.now() - pd.Timedelta(days=days * buffer_days)).strftime('%Y%m%d')
        try:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return None
            df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0) * 100
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0) * 1000
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.sort_values('date')
            out = self._prepare_history_features(
                df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
            ).tail(days)
            self.history_cache[code] = out.copy()
            return out
        except Exception:
            return None

    def _prefetch_history_data_tushare(self, codes: List[str], days: Optional[int] = None) -> None:
        """按交易日批量拉取 watchlist 历史，减少逐只请求。"""
        days = days or int(self.params['history_days_long'])
        uniq_codes = sorted({str(c).zfill(6) for c in codes if str(c).strip()})
        if not uniq_codes:
            return

        ts_codes = {self._to_ts_code(code): code for code in uniq_codes}
        end_date = datetime.now().strftime('%Y%m%d')
        buffer_days = int(self.params['history_prefetch_buffer_days'])
        min_required_days = int(self.params['min_required_history_days'])
        start_date = (datetime.now() - pd.Timedelta(days=days * buffer_days)).strftime('%Y%m%d')

        try:
            cal = self.pro.trade_cal(
                exchange='SSE',
                is_open='1',
                start_date=start_date,
                end_date=end_date,
                fields='cal_date',
            )
            if cal is None or cal.empty:
                return

            trade_dates = sorted([str(v) for v in cal['cal_date'].tolist() if str(v).strip()])
            parts = []
            target_ts_codes = set(ts_codes.keys())
            total_dates = len(trade_dates)
            for idx, trade_date in enumerate(trade_dates, 1):
                if idx == 1 or idx % 10 == 0 or idx == total_dates:
                    print(f"[long_term] 历史预取 {idx}/{total_dates} trade_date={trade_date}", flush=True)
                df = self.pro.daily(trade_date=trade_date)
                if df is None or df.empty:
                    continue
                df = df[df['ts_code'].isin(target_ts_codes)]
                if df.empty:
                    continue
                parts.append(df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']])

            if not parts:
                return

            merged = pd.concat(parts, ignore_index=True)
            for ts_code, group in merged.groupby('ts_code'):
                code = ts_codes.get(str(ts_code))
                if not code:
                    continue
                g = group.rename(columns={'trade_date': 'date', 'vol': 'volume'}).copy()
                g['volume'] = pd.to_numeric(g['volume'], errors='coerce').fillna(0.0) * 100
                g['amount'] = pd.to_numeric(g['amount'], errors='coerce').fillna(0.0) * 1000
                for col in ['open', 'high', 'low', 'close']:
                    g[col] = pd.to_numeric(g[col], errors='coerce')
                g = g.sort_values('date')
                if len(g) < min_required_days:
                    continue
                prepared = self._prepare_history_features(
                    g[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
                ).tail(days).copy()
                self.history_cache[code] = prepared
            print(f"[long_term] 历史预取完成，缓存 {len(self.history_cache)} 只股票", flush=True)
        except Exception:
            return

    @staticmethod
    def _clip01(value: float) -> float:
        """限制到0-1区间。"""
        return max(0.0, min(1.0, value))

    def _normalize_positive(self, value: float, low: float, high: float) -> float:
        """正向标准化，小于low记0，大于high记1。"""
        if high <= low:
            return 0.0
        return self._clip01((value - low) / (high - low))

    def _normalize_symmetric(self, value: float, scale: float) -> float:
        """
        对正负值做平滑标准化。
        0 -> 0.5，明显为正趋近1，明显为负趋近0。
        """
        if scale <= 0:
            return 0.5
        return self._clip01((np.tanh(value / scale) + 1.0) / 2.0)

    @staticmethod
    def _rescale_score(value: float, old_max: float, new_max: float) -> float:
        if old_max <= 0:
            return 0.0
        return max(0.0, min(new_max, value / old_max * new_max))

    def _get_score_weights(self) -> Dict[str, float]:
        return {
            'growth': self._safe_float(self.params.get('growth_weight'), 30.0),
            'quality_moat_proxy': self._safe_float(self.params.get('moat_weight'), 15.0),
            'valuation': self._safe_float(self.params.get('valuation_weight'), 15.0),
            'trend': self._safe_float(self.params.get('trend_weight'), 15.0),
            'fund': self._safe_float(self.params.get('flow_weight'), 10.0),
            'risk': self._safe_float(self.params.get('risk_weight'), 15.0),
        }

    @staticmethod
    def _get_score_raw_max() -> Dict[str, float]:
        return {
            'growth': 30.0,
            'quality_moat_proxy': 15.0,
            'valuation': 15.0,
            'trend': 15.0,
            'fund': 10.0,
            'risk': 15.0,
        }

    def _weight_component_scores(self, raw_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        weights = self._get_score_weights()
        raw_max = self._get_score_raw_max()
        weighted_scores: Dict[str, float] = {}
        normalized_scores: Dict[str, float] = {}
        for name, raw_value in raw_scores.items():
            raw = self._safe_float(raw_value)
            module_max = self._safe_float(raw_max.get(name), 0.0)
            weight = self._safe_float(weights.get(name), module_max)
            normalized = self._clip01(raw / module_max) if module_max > 0 else 0.0
            normalized_scores[name] = round(normalized, 4)
            weighted_scores[name] = round(normalized * weight, 2)

        return {
            'weighted': weighted_scores,
            'normalized': normalized_scores,
            'weights': weights,
            'raw_max': raw_max,
        }

    @staticmethod
    def _industry_profile_map() -> Dict[str, Dict[str, float]]:
        # 未来3年增长、行业集中度、政策方向（0-5），白名单优先
        return {
            'AI算力': {'future_growth': 4.8, 'concentration': 3.8, 'policy': 4.2},
            '算力': {'future_growth': 4.7, 'concentration': 3.7, 'policy': 4.1},
            '电网': {'future_growth': 4.2, 'concentration': 4.0, 'policy': 4.6},
            '军工': {'future_growth': 4.0, 'concentration': 3.5, 'policy': 4.8},
            '创新药': {'future_growth': 4.5, 'concentration': 3.0, 'policy': 4.3},
        }

    @staticmethod
    def _industry_long_term_profiles() -> Dict[str, Dict[str, object]]:
        return {
            'AI算力': {
                'long_term_space': 5.0,
                'cycle_risk': 2.5,
                'policy_support': 4.0,
                'competition_intensity': 3.0,
                'capital_intensity': 4.5,
                'cashflow_visibility': 2.8,
                'cycle_type': 'growth',
                'comment': '需求扩张明确但资本开支重，兑现节奏要跟踪',
            },
            '算力': {
                'long_term_space': 4.8,
                'cycle_risk': 2.8,
                'policy_support': 4.0,
                'competition_intensity': 3.2,
                'capital_intensity': 4.4,
                'cashflow_visibility': 2.8,
                'cycle_type': 'growth',
                'comment': '受益于算力资本开支与国产替代',
            },
            '高端制造': {
                'long_term_space': 4.4,
                'cycle_risk': 2.7,
                'policy_support': 4.1,
                'competition_intensity': 3.1,
                'capital_intensity': 3.8,
                'cashflow_visibility': 3.3,
                'cycle_type': 'quality',
                'comment': '技术壁垒和国产替代共振，但资本开支不低',
            },
            '工控自动化': {
                'long_term_space': 4.3,
                'cycle_risk': 2.4,
                'policy_support': 3.8,
                'competition_intensity': 2.8,
                'capital_intensity': 3.1,
                'cashflow_visibility': 3.8,
                'cycle_type': 'quality',
                'comment': '自动化渗透率提升，商业模式偏稳健',
            },
            '电网': {
                'long_term_space': 4.2,
                'cycle_risk': 1.8,
                'policy_support': 4.8,
                'competition_intensity': 2.8,
                'capital_intensity': 3.0,
                'cashflow_visibility': 4.1,
                'cycle_type': 'stable',
                'comment': '设备更新与新型电力系统驱动，订单可见度较好',
            },
            '军工': {
                'long_term_space': 4.1,
                'cycle_risk': 2.2,
                'policy_support': 4.9,
                'competition_intensity': 2.6,
                'capital_intensity': 3.7,
                'cashflow_visibility': 3.0,
                'cycle_type': 'growth',
                'comment': '订单兑现和节奏管理重要，行业壁垒相对较高',
            },
            '创新药': {
                'long_term_space': 4.7,
                'cycle_risk': 3.2,
                'policy_support': 4.2,
                'competition_intensity': 3.5,
                'capital_intensity': 4.0,
                'cashflow_visibility': 2.3,
                'cycle_type': 'growth',
                'comment': '产品周期与研发兑现差异大，空间大但波动高',
            },
            '品牌消费': {
                'long_term_space': 3.8,
                'cycle_risk': 1.9,
                'policy_support': 2.8,
                'competition_intensity': 2.7,
                'capital_intensity': 2.1,
                'cashflow_visibility': 4.5,
                'cycle_type': 'quality',
                'comment': '品牌力和渠道力决定护城河，现金流通常更稳',
            },
            '公用事业': {
                'long_term_space': 2.9,
                'cycle_risk': 1.4,
                'policy_support': 4.2,
                'competition_intensity': 1.8,
                'capital_intensity': 3.6,
                'cashflow_visibility': 4.4,
                'cycle_type': 'dividend',
                'comment': '增长有限但现金流可见性高，适合红利防守',
            },
            '红利资源': {
                'long_term_space': 2.8,
                'cycle_risk': 4.1,
                'policy_support': 2.7,
                'competition_intensity': 2.3,
                'capital_intensity': 3.9,
                'cashflow_visibility': 3.7,
                'cycle_type': 'dividend',
                'comment': '分红吸引力较强，但商品价格周期风险不能忽视',
            },
            '出海制造': {
                'long_term_space': 4.3,
                'cycle_risk': 2.8,
                'policy_support': 3.2,
                'competition_intensity': 3.0,
                'capital_intensity': 3.3,
                'cashflow_visibility': 3.6,
                'cycle_type': 'growth',
                'comment': '海外份额提升带来成长，但汇率与贸易环境有扰动',
            },
        }

    def _build_market_context(self) -> Dict:
        """市场背景展示模块，不再承担策略门控职责。"""
        sentiment = calculate_market_sentiment(use_demo_data=False)
        if sentiment.get('stats', {}).get('total', 0) == 0:
            sentiment = calculate_market_sentiment(use_demo_data=True)

        score = float(sentiment.get('score', 50.0) or 50.0)
        stats = sentiment.get('stats', {})
        up_ratio = self._safe_float(stats.get('up_ratio'), 50.0)
        avg_change = self._safe_float(stats.get('avg_change'))
        avg_volatility = self._safe_float(stats.get('avg_volatility'))
        limit_up = int(stats.get('limit_up', 0) or 0)
        limit_down = int(stats.get('limit_down', 0) or 0)
        strong_stocks = int(stats.get('strong_stocks', 0) or 0)
        weak_stocks = int(stats.get('weak_stocks', 0) or 0)
        total = max(int(stats.get('total', 0) or 0), 1)
        strong_ratio = strong_stocks / total * 100
        weak_ratio = weak_stocks / total * 100

        regime = 'neutral'

        if score < 35 or avg_change < -1.5 or limit_down > limit_up * 1.5:
            regime = 'defensive'
        elif score < 45 or up_ratio < 42 or weak_ratio > strong_ratio + 4:
            regime = 'weak'
        elif score >= 65 and up_ratio >= 55 and limit_up > limit_down * 2:
            regime = 'strong'

        position_advice = {
            'strong': 'normal_scale_in',
            'neutral': 'selective_scale_in',
            'weak': 'slow_scale_in',
            'defensive': 'observe_first',
        }.get(regime, 'selective_scale_in')

        return {
            'regime': regime,
            'position_advice': position_advice,
            'sentiment': sentiment,
            'summary': {
                'score': score,
                'up_ratio': up_ratio,
                'avg_change': avg_change,
                'avg_volatility': avg_volatility,
                'limit_up': limit_up,
                'limit_down': limit_down,
                'strong_ratio': round(strong_ratio, 2),
                'weak_ratio': round(weak_ratio, 2),
            },
        }

    @staticmethod
    def _normalize_sector_name(name: Optional[str]) -> str:
        if not name:
            return ''
        normalized = str(name).strip()
        for suffix in ['Ⅰ', 'Ⅱ', 'Ⅲ', 'A', 'B']:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        return normalized

    def _match_industry_long_term_profile(self, sector_name: str) -> Optional[Dict[str, object]]:
        sector_name = str(sector_name or '').strip()
        if not sector_name:
            return None
        for key, profile in self._industry_long_term_profiles().items():
            if key in sector_name:
                return {'matched_key': key, **profile}
        return None

    def _load_sector_market_heat(self) -> Dict[str, Dict]:
        if self.sector_market_heat_map is not None:
            return self.sector_market_heat_map

        heat_map: Dict[str, Dict] = {}
        try:
            stock_basic = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,industry',
            )
            if stock_basic is None or stock_basic.empty:
                self.sector_market_heat_map = {}
                return self.sector_market_heat_map

            stock_basic = stock_basic.copy()
            stock_basic['industry'] = stock_basic['industry'].astype(str).map(self._normalize_sector_name)
            stock_basic = stock_basic[stock_basic['industry'].astype(str).str.len() > 0]
            self.ts_code_to_industry = {
                str(r['ts_code']): str(r['industry'])
                for _, r in stock_basic.iterrows()
                if str(r.get('ts_code', '')).strip() and str(r.get('industry', '')).strip()
            }
            self.code_to_industry = {
                str(ts_code).split('.')[0].zfill(6): industry
                for ts_code, industry in self.ts_code_to_industry.items()
            }

            trade_cal = self.pro.trade_cal(
                exchange='SSE',
                is_open='1',
                start_date=(datetime.now() - pd.Timedelta(days=45)).strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d'),
                fields='cal_date',
            )
            if trade_cal is None or trade_cal.empty:
                self.sector_market_heat_map = {}
                return self.sector_market_heat_map

            trade_date = max([str(v) for v in trade_cal['cal_date'].tolist() if str(v).strip()])
            daily = self.pro.daily(trade_date=trade_date)
            if daily is None or daily.empty:
                self.sector_market_heat_map = {}
                return self.sector_market_heat_map

            merged = daily[['ts_code', 'pct_chg']].copy().merge(
                stock_basic[['ts_code', 'industry']], on='ts_code', how='inner'
            )
            if merged.empty:
                self.sector_market_heat_map = {}
                return self.sector_market_heat_map

            merged['pct_chg'] = pd.to_numeric(merged['pct_chg'], errors='coerce').fillna(0.0)
            agg = merged.groupby('industry', as_index=False).agg(
                change_pct=('pct_chg', 'mean'),
                up_count=('pct_chg', lambda s: int((s > 0).sum())),
                down_count=('pct_chg', lambda s: int((s < 0).sum())),
                sample_size=('pct_chg', 'count'),
            )
            agg = agg.sort_values('change_pct', ascending=False).reset_index(drop=True)

            total = len(agg)
            for idx, row in agg.iterrows():
                name = str(row.get('industry', '')).strip()
                if not name:
                    continue
                change_pct = self._safe_float(row.get('change_pct'))
                up_count = int(row.get('up_count') or 0)
                down_count = int(row.get('down_count') or 0)
                sample_size = int(row.get('sample_size') or 0)
                heat_score = change_pct * 1.8 + min(4.0, max(-4.0, (up_count - down_count) / 15.0))
                rank_pct = (idx + 1) / total if total else 1.0
                if rank_pct <= 0.1:
                    level = 'strong'
                elif rank_pct <= 0.3:
                    level = 'warm'
                elif rank_pct >= 0.9:
                    level = 'weak'
                else:
                    level = 'neutral'
                heat_map[name] = {
                    'name': name,
                    'change_pct': round(change_pct, 2),
                    'rank': idx + 1,
                    'rank_pct': round(rank_pct, 4),
                    'up_count': up_count,
                    'down_count': down_count,
                    'sample_size': sample_size,
                    'heat_score': round(heat_score, 2),
                    'level': level,
                    'trade_date': trade_date,
                }
        except Exception:
            heat_map = {}

        self.sector_market_heat_map = heat_map
        return heat_map

    def _load_sector_strengths(self) -> Dict[str, Dict]:
        """兼容旧调用名，实际返回行业市场热度。"""
        return self._load_sector_market_heat()

    def _get_stock_sector_info(self, code: str) -> Dict:
        if code in self.stock_sector_cache:
            return self.stock_sector_cache[code]

        info = {
            'sector': None,
            'normalized_sector': '',
            'market_heat': None,
            'long_term_profile': None,
            'cycle_type': 'quality',
            'industry_prior_type': 'quality',
            'matched_profile_key': None,
            'strength': None,
        }
        try:
            if not self.sector_market_heat_map:
                self._load_sector_market_heat()
            sector = self.code_to_industry.get(str(code).zfill(6))
            normalized = self._normalize_sector_name(sector)
            market_heat = self._load_sector_market_heat().get(normalized)
            long_term_profile = self._match_industry_long_term_profile(normalized)
            info = {
                'sector': sector,
                'normalized_sector': normalized,
                'market_heat': market_heat,
                'long_term_profile': long_term_profile,
                'cycle_type': str((long_term_profile or {}).get('cycle_type') or 'quality'),
                'industry_prior_type': str((long_term_profile or {}).get('cycle_type') or 'quality'),
                'matched_profile_key': (long_term_profile or {}).get('matched_key'),
                'strength': market_heat,
            }
        except Exception:
            pass

        self.stock_sector_cache[code] = info
        return info

    def _score_trend_quality(self, df: pd.DataFrame) -> Dict:
        current = df.iloc[-1]
        current_close = self._safe_float(current.get('close'))
        ma60 = self._safe_float(current.get('ma60'))
        ma120 = self._safe_float(current.get('ma120'))
        ma250 = self._safe_float(current.get('ma250'))
        ret_60d = self._safe_float(current.get('ret_60d'))
        ret_120d = self._safe_float(current.get('ret_120d'))
        drawdown_52w = self._safe_float(current.get('drawdown_from_52w_high'))
        range_position_52w = self._safe_float(current.get('range_position_52w'))

        score = 0.0
        reasons = []

        long_ma_structure = 0.0
        if ma60 > 0 and ma120 > 0 and ma250 > 0:
            if current_close >= ma60 >= ma120 >= ma250:
                long_ma_structure = 1.0
                reasons.append('长期均线结构顺畅')
            elif current_close >= ma120 and ma60 >= ma120 >= ma250:
                long_ma_structure = 0.75
            elif current_close >= ma120 and ma250 > 0:
                long_ma_structure = 0.55
            elif current_close >= ma250:
                long_ma_structure = 0.4
        score += 4.5 * long_ma_structure

        price_position = 0.0
        if ma120 > 0 and ma250 > 0:
            above_ma120 = self._clip01((current_close / ma120 - 0.9) / 0.25)
            above_ma250 = self._clip01((current_close / ma250 - 0.9) / 0.35)
            price_position = 0.55 * above_ma120 + 0.45 * above_ma250
        score += 3.75 * price_position

        return_health = 0.0
        return_health += 0.5 * self._normalize_positive(ret_120d, 0.0, 35.0)
        return_health += 0.5 * self._normalize_positive(ret_60d, -5.0, 20.0)
        score += 3.0 * return_health

        drawdown_health = 0.0
        if drawdown_52w != 0:
            drawdown_health = self._clip01((drawdown_52w + 35.0) / 35.0)
        if range_position_52w > 0:
            drawdown_health = (drawdown_health + self._clip01(range_position_52w)) / 2.0
        score += 2.25 * drawdown_health

        overheat_penalty = 0.0
        if ret_60d > 35:
            overheat_penalty += 1.5
            reasons.append('60日涨幅偏快')
        if ret_120d > 60:
            overheat_penalty += 1.5
            reasons.append('120日涨幅偏快')
        if drawdown_52w > -3:
            overheat_penalty += 1.0
            reasons.append('接近52周高点，回撤保护较弱')
        score -= overheat_penalty

        score = max(0.0, min(15.0, score))
        rating = '长期趋势健康' if score >= 11 else '中长期平衡' if score >= 7 else '趋势一般'
        return {
            'score': round(score, 2),
            'close': round(current_close, 4),
            'ma60': round(ma60, 4),
            'ma120': round(ma120, 4),
            'ma250': round(ma250, 4),
            'returns_60d': round(ret_60d, 2),
            'returns_120d': round(ret_120d, 2),
            'drawdown_from_52w_high': round(drawdown_52w, 2),
            'range_position_52w': round(range_position_52w, 3),
            'reasons': reasons,
            'rating': rating,
        }

    def _get_tushare_fundamental_profile(self, code: str) -> Dict:
        """优先使用 Tushare 拉取基本面，并补充 ROIC、现金流质量。"""
        code = str(code).zfill(6)
        if code in self.fundamental_cache:
            return self.fundamental_cache[code]

        ts_code = self._to_ts_code(code)
        out = {
            'code': code,
            'roe': None,
            'profit_growth': None,
            'revenue_growth': None,
            'dividend_yield': None,
            'pe': None,
            'roic': None,
            'ocf': None,
            'net_profit': None,
            'ocf_np_ratio': None,
            'source': 'tushare',
        }

        try:
            trade_date = datetime.now().strftime('%Y%m%d')
            df_basic = self.pro.daily_basic(
                ts_code=ts_code,
                trade_date=trade_date,
                fields='ts_code,trade_date,pe,pe_ttm,dv_ttm,dv_ratio',
            )
            if df_basic is None or df_basic.empty:
                df_basic = self.pro.daily_basic(
                    ts_code=ts_code,
                    fields='ts_code,trade_date,pe,pe_ttm,dv_ttm,dv_ratio',
                    limit=1,
                )
            if df_basic is not None and not df_basic.empty:
                row = df_basic.iloc[0]
                out['pe'] = self._safe_float_or_none(row.get('pe_ttm'))
                if out['pe'] is None:
                    out['pe'] = self._safe_float_or_none(row.get('pe'))
                out['dividend_yield'] = self._safe_float_or_none(row.get('dv_ttm'))
                if out['dividend_yield'] is None:
                    out['dividend_yield'] = self._safe_float_or_none(row.get('dv_ratio'))
        except Exception:
            pass

        try:
            df_fi = self.pro.fina_indicator(
                ts_code=ts_code,
                fields='ts_code,end_date,roe,q_roe,netprofit_yoy,q_netprofit_yoy,tr_yoy,q_sales_yoy,or_yoy,roic',
                limit=1,
            )
            if df_fi is not None and not df_fi.empty:
                row = df_fi.iloc[0]
                out['roe'] = self._safe_float_or_none(row.get('roe'))
                if out['roe'] is None:
                    out['roe'] = self._safe_float_or_none(row.get('q_roe'))
                out['profit_growth'] = self._safe_float_or_none(row.get('netprofit_yoy'))
                if out['profit_growth'] is None:
                    out['profit_growth'] = self._safe_float_or_none(row.get('q_netprofit_yoy'))
                out['revenue_growth'] = self._safe_float_or_none(row.get('tr_yoy'))
                if out['revenue_growth'] is None:
                    out['revenue_growth'] = self._safe_float_or_none(row.get('q_sales_yoy'))
                if out['revenue_growth'] is None:
                    out['revenue_growth'] = self._safe_float_or_none(row.get('or_yoy'))
                out['roic'] = self._safe_float_or_none(row.get('roic'))
        except Exception:
            pass

        # 现金流质量：经营现金流 / 归母净利润
        try:
            df_cf = self.pro.cashflow(ts_code=ts_code, fields='ts_code,end_date,n_cashflow_act', limit=1)
            if df_cf is not None and not df_cf.empty:
                out['ocf'] = self._safe_float_or_none(df_cf.iloc[0].get('n_cashflow_act'))
        except Exception:
            pass

        try:
            df_inc = self.pro.income(ts_code=ts_code, fields='ts_code,end_date,n_income_attr_p', limit=1)
            if df_inc is not None and not df_inc.empty:
                out['net_profit'] = self._safe_float_or_none(df_inc.iloc[0].get('n_income_attr_p'))
        except Exception:
            pass

        if out['ocf'] is not None and out['net_profit'] not in [None, 0]:
            out['ocf_np_ratio'] = out['ocf'] / out['net_profit']

        self.fundamental_cache[code] = out
        return out

    def _score_quality_moat_proxy(self, data: Dict, sector_info: Dict) -> Dict:
        """质量/护城河代理分（ROIC + 现金流质量 + 行业长期属性），满分 15。"""
        roic = self._safe_float(data.get('roic'))
        ocf_np_ratio = self._safe_float(data.get('ocf_np_ratio'))
        sector_name = str((sector_info or {}).get('normalized_sector') or '')
        long_term_profile = (sector_info or {}).get('long_term_profile') or {}

        score = 0.0
        reasons: List[str] = []

        if roic >= 15:
            score += 6.0
            reasons.append(f'ROIC较高 {roic:.1f}%')
        elif roic >= 10:
            score += 4.5
        elif roic >= 6:
            score += 2.5
        elif roic > 0:
            score += 1.0

        if ocf_np_ratio >= 1.2:
            score += 4.0
            reasons.append(f'现金流质量高 {ocf_np_ratio:.2f}')
        elif ocf_np_ratio >= 1.0:
            score += 3.0
        elif ocf_np_ratio >= 0.8:
            score += 1.5
        elif ocf_np_ratio > 0:
            score += 0.5
        elif ocf_np_ratio < 0:
            score -= 2.0

        if long_term_profile:
            matched_key = str(long_term_profile.get('matched_key') or sector_name or '行业画像')
            reasons.append(f'行业长期画像匹配: {matched_key}')
            industry_score = (
                self._safe_float(long_term_profile.get('long_term_space')) * 0.32
                + self._safe_float(long_term_profile.get('policy_support')) * 0.20
                + self._safe_float(long_term_profile.get('cashflow_visibility')) * 0.23
                + (5.0 - self._safe_float(long_term_profile.get('competition_intensity'))) * 0.15
                + (5.0 - self._safe_float(long_term_profile.get('capital_intensity'))) * 0.10
            )  # 0-5
            score += industry_score
        else:
            industry_score = 0.0

        score = max(0.0, min(15.0, score))
        return {
            'score': round(score, 2),
            'roic': round(roic, 2) if data.get('roic') is not None else None,
            'ocf_np_ratio': round(ocf_np_ratio, 3) if data.get('ocf_np_ratio') is not None else None,
            'industry_score': round(industry_score, 2),
            'industry': sector_name or None,
            'matched_profile_key': long_term_profile.get('matched_key'),
            'reasons': reasons,
        }

    def _score_moat_proxy(self, data: Dict, sector_info: Dict) -> Dict:
        """兼容旧调用名，实际返回质量/护城河代理分。"""
        return self._score_quality_moat_proxy(data, sector_info)

    def _score_growth_quality(self, data: Dict) -> Dict:
        roe = self._safe_float_or_none(data.get('roe'))
        roic = self._safe_float_or_none(data.get('roic'))
        profit_growth = self._safe_float_or_none(data.get('profit_growth'))
        revenue_growth = self._safe_float_or_none(data.get('revenue_growth'))
        dividend_yield = self._safe_float_or_none(data.get('dividend_yield'))
        ocf_np_ratio = self._safe_float_or_none(data.get('ocf_np_ratio'))

        roe_score = 8.0 * self._normalize_positive(self._safe_float(roe), 8.0, 25.0) if roe is not None else 4.0
        roic_score = 7.0 * self._normalize_positive(self._safe_float(roic), 6.0, 20.0) if roic is not None else 3.0
        revenue_score = 5.0 * self._normalize_positive(self._safe_float(revenue_growth), 0.0, 25.0) if revenue_growth is not None else 2.5
        profit_score = 5.0 * self._normalize_positive(self._safe_float(profit_growth), 0.0, 30.0) if profit_growth is not None else 2.5

        growth_gap = None
        match_score = 2.5
        if profit_growth is not None and revenue_growth is not None:
            growth_gap = abs(profit_growth - revenue_growth)
            if revenue_growth < 0 < profit_growth:
                match_score = 0.5
            else:
                match_score = 5.0 * self._clip01(1.0 - growth_gap / 40.0)

        cash_realization_score = 0.0
        if ocf_np_ratio is not None:
            cash_realization_score = 3.0 * self._clip01(ocf_np_ratio / 1.2)
        elif dividend_yield is not None and dividend_yield > 2.0:
            cash_realization_score = 1.5
        else:
            cash_realization_score = 1.0

        score = roe_score + roic_score + revenue_score + profit_score + match_score + cash_realization_score
        score = max(0.0, min(30.0, score))

        return {
            'score': round(score, 2),
            'roe': round(self._safe_float(roe), 2),
            'roic': round(roic, 2) if roic is not None else None,
            'profit_growth': round(self._safe_float(profit_growth), 2),
            'revenue_growth': round(self._safe_float(revenue_growth), 2),
            'dividend_yield': round(self._safe_float(dividend_yield), 2),
            'ocf_np_ratio': round(ocf_np_ratio, 3) if ocf_np_ratio is not None else None,
            'growth_gap': round(self._safe_float(growth_gap), 2) if growth_gap is not None else None,
            'data_confidence': round(
                sum(v is not None for v in [roe, roic, profit_growth, revenue_growth, ocf_np_ratio]) / 5.0,
                2,
            ),
            'source': data.get('source', ''),
        }

    def _build_stock_profile(self, data: Dict, sector_info: Optional[Dict] = None) -> Dict[str, object]:
        sector_info = sector_info or {}
        long_term_profile = sector_info.get('long_term_profile') or {}
        sector_name = str(sector_info.get('normalized_sector') or '')

        profit_growth = self._safe_float_or_none(data.get('profit_growth'))
        revenue_growth = self._safe_float_or_none(data.get('revenue_growth'))
        dividend_yield = self._safe_float_or_none(data.get('dividend_yield'))
        roic = self._safe_float_or_none(data.get('roic'))
        roe = self._safe_float_or_none(data.get('roe'))
        ocf_np_ratio = self._safe_float_or_none(data.get('ocf_np_ratio'))

        industry_prior_type = str(
            sector_info.get('industry_prior_type')
            or long_term_profile.get('cycle_type')
            or 'quality'
        )
        reasons: List[str] = []
        if long_term_profile.get('matched_key'):
            reasons.append(f"行业先验: {long_term_profile.get('matched_key')}")

        financial_style = 'quality'
        if dividend_yield is not None and dividend_yield >= 3.5 and max(self._safe_float(profit_growth), self._safe_float(revenue_growth)) < 12:
            financial_style = 'dividend'
            reasons.append('财务特征偏红利')
        elif min(self._safe_float(profit_growth), self._safe_float(revenue_growth)) >= 18:
            financial_style = 'growth'
            reasons.append('收入利润双高增')
        elif (roic is not None and roic >= 10) or (roe is not None and roe >= 14):
            financial_style = 'quality'
            reasons.append('盈利质量较高')

        if any(key in sector_name for key in ['煤', '钢铁', '有色', '化工', '航运', '油气']):
            industry_prior_type = 'cyclical'
            reasons.append('行业先验偏周期')

        final_type = industry_prior_type
        if industry_prior_type == 'cyclical':
            if ocf_np_ratio is not None and ocf_np_ratio >= 1.0 and dividend_yield is not None and dividend_yield >= 3.0:
                final_type = 'dividend'
                reasons.append('周期属性下现金流与分红较强，按红利处理')
            else:
                final_type = 'cyclical'
        elif industry_prior_type == 'dividend':
            if min(self._safe_float(profit_growth), self._safe_float(revenue_growth)) >= 20 and (roic is not None and roic >= 10):
                final_type = 'quality'
                reasons.append('红利行业内经营质量更强，升级为质量型')
            else:
                final_type = 'dividend'
        elif industry_prior_type == 'growth':
            if (roic is not None and roic < 6) and max(self._safe_float(profit_growth), self._safe_float(revenue_growth)) < 15:
                final_type = 'quality'
                reasons.append('成长行业但经营质量一般，降级为质量型')
            else:
                final_type = 'growth' if financial_style == 'growth' else 'quality'
        else:
            if financial_style == 'dividend':
                final_type = 'dividend'
            elif financial_style == 'growth':
                final_type = 'growth'
            else:
                final_type = 'quality'

        return {
            'industry_prior_type': industry_prior_type,
            'financial_style': financial_style,
            'stock_type': final_type,
            'matched_profile_key': long_term_profile.get('matched_key'),
            'reasons': reasons,
        }

    def _infer_stock_type(self, data: Dict, sector_info: Optional[Dict] = None) -> str:
        return str(self._build_stock_profile(data, sector_info).get('stock_type') or 'quality')

    def _score_valuation(self, data: Dict, stock_type: str = 'quality') -> Dict:
        pe = self._safe_float_or_none(data.get('pe'))
        dividend_yield = self._safe_float_or_none(data.get('dividend_yield'))
        profit_growth = self._safe_float_or_none(data.get('profit_growth'))
        revenue_growth = self._safe_float_or_none(data.get('revenue_growth'))
        growth_anchor = max(self._safe_float(profit_growth), self._safe_float(revenue_growth), 0.0)
        peg = pe / growth_anchor if pe is not None and pe > 0 and growth_anchor > 0 else None

        score = 0.0
        reasons: List[str] = []

        if stock_type == 'growth':
            pe_score = 7.0 * self._clip01(1.0 - max(self._safe_float(pe) - 25.0, 0.0) / 45.0) if pe is not None else 3.5
            peg_score = 8.0 * self._clip01(1.6 - self._safe_float(peg)) / 1.6 if peg is not None else 3.0
            score = pe_score + max(0.0, peg_score)
            if peg is not None and peg <= 1.2:
                reasons.append('成长估值匹配较好')
        elif stock_type == 'dividend':
            pe_score = 6.0 * self._clip01((25.0 - max(self._safe_float(pe), 5.0)) / 20.0) if pe is not None else 3.0
            dividend_score = 9.0 * self._clip01(self._safe_float(dividend_yield) / 5.0) if dividend_yield is not None else 3.0
            score = pe_score + dividend_score
            if dividend_yield is not None and dividend_yield >= 3.0:
                reasons.append('股息率具备防守性')
        elif stock_type == 'cyclical':
            pe_score = 10.0 * self._clip01((18.0 - max(self._safe_float(pe), 3.0)) / 15.0) if pe is not None else 4.0
            safety_score = 5.0 * self._clip01(1.0 - max(growth_anchor - 40.0, 0.0) / 40.0)
            score = pe_score + safety_score
            reasons.append('周期股估值从严处理')
        else:
            pe_score = 9.0 * self._clip01((28.0 - max(self._safe_float(pe), 6.0)) / 22.0) if pe is not None else 4.5
            quality_match = 6.0 * self._clip01(max(growth_anchor, 5.0) / max(self._safe_float(pe), 10.0)) if pe is not None else 3.0
            score = pe_score + quality_match

        score = max(0.0, min(15.0, score))
        return {
            'score': round(score, 2),
            'pe': round(self._safe_float(pe), 2),
            'peg': round(peg, 2) if peg is not None else None,
            'growth_anchor': round(growth_anchor, 2),
            'stock_type': stock_type,
            'level': '低估' if score >= 11 else '合理' if score >= 7 else '偏贵',
            'reasons': reasons,
        }

    def _score_volume_confirmation(self, df: pd.DataFrame, stock_info: Dict, fund_flow: Dict) -> Dict:
        close = pd.to_numeric(df['close'], errors='coerce')
        amount_series = pd.to_numeric(df.get('amount'), errors='coerce') if 'amount' in df.columns else pd.Series(dtype=float)
        current_amount = self._safe_float(stock_info.get('amount'))
        if current_amount <= 0 and not amount_series.empty:
            current_amount = self._safe_float(amount_series.iloc[-1])
        avg_amount_20 = self._safe_float(amount_series.tail(20).mean()) if not amount_series.empty else 0.0
        amount_ratio = current_amount / avg_amount_20 if current_amount > 0 and avg_amount_20 > 0 else 0.0

        obv = self.indicators.calc_obv(df)
        obv_ratio = 0.0
        if len(obv) >= 20 and self._safe_float(abs(obv.iloc[-20]), 0.0) > 0:
            obv_ratio = (self._safe_float(obv.iloc[-1]) - self._safe_float(obv.iloc[-20])) / abs(self._safe_float(obv.iloc[-20])) * 100

        fund_details = self._score_fund_flow(df, stock_info, fund_flow)
        persistence = self._safe_float(fund_details.get('persistence'))

        score = 0.0
        # 长线里资金面主要做确认和排雷，不再强力推高总分
        if 0.8 <= amount_ratio <= 1.8:
            score += 2.5
        elif 1.8 < amount_ratio <= 2.5:
            score += 1.0
        elif amount_ratio > 3.5:
            score -= 2.5
        elif 0 < amount_ratio < 0.5:
            score -= 1.0

        if obv.iloc[-1] > obv.iloc[-20]:
            score += 1.5
        elif obv.iloc[-1] < obv.iloc[-20]:
            score -= 0.5

        if persistence >= 0.7:
            score += 1.5
        elif persistence >= 0.55:
            score += 0.75

        main_ratio = self._safe_float(fund_details.get('main_ratio'))
        if main_ratio >= 3:
            score += 0.75
        elif main_ratio < -3:
            score -= 2.5

        flow_amount_ratio = self._safe_float(fund_details.get('flow_amount_ratio'))
        if flow_amount_ratio < -2.0:
            score -= 1.5
        elif flow_amount_ratio > 2.0:
            score += 0.5

        score = max(0.0, min(10.0, score))
        fund_details['score'] = round(min(fund_details.get('score', 0.0), 8.0), 2)
        return {
            'score': round(score, 2),
            'amount_ratio': round(amount_ratio, 2) if amount_ratio > 0 else None,
            'obv_trend': 'up' if obv.iloc[-1] > obv.iloc[-20] else 'down',
            'obv_ratio': round(obv_ratio, 2),
            'fund_flow': fund_details,
        }

    def _score_total_risk(
        self,
        df: pd.DataFrame,
        trend_details: Dict,
        volume_details: Dict,
        fundamental_data: Dict,
        valuation_details: Dict,
        sector_info: Optional[Dict] = None,
    ) -> Dict:
        close = pd.to_numeric(df['close'], errors='coerce')
        volatility = self._safe_float(close.pct_change().std() * np.sqrt(252) * 100)
        returns_60d = self._safe_float(trend_details.get('returns_60d'))
        returns_120d = self._safe_float(trend_details.get('returns_120d'))
        drawdown_52w = self._safe_float(trend_details.get('drawdown_from_52w_high'))
        amount_ratio = self._safe_float(volume_details.get('amount_ratio'))
        atr = self.indicators.calc_atr(df)
        atr_value = self._safe_float(atr.iloc[-1])
        pe = self._safe_float_or_none(valuation_details.get('pe'))
        ocf_np_ratio = self._safe_float_or_none(fundamental_data.get('ocf_np_ratio'))
        roic = self._safe_float_or_none(fundamental_data.get('roic'))
        roe = self._safe_float_or_none(fundamental_data.get('roe'))
        profit_growth = self._safe_float_or_none(fundamental_data.get('profit_growth'))
        revenue_growth = self._safe_float_or_none(fundamental_data.get('revenue_growth'))
        stock_type = str(valuation_details.get('stock_type') or 'quality')
        long_term_profile = (sector_info or {}).get('long_term_profile') or {}

        trade_penalty = 0.0
        trade_flags: List[str] = []
        if volatility > 45:
            trade_penalty += 3.0
            trade_flags.append('价格波动偏高')
        elif volatility > 35:
            trade_penalty += 1.5
        if returns_60d > 35:
            trade_penalty += 2.0
            trade_flags.append('60日涨幅过快')
        if returns_120d > 60:
            trade_penalty += 2.0
            trade_flags.append('120日涨幅过快')
        if amount_ratio > 3.5:
            trade_penalty += 1.5
            trade_flags.append('成交额过热')
        if drawdown_52w > -3:
            trade_penalty += 1.0
            trade_flags.append('离52周高点过近')
        trade_score = max(0.0, 5.0 - min(5.0, trade_penalty))

        fundamental_penalty = 0.0
        fundamental_flags: List[str] = []
        if ocf_np_ratio is None:
            fundamental_penalty += 0.8
            fundamental_flags.append('现金流质量待确认')
        elif ocf_np_ratio < 0.8:
            fundamental_penalty += 2.0
            fundamental_flags.append('现金流弱于利润')
        if roic is None:
            fundamental_penalty += 0.8
        elif roic < 6:
            fundamental_penalty += 2.0
            fundamental_flags.append('ROIC偏低')
        if roe is not None and roe < 8:
            fundamental_penalty += 1.2
            fundamental_flags.append('ROE偏低')
        if profit_growth is not None and revenue_growth is not None and revenue_growth < 0 < profit_growth:
            fundamental_penalty += 1.5
            fundamental_flags.append('利润增长与营收背离')
        cycle_risk = self._safe_float(long_term_profile.get('cycle_risk'))
        if cycle_risk > 3.5:
            fundamental_penalty += 1.0
            fundamental_flags.append('行业周期风险偏高')
        fundamental_score = max(0.0, 6.0 - min(6.0, fundamental_penalty))

        valuation_penalty = 0.0
        valuation_flags: List[str] = []
        if stock_type == 'growth':
            if pe is not None and pe > 60:
                valuation_penalty += 4.0
                valuation_flags.append('成长估值过高')
            elif pe is not None and pe > 40:
                valuation_penalty += 2.0
        elif stock_type == 'dividend':
            if pe is not None and pe > 25:
                valuation_penalty += 2.0
                valuation_flags.append('红利股估值失去防守性')
        elif stock_type == 'cyclical':
            if pe is not None and pe > 18:
                valuation_penalty += 3.0
                valuation_flags.append('周期股估值偏高')
        else:
            if pe is not None and pe > 35:
                valuation_penalty += 2.5
                valuation_flags.append('估值安全边际不足')
        if self._safe_float(valuation_details.get('score')) < 5.0:
            valuation_penalty += 1.5
        valuation_score = max(0.0, 4.0 - min(4.0, valuation_penalty))

        score = fundamental_score + valuation_score + trade_score
        penalties = trade_flags + fundamental_flags + valuation_flags
        return {
            'score': round(score, 2),
            'annual_volatility': round(volatility, 2),
            'returns_60d': round(returns_60d, 2),
            'returns_120d': round(returns_120d, 2),
            'drawdown_from_52w_high': round(drawdown_52w, 2),
            'amount_ratio': round(amount_ratio, 2) if amount_ratio > 0 else None,
            'atr': round(atr_value, 4),
            'penalties': penalties,
            'trade_risk': {'score': round(trade_score, 2), 'flags': trade_flags},
            'fundamental_risk': {'score': round(fundamental_score, 2), 'flags': fundamental_flags},
            'valuation_risk': {'score': round(valuation_score, 2), 'flags': valuation_flags},
        }

    def _score_risk_control(self, df: pd.DataFrame, trend_details: Dict, volume_details: Dict) -> Dict:
        """兼容旧接口，默认返回中性风险框架。"""
        return self._score_total_risk(df, trend_details, volume_details, {}, {'stock_type': 'quality'})

    def _score_fund_flow(self, df: pd.DataFrame, stock_info: Dict, fund_flow: Dict) -> Dict:
        """
        资金面多因子标准化评分。
        可用数据越多，评分越完整；缺失时自动跳过对应因子。
        """
        factor_scores = {}
        factor_weights = {}

        main_in = self._safe_float(fund_flow.get('main_in')) if fund_flow else 0.0
        main_ratio = self._safe_float(fund_flow.get('main_ratio')) if fund_flow else 0.0
        stock_amount = self._safe_float(stock_info.get('amount'))
        amount_series = pd.to_numeric(df.get('amount'), errors='coerce') if 'amount' in df.columns else pd.Series(dtype=float)
        avg_amount_20 = self._safe_float(amount_series.tail(20).mean()) if not amount_series.empty else 0.0
        current_amount = stock_amount if stock_amount > 0 else self._safe_float(amount_series.iloc[-1]) if not amount_series.empty else 0.0
        flow_amount_ratio = 0.0

        if fund_flow:
            # 绝对主力净流入，使用对数压缩极值，避免大票天然占优过多
            signed_log_main_in = np.sign(main_in) * np.log1p(abs(main_in))
            factor_scores['main_in_strength'] = self._normalize_symmetric(signed_log_main_in, scale=16.0)
            factor_weights['main_in_strength'] = 0.35

            # 主力净流入占比，比纯金额更能反映资金质量
            factor_scores['main_ratio'] = self._normalize_symmetric(main_ratio, scale=8.0)
            factor_weights['main_ratio'] = 0.25

            # 主力净流入 / 成交额，占成交额比例越高越有意义
            if current_amount > 0:
                flow_amount_ratio = main_in / current_amount * 100
                factor_scores['flow_amount_ratio'] = self._normalize_symmetric(flow_amount_ratio, scale=3.0)
                factor_weights['flow_amount_ratio'] = 0.20

        close_series = pd.to_numeric(df['close'], errors='coerce')
        ma20 = close_series.rolling(20).mean()
        returns = close_series.pct_change()

        persistence_parts = []
        if len(close_series) >= 10 and not ma20.dropna().empty:
            recent_close = close_series.tail(10)
            recent_ma20 = ma20.tail(10)
            valid = recent_ma20.notna()
            if valid.any():
                persistence_parts.append((recent_close[valid] > recent_ma20[valid]).mean())

        if len(returns) >= 10:
            recent_returns = returns.tail(10).dropna()
            if not recent_returns.empty:
                persistence_parts.append((recent_returns > 0).mean())

        if not amount_series.empty and avg_amount_20 > 0:
            recent_amount = amount_series.tail(5).dropna()
            if not recent_amount.empty:
                persistence_parts.append((recent_amount >= avg_amount_20 * 0.8).mean())

        persistence_score = None
        if persistence_parts:
            persistence_score = self._clip01(float(np.mean(persistence_parts)))
            factor_scores['persistence'] = persistence_score
            factor_weights['persistence'] = 0.12

        # 成交额活跃度：当前成交额相对20日均额，偏高但不过热更好
        amount_ratio = 0.0
        if current_amount > 0 and avg_amount_20 > 0:
            amount_ratio = current_amount / avg_amount_20
            if amount_ratio <= 1.0:
                amount_activity_score = self._normalize_positive(amount_ratio, 0.5, 1.0)
            elif amount_ratio <= 2.0:
                amount_activity_score = 1.0
            elif amount_ratio <= 3.0:
                amount_activity_score = self._normalize_positive(3.0 - amount_ratio, 0.0, 1.0)
            else:
                amount_activity_score = 0.0

            factor_scores['amount_activity'] = amount_activity_score
            factor_weights['amount_activity'] = 0.08

        if factor_weights:
            total_weight = sum(factor_weights.values())
            normalized_score = sum(
                factor_scores[name] * factor_weights[name]
                for name in factor_weights
            ) / total_weight
            fund_score = round(self._clip01(normalized_score) * 8, 2)
        else:
            fund_score = 0.0

        return {
            'score': fund_score,
            'main_in': main_in / 10000,
            'main_ratio': main_ratio,
            'flow_amount_ratio': round(flow_amount_ratio, 2) if current_amount > 0 else None,
            'amount_ratio': round(amount_ratio, 2) if amount_ratio > 0 else None,
            'persistence': round(persistence_score, 3) if persistence_score is not None else None,
            'factors': {k: round(v, 3) for k, v in factor_scores.items()},
            'available': bool(factor_weights),
        }

    def _score_fundamental(self, code: str) -> Dict:
        """
        基本面评分，轻量纳入 ROE 和利润增长。
        优质基本面加分，劣质基本面扣分。
        数据缺失时仍返回可展示结果，避免中断选股流程。
        """
        data = self._get_tushare_fundamental_profile(code)
        roe = self._safe_float(data.get('roe'))
        profit_growth = self._safe_float(data.get('profit_growth'))

        score = 0.0

        if roe >= 20:
            score += 6.0
        elif roe >= 15:
            score += 5.0
        elif roe >= 10:
            score += 3.5
        elif roe >= 5:
            score += 1.5
        elif roe <= 0:
            score -= 4.0
        elif roe < 5:
            score -= 2.5

        if profit_growth >= 30:
            score += 6.0
        elif profit_growth >= 20:
            score += 5.0
        elif profit_growth >= 10:
            score += 3.5
        elif profit_growth > 0:
            score += 1.5
        elif profit_growth <= -30:
            score -= 4.0
        elif profit_growth < -20:
            score -= 3.0
        elif profit_growth < -10:
            score -= 1.5

        if roe < 5 and profit_growth < 0:
            score -= 1.5

        score = max(-8.0, min(12.0, score))

        return {
            'score': round(score, 2),
            'roe': round(roe, 2),
            'profit_growth': round(profit_growth, 2),
            'revenue_growth': round(self._safe_float(data.get('revenue_growth')), 2),
            'dividend_yield': round(self._safe_float(data.get('dividend_yield')), 2),
            'source': data.get('source', ''),
            'available': bool(data),
        }

    def load_watchlist(self) -> List[str]:
        """加载监控列表，过滤创业板和科创板和北交所"""
        try:
            with open('watchlist.json', 'r') as f:
                raw = json.load(f)

            all_stocks = []
            for item in raw:
                code = self.cache.normalize_code(item)
                if not code:
                    continue
                all_stocks.append(code)
            all_stocks = list(dict.fromkeys(all_stocks))

            # 过滤: 排除3开头(创业板)和688开头(科创板)和北交所920
            filtered = [
                code for code in all_stocks
                if not code.startswith('3')
                and not code.startswith('688')
                and not code.startswith('920')
            ]

            return filtered
        except:
            return []

    def analyze_single_stock(self, code: str) -> Dict:
        """
        分析单只股票
        返回综合评分和详细数据
        """
        try:
            # 获取历史数据
            history_days = int(self.params['history_days_long'])
            min_required_days = int(self.params['min_required_history_days'])
            df = self._get_history_data_tushare(code, days=history_days)
            if df is None or df.empty or len(df) < min_required_days:
                return None
            
            # 获取基础信息
            stock_info = self.cache.get_stock(code)
            if not stock_info:
                return None
            
            details = {}
            market_context = self.market_context

            trend_details = self._score_trend_quality(df)
            growth_raw = self._get_tushare_fundamental_profile(code)
            growth_details = self._score_growth_quality(growth_raw)
            sector_info = self._get_stock_sector_info(code)
            stock_profile = self._build_stock_profile(growth_raw, sector_info)
            stock_type = str(stock_profile.get('stock_type') or 'quality')
            moat_details = self._score_quality_moat_proxy(growth_raw, sector_info)
            valuation_details = self._score_valuation(growth_raw, stock_type=stock_type)
            fund_flow = self.cache.get_fund_flow(code)
            volume_details = self._score_volume_confirmation(df, stock_info, fund_flow)
            risk_details = self._score_total_risk(
                df,
                trend_details,
                volume_details,
                growth_raw,
                valuation_details,
                sector_info=sector_info,
            )

            raw_component_scores = {
                'growth': self._safe_float(growth_details.get('score')),
                'quality_moat_proxy': self._safe_float(moat_details.get('score')),
                'valuation': self._safe_float(valuation_details.get('score')),
                'trend': self._safe_float(trend_details.get('score')),
                'fund': self._safe_float(volume_details.get('score')),
                'risk': self._safe_float(risk_details.get('score')),
            }
            score_engine = self._weight_component_scores(raw_component_scores)
            weighted_scores = score_engine['weighted']
            trend_score = self._safe_float(weighted_scores.get('trend'))
            fund_score = self._safe_float(weighted_scores.get('fund'))
            risk_score = self._safe_float(weighted_scores.get('risk'))
            data_confidence = self._safe_float(growth_details.get('data_confidence'), 1.0)

            details['trend'] = trend_details
            details['growth'] = growth_details
            details['quality_moat_proxy'] = moat_details
            details['moat'] = moat_details
            details['valuation'] = valuation_details
            details['volume_confirmation'] = volume_details
            details['risk_control'] = risk_details
            details['fundamental'] = {
                **growth_details,
                'pe': valuation_details['pe'],
                'peg': valuation_details['peg'],
                'level': valuation_details['level'],
                'roic': growth_raw.get('roic'),
                'ocf_np_ratio': growth_raw.get('ocf_np_ratio'),
                'stock_type': stock_type,
                'stock_profile': stock_profile,
                'data_confidence': data_confidence,
            }
            details['stock_profile'] = stock_profile
            details['fund_flow'] = volume_details['fund_flow']
            details['score_engine'] = score_engine
            details['score_components'] = {
                'growth': round(self._safe_float(weighted_scores.get('growth')), 2),
                'quality_moat_proxy': round(self._safe_float(weighted_scores.get('quality_moat_proxy')), 2),
                'valuation': round(self._safe_float(weighted_scores.get('valuation')), 2),
                'trend': round(trend_score, 2),
                'fund': round(fund_score, 2),
                'risk': round(risk_score, 2),
                'weights': score_engine['weights'],
                'raw_max': score_engine['raw_max'],
                'normalized': score_engine['normalized'],
                'raw': raw_component_scores,
                'growth_30': round(self._safe_float(weighted_scores.get('growth')), 2),
                'quality_moat_proxy_15': round(self._safe_float(weighted_scores.get('quality_moat_proxy')), 2),
                'moat_15': round(self._safe_float(weighted_scores.get('quality_moat_proxy')), 2),
                'valuation_15': round(self._safe_float(weighted_scores.get('valuation')), 2),
                'trend_15': round(trend_score, 2),
                'fund_10': round(fund_score, 2),
                'risk_15': round(risk_score, 2),
            }

            base_score = (
                self._safe_float(weighted_scores.get('growth'))
                + self._safe_float(weighted_scores.get('quality_moat_proxy'))
                + self._safe_float(weighted_scores.get('valuation'))
                + trend_score
                + fund_score
                + risk_score
            )
            # 不再使用市场情绪作为硬性门控，仅保留个股结构化评分
            score = base_score * (0.85 + 0.15 * data_confidence)
            score = max(0.0, min(100.0, score))

            # ====== 10. 计算买卖点（基于ATR动态止损） ======
            raw_price = stock_info.get('price')
            raw_change_pct = stock_info.get('change_pct')
            current_price = float(raw_price) if raw_price is not None else float(df['close'].iloc[-1])
            atr_value = self._safe_float(risk_details.get('atr'))

            # 中长线使用更宽松的止损止盈
            stop_multiplier = 2.5  # ATR*2.5
            profit_multiplier = 4.0  # ATR*4.0

            if atr_value > 0 and current_price > 0:
                stop_loss = current_price - atr_value * stop_multiplier
                take_profit = current_price + atr_value * profit_multiplier
                stop_loss_pct = (stop_loss - current_price) / current_price * 100
                take_profit_pct = (take_profit - current_price) / current_price * 100
                risk = current_price - stop_loss
                reward = take_profit - current_price
                risk_reward_ratio = reward / risk if risk > 0 else 2.5
            else:
                # 默认值
                stop_loss = current_price * 0.92
                take_profit = current_price * 1.20
                stop_loss_pct = -8.0
                take_profit_pct = 20.0
                risk_reward_ratio = 2.5

            details['trade_points'] = {
                'buy_price': round(current_price, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'stop_loss_pct': round(stop_loss_pct, 2),
                'take_profit_pct': round(take_profit_pct, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2)
            }

            # 生成买入信号列表
            buy_signals = []
            if trend_details['score'] >= 11:
                buy_signals.append(f"趋势质量较高 ({trend_details['rating']})")
            if growth_details['roe'] >= 12:
                buy_signals.append(f"ROE较优 ({growth_details['roe']:.1f}%)")
            if self._safe_float(moat_details.get('roic')) >= 10:
                buy_signals.append(f"ROIC较优 ({moat_details['roic']:.1f}%)")
            if moat_details.get('ocf_np_ratio') is not None and self._safe_float(moat_details.get('ocf_np_ratio')) >= 1.0:
                buy_signals.append(f"现金流质量较好 ({moat_details['ocf_np_ratio']:.2f})")
            if growth_details['profit_growth'] >= 15 and growth_details['revenue_growth'] >= 8:
                buy_signals.append(
                    f"成长匹配良好 (利润{growth_details['profit_growth']:+.1f}%, 营收{growth_details['revenue_growth']:+.1f}%)"
                )
            if valuation_details['score'] >= 10:
                buy_signals.append(f"估值约束合理 ({valuation_details['level']})")
            fund_details = volume_details['fund_flow']
            if volume_details.get('amount_ratio') and 0.8 <= volume_details['amount_ratio'] <= 2.5:
                buy_signals.append(f"成交额确认 ({volume_details['amount_ratio']:.2f}x)")
            if fund_details['main_in'] > 0 and self._safe_float(fund_details.get('flow_amount_ratio')) >= 1.0:
                if fund_details.get('flow_amount_ratio') is not None:
                    buy_signals.append(
                        f"主力流入占成交额 {fund_details['flow_amount_ratio']:+.2f}%"
                    )
                else:
                    buy_signals.append(f"主力流入 (+{fund_details['main_in']:.0f}万)")
            if (fund_details.get('persistence') or 0) >= 0.7:
                buy_signals.append(f"走势持续性较强 ({fund_details['persistence']:.2f})")

            # ====== 汇总结果 ======
            result = {
                'code': code,
                'name': stock_info.get('name', 'Unknown'),
                'price': current_price,
                'change_pct': float(raw_change_pct) if raw_change_pct is not None else 0.0,
                'score': round(float(score), 2),
                'base_score': round(float(base_score), 2),
                'rating': self._get_rating(score),
                'details': self._convert_to_json_safe(details),
                'stock_type': stock_type,
                'signals': [trend_details['rating']] + trend_details['reasons'][:2],
                'buy_signals': buy_signals,
                'buy_signal_count': len(buy_signals),
                'market_context': self._convert_to_json_safe(market_context),
                # 买卖点字段
                'buy_price': round(current_price, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'stop_loss_pct': round(stop_loss_pct, 2),
                'take_profit_pct': round(take_profit_pct, 2),
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'recommend': bool(score >= float(self.params['recommend_score_floor']) and len(buy_signals) >= 2),
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return result
            
        except Exception as e:
            print(f"分析{code}失败: {e}")
            return None
    
    def _get_rating(self, score: float) -> str:
        """评级"""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B+'
        elif score >= 60:
            return 'B'
        elif score >= 45:
            return 'C'
        else:
            return 'D'

    def _apply_sector_heat_adjustment(self, stocks: List[Dict]) -> List[Dict]:
        if not stocks:
            return stocks

        refined = []
        for stock in stocks:
            sector_info = self._get_stock_sector_info(stock['code'])
            market_heat = sector_info.get('market_heat')
            adjustment = 0.0
            if market_heat:
                if market_heat.get('level') == 'strong':
                    adjustment = float(self.params['market_heat_adjustment_strong'])
                elif market_heat.get('level') == 'warm':
                    adjustment = float(self.params['market_heat_adjustment_warm'])
                elif market_heat.get('level') == 'weak':
                    adjustment = float(self.params['market_heat_adjustment_weak'])
                stock['sector_info'] = self._convert_to_json_safe({
                    'name': sector_info.get('sector'),
                    'market_heat': market_heat,
                    'long_term_profile': sector_info.get('long_term_profile'),
                    'cycle_type': sector_info.get('cycle_type'),
                    'score_adjustment': adjustment,
                })
            elif sector_info.get('sector'):
                stock['sector_info'] = self._convert_to_json_safe({
                    'name': sector_info.get('sector'),
                    'market_heat': None,
                    'long_term_profile': sector_info.get('long_term_profile'),
                    'cycle_type': sector_info.get('cycle_type'),
                    'score_adjustment': adjustment,
                })

            stock['score'] = round(max(0.0, min(100.0, stock['score'] + adjustment)), 2)
            stock['rating'] = self._get_rating(stock['score'])
            stock['recommend'] = bool(stock['recommend'])
            refined.append(stock)

        return refined

    def _apply_sector_strength(self, stocks: List[Dict]) -> List[Dict]:
        """兼容旧调用名，实际应用的是行业热度微调。"""
        return self._apply_sector_heat_adjustment(stocks)
    
    def _convert_to_json_safe(self, obj):
        """
        转换为JSON安全的数据类型
        处理numpy/pandas类型、布尔值和NaN
        """
        import numpy as np
        import math
        
        if isinstance(obj, dict):
            return {k: self._convert_to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_safe(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            val = float(obj)
            # 处理NaN和Infinity
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        elif isinstance(obj, float):
            # 处理原生float的NaN
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        elif obj is None:
            return None
        else:
            return obj

    def _get_decision_tag(self, stock: Dict) -> str:
        score = self._safe_float(stock.get('score'))
        risk = stock.get('details', {}).get('risk_control', {})
        penalties = list(risk.get('penalties') or [])
        score_parts = stock.get('details', {}).get('score_components', {})
        weights = score_parts.get('weights') or self._get_score_weights()
        valuation_weight = self._safe_float(weights.get('valuation'), 15.0)
        risk_weight = self._safe_float(weights.get('risk'), 15.0)
        valuation_score = self._safe_float(score_parts.get('valuation'))
        risk_score = self._safe_float(score_parts.get('risk'))
        min_valuation_score = valuation_weight * (10.0 / 15.0)
        min_risk_score = risk_weight * (8.0 / 15.0)

        if penalties and any(flag in ''.join(penalties) for flag in ['估值', '现金流', 'ROIC', '背离']):
            return 'exclude'
        if score >= float(self.params['recommend_score_core']) and valuation_score >= min_valuation_score and risk_score >= min_risk_score:
            return 'core'
        if score >= float(self.params['recommend_score_watch']):
            return 'watch'
        return 'exclude'

    def _get_strengths(self, stock: Dict) -> List[str]:
        details = stock.get('details', {})
        strengths = []
        strengths.extend(stock.get('buy_signals', [])[:3])
        moat = details.get('quality_moat_proxy') or details.get('moat', {})
        valuation = details.get('valuation', {})
        trend = details.get('trend', {})
        if moat.get('reasons'):
            strengths.extend(list(moat.get('reasons'))[:2])
        if valuation.get('reasons'):
            strengths.extend(list(valuation.get('reasons'))[:1])
        if trend.get('reasons'):
            strengths.extend(list(trend.get('reasons'))[:1])
        return list(dict.fromkeys([str(x) for x in strengths if str(x).strip()]))[:5]

    def _get_risks_and_red_flags(self, stock: Dict) -> tuple[List[str], List[str]]:
        details = stock.get('details', {})
        risk = details.get('risk_control', {})
        valuation_risk = risk.get('valuation_risk', {})
        fundamental_risk = risk.get('fundamental_risk', {})
        trade_risk = risk.get('trade_risk', {})
        risks = []
        risks.extend(list(trade_risk.get('flags') or []))
        risks.extend(list(fundamental_risk.get('flags') or []))
        risks.extend(list(valuation_risk.get('flags') or []))
        risks = list(dict.fromkeys([str(x) for x in risks if str(x).strip()]))
        red_flags = [x for x in risks if any(k in x for k in ['现金流', '估值', '背离', 'ROIC'])]
        return risks[:5], red_flags[:3]

    def _get_max_falsification_point(self, stock: Dict) -> str:
        risks, red_flags = self._get_risks_and_red_flags(stock)
        if red_flags:
            return red_flags[0]
        if risks:
            return risks[0]
        return '后续财报与现金流兑现不及预期'

    def _build_selection_bundle(self, ranked: List[Dict], top_n: int) -> Dict[str, List[Dict]]:
        core = []
        watch = []
        exclude = []
        for stock in ranked:
            stock['decision_tag'] = self._get_decision_tag(stock)
            stock['strengths'] = self._get_strengths(stock)
            risks, red_flags = self._get_risks_and_red_flags(stock)
            stock['risks'] = risks
            stock['red_flags'] = red_flags
            stock['max_falsification_point'] = self._get_max_falsification_point(stock)
            if stock['decision_tag'] == 'core':
                core.append(stock)
            elif stock['decision_tag'] == 'watch':
                watch.append(stock)
            else:
                exclude.append(stock)

        return {
            'core': core[:top_n],
            'watch': watch[:max(top_n, 10)],
            'all_ranked': ranked,
            'excluded_but_interesting': exclude[:max(5, min(10, len(exclude)))],
        }
    
    def select_top_stocks(self, top_n: int = 10) -> Dict[str, List[Dict]]:
        """
        选择分层候选池
        返回 core/watch/all_ranked 三层结构
        """
        market_context = self.market_context
        print("=" * 60)
        print(f"中长线选股 - TOP {top_n}")
        print("=" * 60)
        print()

        summary = market_context.get('summary', {})
        print(
            f"市场环境: {market_context.get('regime')} | 情绪 {summary.get('score', 50):.1f} | "
            f"涨停/跌停 {summary.get('limit_up', 0)}/{summary.get('limit_down', 0)} | "
            f"上涨占比 {summary.get('up_ratio', 50):.1f}%"
        )
        print()

        watchlist = self.load_watchlist()
        if not watchlist:
            print("监控列表为空")
            return {'core': [], 'watch': [], 'all_ranked': [], 'excluded_but_interesting': []}

        print(f"[long_term] 股票池: 全量 {len(watchlist)}", flush=True)
        print("[long_term] 预拉历史数据中...", flush=True)
        self._prefetch_history_data_tushare(watchlist, days=int(self.params['history_days_long']))
        
        print(f"分析 {len(watchlist)} 只股票...")
        print()
        
        results = []
        for i, code in enumerate(watchlist, 1):
            print(f"[{i}/{len(watchlist)}] {code}...", end=" ")
            
            result = self.analyze_single_stock(code)
            if result:
                print(f"{result['score']:.1f}分 ({result['rating']})")
                results.append(result)
            else:
                print("分析失败")
        
        ranked = sorted(results, key=lambda x: x['score'], reverse=True)
        refined_candidates = self._apply_sector_heat_adjustment(ranked)
        refined_candidates.sort(key=lambda x: x['score'], reverse=True)
        selection = self._build_selection_bundle(refined_candidates, top_n=top_n)
        top_stocks = selection['core']
        self.last_selection_bundle = selection
        
        print()
        print("=" * 60)
        print(f"核心候选 ({len(selection['core'])}) | 观察名单 ({len(selection['watch'])})")
        print("=" * 60)
        print()
        
        for i, stock in enumerate(top_stocks, 1):
            print(f"{i}. {stock['name']} ({stock['code']})")
            print(f"   评分: {stock['score']:.1f} ({stock['rating']})")
            print(f"   价格: ¥{stock['price']:.2f} ({stock['change_pct']:+.2f}%)")
            growth = stock['details']['growth']
            valuation = stock['details']['valuation']
            fund = stock['details']['fund_flow']
            fund_desc = f"{fund['main_in']:+.0f}万"
            if fund.get('flow_amount_ratio') is not None:
                fund_desc += f" / 占额{fund['flow_amount_ratio']:+.2f}%"
            sector_info = stock.get('sector_info') or {}
            sector_text = sector_info.get('name') or '未知行业'
            if sector_info.get('market_heat'):
                sector_text += f" ({sector_info['market_heat']['level']}, {sector_info['market_heat']['change_pct']:+.2f}%)"
            print(f"   趋势: {stock['details']['trend']['rating']} | "
                  f"ROE: {growth['roe']:.1f}% | "
                  f"利润增长: {growth['profit_growth']:+.1f}% | "
                  f"PE: {valuation['pe']:.1f} | "
                  f"资金: {fund_desc}")
            print(f"   行业: {sector_text}")
            print()
        
        return selection
    
    def generate_report(self, selection) -> str:
        """生成推荐报告"""
        if isinstance(selection, dict):
            core = list(selection.get('core') or [])
            watch = list(selection.get('watch') or [])
            all_ranked = list(selection.get('all_ranked') or [])
            excluded = list(selection.get('excluded_but_interesting') or [])
        else:
            core = list(selection or [])
            watch = []
            all_ranked = core
            excluded = []

        report = []
        market_context = self.market_context
        summary = market_context.get('summary', {})
        report.append("=" * 60)
        report.append("中长线选股报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(
            f"市场环境: {market_context.get('regime')} | 情绪{summary.get('score', 50):.1f} | "
            f"涨停/跌停 {summary.get('limit_up', 0)}/{summary.get('limit_down', 0)} | "
            f"上涨占比 {summary.get('up_ratio', 50):.1f}%"
        )
        report.append("=" * 60)
        report.append("")

        report.append("一、分层结果")
        report.append(f"核心候选: {len(core)}")
        report.append(f"观察名单: {len(watch)}")
        report.append(f"全量排序: {len(all_ranked)}")
        report.append("")

        def _render_group(title: str, stocks: List[Dict]):
            report.append(title)
            if not stocks:
                report.append("  无")
                report.append("")
                return
            for i, stock in enumerate(stocks, 1):
                report.append(f"  {i}. {stock['name']} ({stock['code']}) | 分数 {stock['score']:.1f} | 类型 {stock.get('stock_type', '-')}")
            report.append("")

        _render_group("二、核心候选", core)
        _render_group("三、观察名单", watch[:10])

        report.append("四、核心候选详解")
        report.append("")

        for i, stock in enumerate(core, 1):
            report.append(f"{i}. {stock['name']} ({stock['code']})")
            report.append(f"   评级: {stock['rating']} | 结论: {stock.get('decision_tag', '-')}")
            report.append(f"   评分: {stock['score']:.1f}/100 | 类型: {stock.get('stock_type', '-')}")
            report.append(f"   价格: ¥{stock['price']:.2f} ({stock['change_pct']:+.2f}%)")
            report.append("")

            details = stock['details']
            score_parts = details.get('score_components', {})
            weights = score_parts.get('weights') or self._get_score_weights()
            report.append("   分数拆解:")
            quality_moat_score = score_parts.get('quality_moat_proxy', score_parts.get('quality_moat_proxy_15', score_parts.get('moat_15', 0)))
            report.append(
                f"      成长{score_parts.get('growth', score_parts.get('growth_30', 0)):.1f}/{weights.get('growth', 30):.1f} | "
                f"质量代理{quality_moat_score:.1f}/{weights.get('quality_moat_proxy', weights.get('moat', 15)):.1f} | "
                f"估值{score_parts.get('valuation', score_parts.get('valuation_15', 0)):.1f}/{weights.get('valuation', 15):.1f} | "
                f"趋势{score_parts.get('trend', score_parts.get('trend_15', 0)):.1f}/{weights.get('trend', 15):.1f} | "
                f"资金{score_parts.get('fund', score_parts.get('fund_10', 0)):.1f}/{weights.get('fund', 10):.1f} | "
                f"风险{score_parts.get('risk', score_parts.get('risk_15', 0)):.1f}/{weights.get('risk', 15):.1f}"
            )
            report.append("")

            report.append("   优点:")
            strengths = stock.get('strengths') or ['暂无']
            for item in strengths:
                report.append(f"      - {item}")
            report.append("")

            report.append("   风险与红旗:")
            risks = stock.get('risks') or ['暂无明显风险']
            for item in risks:
                report.append(f"      - {item}")
            if stock.get('red_flags'):
                report.append(f"      红旗: {', '.join(stock['red_flags'])}")
            report.append(f"      最大证伪点: {stock.get('max_falsification_point', '-')}")
            report.append("")

            report.append("   成长与质量:")
            growth = details['growth']
            moat = details.get('quality_moat_proxy') or details.get('moat', {})
            report.append(f"      ROE: {growth['roe']:.2f}% | ROIC: {growth.get('roic') if growth.get('roic') is not None else 'N/A'}")
            report.append(f"      利润增长: {growth['profit_growth']:+.2f}% | 营收增长: {growth['revenue_growth']:+.2f}%")
            report.append(f"      现金流质量: {growth.get('ocf_np_ratio') if growth.get('ocf_np_ratio') is not None else 'N/A'} | 数据置信度: {growth.get('data_confidence', 0):.2f}")
            report.append(f"      质量/护城河代理分: {moat.get('score', 0):.1f}/15")
            report.append("")

            report.append("   估值与趋势:")
            valuation = details['valuation']
            trend = details['trend']
            pe_value = valuation.get('pe')
            report.append(f"      PE: {pe_value:.2f}" if isinstance(pe_value, (int, float)) else "      PE: N/A")
            report[-1] += f" | PEG: {valuation.get('peg') if valuation.get('peg') is not None else 'N/A'} | 类型: {valuation.get('stock_type', '-')}"
            report.append(f"      估值等级: {valuation.get('level', '-')} ({self._safe_float(valuation.get('score')):.1f}/15)")
            report.append(f"      趋势: {trend['rating']} | 60日/120日收益 {trend.get('returns_60d', 0):+.2f}% / {trend.get('returns_120d', 0):+.2f}%")
            report.append(f"      52周高点回撤: {trend.get('drawdown_from_52w_high', 0):+.2f}%")
            report.append("")

            report.append("   参与度与风险:")
            volume = details['volume_confirmation']
            fund = volume['fund_flow']
            risk = details['risk_control']
            report.append(f"      成交额/20日均额: {volume['amount_ratio'] if volume['amount_ratio'] is not None else 'N/A'} | OBV {volume['obv_trend']}")
            report.append(f"      主力净流入: {fund['main_in']:+.0f}万 | 流入/成交额: {fund.get('flow_amount_ratio') if fund.get('flow_amount_ratio') is not None else 'N/A'}")
            report.append(f"      综合风险: {risk['score']:.1f}/15 | 波动率 {risk['annual_volatility']:.2f}%")
            report.append(f"      交易风险: {risk['trade_risk']['score']:.1f} | 经营风险: {risk['fundamental_risk']['score']:.1f} | 估值风险: {risk['valuation_risk']['score']:.1f}")
            report.append("")

            sector_info = stock.get('sector_info') or {}
            if sector_info.get('name'):
                report.append("   行业背景:")
                sector_line = f"      {sector_info['name']}"
                if sector_info.get('market_heat'):
                    sector_line += f" | {sector_info['market_heat']['level']} | {sector_info['market_heat']['change_pct']:+.2f}%"
                report.append(sector_line)
                if sector_info.get('long_term_profile'):
                    report.append(f"      长期属性: {sector_info['long_term_profile'].get('comment', '')}")
                report.append("")

            report.append("   建议动作:")
            if stock.get('decision_tag') == 'core':
                report.append("      分批跟踪，优先等回调或财报确认后加仓")
            elif stock.get('decision_tag') == 'watch':
                report.append("      先观察，等待估值或经营信号进一步确认")
            else:
                report.append("      暂不参与")
            report.append("")
            report.append("-" * 60)
            report.append("")

        if excluded:
            report.append("五、剔除但值得观察")
            for i, stock in enumerate(excluded, 1):
                report.append(f"{i}. {stock['name']} ({stock['code']}) | {stock['score']:.1f} | 证伪点: {stock.get('max_falsification_point', '-')}")
            report.append("")

        return "\n".join(report)
    
    def close(self):
        self.cache.close()

    def save_selected_watchlist(self, stocks: List[Dict], filename: str) -> None:
        """将选股结果覆盖写入指定监控列表文件。"""
        watchlist_file = Path(__file__).resolve().parent / filename
        codes = [str(stock['code']).zfill(6) for stock in stocks if stock.get('code')]
        with open(watchlist_file, 'w', encoding='utf-8') as f:
            json.dump(codes, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    selector = LongTermSelector()
    
    # 选择TOP 10
    selection = selector.select_top_stocks(top_n=10)
    top_stocks = selection.get('core', [])
    
    # 生成报告
    if selection.get('all_ranked'):
        report = selector.generate_report(selection)
        print(report)
        
        # 保存到文件
        date_suffix = datetime.now().strftime('%Y%m%d')
        recommend_dir = Path(__file__).resolve().parent / 'recommend'
        recommend_dir.mkdir(parents=True, exist_ok=True)
        report_file = recommend_dir / f'daily_recommendation_{date_suffix}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        selector.save_selected_watchlist(top_stocks, 'watchlist_long_term.json')
        selector.save_selected_watchlist(top_stocks, 'watchlist_long_term_core.json')
        selector.save_selected_watchlist(selection.get('watch', []), 'watchlist_long_term_watch.json')
        try:
            send_email_with_attachment(
                subject=f'中长线选股报告 {date_suffix}',
                body='中长线选股报告见附件。',
                attachment_path=report_file,
            )
            print("已发送中长线报告到QQ邮箱")
        except Exception as e:
            print(f"中长线报告邮件发送失败: {e}")
        
        print(f"报告已保存到 {report_file}")
    
    selector.close()
