#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
短线选股引擎 (优化版)
每日推荐3-5只短线机会股
排除创业板(3开头)和科创板(688开头)和北交所920

优化内容：
1. 新增MACD、布林带指标评分
2. 动态止损止盈（基于ATR）
3. 精确买卖点输出
4. 多指标共振确认

评分体系 (满分100分):
- RSI信号: 20分
- KDJ信号: 20分
- MACD信号: 15分
- 布林带信号: 15分
- 量价异动: 15分
- 资金流向: 15分
0311优化内容
市场情绪接入
直接复用了 market_sentiment.py，把情绪分、涨停数、跌停数接进了短线环境判断
指数环境过滤
新增指数状态识别，基于上证、深成、中证1000 的短期趋势，给出 strong / neutral / weak / crash
动态门槛
市场弱时自动提高入选门槛：
提高最低分
提高最少买入信号数
收缩推荐数量
极弱环境直接不推荐
环境加减分
个股原始分数算完后，会再叠加市场环境修正，不会再在大盘暴跌时机械给出一堆短线票
"""

import pandas as pd
import numpy as np
import tinyshare as ts
from datetime import datetime
import json
from typing import List, Dict, Optional
from pathlib import Path
from stock_cache_db import StockCache
from short_term_indicators import ShortTermIndicators
from market_sentiment import calculate_market_sentiment
from env_settings import TUSHARE_TOKEN
from email_sender import send_email_with_attachment


class ShortTermSelector:
    """短线选股引擎"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.cache = StockCache()
        self.indicators = ShortTermIndicators()
        if TUSHARE_TOKEN:
            ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        self.history_cache: Dict[str, pd.DataFrame] = {}
        self.code_to_industry: Dict[str, str] = {}
        self.code_to_name: Dict[str, str] = {}
        self.code_to_themes: Dict[str, List[str]] = {}
        self.stock_sector_cache: Dict[str, Dict] = {}
        self.sector_heat_map: Dict[str, Dict] = {}
        self.sector_stock_rank_map: Dict[str, Dict] = {}
        self.theme_heat_map: Dict[str, Dict] = {}
        self.theme_stock_rank_map: Dict[str, Dict] = {}
        self.market_context = self._build_market_context()

    @staticmethod
    def _normalize_sector_name(name: Optional[str]) -> str:
        if not name:
            return ''
        normalized = str(name).strip()
        for suffix in ['Ⅰ', 'Ⅱ', 'Ⅲ', 'A', 'B']:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        return normalized

    @staticmethod
    def _theme_keywords() -> Dict[str, List[str]]:
        return {
            '机器人': ['机器人', '减速器', '伺服', '工业母机', '机器视觉', '自动化'],
            '算力': ['算力', '服务器', '光模块', '液冷', 'cpo', '数据中心', '东数西算'],
            '低空经济': ['低空', '无人机', '通航', '飞行汽车', 'evtol'],
            '并购重组': ['重组', '收购', '资产注入', '借壳', '并购'],
            '军工': ['军工', '导弹', '卫星', '航空发动机', '船舶'],
            '创新药': ['创新药', 'adc', '单抗', 'biotech', '医药'],
            '电网': ['电网', '特高压', '变压器', '电力设备', '储能'],
            '新型工业化': ['工业软件', '智能制造', '工控', '工业互联网', '自动化'],
            '芯片半导体': ['芯片', '半导体', '封测', '光刻', '存储'],
            '新能源车': ['新能源车', '锂电', '充电桩', '汽车零部件', '智能驾驶'],
            '消费电子': ['消费电子', 'ai眼镜', '手机', '耳机', '折叠屏'],
        }

    def _infer_themes(self, code: str, name: str, industry: str) -> List[str]:
        haystack = f"{name or ''} {industry or ''}".lower()
        themes: List[str] = []
        for theme, keywords in self._theme_keywords().items():
            if any(str(keyword).lower() in haystack for keyword in keywords):
                themes.append(theme)

        normalized_industry = self._normalize_sector_name(industry)
        if normalized_industry:
            industry_aliases = {
                '通用设备': '新型工业化',
                '专用设备': '新型工业化',
                '软件服务': '算力',
                '互联网服务': '算力',
                '通信设备': '算力',
                '半导体': '芯片半导体',
                '电气设备': '电网',
                '汽车零部件': '新能源车',
            }
            mapped = industry_aliases.get(normalized_industry)
            if mapped:
                themes.append(mapped)
            elif not themes:
                themes.append(normalized_industry)

        return list(dict.fromkeys([theme for theme in themes if str(theme).strip()]))

    def _load_sector_metadata(self) -> None:
        if self.code_to_industry and self.code_to_themes:
            return
        try:
            stock_basic = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,name,industry',
            )
            if stock_basic is None or stock_basic.empty:
                return
            for _, row in stock_basic.iterrows():
                ts_code = str(row.get('ts_code', '')).strip()
                name = str(row.get('name', '') or '').strip()
                industry = self._normalize_sector_name(row.get('industry'))
                if not ts_code:
                    continue
                code = ts_code.split('.')[0].zfill(6)
                self.code_to_name[code] = name
                if industry:
                    self.code_to_industry[code] = industry
                self.code_to_themes[code] = self._infer_themes(code, name, industry)
        except Exception:
            return

    def _build_sector_context(self) -> None:
        if self.sector_heat_map and self.sector_stock_rank_map:
            return

        self._load_sector_metadata()
        all_stocks = self.cache.get_all_stocks(max_age_minutes=1440)
        if not all_stocks:
            self.sector_heat_map = {}
            self.sector_stock_rank_map = {}
            return

        rows = []
        for stock in all_stocks:
            code = str(stock.get('code', '')).zfill(6)
            industry = self.code_to_industry.get(code, '')
            if not industry:
                continue
            fund_flow = self.cache.get_fund_flow(code) or {}
            lhb = self.cache.get_lhb(code) or {}
            change_pct = self._safe_float(stock.get('change_pct'))
            amount = self._safe_float(stock.get('amount'))
            main_ratio = self._safe_float(fund_flow.get('main_ratio'))
            rows.append({
                'code': code,
                'industry': industry,
                'change_pct': change_pct,
                'amount': amount,
                'main_ratio': main_ratio,
                'main_in': self._safe_float(fund_flow.get('main_in')),
                'lhb_net': self._safe_float(lhb.get('net_amount')),
                'is_limit_like': change_pct >= 9.5,
                'is_strong': change_pct >= 5.0,
            })

        if not rows:
            self.sector_heat_map = {}
            self.sector_stock_rank_map = {}
            return

        df = pd.DataFrame(rows)
        sector_heat_map: Dict[str, Dict] = {}
        sector_stock_rank_map: Dict[str, Dict] = {}
        grouped = df.groupby('industry', as_index=False)
        agg = grouped.agg(
            avg_change=('change_pct', 'mean'),
            sample_size=('code', 'count'),
            up_ratio=('change_pct', lambda s: float((s > 0).mean())),
            strong_count=('is_strong', 'sum'),
            limit_like_count=('is_limit_like', 'sum'),
            total_amount=('amount', 'sum'),
        ).sort_values(
            by=['limit_like_count', 'strong_count', 'avg_change', 'up_ratio', 'total_amount'],
            ascending=False,
        ).reset_index(drop=True)

        total = max(len(agg), 1)
        for idx, row in agg.iterrows():
            industry = str(row['industry'])
            rank_pct = (idx + 1) / total
            structure_score = (
                min(8.0, self._safe_float(row['limit_like_count']) * 2.0)
                + min(6.0, self._safe_float(row['strong_count']) * 1.2)
                + min(6.0, self._safe_float(row['up_ratio']) * 8.0)
            )
            heat_score = (
                self._safe_float(row['avg_change']) * 1.5
                + self._safe_float(row['up_ratio']) * 12.0
                + min(4.0, self._safe_float(row['strong_count']) * 0.8)
                + min(5.0, self._safe_float(row['limit_like_count']) * 1.2)
            )
            if rank_pct <= 0.08:
                level = 'hot'
            elif rank_pct <= 0.2:
                level = 'warm'
            elif rank_pct >= 0.9:
                level = 'cold'
            else:
                level = 'neutral'
            sector_heat_map[industry] = {
                'industry': industry,
                'rank': idx + 1,
                'rank_pct': round(rank_pct, 4),
                'level': level,
                'is_mainstream': bool(rank_pct <= 0.2 or self._safe_float(row['limit_like_count']) >= 2),
                'heat_score': round(heat_score, 2),
                'sector_strength_score': round(min(20.0, heat_score), 2),
                'sector_structure_score': round(min(20.0, structure_score), 2),
                'avg_change': round(self._safe_float(row['avg_change']), 2),
                'up_ratio': round(self._safe_float(row['up_ratio']) * 100, 2),
                'strong_count': int(row['strong_count'] or 0),
                'limit_like_count': int(row['limit_like_count'] or 0),
                'sample_size': int(row['sample_size'] or 0),
                'total_amount': round(self._safe_float(row['total_amount']), 2),
            }

        for industry, group in df.groupby('industry'):
            g = group.copy()
            g['leader_score'] = (
                g['change_pct'].rank(method='min', ascending=False, pct=True) * 45.0
                + g['amount'].rank(method='min', ascending=False, pct=True) * 25.0
                + g['main_ratio'].rank(method='min', ascending=False, pct=True) * 20.0
                + (g['lhb_net'] > 0).astype(float) * 10.0
            )
            g = g.sort_values(
                by=['leader_score', 'change_pct', 'amount', 'main_ratio'],
                ascending=False,
            ).reset_index(drop=True)
            total_in_sector = max(len(g), 1)
            leader_feedback_score = 0.0
            if not g.empty:
                top_n = g.head(min(3, len(g)))
                leader_feedback_score = (
                    self._safe_float(top_n['change_pct'].mean()) * 1.3
                    + min(6.0, float((top_n['change_pct'] >= 9.5).sum()) * 2.5)
                    + min(4.0, float((top_n['main_ratio'] >= 0).mean()) * 4.0)
                )
            if industry in sector_heat_map:
                sector_heat_map[industry]['leader_feedback_score'] = round(max(0.0, min(20.0, leader_feedback_score)), 2)
            for pos, (_, row) in enumerate(g.iterrows(), 1):
                sector_stock_rank_map[str(row['code'])] = {
                    'industry': industry,
                    'rank_in_sector': pos,
                    'total_in_sector': total_in_sector,
                    'leader_percentile': round(1.0 - ((pos - 1) / total_in_sector), 4),
                    'leader_score': round(self._safe_float(row['leader_score']), 2),
                    'is_sector_leader': pos <= 3,
                    'is_front_row': pos <= max(5, int(total_in_sector * 0.1)),
                    'change_pct': round(self._safe_float(row['change_pct']), 2),
                    'amount': round(self._safe_float(row['amount']), 2),
                    'main_ratio': round(self._safe_float(row['main_ratio']), 2),
                    'lhb_positive': bool(self._safe_float(row['lhb_net']) > 0),
                }

        self.sector_heat_map = sector_heat_map
        self.sector_stock_rank_map = sector_stock_rank_map

    def _build_theme_context(self) -> None:
        if self.theme_heat_map and self.theme_stock_rank_map:
            return

        self._load_sector_metadata()
        all_stocks = self.cache.get_all_stocks(max_age_minutes=1440)
        if not all_stocks:
            self.theme_heat_map = {}
            self.theme_stock_rank_map = {}
            return

        rows = []
        for stock in all_stocks:
            code = str(stock.get('code', '')).zfill(6)
            themes = self.code_to_themes.get(code) or []
            if not themes:
                continue
            fund_flow = self.cache.get_fund_flow(code) or {}
            lhb = self.cache.get_lhb(code) or {}
            change_pct = self._safe_float(stock.get('change_pct'))
            amount = self._safe_float(stock.get('amount'))
            main_ratio = self._safe_float(fund_flow.get('main_ratio'))
            for theme in themes:
                rows.append({
                    'code': code,
                    'theme': theme,
                    'change_pct': change_pct,
                    'amount': amount,
                    'main_ratio': main_ratio,
                    'main_in': self._safe_float(fund_flow.get('main_in')),
                    'lhb_net': self._safe_float(lhb.get('net_amount')),
                    'is_limit_like': change_pct >= 9.5,
                    'is_strong': change_pct >= 5.0,
                })

        if not rows:
            self.theme_heat_map = {}
            self.theme_stock_rank_map = {}
            return

        df = pd.DataFrame(rows)
        theme_heat_map: Dict[str, Dict] = {}
        theme_stock_rank_map: Dict[str, Dict] = {}
        agg = df.groupby('theme', as_index=False).agg(
            avg_change=('change_pct', 'mean'),
            sample_size=('code', 'count'),
            up_ratio=('change_pct', lambda s: float((s > 0).mean())),
            strong_count=('is_strong', 'sum'),
            limit_like_count=('is_limit_like', 'sum'),
            total_amount=('amount', 'sum'),
        ).sort_values(
            by=['limit_like_count', 'strong_count', 'avg_change', 'up_ratio', 'total_amount'],
            ascending=False,
        ).reset_index(drop=True)

        total = max(len(agg), 1)
        for idx, row in agg.iterrows():
            theme = str(row['theme'])
            rank_pct = (idx + 1) / total
            structure_score = (
                min(8.0, self._safe_float(row['limit_like_count']) * 2.0)
                + min(6.0, self._safe_float(row['strong_count']) * 1.2)
                + min(6.0, self._safe_float(row['up_ratio']) * 8.0)
            )
            heat_score = (
                self._safe_float(row['avg_change']) * 1.5
                + self._safe_float(row['up_ratio']) * 12.0
                + min(4.0, self._safe_float(row['strong_count']) * 0.8)
                + min(5.0, self._safe_float(row['limit_like_count']) * 1.2)
            )
            if rank_pct <= 0.08:
                level = 'hot'
            elif rank_pct <= 0.2:
                level = 'warm'
            elif rank_pct >= 0.9:
                level = 'cold'
            else:
                level = 'neutral'
            theme_heat_map[theme] = {
                'theme': theme,
                'rank': idx + 1,
                'rank_pct': round(rank_pct, 4),
                'level': level,
                'is_mainstream': bool(rank_pct <= 0.2 or self._safe_float(row['limit_like_count']) >= 2),
                'heat_score': round(heat_score, 2),
                'sector_strength_score': round(min(20.0, heat_score), 2),
                'sector_structure_score': round(min(20.0, structure_score), 2),
                'avg_change': round(self._safe_float(row['avg_change']), 2),
                'up_ratio': round(self._safe_float(row['up_ratio']) * 100, 2),
                'strong_count': int(row['strong_count'] or 0),
                'limit_like_count': int(row['limit_like_count'] or 0),
                'sample_size': int(row['sample_size'] or 0),
                'total_amount': round(self._safe_float(row['total_amount']), 2),
            }

        for theme, group in df.groupby('theme'):
            g = group.copy()
            g['leader_score'] = (
                g['change_pct'].rank(method='min', ascending=False, pct=True) * 45.0
                + g['amount'].rank(method='min', ascending=False, pct=True) * 25.0
                + g['main_ratio'].rank(method='min', ascending=False, pct=True) * 20.0
                + (g['lhb_net'] > 0).astype(float) * 10.0
            )
            g = g.sort_values(by=['leader_score', 'change_pct', 'amount', 'main_ratio'], ascending=False).reset_index(drop=True)
            top_n = g.head(min(3, len(g)))
            leader_feedback_score = 0.0
            if not top_n.empty:
                leader_feedback_score = (
                    self._safe_float(top_n['change_pct'].mean()) * 1.3
                    + min(6.0, float((top_n['change_pct'] >= 9.5).sum()) * 2.5)
                    + min(4.0, float((top_n['main_ratio'] >= 0).mean()) * 4.0)
                )
            if theme in theme_heat_map:
                theme_heat_map[theme]['leader_feedback_score'] = round(max(0.0, min(20.0, leader_feedback_score)), 2)
            total_in_theme = max(len(g), 1)
            for pos, (_, row) in enumerate(g.iterrows(), 1):
                code = str(row['code'])
                theme_row = {
                    'theme': theme,
                    'rank_in_sector': pos,
                    'total_in_sector': total_in_theme,
                    'leader_percentile': round(1.0 - ((pos - 1) / total_in_theme), 4),
                    'leader_score': round(self._safe_float(row['leader_score']), 2),
                    'is_sector_leader': pos <= 3,
                    'is_front_row': pos <= max(5, int(total_in_theme * 0.1)),
                    'change_pct': round(self._safe_float(row['change_pct']), 2),
                    'amount': round(self._safe_float(row['amount']), 2),
                    'main_ratio': round(self._safe_float(row['main_ratio']), 2),
                    'lhb_positive': bool(self._safe_float(row['lhb_net']) > 0),
                }
                existing = theme_stock_rank_map.get(code)
                if existing is None or self._safe_float(theme_row['leader_score']) > self._safe_float(existing.get('leader_score')):
                    theme_stock_rank_map[code] = theme_row

        self.theme_heat_map = theme_heat_map
        self.theme_stock_rank_map = theme_stock_rank_map

    def _get_stock_theme_context(self, code: str) -> Dict:
        code = str(code).zfill(6)
        self._build_theme_context()
        themes = list(self.code_to_themes.get(code) or [])
        primary_theme = None
        best_heat = None
        for theme in themes:
            theme_heat = self.theme_heat_map.get(theme)
            if theme_heat and (best_heat is None or self._safe_float(theme_heat.get('heat_score')) > self._safe_float(best_heat.get('heat_score'))):
                best_heat = theme_heat
                primary_theme = theme
        return {
            'themes': themes,
            'primary_theme': primary_theme,
            'theme_heat': best_heat,
            'theme_rank': self.theme_stock_rank_map.get(code),
        }

    def _get_stock_sector_context(self, code: str) -> Dict:
        code = str(code).zfill(6)
        if code in self.stock_sector_cache:
            return self.stock_sector_cache[code]

        theme_context = self._get_stock_theme_context(code)
        self._build_sector_context()
        industry = self.code_to_industry.get(code, '')
        primary_theme = theme_context.get('primary_theme')
        theme_heat = theme_context.get('theme_heat')
        theme_rank = theme_context.get('theme_rank')
        info = {
            'industry': industry or None,
            'themes': theme_context.get('themes') or [],
            'primary_theme': primary_theme,
            'sector_heat': theme_heat if theme_heat else (self.sector_heat_map.get(industry) if industry else None),
            'sector_rank': theme_rank if theme_rank else self.sector_stock_rank_map.get(code),
        }
        self.stock_sector_cache[code] = info
        return info

    def _build_market_context(self) -> Dict:
        """Build market environment context used by the short-term selector."""
        sentiment = calculate_market_sentiment(use_demo_data=False)
        if sentiment.get('stats', {}).get('total', 0) == 0:
            sentiment = calculate_market_sentiment(use_demo_data=True)

        index_regime = self._get_index_regime()
        sentiment_score = float(sentiment.get('score', 50.0) or 50.0)
        stats = sentiment.get('stats', {})
        limit_up = int(stats.get('limit_up', 0) or 0)
        limit_down = int(stats.get('limit_down', 0) or 0)
        up_ratio = float(stats.get('up_ratio', 50.0) or 50.0)

        score_adjustment = 0.0
        top_n_cap = 5
        candidate_pool_size = 20
        avoid_chasing = False
        tradable = True
        regime = index_regime['regime']

        if regime == 'strong':
            score_adjustment += 6
        elif regime == 'weak':
            score_adjustment -= 12
            top_n_cap = 2
            candidate_pool_size = 12
        elif regime == 'crash':
            score_adjustment -= 25
            top_n_cap = 0
            candidate_pool_size = 0
            tradable = False

        if sentiment_score < 35:
            score_adjustment -= 18
            top_n_cap = 0
            candidate_pool_size = 0
            tradable = False
        elif sentiment_score < 45:
            score_adjustment -= 10
            top_n_cap = min(top_n_cap, 2)
            candidate_pool_size = min(candidate_pool_size, 12)
        elif sentiment_score < 60:
            score_adjustment -= 2
            top_n_cap = min(top_n_cap, 3)
        elif sentiment_score >= 75:
            score_adjustment += 3
            candidate_pool_size = max(candidate_pool_size, 24)
        elif sentiment_score >= 65:
            score_adjustment += 1

        if limit_down > 20:
            score_adjustment -= 10
            top_n_cap = min(top_n_cap, 2)
            candidate_pool_size = min(candidate_pool_size, 10)
        elif limit_up < 20 and limit_down > 10:
            score_adjustment -= 8
            top_n_cap = min(top_n_cap, 2)
            candidate_pool_size = min(candidate_pool_size, 12)
        elif limit_up > 45 and limit_down < 5:
            score_adjustment += 4
        elif limit_up <= limit_down:
            score_adjustment -= 5

        if up_ratio < 45:
            avoid_chasing = True
        if up_ratio < 40:
            score_adjustment -= 4
            candidate_pool_size = min(candidate_pool_size, 12)

        strategy_mode = 'momentum_relay'
        if not tradable or regime == 'crash' or sentiment_score < 35:
            strategy_mode = 'defensive'
        elif regime == 'strong' and sentiment_score >= 65 and limit_up > max(20, limit_down * 2):
            strategy_mode = 'momentum_relay'
        elif sentiment_score < 55 or limit_down >= max(8, limit_up) or up_ratio < 0.5:
            strategy_mode = 'dip_rebound'
        else:
            strategy_mode = 'momentum_relay'

        mode_profile = {
            'momentum_relay': {
                'display': '强势接力',
                'prefer_trend': 1.0,
                'prefer_funding': 1.0,
                'prefer_leadership': 1.1,
                'prefer_dip': 0.7,
            },
            'dip_rebound': {
                'display': '分歧低吸',
                'prefer_trend': 0.7,
                'prefer_funding': 0.85,
                'prefer_leadership': 1.0,
                'prefer_dip': 1.15,
            },
            'defensive': {
                'display': '弱市防守',
                'prefer_trend': 0.5,
                'prefer_funding': 0.65,
                'prefer_leadership': 0.9,
                'prefer_dip': 1.2,
            },
        }[strategy_mode]

        return {
            'sentiment': sentiment,
            'index_regime': index_regime,
            'score_adjustment': score_adjustment,
            'top_n_cap': max(0, top_n_cap),
            'candidate_pool_size': max(0, candidate_pool_size),
            'candidate_min_score': {
                'strong': 52.0,
                'neutral': 56.0,
                'weak': 60.0,
                'crash': 999.0,
            }.get(regime, 56.0),
            'avoid_chasing': avoid_chasing,
            'tradable': tradable,
            'regime': regime,
            'strategy_mode': strategy_mode,
            'strategy_display': mode_profile['display'],
            'mode_profile': mode_profile,
            'sentiment_score': sentiment_score,
            'limit_up': limit_up,
            'limit_down': limit_down,
            'up_ratio': up_ratio / 100.0 if up_ratio > 1 else up_ratio,
            'avg_volatility': float(stats.get('avg_volatility', 0.0) or 0.0),
        }

    def _get_index_regime(self) -> Dict:
        """Estimate short-term index environment using major A-share indices."""
        symbols = {
            'sh000001': '000001.SH',
            'sz399001': '399001.SZ',
            'zz1000': '000852.SH',
        }
        details = []
        strength_score = 0
        valid_count = 0

        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - pd.Timedelta(days=45)).strftime('%Y%m%d')

        for name, ts_code in symbols.items():
            try:
                df = self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df is None or df.empty or len(df) < 12:
                    continue

                df = df.sort_values('trade_date')
                close = pd.to_numeric(df['close'], errors='coerce').dropna()
                if len(close) < 12:
                    continue

                last_close = float(close.iloc[-1])
                prev_close = float(close.iloc[-2])
                ma5 = float(close.tail(5).mean())
                ma10 = float(close.tail(10).mean())
                day_change = ((last_close - prev_close) / prev_close) * 100 if prev_close else 0.0

                index_score = 0
                if last_close >= ma5:
                    index_score += 1
                if last_close >= ma10:
                    index_score += 1
                if day_change > -1.0:
                    index_score += 1
                if day_change <= -2.0:
                    index_score -= 2

                strength_score += index_score
                valid_count += 1
                details.append({
                    'name': name,
                    'close': last_close,
                    'ma5': ma5,
                    'ma10': ma10,
                    'change_pct': round(day_change, 2),
                    'score': index_score,
                })
            except Exception:
                continue

        if valid_count == 0:
            regime = 'neutral'
            description = 'index data unavailable'
        elif any(item['change_pct'] <= -2.5 for item in details) or strength_score <= -1:
            regime = 'crash'
            description = 'index weakness'
        elif strength_score >= valid_count * 2:
            regime = 'strong'
            description = 'indices above short-term averages'
        elif strength_score <= valid_count:
            regime = 'weak'
            description = 'indices losing short-term trend'
        else:
            regime = 'neutral'
            description = 'mixed index trend'

        return {
            'regime': regime,
            'description': description,
            'valid_count': valid_count,
            'details': details,
        }

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return default
            return value
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_ts_code(code: str) -> str:
        code = str(code).zfill(6)
        if code.startswith('6'):
            return f'{code}.SH'
        if code.startswith('4') or code.startswith('8'):
            return f'{code}.BJ'
        return f'{code}.SZ'

    def _get_history_data_tushare(self, code: str, days: int = 90) -> Optional[pd.DataFrame]:
        code = str(code).zfill(6)
        cached = self.history_cache.get(code)
        if cached is not None and not cached.empty:
            return cached.tail(days).copy()

        ts_code = self._to_ts_code(code)
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - pd.Timedelta(days=days * 2)).strftime('%Y%m%d')
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
            out = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].tail(days)
            self.history_cache[code] = out.copy()
            return out
        except Exception:
            return None

    def _prefetch_history_data_tushare(self, codes: List[str], days: int = 90) -> None:
        """按交易日批量拉取 watchlist 历史，减少逐只请求。"""
        uniq_codes = sorted({str(c).zfill(6) for c in codes if str(c).strip()})
        if not uniq_codes:
            return

        ts_codes = {self._to_ts_code(code): code for code in uniq_codes}
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - pd.Timedelta(days=days * 2)).strftime('%Y%m%d')

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
                if self.verbose and (idx == 1 or idx % 10 == 0 or idx == total_dates):
                    print(f"[short_term] 历史预取 {idx}/{total_dates} trade_date={trade_date}", flush=True)
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
                self.history_cache[code] = g[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].tail(days).copy()
            if self.verbose:
                print(f"[short_term] 历史预取完成，缓存 {len(self.history_cache)} 只股票", flush=True)
        except Exception:
            return

    def _score_liquidity_activity(self, df: pd.DataFrame, stock_info: Dict) -> Dict:
        amount_series = pd.to_numeric(df.get('amount'), errors='coerce') if 'amount' in df.columns else pd.Series(dtype=float)
        close = pd.to_numeric(df['close'], errors='coerce')
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')

        current_amount = self._safe_float(stock_info.get('amount'))
        if current_amount <= 0 and not amount_series.empty:
            current_amount = self._safe_float(amount_series.iloc[-1])

        avg_amount_5 = self._safe_float(amount_series.tail(5).mean()) if not amount_series.empty else 0.0
        amount_ratio_5 = current_amount / avg_amount_5 if current_amount > 0 and avg_amount_5 > 0 else 0.0

        amplitude = ((high - low) / close.shift(1).replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan)
        amp_10 = self._safe_float(amplitude.tail(10).mean())
        returns = close.pct_change() * 100
        max_abs_return_5 = self._safe_float(returns.tail(5).abs().max())
        activity_proxy = amount_ratio_5 * max(amp_10, 1.0)

        score = 0.0
        tags = []
        tradable = True
        if current_amount >= 2e8:
            score += 8
            tags.append('成交额充足')
        elif current_amount >= 8e7:
            score += 5
            tags.append('成交额达标')
        elif current_amount >= 5e7:
            score += 1
            tags.append('成交额略低')
        else:
            score -= 6
            tags.append('成交额偏低')

        if 1.0 <= amount_ratio_5 <= 3.0:
            score += 6
            tags.append('近5日量能活跃')
        elif amount_ratio_5 > 3.5:
            score += 5
            tags.append('相对放量明显')
        else:
            score -= 2

        if 2.0 <= amp_10 <= 8.0:
            score += 4
        elif amp_10 < 1.5:
            score -= 4
            tags.append('波动不足')
        elif amp_10 > 12:
            score -= 2

        if max_abs_return_5 >= 4.0:
            score += 2
            tags.append('近期有异动')

        if current_amount < 3e7 and amount_ratio_5 < 1.0:
            tradable = False
            tags.append('流动性明显不足')

        return {
            'score': round(max(0.0, min(20.0, score)), 2),
            'tradable': tradable,
            'current_amount': current_amount,
            'avg_amount_5': avg_amount_5,
            'amount_ratio_5': round(amount_ratio_5, 2) if amount_ratio_5 > 0 else 0.0,
            'amp_10': round(amp_10, 2),
            'max_abs_return_5': round(max_abs_return_5, 2),
            'activity_proxy': round(activity_proxy, 2),
            'tags': tags,
        }

    def _score_sector_resonance(self, code: str, stock_info: Dict, sector_context: Dict) -> Dict:
        sector_heat = (sector_context or {}).get('sector_heat') or {}
        change_pct = self._safe_float(stock_info.get('change_pct'))
        score = 0.0
        tags = []

        if not sector_heat:
            return {
                'score': 0.0,
                'level': 'unknown',
                'industry': None,
                'tags': ['缺少板块画像'],
            }

        level = str(sector_heat.get('level') or 'neutral')
        avg_change = self._safe_float(sector_heat.get('avg_change'))
        strong_count = int(sector_heat.get('strong_count') or 0)
        limit_like_count = int(sector_heat.get('limit_like_count') or 0)
        up_ratio = self._safe_float(sector_heat.get('up_ratio'))
        sector_strength_score = self._safe_float(sector_heat.get('sector_strength_score'))
        sector_structure_score = self._safe_float(sector_heat.get('sector_structure_score'))
        leader_feedback_score = self._safe_float(sector_heat.get('leader_feedback_score'))
        is_mainstream = bool(sector_heat.get('is_mainstream'))

        if level == 'hot':
            score += 12.0
            tags.append('板块处于热点前排')
        elif level == 'warm':
            score += 8.0
            tags.append('板块热度较高')
        elif level == 'neutral':
            score += 3.0
        else:
            score -= 4.0
            tags.append('板块热度偏弱')

        if avg_change >= 2.5:
            score += 4.0
            tags.append('板块日内共振明显')
        elif avg_change >= 1.0:
            score += 2.0

        if strong_count >= 3:
            score += 2.5
            tags.append('板块内强势股较多')
        if limit_like_count >= 2:
            score += 2.5
            tags.append('板块梯队活跃')
        if up_ratio >= 65:
            score += 1.5
        elif up_ratio < 40:
            score -= 2.0

        if change_pct >= avg_change + 2.0:
            score += 1.5

        return {
            'score': round(max(0.0, min(20.0, score)), 2),
            'level': level,
            'is_mainstream': is_mainstream,
            'industry': sector_heat.get('industry'),
            'avg_change': round(avg_change, 2),
            'rank': sector_heat.get('rank'),
            'sector_strength_score': round(sector_strength_score, 2),
            'sector_structure_score': round(sector_structure_score, 2),
            'leader_feedback_score': round(leader_feedback_score, 2),
            'strong_count': strong_count,
            'limit_like_count': limit_like_count,
            'tags': tags,
        }

    def _score_leader_position(self, code: str, stock_info: Dict, sector_context: Dict, fund_flow: Optional[Dict], lhb: Optional[Dict]) -> Dict:
        sector_rank = (sector_context or {}).get('sector_rank') or {}
        score = 0.0
        tags = []

        if not sector_rank:
            return {
                'score': 0.0,
                'rank_in_sector': None,
                'leader_score': 0.0,
                'is_sector_leader': False,
                'tags': ['缺少板块内排名'],
            }

        rank_in_sector = int(sector_rank.get('rank_in_sector') or 0)
        leader_score = self._safe_float(sector_rank.get('leader_score'))
        change_pct = self._safe_float(stock_info.get('change_pct'))
        main_ratio = self._safe_float((fund_flow or {}).get('main_ratio'))
        lhb_net = self._safe_float((lhb or {}).get('net_amount'))

        if rank_in_sector == 1:
            score += 14.0
            tags.append('板块龙头')
        elif rank_in_sector <= 3:
            score += 11.0
            tags.append('板块前排')
        elif sector_rank.get('is_front_row'):
            score += 7.0
            tags.append('板块梯队靠前')
        else:
            score += 2.0

        if change_pct >= 7.0:
            score += 2.5
        elif change_pct >= 4.0:
            score += 1.0

        if main_ratio >= 5.0:
            score += 2.0
            tags.append('主力参与度较高')
        elif main_ratio < -3.0:
            score -= 2.0

        if lhb_net > 0:
            score += 1.5
            tags.append('龙虎榜净买')
        elif lhb_net < 0:
            score -= 1.0

        return {
            'score': round(max(0.0, min(20.0, score)), 2),
            'rank_in_sector': rank_in_sector if rank_in_sector > 0 else None,
            'total_in_sector': sector_rank.get('total_in_sector'),
            'leader_score': round(leader_score, 2),
            'is_sector_leader': bool(sector_rank.get('is_sector_leader')),
            'is_front_row': bool(sector_rank.get('is_front_row')),
            'leader_percentile': sector_rank.get('leader_percentile'),
            'tags': tags,
        }

    def _score_leadership_status(
        self,
        code: str,
        df: pd.DataFrame,
        stock_info: Dict,
        sector_context: Dict,
        fund_flow: Optional[Dict],
        lhb: Optional[Dict],
    ) -> Dict:
        close = pd.to_numeric(df['close'], errors='coerce')
        high = pd.to_numeric(df['high'], errors='coerce')
        open_ = pd.to_numeric(df['open'], errors='coerce')
        amount = pd.to_numeric(df.get('amount'), errors='coerce') if 'amount' in df.columns else pd.Series(dtype=float)
        sector_rank = (sector_context or {}).get('sector_rank') or {}

        returns = close.pct_change() * 100
        limit_like_5d = int((returns.tail(5) >= 9.5).sum()) if len(returns) >= 2 else 0

        streak = 0
        for value in reversed(returns.fillna(0).tail(5).tolist()):
            if value >= 9.5:
                streak += 1
            else:
                break

        recent_3d_heat = 0
        for value in returns.tail(3).fillna(0).tolist():
            if value >= 7.0:
                recent_3d_heat += 1
            elif value >= 4.0:
                recent_3d_heat += 0.5

        close_now = self._safe_float(close.iloc[-1]) if not close.empty else 0.0
        high_now = self._safe_float(high.iloc[-1]) if not high.empty else 0.0
        open_now = self._safe_float(open_.iloc[-1]) if not open_.empty else 0.0
        reclose_strength = 0.0
        if close_now > 0 and high_now >= close_now:
            reclose_strength = self._safe_float(close_now / max(high_now, 0.01))
        intraday_reseal = bool(high_now > close_now * 1.01 and close_now >= open_now and reclose_strength >= 0.985)

        amount_rank_bonus = 0.0
        if sector_rank:
            rank_in_sector = int(sector_rank.get('rank_in_sector') or 999)
            if rank_in_sector == 1:
                amount_rank_bonus = 4.0
            elif rank_in_sector <= 3:
                amount_rank_bonus = 2.5
            elif rank_in_sector <= 5:
                amount_rank_bonus = 1.0
        current_amount = self._safe_float(stock_info.get('amount'))
        if current_amount <= 0 and not amount.empty:
            current_amount = self._safe_float(amount.iloc[-1])

        main_ratio = self._safe_float((fund_flow or {}).get('main_ratio'))
        lhb_net = self._safe_float((lhb or {}).get('net_amount'))
        score = 0.0
        tags: List[str] = []

        if limit_like_5d >= 2:
            score += 6.0
            tags.append('近5日有连板/准连板')
        elif limit_like_5d == 1:
            score += 3.0
        if streak >= 2:
            score += 6.0
            tags.append(f'连板高度 {streak}')
        elif streak == 1:
            score += 2.5

        if recent_3d_heat >= 2.0:
            score += 3.0
            tags.append('近3日辨识度较高')
        elif recent_3d_heat >= 1.0:
            score += 1.5

        score += amount_rank_bonus
        if amount_rank_bonus >= 2.5:
            tags.append('成交额位于题材前列')

        if main_ratio >= 5.0:
            score += 2.5
            tags.append('资金聚焦核心')
        elif main_ratio >= 2.0:
            score += 1.0

        if lhb_net > 0:
            score += 1.5
            tags.append('龙虎榜反馈偏正')
        elif lhb_net < 0:
            score -= 1.5

        if intraday_reseal:
            score += 1.5
            tags.append('开板回封/收盘强度较好')

        leader_tier = 'follower'
        if streak >= 2 or (limit_like_5d >= 1 and sector_rank.get('is_sector_leader')):
            leader_tier = 'leader'
        elif sector_rank.get('is_sector_leader') or (sector_rank.get('is_front_row') and recent_3d_heat >= 1.0):
            leader_tier = 'secondary_leader'

        return {
            'score': round(max(0.0, min(20.0, score)), 2),
            'leader_tier': leader_tier,
            'limit_like_5d': limit_like_5d,
            'streak_height': streak,
            'recent_3d_heat_mentions': recent_3d_heat,
            'intraday_reseal': intraday_reseal,
            'current_amount': round(current_amount, 2),
            'main_ratio': round(main_ratio, 2),
            'lhb_net': round(lhb_net, 2),
            'tags': tags,
        }

    def _classify_trend_state(self, df: pd.DataFrame) -> Dict:
        close = pd.to_numeric(df['close'], errors='coerce')
        high = pd.to_numeric(df['high'], errors='coerce')
        volume = pd.to_numeric(df['volume'], errors='coerce')
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        returns_3d = self._safe_float((close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100) if len(close) >= 4 and self._safe_float(close.iloc[-4]) != 0 else 0.0
        returns_10d = self._safe_float((close.iloc[-1] - close.iloc[-11]) / close.iloc[-11] * 100) if len(close) >= 11 and self._safe_float(close.iloc[-11]) != 0 else 0.0
        volume_ratio = self._safe_float(volume.iloc[-1] / volume.tail(5).mean()) if len(volume) >= 5 and self._safe_float(volume.tail(5).mean()) > 0 else 1.0
        rsi = self.indicators.calc_rsi(df)
        rsi_now = self._safe_float(rsi.iloc[-1], 50.0)

        label = '中性整理'
        score = 0.0
        if close.iloc[-1] > ma5.iloc[-1] > ma10.iloc[-1] and volume_ratio >= 1.5 and returns_3d > 2.5:
            label = '放量试图突破'
            score = 18.0
        elif close.iloc[-1] > ma5.iloc[-1] and returns_10d > 2 and 0 < returns_3d <= 5 and 1.0 <= volume_ratio <= 2.2:
            label = '试盘转强'
            score = 14.0
        elif close.iloc[-1] >= ma10.iloc[-1] and 0.75 <= volume_ratio <= 1.1 and -2 <= returns_3d <= 1.5:
            label = '缩量横住'
            score = 12.0
        elif close.iloc[-1] >= ma5.iloc[-1] and returns_3d < 2 and volume_ratio > 1.1 and returns_10d > 0:
            label = '分歧承接'
            score = 13.0
        elif rsi_now < 35 and returns_10d < -5 and -1 <= returns_3d <= 2:
            label = '超跌修复'
            score = 10.0
        elif rsi_now > 75 and returns_10d > 12:
            label = '放量滞涨'
            score = 3.0
        elif returns_3d > 3 and volume_ratio > 1.8 and close.iloc[-1] < high.iloc[-1] * 0.97:
            label = '冲高回落'
            score = -3.0
        elif close.iloc[-1] < ma5.iloc[-1] < ma10.iloc[-1] and returns_3d < -2:
            label = '高位转弱'
            score = -8.0

        return {
            'label': label,
            'score': score,
            'returns_3d': round(returns_3d, 2),
            'returns_10d': round(returns_10d, 2),
            'volume_ratio': round(volume_ratio, 2),
            'rsi': round(rsi_now, 2),
        }

    def _classify_trend_state(self, df: pd.DataFrame) -> Dict:
        close = pd.to_numeric(df['close'], errors='coerce')
        high = pd.to_numeric(df['high'], errors='coerce')
        volume = pd.to_numeric(df['volume'], errors='coerce')
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        returns_3d = self._safe_float((close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100) if len(close) >= 4 and self._safe_float(close.iloc[-4]) != 0 else 0.0
        returns_10d = self._safe_float((close.iloc[-1] - close.iloc[-11]) / close.iloc[-11] * 100) if len(close) >= 11 and self._safe_float(close.iloc[-11]) != 0 else 0.0
        volume_ratio = self._safe_float(volume.iloc[-1] / volume.tail(5).mean()) if len(volume) >= 5 and self._safe_float(volume.tail(5).mean()) > 0 else 1.0
        rsi = self.indicators.calc_rsi(df)
        rsi_now = self._safe_float(rsi.iloc[-1], 50.0)

        label = '中性整理'
        state_code = 'neutral'
        score = 0.0
        if close.iloc[-1] > ma5.iloc[-1] > ma10.iloc[-1] and volume_ratio >= 1.5 and returns_3d > 2.5:
            label = '放量试图突破'
            state_code = 'breakout_attempt'
            score = 18.0
        elif close.iloc[-1] > ma5.iloc[-1] and returns_10d > 2 and 0 < returns_3d <= 5 and 1.0 <= volume_ratio <= 2.2:
            label = '试盘转强'
            state_code = 'test_strength'
            score = 14.0
        elif close.iloc[-1] >= ma10.iloc[-1] and 0.75 <= volume_ratio <= 1.1 and -2 <= returns_3d <= 1.5:
            label = '缩量横住'
            state_code = 'tight_hold'
            score = 12.0
        elif close.iloc[-1] >= ma5.iloc[-1] and returns_3d < 2 and volume_ratio > 1.1 and returns_10d > 0:
            label = '分歧承接'
            state_code = 'dip_support'
            score = 13.0
        elif rsi_now < 35 and returns_10d < -5 and -1 <= returns_3d <= 2:
            label = '超跌修复'
            state_code = 'oversold_rebound'
            score = 10.0
        elif rsi_now > 75 and returns_10d > 12:
            label = '放量滞涨'
            state_code = 'overheated_stall'
            score = 3.0
        elif returns_3d > 3 and volume_ratio > 1.8 and close.iloc[-1] < high.iloc[-1] * 0.97:
            label = '冲高回落'
            state_code = 'failed_breakout'
            score = -3.0
        elif close.iloc[-1] < ma5.iloc[-1] < ma10.iloc[-1] and returns_3d < -2:
            label = '高位转弱'
            state_code = 'high_weakness'
            score = -8.0

        return {
            'label': label,
            'state_code': state_code,
            'score': score,
            'returns_3d': round(returns_3d, 2),
            'returns_10d': round(returns_10d, 2),
            'volume_ratio': round(volume_ratio, 2),
            'rsi': round(rsi_now, 2),
        }

    def _score_funding_confirmation(self, code: str, df: pd.DataFrame, fund_flow: Optional[Dict]) -> Dict:
        close = pd.to_numeric(df['close'], errors='coerce')
        amount = pd.to_numeric(df.get('amount'), errors='coerce') if 'amount' in df.columns else pd.Series(dtype=float)
        day_change = self._safe_float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) >= 2 and self._safe_float(close.iloc[-2]) != 0 else 0.0
        amount_ratio = self._safe_float(amount.iloc[-1] / amount.tail(5).mean()) if len(amount) >= 5 and self._safe_float(amount.tail(5).mean()) > 0 else 1.0
        main_in = self._safe_float(fund_flow.get('main_in')) if fund_flow else 0.0
        main_in_wan = main_in / 10000

        score = 0.0
        tags = []
        if main_in > 0 and day_change > 0:
            score += 10.0 if amount_ratio >= 1.3 else 7.0
            tags.append('资金与价格共振')
        elif main_in > 0 and day_change <= 0:
            score += 3.0
        elif main_in < 0 and day_change < -1:
            score -= 8.0
            tags.append('资金流出且价格走弱')
        elif day_change > 2 and main_in <= 0:
            score += 1.0

        if amount_ratio >= 2.0 and day_change > 2:
            score += 4.0
            tags.append('放量上涨')

        lhb = self.cache.get_lhb(code)
        if lhb:
            net_amount = self._safe_float(lhb.get('net_amount'))
            if net_amount > 0 and day_change > 0:
                score += 7.0
                tags.append('龙虎榜净买')
            elif net_amount < 0:
                score -= 10.0
                tags.append('龙虎榜净卖')
        return {
            'score': round(max(-15.0, min(15.0, score)), 2),
            'main_in': round(main_in_wan, 2),
            'amount_ratio': round(amount_ratio, 2),
            'day_change': round(day_change, 2),
            'lhb': lhb,
            'tags': tags,
        }

    def _risk_veto(self, code: str, df: pd.DataFrame, fund_flow: Optional[Dict], market_context: Dict) -> Dict:
        open_ = pd.to_numeric(df['open'], errors='coerce')
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        amount = pd.to_numeric(df.get('amount'), errors='coerce') if 'amount' in df.columns else pd.Series(dtype=float)
        reasons = []
        penalty = 0.0
        veto = False

        if len(df) >= 2:
            upper_shadow = high.iloc[-1] - max(open_.iloc[-1], close.iloc[-1])
            body = abs(close.iloc[-1] - open_.iloc[-1])
            amount_ratio = self._safe_float(amount.iloc[-1] / amount.tail(5).mean()) if len(amount) >= 5 and self._safe_float(amount.tail(5).mean()) > 0 else 1.0
            if body >= 0 and upper_shadow > body * 1.5 and amount_ratio > 1.5:
                penalty += 8
                reasons.append('长上影放量')

            if open_.iloc[-1] > close.iloc[-2] * 1.02 and close.iloc[-1] < open_.iloc[-1] * 0.985:
                penalty += 8
                reasons.append('高开低走大阴线')

            recent_changes = close.pct_change().tail(3).fillna(0) * 100
            recent_amount_ratio = self._safe_float(amount.tail(3).mean() / amount.tail(8).mean()) if len(amount) >= 8 and self._safe_float(amount.tail(8).mean()) > 0 else 1.0
            if recent_amount_ratio > 1.5 and recent_changes.mean() < 0.5:
                penalty += 6
                reasons.append('连续放量不涨')

        main_in = self._safe_float(fund_flow.get('main_in')) if fund_flow else 0.0
        if main_in < -8_000_000:
            penalty += 6
            reasons.append('主力净流出偏大')

        if len(df) >= 6:
            recent_high = close.tail(5).max()
            amount_ratio = self._safe_float(amount.iloc[-1] / amount.tail(5).mean()) if len(amount) >= 5 and self._safe_float(amount.tail(5).mean()) > 0 else 1.0
            if close.iloc[-1] >= recent_high * 0.985 and amount_ratio > 1.5 and close.iloc[-1] < high.iloc[-1] * 0.975:
                penalty += 7
                reasons.append('高位放量滞涨')

        lhb = self.cache.get_lhb(code)
        if lhb:
            net_amount = self._safe_float(lhb.get('net_amount'))
            if net_amount < 0:
                penalty += 6
                reasons.append('龙虎榜净卖')
                if market_context.get('sentiment_score', 50) < 45:
                    penalty += 4
                    reasons.append('弱市龙虎榜净卖')

        if market_context.get('sentiment', {}).get('score', 50) < 45 and close.iloc[-1] > open_.iloc[-1] and (high.iloc[-1] - close.iloc[-1]) > (close.iloc[-1] - open_.iloc[-1]):
            penalty += 4
            reasons.append('弱市冲高回落')

        if penalty >= 12:
            veto = True

        return {
            'veto': veto,
            'penalty': round(penalty, 2),
            'reasons': reasons,
        }

    def sync_watchlist_from_cache(self) -> int:
        """将 stock_cache.db 中 stocks 表的全部股票代码写入 watchlist.json"""
        watchlist_file = Path(__file__).resolve().parent / 'watchlist.json'

        try:
            cursor = self.cache.conn.cursor()
            cursor.execute('SELECT code FROM stocks ORDER BY code')
            rows = cursor.fetchall()

            codes = []
            for row in rows:
                code = self.cache.normalize_code(row[0])
                if not code:
                    continue
                codes.append(code)

            # 去重并保持顺序
            unique_codes = list(dict.fromkeys(codes))

            with open(watchlist_file, 'w', encoding='utf-8') as f:
                json.dump(unique_codes, f, ensure_ascii=False, indent=2)

            if self.verbose:
                print(f"✅ 已从 stock_cache.db 同步 {len(unique_codes)} 只股票到 watchlist.json")
            return len(unique_codes)
        except Exception as e:
            print(f"❌ 从 stock_cache.db 同步 watchlist 失败: {e}")
            return 0

    def _score_momentum_relay(self, features: Dict) -> Dict:
        sector_resonance = features['sector_resonance']
        leader_position = features['leader_position']
        leadership_status = features['leadership_status']
        trend = features['trend']
        funding = features['funding']
        liquidity = features['liquidity']
        risk = features['risk']

        trend_code = str(trend.get('state_code') or 'neutral')
        leader_tier = str(leadership_status.get('leader_tier') or 'follower')
        sector_mainstream = bool(sector_resonance.get('is_mainstream'))
        front_row = bool(leader_position.get('is_sector_leader') or leader_position.get('is_front_row'))
        risk_penalty = self._safe_float(risk.get('penalty'))

        score = (
            self._safe_float(sector_resonance.get('score')) * 1.1
            + self._safe_float(leader_position.get('score')) * 1.1
            + self._safe_float(leadership_status.get('score')) * 1.15
            + max(0.0, min(12.0, self._safe_float(funding.get('score'))))
            + max(0.0, min(8.0, self._safe_float(trend.get('score')) * 0.7))
            + max(0.0, min(6.0, self._safe_float(liquidity.get('score')) * 0.4))
            - risk_penalty
        )
        candidate_score = max(0.0, min(100.0, 32.0 + score * 0.95))
        if not sector_mainstream:
            candidate_score = min(candidate_score, 60.0)
        if leader_tier == 'follower':
            candidate_score = min(candidate_score, 62.0)

        is_candidate = (
            candidate_score >= 48.0
            and trend_code in ['breakout_attempt', 'test_strength', 'tight_hold']
            and (
                (
                    sector_resonance.get('level') in ['hot', 'warm']
                    and sector_mainstream
                    and self._safe_float(sector_resonance.get('sector_structure_score')) >= 6.0
                    and self._safe_float(sector_resonance.get('leader_feedback_score')) >= 4.0
                    and front_row
                )
                or (
                    leader_tier in ['leader', 'secondary_leader']
                    and front_row
                    and candidate_score >= 58.0
                )
            )
        )
        recommend = (
            is_candidate
            and leader_tier == 'leader'
            and trend_code in ['breakout_attempt', 'test_strength']
            and risk_penalty < 10
        )
        return {
            'candidate_score': round(candidate_score, 2),
            'is_candidate': bool(is_candidate),
            'recommend': bool(recommend),
            'mode_reason': '强势接力优先前排龙头、加速与回封强度',
        }

    def _score_dip_rebound(self, features: Dict) -> Dict:
        sector_resonance = features['sector_resonance']
        leader_position = features['leader_position']
        leadership_status = features['leadership_status']
        trend = features['trend']
        funding = features['funding']
        liquidity = features['liquidity']
        risk = features['risk']

        trend_code = str(trend.get('state_code') or 'neutral')
        leader_tier = str(leadership_status.get('leader_tier') or 'follower')
        front_row = bool(leader_position.get('is_sector_leader') or leader_position.get('is_front_row'))
        risk_penalty = self._safe_float(risk.get('penalty'))
        dip_bonus = 4.0 if trend_code in ['dip_support', 'tight_hold', 'oversold_rebound'] else 0.0

        score = (
            self._safe_float(sector_resonance.get('score')) * 1.0
            + self._safe_float(features['leader_position'].get('score')) * 0.9
            + self._safe_float(leadership_status.get('score')) * 1.0
            + max(0.0, min(9.0, self._safe_float(funding.get('score')) * 0.8))
            + max(0.0, min(6.0, self._safe_float(liquidity.get('score')) * 0.4))
            + dip_bonus
            - risk_penalty
        )
        candidate_score = max(0.0, min(100.0, 34.0 + score * 0.92))
        if trend_code == 'breakout_attempt':
            candidate_score = min(candidate_score, 70.0)
        if leader_tier == 'follower':
            candidate_score = min(candidate_score, 64.0)

        is_candidate = (
            candidate_score >= 40.0
            and trend_code in ['dip_support', 'tight_hold', 'oversold_rebound']
            and (
                (
                    bool(sector_resonance.get('is_mainstream'))
                    and self._safe_float(sector_resonance.get('sector_structure_score')) >= 5.0
                    and self._safe_float(sector_resonance.get('leader_feedback_score')) >= 3.0
                    and leader_tier in ['leader', 'secondary_leader']
                )
                or (
                    bool(sector_resonance.get('is_mainstream'))
                    and front_row
                    and candidate_score >= 52.0
                )
            )
        )
        recommend = (
            is_candidate
            and leader_tier in ['leader', 'secondary_leader']
            and trend_code in ['dip_support', 'tight_hold', 'oversold_rebound']
            and risk_penalty < 10
        )
        return {
            'candidate_score': round(candidate_score, 2),
            'is_candidate': bool(is_candidate),
            'recommend': bool(recommend),
            'mode_reason': '分歧低吸优先主流题材核心的承接与缩量结构',
        }

    def _score_defensive(self, features: Dict) -> Dict:
        sector_resonance = features['sector_resonance']
        leadership_status = features['leadership_status']
        trend = features['trend']
        funding = features['funding']
        liquidity = features['liquidity']
        risk = features['risk']

        trend_code = str(trend.get('state_code') or 'neutral')
        leader_tier = str(leadership_status.get('leader_tier') or 'follower')
        risk_penalty = self._safe_float(risk.get('penalty'))

        score = (
            self._safe_float(sector_resonance.get('score')) * 0.8
            + self._safe_float(features['leader_position'].get('score')) * 0.8
            + self._safe_float(leadership_status.get('score')) * 1.1
            + max(0.0, min(8.0, self._safe_float(funding.get('score')) * 0.6))
            + max(0.0, min(5.0, self._safe_float(liquidity.get('score')) * 0.35))
            - risk_penalty * 1.15
        )
        candidate_score = max(0.0, min(100.0, 30.0 + score * 0.9))
        if leader_tier != 'leader':
            candidate_score = min(candidate_score, 58.0)

        is_candidate = (
            candidate_score >= 54.0
            and bool(sector_resonance.get('is_mainstream'))
            and self._safe_float(sector_resonance.get('leader_feedback_score')) >= 5.0
            and leader_tier in ['leader', 'secondary_leader']
            and trend_code not in ['high_weakness', 'failed_breakout', 'overheated_stall']
            and risk_penalty < 9
        )
        recommend = bool(is_candidate and risk_penalty < 6)
        return {
            'candidate_score': round(candidate_score, 2),
            'is_candidate': bool(is_candidate),
            'recommend': bool(recommend),
            'mode_reason': '弱市防守只保留主流核心龙头，强调低风险与辨识度',
        }

    def load_watchlist(self) -> List[str]:
        """先同步数据库股票池到 watchlist.json，再加载并过滤创业板/科创板和北交所"""
        watchlist_file = Path(__file__).resolve().parent / 'watchlist.json'

        # 每次运行先同步一份最新 watchlist
        self.sync_watchlist_from_cache()

        try:
            with open(watchlist_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)

            if not isinstance(raw, list):
                print("❌ watchlist.json 格式错误：必须是数组")
                return []

            all_stocks = []
            for item in raw:
                code = self.cache.normalize_code(item)
                if not code:
                    continue
                all_stocks.append(code)

            filtered = [
                code for code in all_stocks
                if not code.startswith('3')
                and not code.startswith('688')
                and not code.startswith('920')
            ]

            if not filtered:
                print("❌ 监控列表过滤后为空（可能全部是创业板/科创板和北交所）")
            return filtered
        except FileNotFoundError:
            print("❌ 未找到 watchlist.json")
            return []
        except Exception as e:
            print(f"❌ 读取 watchlist.json 失败: {e}")
            return []

    def _prefilter_short_term_universe(self, watchlist: List[str], max_universe: int = 1500) -> List[str]:
        """短线只分析活跃股票，避免全市场 1w+ 股票导致长时间无输出。"""
        all_stocks = self.cache.get_all_stocks(max_age_minutes=1440)
        if not all_stocks:
            return watchlist[:max_universe]

        watchset = {str(code).zfill(6) for code in watchlist}
        candidates = []
        for stock in all_stocks:
            code = str(stock.get('code', '')).zfill(6)
            if code not in watchset:
                continue
            amount = self._safe_float(stock.get('amount'))
            change_pct = self._safe_float(stock.get('change_pct'))
            activity_score = amount / 1e8 + abs(change_pct) * 2.0
            if amount >= 5e7 or abs(change_pct) >= 3.0:
                candidates.append((activity_score, code))

        if not candidates:
            return watchlist[:max_universe]

        ranked_codes = [code for _, code in sorted(candidates, key=lambda item: item[0], reverse=True)]
        return ranked_codes[:max_universe]

    def analyze_single_stock(self, code: str) -> Dict:
        """一阶量化：先生成短线候选，再决定是否进入最终推荐。"""
        try:
            df = self._get_history_data_tushare(code, days=90)
            if df is None or df.empty or len(df) < 20:
                return None

            stock_info = self.cache.get_stock(code)
            if not stock_info:
                return None

            market_context = self.market_context

            name = stock_info.get('name', 'Unknown')
            current_price = self._safe_float(stock_info.get('price'), self._safe_float(df['close'].iloc[-1]))
            change_pct = self._safe_float(stock_info.get('change_pct'), 0.0)
            fund_flow = self.cache.get_fund_flow(code) or {}
            lhb = self.cache.get_lhb(code)
            sector_context = self._get_stock_sector_context(code)
            strategy_mode = str(market_context.get('strategy_mode') or 'momentum_relay')
            mode_profile = market_context.get('mode_profile') or {}

            liquidity = self._score_liquidity_activity(df, stock_info)
            sector_resonance = self._score_sector_resonance(code, stock_info, sector_context)
            leader_position = self._score_leader_position(code, stock_info, sector_context, fund_flow, lhb)
            leadership_status = self._score_leadership_status(code, df, stock_info, sector_context, fund_flow, lhb)

            trend = self._classify_trend_state(df)
            trend_state = trend.get('label', '中性整理')
            trend_code = trend.get('state_code', 'neutral')
            trend_score = self._safe_float(trend.get('score'), 0.0)

            funding = self._score_funding_confirmation(code, df, fund_flow)
            risk = self._risk_veto(code, df, fund_flow, market_context)

            rsi = self.indicators.calc_rsi(df)
            k, d, j = self.indicators.calc_kdj(df)
            dif, dea, macd_hist = self.indicators.calc_macd_short(df)
            volume_surge = self.indicators.detect_volume_surge(df, ratio=1.5)

            technical_score = 0.0
            buy_signals = []
            sell_signals = []

            rsi_now = self._safe_float(rsi.iloc[-1], 50.0)
            k_now = self._safe_float(k.iloc[-1], 50.0)
            d_now = self._safe_float(d.iloc[-1], 50.0)
            dif_now = self._safe_float(dif.iloc[-1], 0.0)
            dea_now = self._safe_float(dea.iloc[-1], 0.0)
            macd_now = self._safe_float(macd_hist.iloc[-1], 0.0)
            volume_ratio = self._safe_float(volume_surge.get('volume_ratio'), 1.0)

            if 35 <= rsi_now <= 82:
                technical_score += 1.2
                buy_signals.append('RSI健康')
            elif rsi_now < 28:
                technical_score += 0.8
                buy_signals.append('RSI低位')
            elif rsi_now > 90:
                technical_score -= 1.2
                sell_signals.append('RSI过热')

            if dif_now > dea_now and macd_now >= -0.02:
                technical_score += 1.5
                buy_signals.append('MACD偏强')
            elif dif_now < dea_now and macd_now < -0.05:
                technical_score -= 1.8
                sell_signals.append('MACD走弱')

            if k_now > d_now and k_now < 92:
                technical_score += 0.8
                buy_signals.append('KDJ偏强')
            elif k_now < d_now and k_now < 35:
                technical_score -= 0.8
                sell_signals.append('KDJ高位转弱')

            if volume_ratio >= 1.3 and change_pct > 0:
                technical_score += 1.0
                buy_signals.append('量价配合')
            elif volume_ratio >= 1.8 and change_pct <= 0:
                technical_score -= 1.5
                sell_signals.append('放量承接一般')

            technical_score = max(-3.0, min(6.0, technical_score))

            if market_context.get('avoid_chasing'):
                if trend_code in ['breakout_attempt', 'test_strength']:
                    trend_score -= 6.0
                elif trend_code in ['dip_support', 'tight_hold', 'oversold_rebound']:
                    trend_score += 2.0

            liquidity_score = self._safe_float(liquidity.get('score'), 0.0)
            funding_score = self._safe_float(funding.get('score'), 0.0)
            risk_penalty = self._safe_float(risk.get('penalty'), 0.0)
            sector_score = self._safe_float(sector_resonance.get('score'), 0.0)
            leader_score = self._safe_float(leader_position.get('score'), 0.0)
            leadership_score = self._safe_float(leadership_status.get('score'), 0.0)
            leader_tier = str(leadership_status.get('leader_tier') or 'follower')

            features = {
                'liquidity': liquidity,
                'sector_resonance': sector_resonance,
                'leader_position': leader_position,
                'leadership_status': leadership_status,
                'trend': {**trend, 'score': trend_score},
                'funding': funding,
                'risk': risk,
                'technical_score': technical_score,
                'market_context': market_context,
            }

            if strategy_mode == 'momentum_relay':
                mode_result = self._score_momentum_relay(features)
            elif strategy_mode == 'dip_rebound':
                mode_result = self._score_dip_rebound(features)
            else:
                mode_result = self._score_defensive(features)

            candidate_score = self._safe_float(mode_result.get('candidate_score'))
            is_candidate = bool(mode_result.get('is_candidate'))
            recommend = bool(mode_result.get('recommend'))
            mode_reason = str(mode_result.get('mode_reason') or '')

            if lhb and self._safe_float(lhb.get('net_amount')) < 0 and market_context.get('sentiment_score', 50) < 45:
                recommend = False
                risk_penalty += 2
                if '弱市龙虎榜净卖' not in risk['reasons']:
                    risk['reasons'].append('弱市龙虎榜净卖')
            if lhb and self._safe_float(lhb.get('net_amount')) < 0 and market_context.get('sentiment_score', 50) < 45:
                recommend = False
                risk_penalty += 2
                if '弱市龙虎榜净卖' not in risk['reasons']:
                    risk['reasons'].append('弱市龙虎榜净卖')

            reasons = []
            reasons.extend(sector_resonance.get('tags', [])[:2])
            reasons.extend(leader_position.get('tags', [])[:2])
            reasons.extend(leadership_status.get('tags', [])[:2])
            reasons.extend(liquidity.get('tags', [])[:2])
            reasons.append(f'趋势状态: {trend_state}')
            if mode_reason:
                reasons.append(mode_reason)
            reasons.extend(funding.get('tags', [])[:2])
            if technical_score >= 6:
                reasons.append('技术辅助偏强')

            details = {
                'liquidity': liquidity,
                'sector_resonance': sector_resonance,
                'leader_position': leader_position,
                'leadership_status': leadership_status,
                'sector_context': sector_context,
                'trend': trend,
                'funding': funding,
                'risk': risk,
                'technical': {
                    'rsi': round(rsi_now, 2),
                    'k': round(k_now, 2),
                    'd': round(d_now, 2),
                    'dif': round(dif_now, 4),
                    'dea': round(dea_now, 4),
                    'macd': round(macd_now, 4),
                    'volume_ratio': round(volume_ratio, 2),
                },
            }

            return {
                'code': code,
                'name': name,
                'price': current_price,
                'change_pct': change_pct,
                'score': round(candidate_score, 2),
                'candidate_score': round(candidate_score, 2),
                'sector_score': round(sector_score, 2),
                'leader_score': round(leader_score, 2),
                'leadership_score': round(leadership_score, 2),
                'sector_strength_score': round(self._safe_float(sector_resonance.get('sector_strength_score')), 2),
                'sector_structure_score': round(self._safe_float(sector_resonance.get('sector_structure_score')), 2),
                'leader_feedback_score': round(self._safe_float(sector_resonance.get('leader_feedback_score')), 2),
                'liquidity_score': round(liquidity_score, 2),
                'trend_score': round(trend_score, 2),
                'funding_score': round(funding_score, 2),
                'technical_score': round(technical_score, 2),
                'risk_penalty': round(risk_penalty, 2),
                'trend_state': trend_state,
                'industry': sector_context.get('industry'),
                'themes': sector_context.get('themes') or [],
                'primary_theme': sector_context.get('primary_theme'),
                'sector_level': sector_resonance.get('level'),
                'rank_in_sector': leader_position.get('rank_in_sector'),
                'leader_tier': leader_tier,
                'strategy_mode': strategy_mode,
                'mode_reason': mode_reason,
                'risk_flags': risk.get('reasons', []),
                'signals': reasons,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'reasons': reasons,
                'recommend': recommend,
                'is_candidate': is_candidate,
                'market_fit': market_context.get('tradable', True),
                'market_context': self._convert_to_json_safe({
                    'sentiment_score': market_context.get('sentiment_score', 50.0),
                    'regime': market_context.get('regime', 'neutral'),
                    'score_adjustment': market_context.get('score_adjustment', 0.0),
                    'candidate_pool_size': market_context.get('candidate_pool_size', 15),
                    'top_n_cap': market_context.get('top_n_cap', 5),
                    'avoid_chasing': market_context.get('avoid_chasing', False),
                    'strategy_mode': strategy_mode,
                    'strategy_display': market_context.get('strategy_display'),
                }),
                'details': self._convert_to_json_safe(details),
                'rating': self._get_rating(candidate_score),
                'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }

        except Exception as e:
            if self.verbose:
                import traceback
                print(f'分析{code}失败: {e}')
                traceback.print_exc()
            return None

    def _get_rating(self, score: float) -> str:
        """
        评级
        A+/A: 强候选
        B+/B: 可跟踪
        C: 一般观察
        """
        if score >= 78:
            return 'A+'
        elif score >= 70:
            return 'A'
        elif score >= 62:
            return 'B+'
        elif score >= 54:
            return 'B'
        else:
            return 'C'
    
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
    
    def select_top_stocks(self, top_n: int = 10, verbose: bool = None) -> List[Dict]:
        """两段式筛选：先形成候选池，再给出最终推荐。"""
        if verbose is None:
            verbose = self.verbose

        market_context = self.market_context
        watchlist = self.load_watchlist()
        if not watchlist:
            if verbose:
                print('监控列表为空')
            self.last_results = []
            self.last_candidates = []
            return []

        raw_watchlist_count = len(watchlist)
        print(f"[short_term] 股票池: 全量 {raw_watchlist_count}", flush=True)
        print("[short_term] 预拉历史数据中...", flush=True)
        self._prefetch_history_data_tushare(watchlist, days=90)
        print("[short_term] 构建题材/板块上下文...", flush=True)
        self._build_sector_context()

        if verbose:
            print('\n市场环境:')
            print(f"  情绪分: {market_context.get('sentiment_score', 0):.1f}")
            print(f"  状态: {market_context.get('regime', 'neutral')}")
            print(f"  候选池上限: {market_context.get('candidate_pool_size', 0)}")
            print(f"  最终推荐上限: {market_context.get('top_n_cap', 0)}")
            print(f"  避免追高: {market_context.get('avoid_chasing', False)}")

        results = []
        total = len(watchlist)
        for i, code in enumerate(watchlist, 1):
            if verbose:
                print(f'[{i}/{total}] {code}...', end=' ')
            elif i == 1 or i % 100 == 0 or i == total:
                print(f"[short_term] 分析进度 {i}/{total}", flush=True)

            result = self.analyze_single_stock(code)
            if result is None:
                if verbose:
                    print('分析失败')
                continue

            results.append(result)
            if verbose:
                status = '候选' if result.get('is_candidate') else '跳过'
                print(f"{status} {result['candidate_score']:.1f}分 | {result.get('trend_state', '-')}")

        ranked = sorted(results, key=lambda item: item.get('candidate_score', 0.0), reverse=True)

        candidate_pool_size = int(market_context.get('candidate_pool_size', 20) or 0)
        top_n_cap = int(market_context.get('top_n_cap', top_n) or 0)
        target_n = max(0, min(int(top_n), top_n_cap))

        candidate_pool = [
            item for item in ranked
            if item.get('is_candidate')
        ]
        if candidate_pool_size > 0:
            candidates = candidate_pool[:candidate_pool_size]
        else:
            candidates = []

        recommended_pool = [
            item for item in candidates
            if item.get('recommend')
        ]
        final_recommendations = recommended_pool[:target_n] if target_n > 0 else []

        self.last_results = results
        self.last_candidates = candidates

        if verbose:
            print(f'\n全量分析: {len(results)}')
            print(f'一阶候选池: {len(candidates)}')
            print(f'最终推荐池: {len(final_recommendations)}')

        return final_recommendations

    def generate_report(self, stocks: List[Dict]) -> str:
        """生成短线候选与最终推荐报告。"""
        report = []
        market = self.market_context

        report.append('=' * 60)
        report.append('短线候选与推荐报告')
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append('=' * 60)
        report.append('')
        report.append('一、市场环境')
        report.append(f"情绪分数: {market.get('sentiment_score', 0):.1f}")
        report.append(f"市场状态: {market.get('regime', 'neutral')}")
        report.append(f"上涨占比: {market.get('up_ratio', 0):.2%}")
        report.append(f"涨停/跌停: {market.get('limit_up', 0)} / {market.get('limit_down', 0)}")
        report.append(f"平均波动率: {market.get('avg_volatility', 0):.2f}")
        report.append(f"候选池上限: {market.get('candidate_pool_size', 0)}")
        report.append(f"最终推荐上限: {market.get('top_n_cap', 0)}")
        report.append(f"避免追高: {'是' if market.get('avoid_chasing') else '否'}")
        report.append('')

        candidates = getattr(self, 'last_candidates', []) or []
        report.append('二、一阶候选池')
        if not candidates:
            report.append('当前没有满足条件的候选股。')
        else:
            for i, stock in enumerate(candidates, 1):
                report.append(f"{i}. {stock['name']} ({stock['code']})")
                report.append(f"   候选分: {stock.get('candidate_score', 0):.1f}")
                report.append(f"   趋势状态: {stock.get('trend_state', '-')}")
                report.append(
                    f"   分项: 流动性{stock.get('liquidity_score', 0):.1f} | 趋势{stock.get('trend_score', 0):.1f} | 资金{stock.get('funding_score', 0):.1f} | 技术{stock.get('technical_score', 0):.1f} | 风险扣分{stock.get('risk_penalty', 0):.1f}"
                )
                if stock.get('reasons'):
                    report.append(f"   入池原因: {'、'.join(stock['reasons'])}")
                if stock.get('risk_flags'):
                    report.append(f"   风险提示: {'、'.join(stock['risk_flags'])}")
                report.append('')

        report.append('三、最终推荐')
        if not stocks:
            report.append('当前市场环境下无满足条件的短线推荐。')
            return '\n'.join(report)

        for i, stock in enumerate(stocks, 1):
            report.append(f"{i}. {stock['name']} ({stock['code']})")
            report.append(f"   当前价: {stock.get('price', 0):.2f}  涨跌幅: {stock.get('change_pct', 0):+.2f}%")
            report.append(f"   推荐分: {stock.get('candidate_score', 0):.1f} | 评级: {stock.get('rating', '-')}")
            report.append(f"   趋势状态: {stock.get('trend_state', '-')}")
            report.append(
                f"   分项: 流动性{stock.get('liquidity_score', 0):.1f} | 趋势{stock.get('trend_score', 0):.1f} | 资金{stock.get('funding_score', 0):.1f} | 技术{stock.get('technical_score', 0):.1f} | 风险扣分{stock.get('risk_penalty', 0):.1f}"
            )
            if stock.get('reasons'):
                report.append(f"   推荐理由: {'、'.join(stock['reasons'])}")
            if stock.get('buy_signals'):
                report.append(f"   辅助信号: {'、'.join(stock['buy_signals'])}")
            if stock.get('risk_flags'):
                report.append(f"   风险提示: {'、'.join(stock['risk_flags'])}")
            report.append('')

        return '\n'.join(report)

    def close(self):
        self.cache.close()

    def save_selected_watchlist(self, stocks: List[Dict], filename: str) -> None:
        """将选股结果覆盖写入指定监控列表文件。"""
        watchlist_file = Path(__file__).resolve().parent / filename
        codes = [str(stock['code']).zfill(6) for stock in stocks if stock.get('code')]
        with open(watchlist_file, 'w', encoding='utf-8') as f:
            json.dump(codes, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    selector = ShortTermSelector(verbose=True)

    # 仅输出最终结果，避免大量过程日志导致终端卡顿
    top_stocks = selector.select_top_stocks(top_n=10, verbose=True)

    if top_stocks:
        for i, stock in enumerate(top_stocks, 1):
            print(f"{i}. {stock['code']} {stock['name']} | 评分 {stock['score']:.1f} | 现价 {stock['price']:.2f} | 涨跌 {stock['change_pct']:+.2f}%")

        report = selector.generate_report(top_stocks)
        date_suffix = datetime.now().strftime('%Y%m%d')
        recommend_dir = Path(__file__).resolve().parent / 'recommend'
        recommend_dir.mkdir(parents=True, exist_ok=True)
        report_file = recommend_dir / f'short_term_recommendation_{date_suffix}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        selector.save_selected_watchlist(top_stocks, 'watchlist_short_term.json')
        try:
            send_email_with_attachment(
                subject=f'短线选股报告 {date_suffix}',
                body='短线选股报告见附件。',
                attachment_path=report_file,
            )
            print("已发送短线报告到QQ邮箱")
        except Exception as e:
            print(f"短线报告邮件发送失败: {e}")

    selector.close()
