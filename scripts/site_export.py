#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将选股结果导出为静态站点（GitHub Pages）使用的 JSON。

在 long_term_selector.py / short_term_selector.py 的 __main__ 里调用，
产物写入 scripts/site/data/ 下，供 site/index.html 前端读取渲染。
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

SITE_DIR = Path(__file__).resolve().parent / 'site'
DATA_DIR = SITE_DIR / 'data'

# 中长线：从结果字典中挑选前端展示字段
LONG_FIELDS = [
    'code', 'name', 'price', 'change_pct', 'score', 'rating', 'stock_type',
    'signals', 'buy_signals', 'buy_price', 'stop_loss', 'take_profit',
    'stop_loss_pct', 'take_profit_pct', 'risk_reward_ratio',
    'max_falsification_point', 'recommend',
]

# 短线：从结果字典中挑选前端展示字段
SHORT_FIELDS = [
    'code', 'name', 'price', 'change_pct', 'score', 'candidate_score', 'rating',
    'trend_state', 'liquidity_score', 'trend_score', 'funding_score',
    'technical_score', 'risk_penalty', 'sector_score', 'industry',
    'primary_theme', 'leader_tier', 'reasons', 'buy_signals', 'risk_flags',
    'recommend',
]


def _clean(value: Any) -> Any:
    """把 numpy 标量转 python 原生，NaN/Inf 转 None，保证可 JSON 序列化。"""
    try:
        import numpy as np
        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _pick(stock: Dict, fields: List[str]) -> Dict:
    return {key: _clean(stock.get(key)) for key in fields}


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(filename: str, payload: Dict) -> None:
    _ensure_dir()
    with open(DATA_DIR / filename, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _update_meta(**kwargs: Any) -> None:
    """维护一个汇总 meta.json，记录最近生成时间与各榜单数量。"""
    _ensure_dir()
    meta_path = DATA_DIR / 'meta.json'
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            meta = {}
    meta.update(kwargs)
    meta['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def write_long_term(selection: Dict) -> None:
    """导出中长线结果。selection 为 select_top_stocks() 返回的分层结构。"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    payload = {
        'generated_at': now,
        'core': [_pick(s, LONG_FIELDS) for s in (selection.get('core') or [])],
        'watch': [_pick(s, LONG_FIELDS) for s in (selection.get('watch') or [])],
        'excluded': [
            _pick(s, LONG_FIELDS)
            for s in (selection.get('excluded_but_interesting') or [])
        ],
    }
    _write_json('long_term.json', payload)
    _update_meta(long_term_generated_at=now, long_term_count=len(payload['core']))


_MARKET_FIELDS = ['sentiment_score', 'regime', 'up_ratio', 'limit_up', 'limit_down']


def write_short_term(stocks, context=None, note=None, abstain=False) -> None:
    """导出短线结果。

    stocks   : 要展示的股票列表（正常为最终推荐；空仓日为评分最高的参考股）。
    context  : 市场环境字典（用于展示情绪分/涨跌停等）。
    note     : 提示语（空仓日说明为何建议观望）。
    abstain  : True 表示今日建议空仓、下列股票仅供参考、非买入推荐。
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    market = {}
    if context:
        market = {k: _clean(context.get(k)) for k in _MARKET_FIELDS}
    payload = {
        'generated_at': now,
        'stocks': [_pick(s, SHORT_FIELDS) for s in (stocks or [])],
        'abstain': bool(abstain),
        'note': note,
        'market': market,
    }
    _write_json('short_term.json', payload)
    _update_meta(short_term_generated_at=now, short_term_count=len(payload['stocks']))
