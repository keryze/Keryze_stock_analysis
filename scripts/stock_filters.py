#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票风险名称过滤。

用于在选股股票池中剔除 ST / *ST / 退市股。
判定完全基于股票名称（Tushare/东财名称会带 "ST"、"*ST"、"退" 等标记），
是全市场通用规则，短线与中长线选股共用同一套逻辑。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

# 名称中出现以下关键字即视为高风险，予以剔除：
#   'ST'  -> 覆盖 ST 与 *ST（*ST 名称本身即包含 'ST' 子串）
#   '退'  -> 覆盖 "退市XX"、"XX退" 等退市/退市整理期个股
RISK_NAME_KEYWORDS: Tuple[str, ...] = ('ST', '退')


def is_risk_name(name: Optional[str]) -> bool:
    """名称是否命中 ST/退市风险关键字。名称缺失时返回 False（无法判定则不剔除）。"""
    if not name:
        return False
    text = str(name).upper()
    return any(keyword in text for keyword in RISK_NAME_KEYWORDS)


def filter_risk_codes(
    codes: Iterable[str],
    name_map: Dict[str, str],
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    从代码列表中剔除 ST/退市股。

    参数:
        codes: 待过滤的股票代码列表。
        name_map: code -> name 映射（通常来自 StockCache.get_all_names()）。
                  映射中不存在的代码视为名称未知，予以保留（宁可漏剔，不可误杀）。

    返回:
        (kept, removed):
            kept    -> 保留的代码列表，顺序与输入一致。
            removed -> 被剔除的 (code, name) 列表，便于日志与复核。
    """
    kept: List[str] = []
    removed: List[Tuple[str, str]] = []
    for code in codes:
        name = name_map.get(code)
        if name and is_risk_name(name):
            removed.append((code, name))
        else:
            kept.append(code)
    return kept, removed
