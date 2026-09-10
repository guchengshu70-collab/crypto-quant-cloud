"""quant_research/filters.py
无未来函数的入场质量筛选（供 A/B 两变体共用，可组合）。

设计原则：仅使用信号 Bar i 及之前的信息（i-1、i-2、近 N 根），不使用 i 之后任何数据，
杜绝未来函数。所有筛选均为「提升入场质量」的硬规则，不是为抬胜率而针对样本调参。
"""

import numpy as np


def passes_entry_filters(i, direction, c, v, e20, e50, medv, r, flags):
    """返回 bool：该 setup 是否通过全部启用的筛选。

    flags 可含键：respect / stable_bias / rsi_window / vol_confirm / mom2。
    c,v,e20,e50,r 为该交易级 Bar 的 numpy 数组；medv 为成交量中位数。
    """
    if flags.get("respect"):
        # 近 10 根内曾价格高于 EMA20 → 确认为「回调」而非「破位下跌」
        lo = max(0, i - 10)
        if not any(c[j] > e20[j] for j in range(lo, i)):
            return False
    if flags.get("stable_bias"):
        # 偏置需稳定运行：近 5 根 EMA20/EMA50 关系与方向一致
        lo = max(0, i - 5)
        if direction > 0 and not all(e20[j] > e50[j] for j in range(lo, i)):
            return False
        if direction < 0 and not all(e20[j] < e50[j] for j in range(lo, i)):
            return False
    if flags.get("rsi_window"):
        if not (35.0 <= r[i] <= 50.0):
            return False
    if flags.get("vol_confirm"):
        # 反转 Bar 成交量需达中位量的 80% 以上（有资金确认，非缩量假反弹）
        if v[i] < medv * 0.8:
            return False
    if flags.get("mom2"):
        # 动量拐头：相对 2 根前已转向（多：c[i]>c[i-2]；空反之）
        if direction > 0 and not (c[i] > c[i - 2]):
            return False
        if direction < 0 and not (c[i] < c[i - 2]):
            return False
    return True


def vol_proxy(bars):
    """波动率代理：60min/1h 收盘价逐根收益率绝对值的中位数（无需完整 ATR，快速）。"""
    c = np.array([b["c"] for b in bars], dtype=float)
    if len(c) < 3:
        return 0.0
    ret = np.abs(np.diff(c) / c[:-1])
    return float(np.median(ret))
