"""quant_research/signals.py
MTF-TAMR · 最终 exp3 架构（B 扫描家族）—— 数据信号底座（唯一可信信号源）。

本文件是 exp3 交叉 AB 组验证后锁定的**最终架构**（B-bfam 与 B-aparam 均 100% 轮正、
DEV 与 HOLDOUT 双集一致正、两套独立代码收敛到同一结论，通过跨组对抗验证）：

  - 偏置（大周期定方向）：SMA20/SMA50 斜率 + 额外要求 price>EMA100（更深的趋势上下文）。
  - 触发（交易周期抓均值回归）：当根为「阳线」(close>=open) 且价格位于布林中轨下方
    （温和超卖），RSI 阈值更低(45)、回调带更宽(0.2~2.5)、止损乘数更大(2.2)。
  - Regime：拒绝均线纠缠（分离度过低）与极端波动，不使用 ADX。
  - 核心：趋势对齐 + 均值回归至 EMA20 止盈（fixed 退出）。
  - 退出：固定 TP=EMA20，SL=±sl_mult×ATR（**无 trailing / 无 partial**——exp2 已证
    partial 有害，bfam 宽参数 + 严流动才是正期望来源）。
  - 流动性门槛：严流动 5e6（扫描层据此过滤低流动性标的——exp3b 证明这是 B 组主因）。

纪律（不可破坏）：
  - 全部基于已收盘 Bar，无任何未来函数。
  - 成本/滑点/风控由 engine 统一处理；本模块只产出 setup：entry / tp / sl / level / risk / reward。
  - **仅加密（crypto）**。A 股已被持久排除（见项目 MEMORY），本底座不产出任何 A 股信号。

选参说明：PARAMS 为 exp3 验证通过的 B 参数家族（bfam 宽参数），选参后已冻结，
非对回测集调参；逻辑结构（SMA 偏置 + EMA100 + 布林中轨 + 宽带）为通用设计，非针对单品种。
"""

import numpy as np

from indicators import atr, bollinger, ema, rsi, sma
from filters import passes_entry_filters

# ---- 最终 exp3 架构参数（B 参数家族 / bfam 宽参数；选参后冻结） ----
PARAMS = {
    "rsi_long": 45.0,        # 多头触发：RSI(14) <= 该值（温和超卖，比 A 的 50 更低）
    "rsi_short": 55.0,       # 空头触发：RSI(14) >= 该值
    "min_dip": 0.2,          # 最小回调深度（以 ATR 计），过滤噪声
    "max_dip": 2.5,          # 最大回调深度（以 ATR 计），宽带来自 bfam
    "sl_mult": 2.2,          # 止损 = entry ± sl_mult*ATR（宽止损，bfam 来源）
    "ema_long": 100,         # 偏置要求的更深层趋势上下文（price>EMA100）
    "bb_k": 2.0,             # 布林带宽系数（中轨触发）
    "crash_atr_mult": 3.5,   # 崩盘识别：近 3 根跌幅 > crash_atr_mult×ATR 则跳过
    "vol_break": 0.15,       # 波动率熔断：ATR% > 该值跳过（crypto）
    "sep_thr": 0.004,        # 均线分离度门槛：过低（纠缠）跳过
    "cooldown": None,        # 同标的同向冷却根数（按级别设定）
    # 入场质量筛选（无未来函数；均不降低盈亏比）：回调确认 + 量能确认
    "flags": {"respect": True, "vol_confirm": True},
}

# 级别定义：(名称, 交易分钟, 偏置分钟=更高周期K线, 最长持有根数)
# 加密货币：偏置用原生更高周期（含月线 43200=1M）。
LEVELS_CRYPTO = [
    ("1h", 60, 240, 36),
    ("4h", 240, 720, 30),
    ("6h", 360, 720, 26),
    ("12h", 720, 1440, 22),
    ("1w", 10080, 43200, 12),
]
# 注：A 股（LEVELS_ASH / market="ashare"）已被持久排除，本底座不再包含。
LEVELS = LEVELS_CRYPTO

# 严流动门槛（最终 exp3 架构主因）：扫描层据此过滤低美元流动性标的。
# 单位 = 美元(USDT)成交额（median 1h v*c，USDT≈USD）；不再用基础币量。
# 用户拍板（2026-08-10）：5e6 USD/1h-bar 在全市场仅 BTC/ETH/SOL 3 币通过、无法拆 dev/holdout，
# 故放宽至 4e5 USD/1h-bar（≈$10M/日，约 35 个流动性蓝筹 alt），保留"只做高流动币"精神且可验证。
LIQ_STRICT = 4e5
# 波动率代理上限（资产级波动屏，见 filters.vol_proxy）：高于此值跳过（crypto）。
VOL_PROXY_MAX = 0.06


def _bias_direction(bias_bars, params=PARAMS):
    """趋势偏置：SMA20/SMA50 斜率 + price>EMA100（更深的趋势上下文）。

    与 A 扫描（纯 EMA 斜率）的真实对抗分歧：B 要求价格站在 EMA100 之上才判多头偏置，
    过滤掉"均线向上但价格已大幅偏离"的脆弱结构。无未来函数。
    """
    if len(bias_bars) < 60:
        return np.array([])
    c = np.array([b["c"] for b in bias_bars], dtype=float)
    s20 = sma(c, 20)
    s50 = sma(c, 50)
    e100 = ema(c, params["ema_long"])
    d = np.zeros(len(bias_bars))
    d[(s20 > s50) & (c > e100)] = 1
    d[(s20 < s50) & (c < e100)] = -1
    return d


def scan_level(bars, bias_bars, level_cfg, params=PARAMS, market="crypto", live=False):
    """扫描单个交易级别。bars=交易级K线, bias_bars=更高周期原生K线(提供趋势方向)。

    market 仅接受 "crypto"（A 股已被排除）；若传入其他值，波动率/分离度缩放不再生效。
    纯函数：无 I/O、无未来函数。返回 setup 列表（entry/tp/sl/level/risk/reward/trade_limit）。

    live 参数（2026-08-13 用户拍板，方案B）：
    - live=False（默认，回测/验证路径）：扫描窗口截至 n−trade_limit，预留前向
      评估窗口——行为与 exp3 锁定版本逐字节一致，历史统计完全不受影响。
    - live=True（仅实盘 app 接入层 mtf_tamr 传入）：窗口延伸到最新已收盘 bar。
      入场条件数学零改动（偏置/EMA20回调/布林中轨/RSI/回调带/过滤全部相同，
      且每根 bar 仍只用 ≤ 该 bar 的数据，无未来函数）；仅让实盘信号与回测假设
      （信号触发时刻入场）对齐，避免展示数十根 bar 前的陈旧入场参数。
    """
    name, tf_min, bias_min, trade_limit = level_cfg
    n = len(bars)
    warmup = 60
    if n < warmup + trade_limit + 5:
        return []

    close = np.array([b["c"] for b in bars], dtype=float)
    high = np.array([b["h"] for b in bars], dtype=float)
    low = np.array([b["l"] for b in bars], dtype=float)
    openp = np.array([b["o"] for b in bars], dtype=float)
    v = np.array([b["v"] for b in bars], dtype=float)
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e100 = ema(close, params["ema_long"])
    r = rsi(close, 14)
    a = atr(high, low, close, 14)
    upper, mid, lower = bollinger(close, 20, params["bb_k"])
    sep = np.abs(ema(close, 20) - sma(close, 50)) / close
    atr_pct = a / close
    medv = float(np.median(v)) if len(v) else 0.0

    bd = _bias_direction(bias_bars, params)
    has_bias = len(bd) > 0
    bt = np.array([b["t"] for b in bias_bars], dtype=np.int64) if has_bias else np.array([], dtype=np.int64)

    cooldown = params["cooldown"] or max(4, trade_limit // 6)
    last_entry = -999
    setups = []

    crash_mult = params["crash_atr_mult"]
    vol_break = params["vol_break"]           # crypto-only：不再对 ashare 缩放
    sep_thr = params["sep_thr"]

    # live 模式扫描到最新已收盘 bar；回测模式截至 n−trade_limit（预留前向评估窗口）
    scan_end = n if live else n - trade_limit
    for i in range(warmup, scan_end):
        if i - last_entry < cooldown:
            continue
        if np.isnan(e20[i]) or np.isnan(r[i]) or np.isnan(a[i]) or a[i] < 1e-9:
            continue
        if np.isnan(mid[i]) or np.isnan(sep[i]):
            continue
        # Regime：拒绝均线纠缠（分离度过低）
        if sep[i] < sep_thr:
            continue
        # 波动率熔断
        if (not np.isnan(atr_pct[i])) and atr_pct[i] > vol_break:
            continue
        # 崩盘识别
        if i >= 3 and a[i] > 1e-9:
            drop3 = (close[i - 3] - close[i]) / a[i]
            if drop3 > crash_mult:
                continue
        direction = 0
        if has_bias:
            j = int(np.searchsorted(bt, bars[i]["t"], side="right") - 1)
            if 0 <= j < len(bd):
                direction = int(bd[j])
        if direction == 0:
            continue
        if i < 1:
            continue
        # 入场质量筛选（无未来函数）
        if not passes_entry_filters(i, direction, close, v, e20, e50, medv, r, params["flags"]):
            continue
        if direction > 0:
            # 阳线 + EMA20 回调 + 布林中轨下方（温和超卖）
            if not (close[i] >= openp[i]):
                continue
            if not (close[i] < e20[i]):
                continue
            if not (close[i] <= mid[i]):
                continue
            dip = (e20[i] - close[i]) / a[i]
            if not (params["min_dip"] <= dip <= params["max_dip"]):
                continue
            if r[i] > params["rsi_long"]:
                continue
            entry = close[i]
            tp = e20[i]
            sl = entry - params["sl_mult"] * a[i]
            if not (sl < entry < tp):
                continue
        else:
            if not (close[i] <= openp[i]):
                continue
            if not (close[i] > e20[i]):
                continue
            if not (close[i] >= mid[i]):
                continue
            dip = (close[i] - e20[i]) / a[i]
            if not (params["min_dip"] <= dip <= params["max_dip"]):
                continue
            if r[i] < params["rsi_short"]:
                continue
            entry = close[i]
            tp = e20[i]
            sl = entry + params["sl_mult"] * a[i]
            if not (tp < entry < sl):
                continue
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk < 1e-9:
            continue
        setups.append({
            "idx": i, "dir": direction, "entry": entry,
            "tp": tp, "sl": sl, "level": name,
            "risk": risk, "reward": reward, "trade_limit": trade_limit,
        })
        last_entry = i
    return setups


def median_volume(bars):
    """资产级成交量中位数（基础币量；仅归档兼容，新底座请用 median_quote_volume）。"""
    if not bars:
        return 0.0
    v = np.array([b.get("v", 0.0) for b in bars], dtype=float)
    return float(np.median(v)) if len(v) else 0.0


def median_quote_volume(bars):
    """资产级**美元(USDT)流动性**：median(v * c)。

    所有交易对均为 USDT 报价（USDT≈USD），故 基础币量×价格 = 美元成交额代理。
    这是正确锚定美元的流动性度量——修正了早期用"基础币量"导致低价币误放行、
    真流动性币(BTC/ETH)反被跳过的反向过滤问题。无未来函数。
    """
    if not bars:
        return 0.0
    qv = np.array([b.get("v", 0.0) * b.get("c", 0.0) for b in bars], dtype=float)
    return float(np.median(qv)) if len(qv) else 0.0
