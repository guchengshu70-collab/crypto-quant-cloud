"""quant_research/signals_violent.py
MTF-TAMR · 第二套信号系统「暴力拉升版 / 暴跌版」——**启动前**预警（与 exp3 并列，互不覆盖）。

■ 关键语义（务必与"突破追涨"区分）：
  本系统抓的是 **大阳线 / 大阴线出现【之前】** 的那一刻，不是启动之后。
  - 突破型（错误语义）：等大实体 K 线走出来、收盘突破区间 → 才发信号。
    此时最肥的一段已经走完，属于"拉升以后"。
  - 本系统（正确语义）：行情 **仍被压在区间内、大实体尚未出现** 时提前埋伏。

■ 条件来自实测诊断（quant_research 内 25 币 × 5 级别特征分层，非拍脑袋）：
  基准「启动命中率」14.6%。逐特征分层后：
    - 【无效】相对历史压缩分位 sqp、窗口极差/ATR —— 单纯"缩量横盘"几乎无预测力，已弃用。
    - 【最强】距 EMA20 的距离（带方向）：>=3ATR 时命中 29.6%，近均线处仅 13.2%。
    - 【有效】收敛形态 conv（后半段极差 <= 前半段）、贴边蓄力、方向倾斜、缩量。
  组合后启动命中率 14.6% → 36~37%（做多 41% / 做空 27%），信号稀缺度 0.2%。
  经济含义：真正的"启动前"不是缩量横盘，而是
  **价格已离开均线一段距离、在局部区间里收敛并顶着蓄力方向边界**——旗形平台上沿。

■ 已知权衡（写在代码里，避免后人误读）：
  固定止盈下，"胜率 70%" 与 "抓大行情" 数学上冲突（实测）：
    0.5R → 胜率 70.9% 但期望 -0.004R（高胜率陷阱）；3.0R → 期望 +0.257R 但胜率 44.3%。
  因此默认走 **分批止盈**(exit_mode="partial")：首目标 0.5R 落袋锁定高胜率，
  余仓保本 + 远轨跟踪抓启动段。胜率口径与期望由 bt_violent.py 双口径如实报告。

■ 工程约定（与 exp3 同构，便于接入层与回测复用）：
  - 单一可信接口 `scan_level(bars, bias_bars, level_cfg, params, market, live)`，
    纯函数、无 I/O、无未来函数。setup 字段与 exp3 一致，额外附 `atr` 供展示层使用。
  - 导出 `PARAMS / LEVELS_CRYPTO / LIQ_STRICT / VOL_PROXY_MAX / median_quote_volume`。
  - 仅加密（crypto）。复用更高周期原生 K 线作偏置。

■ 纪律（与 exp3 一致，不可破坏）：
  - 全部基于已收盘 bar，无任何未来函数。
  - 成本/滑点/风控由 engine 统一处理；本模块只产出 setup。
  - 不修改 signals.py / exp3 任何核心逻辑 —— 这是并列第二套信号系统。
"""

import numpy as np

from indicators import atr, ema, sma
from filters import passes_entry_filters

# ---- 「启动前」参数（初版；最终由 bt_violent.py 在 DEV 上分阶段单变量扫描选参、冻结） ----
PARAMS = {
    # —— 方向 ——
    "bias_gate": True,          # 必须 HTF 偏置与本方向一致（exp3 已证最强胜率杠杆）

    # —— 窗口 ——
    "coil_bars": 12,            # 蓄势窗口根数（含当前K线）

    # —— ① 动量位置（诊断最强单变量）——
    "ema_d_min": 3.0,           # 收盘沿蓄力方向偏离 EMA20 >= 该 ATR 倍数
    "ema_d_max": 8.0,           # 上限，排除极端乖离（追顶/追底）

    # —— ② 收敛形态 ——
    "conv_max": 0.90,           # 后半段极差 / 前半段极差 <= 该值（<1 = 越走越窄）

    # —— ③ 蓄力方向 ——
    "cpos_min": 0.50,           # 收盘位于「蓄力方向」一侧（按方向归一后 >= 该值）
    "tilt_min": 0.05,           # 方向倾斜强度下限（后半段重心位移 / 窗口极差）

    # —— ④ 贴边蓄力（诊断：在高 ema_d 子集上收紧 edge 反而变差，默认关闭）——
    "edge_max": 9.0,            # 距「蓄力方向」边界 <= 该 ATR 倍数；9.0 = 不启用

    # —— ⑤ 量能（诊断：高 ema_d 时量能已天然萎缩，再卡缩量样本近乎归零，默认关闭）——
    "vol_shrink": 9.0,          # 窗口内量能中位数 <= 长期中位数 * 该值；9.0 = 不启用

    # —— ⑥ 尚未启动（硬性否定，区别于"突破追涨"）——
    "not_launched_body": 1.60,  # 当前K线实体 < 该倍数*ATR → 大阳/大阴还没出现
    "no_launch_lookback": 6,    # 回溯窗口：最近 N 根内也不得出现过启动大实体
    "no_launch_body": 2.00,     # 该窗口内的"已启动"实体阈值 *ATR
    "in_coil": True,            # 收盘必须仍在区间内（未突破 = 启动前，而非启动中）

    # —— 出场 ——
    "sl_buf": 0.25,             # 止损置于区间边界外的缓冲（*ATR）
    "sl_min_atr": 1.0,          # 止损最小距离（*ATR），避免过紧被噪音扫
    "sl_max_atr": 2.0,          # 止损最大距离（*ATR），控制单笔风险
    "tp_r": 2.0,                # 首目标 = tp_r * risk（DEV 选参 / HOLDOUT 验证后的冻结值）
                                # 注意：tp_r=0.5 是唯一能把胜率推到 ~74% 的档位，但期望为负（见模块 docstring）

    # —— 通用过滤 ——
    "atr_period": 14,
    "ema_long": 100,            # 偏置深层趋势上下文（与 exp3 同语义：price>EMA100）
    "vol_break": 0.20,          # 波动率熔断
    "sep_thr": 0.004,           # 均线分离度门槛（拒绝均线纠缠）
    "cooldown": None,           # 同标的同向冷却根数

    # —— 启动命中校验（非交易结果，仅用于评估"预警是否命中真启动"）——
    "launch_check_bars": 8,     # 信号后 N 根内检查是否出现真启动
    "launch_body_mult": 1.50,   # 判定"真启动"的实体阈值 *ATR

    "flags": {"respect": False, "vol_confirm": False},
}

# 与 exp3 同构的级别定义（名称, 交易分钟, 偏置分钟, 最长持有根数）
# 启动前埋伏需要给行情留出启动时间，持仓窗口比 exp3 略长。
LEVELS_CRYPTO = [
    ("1h", 60, 240, 36),
    ("4h", 240, 720, 30),
    ("6h", 360, 720, 26),
    ("12h", 720, 1440, 22),
    ("1w", 10080, 43200, 12),
]
LEVELS = LEVELS_CRYPTO

# 流动性/波动率门槛（与 exp3 同源语义，单一来源）
LIQ_STRICT = 4e5
VOL_PROXY_MAX = 0.06


def _bias_direction(bias_bars, params=PARAMS):
    """趋势偏置：SMA20>SMA50 且 close>EMA100（与 exp3 同语义）。无未来函数。"""
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


def median_quote_volume(bars):
    """资产级美元(USDT)流动性：median(v * c)。无未来函数。"""
    if not bars:
        return 0.0
    qv = np.array([b.get("v", 0.0) * b.get("c", 0.0) for b in bars], dtype=float)
    return float(np.median(qv)) if len(qv) else 0.0


def check_launch(bars, setups, params=PARAMS):
    """启动命中校验（非交易结果）：信号后 launch_check_bars 根内，
    是否真的出现了方向一致的大实体 K 线（|c-o| >= launch_body_mult * ATR）。

    这是评估「启动前预警」是否名副其实的关键指标：
    - 命中 = 预警之后确实来了大阳/大阴（抓的是启动前）；
    - 未命中 = 预警之后行情没启动（预警落空，靠小止损离场）。
    返回 (命中数, 总数, 命中率%)。
    """
    if not setups:
        return 0, 0, 0.0
    high = np.array([b["h"] for b in bars], dtype=float)
    low = np.array([b["l"] for b in bars], dtype=float)
    close = np.array([b["c"] for b in bars], dtype=float)
    openp = np.array([b["o"] for b in bars], dtype=float)
    a = atr(high, low, close, int(params["atr_period"]))
    n = len(bars)
    w = int(params["launch_check_bars"])
    thr = float(params["launch_body_mult"])
    hit = 0
    for s in setups:
        i = int(s["idx"])
        d = int(s["dir"])
        a_i = float(s.get("atr") or (a[i] if i < n else 0.0))
        if a_i <= 0 or np.isnan(a_i):
            continue
        j0, j1 = i + 1, min(i + 1 + w, n)
        if j0 >= j1:
            continue
        body = np.abs(close[j0:j1] - openp[j0:j1])
        brk = body >= thr * a_i
        if d > 0:
            if bool(np.any(brk & (close[j0:j1] > openp[j0:j1]))):
                hit += 1
        else:
            if bool(np.any(brk & (close[j0:j1] < openp[j0:j1]))):
                hit += 1
    return hit, len(setups), (100.0 * hit / len(setups)) if setups else 0.0


def scan_level(bars, bias_bars, level_cfg, params=PARAMS, market="crypto", live=False):
    """扫描单个交易级别，捕获「大阳/大阴启动【前】」的埋伏信号。

    判定链（全部基于已收盘 bar，无未来函数）：
      偏置门控 → 通用过滤 → ⑥尚未启动(硬否定) → ①动量位置(距EMA20带方向)
      → ②收敛形态 → ③蓄力方向(倾斜+归一收盘位) → ④贴边 → ⑤缩量
    满足则在【当前收盘】埋伏：止损贴区间边界外，止盈按 tp_r 倍风险。

    bars=交易级K线, bias_bars=更高周期原生K线(趋势方向)。
    纯函数：无 I/O、无未来函数。返回 setup 列表（字段与 exp3 一致）。
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

    ap = int(params["atr_period"])
    a = atr(high, low, close, ap)
    atr_pct = a / close
    medv = float(np.median(v)) if len(v) else 0.0

    e20 = ema(close, 20)
    s50 = sma(close, 50)

    bd = _bias_direction(bias_bars, params)
    has_bias = len(bd) > 0
    bt = np.array([b["t"] for b in bias_bars], dtype=np.int64) if has_bias else np.array([], dtype=np.int64)

    cooldown = params["cooldown"] or max(4, trade_limit // 6)
    last_entry = -999
    setups = []

    coil_bars = int(params["coil_bars"])
    ema_d_min = float(params["ema_d_min"])
    ema_d_max = float(params["ema_d_max"])
    conv_max = float(params["conv_max"])
    cpos_min = float(params["cpos_min"])
    tilt_min = float(params["tilt_min"])
    edge_max = float(params["edge_max"])
    vol_shrink = float(params["vol_shrink"])
    not_launched_body = float(params["not_launched_body"])
    no_launch_lookback = int(params["no_launch_lookback"])
    no_launch_body = float(params["no_launch_body"])
    in_coil = bool(params["in_coil"])
    sl_buf = float(params["sl_buf"])
    sl_min_atr = float(params["sl_min_atr"])
    sl_max_atr = float(params["sl_max_atr"])
    tp_r = float(params["tp_r"])
    vol_break = float(params["vol_break"])
    sep_thr = float(params["sep_thr"])

    body_all = np.abs(close - openp)

    scan_end = n if live else n - trade_limit

    for i in range(warmup, scan_end):
        if i - last_entry < cooldown:
            continue
        if np.isnan(e20[i]) or np.isnan(a[i]) or a[i] < 1e-9:
            continue
        if np.isnan(s50[i]):
            continue
        # 均线分离度（拒绝纠缠态）
        sep = abs(e20[i] - s50[i]) / close[i]
        if sep < sep_thr:
            continue
        # 波动率熔断
        if (not np.isnan(atr_pct[i])) and atr_pct[i] > vol_break:
            continue
        # 偏置门控（HTF 方向必须一致 —— 最强胜率杠杆）
        direction = 0
        if params["bias_gate"] and has_bias:
            j = int(np.searchsorted(bt, bars[i]["t"], side="right") - 1)
            if 0 <= j < len(bd):
                direction = int(bd[j])
            if direction == 0:
                continue

        # ---- ⑥ 尚未启动（先排除已启动，再谈方向）----
        if body_all[i] > not_launched_body * a[i]:
            continue
        bl0 = max(0, i - no_launch_lookback + 1)
        if float(np.max(body_all[bl0:i + 1])) > no_launch_body * a[i]:
            continue

        # ---- 蓄势窗口（含当前K线）----
        if i < coil_bars + 1:
            continue
        w0 = i - coil_bars + 1
        hi = float(np.max(high[w0:i + 1]))
        lo = float(np.min(low[w0:i + 1]))
        coil_range = hi - lo
        if coil_range <= 1e-12:
            continue
        # 收盘仍在区间内 = 尚未突破 = 启动前（而非启动中）
        if in_coil and (close[i] > hi or close[i] < lo):
            continue

        # ---- ① 动量位置：沿蓄力方向偏离 EMA20（诊断最强单变量）----
        ema_d = (close[i] - e20[i]) / a[i] * direction
        if ema_d < ema_d_min or ema_d > ema_d_max:
            continue

        # ---- ② 收敛形态：后半段极差 <= 前半段 * conv_max ----
        half = coil_bars // 2
        f_rng = float(np.max(high[w0:w0 + half]) - np.min(low[w0:w0 + half]))
        s_rng = float(np.max(high[w0 + half:i + 1]) - np.min(low[w0 + half:i + 1]))
        if f_rng > 1e-12 and (s_rng / f_rng) > conv_max:
            continue

        # ---- ③ 蓄力方向：倾斜 + 归一收盘位 ----
        if direction > 0:
            tilt = (float(np.mean(low[w0 + half:i + 1])) - float(np.mean(low[w0:w0 + half]))) / coil_range
            cpos = (close[i] - lo) / coil_range
            edge = (hi - close[i]) / a[i]
        else:
            tilt = (float(np.mean(high[w0:w0 + half])) - float(np.mean(high[w0 + half:i + 1]))) / coil_range
            cpos = (hi - close[i]) / coil_range
            edge = (close[i] - lo) / a[i]
        if tilt < tilt_min:
            continue
        if cpos < cpos_min:
            continue

        # ---- ④ 贴边蓄力 ----
        if edge > edge_max:
            continue

        # ---- ⑤ 缩量蓄势 ----
        if medv > 0 and float(np.median(v[w0:i + 1])) > medv * vol_shrink:
            continue

        # ---- 埋伏：当前收盘入场，止损贴区间边界外 ----
        entry = float(close[i])
        if direction > 0:
            raw_sl = lo - sl_buf * a[i]
            risk = entry - raw_sl
        else:
            raw_sl = hi + sl_buf * a[i]
            risk = raw_sl - entry
        if risk <= 1e-12:
            continue
        risk = float(min(max(risk, sl_min_atr * a[i]), sl_max_atr * a[i]))
        if direction > 0:
            sl = entry - risk
            tp = entry + tp_r * risk
        else:
            sl = entry + risk
            tp = entry - tp_r * risk
        if not ((direction > 0 and sl < entry < tp) or (direction < 0 and tp < entry < sl)):
            continue

        setups.append({
            "idx": i, "dir": direction, "entry": entry,
            "tp": tp, "sl": sl, "level": name, "atr": float(a[i]),
            "risk": risk, "reward": tp_r * risk, "trade_limit": trade_limit,
        })
        last_entry = i
    return setups
