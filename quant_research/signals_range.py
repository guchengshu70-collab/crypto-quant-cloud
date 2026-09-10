"""quant_research/signals_range.py
第三套信号源 —— 震荡期（箱体）均值回归。

与既有两套的**分工边界**（互不覆盖、互不改动）：
  - signals.py        (exp3)    趋势内回调：HTF 偏置 + LTF 拉回 EMA20，止盈 EMA20。
  - signals_violent.py(violent) 暴力启动：缩波动蓄势 → 大实体突破，测量目标。
  - signals_range.py  (本模块)  震荡期：先证明"价格被关在箱子里、且暂时没有拉升/暴跌、
                                且成交量明显缩小（地量蓄力）"，再于箱体**双向限价挂单**：
                                底部 buy limit（吃下沿回归）+ sell stop（跌破抓暴跌），
                                顶部 sell limit（吃上沿回归）+ buy stop（涨破抓爆拉）。
                                止盈回箱体中枢 / 对称 ATR 距离；破边界即认错。

------------------------------------------------------------------------------
一、震荡期判定（regime）—— 本信号源的灵魂，全部只用「已收盘的过去数据」
------------------------------------------------------------------------------
在 bar i 处，用回溯窗口 [i-L+1 .. i]（L = regime_lookback，按级别取"前几十根"）计算：

  ① 效率比 ER（Kaufman）  = |C[i]-C[i-L]| / Σ|ΔC|
        → 震荡期分子小分母大，ER 趋近 0；趋势期 ER 趋近 1。最经典的震荡判据。
  ② 净漂移 drift_atr      = |C[i]-C[i-L]| / ATR
        → 一段路"净走了多远"。震荡期几乎原地踏步。
  ③ 箱体高度 box_atr      = (max(H)-min(L)) / ATR
        → 区间宽度。过宽说明早已走出趋势，过窄说明没有可赚的空间（双端卡口）。
  ④ 波动未放大 vol_expand = ATR[i] / ATR[i-L]
        → 排除"变盘前夜"（波动刚放大往往意味着箱体即将被打破）。
  ⑤ 暂时无拉升/暴跌 quiet = 最近 quiet_lb 根内单根实体 / ATR 的最大值
        → 用户明确要求：确认"暂时无拉升下跌"。刚出现大阳/大阴的不算震荡。
  ⑥ 箱体位置 pos          = (C[i]-box_low) / (box_high-box_low) ∈ [0,1]
        → 仅作诊断上下文；本信号源**不**按 pos 分边触发，而是箱体一旦确认即
          双侧布防（底部+顶部各两单），故不要求价格必须贴某一边。

**无未来函数声明**：以上全部只用下标 ≤ i 的已收盘 Bar；箱体高低、ATR、ER 均不含 i 之后
的任何信息。回测路径 scan_end = n - trade_limit，为每笔交易预留干净的前向评估窗口。

------------------------------------------------------------------------------
二、双向限价挂单（bracket）口径
------------------------------------------------------------------------------
  箱体确认（含"量能明显缩小"）即一次性挂出四张单，覆盖"震荡延续"与"变盘突破"两种结局：
    - 底部 buy limit  ：entry = 下沿 − limit_buf×ATR，TP = 箱体中枢（均值回归），
                         SL = entry − risk（跌破下沿 = 破位）。
    - 底部 sell stop   ：entry = 下沿 − bracket_buf×ATR，TP = entry − bracket_tp_r×risk（测 movable），
                         SL = entry + risk（假跌破回升即认错）。
    - 顶部 sell limit  ：entry = 上沿 + limit_buf×ATR，TP = 箱体中枢（均值回归），
                         SL = entry + risk（涨破上沿 = 破位）。
    - 顶部 buy stop    ：entry = 上沿 + bracket_buf×ATR，TP = entry + bracket_tp_r×risk（测 movable），
                         SL = entry − risk（假突破回落即认错）。
  risk = 箱外 buf×ATR，并夹在 [sl_min_atr, sl_max_atr]×ATR 之间，避免过紧被噪音扫、过宽亏爆。

------------------------------------------------------------------------------
三、纪律
------------------------------------------------------------------------------
  - 仅加密（crypto）。
  - 纯函数：无 I/O、无随机、无未来函数。
  - 本模块只产出 setup，成本/滑点/杠杆由回测引擎统一处理。
  - 参数由 DEV 集选出后在 HOLDOUT 冻结验证，禁止结果依赖调参。
"""

import numpy as np

from indicators import atr as _atr, rsi as _rsi

# ----------------------------------------------------------------------------
# 参数（默认值 = 研究起点；正式值由 DEV 扫描后冻结，见 bt_range_devhold.py）
# ----------------------------------------------------------------------------
PARAMS = {
    # —— 震荡期判定窗口 ——
    # None = 按级别自动取（LOOKBACK_BY_LEVEL：1h=72根≈3天、4h=42根≈7天、1d=30根≈30天）
    "regime_lookback": None,

    # —— ①②③ 阈值一律用「√L 归一化」后的尺度无关量 ——
    # 实测（DEV 15 币诊断）：随机游走下 ER≈1/√L、drift_atr≈0.67√L、box_atr≈c√L。
    # 归一化后 1h/4h/12h/1d/1w 的分布高度重合（er×√L: 0.85~1.12；drift/√L: 0.377~0.406；
    # box/√L: 1.022~1.078），因此**同一套阈值可跨周期通用**；
    # 若用未归一化的绝对阈值，1w 会被过度过滤（实测导致周线信号归零）。
    "er_n_max": 1.20,           # ER×√L <= 该值（≈1 为随机游走基准，越小越"原地打转"）
    "drift_n_max": 0.35,        # (|净位移|/ATR)/√L <= 该值
    "box_n_min": 0.75,          # (箱体/ATR)/√L >= 该值：太窄 = 没有可赚空间（噪音箱）
    "box_n_max": 1.60,          # <= 该值：太宽 = 其实已经是趋势

    # —— ④ 波动未放大 ——
    "vol_expand_max": 1.60,     # ATR[i]/ATR[i-L] <= 该值（排除变盘前夜）

    # —— ⑤ 暂时无拉升/暴跌 ——
    "quiet_lb": 5,              # 最近 N 根
    "quiet_body_atr": 2.20,     # 单根实体/ATR 上限（超过即认为"已经启动了"）

    # —— ⑥ 触发位置与 RSI ——
    # 做多触发区间 [pos_lo_lo, pos_lo_hi]；做空触发区间 [pos_hi_lo, pos_hi_hi]。
    # **为什么是区间而不是"越贴边越好"**：DEV 200 币实测 R 随 pos 单调上升——
    # 贴到边界（pos<0.1）反而最差（1h R=-0.31、胜率仅 25%），因为那意味着价格
    # 正在**冲击**边界、破位概率高；真正安全的是"已从边界回撤、箱体在起作用"的位置。
    "pos_lo_lo": 0.0,           # 做多位置下界
    "pos_lo_hi": 0.25,          # 做多位置上界
    "pos_hi_lo": 0.75,          # 做空位置下界
    "pos_hi_hi": 1.0,           # 做空位置上界
    "rsi_lo": 50.0,             # 做多：RSI <= 该值
    "rsi_hi": 50.0,             # 做空：RSI >= 该值
    "rsi_period": 14,

    # —— 出场 ——
    "tp_mode": "mid",           # "mid"=箱体中枢  |  "opposite"=对侧边界  |  "fixedR"=固定R
    "tp_r": 1.0,                # tp_mode="fixedR" 时的 R 倍数
    "sl_buf_atr": 0.50,         # 止损置于箱体边界外的缓冲（×ATR）
    "sl_min_atr": 1.00,         # 止损最小距离（×ATR）
    "sl_max_atr": 2.50,         # 止损最大距离（×ATR）

    # —— ② 量能明显缩小（用户条件2：爆拉/暴跌「之前」的蓄力 = 成交量压缩）——
    # 近期均量 / 基线均量；<= vol_shrink_max 才算"量能明显缩小"。
    # 与 ④ vol_expand（看 ATR 波动率）是**两个维度**：后者排除波动放大，
    # 前者排除成交活跃——二者叠加才是"低波动 + 地量"的真·压缩蓄力。
    "vol_shrink_max": 0.65,     # 近期均量 <= 65% × 基线均量 = 量能明显缩小
    "vol_recent_bars": 6,       # 近期窗口（根）
    "vol_baseline_mult": 2.0,   # 基线窗口 = vol_baseline_mult × L（更稳）

    # —— 双向限价挂单（bracket）几何 ——
    "limit_buf_atr": 0.25,      # 限价单置于箱体边界外该倍数×ATR（贴边吃回归）
    "bracket_buf_atr": 0.60,    # 突破单置于箱体边界外该倍数×ATR（破位才触发）
    "bracket_tp_r": 1.5,        # 突破单止盈 = bracket_tp_r × 该单风险

    # —— 通用 ——
    "atr_period": 14,
    "cooldown": None,           # 同标的冷却根数（None → 按级别取 trade_limit//3）
    "direction": "both",        # both / long / short（DEV 选参维度；1h 上多空不对称）
}

# 级别定义：(名称, 交易分钟, 最长持有根数)
# 说明：本套信号源不需要 bias 高周期（震荡期无趋势方向可言），故比 exp3 少一列，
#       但为保持接入层同构，LEVELS_CRYPTO 仍保留三元组。
LEVELS_CRYPTO = [
    ("1h", 60, 36),
    ("4h", 240, 30),
    ("12h", 720, 22),
    ("1d", 1440, 20),
    ("1w", 10080, 12),
]
LEVELS = LEVELS_CRYPTO

# 各级别的震荡判定回溯根数（"前几十根/几十小时/几十天"的级别化落点）
LOOKBACK_BY_LEVEL = {"1h": 72, "4h": 42, "12h": 36, "1d": 30, "1w": 26}

LIQ_STRICT = 4e5          # 与项目既定口径一致：1h 美元成交额中位数下限
VOL_PROXY_MAX = 0.06      # 资产级波动屏


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def _arrays(bars):
    """把 dict bar 列表转成 numpy 数组（t,o,h,l,c,v）。"""
    t = np.array([b["t"] for b in bars], dtype=np.int64)
    o = np.array([b["o"] for b in bars], dtype=float)
    h = np.array([b["h"] for b in bars], dtype=float)
    l = np.array([b["l"] for b in bars], dtype=float)
    c = np.array([b["c"] for b in bars], dtype=float)
    v = np.array([b["v"] for b in bars], dtype=float) if "v" in bars[0] else np.zeros(len(bars))
    return t, o, h, l, c, v


def _rolling_max(a, w):
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    # 前缀朴素滑动最大（w 通常 <= 72，n 可达 10^4；用 stride 视图更快）
    view = np.lib.stride_tricks.sliding_window_view(a, w)
    out[w - 1:] = view.max(axis=1)
    return out


def _rolling_min(a, w):
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    view = np.lib.stride_tricks.sliding_window_view(a, w)
    out[w - 1:] = view.min(axis=1)
    return out


def _rolling_sum(a, w):
    n = len(a)
    out = np.full(n, np.nan)
    if n < w:
        return out
    cs = np.concatenate(([0.0], np.cumsum(a)))
    out[w - 1:] = cs[w:] - cs[:-w]
    return out


def median_quote_volume(bars):
    """1h 美元成交额中位数（流动性口径，与项目一致）。"""
    if not bars:
        return 0.0
    v = np.array([b["v"] for b in bars], dtype=float)
    c = np.array([b["c"] for b in bars], dtype=float)
    return float(np.median(v * c))


def _vol_ratio_array(v, recent, base):
    """成交量压缩比 vol_ratio[i] = 近期均量 / 基线均量（向量化，无未来函数）。

    窗口约定：bar i 用 [i-w+1 .. i]（w = recent 或 base，均只含下标 <= i）。
    - recent：近期窗口（短，捕捉"正在缩量"）。
    - base  ：基线窗口（长，= vol_baseline_mult × L，代表常态成交）。
    vol_ratio <= vol_shrink_max 即"量能明显缩小"。
    无成交量数据（v 全 0 / 长度不足 base）→ 整段返回 NaN，调用方据此"不否决"（中立跳过）。
    """
    n = len(v)
    out = np.full(n, np.nan)
    if n < base or base <= 0:
        return out
    recent = max(1, min(int(recent), base))
    cs = np.concatenate(([0.0], np.cumsum(np.asarray(v, dtype=float))))
    base_sum = cs[base:] - cs[:-base]      # i = base-1 .. n-1
    rec_sum = cs[recent:] - cs[:-recent]   # i = recent-1 .. n-1
    start = base - 1
    idx = np.arange(start, n)
    bm = base_sum[idx - (base - 1)]
    rm = rec_sum[idx - (recent - 1)]
    eps = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        # 注意：base_sum / rec_sum 均为"窗口和"，需各自除以窗口长度才是均值比
        ratio = np.where(bm > eps, (rm / recent) / (bm / base), np.nan)
    out[idx] = ratio
    return out



def regime_flags(o, h, l, c, atr_arr, i, L, params):
    """在 bar i 处计算震荡期判定的全部特征。只用下标 <= i 的数据。

    返回 dict；若数据不足返回 None。
    """
    if i - L + 1 < 0:
        return None
    seg_c = c[i - L + 1:i + 1]
    seg_h = h[i - L + 1:i + 1]
    seg_l = l[i - L + 1:i + 1]
    seg_o = o[i - L + 1:i + 1]
    a_now = atr_arr[i]
    if not np.isfinite(a_now) or a_now <= 1e-12:
        return None

    box_high = float(seg_h.max())
    box_low = float(seg_l.min())
    box_rng = box_high - box_low
    if box_rng <= 1e-12:
        return None
    box_mid = (box_high + box_low) / 2.0

    net = abs(c[i] - c[i - L]) if i - L >= 0 else abs(c[i] - c[0])
    path = float(np.abs(np.diff(seg_c)).sum())
    er = net / path if path > 1e-12 else 0.0
    drift_atr = net / a_now
    box_atr = box_rng / a_now
    pos = (c[i] - box_low) / box_rng

    # 波动是否放大
    a_prev = atr_arr[i - L] if i - L >= 0 else np.nan
    vol_expand = (a_now / a_prev) if (np.isfinite(a_prev) and a_prev > 1e-12) else np.nan

    # 最近 quiet_lb 根内最大单根实体 / 当时 ATR（"暂时无拉升下跌"）
    qlb = int(params["quiet_lb"])
    q_start = max(0, i - qlb + 1)
    q_atr = atr_arr[q_start:i + 1]
    q_body = np.abs(seg_c[-(i - q_start + 1):] - seg_o[-(i - q_start + 1):])
    with np.errstate(divide="ignore", invalid="ignore"):
        q_ratio = np.where(q_atr > 1e-12, q_body / q_atr, np.nan)
    quiet_max = float(np.nanmax(q_ratio)) if q_ratio.size and np.isfinite(q_ratio).any() else np.nan

    sq = float(np.sqrt(max(1, int(L))))
    return {
        "box_high": box_high, "box_low": box_low, "box_mid": box_mid,
        "box_range": box_rng, "box_atr": box_atr,
        "er": float(er), "drift_atr": float(drift_atr),
        "er_n": float(er) * sq, "drift_n": float(drift_atr) / sq,
        "box_n": float(box_atr) / sq,
        "vol_expand": float(vol_expand) if np.isfinite(vol_expand) else np.nan,
        "quiet_max": quiet_max, "pos": float(pos),
    }


def is_ranging(rg, params):
    """震荡期硬判定：六项全过才算"确认处于震荡期且暂时无拉升下跌"。（归一化量）"""
    if rg is None:
        return False
    if rg["er_n"] > params["er_n_max"]:
        return False
    if rg["drift_n"] > params["drift_n_max"]:
        return False
    if not (params["box_n_min"] <= rg["box_n"] <= params["box_n_max"]):
        return False
    ve = rg["vol_expand"]
    if np.isfinite(ve) and ve > params["vol_expand_max"]:
        return False
    qm = rg["quiet_max"]
    if np.isfinite(qm) and qm > params["quiet_body_atr"]:
        return False
    return True


# ----------------------------------------------------------------------------
# 向量化特征面（与主扫描完全同构；regime_flags/is_ranging 为逐点参考实现）
# ----------------------------------------------------------------------------
def regime_surface(o, h, l, c, atr_arr, L, qlb):
    """一次性算出整条序列上的震荡期特征（向量化，无未来函数）。

    窗口约定与 regime_flags 完全一致：bar i 用 [i-L+1 .. i]。
    """
    n = len(c)
    nan = np.full(n, np.nan)

    box_high = _rolling_max(h, L)
    box_low = _rolling_min(l, L)
    box_rng = box_high - box_low
    box_mid = (box_high + box_low) / 2.0

    c_shift = np.full(n, np.nan)
    if n > L:
        c_shift[L:] = c[:-L]
    net = np.abs(c - c_shift)

    d = np.zeros(n)
    if n > 1:
        d[1:] = np.abs(np.diff(c))
    w = max(1, L - 1)
    path = _rolling_sum(d, w)

    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(path > 1e-12, net / path, 0.0)
        drift_atr = np.where(atr_arr > 1e-12, net / atr_arr, np.nan)
        box_atr = np.where(atr_arr > 1e-12, box_rng / atr_arr, np.nan)
        pos = np.where(box_rng > 1e-12, (c - box_low) / box_rng, np.nan)

        atr_shift = np.full(n, np.nan)
        if n > L:
            atr_shift[L:] = atr_arr[:-L]
        vol_expand = np.where(atr_shift > 1e-12, atr_arr / atr_shift, np.nan)

        body_ratio = np.where(atr_arr > 1e-12, np.abs(c - o) / atr_arr, np.nan)
    quiet_max = _rolling_max(np.nan_to_num(body_ratio, nan=0.0), max(1, int(qlb)))

    # √L 归一化（尺度无关，跨周期可比）
    sq = float(np.sqrt(max(1, int(L))))
    er_n = er * sq
    drift_n = drift_atr / sq
    box_n = box_atr / sq

    return {"box_high": box_high, "box_low": box_low, "box_mid": box_mid,
            "box_range": box_rng, "box_atr": box_atr, "er": er,
            "drift_atr": drift_atr, "vol_expand": vol_expand,
            "quiet_max": quiet_max, "pos": pos, "net": net, "path": path,
            "er_n": er_n, "drift_n": drift_n, "box_n": box_n, "sqrt_L": sq}


# ----------------------------------------------------------------------------
# 主扫描（向量化）
# ----------------------------------------------------------------------------
def scan_level(bars, level_cfg, params=PARAMS, market="crypto", live=False):
    """扫描单个级别的震荡期信号。

    bars       : 该级别的已收盘 K 线（list[dict]）
    level_cfg  : (name, tf_min, trade_limit)
    live=False : 回测路径，扫描窗口截至 n-trade_limit（预留前向评估窗口）
    live=True  : 实盘路径，扫描到最新已收盘 bar

    返回 setup 列表，每项含 entry/tp/sl/level/risk/box_*/er/pos/rsi 等诊断字段。
    """
    if market != "crypto":
        return []
    name, tf_min, trade_limit = level_cfg
    n = len(bars)
    if n < 200:
        return []

    # bars 支持 list[dict] 或紧凑 numpy [t,o,h,l,c(,v)]（回测热路径用后者，避免重复解析）
    if isinstance(bars, np.ndarray):
        arr = np.asarray(bars, dtype=float)
        t, o, h, l, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        v = arr[:, 5] if arr.shape[1] > 5 else np.zeros(len(arr))
    else:
        t, o, h, l, c, v = _arrays(bars)
    p = dict(params)
    _lb = p.get("regime_lookback", None)
    L = int(LOOKBACK_BY_LEVEL.get(name, 48)) if _lb is None else int(_lb)
    ap = int(p["atr_period"])
    atr_arr = _atr(h, l, c, ap)
    rsi_arr = _rsi(c, int(p["rsi_period"]))

    # 预热：ATR/RSI 递归 14 根即可收敛；箱体滚动窗口 NaN 会自然过滤更早的位置。
    # 取 60（周线目标仅 150 根，过高的预热会让周线扫不出任何信号）。
    warmup = max(60, L, ap + 15)
    if n < warmup + trade_limit + 5:
        return []

    cooldown = p["cooldown"] or max(4, trade_limit // 3)
    scan_end = n if live else n - trade_limit

    R = regime_surface(o, h, l, c, atr_arr, L, p["quiet_lb"])

    # —— 量能明显缩小（用户条件2：爆拉/暴跌「之前」的蓄力 = 成交量压缩）——
    # 近期均量 / 基线均量；NaN（无成交量数据）时不否决（中立跳过），只否决明确放大者。
    vol_ratio = _vol_ratio_array(v, p["vol_recent_bars"], int(round(p["vol_baseline_mult"] * L)))
    pos = R["pos"]

    ok = np.isfinite(atr_arr) & (atr_arr > 1e-12) & np.isfinite(rsi_arr) & np.isfinite(pos)
    ok &= (R["er_n"] <= p["er_n_max"])
    ok &= (R["drift_n"] <= p["drift_n_max"])
    ok &= (R["box_n"] >= p["box_n_min"]) & (R["box_n"] <= p["box_n_max"])
    # ④ 波动未放大 / ⑤ 近期无大实体：NaN（数据不足）不否决，只否决明确越界者
    ok &= ~(np.isfinite(R["vol_expand"]) & (R["vol_expand"] > p["vol_expand_max"]))
    ok &= ~(np.isfinite(R["quiet_max"]) & (R["quiet_max"] > p["quiet_body_atr"]))
    # ② 量能明显缩小：有成交量数据则要求近期均量 <= vol_shrink_max×基线；无数据不否决
    ok &= ~(np.isfinite(vol_ratio) & (vol_ratio > p["vol_shrink_max"]))

    # 箱体确认（含量能缩小）即触发一个**双向限价挂单** bracket（底部 + 顶部各两单），
    # 不再分 long/short 单边——用户要求"底部挂单 + 顶部挂单"两侧同时布防。
    idxs = np.nonzero(ok & (np.arange(n) >= warmup) & (np.arange(n) < scan_end))[0]

    setups = []
    last_idx = -10 ** 9
    buf = float(p["sl_buf_atr"])
    limit_buf = float(p["limit_buf_atr"])
    brk_buf = float(p["bracket_buf_atr"])
    tp_r = float(p["bracket_tp_r"])

    def _ord(kind, d, e, s, t):
        return {"kind": kind, "dir": int(d), "entry": float(e), "sl": float(s), "tp": float(t),
                "risk": float(abs(e - s)), "reward": float(abs(t - e)),
                "rr": float(abs(t - e) / abs(e - s)) if abs(e - s) > 1e-12 else None}

    for i in idxs:
        i = int(i)
        if i - last_idx < cooldown:
            continue
        a = float(atr_arr[i])
        # 单笔风险：箱外 buffer × ATR，并夹到 [sl_min, sl_max]×ATR（四类单共用，几何一致）
        risk = min(max(buf * a, p["sl_min_atr"] * a), p["sl_max_atr"] * a)
        if risk <= 1e-12:
            continue
        supp = float(R["box_low"][i])
        resist = float(R["box_high"][i])
        mid = float(R["box_mid"][i])

        # 底部：buy limit（贴下沿吃均值回归多）+ sell stop（跌破下沿抓暴跌）
        bl_entry = supp - limit_buf * a
        bl_sl = bl_entry - risk
        bl_tp = mid
        bs_entry = supp - brk_buf * a
        bs_sl = bs_entry + risk
        bs_tp = bs_entry - tp_r * risk
        # 顶部：sell limit（贴上沿吃均值回归空）+ buy stop（涨破上沿抓爆拉）
        sl_entry = resist + limit_buf * a
        sl_sl = sl_entry + risk
        sl_tp = mid
        tb_entry = resist + brk_buf * a
        tb_sl = tb_entry - risk
        tb_tp = tb_entry + tp_r * risk

        # 几何有效性：每个方向必须"止盈比入场更有利、止损在错误侧"
        if not (bl_tp > bl_entry > bl_sl):
            continue
        if not (sl_tp < sl_entry < sl_sl):
            continue
        if not (bs_tp < bs_entry < bs_sl):
            continue
        if not (tb_tp > tb_entry > tb_sl):
            continue

        bracket = {
            "box_low": supp, "box_high": resist, "box_mid": mid, "atr": a,
            "bottom": {
                "buy_limit": _ord("buy_limit", 1, bl_entry, bl_sl, bl_tp),
                "sell_stop": _ord("sell_stop", -1, bs_entry, bs_sl, bs_tp),
            },
            "top": {
                "sell_limit": _ord("sell_limit", -1, sl_entry, sl_sl, sl_tp),
                "buy_stop": _ord("buy_stop", 1, tb_entry, tb_sl, tb_tp),
            },
        }
        # 代表（兼容旧消费方：取底部 buy limit 的 entry/tp/sl，direction=0 表示双向）
        setups.append({
            "idx": i, "bar_t": int(t[i]), "level": name,
            "dir": 0, "signal_type": "range_bracket",
            "bracket": bracket,
            "entry": bl_entry, "tp": bl_tp, "sl": bl_sl,
            "risk": float(risk), "reward": float(abs(bl_tp - bl_entry)),
            "rr": float(abs(bl_tp - bl_entry) / risk) if risk > 0 else None,
            "trade_limit": trade_limit,
            # —— 诊断字段（供 DEV 阶段做特征发现）——
            "pos": float(pos[i]), "rsi": float(rsi_arr[i]), "er": float(R["er"][i]),
            "er_n": float(R["er_n"][i]), "drift_n": float(R["drift_n"][i]),
            "box_n": float(R["box_n"][i]),
            "drift_atr": float(R["drift_atr"][i]), "box_atr": float(R["box_atr"][i]),
            "vol_expand": float(R["vol_expand"][i]) if np.isfinite(R["vol_expand"][i]) else None,
            "quiet_max": float(R["quiet_max"][i]) if np.isfinite(R["quiet_max"][i]) else None,
            "vol_ratio": float(vol_ratio[i]) if np.isfinite(vol_ratio[i]) else None,
            "box_high": resist, "box_low": supp, "box_mid": mid, "atr": a,
        })
        last_idx = i

    return setups
