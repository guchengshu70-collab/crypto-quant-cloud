"""signals_mtop.py —— 第四套独立信号系统：「M顶空头 + MACD/KDJ 共振」（用户指定，与 exp3 / violent / range 完全隔离）。

信号语义（全部参数经 quant_research 回测在 DEV-A→DEV-B→HOLD 三样本验证，预注册冻结）：
  形态：M顶（双顶）= 两个近似等高(±3%)摆动峰 g1<g2 + 中谷 vk + 收盘跌破颈线(vk) 确认（滞后 N=5 根确认摆动，无未来函数）。
  共振（在破位确认日 j0 判定，仅用 ≤j0 数据）：MACD 线处于信号线下方（m<sig，死叉态）且 KDJ J 值拐头向下（J[j0]<J[j0-1]）。
  方向：只做空（dir=-1）。
  开仓：sell-limit 挂于形态最高点 top=max(h[g1],h[g2])（"最尖尖"），等价格回抽触达成交（挂单型）。
  止损：SL = top×1.02（突破新高即认错）——已验证零爆仓（12h: HOLD +4.08%, CI+0.44, n=963；1d: HOLD +3.88%, CI+0.34）。
  止盈：TP = entry − 4×ATR 为参考目标；已验证最优执行为持有 ~15 天（中期空单），TP 仅作展示参考。
  级别：12h（主）/ 1d（次）。

纪律：
  - 无未来函数（摆动滞后确认、破位收盘确认、指标均 ≤i）；
  - 本模块不 import 任何 exp3/range/violent 代码，不修改、不依赖其他三套；
  - 与回测同一份逻辑（quant_research/_pl200_mdswing + _pl200_mtf 的冻结实现）。
"""
import numpy as np

# ── 冻结参数（回测验证，勿改）──────────────────────────────────────
SWING_N = 5          # 摆动确认半窗（±N 根）
EPS = 0.03           # 双峰等高容差 ±3%
W_MAX = 60           # 两峰最大间距（根）
SL_BUF = 0.02        # 止损缓冲：top×1.02
TP_ATR = 4.0         # 参考目标：entry − 4×ATR
WARMUP = 60          # 预热根数

PARAMS = {
    "swing_n": SWING_N, "eps": EPS, "w_max": W_MAX,
    "sl_buf": SL_BUF, "tp_atr": TP_ATR, "warmup": WARMUP,
}

# 级别集：(名称, tf_min, trade_limit)。12h 为主（HOLD n=963 最稳），1d 次之。
LEVELS_CRYPTO = [
    ("12h", 720, 22),
    ("1d", 1440, 20),
]


def _arrays(bars):
    """list[dict] → (t, o, h, l, c) numpy。"""
    n = len(bars)
    t = np.array([b["t"] for b in bars], dtype=np.int64)
    o = np.array([b["o"] for b in bars], dtype=float)
    h = np.array([b["h"] for b in bars], dtype=float)
    l = np.array([b["l"] for b in bars], dtype=float)
    c = np.array([b["c"] for b in bars], dtype=float)
    return t, o, h, l, c


def _atr(h, l, c, period=14):
    """Wilder ATR（滚动，无未来）。"""
    n = len(c)
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    out[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _ema(arr, n):
    """EMA（NaN 前缀容忍：从第一个有限值播种）。"""
    out = np.full(len(arr), np.nan)
    fin = np.where(np.isfinite(arr))[0]
    if fin.size < n:
        return out
    s = fin[0]
    alpha = 2.0 / (n + 1)
    out[s + n - 1] = arr[s:s + n].mean()
    for i in range(s + n, len(arr)):
        if not np.isfinite(arr[i]):
            continue
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _kdj(c, h, l, period=9):
    """KDJ(9,3,3)：J = 3K − 2D。滚动实现，无未来。"""
    n = len(c)
    K = np.full(n, np.nan); D = np.full(n, np.nan); J = np.full(n, np.nan)
    if n < period + 3:
        return J
    rsv = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh = h[i - period + 1:i + 1].max()
        ll = l[i - period + 1:i + 1].min()
        den = hh - ll
        rsv[i] = 50.0 if den < 1e-12 else (c[i] - ll) / den * 100.0
    K[period - 1] = rsv[period - 1]; D[period - 1] = K[period - 1]
    for i in range(period, n):
        K[i] = (2.0 / 3.0) * K[i - 1] + (1.0 / 3.0) * rsv[i]
        D[i] = (2.0 / 3.0) * D[i - 1] + (1.0 / 3.0) * K[i]
    J = 3.0 * K - 2.0 * D
    return J


def _swings(h, l, n):
    """滞后确认的摆动峰/谷（i 在 i+n 之后才可确认）。返回 (谷idx, 峰idx)。"""
    N = len(h)
    sl = np.zeros(N, dtype=bool); sh = np.zeros(N, dtype=bool)
    for i in range(n, N - n):
        if l[i] <= l[i - n:i + n + 1].min() and l[i] < l[i - n:i].min():
            sl[i] = True
        if h[i] >= h[i - n:i + n + 1].max() and h[i] > h[i - n:i].max():
            sh[i] = True
    return np.where(sl)[0], np.where(sh)[0]


def _detect_mtop(h, l, c, p):
    """检测 M顶 + 破位确认。返回 [(g1, g2, vk, top, j0), ...]。
    j0 = 收盘跌破颈线 vk 的确认日（≥ g2+SWING_N，无未来）。"""
    n = len(c)
    _, gp = _swings(h, l, int(p["swing_n"]))
    wmax = int(p["w_max"])
    eps = float(p["eps"])
    n_sw = int(p["swing_n"])
    out = []
    for k in range(1, len(gp)):
        g2 = int(gp[k]); g1 = int(gp[k - 1])
        if g2 - g1 > wmax:
            continue
        seg = slice(g1 + 1, g2)
        if seg.start >= seg.stop:
            continue
        vk = float(l[seg].min())
        if vk >= min(h[g1], h[g2]):
            continue
        if abs(h[g2] - h[g1]) / h[g1] > eps:
            continue
        top = float(max(h[g1], h[g2]))
        # 破位确认：收盘跌破颈线，且第二峰已确认（g2+n_sw）
        start = g2 + n_sw
        if start >= n:
            continue
        hit = np.where(c[start:] < vk)[0]
        if hit.size == 0:
            continue
        j0 = start + int(hit[0])
        out.append((g1, g2, vk, top, j0))
    return out


def _resonance_f1(m, sig, J, j0):
    """F1 共振（回测验证：12h/1d top 挂单 HOLD 三样本显著）：
    MACD 线下降 m[j0]<m[j0-1] 且 KDJ J 拐头向下 J[j0]<J[j0-1]（j0 判定，≤j0 数据）。"""
    if j0 < 2 or not (np.isfinite(m[j0]) and np.isfinite(m[j0 - 1])
                      and np.isfinite(J[j0]) and np.isfinite(J[j0 - 1])):
        return False
    return (m[j0] < m[j0 - 1]) and (J[j0] < J[j0 - 1])


def scan_level(bars, level_cfg, params=PARAMS, market="crypto", live=False):
    """扫描单级别的 M顶空头信号（仅空头）。

    bars      : list[dict]（t/o/h/l/c）或 numpy [t,o,h,l,c(,v)]
    level_cfg : (name, tf_min, trade_limit)
    live=True : 扫描到最新已收盘 bar（实盘）；False：预留前向窗口（回测）
    返回 setups：{kind, dir=-1, entry=top, sl=top×1.02, tp, limit_price, idx=j0, atr, level, bar_t, reason, signal_type}
    """
    if market != "crypto":
        return []
    if isinstance(bars, np.ndarray):
        arr = np.asarray(bars, dtype=float)
        t, o, h, l, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    else:
        t, o, h, l, c = _arrays(bars)
    n = len(c)
    if n < 200:
        return []
    p = dict(params)
    warmup = int(p["warmup"])
    if n < warmup + 30:
        return []
    a = _atr(h, l, c, 14)
    m = _ema(c, 12) - _ema(c, 26)
    sig = _ema(m, 9)
    J = _kdj(c, h, l, 9)
    scan_end = n if live else n - 30

    setups = []
    for g1, g2, vk, top, j0 in _detect_mtop(h, l, c, p):
        if j0 < warmup or j0 >= scan_end:
            continue
        atr = float(a[j0])
        if not (np.isfinite(atr) and atr > 1e-12):
            continue
        if not _resonance_f1(m, sig, J, j0):
            continue
        entry = top
        sl = top * (1.0 + float(p["sl_buf"]))
        tp = entry - float(p["tp_atr"]) * atr
        setups.append({
            "kind": "mtop_short",
            "signal_type": "mtop_short",
            "dir": -1,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "risk": float(abs(entry - sl)),
            "reward": float(abs(tp - entry)),
            "rr": float(abs(tp - entry) / abs(entry - sl)) if abs(entry - sl) > 1e-12 else None,
            "limit_price": float(entry),   # sell-limit 挂单点（形态最高点）
            "entry_mode": "mtop_limit",
            "atr": float(atr),
            "idx": int(j0),
            "bar_t": int(t[j0]),
            "top": float(top),
            "vk": float(vk),
            "reason": (f"M顶确认（双峰等高±3%，第二峰 {g2} 确认后收盘跌破颈线 {vk:.6g}）+ "
                       f"MACD 线下降 + KDJ 拐头向下（F1 共振）→ 形态最高点 {top:.6g} 挂空（最尖尖），"
                       f"止损新高 {sl:.6g} 认错，参考目标 {tp:.6g}（建议持有 ~15 天中期空单）"),
        })
    return setups
