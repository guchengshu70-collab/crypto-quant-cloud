"""quant_research/indicators.py — 轻量、无副作用的技术指标（numpy 实现）。"""

import numpy as np


def ema(arr: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan)
    if len(arr) < n:
        return out
    alpha = 2.0 / (n + 1)
    out[n - 1] = arr[:n].mean()
    for i in range(n, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def sma(arr: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan)
    c = np.cumsum(np.nan_to_num(arr, nan=0.0))
    for i in range(n - 1, len(arr)):
        out[i] = (c[i] - c[i - n]) / n
    return out


def rsi(arr: np.ndarray, n: int = 14) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan)
    if len(arr) < n + 1:
        return out
    delta = np.diff(arr)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.full(len(arr), np.nan)
    al = np.full(len(arr), np.nan)
    ag[n] = gain[:n].mean()
    al[n] = loss[:n].mean()
    for i in range(n + 1, len(arr)):
        ag[i] = (ag[i - 1] * (n - 1) + gain[i - 1]) / n
        al[i] = (al[i - 1] * (n - 1) + loss[i - 1]) / n
    rs = ag / np.where(al < 1e-12, 1e-12, al)
    out[n:] = 100 - 100 / (1 + rs[n:])
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    out = np.full_like(close, np.nan)
    if len(close) < 2:
        return out
    pc = np.empty_like(close)
    pc[0] = close[0]
    pc[1:] = close[:-1]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - pc),
        np.abs(low - pc),
    ])
    out[0] = tr[0]
    for i in range(1, len(close)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def bollinger(close: np.ndarray, n: int = 20, k: float = 2.0):
    close = np.asarray(close, dtype=float)
    mid = sma(close, n)
    std = np.full_like(close, np.nan)
    c = np.cumsum(np.nan_to_num(close, nan=0.0))
    c2 = np.cumsum(np.nan_to_num(close ** 2, nan=0.0))
    for i in range(n - 1, len(close)):
        mean = (c[i] - c[i - n]) / n
        var = (c2[i] - c2[i - n]) / n - mean * mean
        std[i] = np.sqrt(max(var, 0.0))
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder 平滑：前 n-1 根置 NaN，第 n-1 根起为前 n 根均值并递归。"""
    out = np.full_like(x, np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.nanmean(x[:n])
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def _plus_minus_dm(high, low):
    up = high[1:] - high[:-1]
    dn = low[:-1] - low[1:]
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    return plus, minus


def adx(high, low, close, n: int = 14) -> np.ndarray:
    """返回与输入等长的 ADX 数组（前 n*2 根为 NaN）。"""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    N = len(close)
    if N < 2 * n:
        return np.full(N, np.nan)
    tr = atr(high, low, close, 1)
    plus, minus = _plus_minus_dm(high, low)
    plus_dm = np.zeros(N)
    minus_dm = np.zeros(N)
    plus_dm[1:] = plus
    minus_dm[1:] = minus
    atr_s = _wilder(tr, n)
    pdm_s = _wilder(plus_dm, n)
    mdm_s = _wilder(minus_dm, n)
    eps = 1e-12
    pdi = 100 * pdm_s / np.where(atr_s < eps, eps, atr_s)
    mdi = 100 * mdm_s / np.where(atr_s < eps, eps, atr_s)
    dx = 100 * np.abs(pdi - mdi) / np.where((pdi + mdi) < eps, eps, pdi + mdi)
    return _wilder(dx, n)


def resample_bars(bars, group: int) -> list[dict]:
    """将交易级 Bar 列表按连续 group 根合成为更高周期（用于跨周期偏置）。"""
    out = []
    for i in range(0, len(bars) - group + 1, group):
        chunk = bars[i:i + group]
        out.append({
            "t": chunk[0]["t"],
            "o": chunk[0]["o"],
            "h": max(b["h"] for b in chunk),
            "l": min(b["l"] for b in chunk),
            "c": chunk[-1]["c"],
            "v": sum(b["v"] for b in chunk),
            "is_closed": True,
        })
    return out
