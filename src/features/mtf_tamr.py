"""crypto-quant · MTF-TAMR 信号引擎（exp3 最终架构接入层）。

本模块是 app 实盘信号引擎，是 quant_research/signals.py 的**唯一可信调用方**：
直接 import quant_research/signals.py（与回测/验证同一份已锁定代码），杜绝双管线漂移。

职责：
- 复用 app 自带 BinanceRest 客户端取多周期(1h/4h/6h/12h/1w + 偏置 4h/12h/1d/1M)K线。
- 对每个交易级别调用 signals.scan_level 产出 setup，汇聚多周期对齐结果。
- 把 setup 映射为 model.signal 契约字典（同时带顶层字段与 price_zone.recommendation 嵌套，
  以保形前端两条渲染路径：renderAnalysis 与 renderScan/onSignalPush）。
- 严守纪律：仅用已收盘 bar（丢弃每个周期最后一根未收盘 bar，杜绝未来函数）。

纪律（不可破坏，与 signals.py 一致）：
- 无未来函数；成本/滑点/风控由上层 engine 处理；本模块只产出 setup→signal。
- 仅加密（crypto）。A 股已被持久排除（见项目 MEMORY）。
"""

import asyncio
import os
import sys
import time

# 把 quant_research 加入 sys.path，使 `import signals`（及 signals 内部的
# `from indicators/filters import ...`）可解析——这是与回测/验证同一份代码的关键。
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QR = os.path.join(_ROOT, "quant_research")
if _QR not in sys.path:
    sys.path.insert(0, _QR)

from signals import scan_level, LEVELS_CRYPTO, PARAMS  # noqa: E402
import signals_violent  # noqa: E402  新增「暴力拉升/暴跌版」信号系统（与 exp3 并列，互不覆盖）
import signals_range    # noqa: E402  第三套「震荡期」独立信号源（箱体均值回归，与 exp3 完全隔离）
import signals_mtop     # noqa: E402  第四套「M顶空头+MACD/KDJ共振」独立信号源（空头形态，与其余三套完全隔离）

# ---- 信号系统路由（多套并列，互不覆盖 exp3 核心）----
# exp3   : 趋势内回调均值回归（在用，默认）。
# violent: 暴力拉升/暴跌版（抓「大阳/大阴【启动前】」—— 大实体出现之前埋伏，新增第二套）。
# range  : 震荡期独立信号源——先证"处于箱体且暂时无拉升/暴跌"，再于箱体边界做均值回归；
#          与 exp3 完全独立：无 HTF 趋势偏置、不要求多周期共振、不用 EMA20 拉回。
SIGNAL_SYSTEMS = {
    "exp3": {"label": "EXP3 · 趋势内回调均值回归", "reserved": False},
    # EXP4-S：与 exp3 同信号池 / 同入场 / 同止损（SL 2.2×ATR、band k=0.25），
    # 差异仅两处 —— ① 出场改「一次性 EMA20 全平」（关掉 V11 分批）；
    # ② 级别列表 1h/4h/6h/12h（恢复 1h、移除 1w）。共振门槛 ≥2 与 exp3 一致。
    "exp4": {"label": "EXP4-S · 一次性EMA20止盈 + 1h扩容（测试版）", "reserved": False},
    # 以下三套按用户要求（2026-09-06）**前端隐藏、后端保留**：不再出现在选项卡与策略列表，
    # 但路由/信号模块完好保留，随时可一行恢复。后端收到这些 strategy 仍按原语义工作。
    "violent": {"label": "EXP3 · 暴力拉升/暴跌（启动前埋伏）", "reserved": False, "hidden": True},
    "range": {"label": "震荡期 · 箱体均值回归（独立信号源·实验）", "reserved": False, "hidden": True},
    "mtop": {"label": "空头形态 · M顶+MACD/KDJ共振（独立信号源·已验证）", "reserved": False, "hidden": True},
}
DEFAULT_STRATEGY = "exp3"

# 发布聚合的「时间对齐」窗口（毫秒）：级别一根 bar 的时长。
# 修复 2026-09-02：旧实现把整个数据窗内全部历史触发跨月累计投票/对齐，
# 导致方向被数周前的旧触发污染、rep 选中陈旧参数（如 8 月空单挂在 7 万现价）。
# 与回测 grade_of 的 ALIGN_WIN 同一口径——只统计触发时刻 ±1 根 bar 的同向级别。
_ALIGN_WIN_MS = {name: tf_min * 60000 for name, tf_min, *_ in LEVELS_CRYPTO}

# 震荡期级别集（来自 signals_range，5 级别：1h/4h/12h/1d/1w；与 exp3 的 12 级别互不干扰）
RANGE_LEVELS = signals_range.LEVELS_CRYPTO
RANGE_LEVEL_NAMES = [n for n, *_ in RANGE_LEVELS]
RANGE_TL = {n: tl for n, _m, tl in RANGE_LEVELS}

# M顶空头级别集（来自 signals_mtop，2 级别：12h 主 / 1d 次；与其余信号系统互不干扰）
MTOP_LEVELS = signals_mtop.LEVELS_CRYPTO
MTOP_LEVEL_NAMES = [n for n, *_ in MTOP_LEVELS]
MTOP_TL = {n: tl for n, _m, tl in MTOP_LEVELS}


def _signals_module(strategy: str):
    """按 strategy 返回对应 signals 模块；exp3 为兜底。"""
    if strategy == "violent":
        return signals_violent
    if strategy == "range":
        return signals_range
    return sys.modules["signals"]  # exp3 默认/兜底

# ---- 周期 → 币安 interval 映射（覆盖 5m~1M 全周期，供 focus_tf 按需取数） ----
MIN_TO_INTERVAL = {
    5: "5m", 15: "15m", 30: "30m", 60: "1h", 120: "2h", 240: "4h",
    360: "6h", 480: "8h", 720: "12h", 1440: "1d", 4320: "3d",
    10080: "1w", 43200: "1M",
}
INTERVAL_TO_MIN = {v: k for k, v in MIN_TO_INTERVAL.items()}

# focus_tf（用户所选时间线）→ 该级别的偏置大周期（HTF 定方向，exp3 核心）。
# 任何 UI 时间线都映射到一个 (交易级, 偏置级) 对，保证"选 X 就是 X 推荐"。
FOCUS_BIAS = {
    "5m": "15m", "15m": "1h", "30m": "1h", "1h": "4h", "2h": "4h",
    "4h": "12h", "6h": "12h", "8h": "1d", "12h": "1d", "1d": "1w",
    "3d": "1w", "1w": "1M", "1M": "1w",
}

# 取数条数：交易级需 warmup(60)+trade_limit(≤36)+5；偏置级需 ≥60。留足余量。
TRADE_LIMIT_BARS = 160
BIAS_LIMIT_BARS = 120
_KLINES_PAGE = 1000  # Binance klines 单次上限

# 缓存 TTL（秒）：约该周期 0.8 根时长，避免每次 5m 触发都重拉全市场。
_TTL_MIN = {60: 1800, 240: 7200, 360: 10800, 720: 21600,
            1440: 43200, 10080: 86400, 43200: 86400 * 7}

RISK_DISCLAIMER = (
    "本信号由程序基于公开行情数据自动生成，仅供学习研究参考，"
    "不构成任何投资建议。市场有风险，投资需谨慎。"
)

# 级别优先级（低周期在前=触发更近、作为代表级别优先）
LEVEL_PRIORITY = [name for name, *_ in LEVELS_CRYPTO]
LEVEL_CRYPTO_NAMES = [name for name, *_ in LEVELS_CRYPTO]  # 级别名列表（供对齐统计）

# ---- 已验证执行/展示层参数（2026-08-31 采纳：415 币规模化验证的最优配置 V11）----
# 回测来源：quant_research/_bt_exp3_v2.py + _bt_scale_test.py（415 币、G9 盲测通过）：
#   4H·2/5 · V11（分批 50%@+1R + 余 50%@EMA20 + BE@1R）· sl=2.2×ATR · close 入场
#   关键证据：胜率 69.8% 在 415 币 / 2587 笔上稳定；G10 全新币(155) 表现优于全池；
#             拉高止盈 / 收缩止损均为负优化（_bt_exp3_tpsl_sweep.py，285 组网格盲测）。
#   1h 及以下周期信号已下线（毛边缘 0.073% < 往返成本 0.2%，taker 下数学上不划算）。
V11_SL_MULT = 2.2          # 止损：2.2×ATR（V11 验证口径）
V11_TP1_R = 1.0            # 首批止盈：+1R（50% 仓位，锁盈抬胜率）
V11_TP1_FRAC = 0.5         # 首批平仓比例
V11_BE_R = 1.0             # 触及 +1R 后剩余仓位移动止损到成本（BE）
RETIRED_TFS = ("5m", "15m", "30m", "1h")   # 已下线周期（前端据此不画信号）

# ---- 入场价偏移（2026-08-31 采纳：k=0.25×ATR，三独立样本复现） ----
# 回测来源：quant_research/_bt_entry_offset_test.py（DEV G6-G8 选参）+ _bt_k025_final_verify.py
# （325 币独立集 + 415 币全池复现）：多单开仓线压低 close−0.25×ATR、空单抬高 close+0.25×ATR，
# 相对 close 入场收益 +22.6%~+33.6%，限价成交率 ≈78%（8h 内触及按挂单价成交，超时市价回退）。
V11_ENTRY_K = 0.25         # 入场偏移深度（×ATR）；仅 4h 聚焦级别生效（1h 已下线保持 close）
ORDER_WAIT_MIN = 8 * 60    # 限价挂单等待窗（min）：信号后 8h 内触及成交，超时市价回退

# ---- EXP4-S 级别列表（2026-09-06 采纳：恢复 1h、移除 1w）----
# 回测来源：EXP4信号量扩容方案_九条路径实测与推荐EXP4-S.md（127 币 · 92 天 · DEV/HOLDOUT 切分 07-13）
#   恢复 1h：1h 挂在「≥2 共振 + k=0.25 限价挂单」结构下 154 笔，胜率 68.8%、样本外 R 口径
#            PF 1.749、三段子窗口实收 PF 1.462/1.507/1.116 全 >1；净效果 320→489 笔
#            （3.5→5.34 笔/天）、0 信号日 16%→7%、样本外实收 PF 0.837→1.154。
#   移除 1w：92 天窗口 1w 级别 setup 数 = 0（死级别，保留纯占计算）。
#   门槛 ≥2 **保持不变**：降到 ≥1 已被否决（样本外 PF 0.700、P(盈利)=0.15%、MDD 71R）。
# 【隔离红线】禁止修改全局 RETIRED_TFS —— exp3 的 1h 因毛边缘 0.073% < taker 往返成本 0.2%
#   被永久下线；此处仅为 exp4 单独放开 1h 的参与资格，两套系统互不干扰。
EXP4S_LEVELS = ("1h", "4h", "6h", "12h")
# 【方向锚点纪律】exp4 的 1h 只参与「共振计数」，不夺取方向主导权：
# latest 锚点仍只从 4h/6h/12h 中选（沿用 2026-09-03 修复），避免 1h 触发掩盖 4h 信号。
# rep（入场/止损/止盈参数来源）同样跳过 1h —— 与 exp3 同口径，也与回测逐笔配对
# 自洽（配对差 −0.0015R ≈ 0 说明回测中 rep 未变）。


def _apply_v11_exit(setup: dict) -> dict:
    """对 exp3 的 4h / 1h 级别应用已验证的 V11 出场结构 + 入场偏移（signals.py 核心触发逻辑零改动）。

    - 入场：4h 用限价挂单留容错——多单 close−0.25×ATR（压低开仓线）、空单 close+0.25×ATR（抬高开仓线），
      8h 内触及按挂单价成交，超时市价回退（k=0.25 三独立样本验证，收益 +22.6%~+33.6%）。
      1h 保持 close（1h 已下线，维持 close 诚信成交）。
    - 止损：sl = E ∓ 2.2×ATR（V11 验证口径，替代原 1.8）。
    - 止盈①：tp1 = E ± 1R，首批平 50% 仓位（锁盈抬胜率）。
    - 止盈②：tp2 = EMA20（均值回归目标，剩余 50%）。
    - BE：触及 +1R 后，剩余仓位止损移至入场价。
    EMA20 几何失效（sl<E<tp2 不成立）时安全降级为 +1R 单止盈，避免「TP 命中即亏损」。
    """
    level = setup.get("level")
    if level not in ("1h", "4h"):
        return setup
    d = setup["dir"]
    atr = float(setup["risk"]) / PARAMS["sl_mult"]   # 由 core 止损乘数复原 ATR
    E = float(setup["entry"])
    tp2 = float(setup["tp"])                          # EMA20（core 语义）
    risk = V11_SL_MULT * atr
    nsl = E - risk if d > 0 else E + risk
    tp1 = E + V11_TP1_R * risk if d > 0 else E - V11_TP1_R * risk
    if d > 0 and not (nsl < E < tp2):
        tp2 = tp1
    if d < 0 and not (tp2 < E < nsl):
        tp2 = tp1
    setup["entry"] = E
    setup["sl"] = nsl
    setup["tp"] = tp2
    setup["risk"] = abs(E - nsl)
    setup["reward"] = abs(tp2 - E)
    setup["atr"] = atr
    setup["tp1_price"] = round(tp1, 8)
    setup["be_price"] = round(E, 8)
    setup["tp1_frac"] = V11_TP1_FRAC
    setup["exit_plan"] = "v11"
    # 入场偏移：仅 4h 聚焦级别（多单压低 / 空单抬高 0.25×ATR 限价挂单，8h 超时市价回退）
    if level == "4h":
        limit = E - V11_ENTRY_K * atr if d > 0 else E + V11_ENTRY_K * atr
        # 几何保护：挂单价不得越过止损/止盈（限价单语义，防御性）
        if d > 0:
            limit = min(limit, E)          # 多单只压低，不抬高
            limit = max(limit, nsl)        # 不低于止损
            if limit >= tp2:
                limit = max(E - 1e-8, nsl)
        else:
            limit = max(limit, E)          # 空单只抬高，不压低
            limit = min(limit, nsl)        # 不高于止损
            if limit <= tp2:
                limit = min(E + 1e-8, nsl)
        setup["band"] = {
            "band": V11_ENTRY_K, "limit_price": round(limit, 8),
            "order_valid_min": ORDER_WAIT_MIN,
            "entry_desc": f"{'close−' if d > 0 else 'close+'}{V11_ENTRY_K}×ATR",
        }
    return setup


def _apply_exp4s_entry(setup: dict) -> dict:
    """EXP4-S：与 exp3 **同信号池 / 同入场 / 同止损**，唯一变量是出场退回「一次性 EMA20」。

    与 _apply_v11_exit 的差异只有一处：不设 tp1_price / be_price / tp1_frac，
    exit_plan = "single" —— 触及 EMA20 一次性全平，不做 +1R 平半仓、不移动保本止损。

    其余全部对齐 exp3，保证 A/B 只存在「出场方式」这一个变量：
      - 止损 SL = 2.2×ATR（PARAMS["sl_mult"] = 2.2 = V11_SL_MULT，core 已给出，此处不改动）
      - 止盈 TP = EMA20（core 语义，此处不改动）
      - 限价入场偏移 k = 0.25×ATR，8h 内触及成交、超时市价回退（与 exp3 逐字段一致）
    依据：关掉 V11 分批 → 逐笔配对 +0.0735R，95%CI [+0.0269,+0.1220] 不含 0，P=0.9996；
          且在 SL 1.0/1.4/1.8/2.2/2.6/3.0 六种止损宽度下净收益全为正（问题在机制不在参数）。
    """
    level = setup.get("level")
    if level not in ("1h", "4h"):
        return setup
    d = setup["dir"]
    atr = float(setup["risk"]) / PARAMS["sl_mult"]   # 由 core 止损乘数复原 ATR（core sl_mult=2.2）
    E = float(setup["entry"])
    tp = float(setup["tp"])                          # EMA20（core 语义）
    sl = float(setup["sl"])
    setup["atr"] = atr
    setup["exit_plan"] = "single"                    # 一次性止盈，无分批 / 无保本
    limit = E - V11_ENTRY_K * atr if d > 0 else E + V11_ENTRY_K * atr
    # 几何保护：挂单价不得越过止损/止盈（限价单语义，防御性）
    if d > 0:
        limit = min(limit, E)          # 多单只压低，不抬高
        limit = max(limit, sl)         # 不低于止损
        if limit >= tp:
            limit = max(E - 1e-8, sl)
    else:
        limit = max(limit, E)          # 空单只抬高，不压低
        limit = min(limit, sl)         # 不高于止损
        if limit <= tp:
            limit = min(E + 1e-8, sl)
    setup["band"] = {
        "band": V11_ENTRY_K, "limit_price": round(limit, 8),
        "order_valid_min": ORDER_WAIT_MIN,
        "entry_desc": f"{'close−' if d > 0 else 'close+'}{V11_ENTRY_K}×ATR",
    }
    return setup


class MultiTFCache:
    """按 (symbol, interval) 缓存已收盘 K线，带 TTL，减少 REST 调用。"""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], tuple[float, list[dict]]] = {}

    def get(self, symbol: str, interval: str):
        return self._store.get((symbol, interval))

    def put(self, symbol: str, interval: str, bars: list[dict]) -> None:
        self._store[(symbol, interval)] = (time.time(), bars)


async def _fetch_closed(rest, cache: MultiTFCache, symbol: str, interval: str,
                        minutes: int, limit: int) -> list[dict]:
    """拉取某周期的已收盘 bar（丢弃最后一根可能未收盘的当前 bar，杜绝未来函数）。

    limit > 1000 时分页向前拉取（endTime 递减），拼成完整序列（从旧到新）。
    """
    key = (symbol, interval)
    now = time.time()
    c = cache.get(symbol, interval)
    ttl = _TTL_MIN.get(minutes, 3600)
    if c and now - c[0] < ttl:
        return c[1]
    raw: list[dict] = []
    if limit <= _KLINES_PAGE:
        raw = await rest.klines(symbol, interval=interval, limit=limit)
    else:
        got = 0
        end_ts = None
        while got < limit:
            want = min(_KLINES_PAGE, limit - got)
            batch = await rest.klines(symbol, interval=interval, limit=want, end_ts=end_ts)
            if not batch:
                break
            raw = batch + raw          # 从旧到新拼接
            got += len(batch)
            end_ts = batch[0]["t"]     # 下一页取更早的
            if len(batch) < want:
                break
    bars = [{"t": k["t"], "o": k["o"], "h": k["h"], "l": k["l"], "c": k["c"], "v": k["v"]} for k in raw]
    if bars:
        bars = bars[:-1]  # 丢弃可能正在形成的最后一根
    # 缓存防毒：条数明显不足的响应（网络抖动/数据源截断）不入缓存——
    # 否则一次抖动会让该级别在长 TTL 内持续被判"数据不足"而跳过，且不同调用
    # 在不同时间缓存到不同完整度的数据，导致同一级别结果互相打架。
    if len(bars) >= 100:
        cache.put(symbol, interval, bars)
    return bars


async def analyze_symbol(symbol: str, rest, cache: MultiTFCache | None = None,
                         settings=None, tf: str | None = None, strategy: str | None = None) -> dict:
    """对一个币做多周期扫描，产出 model.signal 契约字典（direction=0 表示观望）。

    tf：用户所选时间线（如 "15m"）。传入时进入 focus 模式——只扫描该级别 + 其偏置大周期，
    使"选 X 就是 X 推荐"。不传则聚合全部 LEVELS_CRYPTO 级别（代表性级别=最低触发级）。
    无网络/部分失败时返回中性信号（direction=0, ready=True），不会抛异常。
    """
    cache = cache or MultiTFCache()
    symbol = symbol.upper()

    # 信号系统路由：exp3(默认) / violent(暴力拉升暴跌版) / range(震荡期独立源)。
    strategy = strategy or DEFAULT_STRATEGY
    if strategy not in SIGNAL_SYSTEMS:
        strategy = DEFAULT_STRATEGY

    # 震荡期：独立逻辑（箱体边界均值回归），单独走 analyze_range，不与 exp3 的
    # HTF 偏置 / EMA20 拉回 / ≥2/5 共振 合成管线耦合。
    if strategy == "range":
        return await analyze_range(symbol, rest, cache, settings, tf, "range")

    # M顶空头（独立逻辑，用户新增第四套）：单独走 analyze_mtop，完全隔离。
    if strategy == "mtop":
        return await analyze_mtop(symbol, rest, cache, settings, tf, "mtop")

    sig_mod = _signals_module(strategy)

    # 并行拉取需要的周期（交易级 + 偏置级去重）
    needed: dict[str, int] = {}  # interval -> minutes（用于 TTL）
    level_plan: list[tuple[str, str, str, int]] = []  # (level_name, trade_iv, bias_iv, trade_limit)
    focus = bool(tf and tf in FOCUS_BIAS)
    if focus:
        # focus 模式：只取该级别 + 其偏置大周期（快、与 UI 所选严格对应）
        ti = tf
        bi = FOCUS_BIAS[tf]
        needed[ti] = INTERVAL_TO_MIN.get(ti, 60)
        needed[bi] = INTERVAL_TO_MIN.get(bi, 240)
        level_plan.append((ti, ti, bi, 30))
    else:
        # EXP4-S 级别列表：1h/4h/6h/12h（恢复 1h、移除 1w 死级别）；其余策略用全量 LEVELS_CRYPTO。
        _lv = LEVELS_CRYPTO if strategy != "exp4" else [l for l in LEVELS_CRYPTO if l[0] in EXP4S_LEVELS]
        for name, tf_min, bias_min, tl in _lv:
            ti = MIN_TO_INTERVAL[tf_min]
            bi = MIN_TO_INTERVAL[bias_min]
            needed[ti] = tf_min
            needed[bi] = bias_min
            level_plan.append((name, ti, bi, tl))
    # 波段回踩入场：E = close − 0.75×ATR 由 4h 直接计算（无需 15m 细粒度数据）。
    async def _pull(iv):
        return iv, await _fetch_closed(rest, cache, symbol, iv, needed[iv], 250)

    results = dict(await asyncio.gather(*(_pull(iv) for iv in needed)))
    by_interval = results  # interval -> bars

    setups_by_level: list[dict] = []
    insufficient: list[str] = []  # 因 K线历史不足被跳过的级别（新币/数据缺口）
    for name, ti, bi, tl in level_plan:
        trade_bars = by_interval.get(ti, [])
        bias_bars = by_interval.get(bi, [])
        if len(trade_bars) < 70 or len(bias_bars) < 60:
            insufficient.append(name)
            continue
        try:
            # live=True：实盘扫描窗口延伸到最新已收盘 bar（入场条件数学不变，
            # 回测路径不传此参数——见 signals.scan_level docstring，方案B）
            su = sig_mod.scan_level(trade_bars, bias_bars,
                            (name, INTERVAL_TO_MIN.get(ti, 60), INTERVAL_TO_MIN.get(bi, 240), tl),
                            market="crypto", live=True)
        except Exception:
            su = []
        for s in su:
            s["level"] = name
            if 0 <= s.get("idx", -1) < len(trade_bars):
                s["bar_t"] = trade_bars[s["idx"]]["t"]  # 触发 bar 时间（时效展示/验证用）
                if strategy == "exp3":
                    _apply_v11_exit(s)   # V11 分批止盈结构仅 exp3；暴力/预留系统 setup 已自带 entry/tp/sl
                elif strategy == "exp4":
                    _apply_exp4s_entry(s)  # EXP4-S：同入场 / 同止损，出场改一次性 EMA20 + 1h 同样挂单
            setups_by_level.append(s)

    return _build_signal(symbol, setups_by_level, focus_tf=(tf if focus else None),
                         insufficient=insufficient, strategy=strategy)


# ----------------------------------------------------------------------------
# 震荡期独立信号源（range）—— 与 exp3 / violent 完全隔离的合成管线
# ----------------------------------------------------------------------------
async def analyze_range(symbol: str, rest, cache: MultiTFCache | None = None,
                        settings=None, tf: str | None = None, strategy: str = "range") -> dict:
    """震荡期（箱体）均值回归：单级别箱体边界触发，无 HTF 趋势偏置、不要求多周期共振。

    与 exp3 的 analyze_symbol 并列但互不耦合：
    - 只取各交易级 K线，不取 bias 高周期（震荡期无趋势方向可言）。
    - 调用 signals_range.scan_level（3 元祖 level_cfg，无 bias 参数）。
    - 合成交给 _build_range_signal（独立逻辑，不沿用 EMA20 拉回 / V11 / ≥2/5 共振门控）。
    无网络/部分失败时返回中性信号（direction=0, ready=True），不会抛异常。
    """
    cache = cache or MultiTFCache()
    symbol = symbol.upper()
    focus = bool(tf and tf in RANGE_LEVEL_NAMES)

    needed: dict[str, int] = {}
    level_plan: list[tuple[str, str, int, int]] = []  # (name, interval, tf_min, trade_limit)
    if focus:
        ti = tf
        tmin = INTERVAL_TO_MIN.get(ti, 60)
        tl = RANGE_TL.get(ti, 30)
        needed[ti] = tmin
        level_plan.append((ti, ti, tmin, tl))
    else:
        for name, tf_min, tl in RANGE_LEVELS:
            ti = MIN_TO_INTERVAL[tf_min]
            needed[ti] = tf_min
            level_plan.append((name, ti, tf_min, tl))

    async def _pull(iv):
        return iv, await _fetch_closed(rest, cache, symbol, iv, needed[iv], 250)

    results = dict(await asyncio.gather(*(_pull(iv) for iv in needed)))
    setups_by_level: list[dict] = []
    insufficient: list[str] = []
    for name, ti, tf_min, tl in level_plan:
        bars = results.get(ti, [])
        if len(bars) < 200:
            insufficient.append(name)
            continue
        try:
            su = signals_range.scan_level(
                bars, (name, tf_min, tl), signals_range.PARAMS,
                market="crypto", live=True,
            )
        except Exception:
            su = []
        for s in su:
            s["level"] = name
            if "bar_t" not in s and 0 <= s.get("idx", -1) < len(bars):
                s["bar_t"] = bars[s["idx"]]["t"]
            setups_by_level.append(s)

    return _build_range_signal(
        symbol, setups_by_level, focus_tf=(tf if focus else None),
        insufficient=insufficient, strategy=strategy,
    )


def _build_range_signal(symbol: str, setups: list[dict], focus_tf: str | None = None,
                        insufficient: list[str] | None = None, strategy: str = "range") -> dict:
    """震荡期信号合成：箱体确认（含量能明显缩小）即输出**双向限价挂单 bracket**。

    与 exp3 的 _build_signal 完全隔离——range 不取 HTF 偏置、不要求多周期共振、不沿单边
    触发。研究已证短周期多级别"共振"反而是破位前兆，故只统计"多少级别当前处于箱体"用于
    排序展示，不作为发布门槛（任一级别确认即发布双向 bracket）。
    """
    total = len(RANGE_LEVELS)
    if not setups:
        if insufficient:
            return _neutral(symbol, total,
                            note=f"数据不足：{'/'.join(insufficient)} 级别 K线历史不够，暂无法评估震荡期",
                            strategy=strategy)
        return _neutral(symbol, total,
                        note="当前无震荡期箱体信号（价格未进入可交易箱体区间，或非震荡市 / 量能未明显缩小）",
                        strategy=strategy)

    if focus_tf and focus_tf in RANGE_LEVEL_NAMES:
        at_focus = [s for s in setups if s["level"] == focus_tf]
        if not at_focus:
            return _neutral(symbol, total,
                            note=f"{focus_tf} 级别当前无震荡期箱体触发（观望）", strategy=strategy)
        rep = at_focus[-1]  # 该级别最新触发的箱体
        return _make_range_bracket_signal(symbol, rep, focus_tf, total, strategy)

    # ---- 聚合模式（无 focus）：取级别优先级最低的箱体为代表，统计多少级别处于箱体 ----
    setups_sorted = sorted(
        setups,
        key=lambda s: RANGE_LEVEL_NAMES.index(s["level"]) if s["level"] in RANGE_LEVEL_NAMES else 99,
    )
    rep = setups_sorted[0]
    return _make_range_bracket_signal(symbol, rep, rep["level"], total, strategy, all_setups=setups)


def _round_bracket(b: dict) -> dict:
    """把 bracket 内所有数值四舍五入为 6 位（rr 3 位），便于 JSON 传输与前端展示。"""
    def r(x):
        return round(float(x), 6)

    def o(d):
        return {
            "kind": d["kind"], "dir": d["dir"], "entry": r(d["entry"]), "sl": r(d["sl"]),
            "tp": r(d["tp"]), "risk": r(d["risk"]), "reward": r(d["reward"]),
            "rr": round(d["rr"], 3) if d["rr"] is not None else None,
        }

    return {
        "box_low": r(b["box_low"]), "box_high": r(b["box_high"]),
        "box_mid": r(b["box_mid"]), "atr": r(b["atr"]),
        "bottom": {"buy_limit": o(b["bottom"]["buy_limit"]),
                   "sell_stop": o(b["bottom"]["sell_stop"])},
        "top": {"sell_limit": o(b["top"]["sell_limit"]),
                "buy_stop": o(b["top"]["buy_stop"])},
    }


def _reason_range_bracket(setup: dict, level: str, aligned: int, total: int) -> str:
    """震荡期双向 bracket 的中文原因文本。"""
    vr = setup.get("vol_ratio")
    if vr is not None:
        vol_txt = f"量能比 {vr:.2f}（明显缩小，确认缩量蓄力）"
    else:
        vol_txt = "量能（无成交量数据，未启用量能门控）"
    return (f"{level} 级别：确认处于箱体（效率比/净漂移/箱体高度/波动未放大达标），{vol_txt}，"
            f"判定为爆拉/暴跌「之前」的压缩蓄力。已双向限价挂单——"
            f"底部 buy limit（吃下沿回归，TP 箱体中枢）+ sell stop（跌破抓暴跌）；"
            f"顶部 sell limit（吃上沿回归）+ buy stop（涨破抓爆拉）。触发级别 {aligned}/{total}。")


def _make_range_bracket_signal(symbol: str, setup: dict, focus_tf: str | None,
                               total_levels: int, strategy: str,
                               all_setups: list[dict] | None = None) -> dict:
    """构造震荡期双向 bracket 的 model.signal 契约字典（signal_type=range_bracket）。

    direction=0（双向，无单一方向）；bracket 携带底部/顶部各两单的完整 entry/sl/tp。
    同时保留与旧渲染兼容的顶层 entry_zone/stop_loss/take_profit_*（取底部 buy limit 为代表）。
    """
    b = setup["bracket"]
    level = setup["level"]
    fired = sorted(
        set(s["level"] for s in (all_setups or [setup])),
        key=lambda l: RANGE_LEVEL_NAMES.index(l) if l in RANGE_LEVEL_NAMES else 99,
    )
    aligned = len(fired)
    resonance = round(aligned / total_levels * 100) if total_levels else 0
    lvl = "强" if resonance >= 70 else ("中" if resonance >= 40 else "弱")
    reason = _reason_range_bracket(setup, level, aligned, total_levels)

    bl = b["bottom"]["buy_limit"]
    atr = round(b["atr"], 6)
    rr = bl["rr"]
    entry_lo = round(bl["entry"] * (1 - 0.0005), 6)
    entry_hi = round(bl["entry"] * (1 + 0.0005), 6)

    price_zone = {
        "current": round(bl["entry"], 6),
        "entry_zone": [entry_lo, entry_hi],
        "stop_loss": round(bl["sl"], 6),
        "take_profit_1": round(bl["tp"], 6),
        "take_profit_2": round(bl["tp"], 6),
        "tp1_frac": 1.0, "be_price": None, "exit_plan": "bracket",
        "risk_reward": round(rr, 2) if rr is not None else None,
        "atr": atr, "atr_pct": round(atr / bl["entry"] * 100, 3) if bl["entry"] else 0.0,
        "entry_mode": "limit", "limit_price": round(bl["entry"], 6),
        "order_valid_min": None, "entry_desc": "震荡期箱体双向限价挂单",
        "recommendation": {
            "entry_zone": [entry_lo, entry_hi], "stop_loss": round(bl["sl"], 6),
            "take_profit_1": round(bl["tp"], 6), "take_profit_2": round(bl["tp"], 6),
            "tp1_frac": 1.0, "be_price": None, "exit_plan": "bracket",
            "risk_reward": round(rr, 2) if rr is not None else None,
        },
    }
    return {
        "symbol": symbol, "tf": focus_tf or level, "strategy": strategy,
        "market": "crypto", "ready": True,
        "signal_type": "range_bracket", "direction": 0,
        "resonance": resonance, "level": lvl,
        "aligned_count": aligned, "total_levels": total_levels,
        "levels_fired": fired, "level_dirs": {lv: 0 for lv in fired},
        # 箱体几何 + 量能诊断
        "box_low": round(b["box_low"], 6), "box_high": round(b["box_high"], 6),
        "box_mid": round(b["box_mid"], 6), "atr": atr,
        "vol_ratio": setup.get("vol_ratio"),
        "signal_bar_t": setup.get("bar_t"),
        "bracket": _round_bracket(b),
        # 兼容旧渲染字段（代表 = 底部 buy limit）
        "entry_zone": [entry_lo, entry_hi], "stop_loss": round(bl["sl"], 6),
        "take_profit_1": round(bl["tp"], 6), "take_profit_2": round(bl["tp"], 6),
        "tp1_frac": 1.0, "be_price": None, "exit_plan": "bracket",
        "risk_reward": round(rr, 2) if rr is not None else None,
        "entry_mode": "limit", "limit_price": round(bl["entry"], 6),
        "order_valid_min": None, "entry_desc": "震荡期箱体双向限价挂单",
        "reason": reason, "risk_disclaimer": RISK_DISCLAIMER,
        "price_zone": price_zone,
    }


# ----------------------------------------------------------------------------
# M顶空头独立信号源（mtop）—— 与 exp3 / violent / range 完全隔离的第四套管线
# 来源：quant_research 回测（M顶空头 + MACD/KDJ 共振，DEV-A→DEV-B→HOLD 三样本验证）
#   - 12h top 挂单 + F1/F2 共振 + SL=top×1.02：HOLD +4.08%, CI+0.44, n=963, 零爆仓
#   - 仅空头；不取 HTF 偏置、不要求多周期共振、不用 EMA20 拉回、不参与 exp3 执行。
# ----------------------------------------------------------------------------
async def analyze_mtop(symbol: str, rest, cache: MultiTFCache | None = None,
                       settings=None, tf: str | None = None, strategy: str = "mtop") -> dict:
    """M顶空头（挂单型）：对 12h/1d 扫描 M顶破位 + MACD/KDJ 共振，输出形态最高点 sell-limit 空单。

    与其余三套并列且互不耦合：只取交易级 K线，独立合成（_build_mtop_signal），
    不触碰 exp3 的 HTF 偏置 / EMA20 / V11 / ≥2/5 共振门控，也不影响 violent / range。
    无网络/部分失败时返回中性信号（direction=0, ready=True），不会抛异常。
    """
    cache = cache or MultiTFCache()
    symbol = symbol.upper()
    focus = bool(tf and tf in MTOP_LEVEL_NAMES)

    needed: dict[str, int] = {}
    level_plan: list[tuple[str, str, int, int]] = []  # (name, interval, tf_min, trade_limit)
    if focus:
        ti = tf
        tmin = INTERVAL_TO_MIN.get(ti, 60)
        tl = MTOP_TL.get(ti, 22)
        needed[ti] = tmin
        level_plan.append((ti, ti, tmin, tl))
    else:
        for name, tf_min, tl in MTOP_LEVELS:
            ti = MIN_TO_INTERVAL[tf_min]
            needed[ti] = tf_min
            level_plan.append((name, ti, tf_min, tl))

    async def _pull(iv):
        return iv, await _fetch_closed(rest, cache, symbol, iv, needed[iv], 400)

    results = dict(await asyncio.gather(*(_pull(iv) for iv in needed)))
    setups_by_level: list[dict] = []
    insufficient: list[str] = []
    for name, ti, tf_min, tl in level_plan:
        bars = results.get(ti, [])
        if len(bars) < 200:
            insufficient.append(name)
            continue
        try:
            su = signals_mtop.scan_level(
                bars, (name, tf_min, tl), signals_mtop.PARAMS,
                market="crypto", live=True,
            )
        except Exception:
            su = []
        for s in su:
            s["level"] = name
            if "bar_t" not in s and 0 <= s.get("idx", -1) < len(bars):
                s["bar_t"] = bars[s["idx"]]["t"]
            setups_by_level.append(s)

    return _build_mtop_signal(
        symbol, setups_by_level, focus_tf=(tf if focus else None),
        insufficient=insufficient, strategy=strategy,
    )


def _build_mtop_signal(symbol: str, setups: list[dict], focus_tf: str | None = None,
                       insufficient: list[str] | None = None, strategy: str = "mtop") -> dict:
    """M顶空头信号合成：任一级别触发即发布空头挂单；统计多少级别处于形态（排序用，不作门槛）。"""
    total = len(MTOP_LEVELS)
    if not setups:
        if insufficient:
            return _neutral(symbol, total,
                            note=f"数据不足：{'/'.join(insufficient)} 级别 K线历史不够，暂无法评估 M顶形态",
                            strategy=strategy)
        return _neutral(symbol, total,
                        note="当前无 M顶空头信号（未出现双峰等高形态 + MACD/KDJ 共振，或破位未确认）",
                        strategy=strategy)

    if focus_tf and focus_tf in MTOP_LEVEL_NAMES:
        at_focus = [s for s in setups if s["level"] == focus_tf]
        if not at_focus:
            return _neutral(symbol, total,
                            note=f"{focus_tf} 级别当前无 M顶空头触发（观望）", strategy=strategy)
        rep = at_focus[-1]  # 该级别最新触发
        return _make_mtop_signal(symbol, rep, focus_tf, total, strategy)

    setups_sorted = sorted(
        setups,
        key=lambda s: MTOP_LEVEL_NAMES.index(s["level"]) if s["level"] in MTOP_LEVEL_NAMES else 99,
    )
    rep = setups_sorted[0]
    return _make_mtop_signal(symbol, rep, rep["level"], total, strategy, all_setups=setups)


def _make_mtop_signal(symbol: str, setup: dict, focus_tf: str | None,
                      total_levels: int, strategy: str,
                      all_setups: list[dict] | None = None) -> dict:
    """构造 M顶空头的 model.signal 契约字典（signal_type=mtop_short，仅空头挂单）。

    entry=形态最高点（sell-limit 挂单点），sl=top×1.02（新高认错），tp=参考目标（建议持有 15 天）。
    direction=-1；level_dirs 仅统计 12h/1d 两级形态触发数。
    """
    level = setup["level"]
    fired = sorted(
        set(s["level"] for s in (all_setups or [setup])),
        key=lambda l: MTOP_LEVEL_NAMES.index(l) if l in MTOP_LEVEL_NAMES else 99,
    )
    aligned = len(fired)
    resonance = round(aligned / total_levels * 100) if total_levels else 0
    lvl = "强" if resonance >= 70 else ("中" if resonance >= 40 else "弱")

    entry = float(setup["entry"])
    sl = float(setup["sl"])
    tp = float(setup["tp"])
    atr = float(setup.get("atr") or 0.0)
    rr = setup.get("rr")
    limit_price = float(setup["limit_price"])
    band = max(entry * 0.0005, 1e-6)
    entry_lo = round(entry - band, 6)
    entry_hi = round(entry + band, 6)

    reason = _reason_mtop(setup, level, aligned, total_levels)
    price_zone = {
        "current": round(entry, 6),
        "entry_zone": [entry_lo, entry_hi],
        "stop_loss": round(sl, 6),
        "take_profit_1": round(tp, 6),
        "take_profit_2": round(tp, 6),
        "tp1_frac": 1.0, "be_price": None, "exit_plan": "mtop_hold15",
        "risk_reward": round(rr, 2) if rr is not None else None,
        "atr": round(atr, 6),
        "atr_pct": round(atr / entry * 100, 3) if entry else 0.0,
        "entry_mode": "limit", "limit_price": round(limit_price, 6),
        "order_valid_min": None, "entry_desc": "M顶最高点挂空（最尖尖，回抽成交）",
        "recommendation": {
            "entry_zone": [entry_lo, entry_hi], "stop_loss": round(sl, 6),
            "take_profit_1": round(tp, 6), "take_profit_2": round(tp, 6),
            "tp1_frac": 1.0, "be_price": None, "exit_plan": "mtop_hold15",
            "risk_reward": round(rr, 2) if rr is not None else None,
        },
    }
    return {
        "symbol": symbol, "tf": focus_tf or level, "strategy": strategy,
        "market": "crypto", "ready": True,
        "signal_type": "mtop_short", "direction": -1,
        "resonance": resonance, "level": lvl,
        "aligned_count": aligned, "total_levels": total_levels,
        "levels_fired": fired, "level_dirs": {lv: -1 for lv in fired},
        "signal_bar_t": setup.get("bar_t"),
        # 形态诊断
        "mtop_top": round(float(setup.get("top") or entry), 6),
        "mtop_neck": round(float(setup.get("vk") or 0.0), 6),
        # 兼容旧渲染字段（单边空头）
        "entry_zone": [entry_lo, entry_hi], "stop_loss": round(sl, 6),
        "take_profit_1": round(tp, 6), "take_profit_2": round(tp, 6),
        "tp1_frac": 1.0, "be_price": None, "exit_plan": "mtop_hold15",
        "risk_reward": round(rr, 2) if rr is not None else None,
        "entry_mode": "limit", "limit_price": round(limit_price, 6),
        "order_valid_min": None, "entry_desc": "M顶最高点挂空（最尖尖，回抽成交）",
        "reason": reason, "risk_disclaimer": RISK_DISCLAIMER,
        "price_zone": price_zone,
    }


def _reason_mtop(setup: dict, level: str, aligned: int, total: int) -> str:
    """M顶空头中文原因文本（注明已验证执行语义）。"""
    top = float(setup.get("top") or setup["entry"])
    vk = float(setup.get("vk") or 0.0)
    sl = float(setup["sl"])
    return (
        f"{level} 级别：M顶确认（双峰近似等高 ±3%，颈线 {vk:.6g}，收盘跌破后回抽），"
        f"MACD 线下降 + KDJ 拐头向下（F1 共振过滤，回测三样本验证）→ "
        f"形态最高点 {top:.6g} 挂空（最尖尖），止损 {sl:.6g}（突破新高认错，零爆仓），"
        f"参考目标 {float(setup['tp']):.6g}；已验证最优执行为持有 ~15 天中期空单（L2 年化 ~27-99%）。"
        f"触发级别 {aligned}/{total}。"
    )


def MIN_TO_INTERVAL_REVERSE(iv: str) -> int:
    for m, v in MIN_TO_INTERVAL.items():
        if v == iv:
            return m
    return 60


def _trade_limit_for(name: str) -> int:
    for n, _t, _b, tl in LEVELS_CRYPTO:
        if n == name:
            return tl
    return 30


def _pick_exp4_anchor(setups: list[dict]):
    """EXP4-S：按回测 grade_of 口径挑信号主体（对齐级别数最多，同分取最近触发）。

    与 exp3 的「全局 latest 锚点」唯一差别是把**每个 setup 都当作候选主体**，
    这样恢复的 1h 才能作为主体之一出现（1h 触发 + 4h/6h 在 ±4h/±6h 内同向 = 2 级共振）。

    两道防陈旧保护（对应 2026-09-02 的「跨月累计」修复）：
      1) 候选主体必须落在「全级别最近触发 −12h」之内，杜绝数天前的旧 setup 抢锚点；
      2) 同分时取最近触发，保证发布的是当前而非历史信号。
    返回 (anchor, matched)；无合格候选时返回 (None, [])。
    """
    if not setups:
        return None, []

    def _cluster(s):
        st = s.get("bar_t") or 0
        return [x for x in setups
                if x["dir"] == s["dir"]
                and abs((x.get("bar_t") or 0) - st) <= _ALIGN_WIN_MS.get(x["level"], 12 * 3600 * 1000)]

    # 候选 A：与 exp3 **完全同口径**的锚点（最近的非下线级别 4h/6h/12h）——保证 exp4 信号集 ⊇ exp3。
    pool = [s for s in setups if s.get("level") not in RETIRED_TFS] or setups
    a = max(pool, key=lambda s: s.get("bar_t", 0) or 0)
    ca = _cluster(a)
    na = len(set(x["level"] for x in ca))

    # 候选 B：1h 锚点——**仅当 1h 是全局最新触发**时才考虑，保证发布的是「此刻的新信号」，
    # 而不是翻出几小时前的旧 1h 触发来抢方向（对应 2026-09-02 跨月累计修复的同源风险）。
    h1 = [s for s in setups if s.get("level") == "1h"]
    b = None
    if h1:
        h = max(h1, key=lambda s: s.get("bar_t", 0) or 0)
        if (h.get("bar_t") or 0) > (a.get("bar_t") or 0):
            b = h
    if b is None:
        return a, ca
    cb = _cluster(b)
    nb = len(set(x["level"] for x in cb))
    # 同分时优先 A（4h+ 主导方向），只有 1h 簇**严格更优**才采用 —— 净效果：只增不减。
    return (b, cb) if nb > na else (a, ca)


def _build_signal(symbol: str, setups: list[dict], focus_tf: str | None = None,
                  insufficient: list[str] | None = None, strategy: str = "exp3") -> dict:
    """把多级别 setup 汇聚为 model.signal 契约字典。

    focus_tf：用户所选时间线。传入时只认该级别的 setup（"选 X 就是 X 推荐"）；
    该级别已被其 HTF 偏置确认（scan_level 内部校验），故视为强确认。
    不传则聚合全部级别，代表性级别=最低触发级，共振=对齐级数/总级数。
    insufficient：因 K线历史不足被跳过的级别名——用于把"数据不足"与"无信号"区分开。
    """
    # EXP4-S 级别列表不含 1w（92 天窗口 0 setup 死级别）→ 分母按策略动态取，避免显示 "2/5" 误导（实际只有 4 级参与）。
    total_levels = len(EXP4S_LEVELS) if strategy == "exp4" else len(LEVELS_CRYPTO)

    # 【纪律】下线周期门控（focus 分支此前漏检，与聚合口径不一致）：
    # exp3：1h 及以下已下线（毛边缘 0.073% < 往返成本 0.2%，taker 不划算）；
    # exp4：恢复 1h（EXP4-S 扩容），但 5m/15m/30m 两套一律停用。
    # 复现实证（2026-09-09）：/api/analyze?tf=1h&strategy=exp3 曾输出
    # 「1h 强信号 · aligned=1/5 · bar_t=今晨」——1h 单级触发绕过下线+1/5停用两条纪律。
    # 放在 not-setups 提前返回之前：即使该级别当前无触发，也应返回明确的「已下线」语义。
    if (focus_tf and focus_tf in FOCUS_BIAS and strategy in ("exp3", "exp4")
            and focus_tf in RETIRED_TFS and not (strategy == "exp4" and focus_tf == "1h")):
        if focus_tf == "1h":
            note = (f"1h 级别信号已下线：exp3 仅做 4H+（1h 毛边缘 < 往返成本）；"
                    f"1H 信号请切换 EXP4-S")
        else:
            note = (f"{focus_tf} 级别信号已下线（仅 4H+ 与 EXP4-S 的 1H 参与交易；"
                    f"更短周期毛边缘 < 往返成本，已停用）")
        n = _neutral(symbol, total_levels, note=note, strategy=strategy)
        n["tf"] = focus_tf
        return n

    if not setups:
        if insufficient:
            return _neutral(symbol, total_levels,
                            note=f"数据不足：{'/'.join(insufficient)} 级别 K线历史不够（新币或数据缺口），暂无法评估",
                            strategy=strategy)
        return _neutral(symbol, total_levels, strategy=strategy)

    if focus_tf and focus_tf in FOCUS_BIAS:
        at_focus = [s for s in setups if s["level"] == focus_tf]
        if not at_focus:
            if insufficient and focus_tf in insufficient:
                note = f"{focus_tf} 级别数据不足（K线历史不够，新币或数据缺口），暂无法评估"
            else:
                note = f"{focus_tf} 级别当前无 MTF-TAMR 触发信号（观望）"
            n = _neutral(symbol, total_levels, note=note, strategy=strategy)
            n["tf"] = focus_tf
            return n
        # focus 模式（2026-09-02 修复）：以该级别「最近一次触发」为准，
        # 不再把整个数据窗的历史触发跨月累计投票——旧逻辑会让数周前的
        # 反向触发翻转方向（如 8 月空单在 9 月现价 7.6 万时仍发布做空）。
        latest = max(at_focus, key=lambda s: s.get("bar_t", 0) or 0)
        direction = int(latest["dir"])
        lt = latest.get("bar_t") or 0
        matched = [s for s in at_focus if s["dir"] == direction
                   and (not lt or abs((s.get("bar_t") or 0) - lt) <= _ALIGN_WIN_MS.get(focus_tf, 12 * 3600 * 1000))]
        if not matched:
            matched = [latest]
        resonance = 100  # 该级别已由 HTF 偏置确认
        level = "强"
        reason = _reason_text(direction, focus_tf, FOCUS_BIAS[focus_tf], None, None, None, None,
                              strategy=strategy)
        return _make_signal(symbol, direction, matched, resonance, level, focus_tf, reason,
                            level_dirs={s["level"]: s["dir"] for s in setups}, strategy=strategy)

    # ---- 聚合模式（无 focus）：以最近一次触发为主（live 发布语义） ----
    # 2026-09-02 修复：旧实现把整个数据窗内全部历史触发跨月累计投票，
    # 方向被数周前的旧触发污染（如 8 月空单在 9 月 7.6 万现价仍发布做空），
    # 评级也被跨月触发虚增。现改为「最新触发主导 + 时间窗对齐」——
    # 与回测 grade_of 的 ALIGN_WIN 口径一致：只统计触发时刻 ±1 根 bar 的同向级别。
    # 2026-09-03 再修：latest 锚点只从「非 RETIRED 级别」（4h/6h/12h/1d/1w）选——
    # 1h 已下线但仍在 setups 里，若 latest 锚到 1h 触发会掩盖 4h 信号（方向被 1h 主导）。
    valid_setups = [s for s in setups if s.get("level") not in RETIRED_TFS]
    # ---- EXP4-S 专用：1h 参与共振的锚点选取（回测 grade_of 口径）----
    # 客户端原模型是「每币只发最新一条」：以全局 latest 为锚，各级别按自身 ±1 根 bar 对齐。
    # 该模型下 1h 几乎进不了对齐集（1h 触发常早于 4h 锚点 2~3 根 bar，超出 ±1h 窗口），
    # 恢复 1h 后信号数几乎不变，EXP4-S 的 +53% 会完全落空。
    # 回测口径（_bt_1h_new30.grade_of）是「每个 setup 都是候选主体，各自独立评级」——
    # 故 exp4 单独改为：在所有 setup 中挑**对齐级别数最多**的作为主体，同分取最近触发。
    # 【隔离红线】仅 strategy == "exp4" 走此分支，exp3 的 latest 锚点行为逐字不变。
    if strategy == "exp4":
        anchor, matched = _pick_exp4_anchor(setups)
        if anchor is None:
            return _neutral(symbol, total_levels, strategy=strategy)
        direction = int(anchor["dir"])
        aligned_levels = len(set(s["level"] for s in matched))
    else:
        latest = max(valid_setups or setups, key=lambda s: s.get("bar_t", 0) or 0)
        direction = int(latest["dir"])
        lt = latest.get("bar_t") or 0
        matched = [s for s in setups if s["dir"] == direction
                   and (not lt or abs((s.get("bar_t") or 0) - lt) <= _ALIGN_WIN_MS.get(s["level"], 12 * 3600 * 1000))]
        if not matched:
            matched = [latest]
        aligned_levels = len(set(s["level"] for s in matched))
    # 等级过滤（2026-08-31 采纳：只信 2/5 及以上；1/5 级单周期共振已停用——
    # 回测显示 1/5 级在 taker 成本下无正期望，且 DEV/HOLD 一致性差属噪声特征）
    # 【隔离红线】该过滤仅作用于 exp3 / exp4：violent（暴力拉升/暴跌）与 range（震荡期，独立
    # analyze_range 管线）各有独立信号语义，不得被共振评级口径拦截——各选项卡逻辑与代码保持隔离。
    # exp4 与 exp3 同门槛（≥2），这是 A/B 成立的前提：不放开则 exp4 跑到无门槛劣质池
    # （已证实同配置 PF 从 1.172 掉到 0.930）；门槛值与 exp3 严格相同。
    if strategy in ("exp3", "exp4") and aligned_levels < 2:
        return _neutral(symbol, total_levels,
                        note="当前仅 1/5 级共振（单周期触发）——1/5 级信号已停用，需 ≥2/5 级多周期共振才发布",
                        strategy=strategy)
    ratio = aligned_levels / total_levels if total_levels else 0.0
    resonance = round(ratio * 100)
    level = "强" if resonance >= 70 else ("中" if resonance >= 40 else "弱")
    reason = _reason_text(direction, None, None, None, None, None, None,
                          aligned_levels=aligned_levels, total_levels=total_levels,
                          strategy=strategy)
    return _make_signal(symbol, direction, matched, resonance, level, None, reason,
                        level_dirs={s["level"]: s["dir"] for s in setups}, strategy=strategy)


def _reason_text(direction, level, bias_level, entry, tp, sl, rr,
                 aligned_levels=None, total_levels=None, strategy="exp3") -> str:
    """生成中文'原因'文本（极简信号卡的唯一解释字段）。

    exp3 与 violent 的语义完全不同，文案必须分开：
      exp3    = 趋势内回调 → 均值回归（价格回到 EMA20 下方/上方）
      violent = 大阳/大阴【启动前】埋伏（价格已离开均线、区间收敛蓄力、大实体尚未出现）
    """
    prefix = f"{level} 级别：" if level else ""

    if strategy == "range":
        # 震荡期：箱体边界均值回归（与 exp3 的 EMA20 拉回、violent 的启动前埋伏均不同）
        if direction > 0:
            txt = (f"{prefix}价格回落至箱体下沿附近（震荡期确认、近期无拉升），"
                   f"于下沿做均值回归多；止盈看箱体中枢，跌破下沿止损。")
        elif direction < 0:
            txt = (f"{prefix}价格反弹至箱体上沿附近（震荡期确认、近期无暴跌），"
                   f"于上沿做均值回归空；止盈看箱体中枢，涨破上沿止损。")
        else:
            txt = ""
        if aligned_levels is not None and total_levels:
            txt += f" 触发级别 {aligned_levels}/{total_levels}。"
        return txt

    if strategy == "violent":
        if direction > 0:
            txt = (f"{prefix}价格沿趋势方向偏离 EMA20 且区间收敛蓄力，"
                   f"大阳线尚未出现 → 启动前埋伏做多。")
        elif direction < 0:
            txt = (f"{prefix}价格沿趋势方向偏离 EMA20 且区间收敛蓄力，"
                   f"大阴线尚未出现 → 启动前埋伏做空。")
        else:
            txt = ""
        if aligned_levels is not None and total_levels:
            txt += f" 多周期对齐 {aligned_levels}/{total_levels} 级。"
        return txt

    if direction > 0:
        if bias_level:
            txt = (f"{prefix}价格回调至 EMA20 下方、位于布林中轨下方（温和超卖，RSI≤45），"
                   f"且大周期({bias_level})趋势偏置向上 → 做多。")
        else:
            txt = (f"{prefix}价格回调至 EMA20 下方（温和超卖，RSI≤45），"
                   f"多周期趋势偏置向上 → 做多。")
    elif direction < 0:
        if bias_level:
            txt = (f"{prefix}价格反弹至 EMA20 上方、位于布林中轨上方（温和超买，RSI≥55），"
                   f"且大周期({bias_level})趋势偏置向下 → 做空。")
        else:
            txt = (f"{prefix}价格反弹至 EMA20 上方（温和超买，RSI≥55），"
                   f"多周期趋势偏置向下 → 做空。")
    else:
        txt = ""
    if aligned_levels is not None and total_levels:
        txt += f" 多周期对齐 {aligned_levels}/{total_levels} 级。"
    return txt


def _make_signal(symbol, direction, matched, resonance, level, focus_tf, reason,
                 level_dirs: dict | None = None, strategy: str = "exp3") -> dict:
    """按 direction + rep setup 构造 model.signal 契约字典（纯 exp3：方向/原因/价区）。

    只保留前端实际消费的字段，移除全部四类共振遗留字段
    （confidence/resonance_level/signal_strength/cross_check/notes/regime/veto/agreement/tech_score/cat_*/indicators/price）。
    """
    # 代表 setup：级别优先（低周期在前），同级别内取**最新**触发（-idx）——
    # 旧实现取同级别最早触发，展示的入场/止损/止盈可能是几十根 bar 前的陈旧参数。
    matched = sorted(matched, key=lambda s: (
        LEVEL_PRIORITY.index(s["level"]) if s["level"] in LEVEL_PRIORITY else 99,
        -s.get("idx", 0),
    ))
    # 【exp3 专属】代表级别跳过已下线周期（1h 及以下已停用，其入场参数不展示、不生成挂单）：
    # 保证 4h 聚焦信号以 4h 级别参数为代表（与回测口径一致，k=0.25 挂单正常生效）。
    # 仅改 rep 选取，不影响 matched/aligned_count（等级显示仍按全部触发级别）。
    rep = matched[0]
    if strategy in ("exp3", "exp4"):
        _rep = next((s for s in matched if s["level"] not in RETIRED_TFS), None)
        if _rep is not None:
            rep = _rep
    entry = float(rep["entry"])
    tp = float(rep["tp"])
    sl = float(rep["sl"])
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = round(reward / risk, 2) if risk > 1e-9 else None
    atr = rep.get("atr") or (round(risk / PARAMS["sl_mult"], 8) if risk > 0 else 0.0)

    # V11 分批止盈结构（exp3 已验证；violent/预留系统无此结构时退化为单止盈）
    # 【隔离红线】V11 字段只在 exp3 生效：即便暴力/其他系统 setup 意外携带 exit_plan/tp1_price 等
    # 字段，也强制退化为 single——从契约层保证两套选项卡逻辑与代码完全隔离。
    exit_plan = rep.get("exit_plan") or "single"
    if strategy != "exp3":
        exit_plan = "single"
    tp1 = float(rep.get("tp1_price") or tp)   # 首批止盈（+1R，平 tp1_frac 仓位）
    tp2 = float(rep.get("tp") or tp)          # 剩余止盈（EMA20）
    be_price = rep.get("be_price")
    tp1_frac = float(rep.get("tp1_frac") or 1.0)
    if exit_plan != "v11":
        tp1 = tp2 = tp
        tp1_frac = 1.0
        be_price = None
    # 入场模式：exp3·4h 为限价挂单留容错（k=0.25×ATR，8h 超时市价回退）；其余 close 入场。
    # 【隔离红线】band 挂单只在 exp3 / exp4 生效：其余策略即使 setup 意外携带 band 字段也强制 close。
    # exp4 与 exp3 同为 k=0.25×ATR 限价挂单——不放开会让 exp4 退化成 close 市价入场，
    # 与 exp3 差出「入场价 + 出场方式」两个变量，A/B 结果无法归因。
    bnd = (rep.get("band") or {}) if strategy in ("exp3", "exp4") else {}
    entry_mode = "band" if bnd.get("limit_price") else "close"
    limit_price = float(bnd["limit_price"]) if bnd.get("limit_price") is not None else None
    order_valid_min = int(bnd.get("order_valid_min") or ORDER_WAIT_MIN) if entry_mode == "band" else None
    entry_desc = bnd.get("entry_desc") if entry_mode == "band" else None

    gap_up = abs(tp2 - entry)
    gap_dn = abs(entry - sl)
    band = 0.15 * min(risk, gap_up, gap_dn) if (risk > 0 and gap_up > 0 and gap_dn > 0) else entry * 0.001
    entry_lo = round(entry - band, 6)
    entry_hi = round(entry + band, 6)
    entry_lo = max(entry_lo, round(sl, 6))
    entry_hi = min(entry_hi, round(tp2, 6))
    if entry_hi <= entry_lo:
        entry_lo = round(entry - max(entry * 1e-4, 1e-6), 6)
        entry_hi = round(entry + max(entry * 1e-4, 1e-6), 6)
    if entry_mode == "band":
        # 限价挂单：推荐入场 = 精确挂单价（等待回踩/反弹触达成交），entry_zone 收敛为单点
        entry_lo = entry_hi = limit_price

    recommendation = {
        "entry_zone": [entry_lo, entry_hi],
        "stop_loss": round(sl, 6),
        "take_profit_1": round(tp1, 6),          # 首批止盈（+1R，平 tp1_frac 仓位）
        "take_profit_2": round(tp2, 6),          # 剩余止盈（EMA20）
        "tp1_frac": tp1_frac,
        "be_price": round(be_price, 6) if be_price is not None else None,
        "exit_plan": exit_plan,
        "risk_reward": rr,
    }
    price_zone = {
        "current": round(entry, 6),
        "entry_zone": [entry_lo, entry_hi],
        "stop_loss": round(sl, 6),
        "take_profit_1": round(tp1, 6),
        "take_profit_2": round(tp2, 6),
        "tp1_frac": tp1_frac,
        "be_price": round(be_price, 6) if be_price is not None else None,
        "exit_plan": exit_plan,
        "risk_reward": rr,
        "atr": atr,
        "atr_pct": round(atr / entry * 100, 3) if entry else 0.0,
        "entry_mode": entry_mode,
        "limit_price": limit_price,
        "order_valid_min": order_valid_min,
        "entry_desc": entry_desc,
        "recommendation": recommendation,
    }

    # 多周期对齐信息（纯 exp3 语义：对齐级别数 / 总级别数）
    # EXP4-S 级别列表不含 1w（死级别）→ 分母按策略动态取，避免显示 "2/5" 误导（实际只有 4 级参与）。
    levels_fired = sorted(set(s["level"] for s in matched))
    total_levels = len(EXP4S_LEVELS) if strategy == "exp4" else len(LEVEL_CRYPTO_NAMES)
    aligned_count = len(levels_fired)
    # 各级别方向明细（增量元数据，由 _build_signal 从全部 setups 计算传入，
    # 含多空双方——聚合方向与扫描方向不一致时也能如实统计支持任一方向的基础级数；
    # 不影响任何信号判定——entry/tp/sl/direction 完全不变）。
    level_dirs = level_dirs or {s["level"]: s["dir"] for s in matched}

    # V11 出场结构说明（exp3 已验证方案；violent 保持原 reason）
    final_reason = reason
    if exit_plan == "v11":
        final_reason = (
            f"{reason} 出场=已验证 V11：首批 {int(tp1_frac * 100)}% @ +1R({round(tp1, 6)}) 锁盈，"
            f"触及后剩余仓位止损移至保本价({round(be_price, 6) if be_price is not None else entry})；"
            f"剩余 {int((1 - tp1_frac) * 100)}% 目标 EMA20({round(tp2, 6)})；止损 {V11_SL_MULT}×ATR。"
        )
        if entry_mode == "band" and limit_price:
            final_reason += (
                f" 入场={entry_desc}限价挂单 @ {round(limit_price, 6)}"
                f"（{order_valid_min}min 内触达成交，超时回退市价）。"
            )

    return {
        "symbol": symbol,
        "tf": focus_tf or rep["level"],
        "strategy": strategy,          # 信号归属策略（exp3 / violent / range 预留；执行层按此门控）
        "market": "crypto",
        "ready": True,
        "direction": direction,
        "resonance": resonance,           # 对齐百分比（0~100），供扫盘排序用
        "level": level,                    # 强/中/弱（基于对齐比例）
        "aligned_count": aligned_count,    # 对齐级别数
        "total_levels": total_levels,      # 总级别数
        "levels_fired": levels_fired,      # 触发级别名列表
        "level_dirs": level_dirs,          # {级别名: 方向}（全部 setups，含多空双方）
        "signal_bar_t": rep.get("bar_t"),  # 代表信号触发 bar 时间（ms；时效展示用）
        # 价区（核心）
        "entry_zone": [entry_lo, entry_hi],
        "stop_loss": round(sl, 6),
        "take_profit_1": round(tp1, 6),          # 首批止盈（+1R，平 tp1_frac 仓位）
        "take_profit_2": round(tp2, 6),          # 剩余止盈（EMA20）
        "tp1_frac": tp1_frac,
        "be_price": round(be_price, 6) if be_price is not None else None,
        "exit_plan": exit_plan,
        "risk_reward": rr,
        # 入场模式（V11 统一 close；band 挂单已移除）
        "entry_mode": entry_mode,
        "limit_price": limit_price,
        "order_valid_min": order_valid_min,
        "entry_desc": entry_desc,
        "reason": final_reason,
        "risk_disclaimer": RISK_DISCLAIMER,
        "price_zone": price_zone,
    }


def _neutral(symbol: str, total_levels: int, note: str = "无多周期对齐信号",
             strategy: str = "exp3") -> dict:
    """中性信号（direction=0，观望）。纯字段，无四类共振遗留。strategy 透传。"""
    return {
        "symbol": symbol,
        "tf": "crypto",
        "strategy": strategy,
        "market": "crypto",
        "ready": True,
        "direction": 0,
        "resonance": 0,
        "level": "观望",
        "aligned_count": 0,
        "total_levels": total_levels,
        "levels_fired": [],
        "level_dirs": {},
        "signal_bar_t": None,
        "entry_zone": None,
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "tp1_frac": None,
        "be_price": None,
        "exit_plan": None,
        "risk_reward": None,
        "reason": note,
        "risk_disclaimer": RISK_DISCLAIMER,
        "price_zone": {
            "current": None, "entry_zone": None, "stop_loss": None,
            "take_profit_1": None, "take_profit_2": None, "tp1_frac": None,
            "be_price": None, "exit_plan": None, "risk_reward": None,
            "atr": None, "atr_pct": None, "entry_mode": None, "limit_price": None,
            "order_valid_min": None, "entry_desc": None, "recommendation": {
                "entry_zone": None, "stop_loss": None,
                "take_profit_1": None, "take_profit_2": None, "risk_reward": None,
            },
        },
    }


async def scan_universe(rest, cache: MultiTFCache | None, symbols: list[str],
                        settings=None, limit: int = 30, tf: str | None = None,
                        strategy: str = "exp3") -> list[dict]:
    """并发扫描给定宇宙，返回扫盘行（仅含 direction!=0 的可执行信号）。

    tf：扫描周期（用户所选时间线）。strategy：信号系统（exp3/violent/range）。
    推荐等级（aligned_count/total_levels/levels_fired）取自**全级别聚合**结果中
    与焦点方向一致的基础级数（1~5 级推荐），而非焦点模式恒定的 1/5——
    使扫盘列表能真实区分多周期共识强度。两次分析共享 MultiTFCache，
    首个周期后 K线命中 TTL 缓存，成本可控。
    """
    cache = cache or MultiTFCache()
    syms = [s.upper() for s in symbols][:limit]

    # 并发闸门（2026-09-03）：全市场扫描若不限流，limit×~8 个 K线请求瞬间并发，
    # 极易触发币安 418 IP 封禁（共享 VPN 出口 IP 常被预限流，叠加后冷启动全卡死）。
    # 4 并发 ≈ 峰值 ~32 个在途请求，冷启动稍慢但稳定；命中 TTL 缓存后近乎零成本。
    # 仅限流，不改任何信号计算逻辑（exp3 核心不动）。
    sem = asyncio.Semaphore(4)

    async def _gated(sym):
        async with sem:
            return await _one(sym)

    async def _one(sym):
        try:
            focus_sig, agg_sig = await asyncio.gather(
                analyze_symbol(sym, rest, cache, settings, tf=tf, strategy=strategy),
                analyze_symbol(sym, rest, cache, settings, tf=None, strategy=strategy),
            )
            d = focus_sig.get("direction", 0)
            if d == 0 and focus_sig.get("signal_type") != "range_bracket":
                return None
            # 推荐等级 = 聚合 5 基础级中与焦点方向一致的级别数
            level_dirs = agg_sig.get("level_dirs") or {}
            same_levels = sorted(
                [lv for lv, dd in level_dirs.items() if dd == d],
                key=lambda l: LEVEL_PRIORITY.index(l) if l in LEVEL_PRIORITY else 99,
            )
            aligned = len(same_levels)
            total = agg_sig.get("total_levels") or len(LEVELS_CRYPTO)
            resonance = round(aligned / total * 100) if total else 0
            return {
                "symbol": sym,
                "name": focus_sig.get("symbol", sym),
                "direction": d,
                "resonance": resonance,
                "level": "强" if resonance >= 70 else ("中" if resonance >= 40 else "弱"),
                "aligned_count": aligned,
                "total_levels": total,
                "levels_fired": same_levels,
                "entry_zone": focus_sig.get("entry_zone"),
                "stop_loss": focus_sig.get("stop_loss"),
                "take_profit_1": focus_sig.get("take_profit_1"),
                "risk_reward": focus_sig.get("risk_reward"),
                "entry_mode": focus_sig.get("entry_mode"),
                "limit_price": focus_sig.get("limit_price"),
                "order_valid_min": focus_sig.get("order_valid_min"),
                "entry_desc": focus_sig.get("entry_desc"),
                "reason": focus_sig.get("reason"),
                "signal_bar_t": focus_sig.get("signal_bar_t"),
                "signal_type": focus_sig.get("signal_type"),
                "bi_dir": focus_sig.get("signal_type") == "range_bracket",
            }
        except Exception:
            return None

    rows = await asyncio.gather(*(_gated(s) for s in syms))
    rows = [r for r in rows if r]
    # 等级过滤：exp3 / exp4 只保留 ≥2 级多周期共振（1 级单周期共振已停用）；
    # violent/预留系统保留自身语义，不受此过滤影响。
    if strategy in ("exp3", "exp4"):
        rows = [r for r in rows if (r.get("aligned_count") or 0) >= 2]
    # exp3 排序：推荐等级(多周期对齐数,主要) → 盈亏比(次要)，高等级推荐优先
    rows.sort(key=lambda x: (x.get("aligned_count") or 0, x.get("risk_reward") or 0), reverse=True)
    return rows
