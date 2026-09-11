#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""crypto-quant 云端信号检查器（GitHub Actions 定时运行）

流程：拉取币安K线 → MTF-TAMR 多周期共振（与本地客户端同一份算法）
      → 发现新信号 → 邮件通知 → 去重状态写入 state/notified.json

与客户端「4小时」页签严格对齐：
  - 仅 EXP3（趋势内回调均值回归）与 EXP4-S（一次性EMA20止盈+1h扩容）两套策略
  - 聚焦 4h 级别（交易级 4h + 偏置级 12h），复用 mtf_tamr.scan_universe(tf="4h")
  - 扫盘币池 = 全市场 USDT 永续（按 24h 成交额降序），与原客户端 _scan_pool 一致

环境变量（必填；密钥只走环境变量/Secrets，不进代码）：
  MAIL_HOST      SMTP 服务器（QQ邮箱：smtp.qq.com）
  MAIL_PORT      SMTP 端口（QQ邮箱 SSL：465）
  MAIL_USER      发件邮箱
  MAIL_PASS      SMTP 授权码（不是QQ登录密码）
  MAIL_TO        收件邮箱，多个用逗号分隔
  WATCH_SYMBOLS  监控币种，逗号分隔（默认 BTC/ETH/SOL/XRP/BNB/DOGE/ADA/LTC/AVAX/LINK）

用法：
  python cloud_signal_checker.py              # 正常检查一轮
  python cloud_signal_checker.py --test-mail  # 发一封测试邮件，验证SMTP配置
"""
import asyncio
import json
import os
import smtplib
import sys
import time
from datetime import datetime, timedelta, timezone
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "quant_research"))

STATE_FILE = ROOT / "state" / "notified.json"
STATE_MAX = 500  # 全市场扫描下保留更多指纹，防止状态文件过快增长

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,DOGEUSDT,ADAUSDT,LTCUSDT,AVAXUSDT,LINKUSDT"

# 与客户端「4小时」页签对齐：仅两套策略 + 聚焦 4h（偏置 12h）
STRATEGIES = ("exp3", "exp4")
FOCUS_TF = "4h"
STRATEGY_LABEL = {"exp3": "EXP3", "exp4": "EXP4-S"}

CST = timezone(timedelta(hours=8))


class RateLimiter:
    """简单令牌桶限流：保护币安 REST 配额（约 15 req/s = 1800 weight/min < 2400 上限）。"""

    def __init__(self, rate: float = 15.0, burst: int = 30) -> None:
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.updated = time.monotonic()

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            await asyncio.sleep(0.05)


class CloudRest:
    """极简币安 REST 客户端（K线 + 全市场 24h 快照；GitHub 海外 runner 直连，无需代理）。"""

    BASE = "https://fapi.binance.com"

    def __init__(self) -> None:
        self._session = requests.Session()
        self._limiter = RateLimiter()

    async def _get(self, path: str, params: dict | None = None, retries: int = 3) -> list | dict:
        url = f"{self.BASE}{path}"
        for attempt in range(retries):
            await self._limiter.acquire()
            try:
                resp = self._session.get(url, params=params, timeout=25)
                if resp.status_code in (418, 429):
                    wait = 5 * (attempt + 1)
                    print(f"  [rest] {resp.status_code} 限流，等待 {wait}s 重试")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as exc:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
        raise RuntimeError(f"GET {path} 失败")

    async def klines(self, symbol: str, interval: str = "5m", limit: int = 500,
                     end_ts: int | None = None) -> list[dict]:
        params = {"symbol": symbol.upper(), "interval": interval, "limit": min(int(limit), 1000)}
        if end_ts:
            params["endTime"] = int(end_ts)
        raw = await self._get("/fapi/v1/klines", params)
        return [
            {"t": int(k[0]), "o": float(k[1]), "h": float(k[2]), "l": float(k[3]),
             "c": float(k[4]), "v": float(k[5]), "close_ts": int(k[6]), "is_closed": True}
            for k in raw
        ]

    async def market_pool(self) -> list[str]:
        """全市场 USDT 永续，按 24h 成交额降序（与原客户端 _scan_pool 的 REST 分支一致）。"""
        data = await self._get("/fapi/v1/ticker/24hr")
        arr = sorted(
            (t for t in data if str(t.get("symbol", "")).endswith("USDT")
             and float(t.get("quoteVolume", 0) or 0) > 0),
            key=lambda t: float(t.get("quoteVolume", 0) or 0), reverse=True,
        )
        pool = [t["symbol"] for t in arr]
        print(f"[checker] 全市场 USDT 永续币池: {len(pool)} 个")
        return pool


# ---------------- 去重状态 ----------------

def load_state() -> set[str]:
    if STATE_FILE.exists():
        try:
            return set(json.loads(STATE_FILE.read_text(encoding="utf-8")).get("notified", []))
        except Exception:
            pass
    return set()


def save_state(fps: set[str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"updated": datetime.now(CST).isoformat(), "notified": sorted(fps)[-STATE_MAX:]}
    STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprint(sig: dict) -> str:
    """信号指纹：策略 + 币种 + 方向 + 触发bar时间 + 触发级别集。同一信号重复扫描不会重复通知。"""
    bar_t = sig.get("signal_bar_t") or 0
    lvs = ",".join(sorted(sig.get("levels_fired") or []))
    return f"{sig.get('strategy')}|{sig['symbol']}|{sig.get('direction')}|{bar_t}|{lvs}"


# ---------------- 邮件 ----------------

def fmt_signal(sig: dict) -> str:
    side = "做多" if sig.get("direction", 0) > 0 else "做空"
    ez = sig.get("entry_zone") or ["-", "-"]
    bt = sig.get("signal_bar_t")
    t = datetime.fromtimestamp(bt / 1000, CST).strftime("%m-%d %H:%M") if bt else "-"
    label = STRATEGY_LABEL.get(sig.get("strategy"), sig.get("strategy", "?"))
    return (
        f"■ [{label}] {sig['symbol']}  {side}   ·   共振 {sig.get('aligned_count')}/{sig.get('total_levels')} 级"
        f"（{sig.get('level')}）\n"
        f"    触发时间: {t}    触发级别: {','.join(sig.get('levels_fired') or ['-'])}\n"
        f"    入场区间: {ez[0]} ~ {ez[1]}    止损: {sig.get('stop_loss')}    止盈1: {sig.get('take_profit_1')}"
        f"    盈亏比: {sig.get('risk_reward')}\n"
        f"    原因: {sig.get('reason', '')}"
    )


def send_mail(subject: str, body: str) -> None:
    host = os.environ["MAIL_HOST"]
    port = int(os.environ.get("MAIL_PORT", "465"))
    user = os.environ["MAIL_USER"]
    pwd = os.environ["MAIL_PASS"].replace("\ufeff", "").strip()  # 防御 BOM/空白污染
    to = [x.strip() for x in os.environ["MAIL_TO"].split(",") if x.strip()]
    if not to:
        raise RuntimeError("MAIL_TO 为空")
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = user
    msg["To"] = ", ".join(to)
    if port == 465:
        s = smtplib.SMTP_SSL(host, port, timeout=30)
    else:
        s = smtplib.SMTP(host, port, timeout=30)
        s.starttls()
    try:
        s.login(user, pwd)
        s.sendmail(user, to, msg.as_string())
    finally:
        s.quit()


# ---------------- 主检查流程 ----------------

async def run_check() -> int:
    from features.mtf_tamr import scan_universe

    rest = CloudRest()

    # 币池：默认全市场 USDT 永续；若显式设置 WATCH_SYMBOLS 则只用列表（调试用）
    watch = os.environ.get("WATCH_SYMBOLS", "").strip()
    if watch:
        symbols = [x.strip().upper() for x in watch.split(",") if x.strip()]
        print(f"[checker] 使用显式关注列表: {len(symbols)} 个")
    else:
        symbols = await rest.market_pool()
    if not symbols:
        print("[checker] 币池为空，跳过")
        return 0

    rows: list[dict] = []
    t0 = time.monotonic()
    for strat in STRATEGIES:
        print(f"[checker] 策略 {strat} @ {FOCUS_TF} 扫描 {len(symbols)} 个币种 …")
        r = await scan_universe(rest, None, symbols, settings=None,
                                limit=len(symbols), tf=FOCUS_TF, strategy=strat)
        for x in r:
            x["strategy"] = strat
        rows.extend(r)
        print(f"[checker] {strat} 产出 {len(r)} 条信号，累计耗时 {time.monotonic()-t0:.0f}s")
    print(f"[checker] 本轮可执行信号 {len(rows)} 条（耗时 {time.monotonic()-t0:.0f}s）")

    if not rows:
        return 0

    old = load_state()
    new = [r for r in rows if fingerprint(r) not in old]
    if not new:
        print("[checker] 均为已通知信号，不发邮件")
        return 0

    print(f"[checker] 新信号 {len(new)} 条，发送邮件…")
    body = "crypto-quant 云端信号监视\n" + "=" * 40 + "\n\n"
    for i, r in enumerate(new, 1):
        body += f"【{i}】\n" + fmt_signal(r) + "\n\n"
    body += "-" * 40 + "\n本邮件由程序自动生成，信号仅供学习研究参考，不构成投资建议。"

    send_mail(f"【crypto-quant 信号】{len(new)} 条新信号", body)

    for r in new:
        old.add(fingerprint(r))
    save_state(old)
    print(f"[checker] 邮件已发送，去重状态已更新（共 {len(old)} 条指纹）")
    return len(new)


def main() -> int:
    if os.environ.get("CHECKER_MODE") == "test-mail" or "--test-mail" in sys.argv:
        send_mail("【crypto-quant】测试邮件",
                  "这是一封来自云端信号监视器的测试邮件。\n收到即代表 SMTP 配置正常，后续有新信号会发到这里。\n\n—— crypto-quant 信号监视器")
        print("[checker] 测试邮件已发送")
        return 0
    try:
        return asyncio.run(run_check())
    except Exception as exc:
        print(f"[checker] 执行出错: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
