#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GitHub Actions runner 网络诊断：测试币安各端点 + 备选交易所可用性。"""
import requests

ENDPOINTS = [
    ("binance ping", "https://fapi.binance.com/fapi/v1/ping", None),
    ("binance time", "https://fapi.binance.com/fapi/v1/time", None),
    ("binance klines BTC 4h", "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=4h&limit=5", None),
    ("binance exchangeInfo", "https://fapi.binance.com/fapi/v1/exchangeInfo", None),
    ("binance ticker24h BTC", "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT", None),
    ("okx instruments", "https://www.okx.com/api/v5/public/instruments?instType=SWAP", None),
    ("okx candles BTC", "https://www.okx.com/api/v5/market/candles?instId=BTC-USDT-SWAP&bar=4H&limit=5", None),
    ("bybit kline BTC", "https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=240&limit=5", None),
]

for name, url, _ in ENDPOINTS:
    try:
        r = requests.get(url, timeout=25)
        preview = r.text[:120].replace("\n", " ")
        print(f"{name}: HTTP {r.status_code} | len={len(r.text)} | {preview}")
    except Exception as exc:
        print(f"{name}: ERROR {exc}")
