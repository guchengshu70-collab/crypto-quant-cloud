# crypto-quant 云端信号监视器 · 完整复刻指南

> 本文档面向**另一个 AI（如 DeepSeek）或工程师**，目标：仅凭本文档 + 仓库源码，
> 就能从零复刻一套"全市场加密货币 4H 级别信号云端监视器"。
> 本系统已在 GitHub Actions 上实际运行并通过验证（币池 463 个，单轮约 8 分钟）。

---

## 0. 一句话目标

把本地量化客户端（crypto-quant，基于币安数据 + MTF-TAMR 信号引擎）的**信号检测逻辑**，
抽离成一个**免费、24 小时、无需用户开电脑**的云端定时任务：
每 30 分钟拉全市场 USDT 永续 K 线 → 用同一套信号算法计算 EXP3 / EXP4-S 两套策略的 4H 信号 →
（可选）邮件通知用户。当前邮件通知已关闭，仅保留扫描与去重状态。

---

## 1. 整体架构

```
GitHub Actions（每30分钟 cron 触发）
   │
   ▼
cloud_signal_checker.py（单进程 Python 3.12）
   │ 1. OKX REST 拉全市场 USDT 永续币池（按24h成交额降序，463个）
   │ 2. 对每个币拉多周期已收盘K线（4h/12h + 1h/6h/1w/1d/1M 等，共约10个周期）
   │ 3. 调用 MTF-TAMR 信号引擎（quant_research/，与本地客户端同一份代码）
   │    - exp3：趋势内回调均值回归
   │    - exp4：EXP4-S 一次性EMA20止盈 + 1h扩容
   │ 4. 聚焦 4H 级别（交易级4h + 偏置12h），共振≥2/5 才算可执行信号
   │ 5. 与 state/notified.json 比对去重（指纹：策略|币种|方向|触发bar|级别集）
   │ 6. 邮件通知（可选，默认关）；把新指纹写回 state/notified.json 并 git push
   ▼
state/notified.json（仓库内持久化去重状态）
```

关键决策：
- **数据源用 OKX 而非币安**：币安 fapi 对 GitHub Actions runner 的 IP 全量返回 HTTP 451（地域封锁），
  OKX 公共接口全通（实测 200）。算法不变，仅行情来源不同。
- **共享 K 线缓存**：exp3 与 exp4 共用同一个 MultiTFCache 实例，K 线请求量减半，
  单轮耗时从 18 分钟降到 8 分钟。
- **退出码必须为 0**：脚本正常完成（无论有无信号）返回 0，否则 GitHub Actions 判失败。

---

## 2. 仓库文件结构

```
crypto-quant-cloud/
├── cloud_signal_checker.py         # 主脚本（全部云端逻辑，约 250 行）
├── requirements.txt                # requests, numpy
├── README.md                       # 使用说明
├── state/notified.json             # 去重状态（脚本自动提交更新）
├── .github/workflows/
│   ├── signal-check.yml            # 主定时任务（cron */30 + workflow_dispatch）
│   └── diag.yml                    # 网络诊断用（可选，复刻时可删除）
├── src/
│   ├── __init__.py                 # 空
│   └── features/
│       ├── __init__.py             # 空
│       └── mtf_tamr.py             # MTF-TAMR 多周期聚合引擎（原样拷贝自本地客户端）
└── quant_research/
    ├── signals.py                  # EXP3 信号核心（原样拷贝）
    ├── signals_violent.py          # 暴力拉升/暴跌信号（原样拷贝，未被云端使用）
    ├── signals_range.py            # 震荡期信号（原样拷贝）
    ├── signals_mtop.py             # M顶空头信号（原样拷贝）
    ├── indicators.py               # 指标库 EMA/MACD/RSI/ATR/布林等
    └── filters.py                  # 入场过滤器
```

> **重要**：`mtf_tamr.py` 与 `quant_research/*.py` 是**从用户本地客户端原样拷贝**的算法文件，
> 云端只改数据源适配层（CloudRest 类），**信号算法零改动**——这是"和本地客户端信号一致"的保证。

---

## 3. 数据源：OKX REST API

### 3.1 币池（全市场 USDT 永续）

```
GET https://www.okx.com/api/v5/public/instruments?instType=SWAP&quoteCcy=USDT&state=live
```
- 筛选 `settleCcy == "USDT"`，instId 形如 `BTC-USDT-SWAP` → 转换为 `BTCUSDT`（去掉 `-USDT-SWAP`）
- 实测返回 **463 个**（OKX USDT 永续总数，覆盖全部主流币）

成交额降序（可选，用于扫描优先级）：
```
GET https://www.okx.com/api/v5/market/tickers?instType=SWAP
```
- 用 `volCcyQuote24h`（24h 成交额，USDT 计价）排序

### 3.2 K 线

```
GET https://www.okx.com/api/v5/market/candles?instId=BTC-USDT-SWAP&bar=4H&limit=300
```
- bar 参数：`1H/4H/6H/12H/1D/1W/1M`（分钟级用 `5m/15m/30m` 小写）
- limit 上限 300（信号引擎需要 ≥160 根，300 足够）
- **返回为倒序（最新在前），必须 `reversed()` 转正序**
- 每根 K 线数组：`[ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]`
- **`confirm` 字段：`"1"` 表示已收盘，`"0"` 表示未收盘——只保留已收盘 K 线**（与本地客户端语义一致）

### 3.3 限流

- OKX 公共接口约 10 req/s 上限，用简单令牌桶限流（`RateLimiter(rate=8, burst=20)`）
- 463 币 × ~10 周期 ≈ 4600 次请求，8 req/s ≈ 10 分钟；配合共享缓存实际单轮约 8 分钟
- 429 时指数退避重试

---

## 4. 信号引擎：MTF-TAMR（核心算法，勿改动）

来源：本地客户端 `crypto-quant` 项目，`src/features/mtf_tamr.py` + `quant_research/`。

### 4.1 信号流程

1. **K 线准备**：对每个币、每个周期（4h/12h/1h/6h/1w/1d/1M）拉已收盘 K 线（最多 250 根）
2. **scan_level**（signals.py）：每个周期算一套 setup（方向、入场、止损、止盈、盈亏比、原因）
3. **多周期聚合**（mtf_tamr.py）：
   - 交易级：1h/4h/6h/12h/1w（1h 已退役，不参与方向锚定但参与共振计数）
   - 偏置级：4h/12h/1d/1w/1M
   - 方向 = 最新 setup 的方向；`aligned_count` = 聚合结果中与该方向一致的级别数
4. **聚焦 4H 模式**（tf="4h"）：
   - 只认 4h 级别信号（4h + 偏置 12h）
   - 可执行信号过滤条件：`aligned_count >= 2`（即 ≥2/5 级别同向共振）
5. **输出字段**：symbol, direction(±1), level, entry_zone[lo,hi], stop_loss, take_profit_1,
   risk_reward, reason, signal_bar_t(ms), aligned_count, total_levels, levels_fired

### 4.2 两套策略

| 策略 | 含义 | 差异点 |
|---|---|---|
| exp3 | 趋势内回调均值回归 | 标准入场 + V11 分批出场 |
| exp4 | EXP4-S 一次性EMA20止盈 + 1h扩容 | 一次性止盈 + 1h 扩容（4h 焦点下该差异即 EXP4-S 命名来源） |

调用方式（共享缓存，重要）：
```python
cache = MultiTFCache()          # 两个策略共用，K线请求减半
for strat in ("exp3", "exp4"):
    rows = await scan_universe(rest, cache, symbols, settings=None,
                               limit=len(symbols), tf="4h", strategy=strat)
```

---

## 5. 主脚本逻辑（cloud_signal_checker.py）

```
main()
 ├─ CHECKER_MODE == "test-mail" → 发一封测试邮件后退出（验证SMTP用）
 └─ run_check():
     1. 币池：WATCH_SYMBOLS 显式列表（调试用）或 OKX 全市场 463 币
     2. 共享 cache，对 exp3 / exp4 各跑一次 scan_universe(tf="4h")
     3. 合并两策略结果，标注 strategy 字段
     4. 加载 state/notified.json → 过滤出"新信号"（指纹未出现过）
     5. 若 MAIL_ENABLED=1：发邮件（最多列 25 条，其余提示省略）
        若 MAIL_ENABLED=0：不发邮件（默认），仅记录
     6. 新指纹并入状态 → 写回 state/notified.json → git add/commit/push（自动持久化）
     7. 正常结束返回 0
```

### 5.1 去重指纹

```
fingerprint = f"{strategy}|{symbol}|{direction}|{signal_bar_t}|{sorted(levels_fired)}"
```
- 同一币种同一方向在 4H bar 未滚动前指纹不变 → 下一轮自动去重，不重复通知
- state/notified.json 保留最近 500 条指纹

### 5.2 邮件（当前默认关闭）

- 环境变量 `MAIL_ENABLED=1` 才发送
- SMTP：`smtplib`，支持 465 SSL 与 587 STARTTLS
- 配置全部来自环境变量（MAIL_HOST/MAIL_PORT/MAIL_USER/MAIL_PASS/MAIL_TO），**密钥不进代码**

---

## 6. GitHub Actions 配置

### 6.1 signal-check.yml（主任务）

```yaml
name: crypto-quant-signal-monitor
on:
  schedule:
    - cron: '*/30 * * * *'     # 每30分钟（单轮约8分钟，不会重叠）
  workflow_dispatch:            # 支持手动触发
    inputs:
      mode: {type: choice, options: [check, test-mail]}
permissions:
  contents: write               # 必须：脚本要 git push 去重状态
jobs:
  scan:
    runs-on: ubuntu-latest
    timeout-minutes: 90
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.12'}
      - run: pip install -q requests numpy
      - run: python cloud_signal_checker.py
        env:
          CHECKER_MODE: ${{ inputs.mode }}
          MAIL_ENABLED: ${{ vars.MAIL_ENABLED }}   # 默认空 → 脚本默认关闭
          MAIL_HOST: ${{ secrets.MAIL_HOST }}
          MAIL_PORT: ${{ secrets.MAIL_PORT }}
          MAIL_USER: ${{ secrets.MAIL_USER }}
          MAIL_PASS: ${{ secrets.MAIL_PASS }}
          MAIL_TO: ${{ secrets.MAIL_TO }}
```

### 6.2 Secrets / Variables

| 类型 | 名称 | 说明 |
|---|---|---|
| Secret | MAIL_HOST | SMTP 服务器（QQ邮箱 smtp.qq.com） |
| Secret | MAIL_PORT | 465 |
| Secret | MAIL_USER | 发件邮箱 |
| Secret | MAIL_PASS | SMTP 授权码（非登录密码） |
| Secret | MAIL_TO | 收件邮箱（可逗号分隔多个） |
| Variable | MAIL_ENABLED | 设为 1 开启邮件，删除/留空为关闭 |
| Variable | WATCH_SYMBOLS | 可选：显式币种列表（调试用），不设则全市场 |

---

## 7. 部署步骤（复刻流程）

1. 创建 GitHub 公开仓库（公共仓库 Actions 免费额度无限；私有仓库每月 2000 分钟可能不够）
2. 上传全部文件（见 §2 文件结构）
3. 配置 Secrets/Variables（见 §6.2）
4. 页面 Actions → 手动 Run workflow（mode=check）触发首轮
5. 观察运行日志：应看到
   ```
   [checker] OKX 全市场 USDT 永续币池: 463 个（按24h成交额降序）
   [checker] 策略 exp3 @ 4h 扫描 463 个币种 …
   [checker] exp3 产出 N 条信号，累计耗时 XXXs
   [checker] 本轮可执行信号 N 条（耗时 XXXs）
   ```
6. 验证 `state/notified.json` 在运行后自动更新并出现 `chore: update notified signal state` 提交
7. 之后 cron 每 30 分钟自动运行

---

## 8. 已知坑与决策记录（复刻者必读）

1. **币安 451 封锁**：`fapi.binance.com` 对 GitHub Actions 的美国 runner IP 全部返回 451
   （含 ping/klines/exchangeInfo/ticker），**不可用**。备选 OKX（实测 200 全通）；Bybit 403（CloudFront 封锁）。
2. **币池数量**：OKX USDT 永续 463 个 < 币安 600+（币安多出的为低流动性小币）。
   要币安全量需付费云服务器，免费 GitHub Actions 做不到。
3. **共享缓存**：exp3/exp4 若不共享 MultiTFCache，请求量翻倍、单轮 18 分钟，30 分钟 cron 会重叠。
4. **退出码**：`sys.exit(信号数量)` 是错误写法，GitHub Actions 会把非 0 退出码判为失败。
   正常完成必须 return 0。
5. **BOM 陷阱**：Windows PowerShell `Set-Content -Encoding utf8` 会写入 BOM（\ufeff），
   存进 Secret 后 SMTP 登录 base64 编码崩溃。写敏感文件用 `UTF8Encoding($false)` 或 utf-8-sig 读取，
   并在代码里 `pwd.replace("\ufeff","")` 防御。
6. **OKX K 线倒序**：candles 返回最新在前，必须 reversed；confirm 字段过滤未收盘 bar。
7. **状态持久化必须在脚本内**：不能依赖 workflow 的"Commit 步骤"——脚本异常退出会跳过该步骤，
   导致去重状态丢失、下轮重复发邮件。脚本内 `save_state()` 后直接 git push。

---

## 9. 复刻验证清单

- [ ] 币池能拉到 ≥400 个 USDT 永续
- [ ] exp3/exp4 各能产出信号（或正确返回 0 条）
- [ ] 单轮耗时 < 15 分钟（共享缓存后实测 ~8 分钟）
- [ ] 运行结论为 success（退出码 0）
- [ ] state/notified.json 自动提交更新
- [ ] 同一信号连跑两轮只通知一次（去重生效）
- [ ] MAIL_ENABLED=1 + test-mail 模式能收到测试邮件

---

## 10. 当前运行状态（2026-09-11 实测）

- 币池：463 个 OKX USDT 永续
- 信号量：EXP3 约 77 条 / EXP4-S 约 79 条（4H 级别、共振≥2/5）
- 单轮耗时：约 8 分钟（共享缓存）
- 邮件：已关闭（MAIL_ENABLED 默认 0），扫描与去重照常运行
- 定时：cron `*/30 * * * *`，GitHub Actions 免费运行

> 免责声明：信号由程序基于公开行情自动生成，仅供学习研究参考，不构成投资建议。
