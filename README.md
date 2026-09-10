# crypto-quant 云端信号监视器

基于你的本地 crypto-quant 客户端抽离的**云端版信号监视器**，运行在 GitHub Actions 上，**免费、24小时不间断、不需要你开电脑**。

## 功能

- **与本地客户端算法完全一致**：复用 `quant_research/signals.py` 的 MTF-TAMR 信号引擎（同一份代码，无重写）
- **仅两套策略**：EXP3（趋势内回调均值回归）+ EXP4-S（一次性EMA20止盈+1h扩容）
- **仅 4 小时级别**：聚焦 4h（偏置 12h），与客户端「4小时」页签严格对齐
- **每 15 分钟**自动扫描一次监控币种
- **有新信号自动发邮件**到你的 QQ 邮箱，同一信号不会重复轰炸

## 文件结构

```
crypto-quant-cloud/
├── cloud_signal_checker.py      # 主检查脚本（拉K线→算信号→去重→发邮件）
├── src/features/mtf_tamr.py     # MTF-TAMR 聚合引擎（与本地同源）
├── quant_research/              # 信号算法（signals/indicators/filters，与本地同源）
├── state/notified.json          # 已通知信号指纹（去重状态）
└── .github/workflows/signal-check.yml  # 定时任务（每15分钟）
```

## 配置（GitHub Actions Secrets）

| Secret | 说明 |
|---|---|
| `MAIL_HOST` | SMTP 服务器，QQ邮箱填 `smtp.qq.com` |
| `MAIL_PORT` | `465` |
| `MAIL_USER` | 发件邮箱，如 `454075863@qq.com` |
| `MAIL_PASS` | SMTP 授权码（QQ邮箱设置→账户→开启SMTP生成） |
| `MAIL_TO` | 收件邮箱（可多个，逗号分隔） |

监控币种通过仓库变量 `WATCH_SYMBOLS` 配置（逗号分隔）。

## 本地调试

```bash
pip install -r requirements.txt
python cloud_signal_checker.py --test-mail   # 发测试邮件验证SMTP
python cloud_signal_checker.py               # 手动检查一轮
```

## 免责声明

信号由程序基于公开行情数据自动生成，仅供学习研究参考，不构成投资建议。
