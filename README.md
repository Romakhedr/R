<div align="center">

# 🤖 R (Trend-Following Trading Bot)

> *A Python-based algorithmic trading bot designed for real-time market tracking, executing EMA crossover strategies with RSI and volume confirmation, featuring comprehensive built-in risk management.*

[![GitHub Repository](https://img.shields.io/badge/GitHub-Source_Code-blue?style=for-the-badge&logo=github)
(https://github.com/Romakhedr/R)
</div>
---

## ⚠️ Important Disclaimer
> This code is for educational and programming purposes only; it does not constitute financial or investment advice.
---
## 📦 Installation

```bash

pip install -r requirements.txt
```
## 🛡️ Risk Management
* **Position Sizing:** Trade size is automatically calculated to ensure the potential loss does not exceed `risk_per_trade_pct` (default: `1%`).
* **Stop-Loss:** Set dynamically via `Stop-loss = Entry price minus (ATR × 2)`.
* **Take-Profit:** Calculated using `Take-profit = Risk amount × risk_reward_ratio` (default: `2`).
* **Daily Loss Limit:** A daily loss limit (`max_daily_loss_pct`, default: `5%`) automatically stops the bot.
