"""
اختبارات أساسية لبوت R — تتحقق من صحة المؤشرات الفنية وحسابات إدارة المخاطر.
تشغيل: pytest tests/
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trend_bot import ema, rsi, atr, BotConfig, R


def make_sample_df(n=60):
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    prices = np.clip(prices, 1, None)
    return pd.DataFrame({
        "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="15min"),
        "open": prices,
        "high": prices + rng.uniform(0, 1, n),
        "low": prices - rng.uniform(0, 1, n),
        "close": prices + rng.normal(0, 0.3, n),
        "volume": rng.uniform(50, 200, n),
    })


def test_ema_length_matches_input():
    df = make_sample_df()
    result = ema(df["close"], 12)
    assert len(result) == len(df)


def test_rsi_bounded_between_0_and_100():
    df = make_sample_df()
    result = rsi(df["close"], 14)
    assert result.min() >= 0
    assert result.max() <= 100


def test_atr_is_non_negative():
    df = make_sample_df()
    result = atr(df, 14)
    assert (result.dropna() >= 0).all()


def test_position_size_respects_risk_limit():
    cfg = BotConfig(paper_trading=True, risk_per_trade_pct=1.0, starting_balance=10_000)
    bot = R(cfg)
    entry_price = 100.0
    stop_loss = 95.0
    size = bot.calculate_position_size(entry_price, stop_loss)

    max_risk = cfg.starting_balance * (cfg.risk_per_trade_pct / 100)
    actual_risk = size * abs(entry_price - stop_loss)

    assert abs(actual_risk - max_risk) < 0.01


def test_position_size_zero_when_stop_equals_entry():
    cfg = BotConfig(paper_trading=True)
    bot = R(cfg)
    size = bot.calculate_position_size(100.0, 100.0)
    assert size == 0.0


def test_daily_loss_limit_halts_bot():
    cfg = BotConfig(paper_trading=True, max_daily_loss_pct=5.0, starting_balance=10_000)
    bot = R(cfg)
    bot.balance = 9_400
    assert bot.check_daily_loss_limit() is True


def test_daily_loss_limit_allows_trading_within_range():
    cfg = BotConfig(paper_trading=True, max_daily_loss_pct=5.0, starting_balance=10_000)
    bot = R(cfg)
    bot.balance = 9_800
    assert bot.check_daily_loss_limit() is False


def test_generate_signal_returns_valid_value():
    df = make_sample_df()
    from trend_bot import add_indicators
    cfg = BotConfig()
    df = add_indicators(df, cfg)
    df = df.dropna().reset_index(drop=True)

    bot = R(cfg)
    signal = bot.generate_signal(df)
    assert signal in ("buy", "sell", "hold")
