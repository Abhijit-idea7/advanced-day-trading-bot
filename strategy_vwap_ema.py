"""
strategy_vwap_ema.py
--------------------
VWAP + 9/20 EMA Pullback Strategy.

SOURCE: Andrew Aziz, "Advanced Techniques in Day Trading"
        Chapters on VWAP trading and Moving Average trend strategies.

CONCEPT
-------
VWAP (Volume-Weighted Average Price) is the benchmark used by institutional
traders to evaluate execution quality. When a stock is trending, it will
pull back to VWAP and find buyers (in an uptrend) or sellers (in a downtrend).

The 9 EMA and 20 EMA act as a dual-confirmation trend filter:
  9 EMA > 20 EMA  →  bullish intraday trend  →  look for LONG pullbacks
  9 EMA < 20 EMA  →  bearish intraday trend  →  look for SHORT pullbacks

This combination (EMA trend filter + VWAP pullback entry) is one of the
highest win-rate setups described in the book because:
  1. Trend direction is confirmed before entry (no counter-trend trades)
  2. VWAP pullback provides a low-risk entry with a logical, tight stop
  3. Institutional support/resistance at VWAP means bounces are reliable
  4. Volume confirmation on the bounce candle filters weak setups

ENTRY RULES — LONG (mirror for SHORT)
--------------------------------------
  1. Trend confirmed: 9 EMA > 20 EMA (uptrend)
  2. Price has pulled back to VWAP (within VWAP_PROXIMITY_PCT = 0.4%)
  3. Bounce confirmed:
       a. Current close > VWAP  (price bounced above, not just touched)
       b. Current close > previous close  (momentum turning up)
  4. RSI in neutral zone (35–65) — not overbought, not in free-fall
  5. Volume spike on bounce candle >= 1.3× 10-candle average
  6. Before VWAP_ENTRY_CUTOFF_TIME (12:30 IST)

STOP LOSS
---------
  LONG:  VWAP × (1 - 0.5%)  — just below VWAP; if VWAP breaks, thesis is gone
  SHORT: VWAP × (1 + 0.5%)  — just above VWAP

TARGET
------
  LONG:  Entry + (2 × risk)   [RISK_REWARD_RATIO = 2.0]
  SHORT: Entry - (2 × risk)

EXIT SIGNALS (in addition to SL/Target)
-----------------------------------------
  EMA_REVERSAL: 9 EMA crosses back through 20 EMA against the trade.
                This means the intraday trend has reversed — the trade
                premise is invalid and should be closed regardless of P&L.

INDIA-SPECIFIC NOTES
--------------------
  - VWAP is displayed by default on Zerodha Kite, Angel One, and most Indian
    trading platforms — so it has self-fulfilling institutional significance
  - The 9 EMA on 2-min NSE charts corresponds to 18 minutes of data; it's a
    very responsive trailing guide of intraday momentum
  - This strategy is most reliable between 10:00–12:30 IST after the open
    settles. The early morning (9:15–10:00) VWAP is still being "discovered"
    and is less meaningful as a support/resistance level
  - Avoid on major event days: RBI policy, Union Budget, stock-specific
    results — VWAP can be distorted by volume surges in non-directional ways
"""

import logging
from datetime import datetime

import pandas as pd
import pytz

from config import (
    RISK_REWARD_RATIO,
    VWAP_ENTRY_CUTOFF_TIME,
    VWAP_PROXIMITY_PCT,
    VWAP_RSI_MAX,
    VWAP_RSI_MIN,
    VWAP_VOLUME_MULTIPLIER,
)
from indicators import (
    EMA_FAST_COL,
    EMA_SLOW_COL,
    RSI_COL,
    VOLAVG_COL,
    VWAP_COL,
)

IST    = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)
_HOLD  = {"action": "HOLD", "sl": 0.0, "target": 0.0}

STRATEGY_NAME = "VWAP_EMA"
_VWAP_SL_PCT  = 0.005   # Stop-loss placed 0.5% beyond VWAP


def generate_signal(df: pd.DataFrame, symbol: str = "") -> dict:
    """
    Evaluate the last COMPLETED candle (iloc[-2]) for a VWAP+EMA pullback signal.

    iloc[-1] = currently forming candle (incomplete — never use for signals)
    iloc[-2] = last fully closed candle  ← signal candle
    iloc[-3] = candle before signal candle (used for bounce confirmation)
    """
    # Entry cutoff gate
    now_ist = datetime.now(IST)
    h, m = map(int, VWAP_ENTRY_CUTOFF_TIME.split(":"))
    if now_ist >= now_ist.replace(hour=h, minute=m, second=0, microsecond=0):
        return _HOLD

    if len(df) < 4:
        return _HOLD

    row      = df.iloc[-2]   # last completed candle (signal candle)
    prev_row = df.iloc[-3]   # candle before signal

    # Guard: NaN checks on all required indicators
    required = [EMA_FAST_COL, EMA_SLOW_COL, VWAP_COL, RSI_COL, VOLAVG_COL]
    if any(pd.isna(row.get(col)) for col in required):
        return _HOLD
    if pd.isna(prev_row.get("Close")):
        return _HOLD

    close      = float(row["Close"])
    prev_close = float(prev_row["Close"])
    ema_fast   = float(row[EMA_FAST_COL])
    ema_slow   = float(row[EMA_SLOW_COL])
    vwap       = float(row[VWAP_COL])
    rsi        = float(row[RSI_COL])
    volume     = float(row["Volume"])
    vol_avg    = float(row[VOLAVG_COL])

    vol_ratio       = (volume / vol_avg) if vol_avg > 0 else 0.0
    vwap_proximity  = abs(close - vwap) / vwap

    logger.info(
        f"{symbol} VWAP: close={close:.2f} vwap={vwap:.2f} prox={vwap_proximity:.3%} "
        f"ema9={ema_fast:.2f} ema20={ema_slow:.2f} "
        f"rsi={rsi:.1f} vol={vol_ratio:.2f}x"
    )

    # ---- LONG: uptrend + pullback to VWAP + bounce ----
    if (
        ema_fast > ema_slow                           # uptrend confirmed
        and vwap_proximity <= VWAP_PROXIMITY_PCT      # price near VWAP
        and close > vwap                              # bounced above VWAP
        and close > prev_close                        # upward momentum on signal candle
        and VWAP_RSI_MIN <= rsi <= VWAP_RSI_MAX       # RSI neutral (not extended)
        and vol_ratio >= VWAP_VOLUME_MULTIPLIER       # volume confirms the bounce
    ):
        sl   = vwap * (1 - _VWAP_SL_PCT)
        risk = close - sl
        if risk <= 0:
            return _HOLD
        target = close + (RISK_REWARD_RATIO * risk)
        logger.info(
            f"{symbol} VWAP: *** BUY SIGNAL *** "
            f"entry={close:.2f} sl={sl:.2f} target={target:.2f} "
            f"risk=₹{risk:.2f} ema9>ema20 vol={vol_ratio:.2f}x"
        )
        return {"action": "BUY", "sl": sl, "target": target, "strategy": STRATEGY_NAME}

    # ---- SHORT: downtrend + rally to VWAP + rejection ----
    if (
        ema_fast < ema_slow                           # downtrend confirmed
        and vwap_proximity <= VWAP_PROXIMITY_PCT      # price near VWAP
        and close < vwap                              # rejected below VWAP
        and close < prev_close                        # downward momentum on signal candle
        and VWAP_RSI_MIN <= rsi <= VWAP_RSI_MAX       # RSI neutral
        and vol_ratio >= VWAP_VOLUME_MULTIPLIER       # volume confirms the rejection
    ):
        sl   = vwap * (1 + _VWAP_SL_PCT)
        risk = sl - close
        if risk <= 0:
            return _HOLD
        target = close - (RISK_REWARD_RATIO * risk)
        logger.info(
            f"{symbol} VWAP: *** SELL SIGNAL *** "
            f"entry={close:.2f} sl={sl:.2f} target={target:.2f} "
            f"risk=₹{risk:.2f} ema9<ema20 vol={vol_ratio:.2f}x"
        )
        return {"action": "SELL", "sl": sl, "target": target, "strategy": STRATEGY_NAME}

    return _HOLD


def check_exit_signal(df: pd.DataFrame, position: dict) -> str | None:
    """
    Exit conditions for an open VWAP+EMA position.

    Priority order:
      1. Target hit
      2. Stop-loss hit
      3. EMA trend reversal (9 EMA crosses through 20 EMA against the trade)
         — If the intraday trend has flipped, the trade premise is gone.
           Exit before waiting for SL, preserving more capital.

    Uses iloc[-2] (last completed candle) for consistency.
    """
    if len(df) < 2:
        return None

    row       = df.iloc[-2]
    close     = float(row["Close"])
    direction = position["direction"]
    sl        = float(position["sl"])
    target    = float(position["target"])

    ema_fast = float(row[EMA_FAST_COL]) if not pd.isna(row.get(EMA_FAST_COL)) else None
    ema_slow = float(row[EMA_SLOW_COL]) if not pd.isna(row.get(EMA_SLOW_COL)) else None

    if direction == "BUY":
        if close >= target:
            return "TARGET"
        if close <= sl:
            return "STOP_LOSS"
        # EMA trend reversed against our long position
        if ema_fast is not None and ema_slow is not None and ema_fast < ema_slow:
            return "EMA_REVERSAL"
    else:  # SELL
        if close <= target:
            return "TARGET"
        if close >= sl:
            return "STOP_LOSS"
        # EMA trend reversed against our short position
        if ema_fast is not None and ema_slow is not None and ema_fast > ema_slow:
            return "EMA_REVERSAL"

    return None
