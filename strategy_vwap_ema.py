"""
strategy_vwap_ema.py
--------------------
VWAP + 9/20/50 EMA Pullback Strategy.

SOURCE: Andrew Aziz, "Advanced Techniques in Day Trading"
        Chapters on VWAP trading and Moving Average trend strategies.

CONCEPT
-------
When a stock is trending intraday, it will pull back to VWAP and bounce.
The 9/20 EMA cross confirms short-term trend direction, while the 50 EMA
acts as a macro session filter — only trade in the direction the stock has
been trending for most of the session.

PROFITABILITY IMPROVEMENTS (vs v1)
------------------------------------
1. 50 EMA macro trend filter (new):
   Only LONG if 9 EMA > 50 EMA (stock has been bullish most of the session).
   Only SHORT if 9 EMA < 50 EMA.
   Eliminates VWAP trades taken against the session's established direction —
   those were the main source of losing trades in the original version.

2. Stronger EMA_REVERSAL exit condition:
   Previously exited the moment 9/20 EMAs crossed — which happens dozens
   of times per day on 2-min charts, cutting many winning trades short.
   Now requires BOTH the EMA cross AND price to confirm:
   For LONG: 9 EMA < 20 EMA AND close < 20 EMA (not just a cross).
   This keeps trades alive through minor pullbacks while exiting true reversals.

3. Minimum risk filter (new):
   Skips trades where risk is < VWAP_MIN_RISK_PCT (0.2%) of entry price.
   Tiny-risk setups are usually price hugging VWAP — the Rs40 brokerage
   becomes a disproportionate cost relative to the small profit potential.

4. Loosened entry filters:
   VWAP proximity: 0.4% → 0.7%  (catches more valid pullbacks)
   RSI range: 35–65 → 30–70      (more entries, still avoids extremes)
   Volume multiplier: 1.3× → 1.1× (removes overfitting on volume)

ENTRY RULES — LONG (mirror for SHORT)
--------------------------------------
  1. Macro trend: 9 EMA > 50 EMA (bullish session)
  2. Short-term trend: 9 EMA > 20 EMA
  3. Price pulled back to VWAP (within 0.7%)
  4. Bounce: close > VWAP AND close > previous close
  5. RSI neutral: 30–70
  6. Volume >= 1.1× average
  7. Risk >= 0.2% of entry (minimum trade quality gate)
  8. Before 12:30 IST

STOP LOSS   LONG: VWAP × 0.995  |  SHORT: VWAP × 1.005
TARGET      LONG: Entry + 2× risk  |  SHORT: mirror

EXIT SIGNALS
------------
  TARGET       : Price hits calculated target
  STOP_LOSS    : Price hits 0.5% beyond VWAP
  EMA_REVERSAL : 9 EMA crossed below 20 EMA AND price is below 20 EMA
                 (confirmed trend reversal, not just a momentary cross)
"""

import logging
from datetime import datetime

import pandas as pd
import pytz

from config import (
    RISK_REWARD_RATIO,
    VWAP_ENTRY_CUTOFF_TIME,
    VWAP_MIN_RISK_PCT,
    VWAP_PROXIMITY_PCT,
    VWAP_RSI_MAX,
    VWAP_RSI_MIN,
    VWAP_VOLUME_MULTIPLIER,
)
from indicators import (
    EMA_FAST_COL,
    EMA_MACRO_COL,
    EMA_SLOW_COL,
    RSI_COL,
    VOLAVG_COL,
    VWAP_COL,
)

IST    = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)
_HOLD  = {"action": "HOLD", "sl": 0.0, "target": 0.0}

STRATEGY_NAME = "VWAP_EMA"
_VWAP_SL_PCT  = 0.005   # Stop-loss 0.5% beyond VWAP


def generate_signal(df: pd.DataFrame, symbol: str = "", sim_time=None) -> dict:
    """
    Evaluate the last COMPLETED candle (iloc[-2]) for a VWAP+EMA pullback signal.

    iloc[-1] = currently forming candle (incomplete — never use for signals)
    iloc[-2] = last fully closed candle  ← signal candle
    iloc[-3] = prior candle (for bounce momentum check)

    sim_time: candle timestamp for backtesting (avoids datetime.now() in backtest).
    """
    # Entry cutoff gate
    now_ist = sim_time if sim_time is not None else datetime.now(IST)
    if hasattr(now_ist, "tzinfo") and now_ist.tzinfo is None:
        now_ist = IST.localize(now_ist)
    h, m = map(int, VWAP_ENTRY_CUTOFF_TIME.split(":"))
    if now_ist >= now_ist.replace(hour=h, minute=m, second=0, microsecond=0):
        return _HOLD

    if len(df) < 4:
        return _HOLD

    row      = df.iloc[-2]   # signal candle
    prev_row = df.iloc[-3]   # prior candle

    # Guard: NaN checks
    required = [EMA_FAST_COL, EMA_SLOW_COL, EMA_MACRO_COL, VWAP_COL, RSI_COL, VOLAVG_COL]
    if any(pd.isna(row.get(col)) for col in required):
        return _HOLD
    if pd.isna(prev_row.get("Close")):
        return _HOLD

    close      = float(row["Close"])
    prev_close = float(prev_row["Close"])
    ema_fast   = float(row[EMA_FAST_COL])
    ema_slow   = float(row[EMA_SLOW_COL])
    ema_macro  = float(row[EMA_MACRO_COL])
    vwap       = float(row[VWAP_COL])
    rsi        = float(row[RSI_COL])
    volume     = float(row["Volume"])
    vol_avg    = float(row[VOLAVG_COL])

    vol_ratio      = (volume / vol_avg) if vol_avg > 0 else 0.0
    vwap_proximity = abs(close - vwap) / vwap

    logger.info(
        f"{symbol} VWAP: close={close:.2f} vwap={vwap:.2f} prox={vwap_proximity:.3%} "
        f"ema9={ema_fast:.2f} ema20={ema_slow:.2f} ema50={ema_macro:.2f} "
        f"rsi={rsi:.1f} vol={vol_ratio:.2f}x"
    )

    # ---- LONG: macro uptrend + pullback to VWAP + bounce ----
    if (
        ema_fast > ema_macro                          # macro session uptrend (9 > 50 EMA)
        and ema_fast > ema_slow                       # short-term uptrend confirmed
        and vwap_proximity <= VWAP_PROXIMITY_PCT      # pulled back near VWAP
        and close > vwap                              # bounced above VWAP
        and close > prev_close                        # momentum turning up on signal candle
        and VWAP_RSI_MIN <= rsi <= VWAP_RSI_MAX       # RSI neutral
        and vol_ratio >= VWAP_VOLUME_MULTIPLIER       # volume confirms the bounce
    ):
        sl   = vwap * (1 - _VWAP_SL_PCT)
        risk = close - sl
        if risk <= 0:
            return _HOLD
        # Minimum risk filter: avoid tiny setups eaten by brokerage
        if risk / close < VWAP_MIN_RISK_PCT:
            logger.info(f"{symbol} VWAP: LONG rejected — risk too small ({risk/close:.3%} < {VWAP_MIN_RISK_PCT:.3%})")
            return _HOLD
        target = close + (RISK_REWARD_RATIO * risk)
        logger.info(
            f"{symbol} VWAP: *** BUY SIGNAL *** "
            f"entry={close:.2f} sl={sl:.2f} target={target:.2f} "
            f"risk=Rs{risk:.2f} R:R={RISK_REWARD_RATIO} ema9>ema50 vol={vol_ratio:.2f}x"
        )
        return {"action": "BUY", "sl": sl, "target": target, "strategy": STRATEGY_NAME}

    # ---- SHORT: macro downtrend + rally to VWAP + rejection ----
    if (
        ema_fast < ema_macro                          # macro session downtrend (9 < 50 EMA)
        and ema_fast < ema_slow                       # short-term downtrend confirmed
        and vwap_proximity <= VWAP_PROXIMITY_PCT      # price rallied near VWAP
        and close < vwap                              # rejected below VWAP
        and close < prev_close                        # momentum turning down on signal candle
        and VWAP_RSI_MIN <= rsi <= VWAP_RSI_MAX       # RSI neutral
        and vol_ratio >= VWAP_VOLUME_MULTIPLIER       # volume confirms the rejection
    ):
        sl   = vwap * (1 + _VWAP_SL_PCT)
        risk = sl - close
        if risk <= 0:
            return _HOLD
        if risk / close < VWAP_MIN_RISK_PCT:
            logger.info(f"{symbol} VWAP: SHORT rejected — risk too small ({risk/close:.3%} < {VWAP_MIN_RISK_PCT:.3%})")
            return _HOLD
        target = close - (RISK_REWARD_RATIO * risk)
        logger.info(
            f"{symbol} VWAP: *** SELL SIGNAL *** "
            f"entry={close:.2f} sl={sl:.2f} target={target:.2f} "
            f"risk=Rs{risk:.2f} R:R={RISK_REWARD_RATIO} ema9<ema50 vol={vol_ratio:.2f}x"
        )
        return {"action": "SELL", "sl": sl, "target": target, "strategy": STRATEGY_NAME}

    return _HOLD


def check_exit_signal(df: pd.DataFrame, position: dict) -> str | None:
    """
    Exit conditions for an open VWAP+EMA position.

    Priority:
      1. Target hit
      2. Stop-loss hit
      3. EMA_REVERSAL — CONFIRMED trend reversal (not just a momentary cross).
         Requires BOTH:
           a. 9 EMA has crossed back through 20 EMA against the trade
           b. Price is on the wrong side of the 20 EMA
         This prevents EMA_REVERSAL from firing on normal pullbacks within a trend.
    """
    if len(df) < 2:
        return None

    row       = df.iloc[-2]
    close     = float(row["Close"])
    direction = position["direction"]
    sl        = float(position["sl"])
    target    = float(position["target"])

    ema_fast  = float(row[EMA_FAST_COL])  if not pd.isna(row.get(EMA_FAST_COL))  else None
    ema_slow  = float(row[EMA_SLOW_COL])  if not pd.isna(row.get(EMA_SLOW_COL))  else None

    if direction == "BUY":
        if close >= target:
            return "TARGET"
        if close <= sl:
            return "STOP_LOSS"
        # Confirmed reversal: 9 EMA crossed below 20 EMA AND price is below 20 EMA.
        # Both conditions required — the EMA cross alone on a 2-min chart is too noisy.
        if ema_fast is not None and ema_slow is not None:
            if ema_fast < ema_slow and close < ema_slow:
                return "EMA_REVERSAL"
    else:  # SELL
        if close <= target:
            return "TARGET"
        if close >= sl:
            return "STOP_LOSS"
        # Confirmed reversal: 9 EMA crossed above 20 EMA AND price is above 20 EMA.
        if ema_fast is not None and ema_slow is not None:
            if ema_fast > ema_slow and close > ema_slow:
                return "EMA_REVERSAL"

    return None
