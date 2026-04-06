"""
strategy_orb.py
---------------
Opening Range Breakout (ORB) Strategy.

SOURCE: Andrew Aziz, "Advanced Techniques in Day Trading"
        Chapter on Opening Range Breakout and Gap-and-Go setups.

CONCEPT
-------
The first 15 minutes of the NSE session (9:15–9:30 IST) define the
"opening range" — the high and low set while institutional order flow from
the pre-market auction resolves. Once this range is established, a breakout
above the high or breakdown below the low signals that one side has taken
control for the day. This is one of the highest-probability setups in
intraday trading because:
  1. The range reflects genuine supply/demand discovery
  2. Breakout direction aligns with the day's institutional bias
  3. The stop-loss (other side of range) is logically grounded
  4. Volume confirmation filters out false breakouts

ENTRY RULES — LONG (mirror for SHORT)
--------------------------------------
  1. ORB period has ended (>= 9:30 IST)
  2. Close > ORB High (price has broken out of the range)
  3. Extension check: close has NOT moved >1% beyond ORB high (no chasing)
  4. ORB range is meaningful: between 0.3% and 4% of price
  5. Volume on breakout candle >= 1.5× 10-candle average
  6. Before ORB_ENTRY_CUTOFF_TIME (11:00 IST)
     — ORBs that haven't fired by 11 AM are stale and unreliable

STOP LOSS
---------
  LONG:  ORB Low  (the full range is the risk; if price re-enters range, thesis failed)
  SHORT: ORB High

TARGET
------
  LONG:  Entry + (ORB range × ORB_TARGET_MULTIPLIER)  [default 1.5× range]
  SHORT: Entry - (ORB range × ORB_TARGET_MULTIPLIER)

EXIT SIGNALS (in addition to SL/Target)
-----------------------------------------
  ORB_FAILED: Price closes back inside the ORB (false breakout confirmed).
              This early exit preserves capital vs waiting for SL.

INDIA-SPECIFIC NOTES
--------------------
  - NSE pre-open auction runs 9:00–9:15; continuous session starts 9:15 IST
  - The first 15-min range on NSE captures the post-auction price discovery
  - Works particularly well on F&O-eligible high-beta stocks (Nifty50, BankNifty components)
  - On event days (RBI policy, Budget, earnings), widen ORB_MAX_RANGE_PCT or skip
"""

import logging
from datetime import datetime

import pandas as pd
import pytz

from config import (
    ORB_CHASE_LIMIT_PCT,
    ORB_ENTRY_CUTOFF_TIME,
    ORB_MAX_RANGE_PCT,
    ORB_MIN_RANGE_PCT,
    ORB_TARGET_MULTIPLIER,
    ORB_VOLUME_MULTIPLIER,
)
from indicators import (
    ORB_ESTABLISHED_COL,
    ORB_HIGH_COL,
    ORB_LOW_COL,
    VOLAVG_COL,
    VWAP_COL,
)

IST    = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)
_HOLD  = {"action": "HOLD", "sl": 0.0, "target": 0.0}

STRATEGY_NAME = "ORB"


def generate_signal(df: pd.DataFrame, symbol: str = "") -> dict:
    """
    Evaluate the last COMPLETED candle (iloc[-2]) for an ORB entry signal.

    iloc[-1] = currently forming candle (incomplete — never use for signals)
    iloc[-2] = last fully closed candle  ← signal candle
    """
    # Entry cutoff gate
    now_ist = datetime.now(IST)
    h, m = map(int, ORB_ENTRY_CUTOFF_TIME.split(":"))
    if now_ist >= now_ist.replace(hour=h, minute=m, second=0, microsecond=0):
        return _HOLD

    if len(df) < 3:
        return _HOLD

    row = df.iloc[-2]   # last completed candle

    # Guard: ORB must be established (past the 15-min window)
    if not bool(row.get(ORB_ESTABLISHED_COL, False)):
        return _HOLD

    orb_high = row.get(ORB_HIGH_COL)
    orb_low  = row.get(ORB_LOW_COL)
    if pd.isna(orb_high) or pd.isna(orb_low):
        return _HOLD

    orb_high  = float(orb_high)
    orb_low   = float(orb_low)
    orb_range = orb_high - orb_low

    if orb_range <= 0:
        return _HOLD

    close   = float(row["Close"])
    volume  = float(row["Volume"])
    vol_avg = float(row[VOLAVG_COL]) if not pd.isna(row.get(VOLAVG_COL)) else 0.0

    # ---- Range quality filter ----
    range_pct = orb_range / orb_high
    if range_pct < ORB_MIN_RANGE_PCT:
        logger.info(f"{symbol} ORB: range too narrow ({range_pct:.2%}) — skipping flat open")
        return _HOLD
    if range_pct > ORB_MAX_RANGE_PCT:
        logger.info(f"{symbol} ORB: range too wide ({range_pct:.2%}) — skipping gap event")
        return _HOLD

    vol_ratio = (volume / vol_avg) if vol_avg > 0 else 0.0

    logger.info(
        f"{symbol} ORB: close={close:.2f} orb=[{orb_low:.2f}–{orb_high:.2f}] "
        f"range={range_pct:.2%} vol={vol_ratio:.2f}x"
    )

    # ---- LONG: breakout above ORB high ----
    if close > orb_high:
        extension = (close - orb_high) / orb_high
        if extension > ORB_CHASE_LIMIT_PCT:
            logger.info(f"{symbol} ORB: LONG rejected — chasing {extension:.2%} above ORB high")
            return _HOLD
        if vol_ratio < ORB_VOLUME_MULTIPLIER:
            logger.info(f"{symbol} ORB: LONG rejected — weak volume {vol_ratio:.2f}x")
            return _HOLD

        sl     = orb_low
        risk   = close - sl
        if risk <= 0:
            return _HOLD
        target = close + (orb_range * ORB_TARGET_MULTIPLIER)
        logger.info(
            f"{symbol} ORB: *** BUY SIGNAL *** "
            f"entry={close:.2f} sl={sl:.2f} target={target:.2f} "
            f"risk=₹{risk:.2f} vol={vol_ratio:.2f}x"
        )
        return {"action": "BUY", "sl": sl, "target": target, "strategy": STRATEGY_NAME}

    # ---- SHORT: breakdown below ORB low ----
    if close < orb_low:
        extension = (orb_low - close) / orb_low
        if extension > ORB_CHASE_LIMIT_PCT:
            logger.info(f"{symbol} ORB: SHORT rejected — chasing {extension:.2%} below ORB low")
            return _HOLD
        if vol_ratio < ORB_VOLUME_MULTIPLIER:
            logger.info(f"{symbol} ORB: SHORT rejected — weak volume {vol_ratio:.2f}x")
            return _HOLD

        sl     = orb_high
        risk   = sl - close
        if risk <= 0:
            return _HOLD
        target = close - (orb_range * ORB_TARGET_MULTIPLIER)
        logger.info(
            f"{symbol} ORB: *** SELL SIGNAL *** "
            f"entry={close:.2f} sl={sl:.2f} target={target:.2f} "
            f"risk=₹{risk:.2f} vol={vol_ratio:.2f}x"
        )
        return {"action": "SELL", "sl": sl, "target": target, "strategy": STRATEGY_NAME}

    return _HOLD


def check_exit_signal(df: pd.DataFrame, position: dict) -> str | None:
    """
    Exit conditions for an open ORB position.

    Priority order:
      1. Target hit
      2. Stop-loss hit
      3. Failed breakout — price closes back inside the opening range

    A failed breakout exit is unique to ORB: if the breakout reverses
    and price re-enters the range, the trade thesis is invalid. Exiting
    early here protects capital rather than waiting for the full SL.

    Uses iloc[-2] (last completed candle) for consistency.
    """
    if len(df) < 2:
        return None

    row       = df.iloc[-2]
    close     = float(row["Close"])
    direction = position["direction"]
    sl        = float(position["sl"])
    target    = float(position["target"])

    orb_high = float(row[ORB_HIGH_COL]) if not pd.isna(row.get(ORB_HIGH_COL)) else None
    orb_low  = float(row[ORB_LOW_COL])  if not pd.isna(row.get(ORB_LOW_COL))  else None

    if direction == "BUY":
        if close >= target:
            return "TARGET"
        if close <= sl:
            return "STOP_LOSS"
        # Failed breakout: price re-enters the ORB
        if orb_high is not None and close < orb_high:
            return "ORB_FAILED"
    else:  # SELL
        if close <= target:
            return "TARGET"
        if close >= sl:
            return "STOP_LOSS"
        # Failed breakdown: price re-enters the ORB
        if orb_low is not None and close > orb_low:
            return "ORB_FAILED"

    return None
