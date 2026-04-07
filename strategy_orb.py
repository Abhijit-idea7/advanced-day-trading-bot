"""
strategy_orb.py
---------------
Opening Range Breakout (ORB) Strategy — with Gap-Direction Filter.

SOURCE: Andrew Aziz, "Advanced Techniques in Day Trading"
        Chapters on Opening Range Breakout and Gap-and-Go setups.

CONCEPT
-------
The first 15 minutes of the NSE session (9:15–9:30 IST) define the
"opening range". A breakout above the high or breakdown below the low,
confirmed by volume, signals that one side has taken control for the day.

PROFITABILITY IMPROVEMENTS (vs v1)
------------------------------------
1. Gap-direction filter (biggest win-rate lift):
   If a stock gaps UP on open vs prior close → only take LONG ORB breakouts.
   If a stock gaps DOWN on open vs prior close → only take SHORT ORB breakdowns.
   Trading WITH the gap direction aligns us with institutional order flow.
   Counter-gap ORBs (e.g. shorting a gap-up open) have far lower win rates.

2. VWAP alignment check:
   For LONG breakouts, price at breakout candle must be above VWAP.
   For SHORT breakdowns, price must be below VWAP.
   VWAP is the session's fair-value anchor — entering on the correct side
   of it adds an extra layer of confirmation.

3. Higher target multiplier (2.5× instead of 1.5×):
   Better R:R. With SL at the full range and target at 2.5× range,
   a 40% win rate still produces net profit after brokerage.

4. ORB_FAILED exit buffer (0.3%):
   Prevents whipsaw exits when price briefly dips back below the breakout
   level on a healthy retest then continues in the breakout direction.

ENTRY RULES — LONG (mirror for SHORT)
--------------------------------------
  1. ORB period has ended (>= 9:30 IST)
  2. Gap-direction: today's open gapped UP vs prior close (>= ORB_MIN_GAP_PCT)
     — OR — open is flat (±ORB_MIN_GAP_PCT) and VWAP alignment confirms
  3. Close > ORB High (price has broken out above the opening range)
  4. Extension check: close has NOT moved >0.8% beyond ORB high (no chasing)
  5. VWAP alignment: close > VWAP at time of breakout (above fair value)
  6. Volume on breakout candle >= 1.3× 10-candle average
  7. ORB range is meaningful: between 0.3% and 4% of price
  8. Before ORB_ENTRY_CUTOFF_TIME (11:00 IST)

STOP LOSS   LONG: ORB Low  |  SHORT: ORB High
TARGET      LONG: Entry + (ORB range × 2.5)  |  SHORT: mirror

EXIT SIGNALS
------------
  TARGET     : Candle High (LONG) / Low (SHORT) touches the calculated target
               — intrabar detection so no fills are missed
  STOP_LOSS  : Candle Low (LONG) / High (SHORT) touches the effective SL,
               which starts at ORB Low/High and moves to entry (breakeven)
               once the trade gains ORB_BREAKEVEN_TRIGGER_R × initial risk
  ORB_FAILED : Price CLOSES more than ORB_FAILED_BUFFER_PCT (0.8%) back
               inside the opening range — confirmed false breakout
               (close-based intentionally; intrabar wicks are noise)
  TIME_EXIT  : Position is AT A LOSS at ORB_MAX_HOLD_TIME (12:30 IST) —
               the breakout failed to gain momentum; exit to avoid the
               dead lunch period (12:00–13:30) making losses worse.
               Profitable positions at 12:30 are NOT exited here — they
               continue running toward target or are protected by the
               breakeven SL. Handled in the main loop / backtest engine.

INDIA-SPECIFIC NOTES
--------------------
  - NSE pre-open auction runs 9:00–9:15; continuous session starts 9:15 IST
  - The gap filter works especially well on F&O stocks where overnight
    futures positioning creates strong directional gaps
  - Avoid ORB on stocks with circuit breakers hit — gaps can be misleading
"""

import logging
from datetime import datetime

import pandas as pd
import pytz

from config import (
    ORB_BREAKEVEN_TRIGGER_R,
    ORB_CHASE_LIMIT_PCT,
    ORB_ENTRY_CUTOFF_TIME,
    ORB_FAILED_BUFFER_PCT,
    ORB_MAX_RANGE_PCT,
    ORB_MIN_GAP_PCT,
    ORB_MIN_RANGE_PCT,
    ORB_TARGET_MULTIPLIER,
    ORB_VOLUME_MULTIPLIER,
)
from indicators import (
    DAY_OPEN_COL,
    ORB_ESTABLISHED_COL,
    ORB_HIGH_COL,
    ORB_LOW_COL,
    PREV_DAY_CLOSE_COL,
    VOLAVG_COL,
    VWAP_COL,
)

IST    = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)
_HOLD  = {"action": "HOLD", "sl": 0.0, "target": 0.0}

STRATEGY_NAME = "ORB"


def generate_signal(df: pd.DataFrame, symbol: str = "", sim_time=None) -> dict:
    """
    Evaluate the last COMPLETED candle (iloc[-2]) for an ORB entry signal.

    iloc[-1] = currently forming candle (incomplete — never use for signals)
    iloc[-2] = last fully closed candle  ← signal candle

    sim_time: candle timestamp for backtesting (avoids datetime.now() in backtest).
    """
    # Entry cutoff gate
    now_ist = sim_time if sim_time is not None else datetime.now(IST)
    if hasattr(now_ist, "tzinfo") and now_ist.tzinfo is None:
        now_ist = IST.localize(now_ist)
    h, m = map(int, ORB_ENTRY_CUTOFF_TIME.split(":"))
    if now_ist >= now_ist.replace(hour=h, minute=m, second=0, microsecond=0):
        return _HOLD

    if len(df) < 3:
        return _HOLD

    row = df.iloc[-2]   # last completed candle (signal candle)

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
    vwap    = float(row[VWAP_COL])   if not pd.isna(row.get(VWAP_COL))   else 0.0

    # ---- Range quality filter ----
    range_pct = orb_range / orb_high
    if range_pct < ORB_MIN_RANGE_PCT:
        logger.info(f"{symbol} ORB: range too narrow ({range_pct:.2%}) — flat open, skipping")
        return _HOLD
    if range_pct > ORB_MAX_RANGE_PCT:
        logger.info(f"{symbol} ORB: range too wide ({range_pct:.2%}) — gap event, skipping")
        return _HOLD

    vol_ratio = (volume / vol_avg) if vol_avg > 0 else 0.0

    # ---- Gap-direction filter  (the book's "Gap-and-Go" rule) ----
    # Determine today's directional bias from the gap vs prior close.
    prev_close = row.get(PREV_DAY_CLOSE_COL)
    day_open   = row.get(DAY_OPEN_COL)

    gap_pct = 0.0
    if not pd.isna(prev_close) and not pd.isna(day_open) and float(prev_close) > 0:
        gap_pct = (float(day_open) - float(prev_close)) / float(prev_close)

    # gap_pct > +ORB_MIN_GAP_PCT  → gapped UP  → bullish bias → LONG only
    # gap_pct < -ORB_MIN_GAP_PCT  → gapped DOWN → bearish bias → SHORT only
    # within ±ORB_MIN_GAP_PCT     → flat open   → both directions (VWAP decides)
    gap_up   = gap_pct >= ORB_MIN_GAP_PCT
    gap_down = gap_pct <= -ORB_MIN_GAP_PCT
    flat_open = not gap_up and not gap_down

    logger.info(
        f"{symbol} ORB: close={close:.2f} orb=[{orb_low:.2f}–{orb_high:.2f}] "
        f"range={range_pct:.2%} gap={gap_pct:+.2%} vol={vol_ratio:.2f}x vwap={vwap:.2f}"
    )

    # ---- LONG: breakout above ORB high ----
    if close > orb_high:
        # Direction filter: must match gap bias (or flat open with VWAP above)
        if gap_down:
            logger.info(f"{symbol} ORB: LONG rejected — stock gapped down ({gap_pct:+.2%}), no counter-gap longs")
            return _HOLD

        extension = (close - orb_high) / orb_high
        if extension > ORB_CHASE_LIMIT_PCT:
            logger.info(f"{symbol} ORB: LONG rejected — chasing {extension:.2%} above ORB high")
            return _HOLD
        if vol_ratio < ORB_VOLUME_MULTIPLIER:
            logger.info(f"{symbol} ORB: LONG rejected — weak volume {vol_ratio:.2f}x")
            return _HOLD
        # VWAP alignment: price should be above VWAP at breakout
        if vwap > 0 and close < vwap:
            logger.info(f"{symbol} ORB: LONG rejected — close {close:.2f} below VWAP {vwap:.2f}")
            return _HOLD

        sl   = orb_low
        risk = close - sl
        if risk <= 0:
            return _HOLD
        target = close + (orb_range * ORB_TARGET_MULTIPLIER)
        rr     = (target - close) / risk
        logger.info(
            f"{symbol} ORB: *** BUY SIGNAL *** "
            f"entry={close:.2f} sl={sl:.2f} target={target:.2f} "
            f"R:R={rr:.1f} risk=Rs{risk:.2f} vol={vol_ratio:.2f}x gap={gap_pct:+.2%}"
        )
        return {"action": "BUY", "sl": sl, "target": target, "strategy": STRATEGY_NAME}

    # ---- SHORT: breakdown below ORB low ----
    if close < orb_low:
        # Direction filter: must match gap bias (or flat open with VWAP below)
        if gap_up:
            logger.info(f"{symbol} ORB: SHORT rejected — stock gapped up ({gap_pct:+.2%}), no counter-gap shorts")
            return _HOLD

        extension = (orb_low - close) / orb_low
        if extension > ORB_CHASE_LIMIT_PCT:
            logger.info(f"{symbol} ORB: SHORT rejected — chasing {extension:.2%} below ORB low")
            return _HOLD
        if vol_ratio < ORB_VOLUME_MULTIPLIER:
            logger.info(f"{symbol} ORB: SHORT rejected — weak volume {vol_ratio:.2f}x")
            return _HOLD
        # VWAP alignment: price should be below VWAP at breakdown
        if vwap > 0 and close > vwap:
            logger.info(f"{symbol} ORB: SHORT rejected — close {close:.2f} above VWAP {vwap:.2f}")
            return _HOLD

        sl   = orb_high
        risk = sl - close
        if risk <= 0:
            return _HOLD
        target = close - (orb_range * ORB_TARGET_MULTIPLIER)
        rr     = (close - target) / risk
        logger.info(
            f"{symbol} ORB: *** SELL SIGNAL *** "
            f"entry={close:.2f} sl={sl:.2f} target={target:.2f} "
            f"R:R={rr:.1f} risk=Rs{risk:.2f} vol={vol_ratio:.2f}x gap={gap_pct:+.2%}"
        )
        return {"action": "SELL", "sl": sl, "target": target, "strategy": STRATEGY_NAME}

    return _HOLD


def check_exit_signal(df: pd.DataFrame, position: dict) -> str | None:
    """
    Exit conditions for an open ORB position.

    Priority:
      1. Target hit     — uses candle HIGH (BUY) / LOW (SELL) so intrabar
                          touches of the target are captured, not just closes.
      2. Stop-loss hit  — uses candle LOW (BUY) / HIGH (SELL) for the same
                          reason. SL is the *effective* SL, which may be the
                          original orb_low/orb_high OR entry price (breakeven)
                          once the trade has moved ORB_BREAKEVEN_TRIGGER_R × risk.
      3. ORB_FAILED     — uses CLOSE (intentional): a confirmed close back inside
                          the range is required, not just an intrabar wick.
                          Buffer is ORB_FAILED_BUFFER_PCT (0.8%) to survive retests.

    Breakeven logic:
      Once the candle HIGH (BUY) / LOW (SELL) has moved at least
      ORB_BREAKEVEN_TRIGGER_R (0.6) × initial_risk beyond entry, the effective
      SL is raised to entry_price — locking in a no-loss trade even if the target
      is not reached.
    """
    if len(df) < 2:
        return None

    row       = df.iloc[-2]
    close     = float(row["Close"])
    candle_h  = float(row["High"])
    candle_l  = float(row["Low"])
    direction = position["direction"]
    target    = float(position["target"])

    # Original SL (orb_low for BUY, orb_high for SELL)
    original_sl    = float(position["sl"])
    entry_price    = float(position.get("entry_price", original_sl))   # fallback: original SL
    initial_risk   = abs(entry_price - original_sl)

    orb_high = float(row[ORB_HIGH_COL]) if not pd.isna(row.get(ORB_HIGH_COL)) else None
    orb_low  = float(row[ORB_LOW_COL])  if not pd.isna(row.get(ORB_LOW_COL))  else None

    if direction == "BUY":
        # ---- Breakeven: move SL to entry once trade gained 60% of initial risk ----
        if initial_risk > 0 and candle_h >= entry_price + ORB_BREAKEVEN_TRIGGER_R * initial_risk:
            effective_sl = max(original_sl, entry_price)   # never lower than original SL
        else:
            effective_sl = original_sl

        # 1. Target — intrabar high touched target
        if candle_h >= target:
            return "TARGET"
        # 2. Stop-loss — intrabar low breached effective SL
        if candle_l <= effective_sl:
            return "STOP_LOSS"
        # 3. ORB_FAILED — confirmed close back inside the range (close-based, intentional)
        if orb_high is not None and close < orb_high * (1 - ORB_FAILED_BUFFER_PCT):
            return "ORB_FAILED"

    else:  # SELL
        # ---- Breakeven: move SL to entry once trade gained 60% of initial risk ----
        if initial_risk > 0 and candle_l <= entry_price - ORB_BREAKEVEN_TRIGGER_R * initial_risk:
            effective_sl = min(original_sl, entry_price)   # never higher than original SL
        else:
            effective_sl = original_sl

        # 1. Target — intrabar low touched target
        if candle_l <= target:
            return "TARGET"
        # 2. Stop-loss — intrabar high breached effective SL
        if candle_h >= effective_sl:
            return "STOP_LOSS"
        # 3. ORB_FAILED — confirmed close back inside the range (close-based, intentional)
        if orb_low is not None and close > orb_low * (1 + ORB_FAILED_BUFFER_PCT):
            return "ORB_FAILED"

    return None
