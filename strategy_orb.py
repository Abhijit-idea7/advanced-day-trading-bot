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

BOTH LONG AND SHORT TRADES
---------------------------
  LONG  — price breaks ABOVE the ORB high on a gap-up or flat-open day.
  SHORT — price breaks BELOW the ORB low on a gap-down or flat-open day.
  The gap-direction filter ensures trades align with institutional order flow.

KEY IMPROVEMENTS
----------------
1. Gap-direction filter (biggest win-rate lift):
   If a stock gaps UP vs prior close  → only take LONG ORB breakouts.
   If a stock gaps DOWN vs prior close → only take SHORT ORB breakdowns.
   Counter-gap ORBs have far lower win rates and are blocked entirely.

2. VWAP alignment check:
   LONG  entry: price must be ABOVE VWAP at breakout.
   SHORT entry: price must be BELOW VWAP at breakdown.
   VWAP is the session's fair-value anchor; being on the correct side adds
   an extra layer of institutional confirmation.

3. 2.5× target multiplier:
   Target = entry ± (ORB range × 2.5). Even a 40% win rate produces net
   profit after brokerage at this R:R.

4. Intrabar exit detection:
   TARGET and STOP_LOSS use candle High/Low (not just Close), so intrabar
   fills are captured accurately.

5. Breakeven trailing stop (ORB_BREAKEVEN_TRIGGER_R = 0.6):
   Once the trade gains 60% of initial risk, the effective SL is raised to
   entry price — locking in a no-loss trade.

6. ORB_FAILED buffer (0.8%):
   Requires price to close 0.8% back inside the range before exiting.
   Shallow retests of the breakout level are treated as normal behaviour.

ENTRY RULES — LONG (mirror for SHORT)
--------------------------------------
  1. ORB period has ended (>= 9:30 IST)
  2. Gap-direction: today's open gapped UP vs prior close (>= ORB_MIN_GAP_PCT)
     — OR — open is flat (±ORB_MIN_GAP_PCT) and VWAP alignment confirms
  3. Close > ORB High (price has broken out above the opening range)
  4. Extension: close has NOT moved >0.8% beyond ORB high (no chasing)
  5. VWAP alignment: close > VWAP at breakout
  6. Volume on breakout candle >= 1.3× 10-candle average
  7. ORB range is meaningful: between 0.3% and 4% of price
  8. Before ORB_ENTRY_CUTOFF_TIME (11:00 IST)

STOP LOSS   LONG: ORB Low  |  SHORT: ORB High
TARGET      LONG: Entry + (ORB range × 2.5)  |  SHORT: mirror

EXIT SIGNALS (in priority order)
---------------------------------
  TARGET     : Candle High (LONG) / Low (SHORT) touches the calculated target.
               Intrabar detection — no fills missed because candle closed short.
  STOP_LOSS  : Candle Low (LONG) / High (SHORT) breaches the effective SL.
               The effective SL starts at ORB Low/High and is raised to entry
               (breakeven) once the trade gains ORB_BREAKEVEN_TRIGGER_R × risk.
  ORB_FAILED : Price CLOSES more than 0.8% back inside the opening range.
               Close-based intentionally — intrabar wicks back inside are noise.
  SQUARE_OFF : Hard close at 15:15 IST (handled by the main loop, not here).

INDIA-SPECIFIC NOTES
--------------------
  - NSE pre-open auction runs 9:00–9:15; continuous session starts 9:15 IST.
  - The gap filter works especially well on F&O stocks where overnight
    futures positioning creates strong directional gaps.
  - Avoid ORB on stocks that hit circuit breakers — gaps can be misleading.
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
    ORB_POSITION_SCALE,
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

    # ---- Gap-direction filter  (Andrew Aziz "Gap-and-Go" rule) ----
    prev_close = row.get(PREV_DAY_CLOSE_COL)
    day_open   = row.get(DAY_OPEN_COL)

    gap_pct = 0.0
    if not pd.isna(prev_close) and not pd.isna(day_open) and float(prev_close) > 0:
        gap_pct = (float(day_open) - float(prev_close)) / float(prev_close)

    # gap_pct > +ORB_MIN_GAP_PCT  → gapped UP   → bullish bias  → LONG only
    # gap_pct < -ORB_MIN_GAP_PCT  → gapped DOWN → bearish bias  → SHORT only
    # within ±ORB_MIN_GAP_PCT     → flat open   → both directions (VWAP decides)
    gap_up   = gap_pct >= ORB_MIN_GAP_PCT
    gap_down = gap_pct <= -ORB_MIN_GAP_PCT

    logger.info(
        f"{symbol} ORB: close={close:.2f} orb=[{orb_low:.2f}–{orb_high:.2f}] "
        f"range={range_pct:.2%} gap={gap_pct:+.2%} vol={vol_ratio:.2f}x vwap={vwap:.2f}"
    )

    # ---- LONG: breakout above ORB high ----
    if close > orb_high:
        # Gap-direction filter: block counter-gap longs
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
        return {"action": "BUY", "sl": sl, "target": target, "strategy": STRATEGY_NAME, "quantity_scale": ORB_POSITION_SCALE}

    # ---- SHORT: breakdown below ORB low ----
    if close < orb_low:
        # Gap-direction filter: block counter-gap shorts
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
        return {"action": "SELL", "sl": sl, "target": target, "strategy": STRATEGY_NAME, "quantity_scale": ORB_POSITION_SCALE}

    return _HOLD


def check_exit_signal(df: pd.DataFrame, position: dict) -> str | None:
    """
    Exit conditions for an open ORB position.

    Priority:
      1. TARGET     — candle High (BUY) / Low (SELL) touched the target.
                      Intrabar detection so fills are not missed because the
                      candle closed short of the target.
      2. STOP_LOSS  — candle Low (BUY) / High (SELL) breached effective SL.
                      Effective SL = original SL (orb_low / orb_high) until
                      the trade gains ORB_BREAKEVEN_TRIGGER_R × initial risk,
                      at which point it is raised to entry_price (breakeven).
      3. ORB_FAILED — price CLOSES more than ORB_FAILED_BUFFER_PCT (0.8%)
                      back inside the opening range. Close-based intentionally:
                      intrabar wicks back inside the range are normal behaviour
                      on a valid breakout; only a confirmed close signals failure.
    """
    if len(df) < 2:
        return None

    row      = df.iloc[-2]
    close    = float(row["Close"])
    candle_h = float(row["High"])
    candle_l = float(row["Low"])

    direction    = position["direction"]
    target       = float(position["target"])
    original_sl  = float(position["sl"])
    entry_price  = float(position.get("entry_price", original_sl))
    initial_risk = abs(entry_price - original_sl)

    orb_high = float(row[ORB_HIGH_COL]) if not pd.isna(row.get(ORB_HIGH_COL)) else None
    orb_low  = float(row[ORB_LOW_COL])  if not pd.isna(row.get(ORB_LOW_COL))  else None

    if direction == "BUY":
        # Breakeven: once High reaches entry + 60% of initial risk, SL → entry
        if initial_risk > 0 and candle_h >= entry_price + ORB_BREAKEVEN_TRIGGER_R * initial_risk:
            effective_sl = max(original_sl, entry_price)
        else:
            effective_sl = original_sl

        if candle_h >= target:
            return "TARGET"
        if candle_l <= effective_sl:
            return "STOP_LOSS"
        # Failed breakout: confirmed close back inside the range
        if orb_high is not None and close < orb_high * (1 - ORB_FAILED_BUFFER_PCT):
            return "ORB_FAILED"

    else:  # SELL
        # Breakeven: once Low reaches entry - 60% of initial risk, SL → entry
        if initial_risk > 0 and candle_l <= entry_price - ORB_BREAKEVEN_TRIGGER_R * initial_risk:
            effective_sl = min(original_sl, entry_price)
        else:
            effective_sl = original_sl

        if candle_l <= target:
            return "TARGET"
        if candle_h >= effective_sl:
            return "STOP_LOSS"
        # Failed breakdown: confirmed close back inside the range
        if orb_low is not None and close > orb_low * (1 + ORB_FAILED_BUFFER_PCT):
            return "ORB_FAILED"

    return None
