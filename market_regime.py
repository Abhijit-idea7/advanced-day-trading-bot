"""
market_regime.py
----------------
Intraday NIFTY50 market regime detection for the ALPHA_COMBO strategy.

MOTIVATION (from the backtest data)
────────────────────────────────────
9 out of 30 backtest days had a ≤20% win rate and collectively lost ~Rs15,700 net —
more than the entire strategy's net profit. These were days when the broad market
(NIFTY50) was in a downtrend or high-volatility state, causing most individual
long setups to fail regardless of their alpha score.

The article's 11-step engine (Step 6) describes cross-sectional demeaning:
removing the market-wide component shared by all signals at a given timestamp.
This module implements that concept at the market level: if NIFTY itself is
strongly directional, that directional information should override or constrain
individual stock signals.

REGIME CLASSIFICATION
─────────────────────
  BULL    → NIFTY above VWAP, 9 EMA > 50 EMA, positive day change
            → Full 10-position exposure, standard 0.30 alpha threshold
            → LONG_ONLY direction filter (individual short setups rare on bull days)

  NEUTRAL → Mixed signals, no dominant direction
            → 8-position cap, slightly stricter 0.33 threshold
            → BOTH directions allowed

  BEAR    → NIFTY below VWAP, 9 EMA < 50 EMA, negative day change
            → 4-position cap, strict 0.42 threshold (only highest-conviction trades)
            → SHORT_ONLY direction filter (prevents adding losing longs against market)

The direction filter is the most powerful lever: on the Feb-27 day that produced
0 wins in 10 trades, all 10 were likely longs in a bear market. Blocking longs
in BEAR regime prevents the entire loss on such days.

NIFTY TICKER
────────────
Yahoo Finance ticker for NIFTY50 is ^NSEI (no .NS suffix — it's an index).
We reuse the same indicator machinery (add_indicators) so VWAP, EMA 9/50,
and VWAP are computed identically to the stock signals.
"""

import logging
import time

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from config import (
    ALPHA_ENTRY_THRESHOLD,
    CANDLE_INTERVAL,
    MAX_POSITIONS,
    REGIME_BEAR_ALPHA_THRESHOLD,
    REGIME_BEAR_MAX_POSITIONS,
    REGIME_BEAR_THRESHOLD,
    REGIME_BULL_THRESHOLD,
    REGIME_NEUTRAL_ALPHA_THRESHOLD,
    REGIME_NEUTRAL_MAX_POSITIONS,
)
from indicators import add_indicators

IST          = pytz.timezone("Asia/Kolkata")
logger       = logging.getLogger(__name__)
NIFTY_TICKER = "^NSEI"   # NIFTY50 — no .NS suffix (index, not stock)

# Default regime returned when NIFTY data is unavailable
_NEUTRAL_REGIME = {
    "regime":           "NEUTRAL",
    "score":            0.0,
    "max_positions":    MAX_POSITIONS,
    "alpha_threshold":  ALPHA_ENTRY_THRESHOLD,
    "direction_filter": "BOTH",
}


def _fetch_nifty(period: str = "5d") -> pd.DataFrame | None:
    """
    Fetch NIFTY50 2-min OHLCV candles.

    Uses the same retry logic as data_feed.py but without the .NS suffix,
    since ^NSEI is a Yahoo Finance index symbol.
    """
    for attempt in range(3):
        try:
            df = yf.Ticker(NIFTY_TICKER).history(interval=CANDLE_INTERVAL, period=period)
            if df.empty:
                logger.warning(f"NIFTY: empty data on attempt {attempt + 1}")
                time.sleep(2)
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert(IST)
            else:
                df.index = df.index.tz_convert(IST)
            return df
        except Exception as e:
            logger.warning(f"NIFTY fetch error attempt {attempt + 1}: {e}")
            time.sleep(2)
    logger.error("NIFTY: all fetch attempts failed")
    return None


def get_nifty_regime() -> dict:
    """
    Compute the current intraday NIFTY50 market regime.

    Three scoring components (each ∈ [-1.0, +1.0]):
      1. VWAP position  (weight 0.40) — above/below daily-anchored VWAP
      2. EMA trend      (weight 0.40) — 9 EMA vs 50 EMA alignment
      3. Day change     (weight 0.20) — price vs first candle open

    Weighted score → regime classification:
      score > +REGIME_BULL_THRESHOLD → BULL
      score < -REGIME_BEAR_THRESHOLD → BEAR
      otherwise                      → NEUTRAL

    Returns a dict that main.py injects into strategy_alpha_combo and
    uses to cap max_positions in scan_for_entries.
    """
    df = _fetch_nifty()
    if df is None or len(df) < 10:
        logger.warning("NIFTY data unavailable — defaulting to NEUTRAL regime")
        return dict(_NEUTRAL_REGIME)

    try:
        df_ind   = add_indicators(df)
        today    = df_ind.index[-1].date()
        today_df = df_ind[df_ind.index.date == today]

        if len(today_df) < 3:
            return dict(_NEUTRAL_REGIME)

        row   = today_df.iloc[-2]   # last completed candle
        close = float(row["Close"])

        components: list[float] = []
        weights:    list[float] = []

        # --- Component 1: VWAP position (weight 0.40) ---
        vwap = row.get("vwap")
        if not pd.isna(vwap) and float(vwap) > 0:
            vwap_dev = (close - float(vwap)) / float(vwap)
            components.append(float(np.tanh(vwap_dev * 100)))
            weights.append(0.40)

        # --- Component 2: EMA trend 9 vs 50 (weight 0.40) ---
        ema9  = row.get("ema_fast")
        ema50 = row.get("ema_macro")
        if not any(pd.isna(x) for x in (ema9, ema50)):
            components.append(1.0 if float(ema9) > float(ema50) else -1.0)
            weights.append(0.40)

        # --- Component 3: Day change from open (weight 0.20) ---
        day_open = row.get("day_open")
        if not pd.isna(day_open) and float(day_open) > 0:
            day_chg = (close - float(day_open)) / float(day_open)
            components.append(float(np.tanh(day_chg * 50)))
            weights.append(0.20)

        if not components:
            return dict(_NEUTRAL_REGIME)

        total_w = sum(weights)
        score   = float(np.clip(
            sum(c * w / total_w for c, w in zip(components, weights)),
            -1.0, 1.0
        ))

        # --- Classify regime ---
        if score > REGIME_BULL_THRESHOLD:
            result = {
                "regime":           "BULL",
                "score":            score,
                "max_positions":    MAX_POSITIONS,            # full 10 slots
                "alpha_threshold":  ALPHA_ENTRY_THRESHOLD,   # standard 0.30
                "direction_filter": "LONG_ONLY",             # block shorts on bull days
            }
        elif score < REGIME_BEAR_THRESHOLD:
            result = {
                "regime":           "BEAR",
                "score":            score,
                "max_positions":    REGIME_BEAR_MAX_POSITIONS,    # 4 slots max
                "alpha_threshold":  REGIME_BEAR_ALPHA_THRESHOLD,  # strict 0.42
                "direction_filter": "SHORT_ONLY",                 # block longs on bear days
            }
        else:
            result = {
                "regime":           "NEUTRAL",
                "score":            score,
                "max_positions":    REGIME_NEUTRAL_MAX_POSITIONS,    # 8 slots
                "alpha_threshold":  REGIME_NEUTRAL_ALPHA_THRESHOLD,  # 0.33
                "direction_filter": "BOTH",
            }

        nifty_close = close
        logger.info(
            f"NIFTY regime: {result['regime']} (score={score:+.3f} nifty={nifty_close:.0f}) | "
            f"max_pos={result['max_positions']} "
            f"threshold={result['alpha_threshold']:.2f} "
            f"dir={result['direction_filter']}"
        )
        return result

    except Exception as e:
        logger.warning(f"Regime computation error: {e} — defaulting to NEUTRAL")
        return dict(_NEUTRAL_REGIME)
