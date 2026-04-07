"""
indicators.py
-------------
All indicator implementations for the multi-strategy intraday bot.

Indicators computed:
  EMA Fast   — Exponential Moving Average (9-period)
  EMA Slow   — Exponential Moving Average (20-period)
  EMA Macro  — Exponential Moving Average (50-period) — session macro trend
  VWAP       — Volume-Weighted Average Price (anchored per day, resets daily)
  RSI        — Relative Strength Index (Wilder's smoothing, 14-period)
  Volume Avg — Rolling mean volume over VOLUME_LOOKBACK candles
  ORB        — Opening Range High/Low for first ORB_MINUTES of session
  Prev Close — Previous trading day's closing price (for gap calculation)
  Day Open   — First candle Open of the trading day (for gap calculation)

All implementations match TradingView output where applicable.
Column name constants are defined here and imported by strategy modules.
"""

import numpy as np
import pandas as pd
import pytz

from config import (
    EMA_FAST,
    EMA_MACRO,
    EMA_SLOW,
    ORB_MINUTES,
    RSI_PERIOD,
    VOLUME_LOOKBACK,
)

IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# Column name constants  (imported by strategy modules)
# ---------------------------------------------------------------------------
EMA_FAST_COL        = "ema_fast"          # 9-period EMA
EMA_SLOW_COL        = "ema_slow"          # 20-period EMA
EMA_MACRO_COL       = "ema_macro"         # 50-period EMA (session macro trend)
VWAP_COL            = "vwap"              # Daily-anchored VWAP
RSI_COL             = "rsi"               # RSI(14)
VOLAVG_COL          = "vol_avg"           # Rolling avg volume
ORB_HIGH_COL        = "orb_high"          # Opening range high
ORB_LOW_COL         = "orb_low"           # Opening range low
ORB_ESTABLISHED_COL = "orb_established"   # True once ORB window has closed
PREV_DAY_CLOSE_COL  = "prev_day_close"    # Prior trading day's last close
DAY_OPEN_COL        = "day_open"          # First candle Open of the trading day


# ---------------------------------------------------------------------------
# EMA  — matches TradingView's ta.ema()
# ---------------------------------------------------------------------------
def _ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# RSI  — Wilder's smoothing, matches TradingView's ta.rsi()
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# VWAP  — anchored to each calendar day, resets at midnight
# ---------------------------------------------------------------------------
def _vwap_daily(df: pd.DataFrame) -> pd.Series:
    """
    Calculate VWAP per trading day. Groups rows by date so VWAP resets
    correctly when the DataFrame spans multiple days (e.g. 5d warmup fetch).
    """
    result = pd.Series(np.nan, index=df.index)
    for _, group_idx in df.groupby(df.index.date).groups.items():
        grp       = df.loc[group_idx]
        tp        = (grp["High"] + grp["Low"] + grp["Close"]) / 3
        cum_vol   = grp["Volume"].cumsum()
        cum_tpvol = (tp * grp["Volume"]).cumsum()
        result.loc[group_idx] = (cum_tpvol / cum_vol.replace(0, np.nan)).values
    return result


# ---------------------------------------------------------------------------
# ORB  — Opening Range Breakout levels per trading day
# ---------------------------------------------------------------------------
def _opening_range(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute ORB high and low for each trading day.

    Opening Range = High and Low of the first ORB_MINUTES of trading.
    NSE continuous session opens at 9:15 IST, so:
      ORB_MINUTES=15  →  range covers  9:15–9:29 IST  (8 two-minute candles)

    Returns three Series indexed to df:
      orb_high         — ORB high value (filled for all candles of the day)
      orb_low          — ORB low value  (filled for all candles of the day)
      orb_established  — True only for candles AFTER the ORB window has closed
    """
    orb_high        = pd.Series(np.nan,  index=df.index)
    orb_low         = pd.Series(np.nan,  index=df.index)
    orb_established = pd.Series(False,   index=df.index)

    for date, group_idx in df.groupby(df.index.date).groups.items():
        grp = df.loc[group_idx].sort_index()
        if grp.empty:
            continue

        first_ts  = grp.index[0]
        orb_end   = first_ts + pd.Timedelta(minutes=ORB_MINUTES)

        # Candles inside the ORB window
        orb_candles = grp[grp.index < orb_end]
        if orb_candles.empty:
            continue

        high = orb_candles["High"].max()
        low  = orb_candles["Low"].min()

        # Populate ORB values for the entire day
        orb_high.loc[group_idx] = high
        orb_low.loc[group_idx]  = low

        # Mark candles AFTER the ORB window as "established"
        post_orb_idx = grp[grp.index >= orb_end].index
        orb_established.loc[post_orb_idx] = True

    return orb_high, orb_low, orb_established


# ---------------------------------------------------------------------------
# Prev Day Close + Day Open  — for gap-direction filter in ORB strategy
# ---------------------------------------------------------------------------
def _prev_day_close_and_day_open(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    For each trading day in df:
      prev_day_close — the last closing price of the PREVIOUS trading day.
                       NaN for the first day in the dataset.
      day_open       — the Open price of the very first candle of that day.

    These two values are used to compute the gap percentage:
      gap = (day_open - prev_day_close) / prev_day_close

    A positive gap (day opened above prior close) signals bullish bias.
    A negative gap signals bearish bias — the ORB direction filter uses this.
    """
    prev_close_series = pd.Series(np.nan, index=df.index)
    day_open_series   = pd.Series(np.nan, index=df.index)

    dates = sorted(set(df.index.date))

    for i, date in enumerate(dates):
        today_mask = df.index.date == date
        today_idx  = df[today_mask].index

        # Day open = Open price of the first candle of this day
        today_df  = df.loc[today_idx].sort_index()
        if today_df.empty:
            continue
        day_open_series.loc[today_idx] = float(today_df["Open"].iloc[0])

        # Prev day close = last Close of the prior trading day
        if i == 0:
            continue   # No prior day available in the dataset
        prev_date = dates[i - 1]
        prev_mask = df.index.date == prev_date
        prev_df   = df[prev_mask].sort_index()
        if prev_df.empty:
            continue
        prev_close = float(prev_df["Close"].iloc[-1])
        prev_close_series.loc[today_idx] = prev_close

    return prev_close_series, day_open_series


# ---------------------------------------------------------------------------
# Public interface  — add all indicators to a DataFrame
# ---------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and append all indicators to a copy of df.

    Designed to work on multi-day DataFrames so that EMA/RSI have
    sufficient history for proper warmup. VWAP, ORB, PREV_DAY_CLOSE and
    DAY_OPEN all reset/recalculate each day via their per-day logic.
    """
    df = df.copy()

    df[EMA_FAST_COL]  = _ema(df["Close"], EMA_FAST)
    df[EMA_SLOW_COL]  = _ema(df["Close"], EMA_SLOW)
    df[EMA_MACRO_COL] = _ema(df["Close"], EMA_MACRO)   # 50 EMA — macro trend
    df[VWAP_COL]      = _vwap_daily(df)
    df[RSI_COL]       = _rsi(df["Close"])
    df[VOLAVG_COL]    = df["Volume"].rolling(window=VOLUME_LOOKBACK).mean()

    orb_h, orb_l, orb_est = _opening_range(df)
    df[ORB_HIGH_COL]        = orb_h
    df[ORB_LOW_COL]         = orb_l
    df[ORB_ESTABLISHED_COL] = orb_est

    prev_close, day_open = _prev_day_close_and_day_open(df)
    df[PREV_DAY_CLOSE_COL]  = prev_close
    df[DAY_OPEN_COL]        = day_open

    return df
