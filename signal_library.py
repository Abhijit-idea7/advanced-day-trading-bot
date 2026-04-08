"""
signal_library.py
-----------------
Seven independent alpha signals for the NSE intraday Alpha Combination strategy.

Each signal function returns a score in [-1.0, +1.0]:
  +1.0 = strong bullish evidence
  -1.0 = strong bearish evidence
   0.0 = no information / neutral

Signal taxonomy (maps to the article's five institutional signal categories):

  CATEGORY               SIGNAL              BASE IC   INDEPENDENCE RATIONALE
  ─────────────────────────────────────────────────────────────────────────────
  Price / Structure      orb                 0.10      Purely structural; fires on
                                                       first post-range candle.
  Behavioural / Flow     gap                 0.08      Overnight institutional bias;
                                                       fully resolved before open.
  Mean Reversion         vwap_deviation      0.07      Fair-value anchor; orthogonal
                                                       to trend signals.
  Factor (Trend)         ema_trend           0.07      EMA alignment = session-long
                                                       trend factor exposure.
  Price / Momentum       momentum            0.06      Rate-of-change over 10 bars;
                                                       different timescale from EMA.
  Microstructure         volume_pressure     0.06      Volume aggression; only
                                                       signal using raw volume.
  Momentum Quality       rsi                 0.05      RSI = independent momentum
                                                       oscillator (Wilder smoothing).

Fundamental Law of Active Management:
    IR = IC × sqrt(N)
    7 signals at IC = 0.07 avg → IR = 0.07 × sqrt(7) = 0.185
    Single signal at IC = 0.12 → IR = 0.12
    Combined system is 54% stronger despite each component being weaker.

References:
  "The Math Behind Combining 50 Weak Signals Into One Winning Trade" (Roan, 2025)
  Grinold & Kahn, "Active Portfolio Management" (Fundamental Law)
"""

import numpy as np
import pandas as pd

from indicators import (
    DAY_OPEN_COL,
    EMA_FAST_COL,
    EMA_MACRO_COL,
    EMA_SLOW_COL,
    ORB_ESTABLISHED_COL,
    ORB_HIGH_COL,
    ORB_LOW_COL,
    PREV_DAY_CLOSE_COL,
    RSI_COL,
    VOLAVG_COL,
    VWAP_COL,
)

# Ordered list of all signal names — order matters for array operations in ICWeightTracker
SIGNAL_NAMES = [
    "orb",
    "gap",
    "vwap_deviation",
    "ema_trend",
    "momentum",
    "volume_pressure",
    "rsi",
]


def _tanh_scale(x: float, scale: float = 1.0) -> float:
    """Squash x × scale through tanh to guarantee output ∈ (-1, +1)."""
    return float(np.tanh(x * scale))


# ---------------------------------------------------------------------------
# Signal 1: ORB — Opening Range Breakout (structural momentum)
# ---------------------------------------------------------------------------

def orb_score(row: pd.Series) -> float:
    """
    Score based on where price sits relative to the first-15-min opening range.

    After ORB is established (post 9:30 IST):
      Price above ORB high → positive  (breakout confirmed)
      Price below ORB low  → negative  (breakdown confirmed)
      Price inside range   → weak ±0.25 (positioning within range)

    Scaling: extension of 1× ORB range beyond the boundary → tanh(3) ≈ +0.995.
    At 0.33× range extension → tanh(1) ≈ +0.76.

    Independence: this is the only signal that uses the ORB high/low levels
    directly. Gap signal uses prev_close/day_open, VWAP uses the session average.

    IC proxy: 0.10 — strongest structural signal in the first two hours.
    """
    if not bool(row.get(ORB_ESTABLISHED_COL, False)):
        return 0.0

    orb_high = row.get(ORB_HIGH_COL)
    orb_low  = row.get(ORB_LOW_COL)
    if pd.isna(orb_high) or pd.isna(orb_low):
        return 0.0

    orb_high  = float(orb_high)
    orb_low   = float(orb_low)
    orb_range = orb_high - orb_low
    if orb_range <= 0:
        return 0.0

    close = float(row["Close"])

    if close > orb_high:
        extension = (close - orb_high) / orb_range
        return _tanh_scale(extension, scale=3.0)

    if close < orb_low:
        extension = (orb_low - close) / orb_range
        return -_tanh_scale(extension, scale=3.0)

    # Inside range: weak directional signal (position within range)
    mid = (orb_high + orb_low) / 2.0
    return float(np.clip((close - mid) / (orb_range / 2.0), -1.0, 1.0)) * 0.25


# ---------------------------------------------------------------------------
# Signal 2: GAP — Opening gap direction (institutional/behavioural bias)
# ---------------------------------------------------------------------------

def gap_score(row: pd.Series, session_elapsed_min: float = 0.0) -> float:
    """
    Gap between today's opening price and yesterday's close, with time decay.

    This captures the net effect of overnight institutional positioning,
    futures carry, and news events processed outside cash hours.

    TIME DECAY — the gap loses relevance as the session progresses:
      At 9:15 IST (session open, t=0):    weight = 1.0  (full strength)
      At 12:00 IST (165 min in):          weight = 0.30 (30% residual)
      After 12:00 IST:                    weight = 0.30 (floors there)

    Scaling: 1% gap up at open → tanh(1) ≈ +0.76; same gap at 12:00 → +0.23.

    Independence: uses prev_day_close and day_open — purely pre-session prices
    unavailable to any intrabar signal.

    IC proxy: 0.08 at open; effectively ~0.024 by midday.
    Category: Behavioural Bias / Institutional Flow
    """
    prev_close = row.get(PREV_DAY_CLOSE_COL)
    day_open   = row.get(DAY_OPEN_COL)

    if pd.isna(prev_close) or pd.isna(day_open) or float(prev_close) <= 0:
        return 0.0

    gap_pct   = (float(day_open) - float(prev_close)) / float(prev_close)
    raw_score = _tanh_scale(gap_pct, scale=100.0)

    # Linear decay from 1.0 → 0.30 over 165 min (9:15 to 12:00), then floors
    decay = max(0.30, 1.0 - 0.70 * min(session_elapsed_min, 165.0) / 165.0)

    return raw_score * decay


# ---------------------------------------------------------------------------
# Signal 3: VWAP Deviation — mean reversion / fair-value signal
# ---------------------------------------------------------------------------

def vwap_deviation_score(row: pd.Series) -> float:
    """
    Signed percentage deviation of closing price from daily-anchored VWAP.

    Above VWAP → bullish (trend continuation / premium over fair value)
    Below VWAP → bearish (trend continuation / discount to fair value)

    Scaling: +0.5% above VWAP → +0.46; +1.0% → +0.76; +2.0% → +0.96.

    Independence: VWAP is a cumulative average of all-session volume × price,
    unlike EMA (exponential price-only average) or ORB (range extremes).

    IC proxy: 0.07 — steady across the full session window.
    """
    vwap  = row.get(VWAP_COL)
    close = float(row["Close"])

    if pd.isna(vwap) or float(vwap) <= 0:
        return 0.0

    deviation_pct = (close - float(vwap)) / float(vwap)
    return _tanh_scale(deviation_pct, scale=100.0)


# ---------------------------------------------------------------------------
# Signal 4: EMA Trend — factor / session trend direction
# ---------------------------------------------------------------------------

def ema_trend_score(row: pd.Series) -> float:
    """
    Score from 9 / 20 / 50 EMA alignment.

    Component weights:
      9 EMA vs 50 EMA: ±0.50  (macro session trend — most important)
      9 EMA vs 20 EMA: ±0.30  (short-term trend confirmation)
      20 EMA vs 50 EMA: ±0.20 (medium-term alignment)

    Full uptrend  (9 > 20 > 50): +1.00
    Full downtrend (9 < 20 < 50): -1.00
    Mixed signals produce intermediate scores.

    Independence: EMA uses pure closing-price exponential averaging,
    which is mathematically distinct from VWAP (volume-weighted cumulative
    average) and from the ORB/gap structural levels.

    IC proxy: 0.07 — reliable trend factor throughout the session.
    """
    ema9  = row.get(EMA_FAST_COL)
    ema20 = row.get(EMA_SLOW_COL)
    ema50 = row.get(EMA_MACRO_COL)

    if any(pd.isna(x) for x in (ema9, ema20, ema50)):
        return 0.0

    ema9, ema20, ema50 = float(ema9), float(ema20), float(ema50)

    score  = 0.50 * (1.0 if ema9  > ema50 else -1.0)   # macro
    score += 0.30 * (1.0 if ema9  > ema20 else -1.0)   # short-term
    score += 0.20 * (1.0 if ema20 > ema50 else -1.0)   # medium-term

    return float(score)   # already ∈ [-1.0, +1.0]


# ---------------------------------------------------------------------------
# Signal 5: Momentum — price rate-of-change
# ---------------------------------------------------------------------------

def momentum_score(df: pd.DataFrame, lookback: int = 10) -> float:
    """
    Simple rate-of-change of the closing price over `lookback` candles,
    evaluated on the last completed candle (iloc[-2]).

    momentum_score uses iloc[-2] vs iloc[-2 - lookback].

    Scaling: +2% ROC over 10 candles → tanh(1.0) ≈ +0.76.
             +4% ROC → tanh(2.0) ≈ +0.96.

    Independence: momentum captures trend inertia over a fixed lookback
    horizon — distinct from EMA (which weights recent closes) and RSI
    (which measures up/down gains separately via Wilder smoothing).

    IC proxy: 0.06 — reliable in trending regimes.
    """
    required = lookback + 2
    if len(df) < required:
        return 0.0

    current = float(df.iloc[-2]["Close"])
    past    = float(df.iloc[-2 - lookback]["Close"])

    if past <= 0:
        return 0.0

    roc = (current - past) / past
    return _tanh_scale(roc, scale=50.0)


# ---------------------------------------------------------------------------
# Signal 6: Volume Pressure — microstructure / order-flow signal
# ---------------------------------------------------------------------------

def volume_pressure_score(row: pd.Series, prev_close: float) -> float:
    """
    Directional volume signal: relative volume × price direction.

    Buying pressure  = up-close candle with volume above average → positive
    Selling pressure = down-close candle with volume above average → negative

    Scaling: 2× average volume on an up-close → tanh(1) ≈ +0.76.
             1× average volume → tanh(0) = 0 (neutral).

    Independence: the only signal that directly incorporates raw tick volume.
    EMA, RSI, momentum, VWAP all use price; ORB and gap use structural levels.
    Volume is orthogonal to all price-derived signals.

    IC proxy: 0.06 — provides microstructure confirmation layer.
    """
    volume  = float(row.get("Volume", 0))
    vol_avg = float(row.get(VOLAVG_COL, 0))
    close   = float(row["Close"])

    if vol_avg <= 0:
        return 0.0

    vol_ratio = volume / vol_avg
    direction = 1.0 if close >= prev_close else -1.0

    # vol_ratio - 1: at average volume = 0, at 2× average = 1
    return direction * _tanh_scale(vol_ratio - 1.0, scale=1.0)


# ---------------------------------------------------------------------------
# Signal 7: RSI — momentum quality / relative strength
# ---------------------------------------------------------------------------

def rsi_score(row: pd.Series) -> float:
    """
    RSI centered at 50, used as a directional momentum indicator (not
    as an overbought/oversold filter).

    RSI 70 → +0.64; RSI 80 → +0.83; RSI 50 → 0; RSI 30 → -0.64.

    NOTE: we treat RSI > 50 as bullish momentum and RSI < 50 as bearish —
    the opposite of mean-reversion usage. In a trending intraday context,
    high RSI confirms trend strength rather than signalling exhaustion.

    Independence: RSI uses Wilder's exponential smoothing of up vs down gains
    separately, making it mathematically distinct from all other signals.

    IC proxy: 0.05 — weakest individually but adds marginal independent N.
    """
    rsi = row.get(RSI_COL)
    if pd.isna(rsi):
        return 0.0

    normalized = (float(rsi) - 50.0) / 50.0   # ∈ [-1, +1]
    return _tanh_scale(normalized, scale=2.0)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_all_signals(df: pd.DataFrame, momentum_lookback: int = 10) -> dict:
    """
    Compute all 7 alpha signals for the last completed candle (iloc[-2]).

    Returns a dict mapping signal name → float score ∈ [-1.0, +1.0].
    Missing data produces 0.0 (neutral) — never excludes an observation.

    The gap signal automatically applies time decay using the timestamp of
    the signal candle (iloc[-2]) relative to 9:15 IST (session start).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe with all indicator columns already added by
        indicators.add_indicators().  Must have >= 3 rows.
    momentum_lookback : int
        Lookback period for momentum_score (default 10 candles).
    """
    zero_scores = {name: 0.0 for name in SIGNAL_NAMES}

    if len(df) < 3:
        return zero_scores

    row      = df.iloc[-2]   # last completed candle
    prev_row = df.iloc[-3]   # prior candle (for volume_pressure direction)

    # Compute session elapsed minutes for gap signal time decay.
    # Uses the signal candle's timestamp; falls back to 0 if not available.
    session_elapsed_min = 0.0
    try:
        ts = df.index[-2]
        session_elapsed_min = max(0.0, (ts.hour * 60 + ts.minute) - 9 * 60 - 15)
    except Exception:
        pass

    return {
        "orb":             orb_score(row),
        "gap":             gap_score(row, session_elapsed_min=session_elapsed_min),
        "vwap_deviation":  vwap_deviation_score(row),
        "ema_trend":       ema_trend_score(row),
        "momentum":        momentum_score(df, lookback=momentum_lookback),
        "volume_pressure": volume_pressure_score(row, float(prev_row["Close"])),
        "rsi":             rsi_score(row),
    }
