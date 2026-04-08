"""
strategy_alpha_combo.py
-----------------------
NSE Intraday Alpha Combination Strategy — 7-Signal Ensemble.

THEORETICAL BASIS
─────────────────
The Fundamental Law of Active Management (Grinold & Kahn):
    IR = IC × sqrt(N)

Where:
    IR = Information Ratio (risk-adjusted edge of the combined system)
    IC = average Information Coefficient of individual signals
    N  = number of GENUINELY INDEPENDENT signals

Our 7 signals at IC = 0.07 average:
    IR = 0.07 × sqrt(7) = 0.185

vs a single strong signal at IC = 0.12:
    IR = 0.12

The 7-signal system delivers 54% more risk-adjusted edge despite each
component being individually weaker. This is why hedge fund desks combine
hundreds of signals rather than searching for the one perfect indicator.

SIMPLIFIED 11-STEP ALPHA COMBINATION PROCEDURE
───────────────────────────────────────────────
The full institutional procedure (Steps 1-11 from the article) is adapted
for live intraday deployment where only a few hundred observations are
available per day:

  Step 1:  Record realized returns per completed bar.
  Step 2:  Demean returns (remove systematic market drift).
  Step 3:  Compute variance σ² for each signal's score series.
  Step 4:  Normalize signal scores by σ (unit-variance scaling).
  Step 9:  Pearson correlation(scores_i, returns) → IC_i.
           This is the "residual after removing shared variance" proxy —
           Pearson IC measures independent predictive power directly.
  Step 10: weight_i ∝ max(IC_i, 0) / σ_i
           (independent edge / noise penalization, per article Step 10).
  Step 11: Normalize so Σ|weights| = 1.

  Steps 5–8 (full cross-sectional demeaning and regression) are partially
  absorbed by the VWAP deviation signal (which is already cross-sectionally
  normalized relative to the session fair value).

  Stability blend: weights = 60% IC-derived + 40% base priors, then renorm.
  This prevents weights from collapsing to a single signal in short regimes.

IC WEIGHT PERSISTENCE
─────────────────────
Updated weights are saved to alpha_weights.json at end of each session.
On startup, persisted weights are loaded so the system improves every day.

ENTRY LOGIC
───────────
    combined_alpha = Σ(w_i × score_i) / Σ(w_i for active scores)

  |alpha| >= ALPHA_ENTRY_THRESHOLD → trade in the direction of the sign.
  Position size scales with |alpha| (conviction sizing).

EXIT LOGIC (priority order)
────────────────────────────
  1. TARGET      — close reaches entry ± ATR × ALPHA_TARGET_RR
  2. STOP_LOSS   — close breaches entry ∓ ATR × ALPHA_ATR_SL_MULT
  3. ALPHA_EXIT  — alpha score reverses past ALPHA_EXIT_THRESHOLD
  4. EMA_FLIP    — 9 EMA crosses 50 EMA against the trade (macro trend flip)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from config import (
    ALPHA_ATR_LOOKBACK,
    ALPHA_ATR_SL_MULT,
    ALPHA_ENTRY_CUTOFF_TIME,
    ALPHA_ENTRY_THRESHOLD,
    ALPHA_EXIT_THRESHOLD,
    ALPHA_IC_WINDOW,
    ALPHA_MOMENTUM_LOOKBACK,
    ALPHA_TARGET_RR,
    ALPHA_WEIGHTS,
)
from indicators import EMA_FAST_COL, EMA_MACRO_COL
from signal_library import SIGNAL_NAMES, compute_all_signals

IST           = pytz.timezone("Asia/Kolkata")
logger        = logging.getLogger(__name__)
STRATEGY_NAME = "ALPHA_COMBO"

_HOLD         = {"action": "HOLD", "sl": 0.0, "target": 0.0}
_WEIGHTS_FILE = Path(__file__).parent / "alpha_weights.json"


# ---------------------------------------------------------------------------
# IC Weight Tracker — simplified 11-step combination engine
# ---------------------------------------------------------------------------

class ICWeightTracker:
    """
    Maintains rolling Pearson IC estimates for each signal and updates
    optimal combination weights following the institutional 11-step procedure.

    One singleton instance is shared across all symbols in a session.
    Observations from ALL symbols are pooled, which:
      a) Increases the sample size N rapidly (20 stocks × 90 bars = 1800/day)
      b) Gives the IC estimate cross-sectional breadth, matching Step 6 intent
         of the institutional procedure.

    Usage
    -----
    record(scores_dict, realized_return)   → call after each completed bar
    update_weights()                       → call at end of session
    save() / load()                        → JSON persistence
    combine(scores_dict)                   → returns alpha ∈ [-1, +1]
    """

    def __init__(self, base_weights: dict, window: int = 200) -> None:
        self.window       = window
        self.base_weights = {k: float(v) for k, v in base_weights.items()}
        self.weights      = dict(self.base_weights)
        # Rolling deque: each entry is {signal_name: score, ..., "_ret": return}
        self.observations: list[dict] = []

    # ------------------------------------------------------------------
    # Data accumulation
    # ------------------------------------------------------------------

    def record(self, scores: dict, realized_return: float) -> None:
        """
        Record one bar's signal scores alongside the realized return of
        the NEXT bar. Call this each time a bar closes on an open position.
        """
        entry = {k: scores.get(k, 0.0) for k in SIGNAL_NAMES}
        entry["_ret"] = float(realized_return)
        self.observations.append(entry)
        if len(self.observations) > self.window:
            self.observations.pop(0)

    # ------------------------------------------------------------------
    # Weight update (end of day)
    # ------------------------------------------------------------------

    def update_weights(self) -> dict:
        """
        Recompute IC-optimal weights from accumulated observations.

        Implements Steps 2–4, 9–11 of the 11-step combination engine:
          Step 2:  Demean returns
          Step 3:  Variance per signal
          Step 4:  Normalize signals
          Step 9:  Pearson correlation = IC
          Step 10: weight ∝ max(IC, 0) / σ
          Step 11: Normalize, blend with base weights for stability

        Returns a dict of computed ICs for logging.
        """
        n = len(self.observations)
        if n < 30:
            logger.info(f"ALPHA: only {n} observations — skipping weight update (need 30+)")
            return {}

        returns = np.array([o["_ret"] for o in self.observations])

        # Step 2: demean returns to remove systematic market drift
        returns_dm = returns - returns.mean()

        ics:  dict[str, float] = {}
        stds: dict[str, float] = {}

        for name in SIGNAL_NAMES:
            scores_arr = np.array([o.get(name, 0.0) for o in self.observations])

            # Step 3 & 4: standard deviation (σ) for normalization
            std = float(scores_arr.std())
            stds[name] = std if std > 1e-8 else 1.0

            # Step 9: Pearson IC = corr(signal_scores, next_bar_return)
            if std > 1e-8 and returns_dm.std() > 1e-8:
                corr_matrix = np.corrcoef(scores_arr, returns_dm)
                ic = float(corr_matrix[0, 1])
            else:
                ic = 0.0

            ics[name] = ic

        # Step 10: weight ∝ max(IC, 0) / σ
        raw_weights = {k: max(ics[k], 0.0) / stds[k] for k in SIGNAL_NAMES}
        total = sum(raw_weights.values())

        if total > 1e-8:
            ic_weights = {k: v / total for k, v in raw_weights.items()}
        else:
            # All ICs are non-positive — unusual regime; fall back to flat
            logger.warning("ALPHA: all signal ICs are non-positive — keeping base weights")
            ic_weights = {k: 1.0 / len(SIGNAL_NAMES) for k in SIGNAL_NAMES}

        # Step 11: blend 60% IC-derived + 40% base priors, then renormalize
        blended = {
            k: 0.60 * ic_weights[k] + 0.40 * self.base_weights.get(k, 0.0)
            for k in SIGNAL_NAMES
        }
        total_blend = sum(blended.values())
        if total_blend > 1e-8:
            self.weights = {k: v / total_blend for k, v in blended.items()}

        logger.info(
            f"ALPHA weights updated ({n} obs) | "
            f"IC: {_fmt(ics)} | "
            f"weights: {_fmt(self.weights)}"
        )
        return ics

    # ------------------------------------------------------------------
    # Combination
    # ------------------------------------------------------------------

    def combine(self, scores: dict) -> float:
        """
        Weighted combination of 7 signal scores → alpha ∈ [-1.0, +1.0].

        The denominator is the sum of weights for signals that actually fired
        (non-zero score). Signals that cannot fire in the current bar (e.g.,
        ORB before the window closes) do NOT dilute the combined score — their
        weight is excluded from the denominator so remaining signals retain
        full conviction.

        This matches the article's emphasis on N being the effective count
        of active independent signals, not the nominal count.
        """
        num   = sum(self.weights.get(k, 0.0) * scores.get(k, 0.0) for k in SIGNAL_NAMES)
        denom = sum(self.weights.get(k, 0.0) for k in SIGNAL_NAMES
                    if abs(scores.get(k, 0.0)) > 1e-8)
        if denom < 1e-8:
            return 0.0
        return float(np.clip(num / denom, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist current weights to JSON so they survive between sessions."""
        try:
            with open(_WEIGHTS_FILE, "w") as f:
                json.dump({"weights": self.weights, "n_obs": len(self.observations)}, f, indent=2)
            logger.info(f"ALPHA weights saved → {_WEIGHTS_FILE.name}")
        except OSError as e:
            logger.warning(f"Could not save alpha weights: {e}")

    def load(self) -> None:
        """Load persisted weights from the previous session if available."""
        if not _WEIGHTS_FILE.exists():
            return
        try:
            with open(_WEIGHTS_FILE) as f:
                data = json.load(f)
            loaded = data.get("weights", {})
            if set(loaded.keys()) == set(SIGNAL_NAMES):
                self.weights = {k: float(v) for k, v in loaded.items()}
                prev_obs = data.get("n_obs", 0)
                logger.info(
                    f"ALPHA: loaded persisted weights (prev session had {prev_obs} obs) "
                    f"| {_fmt(self.weights)}"
                )
            else:
                logger.warning("ALPHA: persisted weight keys mismatch — using defaults")
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"ALPHA: could not load persisted weights: {e}")


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all symbols in one session
# ---------------------------------------------------------------------------

_tracker = ICWeightTracker(base_weights=ALPHA_WEIGHTS, window=ALPHA_IC_WINDOW)
_tracker.load()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(d: dict) -> str:
    """Format a dict of floats for compact logging."""
    return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in d.items()) + "}"


def _compute_atr(df: pd.DataFrame, lookback: int) -> float:
    """
    Average True Range over `lookback` completed candles.

    Uses iloc[-2-lookback : -1] — the last `lookback` closed candles,
    excluding iloc[-1] (currently forming) and iloc[-2] (signal candle).
    """
    if len(df) < lookback + 2:
        return 0.0

    window = df.iloc[-2 - lookback: -1].copy()
    prev_close = window["Close"].shift(1)

    tr = pd.concat([
        window["High"] - window["Low"],
        (window["High"] - prev_close).abs(),
        (window["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    val = float(tr.mean())
    return val if np.isfinite(val) else 0.0


# ---------------------------------------------------------------------------
# Public strategy interface  (matches the contract expected by strategy_factory)
# ---------------------------------------------------------------------------

def generate_signal(df: pd.DataFrame, symbol: str = "", sim_time=None) -> dict:
    """
    Evaluate the last completed candle using the 7-signal alpha combination.

    Returns one of:
      BUY  signal: {"action": "BUY",  "sl": float, "target": float,
                    "strategy": "ALPHA_COMBO", "alpha_score": float,
                    "signal_scores": dict, "quantity_scale": float}
      SELL signal: same structure with action="SELL"
      HOLD:        {"action": "HOLD", "sl": 0.0, "target": 0.0}

    quantity_scale ∈ (0, 1]: multiply POSITION_SIZE_INR by this value when
    sizing the order. Stronger alpha conviction → larger position.

    sim_time: override datetime.now() for backtesting.
    """
    # 1. Entry cutoff gate
    now_ist = sim_time if sim_time is not None else datetime.now(IST)
    if hasattr(now_ist, "tzinfo") and now_ist.tzinfo is None:
        now_ist = IST.localize(now_ist)
    h, m = map(int, ALPHA_ENTRY_CUTOFF_TIME.split(":"))
    if now_ist >= now_ist.replace(hour=h, minute=m, second=0, microsecond=0):
        return _HOLD

    if len(df) < 20:
        return _HOLD

    # 2. Compute all 7 signal scores
    scores = compute_all_signals(df, momentum_lookback=ALPHA_MOMENTUM_LOOKBACK)

    # 3. Weighted combination → alpha score ∈ [-1, +1]
    alpha = _tracker.combine(scores)

    logger.info(
        f"{symbol} ALPHA score={alpha:+.3f} | "
        + " | ".join(f"{k}={v:+.2f}" for k, v in scores.items())
    )

    if abs(alpha) < ALPHA_ENTRY_THRESHOLD:
        return _HOLD

    # 4. ATR-based stop-loss and target
    atr = _compute_atr(df, ALPHA_ATR_LOOKBACK)
    entry_price = float(df.iloc[-2]["Close"])

    if atr <= 0:
        atr = entry_price * 0.008   # fallback: 0.8% proxy

    sl_distance     = atr * ALPHA_ATR_SL_MULT
    target_distance = sl_distance * ALPHA_TARGET_RR

    # 5. Conviction-proportional position scale
    quantity_scale = round(float(np.clip(abs(alpha), 0.3, 1.0)), 2)

    action = "BUY" if alpha > 0 else "SELL"
    sl     = (entry_price - sl_distance) if action == "BUY" else (entry_price + sl_distance)
    target = (entry_price + target_distance) if action == "BUY" else (entry_price - target_distance)
    rr     = target_distance / sl_distance if sl_distance > 0 else 0.0

    logger.info(
        f"{symbol} ALPHA *** {action} *** "
        f"entry={entry_price:.2f} sl={sl:.2f} target={target:.2f} "
        f"R:R={rr:.1f}x | alpha={alpha:+.3f} | scale={quantity_scale:.2f} "
        f"| atr={atr:.2f}"
    )

    return {
        "action":         action,
        "sl":             sl,
        "target":         target,
        "strategy":       STRATEGY_NAME,
        "alpha_score":    alpha,
        "signal_scores":  scores,
        "quantity_scale": quantity_scale,
    }


def check_exit_signal(df: pd.DataFrame, position: dict) -> str | None:
    """
    Exit conditions for an open ALPHA_COMBO position.

    Priority:
      1. TARGET      — close touches / exceeds calculated target
      2. STOP_LOSS   — close touches / breaches stop-loss level
      3. ALPHA_EXIT  — combined alpha score reverses past ALPHA_EXIT_THRESHOLD
      4. EMA_FLIP    — 9 EMA crosses the 50 EMA against trade direction

    Uses iloc[-2] (last completed candle), matching all other strategy modules.
    """
    if len(df) < 3:
        return None

    row       = df.iloc[-2]
    close     = float(row["Close"])
    direction = position["direction"]
    target    = float(position["target"])
    sl        = float(position["sl"])

    # 1. Target
    if direction == "BUY"  and close >= target:
        return "TARGET"
    if direction == "SELL" and close <= target:
        return "TARGET"

    # 2. Stop-loss
    if direction == "BUY"  and close <= sl:
        return "STOP_LOSS"
    if direction == "SELL" and close >= sl:
        return "STOP_LOSS"

    # 3. Alpha score reversal
    scores = compute_all_signals(df, momentum_lookback=ALPHA_MOMENTUM_LOOKBACK)
    alpha  = _tracker.combine(scores)

    if direction == "BUY"  and alpha <= -ALPHA_EXIT_THRESHOLD:
        logger.info(
            f"ALPHA_EXIT: score reversed to {alpha:+.3f} "
            f"(threshold -{ALPHA_EXIT_THRESHOLD:.2f}) against BUY"
        )
        return "ALPHA_EXIT"
    if direction == "SELL" and alpha >= ALPHA_EXIT_THRESHOLD:
        logger.info(
            f"ALPHA_EXIT: score reversed to {alpha:+.3f} "
            f"(threshold +{ALPHA_EXIT_THRESHOLD:.2f}) against SELL"
        )
        return "ALPHA_EXIT"

    # 4. EMA macro flip (9 EMA crosses the 50 EMA)
    ema9  = row.get(EMA_FAST_COL)
    ema50 = row.get(EMA_MACRO_COL)
    if not any(pd.isna(x) for x in (ema9, ema50)):
        ema9  = float(ema9)
        ema50 = float(ema50)
        if direction == "BUY"  and ema9 < ema50:
            return "EMA_FLIP"
        if direction == "SELL" and ema9 > ema50:
            return "EMA_FLIP"

    return None


# ---------------------------------------------------------------------------
# IC tracker public API (called from main.py)
# ---------------------------------------------------------------------------

def record_outcome(signal_scores: dict, direction: str, entry_price: float,
                   exit_price: float) -> None:
    """
    Record a completed trade's signal scores and realized return for IC tracking.

    Call from main.py's check_exits() immediately after a position is closed.
    The return is computed here from (direction, entry_price, exit_price).

    Parameters
    ----------
    signal_scores : dict
        The signal scores dict stored in the Position at entry time.
    direction : str
        "BUY" or "SELL" (the direction of the closed position).
    entry_price : float
        Trade entry price.
    exit_price : float
        Trade exit price.
    """
    if not signal_scores:
        return
    direction_mult = 1.0 if direction == "BUY" else -1.0
    ret = direction_mult * (exit_price - entry_price) / entry_price
    _tracker.record(signal_scores, ret)


def end_of_day_weight_update() -> dict:
    """
    Update IC-based signal weights using today's trade observations.
    Persist updated weights to JSON.

    Call once at session close (after SQUARE_OFF_TIME and all exits).
    Returns dict of computed IC values (for logging / audit trail).
    """
    ics = _tracker.update_weights()
    _tracker.save()
    return ics
