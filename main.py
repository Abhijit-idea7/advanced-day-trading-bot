"""
main.py
-------
Multi-strategy intraday trading bot entry point.

Supported strategies (configure via --strategy arg or ACTIVE_STRATEGY env var):
  ORB         — Opening Range Breakout  (fires 9:30–11:00 IST)
  VWAP_EMA    — VWAP + 9/20 EMA Pullback  (fires 9:20–12:30 IST)
  COMBINED    — ORB and VWAP_EMA simultaneously, first signal wins per symbol
  ALPHA_COMBO — 7-signal IC-weighted ensemble (fires 9:20–13:00 IST)
  ALPHA_ORB   — ORB morning breakouts + ALPHA_COMBO midday, shared position pool

Production default (trading.yml): runs ORB and ALPHA_COMBO as TWO separate
parallel processes, each with its own independent position pool and log file:
  python main.py --strategy ORB         → performance_log_ORB.csv
  python main.py --strategy ALPHA_COMBO → performance_log_ALPHA_COMBO.csv

Lifecycle (runs as a single long-lived process via GitHub Actions):
  1. GitHub Actions cron starts the runner at 01:00 UTC = 06:30 IST
  2. Script waits in 30-second poll until TRADE_START_TIME (09:20 IST)
  3. Selects today's top candidates by ATR% volatility ranking
  4. Loop every 2 minutes between 09:20 and 15:15 IST:
       a. Check exits for all open positions (target / SL / strategy-specific exit)
       b. Scan candidates for new entry signals (all active strategies)
  5. At 15:15 IST: force-close all open positions (SQUARE_OFF)
  6. If ALPHA_COMBO: update IC-based signal weights from today's trade data
  7. Print daily P&L summary and save to performance_log_<STRATEGY>.csv
  8. CSV is committed back to the repo by the GitHub Actions workflow step
"""

# ---------------------------------------------------------------------------
# IMPORTANT: parse --strategy and set os.environ BEFORE any other imports.
# config.py reads ACTIVE_STRATEGY from os.environ at import time, so this
# must happen first so both this process and all imported modules see the
# correct strategy name.
# ---------------------------------------------------------------------------
import argparse as _argparse
import os as _os
import sys as _sys

_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("--strategy", type=str, default=None)
_early_args, _ = _parser.parse_known_args()
if _early_args.strategy:
    _os.environ["ACTIVE_STRATEGY"] = _early_args.strategy.upper()

# ---------------------------------------------------------------------------
# Standard imports — config is now imported with the correct ACTIVE_STRATEGY
# ---------------------------------------------------------------------------
import logging
import time
from datetime import datetime

import pytz

from config import (
    ACTIVE_STRATEGY,
    LOOP_SLEEP_SECONDS,
    MAX_POSITIONS,
    ONE_TRADE_PER_STOCK_PER_DAY,
    ORB_MAX_POSITIONS,
    ORB_STOCK_UNIVERSE,
    ORB_TOP_N_STOCKS,
    SQUARE_OFF_TIME,
    TRADE_START_TIME,
)
from data_feed import fetch_candles_for_warmup, get_top_candidates
from indicators import add_indicators
from order_manager import calculate_quantity, place_order, square_off
from performance_tracker import PerformanceTracker
from strategy_factory import get_strategies, get_strategy_name
from trade_tracker import TradeTracker

# ---------------------------------------------------------------------------
# Logging — structured output goes to GitHub Actions console
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

IST         = pytz.timezone("Asia/Kolkata")
MIN_CANDLES = 20   # Minimum candles for indicators to be meaningful


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ist_now() -> datetime:
    return datetime.now(IST)


def current_time_str() -> str:
    return ist_now().strftime("%H:%M")


def is_past(hhmm: str) -> bool:
    now  = ist_now()
    h, m = map(int, hhmm.split(":"))
    return now >= now.replace(hour=h, minute=m, second=0, microsecond=0)


def fetch_and_prepare(symbol: str):
    """
    Fetch 5 days of 2-min candles for EMA/RSI warmup, compute all indicators,
    then return only today's candles with indicators for signal generation.

    The multi-day fetch (without today-only filter) is critical for the EMA
    and ORB indicators to be properly seeded before today's trading session.
    """
    df_full = fetch_candles_for_warmup(symbol, period="5d")
    if df_full is None or len(df_full) < MIN_CANDLES:
        logger.info(
            f"{symbol}: only {len(df_full) if df_full is not None else 0} candles "
            f"— need {MIN_CANDLES}, skipping."
        )
        return None
    try:
        df_ind   = add_indicators(df_full)
        today    = df_ind.index[-1].date()
        df_today = df_ind[df_ind.index.date == today]
        if len(df_today) < 3:
            logger.info(f"{symbol}: only {len(df_today)} candles today — waiting.")
            return None
        return df_today
    except Exception as e:
        logger.warning(f"{symbol}: indicator calculation failed — {e}")
        return None


# ---------------------------------------------------------------------------
# Exit management
# ---------------------------------------------------------------------------

def check_exits(
    tracker:      TradeTracker,
    perf:         PerformanceTracker,
    closed_today: set,
    strategies:   list,
) -> None:
    """Evaluate all open positions and close any that hit their exit condition."""
    for position in tracker.all_positions():
        symbol = position.symbol
        try:
            df = fetch_and_prepare(symbol)
            if df is None:
                continue

            # Route to the strategy module that opened this position
            strategy_module = next(
                (s for s in strategies if get_strategy_name(s) == position.strategy_name),
                strategies[0],   # fallback to first strategy if not found
            )

            reason = strategy_module.check_exit_signal(df, position.__dict__)
            if reason:
                exit_price = float(df["Close"].iloc[-2])
                ok = square_off(symbol, position.direction, position.quantity)
                if ok:
                    # Record outcome for ALPHA_COMBO IC tracking
                    if position.strategy_name == "ALPHA_COMBO":
                        from strategy_alpha_combo import record_outcome
                        record_outcome(
                            signal_scores = position.signal_scores,
                            direction     = position.direction,
                            entry_price   = position.entry_price,
                            exit_price    = exit_price,
                        )
                    tracker.record_closed_pnl(
                        position.entry_price, exit_price,
                        position.quantity, position.direction,
                    )
                    tracker.remove_position(symbol)
                    closed_today.add(symbol)
                    perf.record_trade(
                        symbol      = symbol,
                        direction   = position.direction,
                        entry_price = position.entry_price,
                        exit_price  = exit_price,
                        quantity    = position.quantity,
                        entry_time  = position.entry_time,
                        exit_reason = reason,
                        strategy    = position.strategy_name,
                    )
        except Exception as e:
            logger.error(f"Error checking exit for {symbol}: {e}")


def square_off_all(
    tracker:      TradeTracker,
    perf:         PerformanceTracker,
    closed_today: set,
) -> None:
    """
    Force-close every open position at SQUARE_OFF_TIME.

    Retries each square_off() up to 3 times (5s apart) before giving up.
    Even on failure the trade is recorded at last known price with reason
    SQUARE_OFF_FAILED so P&L tracking remains consistent. Zerodha will
    close the position via MIS auto-square-off at ~15:20 regardless.
    """
    logger.info("=== SQUARE-OFF TIME: closing all open positions ===")
    for position in tracker.all_positions():
        try:
            df         = fetch_and_prepare(position.symbol)
            exit_price = (
                float(df["Close"].iloc[-2]) if df is not None else position.entry_price
            )

            # Retry up to 3 times — webhook can fail transiently
            ok = False
            for attempt in range(1, 4):
                ok = square_off(position.symbol, position.direction, position.quantity)
                if ok:
                    break
                logger.warning(
                    f"{position.symbol}: square_off attempt {attempt}/3 failed"
                    + (", retrying in 5s..." if attempt < 3 else " — giving up.")
                )
                if attempt < 3:
                    time.sleep(5)

            reason = "SQUARE_OFF" if ok else "SQUARE_OFF_FAILED"
            if not ok:
                logger.error(
                    f"{position.symbol}: all square_off attempts failed. "
                    f"Zerodha will close via MIS auto-square-off at ~15:20. "
                    f"Recording P&L at last known price Rs{exit_price:.2f}."
                )

            tracker.record_closed_pnl(
                position.entry_price, exit_price,
                position.quantity, position.direction,
            )
            tracker.remove_position(position.symbol)
            closed_today.add(position.symbol)
            perf.record_trade(
                symbol      = position.symbol,
                direction   = position.direction,
                entry_price = position.entry_price,
                exit_price  = exit_price,
                quantity    = position.quantity,
                entry_time  = position.entry_time,
                exit_reason = reason,
                strategy    = position.strategy_name,
            )
        except Exception as e:
            logger.error(f"Error squaring off {position.symbol}: {e}")
    logger.info("Square-off complete.")


# ---------------------------------------------------------------------------
# Entry management
# ---------------------------------------------------------------------------

def scan_for_entries(
    candidates:            list[str],
    tracker:               TradeTracker,
    closed_today:          set,
    strategies:            list,
    regime:                dict | None = None,
    max_positions_override: int | None = None,
) -> None:
    """
    Check each candidate against all active strategies for a fresh entry signal.

    regime                 : dict from market_regime.get_nifty_regime()
    max_positions_override : use strategy-specific cap (e.g. ORB_MAX_POSITIONS)
                             instead of the regime/config default
    """
    regime            = regime or {}
    effective_max     = max_positions_override or regime.get("max_positions", MAX_POSITIONS)
    direction_filter  = regime.get("direction_filter", "BOTH")

    for symbol in candidates:
        if tracker.open_count() >= effective_max:
            logger.info(
                f"Position cap reached ({tracker.open_count()}/{effective_max} "
                f"regime={regime.get('regime', 'N/A')}) — pausing entries."
            )
            break

        if tracker.has_position(symbol):
            continue

        if ONE_TRADE_PER_STOCK_PER_DAY and symbol in closed_today:
            continue

        try:
            df = fetch_and_prepare(symbol)
            if df is None:
                continue

            # Try each active strategy in order — take the first valid signal
            for strategy_module in strategies:
                signal = strategy_module.generate_signal(df, symbol=symbol)

                if signal["action"] not in ("BUY", "SELL"):
                    continue

                # Apply NIFTY regime direction filter
                if direction_filter == "LONG_ONLY"  and signal["action"] == "SELL":
                    logger.info(f"{symbol}: SELL blocked — BULL regime (LONG_ONLY)")
                    continue
                if direction_filter == "SHORT_ONLY" and signal["action"] == "BUY":
                    logger.info(f"{symbol}: BUY blocked — BEAR regime (SHORT_ONLY)")
                    continue

                entry_price    = float(df["Close"].iloc[-2])
                quantity_scale = signal.get("quantity_scale", 1.0)
                quantity       = calculate_quantity(entry_price, scale=quantity_scale)

                if quantity < 1:
                    logger.warning(f"{symbol}: qty rounds to 0 at Rs{entry_price:.2f}, skipping.")
                    break

                ok = place_order(symbol, signal["action"], quantity)
                if ok:
                    tracker.add_position(
                        symbol        = symbol,
                        direction     = signal["action"],
                        entry_price   = entry_price,
                        sl            = signal["sl"],
                        target        = signal["target"],
                        quantity      = quantity,
                        strategy_name = get_strategy_name(strategy_module),
                        signal_scores = signal.get("signal_scores", {}),
                    )
                break  # one signal per symbol per loop tick

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    logger.info("=" * 65)
    logger.info(f"Advanced Multi-Strategy Intraday Bot")
    logger.info(f"Active strategy : {ACTIVE_STRATEGY}")
    logger.info(f"Trade window    : {TRADE_START_TIME} — {SQUARE_OFF_TIME} IST")
    logger.info(f"Max positions   : {MAX_POSITIONS}")
    logger.info("=" * 65)

    strategies = get_strategies()

    # ORB uses its own larger universe and position cap
    is_orb = (ACTIVE_STRATEGY.upper() == "ORB")
    effective_max_positions = ORB_MAX_POSITIONS if is_orb else MAX_POSITIONS

    logger.info(f"Max positions   : {effective_max_positions}")

    # Wait for trade start (GitHub Actions runner may start 30+ min early)
    while not is_past(TRADE_START_TIME):
        logger.info(f"Waiting for {TRADE_START_TIME} IST... (now {current_time_str()})")
        time.sleep(30)

    # Select today's candidates once at session start
    # ORB scans a larger gap-optimised universe; ALPHA_COMBO uses default
    logger.info("Selecting today's top candidates by ATR%...")
    if is_orb:
        candidates = get_top_candidates(
            universe=ORB_STOCK_UNIVERSE,
            top_n=ORB_TOP_N_STOCKS,
        )
    else:
        candidates = get_top_candidates()
    logger.info(f"Watchlist ({len(candidates)} stocks): {candidates}")

    # Strategy-specific log file so parallel processes don't conflict
    log_file     = f"performance_log_{ACTIVE_STRATEGY}.csv"
    tracker      = TradeTracker()
    perf         = PerformanceTracker(log_file=log_file)
    closed_today: set = set()

    # Main strategy loop — runs every LOOP_SLEEP_SECONDS seconds
    while True:
        logger.info(f"--- Loop tick at {current_time_str()} IST [{ACTIVE_STRATEGY}] ---")

        # Hard square-off gate
        if is_past(SQUARE_OFF_TIME):
            square_off_all(tracker, perf, closed_today)
            break

        # 1. Fetch NIFTY50 market regime — active for ALL strategies:
        #    ORB        : direction_filter gates trade direction each loop tick
        #                 BULL  → LONG_ONLY  (block short breakdowns on bull days)
        #                 BEAR  → SHORT_ONLY (block long breakouts on bear days)
        #                 NEUTRAL → BOTH     (no restriction)
        #    ALPHA_COMBO: also uses alpha_threshold and max_positions from regime
        regime = None
        try:
            from market_regime import get_nifty_regime
            regime = get_nifty_regime()
            if "ALPHA_COMBO" in ACTIVE_STRATEGY.upper():
                from strategy_alpha_combo import set_regime
                set_regime(regime)
            logger.info(
                f"NIFTY regime: {regime['regime']} (score={regime['score']:+.3f})"
                f" → direction_filter={regime['direction_filter']}"
            )
        except Exception as e:
            logger.warning(f"Regime fetch failed: {e} — proceeding with NEUTRAL defaults")

        # 2. Check exits first (always before entries)
        if tracker.open_count() > 0:
            check_exits(tracker, perf, closed_today, strategies)

        # 3. Scan for new entries
        if tracker.can_open_new_trade():
            scan_for_entries(
                candidates, tracker, closed_today, strategies,
                regime=regime,
                max_positions_override=effective_max_positions,
            )

        # 4. Log current state
        logger.info(tracker.summary())

        # 5. Sleep until next candle
        logger.info(f"Sleeping {LOOP_SLEEP_SECONDS}s until next candle...")
        time.sleep(LOOP_SLEEP_SECONDS)

    # End of day — ALPHA_COMBO: update IC weights from today's trade data
    if "ALPHA_COMBO" in ACTIVE_STRATEGY.upper():
        from strategy_alpha_combo import end_of_day_weight_update
        ics = end_of_day_weight_update()
        if ics:
            logger.info(f"ALPHA IC update complete: {ics}")

    perf.daily_summary()
    perf.save_to_csv()
    logger.info("Bot exited cleanly.")


if __name__ == "__main__":
    run()
