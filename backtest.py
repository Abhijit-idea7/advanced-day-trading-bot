#!/usr/bin/env python3
"""
backtest.py
-----------
Offline backtest of the multi-strategy intraday bot over the last N trading days.

Usage:
    python backtest.py                        # ORB, 30 days
    python backtest.py --strategy VWAP_EMA
    python backtest.py --strategy COMBINED --days 45
    python backtest.py --strategy ORB --days 15

How it works:
  1. Fetches 59 days of 2-min candles from yfinance for all STOCK_UNIVERSE stocks
     (maximum available; gives EMA/RSI full warmup history)
  2. Computes all indicators once on the full dataset per symbol
  3. Simulates the live-bot loop day by day, candle by candle:
       - Ranks stocks daily by ATR% (same as live bot)
       - Entries: checks each active strategy's generate_signal()
       - Exits:   checks check_exit_signal() for the strategy that opened the trade
       - Respects MAX_POSITIONS simultaneous open positions
       - Hard square-off at 15:15 IST
  4. Prints per-day P&L table + overall summary
  5. Saves all trades to backtest_results.csv (artifact in GitHub Actions)

Note on yfinance data:
  Yahoo Finance provides up to ~59 days of 2-min candles.
  We always fetch the full 59d window regardless of --days so EMA/RSI are warmed up.
"""

import argparse
import csv
import logging
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from config import (
    ACTIVE_STRATEGY,
    DAILY_LOSS_CIRCUIT_BREAKER,
    MAX_POSITIONS,
    ONE_TRADE_PER_STOCK_PER_DAY,
    ORB_MAX_POSITIONS,
    ORB_STOCK_UNIVERSE,
    ORB_TOP_N_STOCKS,
    POSITION_SIZE_INR,
    REGIME_BEAR_THRESHOLD,
    REGIME_BULL_THRESHOLD,
    STOCK_UNIVERSE,
    TOP_N_STOCKS,
)
from indicators import add_indicators
from strategy_factory import get_strategies, get_strategy_name

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")

IST          = pytz.timezone("Asia/Kolkata")
TRADE_START  = time(9, 20)
SQUARE_OFF   = time(15, 15)
BROKERAGE    = 40         # Rs20 × 2 legs per trade (Zerodha intraday)
OUTPUT_CSV   = Path("backtest_results.csv")
NIFTY_TICKER = "^NSEI"   # NIFTY50 index — no .NS suffix

CSV_FIELDS = [
    "date", "symbol", "strategy", "direction",
    "entry_time", "exit_time",
    "entry_price", "exit_price",
    "quantity", "pnl_inr", "exit_reason",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BtPosition:
    symbol:        str
    direction:     str
    entry_price:   float
    sl:            float
    target:        float
    quantity:      int
    entry_time:    str
    strategy_name: str


@dataclass
class BtTrade:
    date:        str
    symbol:      str
    strategy:    str
    direction:   str
    entry_time:  str
    exit_time:   str
    entry_price: float
    exit_price:  float
    quantity:    int
    pnl_inr:     float
    exit_reason: str


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------
def _ns(symbol: str) -> str:
    return f"{symbol}.NS"


def fetch_with_indicators(symbol: str) -> pd.DataFrame | None:
    """
    Fetch ~59 days of 2-min candles, compute all indicators,
    return a timezone-aware (IST) DataFrame or None on failure.
    """
    try:
        df = yf.Ticker(_ns(symbol)).history(interval="2m", period="59d")
        if df is None or df.empty:
            logger.warning(f"{symbol}: no data from yfinance")
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)

        if len(df) < 30:
            logger.warning(f"{symbol}: only {len(df)} candles — too few for warmup")
            return None

        df = add_indicators(df)
        logger.info(f"{symbol}: {len(df)} candles with indicators")
        return df

    except Exception as e:
        logger.error(f"{symbol}: fetch error — {e}")
        return None


def rank_by_atr(symbol_dfs: dict, date_: datetime.date, top_n: int = TOP_N_STOCKS) -> list[str]:
    """Rank symbols by ATR% using daily data prior to backtest date."""
    scores: dict[str, float] = {}
    for symbol, df in symbol_dfs.items():
        try:
            hist = df[df.index.date < date_]
            if hist.empty:
                continue
            daily       = hist["Close"].resample("1D").last().dropna()
            daily_high  = hist["High"].resample("1D").max().dropna()
            daily_low   = hist["Low"].resample("1D").min().dropna()
            daily, daily_high = daily.align(daily_high, join="inner")
            daily, daily_low  = daily.align(daily_low, join="inner")
            if len(daily) < 3:
                continue
            prev_close = daily.shift(1)
            tr = pd.concat([
                daily_high - daily_low,
                (daily_high - prev_close).abs(),
                (daily_low  - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr     = tr.iloc[-10:].mean()
            atr_pct = atr / daily.iloc[-1] if daily.iloc[-1] > 0 else 0
            scores[symbol] = atr_pct
        except Exception:
            pass

    if not scores:
        return list(symbol_dfs.keys())[:top_n]
    return sorted(scores, key=lambda s: scores[s], reverse=True)[:top_n]


def calculate_quantity(price: float, scale: float = 1.0) -> int:
    return int(POSITION_SIZE_INR * scale // price) if price > 0 else 0


# ---------------------------------------------------------------------------
# NIFTY Regime helpers (used when ORB_REGIME_FILTER=true)
# ---------------------------------------------------------------------------
def fetch_nifty_with_indicators() -> "pd.DataFrame | None":
    """
    Fetch ~59 days of 2-min NIFTY50 candles with indicators computed.
    Uses ^NSEI (Yahoo Finance index ticker — no .NS suffix).
    """
    try:
        df = yf.Ticker(NIFTY_TICKER).history(interval="2m", period="59d")
        if df is None or df.empty:
            logger.warning("NIFTY: empty response from yfinance")
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
        if len(df) < 30:
            logger.warning(f"NIFTY: only {len(df)} candles — too few")
            return None
        df = add_indicators(df)
        logger.info(f"NIFTY: {len(df)} candles fetched with indicators")
        return df
    except Exception as e:
        logger.error(f"NIFTY fetch error: {e}")
        return None


def compute_day_regime(nifty_df: pd.DataFrame, date_) -> dict:
    """
    Classify the NIFTY regime for a given backtest date using data up to 10:00 IST.

    Using 10:00 cutoff simulates what the live bot would see before most ORB entries
    (ORB_ENTRY_CUTOFF_TIME = 11:00). This avoids look-ahead bias while still giving
    a clear regime signal from the first 45 minutes of trading.

    Scoring mirrors market_regime.py exactly:
      VWAP position (0.40) + EMA 9 vs 50 trend (0.40) + Day change (0.20)
      score > REGIME_BULL_THRESHOLD  → BULL  → LONG_ONLY
      score < REGIME_BEAR_THRESHOLD  → BEAR  → SHORT_ONLY
      else                           → NEUTRAL → BOTH
    """
    _neutral = {"regime": "NEUTRAL", "score": 0.0, "direction_filter": "BOTH"}

    try:
        day_df = nifty_df[nifty_df.index.date == date_]
        if day_df.empty:
            return _neutral

        # Use data up to 10:00 IST — enough time for regime to establish but
        # well before the 11:00 ORB entry cutoff.
        cutoff_time = time(10, 0)
        day_df = day_df[day_df.index.time <= cutoff_time]
        if len(day_df) < 3:
            return _neutral

        row   = day_df.iloc[-1]   # last candle on or before 10:00 IST
        close = float(row["Close"])

        components: list = []
        weights:    list = []

        # Component 1: VWAP position (weight 0.40)
        vwap = row.get("vwap")
        if not pd.isna(vwap) and float(vwap) > 0:
            vwap_dev = (close - float(vwap)) / float(vwap)
            components.append(float(np.tanh(vwap_dev * 100)))
            weights.append(0.40)

        # Component 2: EMA 9 vs EMA 50 (weight 0.40)
        ema9  = row.get("ema_fast")
        ema50 = row.get("ema_macro")
        if not any(pd.isna(x) for x in (ema9, ema50)):
            components.append(1.0 if float(ema9) > float(ema50) else -1.0)
            weights.append(0.40)

        # Component 3: Day change from open (weight 0.20)
        day_open = row.get("day_open")
        if not pd.isna(day_open) and float(day_open) > 0:
            day_chg = (close - float(day_open)) / float(day_open)
            components.append(float(np.tanh(day_chg * 50)))
            weights.append(0.20)

        if not components:
            return _neutral

        total_w = sum(weights)
        score   = float(np.clip(
            sum(c * w / total_w for c, w in zip(components, weights)),
            -1.0, 1.0
        ))

        if score > REGIME_BULL_THRESHOLD:
            return {"regime": "BULL",    "score": score, "direction_filter": "LONG_ONLY"}
        elif score < REGIME_BEAR_THRESHOLD:
            return {"regime": "BEAR",    "score": score, "direction_filter": "SHORT_ONLY"}
        else:
            return {"regime": "NEUTRAL", "score": score, "direction_filter": "BOTH"}

    except Exception as e:
        logger.warning(f"compute_day_regime({date_}) error: {e} — using NEUTRAL")
        return _neutral


# ---------------------------------------------------------------------------
# Single-day simulation
# ---------------------------------------------------------------------------
def simulate_day(
    date_:            datetime.date,
    candidates:       list[str],
    symbol_dfs:       dict,
    strategies:       list,
    max_positions:    int = MAX_POSITIONS,
    direction_filter: str = "BOTH",
) -> list[BtTrade]:
    """
    Simulate a full trading day candle-by-candle across all candidates.
    Returns a list of BtTrade records for all closed trades that day.
    """
    trades: list[BtTrade]         = []
    open_positions: dict[str, BtPosition] = {}
    traded_today:   set[str]      = set()
    daily_realized_pnl: float     = 0.0

    # Build per-symbol DataFrames for this day
    day_data: dict[str, pd.DataFrame] = {}
    for sym in candidates:
        if sym not in symbol_dfs:
            continue
        df     = symbol_dfs[sym]
        day_df = df[df.index.date == date_].copy()
        if len(day_df) >= 3:
            day_data[sym] = day_df

    if not day_data:
        return trades

    # Unified timeline across all symbols
    all_times = sorted({ts for df in day_data.values() for ts in df.index})

    def _close_position(symbol: str, exit_px: float, reason: str, ts_str: str):
        nonlocal daily_realized_pnl
        pos = open_positions.pop(symbol)
        traded_today.add(symbol)
        pnl = (
            (exit_px - pos.entry_price) * pos.quantity
            if pos.direction == "BUY"
            else (pos.entry_price - exit_px) * pos.quantity
        )
        daily_realized_pnl += pnl
        trades.append(BtTrade(
            date        = date_.isoformat(),
            symbol      = symbol,
            strategy    = pos.strategy_name,
            direction   = pos.direction,
            entry_time  = pos.entry_time,
            exit_time   = ts_str,
            entry_price = pos.entry_price,
            exit_price  = exit_px,
            quantity    = pos.quantity,
            pnl_inr     = round(pnl, 2),
            exit_reason = reason,
        ))

    for ts in all_times:
        ts_time = ts.time()
        ts_str  = ts.strftime("%H:%M")

        # Hard square-off gate
        if ts_time >= SQUARE_OFF:
            for symbol in list(open_positions.keys()):
                df     = day_data.get(symbol)
                px     = float(df.loc[ts, "Close"]) if df is not None and ts in df.index else open_positions[symbol].entry_price
                _close_position(symbol, px, "SQUARE_OFF", ts_str)
            break

        if ts_time < TRADE_START:
            continue

        # --- Exit checks ---
        # Bug fix #3: use df.index < ts (strictly less than) so the currently
        # forming candle at `ts` is never included — only fully-closed bars.
        for symbol in list(open_positions.keys()):
            df = day_data.get(symbol)
            if df is None:
                continue
            pos = open_positions[symbol]
            df_slice = df[df.index < ts]   # strictly less than → only closed candles
            if len(df_slice) < 2:
                continue

            # Route exit check to the strategy that opened this position
            strategy_module = next(
                (s for s in strategies if get_strategy_name(s) == pos.strategy_name),
                strategies[0],
            )
            reason = strategy_module.check_exit_signal(df_slice, pos.__dict__)
            if reason:
                # Exit price selection — match the detection method used in
                # check_exit_signal so simulated P&L reflects real fills:
                #   TARGET    → filled at the target level (High/Low touched it)
                #   STOP_LOSS → filled at SL level (Low/High touched it)
                #   ORB_FAILED / SQUARE_OFF / other → filled at Close
                sig_candle = df_slice.iloc[-1]   # last closed candle (signal candle)
                if reason == "TARGET":
                    exit_px = float(pos.target)
                elif reason == "STOP_LOSS":
                    exit_px = float(pos.sl)
                else:
                    exit_px = float(sig_candle["Close"])
                _close_position(symbol, exit_px, reason, ts_str)

        # --- Entry checks ---
        # Bug fix #3 (entry): same df_slice boundary — strictly less than ts.
        circuit_tripped = daily_realized_pnl <= DAILY_LOSS_CIRCUIT_BREAKER
        if len(open_positions) < max_positions and not circuit_tripped:
            for symbol in candidates:
                if len(open_positions) >= max_positions:   # fixed: was MAX_POSITIONS
                    break
                if symbol in open_positions:
                    continue
                if ONE_TRADE_PER_STOCK_PER_DAY and symbol in traded_today:
                    continue

                df = day_data.get(symbol)
                if df is None:
                    continue

                df_slice = df[df.index < ts]   # strictly less than → only closed candles
                if len(df_slice) < 3:
                    continue

                # Try each strategy in order — take the first valid signal
                for strategy_module in strategies:
                    # Bug fix #1: pass sim_time=ts so the cutoff gate inside
                    # generate_signal() uses the simulated candle time, not
                    # datetime.now() (which would always be after market hours
                    # and block every single signal during backtesting).
                    signal = strategy_module.generate_signal(
                        df_slice, symbol=symbol, sim_time=ts
                    )
                    if signal["action"] not in ("BUY", "SELL"):
                        continue

                    # Regime direction filter: block counter-regime trades
                    # BULL  → LONG_ONLY  → skip SELL signals
                    # BEAR  → SHORT_ONLY → skip BUY signals
                    # NEUTRAL → BOTH    → no restriction
                    if direction_filter == "LONG_ONLY" and signal["action"] == "SELL":
                        logger.info(f"{symbol} ORB: SELL blocked — BULL regime (LONG_ONLY)")
                        continue
                    if direction_filter == "SHORT_ONLY" and signal["action"] == "BUY":
                        logger.info(f"{symbol} ORB: BUY blocked — BEAR regime (SHORT_ONLY)")
                        continue

                    # Bug fix #2: entry price from the signal candle (iloc[-2]),
                    # matching the live bot which also uses iloc[-2] close.
                    # Previously used iloc[-1] (the forming candle) — wrong.
                    entry_price = float(df_slice.iloc[-1]["Close"])
                    quantity    = calculate_quantity(entry_price, scale=signal.get("quantity_scale", 1.0))
                    if quantity < 1:
                        continue   # Bug fix #4: was `break` — skipped ALL strategies
                                   # for this symbol; changed to `continue` so the
                                   # next strategy is still evaluated.

                    open_positions[symbol] = BtPosition(
                        symbol        = symbol,
                        direction     = signal["action"],
                        entry_price   = entry_price,
                        sl            = signal["sl"],
                        target        = signal["target"],
                        quantity      = quantity,
                        entry_time    = ts_str,
                        strategy_name = get_strategy_name(strategy_module),
                    )
                    break   # one position per symbol per loop tick

    return trades


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_overall_summary(
    all_trades:     list[BtTrade],
    days_tested:    int,
    strategy_label: str,
    universe:       list,
    top_n:          int,
    max_pos:        int,
) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  BACKTEST SUMMARY — Strategy: {strategy_label}")
    print(sep)

    if not all_trades:
        print("  No trades generated in the backtest period.")
        print(sep)
        return

    total     = len(all_trades)
    gross     = sum(t.pnl_inr for t in all_trades)
    brokerage = total * BROKERAGE
    net       = gross - brokerage
    wins      = [t for t in all_trades if t.pnl_inr > 0]
    win_rate  = len(wins) / total * 100

    by_reason: dict[str, int] = {}
    for t in all_trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    by_strategy: dict[str, list] = {}
    for t in all_trades:
        by_strategy.setdefault(t.strategy, []).append(t)

    best  = max(all_trades, key=lambda t: t.pnl_inr)
    worst = min(all_trades, key=lambda t: t.pnl_inr)
    avg_per_day = net / days_tested if days_tested else 0

    print(f"  Backtest period  : {days_tested} trading days")
    print(f"  Universe         : {len(universe)} stocks (top {top_n}/day by ATR%)")
    print(f"  Capital per trade: Rs{POSITION_SIZE_INR:,.0f} (max {max_pos} simultaneous)")
    print(sep)
    print(f"  Total trades     : {total}")
    print(f"  Win rate         : {len(wins)}/{total} = {win_rate:.1f}%")
    print(f"  Exit breakdown   : {by_reason}")
    print(sep)

    # Per-strategy breakdown
    for strat, strat_trades in by_strategy.items():
        s_wins = sum(1 for t in strat_trades if t.pnl_inr > 0)
        s_wr   = s_wins / len(strat_trades) * 100
        s_pnl  = sum(t.pnl_inr for t in strat_trades)
        print(f"  [{strat}] trades={len(strat_trades)} wins={s_wins} ({s_wr:.0f}%) gross=Rs{s_pnl:+,.0f}")

    print(sep)
    print(f"  Gross P&L        : Rs{gross:+,.0f}")
    print(f"  Brokerage (est.) : -Rs{brokerage:,.0f}  (Rs{BROKERAGE}/trade × {total})")
    print(f"  Net P&L (est.)   : Rs{net:+,.0f}")
    print(f"  Avg net/day      : Rs{avg_per_day:+,.0f}")
    print(sep)
    print(f"  Best  : {best.symbol} {best.direction} {best.date} Rs{best.pnl_inr:+,.0f} [{best.exit_reason}]")
    print(f"  Worst : {worst.symbol} {worst.direction} {worst.date} Rs{worst.pnl_inr:+,.0f} [{worst.exit_reason}]")
    print(sep)


def save_to_csv(all_trades: list[BtTrade]) -> None:
    if not all_trades:
        return
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for t in all_trades:
            writer.writerow({
                "date":        t.date,
                "symbol":      t.symbol,
                "strategy":    t.strategy,
                "direction":   t.direction,
                "entry_time":  t.entry_time,
                "exit_time":   t.exit_time,
                "entry_price": t.entry_price,
                "exit_price":  t.exit_price,
                "quantity":    t.quantity,
                "pnl_inr":     t.pnl_inr,
                "exit_reason": t.exit_reason,
            })
    logger.info(f"Saved {len(all_trades)} trades to {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(days: int, strategy_override: str | None = None) -> None:
    import os
    if strategy_override:
        os.environ["ACTIVE_STRATEGY"] = strategy_override

    strategies     = get_strategies()
    strategy_label = " + ".join(get_strategy_name(s) for s in strategies)

    # Resolve universe, top_n and max_positions based on active strategy
    active = os.environ.get("ACTIVE_STRATEGY", ACTIVE_STRATEGY).upper()
    is_orb = (active == "ORB")
    bt_universe      = ORB_STOCK_UNIVERSE if is_orb else STOCK_UNIVERSE
    bt_top_n         = ORB_TOP_N_STOCKS   if is_orb else TOP_N_STOCKS
    bt_max_positions = ORB_MAX_POSITIONS  if is_orb else MAX_POSITIONS

    print(f"\n{'=' * 70}")
    print(f"  INTRADAY BACKTEST — last {days} trading days")
    print(f"  Strategy : {strategy_label}")
    print(f"  Universe : {bt_universe}")
    print(f"{'=' * 70}\n")

    # 1. Fetch data + indicators
    logger.info("Fetching 59-day 2-min data (this may take ~1 minute)...")
    symbol_dfs: dict[str, pd.DataFrame] = {}
    for symbol in bt_universe:
        df = fetch_with_indicators(symbol)
        if df is not None:
            symbol_dfs[symbol] = df

    if not symbol_dfs:
        logger.error("No data fetched. Exiting.")
        return

    # 2. Determine trading days
    all_dates = sorted({d for df in symbol_dfs.values() for d in df.index.date})
    backtest_dates = all_dates[-days:]
    logger.info(f"Testing {len(backtest_dates)} days: {backtest_dates[0]} — {backtest_dates[-1]}")

    # 2b. NIFTY regime filter (ORB only, opt-in via ORB_REGIME_FILTER=true)
    import os as _os
    use_regime = _os.getenv("ORB_REGIME_FILTER", "false").lower() == "true" and is_orb
    day_regimes: dict = {}   # date → {"regime": str, "score": float, "direction_filter": str}

    if use_regime:
        logger.info("ORB_REGIME_FILTER=true — fetching NIFTY50 data for daily regime classification...")
        nifty_df = fetch_nifty_with_indicators()
        if nifty_df is not None:
            for d in backtest_dates:
                day_regimes[d] = compute_day_regime(nifty_df, d)
            regime_counts: dict[str, int] = {}
            for r in day_regimes.values():
                lbl = r.get("regime", "NEUTRAL")
                regime_counts[lbl] = regime_counts.get(lbl, 0) + 1
            logger.info(f"Regime classification over {len(backtest_dates)} days: {regime_counts}")
            print(f"\n  Regime filter ON — NIFTY classification at 10:00 IST: {regime_counts}")
            print(f"  BULL → LONG_ONLY  |  BEAR → SHORT_ONLY  |  NEUTRAL → BOTH")
        else:
            logger.warning("NIFTY fetch failed — regime filter disabled, all days use BOTH")
            use_regime = False

    # 3. Simulate
    all_trades: list[BtTrade] = []
    if use_regime:
        print(f"\n  {'Date':12s}  {'Rgm':>4}  {'Trades':>6}  {'Wins':>4}  {'Losses':>6}  {'Win%':>5}  {'Gross':>12}  {'Net':>12}")
        print(f"  {'-' * 84}")
    else:
        print(f"\n  {'Date':12s}  {'Trades':>6}  {'Wins':>4}  {'Losses':>6}  {'Win%':>5}  {'Gross':>12}  {'Net':>12}")
        print(f"  {'-' * 78}")

    for date_ in backtest_dates:
        regime         = day_regimes.get(date_, {"regime": "BOTH", "direction_filter": "BOTH"})
        dir_filter     = regime.get("direction_filter", "BOTH")
        regime_label   = regime.get("regime", "BOTH")[:4]   # BULL / BEAR / NEUT / BOTH

        candidates  = rank_by_atr(symbol_dfs, date_, top_n=bt_top_n)
        day_trades  = simulate_day(
            date_, candidates, symbol_dfs, strategies,
            max_positions=bt_max_positions,
            direction_filter=dir_filter,
        )
        all_trades.extend(day_trades)

        if day_trades:
            g    = sum(t.pnl_inr for t in day_trades)
            n    = g - BROKERAGE * len(day_trades)
            wins = sum(1 for t in day_trades if t.pnl_inr > 0)
            wr   = wins / len(day_trades) * 100
            if use_regime:
                print(
                    f"  {str(date_):12s}  {regime_label:>4s}  {len(day_trades):>6d}  {wins:>4d}  "
                    f"{len(day_trades)-wins:>6d}  {wr:>4.0f}%  "
                    f"Rs{g:>+9,.0f}  Rs{n:>+9,.0f}"
                )
            else:
                print(
                    f"  {str(date_):12s}  {len(day_trades):>6d}  {wins:>4d}  "
                    f"{len(day_trades)-wins:>6d}  {wr:>4.0f}%  "
                    f"Rs{g:>+9,.0f}  Rs{n:>+9,.0f}"
                )
        else:
            if use_regime:
                print(f"  {str(date_):12s}  {regime_label:>4s}  {'—':>6}  {'—':>4}  {'—':>6}  {'—':>5}  {'Rs0':>12}  {'Rs0':>12}")
            else:
                print(f"  {str(date_):12s}  {'—':>6}  {'—':>4}  {'—':>6}  {'—':>5}  {'Rs0':>12}  {'Rs0':>12}")

    # 4. Summary + CSV
    print_overall_summary(
        all_trades, len(backtest_dates), strategy_label,
        universe=bt_universe, top_n=bt_top_n, max_pos=bt_max_positions,
    )
    save_to_csv(all_trades)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest multi-strategy intraday bot")
    parser.add_argument("--days", type=int, default=30, choices=range(1, 60), metavar="N",
                        help="Trading days to backtest (1–59, default: 30)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Override ACTIVE_STRATEGY: ORB | VWAP_EMA | COMBINED")
    args = parser.parse_args()
    run(args.days, args.strategy)
