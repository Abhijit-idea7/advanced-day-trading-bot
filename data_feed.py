"""
data_feed.py
------------
Fetches OHLCV candle data from Yahoo Finance for NSE-listed stocks.
All symbols are automatically suffixed with ".NS" for NSE.

Key difference from the original strategy's data_feed:
  fetch_candles()           — today's session only (for live signal generation)
  fetch_candles_for_warmup()— 5-day multi-day fetch WITHOUT today-only filter,
                              so indicators (EMA, RSI, ORB) have proper warmup history.
"""

import logging
import time

import pandas as pd
import yfinance as yf

from config import CANDLE_INTERVAL, ORB_STOCK_UNIVERSE, ORB_TOP_N_STOCKS, STOCK_UNIVERSE, TOP_N_STOCKS

logger = logging.getLogger(__name__)


def _ns(symbol: str) -> str:
    """Return Yahoo Finance ticker string for NSE."""
    return f"{symbol}.NS"


def fetch_candles(symbol: str, interval: str = CANDLE_INTERVAL, period: str = "1d") -> pd.DataFrame | None:
    """
    Fetch intraday OHLCV candles for a single trading day.
    Returns today's session only (prevents VWAP anchor bleed from prior days).
    """
    for attempt in range(3):
        try:
            df = yf.Ticker(_ns(symbol)).history(interval=interval, period=period)
            if df.empty:
                logger.warning(f"{symbol}: empty data returned (attempt {attempt + 1})")
                time.sleep(2)
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            # Keep only today's session
            if not df.empty:
                last_date = df.index[-1].normalize()
                df = df[df.index >= last_date]
            return df
        except Exception as e:
            logger.warning(f"{symbol}: fetch error on attempt {attempt + 1} — {e}")
            time.sleep(2)
    logger.error(f"{symbol}: all fetch attempts failed, skipping.")
    return None


def fetch_candles_for_warmup(symbol: str, period: str = "5d") -> pd.DataFrame | None:
    """
    Fetch multiple days of 2-min candles WITHOUT filtering to today.
    Used in fetch_and_prepare() so EMA, RSI, and ORB indicators have
    sufficient history for accurate warmup before today's signal candles.

    Why 5d? The 9/20 EMA and RSI(14) need ~20–30 candles to converge;
    5 days gives ~300+ 2-min candles — well beyond any warmup period.
    VWAP and ORB still reset each day via their per-day grouping logic.
    """
    for attempt in range(3):
        try:
            df = yf.Ticker(_ns(symbol)).history(interval=CANDLE_INTERVAL, period=period)
            if df.empty:
                logger.warning(f"{symbol}: empty warmup data (attempt {attempt + 1})")
                time.sleep(2)
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            # Ensure timezone-aware index in IST
            if df.index.tz is None:
                import pytz
                df.index = df.index.tz_localize("UTC").tz_convert(pytz.timezone("Asia/Kolkata"))
            else:
                import pytz
                df.index = df.index.tz_convert(pytz.timezone("Asia/Kolkata"))
            return df
        except Exception as e:
            logger.warning(f"{symbol}: warmup fetch error attempt {attempt + 1} — {e}")
            time.sleep(2)
    logger.error(f"{symbol}: all warmup fetch attempts failed.")
    return None


def fetch_daily_candles(symbol: str, period: str = "10d") -> pd.DataFrame | None:
    """
    Fetch daily OHLCV candles. Used by get_top_candidates() for ATR% ranking.
    """
    try:
        df = yf.Ticker(_ns(symbol)).history(interval="1d", period=period)
        if df.empty:
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        logger.warning(f"{symbol}: daily fetch error — {e}")
        return None


def get_top_candidates(
    universe: list[str] | None = None,
    top_n:    int | None       = None,
) -> list[str]:
    """
    Rank universe by ATR% (ATR / Close price) using recent daily candles.
    Returns the top top_n symbols — most volatile stocks for today.

    universe : stock list to rank (defaults to STOCK_UNIVERSE)
    top_n    : how many to return (defaults to TOP_N_STOCKS)

    Pass ORB_STOCK_UNIVERSE / ORB_TOP_N_STOCKS from main.py when running
    the ORB strategy so it scans a larger, gap-optimised set.
    """
    universe = universe if universe is not None else STOCK_UNIVERSE
    top_n    = top_n    if top_n    is not None else TOP_N_STOCKS

    scores: dict[str, float] = {}

    for symbol in universe:
        df = fetch_daily_candles(symbol, period="10d")
        if df is None or len(df) < 3:
            continue
        try:
            prev_close = df["Close"].shift(1)
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"]  - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr     = tr.mean()
            atr_pct = atr / df["Close"].iloc[-1]
            scores[symbol] = atr_pct
        except Exception as e:
            logger.warning(f"{symbol}: ATR calculation error — {e}")

    if not scores:
        logger.warning("Could not score any stocks; falling back to full universe.")
        return universe[:top_n]

    ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
    top    = ranked[:top_n]
    logger.info(f"Today's top {top_n} candidates by ATR%: {top}")
    logger.info({s: f"{scores[s]*100:.2f}%" for s in top})
    return top
