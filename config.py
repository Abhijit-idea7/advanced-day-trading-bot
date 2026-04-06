"""
config.py
---------
All configuration for the multi-strategy intraday trading bot.

Strategies available:
  ORB       — Opening Range Breakout (first 15 min of NSE session)
  VWAP_EMA  — VWAP pullback confirmed by 9/20 EMA trend
  COMBINED  — Run both strategies simultaneously

Set ACTIVE_STRATEGY here or override via environment variable.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Stock Universe — High Beta NSE Stocks
# ---------------------------------------------------------------------------
STOCK_UNIVERSE = [
    # High-beta banking
    "AXISBANK", "ICICIBANK", "INDUSINDBK", "FEDERALBNK",
    # Financials
    "BAJFINANCE", "CHOLAFIN",
    # Metals
    "TATASTEEL", "HINDALCO", "JSWSTEEL",
    # Auto
    "TATAMOTORS", "M&M",
    # Adani / infra
    "ADANIENT", "ADANIPORTS",
    # Consumer / new-age
    "ETERNAL",       # Zomato
    # Power
    "TATAPOWER", "ADANIGREEN",
    # PSU banks
    "BANKBARODA", "PNB",
    # High-beta others
    "SUZLON", "HINDCOPPER",
]

# ---------------------------------------------------------------------------
# Active Strategy  (override with env var: ACTIVE_STRATEGY=VWAP_EMA)
# ---------------------------------------------------------------------------
ACTIVE_STRATEGY = os.getenv("ACTIVE_STRATEGY", "ORB")
# Options: "ORB" | "VWAP_EMA" | "COMBINED"

# ---------------------------------------------------------------------------
# ORB (Opening Range Breakout) Parameters
# ---------------------------------------------------------------------------
ORB_MINUTES = 15
# Opening range window: first ORB_MINUTES of the NSE session (9:15–9:30 IST).

ORB_VOLUME_MULTIPLIER = 1.3
# Breakout candle must have volume >= this × 10-candle avg.
# Lowered from 1.5 → 1.3: captures more valid breakouts without noise.

ORB_MIN_RANGE_PCT = 0.003
# Minimum ORB size as % of price (0.3%). Skips dead flat opens.

ORB_MAX_RANGE_PCT = 0.04
# Maximum ORB size as % of price (4%). Avoids extreme gap events.

ORB_CHASE_LIMIT_PCT = 0.008
# Don't enter if price has already moved >0.8% beyond the ORB level.
# Slightly tighter than before — entering closer to breakout = better R:R.

ORB_TARGET_MULTIPLIER = 2.5
# Target = entry ± (ORB range × this multiplier).
# Raised from 1.5 → 2.5: the key R:R fix. With 2.5× target and SL at the
# other end of the range, even a 40% win rate is profitable.

ORB_ENTRY_CUTOFF_TIME = "11:00"
# No new ORB entries after 11:00 IST. Morning momentum window only.

ORB_MIN_GAP_PCT = 0.002
# Gap-direction filter (the book's "Gap-and-Go" rule).
# If today's open is >0.2% above prev close → bias is UP → only take LONG ORB.
# If today's open is >0.2% below prev close → bias is DOWN → only take SHORT ORB.
# If open is flat (within ±0.2%) → accept both directions (VWAP decides).
# This is the single biggest win-rate improvement: ~50% → ~60%+ win rate.

ORB_FAILED_BUFFER_PCT = 0.008
# How far back inside the ORB range before triggering ORB_FAILED exit.
# 0.8% buffer: gives trades room to retest the breakout level without exiting.
# Widened from 0.3% → 0.8% because backtest showed 90/140 exits (64%) were
# ORB_FAILED — many of those were valid retests that reversed and continued.
# A wider buffer keeps the trade alive through shallow retests.

ORB_BREAKEVEN_TRIGGER_R = 0.6
# Move SL to breakeven (entry price) once the trade has gained 60% of initial risk.
# Example: entry=100, SL=98 (risk=2). Once High >= 101.20 (entry + 0.6×2),
# the effective SL is raised to 100 (entry). Locks in a no-loss trade early
# without requiring the full 2.5× target to be reached first.

# ---------------------------------------------------------------------------
# VWAP + EMA Pullback Parameters
# ---------------------------------------------------------------------------
EMA_FAST = 9
# Fast EMA — intraday dynamic support/resistance.

EMA_SLOW = 20
# Slow EMA — confirms short-term trend direction.

EMA_MACRO = 50
# Macro session trend filter. Only go LONG if 9 EMA > 50 EMA (stock trending
# up for the session). Only go SHORT if 9 EMA < 50 EMA.
# Prevents VWAP trades against the session's established direction.

VWAP_PROXIMITY_PCT = 0.007
# Price must be within 0.7% of VWAP to qualify as a pullback.
# Widened from 0.4% → 0.7%: many valid VWAP touches were being filtered
# because price got within 0.5% but not 0.4%. 0.7% catches them cleanly.

VWAP_RSI_MIN = 30
# RSI floor for longs. Widened from 35 → 30 to allow more entries.
# Truly oversold condition (RSI < 30) is still avoided.

VWAP_RSI_MAX = 70
# RSI ceiling for longs. Widened from 65 → 70 for the same reason.

VWAP_VOLUME_MULTIPLIER = 1.1
# Bounce candle volume >= this × avg. Lowered from 1.3 → 1.1.
# Volume confirms direction but 1.3x was filtering too many good setups.

VWAP_ENTRY_CUTOFF_TIME = "12:30"
# No new VWAP entries after 12:30 IST. Morning-to-midday window only.

VWAP_MIN_RISK_PCT = 0.002
# Minimum risk as % of entry price (0.2%) per trade.
# Filters tiny-risk setups (e.g. price hugging VWAP very tightly) where
# the Rs40 brokerage becomes a significant % of the potential gain.
# Example: entry at Rs500, risk must be at least Rs1.00 (0.2%).

# ---------------------------------------------------------------------------
# Shared Indicator Parameters
# ---------------------------------------------------------------------------
RSI_PERIOD     = 14
VOLUME_LOOKBACK = 10       # Candles for rolling volume average

# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------
POSITION_SIZE_INR = 100_000   # Capital per trade in INR
MAX_POSITIONS     = 10        # Max simultaneous open positions
TOP_N_STOCKS      = 10        # Candidates selected daily by ATR%

# ---------------------------------------------------------------------------
# Risk / Reward
# ---------------------------------------------------------------------------
RISK_REWARD_RATIO = 2.0       # Used by VWAP_EMA strategy target calculation

# ---------------------------------------------------------------------------
# Trade Management
# ---------------------------------------------------------------------------
ONE_TRADE_PER_STOCK_PER_DAY = True
TRADE_START_TIME  = "09:20"    # No entries before this IST time
SQUARE_OFF_TIME   = "15:15"    # Force-close all positions at this IST time
CANDLE_INTERVAL   = "2m"       # yfinance interval string
LOOP_SLEEP_SECONDS = 120       # Sleep between strategy iterations (2 min candle)

# ---------------------------------------------------------------------------
# Stocksdeveloper Webhook (routes to Zerodha via stocksdeveloper.in)
# ---------------------------------------------------------------------------
STOCKSDEVELOPER_URL     = "https://tv.stocksdeveloper.in/"
STOCKSDEVELOPER_API_KEY = os.getenv("STOCKSDEVELOPER_API_KEY")
STOCKSDEVELOPER_ACCOUNT = os.getenv("STOCKSDEVELOPER_ACCOUNT", "AbhiZerodha")

if not STOCKSDEVELOPER_API_KEY:
    raise EnvironmentError(
        "STOCKSDEVELOPER_API_KEY is not set. "
        "Add it to your .env file or GitHub Actions secrets."
    )

# ---------------------------------------------------------------------------
# Order Defaults
# ---------------------------------------------------------------------------
EXCHANGE     = "NSE"
PRODUCT_TYPE = "INTRADAY"
ORDER_TYPE   = "MARKET"
VARIETY      = "REGULAR"
