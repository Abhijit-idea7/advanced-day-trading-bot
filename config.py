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
# Opening range window: first ORB_MINUTES of the NSE session (9:15–9:30 IST)
# 15 min is the classic ORB window; captures institutional order flow settlement.

ORB_VOLUME_MULTIPLIER = 1.5
# Breakout candle must have volume >= this × 10-candle avg.
# Filters false breakouts on low-volume fades.

ORB_MIN_RANGE_PCT = 0.003
# Minimum ORB size as % of price (0.3%).
# Skips "dead" opens with no meaningful range to break.

ORB_MAX_RANGE_PCT = 0.04
# Maximum ORB size as % of price (4%).
# Avoids trading after extreme gap-and-run opens where R:R is poor.

ORB_CHASE_LIMIT_PCT = 0.01
# Don't enter if price has already moved >1% beyond the ORB level.
# Prevents chasing extended breakouts with poor risk.

ORB_TARGET_MULTIPLIER = 1.5
# Target = entry ± (ORB range × this multiplier).
# 1.5× range gives a realistic intraday measured move target.

ORB_ENTRY_CUTOFF_TIME = "11:00"
# No new ORB entries after 11:00 IST.
# ORB setups lose statistical edge once morning momentum fades.

# ---------------------------------------------------------------------------
# VWAP + EMA Pullback Parameters
# ---------------------------------------------------------------------------
EMA_FAST = 9
# Fast EMA — acts as dynamic intraday support/resistance.
# The 9 EMA on 2-min charts is the standard for NSE day traders.

EMA_SLOW = 20
# Slow EMA — confirms trend direction.
# 9 EMA > 20 EMA = uptrend, 9 EMA < 20 EMA = downtrend.

VWAP_PROXIMITY_PCT = 0.004
# Price must be within 0.4% of VWAP to qualify as a "pullback".
# VWAP is the institutional fair value benchmark; pullbacks here attract buyers.

VWAP_RSI_MIN = 35
# RSI floor for longs / ceiling-mirror for shorts.
# Ensures we're not buying already-oversold or selling already-overbought.

VWAP_RSI_MAX = 65
# RSI ceiling for longs.
# Above 65 on 2-min chart = extended; pullback likely to be weak.

VWAP_VOLUME_MULTIPLIER = 1.3
# Bounce candle must confirm with above-average volume.

VWAP_ENTRY_CUTOFF_TIME = "12:30"
# No new VWAP entries after 12:30 IST.
# Afternoon VWAP bounces near session midpoint become unreliable
# as institutional interest drops and noise increases.

# ---------------------------------------------------------------------------
# Shared Indicator Parameters
# ---------------------------------------------------------------------------
RSI_PERIOD = 14
VOLUME_LOOKBACK = 10       # Candles for rolling volume average

# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------
POSITION_SIZE_INR = 100_000   # Capital per trade in INR
MAX_POSITIONS = 10            # Max simultaneous open positions
TOP_N_STOCKS = 10             # Candidates selected daily by ATR%

# ---------------------------------------------------------------------------
# Risk / Reward
# ---------------------------------------------------------------------------
RISK_REWARD_RATIO = 2.0       # Used by VWAP_EMA strategy target calculation

# ---------------------------------------------------------------------------
# Trade Management
# ---------------------------------------------------------------------------
ONE_TRADE_PER_STOCK_PER_DAY = True
TRADE_START_TIME = "09:20"    # No entries before this IST time
SQUARE_OFF_TIME  = "15:15"    # Force-close all positions at this IST time
CANDLE_INTERVAL  = "2m"       # yfinance interval string
LOOP_SLEEP_SECONDS = 120      # Sleep between strategy iterations (2 min candle)

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
