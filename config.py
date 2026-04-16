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
# Stock Universe — ALPHA_COMBO (IC-weighted ensemble, 9:20–13:00 IST)
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
# ORB Stock Universe — larger set optimised for gap-and-breakout plays
# ---------------------------------------------------------------------------
# ORB works best on stocks that:
#   1. Have large overnight gaps (reacts to US markets, commodity prices)
#   2. Have sufficient volume for clean breakouts and fills
#   3. Exhibit clear opening range structure (not sideways from open)
# The ATR% ranker picks the top ORB_TOP_N_STOCKS each day from this list.
ORB_STOCK_UNIVERSE = [
    # Large-cap banking — most liquid, clean ORB patterns
    "HDFCBANK", "SBIN", "AXISBANK", "ICICIBANK", "BANKBARODA",
    "PNB", "INDUSINDBK", "FEDERALBNK",
    # Financials
    "BAJFINANCE", "CHOLAFIN",
    # IT — US market cues drive strong overnight gaps
    "INFY", "WIPRO", "HCLTECH",
    # Auto — monthly sales data, commodity input costs create gaps
    "TATAMOTORS", "M&M", "MARUTI", "BAJAJ-AUTO",
    # Metals — commodity-driven overnight moves (LME copper/steel)
    "TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "HINDCOPPER",
    # Oil & Gas — crude oil prices create sharp gaps
    "RELIANCE", "ONGC", "BPCL",
    # Power / Infra — policy-sensitive, gap on news
    "TATAPOWER", "ADANIGREEN", "ADANIPORTS", "ADANIENT", "NTPC",
    # High-beta / momentum
    "SUZLON", "ETERNAL",
]

# ---------------------------------------------------------------------------
# Active Strategy  (override with env var: ACTIVE_STRATEGY=VWAP_EMA)
# ---------------------------------------------------------------------------
ACTIVE_STRATEGY = os.getenv("ACTIVE_STRATEGY", "ORB")
# Options: "ORB" | "VWAP_EMA" | "COMBINED"

# ---------------------------------------------------------------------------
# ORB (Opening Range Breakout) Parameters
# ---------------------------------------------------------------------------
# All key ORB params can be overridden via environment variables so that
# separate backtest workflows can test different configs without code changes.
# ---------------------------------------------------------------------------
ORB_MINUTES           = int(os.getenv("ORB_MINUTES",            "15"))
# Opening range window: first ORB_MINUTES of the NSE session (9:15–9:30 IST).
# Default 15 → range covers 9:15–9:29. Set to 5 for a tighter 9:15–9:19 range.

ORB_VOLUME_MULTIPLIER = float(os.getenv("ORB_VOLUME_MULTIPLIER", "1.3"))
# Breakout candle must have volume >= this × 10-candle avg.

ORB_MIN_RANGE_PCT     = float(os.getenv("ORB_MIN_RANGE_PCT",     "0.003"))
# Minimum ORB size as % of price (0.3%). Skips dead flat opens.

ORB_MAX_RANGE_PCT     = float(os.getenv("ORB_MAX_RANGE_PCT",     "0.04"))
# Maximum ORB size as % of price (4%). Avoids extreme gap events.

ORB_CHASE_LIMIT_PCT   = float(os.getenv("ORB_CHASE_LIMIT_PCT",   "0.008"))
# Don't enter if price has already moved >0.8% beyond the ORB level.

ORB_TARGET_MULTIPLIER = float(os.getenv("ORB_TARGET_MULTIPLIER", "2.5"))
# Target = entry ± (ORB range × this multiplier).
# Default 2.5×. Backtest-1 tests 1.5× via env var.

ORB_ENTRY_CUTOFF_TIME = os.getenv("ORB_ENTRY_CUTOFF_TIME",       "11:00")
# No new ORB entries after this IST time. Default 11:00. Backtest-2 tests 12:00.

ORB_MIN_GAP_PCT       = float(os.getenv("ORB_MIN_GAP_PCT",       "0.002"))
# Gap-direction filter threshold (0.2%). See Gap-and-Go rule in strategy_orb.py.

ORB_POSITION_SCALE    = float(os.getenv("ORB_POSITION_SCALE",    "1.5"))
# ORB signals use 1.5× normal position size (Rs150k → Rs225k per trade).

ORB_FAILED_BUFFER_PCT = float(os.getenv("ORB_FAILED_BUFFER_PCT", "0.008"))
# How far back inside the ORB range triggers ORB_FAILED exit.
# Default 0.8%. Backtest-2 tests 0.5% for faster exit on failed breakouts.

ORB_BREAKEVEN_TRIGGER_R = float(os.getenv("ORB_BREAKEVEN_TRIGGER_R", "0.6"))
# Move SL to breakeven once trade gains this × initial risk. Default 60%.

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

VWAP_PROXIMITY_PCT = 0.007
# Price must be within 0.7% of VWAP to qualify as a pullback.

VWAP_RSI_MIN = 30
# RSI floor. Avoids entering when momentum is truly oversold (RSI < 30).

VWAP_RSI_MAX = 70
# RSI ceiling. Avoids entering when momentum is truly overbought (RSI > 70).

VWAP_VOLUME_MULTIPLIER = 1.1
# Bounce candle volume >= this × rolling avg. Confirms the bounce.

VWAP_ENTRY_CUTOFF_TIME = "12:30"
# No new VWAP entries after 12:30 IST. Morning-to-midday window only.

VWAP_MIN_RISK_PCT = 0.002
# Minimum risk as % of entry price (0.2%) per trade.
# Filters tiny-risk setups where Rs40 brokerage is disproportionate.

# ---------------------------------------------------------------------------
# Shared Indicator Parameters
# ---------------------------------------------------------------------------
RSI_PERIOD      = 14
VOLUME_LOOKBACK = 10   # Candles for rolling volume average

# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------
POSITION_SIZE_INR = 150_000   # Capital per trade in INR
# Raised from Rs100k → Rs150k alongside MAX_POSITIONS drop from 10 → 6.
# Total capital deployment stays similar (6 × 150k ≈ 10 × 100k = Rs1M).
# Since brokerage is flat Rs40/trade regardless of size, each trade now
# earns 1.5× more gross while paying the same Rs40 — improving the ratio.

MAX_POSITIONS     = 6         # Max simultaneous open positions (ALPHA_COMBO)
ORB_MAX_POSITIONS = 5         # ORB runs independently — up to 5 concurrent breakout trades
# Reduced from 10 → 6.  Combined with the higher ALPHA_ENTRY_THRESHOLD,
# this caps daily trades at ~4–6 rather than always filling 10 slots.
# Backtest: 300 trades × Rs40 = Rs12,000 brokerage consumed 34% of gross.
# At ~150 trades: Rs6,000 brokerage on the same-quality gross P&L.

TOP_N_STOCKS      = 10        # Candidates selected daily by ATR% (ALPHA_COMBO)
ORB_TOP_N_STOCKS  = 15        # ORB scans more stocks — breakout setup is selective
# Reverted to 10 after comparing backtest results.
# 6-stock version saved Rs4,560 brokerage but lost Rs11,512 gross P&L —
# a net Rs6,952 worse outcome. Fewer candidates meant missing good trades,
# not just marginal ones. Win rate was identical at 42.4% vs 42.6%.
# Version A (10 stocks): Rs43,279 net | Version B (6 stocks): Rs36,327 net.

# ---------------------------------------------------------------------------
# Risk / Reward
# ---------------------------------------------------------------------------
RISK_REWARD_RATIO = 2.0   # Used by VWAP_EMA strategy target calculation

# ---------------------------------------------------------------------------
# Trade Management
# ---------------------------------------------------------------------------
ONE_TRADE_PER_STOCK_PER_DAY = True

DAILY_LOSS_CIRCUIT_BREAKER = -999_999
# Disabled: backtest showed the breaker backfired — with MAX_POSITIONS=6,
# trades open in batches so the threshold isn't crossed until mid-session,
# by which point it blocked recovery winners (Feb 27: missed 2 wins worth
# Rs633 gross by stopping at trade 7). Net effect was -Rs2,616 worse.
# Set to -999,999 so it never triggers. Re-enable only if a smarter
# consecutive-loss variant is implemented.
TRADE_START_TIME   = "09:20"   # No entries before this IST time
SQUARE_OFF_TIME    = "15:10"   # Force-close all positions at this IST time
# Set to 15:10 (not 15:15) to give a 10-minute buffer before Zerodha's
# MIS auto-square-off at ~15:20. This ensures the bot's orders reach
# Zerodha while the market is still liquid and prices are representative.
CANDLE_INTERVAL    = "2m"      # yfinance interval string
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

# ---------------------------------------------------------------------------
# Alpha Combination Strategy Parameters
# ---------------------------------------------------------------------------
# Initial IC-based signal weights.
# Based on known signal quality for NSE intraday trading (normalized to 1.0).
# The ICWeightTracker in strategy_alpha_combo.py updates these automatically
# at end of each session using Pearson IC regression (simplified Step 10/11
# of the 11-step institutional alpha combination procedure).
#
# Weight rationale (matches IC priors in signal_library.py):
#   orb             0.20 → IC≈0.10 (strongest structural signal)
#   gap             0.16 → IC≈0.08 (institutional overnight bias)
#   vwap_deviation  0.14 → IC≈0.07 (session fair-value anchor)
#   ema_trend       0.14 → IC≈0.07 (session trend direction)
#   momentum        0.13 → IC≈0.06 (price rate-of-change)
#   volume_pressure 0.13 → IC≈0.06 (microstructure order flow)
#   rsi             0.10 → IC≈0.05 (momentum quality)
ALPHA_WEIGHTS = {
    "orb":             0.20,
    "gap":             0.16,
    "vwap_deviation":  0.14,
    "ema_trend":       0.14,
    "momentum":        0.13,
    "volume_pressure": 0.13,
    "rsi":             0.10,
}

ALPHA_ENTRY_THRESHOLD   = 0.40
# Minimum |combined alpha score| to trigger a trade.
# Raised from 0.30 → 0.40 to cut the lowest-conviction trades.
# At 0.30, the strategy always filled all 10 slots (300 trades/30 days).
# At 0.40, only the clearest signal alignments fire — expected ~4–6/day.
# The IC framework guarantees alpha > 0.40 trades carry more independent
# signal evidence than alpha 0.30–0.40 trades. Cutting the weaker ones
# should maintain gross P&L quality while reducing brokerage cost.
# NOTE: this is the default; regime filter overrides it dynamically.

ALPHA_EXIT_THRESHOLD    = 0.20
# Alpha score reversal magnitude that triggers ALPHA_EXIT.
# 0.25 was too high (only 2/300 fires in backtest — almost never used).
# 0.15 was too low (fires on normal signal noise, causing premature exits).
# 0.20 is the middle ground: requires a genuine shift in signal consensus
# without being triggered by per-candle oscillation.

ALPHA_BREAKEVEN_TRIGGER_R = 0.6
# Move stop-loss to breakeven (entry price) once the trade gains
# ALPHA_BREAKEVEN_TRIGGER_R × initial_risk in its favour.
# e.g. entry=100, SL=96 (risk=4). Once price reaches 102.40 (entry + 0.6×4),
# effective SL is raised to 100. Converts a potential loss into breakeven.
# Based on backtest: 160/300 exits were STOP_LOSS; breakeven stop should
# convert ~20-30 of those into zero-loss outcomes.

ALPHA_ATR_LOOKBACK      = 14
# Candles for ATR computation (matches RSI_PERIOD for consistency).

ALPHA_ATR_SL_MULT       = 1.5
# Stop-loss = ATR × this multiple.
# 1.5× ATR gives the trade breathing room while keeping risk well-defined.

ALPHA_TARGET_RR         = 2.5
# Target = SL_distance × this R:R multiplier.
# A 40% win rate is profitable at 2.5× R:R, matching the ORB strategy.

ALPHA_IC_WINDOW         = 200
# Rolling observation window for IC computation.
# At 20 stocks × ~90 bars/day, we accumulate 1,800 observations per session,
# so the window fills within a single day and becomes meaningful immediately.

ALPHA_ENTRY_CUTOFF_TIME = "13:00"
# No new ALPHA_COMBO entries after 13:00 IST.
# Wider than ORB (11:00) and VWAP_EMA (12:30) because ALPHA_COMBO combines
# both structural and trend signals that remain informative into midday.

ALPHA_MOMENTUM_LOOKBACK = 10
# Candles for the momentum signal rate-of-change calculation (10 × 2 min = 20 min).

# ---------------------------------------------------------------------------
# NIFTY50 Market Regime Filter Parameters
# ---------------------------------------------------------------------------
# The regime filter detects intraday NIFTY50 direction and adjusts:
#   - Maximum simultaneous positions (reduces exposure on bear days)
#   - Alpha entry threshold (higher quality bar when market is bearish)
#   - Direction filter (LONG_ONLY / SHORT_ONLY / BOTH)
#
# Based on backtest analysis: 9/30 days had ≤20% win rate and lost ~Rs15,700.
# A regime filter targeting those bear days can approximate doubling net P&L.

REGIME_BULL_THRESHOLD = 0.20
# NIFTY score above this → BULL regime.
# Score = weighted avg of (VWAP position, EMA trend, day change) ∈ [-1, +1].

REGIME_BEAR_THRESHOLD = -0.20
# NIFTY score below this → BEAR regime.

REGIME_BEAR_MAX_POSITIONS    = 4
# Max simultaneous open positions in BEAR regime (vs 10 normally).
# Reduces capital at risk when broad market is unfavourable.

REGIME_NEUTRAL_MAX_POSITIONS = 8
# Max positions in NEUTRAL regime (between BULL and BEAR).

REGIME_BEAR_ALPHA_THRESHOLD    = 0.42
# Stricter entry threshold in BEAR regime: only very high-conviction trades.

REGIME_NEUTRAL_ALPHA_THRESHOLD = 0.33
# Slightly stricter in NEUTRAL (vs 0.30 default in BULL).
