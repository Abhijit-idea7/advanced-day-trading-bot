"""
trade_tracker.py
----------------
In-memory state for the current trading session.

Tracks open positions (symbol → Position) for the live session.
A fresh TradeTracker is created each time the bot starts (once per market day
via GitHub Actions), so no cross-day persistence is needed here.

Extended from the original to include:
  - strategy_name field on Position (which strategy opened the trade)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pytz

from config import MAX_POSITIONS

IST    = pytz.timezone("Asia/Kolkata")
logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol:        str
    direction:     str    # "BUY" (long) or "SELL" (short)
    entry_price:   float
    sl:            float  # stop-loss price
    target:        float  # target price
    quantity:      int
    entry_time:    str    # "HH:MM" IST
    strategy_name: str = "UNKNOWN"   # strategy that opened this position
    signal_scores: dict = field(default_factory=dict)  # ALPHA_COMBO IC tracking


class TradeTracker:
    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}
        self.daily_trades: int = 0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def open_count(self) -> int:
        return len(self._positions)

    def can_open_new_trade(self) -> bool:
        """True if fewer than MAX_POSITIONS stocks are currently open."""
        return self.open_count() < MAX_POSITIONS

    def all_positions(self) -> list[Position]:
        return list(self._positions.values())

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_position(
        self,
        symbol:        str,
        direction:     str,
        entry_price:   float,
        sl:            float,
        target:        float,
        quantity:      int,
        strategy_name: str = "UNKNOWN",
        signal_scores: dict = None,
    ) -> None:
        entry_time = datetime.now(IST).strftime("%H:%M")
        self._positions[symbol] = Position(
            symbol        = symbol,
            direction     = direction,
            entry_price   = entry_price,
            sl            = sl,
            target        = target,
            quantity      = quantity,
            entry_time    = entry_time,
            strategy_name = strategy_name,
            signal_scores = signal_scores or {},
        )
        self.daily_trades += 1
        logger.info(
            f"[TRACKER] {strategy_name} | Added {direction} {symbol} | "
            f"entry={entry_price:.2f} sl={sl:.2f} tgt={target:.2f} qty={quantity} | "
            f"open={self.open_count()}/{MAX_POSITIONS} | total today={self.daily_trades}"
        )

    def remove_position(self, symbol: str) -> None:
        if symbol in self._positions:
            del self._positions[symbol]
            logger.info(f"[TRACKER] Removed {symbol} | open={self.open_count()}")

    def summary(self) -> str:
        if not self._positions:
            return "No open positions."
        lines = [f"Open positions ({self.open_count()}):"]
        for p in self._positions.values():
            arrow = "↑" if p.direction == "BUY" else "↓"
            lines.append(
                f"  {arrow} {p.symbol} [{p.strategy_name}] | "
                f"dir={p.direction} qty={p.quantity} "
                f"entry={p.entry_price:.2f} sl={p.sl:.2f} tgt={p.target:.2f} "
                f"@ {p.entry_time}"
            )
        lines.append(
            f"Open slots: {self.open_count()}/{MAX_POSITIONS} | "
            f"Total trades today: {self.daily_trades}"
        )
        return "\n".join(lines)
