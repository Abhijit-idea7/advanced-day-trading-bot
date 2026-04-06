"""
strategy_factory.py
-------------------
Selects the active strategy module(s) based on ACTIVE_STRATEGY in config.

Each strategy module must expose:
  generate_signal(df, symbol)   → dict with action/sl/target
  check_exit_signal(df, position) → str reason or None

Adding a new strategy:
  1. Create strategy_<name>.py with the two functions above
  2. Import it here and add to the STRATEGIES dict
  3. Set ACTIVE_STRATEGY = "<NAME>" in config.py or as env var
"""

import logging

from config import ACTIVE_STRATEGY

logger = logging.getLogger(__name__)


def get_strategies() -> list:
    """
    Return a list of active strategy modules based on ACTIVE_STRATEGY config.

    In COMBINED mode, both ORB and VWAP_EMA run simultaneously.
    The main loop tries each strategy in order for each candidate symbol
    and takes the first signal generated. Position slots are shared.
    """
    import strategy_orb
    import strategy_vwap_ema

    registry = {
        "ORB":      [strategy_orb],
        "VWAP_EMA": [strategy_vwap_ema],
        "COMBINED": [strategy_orb, strategy_vwap_ema],
    }

    key = ACTIVE_STRATEGY.upper().strip()
    selected = registry.get(key)

    if selected is None:
        logger.warning(
            f"Unknown ACTIVE_STRATEGY='{ACTIVE_STRATEGY}'. "
            f"Valid options: {list(registry.keys())}. Defaulting to ORB."
        )
        selected = [strategy_orb]

    names = [m.__name__ for m in selected]
    logger.info(f"Active strategies: {names}")
    return selected


def get_strategy_name(module) -> str:
    """Return a human-readable name for a strategy module."""
    return getattr(module, "STRATEGY_NAME", module.__name__)
