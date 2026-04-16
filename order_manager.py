"""
order_manager.py
----------------
Sends order webhooks to stocksdeveloper.in which routes them to Zerodha.

Payload format:
{
    "command": "PLACE_ORDERS",
    "orders": [{
        "variety":     "REGULAR",
        "exchange":    "NSE",
        "symbol":      "TATAMOTORS",
        "tradeType":   "BUY" | "SELL",
        "orderType":   "MARKET",
        "productType": "INTRADAY",
        "quantity":    100
    }]
}
"""

import logging

import requests

from config import (
    EXCHANGE,
    ORDER_TYPE,
    POSITION_SIZE_INR,
    PRODUCT_TYPE,
    STOCKSDEVELOPER_ACCOUNT,
    STOCKSDEVELOPER_API_KEY,
    STOCKSDEVELOPER_URL,
    VARIETY,
)

logger   = logging.getLogger(__name__)
_TIMEOUT = 10  # seconds


def _build_payload(symbol: str, trade_type: str, quantity: int) -> dict:
    return {
        "command": "PLACE_ORDERS",
        "orders": [
            {
                "variety":     VARIETY,
                "exchange":    EXCHANGE,
                "symbol":      symbol,
                "tradeType":   trade_type,
                "orderType":   ORDER_TYPE,
                "productType": PRODUCT_TYPE,
                "quantity":    quantity,
            }
        ],
    }


def _send_webhook(payload: dict) -> bool:
    """POST to stocksdeveloper and return True on HTTP 200."""
    params = {
        "apiKey":  STOCKSDEVELOPER_API_KEY,
        "account": STOCKSDEVELOPER_ACCOUNT,
        "group":   "false",
    }
    try:
        resp = requests.post(
            STOCKSDEVELOPER_URL,
            params=params,
            json=payload,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            logger.info(f"Webhook OK [{resp.status_code}]: {payload['orders'][0]}")
            return True
        else:
            logger.error(f"Webhook FAILED [{resp.status_code}]: {resp.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"Webhook exception: {e}")
        return False


def place_order(symbol: str, trade_type: str, quantity: int) -> bool:
    """Place a new entry order. trade_type: 'BUY' for long, 'SELL' for short."""
    if quantity < 1:
        logger.warning(f"{symbol}: quantity {quantity} is invalid, skipping order.")
        return False
    payload = _build_payload(symbol, trade_type, quantity)
    logger.info(f"Placing {trade_type} order — {symbol} × {quantity}")
    return _send_webhook(payload)


def square_off(symbol: str, open_direction: str, quantity: int) -> bool:
    """Close an existing position with the reverse order."""
    exit_type = "SELL" if open_direction == "BUY" else "BUY"
    payload   = _build_payload(symbol, exit_type, quantity)
    logger.info(f"Squaring off {symbol} ({open_direction} → {exit_type}) × {quantity}")
    return _send_webhook(payload)


def calculate_quantity(price: float, scale: float = 1.0) -> int:
    """
    Compute share quantity for a trade.

    For standard strategies: scale = 1.0 → POSITION_SIZE_INR / price.
    For ALPHA_COMBO: scale = signal["quantity_scale"] ∈ (0, 1] so that
    stronger alpha conviction allocates more capital within the cap.

    Always returns at least 0 (never negative).
    """
    if price <= 0:
        return 0
    # scale > 1.0 is intentional for ORB (ORB_POSITION_SCALE = 1.5)
    # scale < 1.0 is used by ALPHA_COMBO for conviction-proportional sizing
    effective_capital = POSITION_SIZE_INR * max(0.1, scale)
    return int(effective_capital // price)
