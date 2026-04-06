"""
test_webhook.py
---------------
Quick connectivity test for the stocksdeveloper.in webhook.
Sends a single test order for 1 share of SAIL (low price, safe for testing).
If the market is closed, the webhook still confirms the connection works.

Run manually before market open:
    python test_webhook.py
Or trigger via the "Test Webhook Connectivity" GitHub Actions workflow.
"""

import os
from dotenv import load_dotenv
import requests

load_dotenv()

API_KEY = os.getenv("STOCKSDEVELOPER_API_KEY")
ACCOUNT = os.getenv("STOCKSDEVELOPER_ACCOUNT", "AbhiZerodha")
URL     = "https://tv.stocksdeveloper.in/"

if not API_KEY:
    print("ERROR: STOCKSDEVELOPER_API_KEY not set.")
    raise SystemExit(1)

payload = {
    "command": "PLACE_ORDERS",
    "orders": [
        {
            "variety":     "REGULAR",
            "exchange":    "NSE",
            "symbol":      "SAIL",
            "tradeType":   "BUY",
            "orderType":   "MARKET",
            "productType": "INTRADAY",
            "quantity":    1,
        }
    ],
}

params = {
    "apiKey":  API_KEY,
    "account": ACCOUNT,
    "group":   "false",
}

print(f"Testing webhook: {URL}")
print(f"Account        : {ACCOUNT}")
print(f"Payload        : {payload}")
print()

try:
    resp = requests.post(URL, params=params, json=payload, timeout=10)
    print(f"HTTP Status : {resp.status_code}")
    print(f"Response    : {resp.text}")
    if resp.status_code == 200:
        print("\nWebhook connectivity: OK")
    else:
        print("\nWebhook connectivity: FAILED (check API key and account name)")
except requests.RequestException as e:
    print(f"\nWebhook exception: {e}")
    raise SystemExit(1)
