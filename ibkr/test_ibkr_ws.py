import time
import json
from ibkr_ws import IBWebSocketClient
from ibkr import IBWebAPIClient


def market_data_callback(data):
    """Callback for market data updates."""
    print(f"\n[MARKET DATA] {json.dumps(data, indent=2)}")


def order_callback(data):
    """Callback for order updates."""
    print(f"\n[ORDER] {json.dumps(data, indent=2)}")


def pnl_callback(data):
    """Callback for P&L updates."""
    print(f"\n[P&L] {json.dumps(data, indent=2)}")


def account_callback(data):
    """Callback for account updates."""
    print(f"\n[ACCOUNT] {json.dumps(data, indent=2)}")


def main():
    """
    Test script for WebSocket API.
    """
    print("=" * 60)
    print("Interactive Brokers WebSocket API Test")
    print("=" * 60)
    
    # First, authenticate using REST API
    print("\n1. Authenticating via REST API...")
    rest_client = IBWebAPIClient()
    
    if not rest_client.authenticate():
        print("Authentication failed. Please authenticate first.")
        return
    
    if not rest_client.setup_account():
        print("Account setup failed.")
        return
    
    account_id = rest_client.account_id
    print(f"Using account: {account_id}")
    
    # Initialize WebSocket client
    print("\n2. Connecting to WebSocket...")
    ws_client = IBWebSocketClient()
    
    if not ws_client.connect():
        print("Failed to connect to WebSocket")
        return
    
    print("WebSocket connected!")
    
    # Subscribe to different topics
    print("\n3. Setting up subscriptions...")
    
    # Subscribe to orders
    print("Subscribing to order updates...")
    ws_client.subscribe_orders(callback=order_callback)
    
    # Subscribe to P&L
    print(f"Subscribing to P&L updates for account {account_id}...")
    ws_client.subscribe_pnl(account_id, callback=pnl_callback)
    
    # Subscribe to account updates
    print("Subscribing to account updates...")
    ws_client.subscribe_account(callback=account_callback)
    
    # Example: Subscribe to market data for a specific contract
    # You would need to replace this with an actual contract ID (conid)
    # To get conid, you can use the REST API to search for a symbol
    # Example conid for AAPL: 265598
    print("\nTo subscribe to market data, you need a contract ID (conid).")
    print("Example: ws_client.subscribe_market_data(265598, ['31', '84', '86'], callback=market_data_callback)")
    print("Field IDs: 31=Last Price, 84=Bid, 85=Bid Size, 86=Ask, 88=Ask Size")
    
    # Show active subscriptions
    print(f"\n4. Active subscriptions: {ws_client.get_active_subscriptions()}")
    
    # Keep running to receive updates
    print("\n5. Listening for updates... (Press Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        ws_client.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()