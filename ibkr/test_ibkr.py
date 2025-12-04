import json
from ibkr import IBWebAPIClient


def main():
    """
    Test script for IB Web API Client
    """
    # Initialize client (will load from ibkr.conf)
    client = IBWebAPIClient()
    
    print("=" * 60)
    print("Interactive Brokers Web API - Account Balance Test")
    print("=" * 60)
    
    # Step 1: Check authentication
    print("\n1. Checking authentication...")
    if not client.authenticate():
        print("\nPlease complete these steps:")
        print("1. Download Client Portal Gateway from:")
        print("   https://www.interactivebrokers.com/en/trading/cpgw.php")
        print("2. Start the gateway application")
        host = client.config.get('api', 'host')
        port = client.config.get('api', 'port')
        print(f"3. Open https://{host}:{port} in your browser")
        print("4. Log in with your IB credentials")
        print("5. Run this script again")
        return
    
    # Step 2: Keep session alive
    print("\n2. Keeping session alive...")
    client.tickle()
    
    # Step 3: Setup account
    print("\n3. Setting up account...")
    if not client.setup_account():
        print("Failed to setup account. Check configuration.")
        return
    
    print(f"\nUsing account: {client.account_id}")
    
    # Step 4: Get account info
    print("\n4. Account Information:")
    account_info = client.get_account_info()
    if account_info:
        print(f"  Account ID: {account_info['accountId']}")
        print(f"  Title: {account_info['accountTitle']}")
        print(f"  Type: {account_info['type']}")
        print(f"  Trading Type: {account_info['tradingType']}")
        print(f"  Currency: {account_info['currency']}")
    
    # Step 5: Get account balance
    print(f"\n{'=' * 60}")
    print("Account Summary")
    print(f"{'=' * 60}")
    balance = client.get_account_balance()
    if balance:
        print(json.dumps(balance, indent=2))
    
    # Step 6: Get account ledger
    print(f"\n{'=' * 60}")
    print("Account Ledger")
    print(f"{'=' * 60}")
    ledger = client.get_account_ledger()
    if ledger:
        print(json.dumps(ledger, indent=2))
    
    # Step 7: Get positions
    print(f"\n{'=' * 60}")
    print("Current Positions")
    print(f"{'=' * 60}")
    
    positions = client.get_positions()
    if positions:
        print(f"\nTotal positions: {len(positions)}")
        print("\nDetailed positions:")
        print(json.dumps(positions, indent=2))
    else:
        print("No positions found or unable to retrieve positions")
    
    # Step 8: Get position summary (formatted)
    print(f"\n{'=' * 60}")
    print("Position Summary")
    print(f"{'=' * 60}")
    
    position_summary = client.get_position_summary()
    if position_summary:
        print(f"\nTotal Positions: {position_summary['total_positions']}\n")
        for idx, pos in enumerate(position_summary['positions'], 1):
            print(f"Position {idx}:")
            print(f"  Symbol: {pos['symbol']} ({pos['ticker']})")
            print(f"  Asset Class: {pos['asset_class']}")
            print(f"  Quantity: {pos['position']}")
            print(f"  Market Price: {pos['market_price']} {pos['currency']}")
            print(f"  Market Value: {pos['market_value']} {pos['currency']}")
            print(f"  Average Cost: {pos['average_cost']}")
            print(f"  Unrealized P&L: {pos['unrealized_pnl']} {pos['currency']}")
            print(f"  Realized P&L: {pos['realized_pnl']} {pos['currency']}")
            print()


if __name__ == "__main__":
    main()