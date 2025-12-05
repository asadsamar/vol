import sys
import os

# Add vol directory to path - MUST BE FIRST
vol_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Remove the strats directory if it's in the path
strats_dir = os.path.dirname(os.path.abspath(__file__))
if strats_dir in sys.path:
    sys.path.remove(strats_dir)

# Insert vol_dir at the very beginning
if vol_dir in sys.path:
    sys.path.remove(vol_dir)
sys.path.insert(0, vol_dir)

from utils.book import Book
from utils.option import Option
from utils.put import Put
from utils.portfolio import OptionPortfolio

from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import threading
import time

from ibkr.ibkr import IBWebAPIClient
from ibkr.ibkr_ws import IBWebSocketClient
from vol_risk.risk_lib import (
    PositionRiskManager,
    analyze_risk_coverage
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PutSellingStrategy:
    """
    Short Put Strategy - Sell cash-secured puts for premium collection.
    
    This strategy:
    1. Loads initial positions and creates an OptionPortfolio
    2. Uses PositionRiskManager to analyze portfolio-level risk
    3. Monitors changes in real-time via WebSocket (if available) or polling
    4. Subscribes to market data for all positions
    5. Calculates cash requirements if all puts are exercised
    6. Tracks expiry dates and cash requirement timelines
    """
    
    def __init__(self, config_file: str = "ibkr/ibkr.conf", debug_symbol: Optional[str] = None):
        """
        Initialize the put selling strategy.
        
        Args:
            config_file: Path to IBKR configuration file
            debug_symbol: Optional symbol to debug (e.g., 'SPY', 'AAPL')
        """
        self.config_file = config_file
        self.client = IBWebAPIClient(config_file)
        self.ws_client = None
        self.account_id = None
        self.risk_manager = None
        self.running = False
        self.debug_symbol = debug_symbol
        
        # Portfolio management
        self.portfolio = OptionPortfolio()
        
        # Cache for current data
        self.current_cash_info = None
        self.data_lock = threading.Lock()
        
        # Market data cache
        self.market_data = {}
        self.market_data_lock = threading.Lock()
    
    def connect(self) -> bool:
        """
        Connect and authenticate with IBKR via REST and optionally WebSocket.
        Loads initial data via REST API.
        
        Returns:
            True if connection successful
        """
        # First authenticate via REST API
        if not self.client.authenticate():
            logger.error("REST API authentication failed")
            return False
        
        if not self.client.setup_account():
            logger.error("Account setup failed")
            return False
        
        self.account_id = self.client.account_id
        logger.info(f"Connected to account: {self.account_id}")
        
        # Load initial data via REST (once!)
        logger.info("Loading initial account data via REST API...")
        if not self._load_initial_data():
            logger.error("Failed to load initial data")
            return False
        
        # Try to connect WebSocket (optional - not required for basic functionality)
        logger.info("Attempting WebSocket connection for real-time updates...")
        self.ws_client = IBWebSocketClient(config_file=self.config_file, rest_client=self.client)
        
        ws_connected = self.ws_client.connect()
        
        if ws_connected:
            # Subscribe to account and portfolio updates
            self._setup_websocket_subscriptions()
            logger.info("✓ WebSocket enabled - real-time updates active")
        else:
            logger.warning("✗ WebSocket not available - using REST API polling mode")
            logger.warning("This is normal if your Gateway doesn't support WebSocket")
            self.ws_client = None
        
        logger.info("Strategy initialized successfully")
        return True
    
    def _load_initial_data(self) -> bool:
        """
        Load initial positions and account balances via REST API.
        Creates portfolio from positions.
        
        Returns:
            True if successful
        """
        try:
            # Load account balances
            cash_summary = self.client.get_account_cash_summary()
            if not cash_summary:
                logger.error("Failed to get account summary")
                return False
            
            settled_cash = cash_summary['settled_cash']
            buying_power = cash_summary['buying_power']
            net_liq_value = cash_summary['net_liq_value']
            
            with self.data_lock:
                self.current_cash_info = (settled_cash, buying_power, net_liq_value)
            
            logger.info(f"Loaded account balances: cash=${settled_cash:,.2f}, BP=${buying_power:,.2f}")
            
            # Load positions and create portfolio
            positions = self.client.get_positions()
            if not positions:
                logger.warning("No positions found")
                with self.data_lock:
                    self.portfolio = OptionPortfolio()
            else:
                with self.data_lock:
                    # Load positions into portfolio
                    self.portfolio.load_from_ibkr_positions(positions)
                
                logger.info(f"Loaded portfolio: {len(self.portfolio)} positions")
                logger.info(f"  - {len(self.portfolio.short_puts)} short puts")
                logger.info(f"  - {len(self.portfolio.long_puts)} long puts")
            
            # Initialize risk manager
            self.risk_manager = PositionRiskManager(settled_cash, buying_power)
            
            with self.data_lock:
                # Set portfolio in risk manager
                self.risk_manager.set_portfolio(self.portfolio)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading initial data: {e}", exc_info=True)
            return False
    
    def _setup_websocket_subscriptions(self):
        """Set up WebSocket subscriptions for real-time updates."""
        if not self.ws_client or not self.ws_client.is_connected:
            return
        
        # Wait for WebSocket to fully initialize
        logger.info("Waiting for WebSocket to stabilize...")
        time.sleep(5)
        
        # Subscribe to account updates
        self.ws_client.subscribe('act', self._handle_account_websocket)
        logger.info("Subscribed to account updates")
        
        # Subscribe to portfolio/position updates
        self.ws_client.subscribe('pnl', self._handle_pnl_websocket)
        logger.info("Subscribed to P&L updates")
        
        # Subscribe to order updates
        self.ws_client.subscribe('ord', self._handle_order_websocket)
        logger.info("Subscribed to order updates")
        
        # Subscribe to system status messages
        self.ws_client.subscribe('system', self._handle_system_websocket)
        logger.info("Subscribed to system updates")
        
        # Subscribe to server status messages
        self.ws_client.subscribe('sts', self._handle_server_status_websocket)
        logger.info("Subscribed to server status")
        
        # Wait a bit more before subscribing to market data
        logger.info("Waiting before subscribing to market data...")
        time.sleep(5)
        
        # Subscribe to market data for all positions
        self._subscribe_position_market_data()
    
    def _subscribe_position_market_data(self):
        """Subscribe to market data for all positions or a specific debug symbol."""
        try:
            with self.data_lock:
                options = list(self.portfolio.options)
            
            if not options:
                logger.info("No positions to subscribe to")
                return
            
            logger.info(f"Total positions in portfolio: {len(options)}")
            
            # Log all underlyers available
            if self.debug_symbol:
                underlyers = set(opt.underlyer for opt in options)
                logger.info(f"Available underlyers: {', '.join(sorted(underlyers))}")
            
            # Filter to debug symbol if specified
            if self.debug_symbol:
                filtered_options = [opt for opt in options if opt.underlyer == self.debug_symbol]
                logger.info(f"Filtering for {self.debug_symbol}: found {len(filtered_options)} position(s)")
                
                if not filtered_options:
                    logger.warning(f"❌ No positions found for debug symbol: {self.debug_symbol}")
                    logger.warning(f"Available symbols: {', '.join(sorted(set(opt.underlyer for opt in options)))}")
                    return
                
                options = filtered_options
                logger.info(f"✓ DEBUG MODE: Subscribing only to {self.debug_symbol} ({len(options)} position(s))")
            else:
                logger.info(f"Subscribing to market data for all {len(options)} positions...")
            
            # Subscribe to market data for each contract
            # Pass the option's Book object directly - no callback needed!
            subscribed = 0
            failed = 0
            for opt in options:
                if not opt.conid:
                    logger.warning(f"Skipping {opt.symbol} - no conid")
                    continue
                
                logger.info(f"Subscribing to market data for {opt.symbol} (conid={opt.conid})...")
                
                # Pass the option's book object - it will be updated automatically
                if self.ws_client.subscribe_market_data(opt.conid, opt.option_book):
                    subscribed += 1
                    logger.info(f"  ✓ Subscribed successfully")
                else:
                    failed += 1
                    logger.warning(f"  ✗ Subscription failed")
                
                time.sleep(0.1)
            
            logger.info(f"Market data subscription summary: {subscribed} succeeded, {failed} failed")
            
        except Exception as e:
            logger.error(f"Error subscribing to position market data: {e}", exc_info=True)
    
    def _handle_system_websocket(self, data: Dict):
        """Handle system message from WebSocket."""
        # Filter out heartbeat messages
        if 'hb' in data:
            logger.debug(f"Heartbeat: {data.get('hb')}")
            return
        
        logger.info(f"System update: {data}")
        is_paper = data.get('isPaper', False)
        username = data.get('success', 'unknown')
        
        if is_paper:
            logger.warning(f"Connected to PAPER trading account: {username}")
        else:
            logger.info(f"Connected to LIVE trading account: {username}")
    
    def _handle_server_status_websocket(self, data: Dict):
        """Handle server status message from WebSocket."""
        args = data.get('args', {})
        connected = args.get('connected', False)
        authenticated = args.get('authenticated', False)
        competing = args.get('competing', False)
        
        if not connected or not authenticated:
            logger.warning(f"Server status issue - Connected: {connected}, Authenticated: {authenticated}")
        
        if competing:
            logger.warning("Competing session detected - another connection is active")
    
    def _handle_account_websocket(self, data: Dict):
        """Handle account update from WebSocket."""
        logger.info("Account WebSocket update received")
        # Account updates will trigger a portfolio refresh
        self._print_risk_report_async()
    
    def _handle_pnl_websocket(self, data: Dict):
        """Handle P&L/position update from WebSocket."""
        logger.info("P&L WebSocket update received")
        # Position updates will trigger a portfolio refresh
        self._print_risk_report_async()
    
    def _handle_order_websocket(self, data: Dict):
        """Handle order update from WebSocket."""
        logger.info(f"Order WebSocket update: {data}")
        status = data.get('status', '').lower()
        if status == 'filled':
            logger.info("Order filled - position may have changed")
            self._print_risk_report_async()
    
    def _print_risk_report_async(self):
        """Print risk report asynchronously."""
        threading.Thread(target=self._print_risk_report_safe, daemon=True).start()
    
    def _print_risk_report_safe(self):
        """Print risk report with proper locking."""
        try:
            with self.data_lock:
                cash_info = self.current_cash_info
                short_puts = self.portfolio.short_puts.copy()
            
            if cash_info:
                self.print_risk_report(short_puts, cash_info)
        except Exception as e:
            logger.error(f"Error printing risk report: {e}", exc_info=True)
    
    def get_account_cash(self) -> Optional[Tuple[float, float, float]]:
        """Get settled cash, buying power, and net liquidation value."""
        with self.data_lock:
            return self.current_cash_info
    
    def get_short_put_positions(self) -> List[Put]:
        """Get short put positions from portfolio."""
        with self.data_lock:
            return self.portfolio.short_puts.copy()
    
    def print_risk_report(self, short_puts: List[Put], cash_info: Tuple[float, float, float]):
        """Print comprehensive risk report using portfolio data."""
        settled_cash, buying_power, net_liq_value = cash_info
        
        print("\n" + "=" * 120)
        print(f"PUT SELLING STRATEGY - RISK MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)
        
        # Account balances
        print("\n1. ACCOUNT BALANCES")
        print("-" * 120)
        print(f"Settled Cash:         ${settled_cash:>18,.2f}")
        print(f"Buying Power:         ${buying_power:>18,.2f}")
        print(f"Net Liquidation:      ${net_liq_value:>18,.2f}")
        
        # Portfolio summary
        print("\n2. PORTFOLIO SUMMARY")
        print("-" * 120)
        with self.data_lock:
            summary = self.portfolio.get_portfolio_summary()
        
        print(f"Total Positions:      {summary['total_positions']:>18}")
        print(f"  - Short Puts:       {summary['short_puts']:>18}")
        print(f"  - Long Puts:        {summary['long_puts']:>18}")
        print(f"Unique Underlyers:    {summary['underlyers']:>18}")
        print(f"Total Notional:       ${summary['total_notional']:>18,.2f}")
        
        if not short_puts:
            print("\n3. SHORT PUT POSITIONS")
            print("-" * 120)
            print("No short put positions found")
            print("\n" + "=" * 120)
            return
        
        # Short put positions
        print("\n3. SHORT PUT POSITIONS")
        print("-" * 120)
        print(f"{'Symbol':<25} {'Qty':<8} {'Strike':<12} {'Days':<8} {'Moneyness':<12} "
              f"{'Option Bid/Ask':<20} {'Exercise Risk':<18}")
        print("-" * 120)
        
        for put in short_puts:
            bid_ask = f"${put.option_book.bid:.2f}/${put.option_book.ask:.2f}" if put.option_book.has_quotes else "N/A"
            
            print(f"{str(put):<25} {abs(put.quantity):<8.0f} ${put.strike:<11.2f} "
                  f"{put.days_to_expiry:<8} {put.moneyness_description:<12} "
                  f"{bid_ask:<20} ${put.exercise_cash_requirement:>16,.2f}")
        
        # Risk analysis
        print("\n4. EXERCISE RISK ANALYSIS")
        print("-" * 120)
        total_exercise_risk = summary['total_exercise_risk']
        total_margin = summary['total_margin_requirement']
        print(f"Total Exercise Risk:  ${total_exercise_risk:>18,.2f}")
        print(f"Total Margin Req:     ${total_margin:>18,.2f}")
        
        # Expiry timeline
        print("\n5. EXPIRY TIMELINE")
        print("-" * 120)
        with self.data_lock:
            timeline = self.portfolio.get_expiry_timeline()
        
        if timeline:
            print(f"{'Expiry Date':<15} {'Days':<8} {'# Positions':<15} "
                  f"{'Cash Needed':<20} {'Cumulative':<20}")
            print("-" * 120)
            for item in timeline:
                expiry_str = item['expiry'].strftime('%Y-%m-%d')
                days = item['days']
                num_positions = len(item['positions'])
                requirement = item['cash_requirement']
                cumulative = item['cumulative']
                
                print(f"{expiry_str:<15} {days:<8} {num_positions:<15} "
                      f"${requirement:>18,.2f} ${cumulative:>18,.2f}")
        
        # Coverage analysis
        print("\n6. COVERAGE ANALYSIS")
        print("-" * 120)
        risk_analysis = analyze_risk_coverage(settled_cash, buying_power, total_exercise_risk)
        
        print(f"Settled Cash Coverage:    {risk_analysis['cash_coverage_pct']:>6.2f}%")
        print(f"Buying Power Coverage:    {risk_analysis['buying_power_coverage_pct']:>6.2f}%")
        
        # Risk assessment
        print("\n7. RISK ASSESSMENT")
        print("-" * 120)
        
        status_symbols = {
            'FULLY_SECURED': '✓',
            'MARGIN_SECURED': '⚠',
            'INSUFFICIENT': '✗'
        }
        
        symbol = status_symbols.get(risk_analysis['status'], '?')
        print(f"Status: {symbol} {risk_analysis['status'].replace('_', ' ')}")
        print(f"Risk Level: {risk_analysis['risk_level']}")
        
        if risk_analysis['excess_cash'] > 0:
            print(f"Excess Cash: ${risk_analysis['excess_cash']:,.2f}")
        elif risk_analysis['shortfall'] > 0:
            if risk_analysis['status'] == 'MARGIN_SECURED':
                print(f"Cash Shortfall: ${risk_analysis['shortfall']:,.2f}")
                print(f"Excess Buying Power: ${buying_power - total_exercise_risk:,.2f}")
            else:
                print(f"Total Shortfall: ${risk_analysis['shortfall']:,.2f}")
                print("\n⚠ WARNING: Insufficient funds to cover all assignments!")
        
        # Risk manager status
        if self.risk_manager:
            print("\n8. CAPACITY ANALYSIS")
            print("-" * 120)
            risk_summary = self.risk_manager.get_risk_summary()
            print(f"Cash Available:           ${risk_summary['cash_available']:>18,.2f}")
            print(f"Cash Utilization:         {risk_summary['cash_utilization_pct']:>18.2f}%")
            print(f"Max Contracts @ $100:     {self.risk_manager.get_max_contracts(100.0):>18}")
            print(f"Max Contracts @ $50:      {self.risk_manager.get_max_contracts(50.0):>18}")
            print(f"Max Contracts @ $25:      {self.risk_manager.get_max_contracts(25.0):>18}")
        
        # WebSocket status
        if self.ws_client:
            print("\n9. CONNECTION STATUS")
            print("-" * 120)
            ws_status = "Connected" if self.ws_client.is_connected else "Disconnected"
            print(f"WebSocket Status:         {ws_status:>18}")
        
        print("\n" + "=" * 120)
    
    def check_new_order(self, strike: float, quantity: int, use_cash_limit: bool = True) -> Dict:
        """Check if a new short put order is within risk limits."""
        if not self.risk_manager:
            logger.error("Risk manager not initialized")
            return {'approved': False, 'message': 'Risk manager not initialized'}
        
        return self.risk_manager.check_new_put_order(strike, quantity, use_cash_limit=use_cash_limit)
    
    def monitor_positions(self):
        """Monitor positions continuously using WebSocket or REST API polling."""
        logger.info("=" * 80)
        if self.ws_client and self.ws_client.is_connected:
            logger.info("Starting REAL-TIME position monitor (WebSocket mode)")
            logger.info("Updates arrive via WebSocket callbacks")
            mode = 'websocket'
        else:
            logger.info("Starting position monitor (REST API polling mode)")
            logger.info("Refreshing every 10 seconds")
            mode = 'polling'
        
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80)
        
        self.running = True
        
        # Print initial report
        cash_info = self.get_account_cash()
        short_puts = self.get_short_put_positions()
        if cash_info:
            self.print_risk_report(short_puts, cash_info)
        
        try:
            if mode == 'websocket':
                # WebSocket mode - just keep alive
                while self.running:
                    time.sleep(1)
            else:
                # Polling mode - refresh periodically
                while self.running:
                    time.sleep(10)
                    logger.info("Refreshing positions...")
                    if self._load_initial_data():
                        cash_info = self.get_account_cash()
                        short_puts = self.get_short_put_positions()
                        if cash_info:
                            self.print_risk_report(short_puts, cash_info)
                
        except KeyboardInterrupt:
            logger.info("\nStopping position monitor...")
            self.running = False
        except Exception as e:
            logger.error(f"Error in position monitor: {e}", exc_info=True)
            self.running = False
        finally:
            if self.ws_client and self.ws_client.is_connected:
                self.ws_client.disconnect()
    
    def analyze_positions_once(self, debug: bool = False):
        """Analyze current positions once and print report."""
        cash_info = self.get_account_cash()
        if not cash_info:
            print("No account cash information available")
            return
        
        short_puts = self.get_short_put_positions()
        self.print_risk_report(short_puts, cash_info)


def main():
    """Main entry point for the put selling strategy."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Put Selling Strategy Real-Time Risk Monitor')
    parser.add_argument('--config', type=str, default='ibkr/ibkr.conf',
                       help='Path to IBKR configuration file (default: ibkr/ibkr.conf)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--monitor', action='store_true', help='Run in continuous monitoring mode')
    parser.add_argument('--debug-symbol', type=str, metavar='SYMBOL',
                       help='Subscribe to market data for only this symbol (e.g., SPY, AAPL)')
    parser.add_argument('--check-order', nargs=2, type=float, metavar=('STRIKE', 'QTY'),
                       help='Check if a new order would be approved (strike quantity)')
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    strategy = PutSellingStrategy(config_file=args.config, debug_symbol=args.debug_symbol)
    
    if not strategy.connect():
        print("Failed to connect to IBKR")
        return
    
    if args.check_order:
        strike, quantity = args.check_order
        print("\n" + "=" * 100)
        print("NEW ORDER RISK CHECK")
        print("=" * 100)
        result = strategy.check_new_order(strike, int(quantity))
        print(f"\n{result['message']}")
        print("\n" + "=" * 100)
        return
    
    if args.monitor:
        strategy.monitor_positions()
    else:
        strategy.analyze_positions_once(debug=args.debug)


if __name__ == "__main__":
    main()