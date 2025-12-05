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
import websocket
import json
import requests
import configparser

from ibkr.ibkr import IBWebAPIClient
from ibkr.ibkr_ws import IBWebSocketClient
from vol_risk.risk_lib import (
    PositionRiskManager,
    analyze_risk_coverage
)

from ibkr.fields import IBKRMarketDataFields

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
        
        # Last report state for change detection
        self.last_report_state = None
        self.last_report_lock = threading.Lock()
    
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
        time.sleep(2)
        
        # Subscribe to account updates
        #self.ws_client.subscribe_account_updates(self._handle_account_update)
        
        # Subscribe to portfolio updates
        #self.ws_client.subscribe_portfolio_updates(self._handle_portfolio_update)
        
        # Subscribe to order updates
        #self.ws_client.subscribe_order_updates(self._handle_order_update)
        
        # Wait before subscribing to market data
        logger.info("Waiting before subscribing to market data...")
        time.sleep(2)
        
        # Subscribe to market data for positions
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
            
            # Filter to debug symbol if specified
            if self.debug_symbol:
                underlyers = set(opt.underlyer for opt in options)
                logger.info(f"Available underlyers: {', '.join(sorted(underlyers))}")
                
                filtered_options = [opt for opt in options if opt.underlyer == self.debug_symbol]
                logger.info(f"Filtering for {self.debug_symbol}: found {len(filtered_options)} position(s)")
                
                if not filtered_options:
                    logger.warning(f"❌ No positions found for debug symbol: {self.debug_symbol}")
                    logger.warning(f"Available symbols: {', '.join(sorted(underlyers))}")
                    return
                
                options = filtered_options
                logger.info(f"✓ DEBUG MODE: Subscribing only to {self.debug_symbol} ({len(options)} position(s))")
            else:
                logger.info(f"Subscribing to market data for all {len(options)} positions...")
            
            # Use the simple interface - no need to know about topics or subscription details
            subscribed, failed = self.ws_client.subscribe_all_position_market_data(options)
            
        except Exception as e:
            logger.error(f"Error subscribing to position market data: {e}", exc_info=True)
    
    def _handle_account_update(self, data: Dict):
        """Handle account update."""
        logger.info("Account update received")
        self._print_risk_report_async()
    
    def _handle_portfolio_update(self, data: Dict):
        """Handle portfolio update."""
        logger.info("Portfolio update received")
        self._print_risk_report_async()
    
    def _handle_order_update(self, data: Dict):
        """Handle order update."""
        logger.info(f"Order update: {data.get('status', 'unknown')}")
        status = data.get('status', '').lower()
        if status == 'filled':
            logger.info("Order filled - position changed")
            self._print_risk_report_async()
    
    def _get_report_state(self) -> Dict:
        """
        Get current state for change detection.
        
        Returns:
            Dictionary with key state values
        """
        with self.data_lock:
            cash_info = self.current_cash_info
            summary = self.portfolio.get_portfolio_summary()
            
            # Get position details
            positions_state = []
            for opt in self.portfolio.options:
                positions_state.append({
                    'symbol': str(opt),
                    'quantity': opt.quantity,
                    'strike': opt.strike,
                    'expiry': opt.expiry,
                    'bid': opt.option_book.bid,
                    'ask': opt.option_book.ask,
                })
        
        return {
            'cash': cash_info,
            'summary': summary,
            'positions': positions_state,
            'timestamp': datetime.now()
        }
    
    def _has_state_changed(self, new_state: Dict) -> bool:
        """
        Check if report state has changed significantly.
        
        Args:
            new_state: New state to compare
            
        Returns:
            True if state has changed
        """
        with self.last_report_lock:
            if self.last_report_state is None:
                return True  # First time, always print
            
            old_state = self.last_report_state
            
            # Check if cash changed
            if new_state['cash'] != old_state['cash']:
                logger.debug("Cash info changed")
                return True
            
            # Check if summary changed (position count, notional, etc.)
            old_summary = old_state['summary']
            new_summary = new_state['summary']
            
            if (old_summary['total_positions'] != new_summary['total_positions'] or
                old_summary['total_notional'] != new_summary['total_notional'] or
                old_summary['total_exercise_risk'] != new_summary['total_exercise_risk']):
                logger.debug("Portfolio summary changed")
                return True
            
            # Check if positions changed
            old_positions = {p['symbol']: p for p in old_state['positions']}
            new_positions = {p['symbol']: p for p in new_state['positions']}
            
            # Check for added/removed positions
            if set(old_positions.keys()) != set(new_positions.keys()):
                logger.debug("Positions added or removed")
                return True
            
            # Check if any position details changed (quantity, strike, or prices)
            for symbol, new_pos in new_positions.items():
                old_pos = old_positions[symbol]
                
                # Check quantity or strike
                if (old_pos['quantity'] != new_pos['quantity'] or
                    old_pos['strike'] != new_pos['strike']):
                    logger.debug(f"Position {symbol} details changed")
                    return True
                
                # Check if prices changed significantly (more than $0.01)
                if old_pos['bid'] is not None and new_pos['bid'] is not None:
                    if abs(old_pos['bid'] - new_pos['bid']) > 0.01:
                        logger.debug(f"Position {symbol} bid changed: {old_pos['bid']:.2f} -> {new_pos['bid']:.2f}")
                        return True
                
                if old_pos['ask'] is not None and new_pos['ask'] is not None:
                    if abs(old_pos['ask'] - new_pos['ask']) > 0.01:
                        logger.debug(f"Position {symbol} ask changed: {old_pos['ask']:.2f} -> {new_pos['ask']:.2f}")
                        return True
            
            # Check if too much time has passed (force refresh every 5 minutes)
            time_since_last = (new_state['timestamp'] - old_state['timestamp']).total_seconds()
            if time_since_last > 300:  # 5 minutes
                logger.debug("Force refresh - 5 minutes elapsed")
                return True
            
            return False
    
    def _print_risk_report_async(self):
        """Print risk report asynchronously only if state changed."""
        threading.Thread(target=self._print_risk_report_safe, daemon=True).start()
    
    def _print_risk_report_safe(self):
        """Print risk report with proper locking and change detection."""
        try:
            # Get current state
            new_state = self._get_report_state()
            
            # Check if state changed
            if not self._has_state_changed(new_state):
                logger.debug("No significant changes, skipping report")
                return
            
            # Update last state
            with self.last_report_lock:
                self.last_report_state = new_state
            
            # Print the report
            with self.data_lock:
                cash_info = self.current_cash_info
                short_puts = self.portfolio.short_puts.copy()
            
            if cash_info:
                logger.info("State changed - printing risk report")
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
        """Monitor positions in real-time with periodic summary updates."""
        try:
            self.running = True
            logger.info("Starting real-time position monitoring...")
            logger.info("Press Ctrl+C to stop")
            
            last_summary_time = 0
            summary_interval = 5  # seconds
            
            while self.running:
                try:
                    current_time = time.time()
                    
                    # Print one-line summary every 5 seconds
                    if current_time - last_summary_time >= summary_interval:
                        self._print_one_line_summary()
                        last_summary_time = current_time
                    
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info("\nStopping monitoring...")
                    self.running = False
                    break
                    
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}", exc_info=True)
        finally:
            self.running = False
            if self.ws_client:
                self.ws_client.disconnect()
    
    def _print_one_line_summary(self):
        """Print a single-line portfolio summary."""
        try:
            with self.data_lock:
                cash_info = self.current_cash_info
            
            if not cash_info:
                return
            
            # Let the portfolio handle its own summary
            self.portfolio.print_one_line_summary(cash_info)
            
            # Check for high delta short puts that should be rolled
            self._print_high_delta_positions()
            
        except Exception as e:
            logger.error(f"Error printing summary: {e}", exc_info=True)
    
    def _print_high_delta_positions(self, delta_threshold: float = 0.5):
        """
        Print short puts with delta above threshold that should be considered for rolling.
        
        Args:
            delta_threshold: Delta threshold (default: 0.5 means 50% probability of exercise)
        """
        try:
            with self.data_lock:
                short_puts = self.portfolio.short_puts
            
            # Find positions with high delta
            high_delta_positions = []
            for put in short_puts:
                if put.delta is not None and abs(put.delta) > delta_threshold:
                    high_delta_positions.append(put)
            
            if not high_delta_positions:
                return
            
            # Sort by delta (highest risk first)
            high_delta_positions.sort(key=lambda p: abs(p.delta), reverse=True)
            
            print(f"\n⚠️  HIGH DELTA SHORT PUTS - CONSIDER ROLLING ({len(high_delta_positions)} positions):")
            print(f"{'Symbol':<10} {'Strike':<10} {'Expiry':<12} {'Days':<6} {'Delta':<8} {'Prob%':<7} {'Bid/Ask':<12} {'Action':<30}")
            print("-" * 100)
            
            for put in high_delta_positions:
                # Calculate next Friday
                days_to_expiry = put.days_to_expiry
                today = datetime.now().date()
                days_until_friday = (4 - today.weekday()) % 7  # Friday is 4
                if days_until_friday == 0:
                    days_until_friday = 7  # If today is Friday, target next Friday
                next_friday = today + timedelta(days=days_until_friday)
                
                # Suggest lower strike (5-10% below current)
                suggested_strike = put.strike * 0.95
                
                prob_pct = abs(put.delta) * 100
                bid_ask = f"${put.option_book.bid:.2f}/${put.option_book.ask:.2f}" if put.option_book.has_quotes else "N/A"
                
                action = f"Roll to {next_friday.strftime('%m/%d')} ${suggested_strike:.0f}"
                
                print(f"{put.underlyer:<10} ${put.strike:<9.2f} {put.expiry.strftime('%Y-%m-%d'):<12} "
                      f"{days_to_expiry:<6} {put.delta:<8.3f} {prob_pct:<7.1f} {bid_ask:<12} {action:<30}")
            
            print()
            
        except Exception as e:
            logger.error(f"Error printing high delta positions: {e}", exc_info=True)
    
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