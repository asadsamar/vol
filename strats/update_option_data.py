"""
Option Data Updater - Continuously fetch and cache option chain data.

This script:
1. Reads configuration for indices, expiries, and target delta
2. Gets initial snapshot of option data via REST API
3. Subscribes to WebSocket for real-time updates
4. Saves updated data to cache files with file locking
"""
import sys
import os

# Add vol directory to path
vol_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if vol_dir in sys.path:
    sys.path.remove(vol_dir)
sys.path.insert(0, vol_dir)

import logging
import time
import signal
import configparser
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
import threading

from ibkr.ibkr import IBWebAPIClient
from ibkr.ibkr_ws import IBWebSocketClient
from utils.option_data import OptionDataManager, OptionChainConfig
from utils.index_constituents import IndexConstituents

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptionDataUpdater:
    """Manages continuous option data updates via REST + WebSocket."""
    
    def __init__(self, config_file: str):
        """
        Initialize the option data updater.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        
        # Load configuration
        config_parser = configparser.ConfigParser()
        config_parser.read(config_file)
        
        # Load option data config
        self.config = OptionChainConfig.from_config(config_parser)
        
        # IBKR clients
        ibkr_config_file = config_parser.get('IBKR', 'config_file', fallback='ibkr/ibkr.conf')
        self.rest_client = IBWebAPIClient(ibkr_config_file)
        self.ws_client = None
        
        # Option data manager
        self.data_manager = OptionDataManager(self.rest_client, self.config)
        
        # State
        self.running = False
        self.symbols_to_track = []
        self.option_conids = {}  # symbol -> {expiry -> [conids]}
        self.market_data_lock = threading.Lock()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Shutdown signal received, stopping...")
        self.running = False
    
    def connect(self) -> bool:
        """Connect to IBKR REST and WebSocket APIs."""
        # Connect to REST API
        if not self.rest_client.authenticate():
            logger.error("REST API authentication failed")
            return False
        
        if not self.rest_client.setup_account():
            logger.error("Account setup failed")
            return False
        
        logger.info(f"Connected to IBKR account: {self.rest_client.account_id}")
        
        # Connect to WebSocket
        logger.info("Connecting to WebSocket for real-time updates...")
        self.ws_client = IBWebSocketClient(
            config_file='ibkr/ibkr.conf',
            rest_client=self.rest_client
        )
        
        if not self.ws_client.connect():
            logger.error("WebSocket connection failed")
            return False
        
        logger.info("✓ WebSocket connected")
        
        # Wait for WebSocket to stabilize
        time.sleep(2)
        
        return True
    
    def initialize_symbols(self) -> bool:
        """
        Get list of symbols to track from configured indices or test_symbols.
        
        Returns:
            True if successful
        """
        # Check if test mode
        if self.config.test_symbols:
            self.symbols_to_track = self.config.test_symbols
            logger.info(f"TEST MODE: Tracking {len(self.symbols_to_track)} symbols: {', '.join(self.symbols_to_track)}")
            return True
        
        # Otherwise, get from indices
        logger.info("Initializing symbol list from indices...")
        
        all_symbols = set()
        for index in self.config.indices:
            symbols = IndexConstituents.get_constituents(index)
            if symbols:
                all_symbols.update(symbols)
                logger.info(f"  {index}: {len(symbols)} symbols")
        
        if not all_symbols:
            logger.error("No symbols found from indices")
            return False
        
        self.symbols_to_track = sorted(list(all_symbols))
        logger.info(f"Total unique symbols to track: {len(self.symbols_to_track)}")
        
        return True
    
    def fetch_initial_data(self) -> bool:
        """
        Fetch initial option chain data for all symbols via REST API.
        
        Returns:
            True if successful
        """
        logger.info("Fetching initial option chain data...")
        
        # Calculate target expiry
        max_expiry = date.today() + timedelta(days=self.config.max_days_to_expiry)
        logger.info(f"Target expiry: {max_expiry.strftime('%Y-%m-%d')} ({self.config.max_days_to_expiry} days)")
        
        # Get stock data for all symbols
        stock_data = self.data_manager.get_stock_data_batch(self.symbols_to_track)
        
        if not stock_data:
            logger.error("Failed to get stock data")
            return False
        
        logger.info(f"Got stock data for {len(stock_data)} symbols")
        
        # Fetch option chains for each symbol
        success_count = 0
        for symbol, data in stock_data.items():
            try:
                chain_data_list = self.data_manager.fetch_option_chains_for_symbol(
                    symbol=symbol,
                    stock_conid=data['conid'],
                    stock_price=data['price'],
                    stock_ivhv=data['ivhv']
                )
                
                if chain_data_list:
                    for chain_data in chain_data_list:
                        # Save to cache
                        self.data_manager.save_option_chain(chain_data)
                        
                        # Store option conids for WebSocket subscription
                        expiry_str = chain_data['expiry']
                        if symbol not in self.option_conids:
                            self.option_conids[symbol] = {}
                        
                        # Store the full option data, not just conids
                        if expiry_str not in self.option_conids[symbol]:
                            self.option_conids[symbol][expiry_str] = []
                        
                        self.option_conids[symbol][expiry_str].extend(chain_data['options'])
                    
                    total_options = sum(len(chain['options']) for chain in chain_data_list)
                    logger.info(f"✓ {symbol}: {total_options} options across {len(chain_data_list)} expiries")
                    success_count += 1
                else:
                    logger.warning(f"✗ {symbol}: No option data")
                    
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {success_count}/{len(stock_data)} symbols")
        
        return success_count > 0
    
    def subscribe_to_market_data(self) -> bool:
        """
        Subscribe to real-time market data for all option conids.
        
        Returns:
            True if successful
        """
        if not self.ws_client or not self.ws_client.is_connected:
            logger.error("WebSocket not connected")
            return False
        
        logger.info("Subscribing to option market data...")
        
        # Build list of Option objects from our cached data
        from utils.put import Put
        from utils.call import Call
        from datetime import datetime
        
        options_to_subscribe = []
        
        for symbol, expiries in self.option_conids.items():
            for expiry_str, option_data_list in expiries.items():
                expiry_date = datetime.strptime(expiry_str, '%Y%m%d').date()
                
                for opt_data in option_data_list:
                    conid = opt_data.get('conid')
                    if not conid:
                        continue
                    
                    right = opt_data.get('right', 'P')
                    
                    # Create appropriate option object based on right
                    if right == 'P':
                        opt = Put(
                            underlyer=symbol,
                            strike=opt_data['strike'],
                            expiry=expiry_date,
                            quantity=0,  # Not needed for subscription
                            conid=conid
                        )
                    else:  # right == 'C'
                        opt = Call(
                            underlyer=symbol,
                            strike=opt_data['strike'],
                            expiry=expiry_date,
                            quantity=0,  # Not needed for subscription
                            conid=conid
                        )
                    
                    # Update greeks from the fetched data
                    opt.delta = opt_data.get('delta')
                    opt.gamma = opt_data.get('gamma')
                    opt.theta = opt_data.get('theta')
                    opt.vega = opt_data.get('vega')
                    opt.implied_vol = opt_data.get('implied_vol')
                    
                    # Update option book with prices (only settable attributes)
                    if opt_data.get('bid') is not None:
                        opt.option_book.bid = opt_data['bid']
                    if opt_data.get('ask') is not None:
                        opt.option_book.ask = opt_data['ask']
                    # mid is calculated from bid/ask, don't set it directly
                    
                    options_to_subscribe.append(opt)
        
        if not options_to_subscribe:
            logger.warning("No option conids to subscribe to")
            return False
        
        logger.info(f"Subscribing to {len(options_to_subscribe)} option contracts...")
        
        try:
            # Use the same method as naked_put_strat
            subscribed, failed = self.ws_client.subscribe_all_position_market_data(options_to_subscribe)
            
            logger.info(f"✓ Subscribed to {subscribed} option contracts")
            if failed > 0:
                logger.warning(f"Failed to subscribe to {failed} contracts")
            
            return subscribed > 0
            
        except Exception as e:
            logger.error(f"Error subscribing to market data: {e}")
            return False
    
    def _handle_market_data_update(self, data: Dict):
        """
        Handle real-time market data update from WebSocket.
        
        Args:
            data: Market data update
        """
        try:
            conid = data.get('conid')
            if not conid:
                return
            
            # Find which symbol/expiry this conid belongs to
            symbol = None
            expiry = None
            
            for sym, expiries in self.option_conids.items():
                for exp, conids in expiries.items():
                    if conid in conids:
                        symbol = sym
                        expiry = exp
                        break
                if symbol:
                    break
            
            if not symbol or not expiry:
                logger.debug(f"Received update for unknown conid: {conid}")
                return
            
            # Load current cache
            expiry_date = datetime.strptime(expiry, '%Y%m%d').date()
            chain_data = self.data_manager.load_option_chain(symbol, expiry_date)
            
            if not chain_data:
                logger.warning(f"No cached data for {symbol} expiry {expiry}")
                return
            
            # Update the option in the chain
            updated = False
            for opt in chain_data['options']:
                if opt.get('conid') == conid:
                    # Update fields from market data
                    if '31' in data:  # Last price
                        opt['last'] = float(data['31'])
                    if '84' in data:  # Bid
                        opt['bid'] = float(data['84'])
                    if '86' in data:  # Ask
                        opt['ask'] = float(data['86'])
                    
                    # Recalculate mid
                    if opt.get('bid') is not None and opt.get('ask') is not None:
                        opt['mid'] = (opt['bid'] + opt['ask']) / 2
                    
                    updated = True
                    break
            
            if updated:
                # Update timestamp
                chain_data['timestamp'] = datetime.now().isoformat()
                
                # Save back to cache
                with self.market_data_lock:
                    self.data_manager.save_option_chain(chain_data)
                
                logger.debug(f"Updated {symbol} option conid {conid}")
            
        except Exception as e:
            logger.error(f"Error handling market data update: {e}", exc_info=True)
    
    def run(self):
        """
        Main run loop - maintains WebSocket connection and refreshes data periodically.
        """
        logger.info("Starting option data updater...")
        self.running = True
        
        last_refresh_time = time.time()
        
        try:
            while self.running:
                current_time = time.time()
                
                # Refresh data periodically
                if current_time - last_refresh_time >= self.config.update_interval:
                    logger.info("Periodic data refresh...")
                    self.fetch_initial_data()
                    last_refresh_time = current_time
                
                # Print status
                self._print_status()
                
                # Sleep
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("\nStopping updater...")
        finally:
            self.running = False
            if self.ws_client:
                self.ws_client.disconnect()
            logger.info("Updater stopped")
    
    def _print_status(self):
        """Print current status."""
        ws_status = "Connected" if (self.ws_client and self.ws_client.is_connected) else "Disconnected"
        
        total_options = sum(
            len(conids) 
            for expiries in self.option_conids.values() 
            for conids in expiries.values()
        )
        
        logger.info(f"Status: {len(self.symbols_to_track)} symbols, "
                   f"{total_options} options tracked, "
                   f"WebSocket: {ws_status}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Option Data Updater - Continuously fetch and cache option data'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create updater
    updater = OptionDataUpdater(args.config)
    
    # Connect to IBKR
    if not updater.connect():
        logger.error("Failed to connect to IBKR")
        return 1
    
    # Initialize symbols
    if not updater.initialize_symbols():
        logger.error("Failed to initialize symbols")
        return 1
    
    # Fetch initial data
    if not updater.fetch_initial_data():
        logger.error("Failed to fetch initial data")
        return 1
    
    # Subscribe to market data
    if not updater.subscribe_to_market_data():
        logger.error("Failed to subscribe to market data")
        return 1
    
    # Run main loop
    updater.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())