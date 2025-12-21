"""
Utility module for fetching and managing option chain data.
"""
import logging
import time
import configparser
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import fcntl

from utils.index_constituents import IndexConstituents

logger = logging.getLogger(__name__)


@dataclass
class OptionChainConfig:
    """Configuration for option chain data collection."""
    indices: List[str]
    max_days_to_expiry: int
    max_delta: float
    strike_pct_below: float
    num_strikes: int
    data_cache_dir: str
    update_interval: int  # seconds between updates
    test_symbols: Optional[List[str]] = None  # Add this line
    
    @classmethod
    def from_config(cls, config: configparser.ConfigParser):
        """Load configuration from config file."""
        section = config['OPTION_DATA']
        
        indices = [i.strip().upper() for i in section.get('indices').split(',')]
        max_days_to_expiry = section.getint('max_days_to_expiry')
        max_delta = section.getfloat('max_delta')
        strike_pct_below = section.getfloat('strike_pct_below', fallback=5.0)
        num_strikes = section.getint('num_strikes', fallback=10)
        data_cache_dir = section.get('data_cache_dir', fallback='data_cache')
        update_interval = section.getint('update_interval', fallback=60)
        
        # Parse test_symbols (comma-separated)
        test_symbols_str = section.get('test_symbols', fallback=None)
        test_symbols = None
        if test_symbols_str:
            test_symbols = [s.strip().upper() for s in test_symbols_str.split(',')]
        
        return cls(
            indices=indices,
            max_days_to_expiry=max_days_to_expiry,
            max_delta=max_delta,
            strike_pct_below=strike_pct_below,
            num_strikes=num_strikes,
            data_cache_dir=data_cache_dir,
            update_interval=update_interval,
            test_symbols=test_symbols
        )


class OptionDataManager:
    """Manages option chain data fetching and caching."""
    
    def __init__(self, client: 'IBWebAPIClient', config: OptionChainConfig):
        """
        Initialize the option data manager.
        
        Args:
            client: IBWebAPIClient instance
            config: OptionChainConfig instance
        """
        self.client = client
        self.config = config
        
        # Create cache directory if it doesn't exist
        Path(config.data_cache_dir).mkdir(parents=True, exist_ok=True)
    
    def get_cache_filepath(self, symbol: str, expiry_date: date) -> str:
        """Get the cache file path for a symbol and expiry."""
        expiry_str = expiry_date.strftime('%Y%m%d')
        return os.path.join(self.config.data_cache_dir, f"{symbol}_{expiry_str}_options.json")
    
    def get_stock_data_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get stock data (price, conid, IV/HV) for multiple symbols in batch.
        This is copied from put_scanner.py STEP 1-3.
        
        Returns:
            Dict mapping symbol to {price, conid, ivhv}
        """
        logger.info(f"Getting stock data for {len(symbols)} symbols...")
        
        # STEP 1: Search for all symbols and get conids
        symbol_to_conid = {}
        
        for symbol in symbols:
            try:
                contracts = self.client.search_contracts(symbol)
                
                if not contracts:
                    logger.debug(f"  ✗ {symbol}: No contracts found")
                    continue
                
                stock_conid = None
                for contract in contracts:
                    if contract.get('symbol') == symbol:
                        sections = contract.get('sections', [])
                        if any(s.get('secType') == 'STK' for s in sections):
                            stock_conid = int(contract.get('conid'))
                            break
                
                if stock_conid:
                    symbol_to_conid[symbol] = stock_conid
                else:
                    logger.debug(f"  ✗ {symbol}: No stock contract found")
                    
            except Exception as e:
                logger.debug(f"  ✗ {symbol}: Error searching - {e}")
                continue
        
        if not symbol_to_conid:
            logger.warning("No valid contracts found")
            return {}
        
        # STEP 2: Get market data for all stocks
        conids = list(symbol_to_conid.values())
        
        # Wait before requesting market data
        time.sleep(5)
        
        # Get market data for all stocks in one batch call
        snapshots = self.client.get_market_data_snapshot(
            conids,
            fields=['31', '84', '86', '7283', '7088']
        )
        
        if not snapshots:
            logger.warning("No market data returned")
            return {}
        
        # Check if we got pre-flight responses and retry if needed (up to 3 times)
        metadata_keys = {'conid', 'conidEx', '_updated', 'server_id', '6119', '6509'}
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            has_data = any(
                any(key not in metadata_keys for key in snapshot.keys())
                for snapshot in snapshots
            )
            
            if has_data:
                logger.debug("Received real market data, proceeding...")
                break  # Got real data, exit retry loop
            
            # All snapshots are pre-flight, need to retry
            retry_count += 1
            wait_time = 2 * retry_count  # Increasing wait time: 2s, 4s, 6s
            logger.debug(f"Pre-flight response received, waiting {wait_time}s and retrying ({retry_count}/{max_retries})...")
            time.sleep(wait_time)
            
            snapshots = self.client.get_market_data_snapshot(
                conids,
                fields=['31', '84', '86', '7283', '7088']
            )
            
            if not snapshots:
                logger.warning("No market data returned on retry")
                return {}
        
        # Check if we still don't have data after all retries
        if not has_data:
            logger.warning(f"Still no real data after {max_retries} retries, proceeding with pre-flight data")
        
        # STEP 3: Parse stock data
        conid_to_symbol = {v: k for k, v in symbol_to_conid.items()}
        stock_data = {}
        
        def to_float(val):
            if val is None:
                return None
            try:
                if isinstance(val, str):
                    val = val.lstrip('CH').replace(',', '')
                return float(val)
            except (ValueError, TypeError):
                return None

        def parse_pct(val):
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                val = val.replace('%', '').strip()
                if val in ('N/A', 'n/a', '', '-'):
                    return None
                try:
                    return float(val)
                except ValueError:
                    return None
            return None
        
        for snapshot in snapshots:
            conid = snapshot.get('conid')
            if conid not in conid_to_symbol:
                continue
            
            symbol = conid_to_symbol[conid]
            
            # Extract stock price
            stock_price = to_float(snapshot.get('31'))
            if stock_price is None:
                stock_price = to_float(snapshot.get('84'))
            
            if not stock_price:
                logger.debug(f"  ✗ {symbol}: No price data")
                continue
            
            # Extract IV/HV
            iv_str = snapshot.get('7283')
            hv_str = snapshot.get('7088')
            
            stock_ivhv = None
            if iv_str and hv_str:
                iv = parse_pct(iv_str)
                hv = parse_pct(hv_str)
                
                if iv is not None and hv is not None and hv != 0:
                    stock_ivhv = (iv / hv) * 100
            
            stock_data[symbol] = {
                'price': stock_price,
                'conid': conid,
                'ivhv': stock_ivhv
            }
        
        return stock_data

    def fetch_option_chains_for_symbol(
        self, 
        symbol: str, 
        stock_conid: int, 
        stock_price: float,
        stock_ivhv: Optional[float],
        right: str = 'P'  # Add this parameter
    ) -> List[Dict[str, Any]]:
        """
        Fetch option chains for a single symbol across all valid expiries.
        
        Args:
            symbol: Stock symbol
            stock_conid: Stock contract ID
            stock_price: Current stock price
            stock_ivhv: Stock IV/HV ratio
            right: 'P' for puts, 'C' for calls
        """
        logger.debug(f"Fetching option chains for {symbol}")
        
        # Calculate max strike (below current price)
        max_strike = stock_price * (1 - self.config.strike_pct_below / 100)
        
        # Get max expiry
        today = date.today()
        max_expiry = today + timedelta(days=self.config.max_days_to_expiry)
        
        try:
            logger.debug(f"{symbol}: Getting options with strike <= ${max_strike:.2f}, expiry <= {max_expiry}")
            
            # Get all options meeting our criteria
            options_list = self.client.get_options_by_expiry_and_strike(
                stock_conid=stock_conid,
                max_expiry_date=max_expiry,
                max_strike=max_strike,
                right=right,  # Pass it through
                exchange='SMART',
                num_strikes=self.config.num_strikes
            )
            
            if not options_list:
                logger.debug(f"{symbol}: No options found")
                return []
            
            logger.debug(f"{symbol}: Found {len(options_list)} options")
            
            # Get market data for ALL options in ONE call
            conids = [opt['conid'] for opt in options_list]
            
            time.sleep(0.5)
            
            snapshots = self.client.get_market_data_snapshot(
                conids,
                fields=['31', '84', '86', '87', '7308', '7309', '7310', '7311', '7633']
            )
            
            if not snapshots:
                logger.debug(f"{symbol}: No market data returned")
                return []
            
            # Check if we got pre-flight responses and retry if needed (up to 3 times)
            metadata_keys = {'conid', 'conidEx', '_updated', 'server_id', '6119', '6509'}
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                has_data = any(
                    any(key not in metadata_keys for key in snapshot.keys())
                    for snapshot in snapshots
                )
                
                if has_data:
                    break  # Got real data, exit retry loop
                
                # All snapshots are pre-flight, need to retry
                retry_count += 1
                
                if retry_count >= max_retries:
                    logger.warning(f"{symbol}: Still no data after {max_retries} retries")
                    break
                
                wait_time = 2 * retry_count  # Increasing wait time
                logger.debug(f"{symbol}: Pre-flight response, waiting {wait_time}s and retrying ({retry_count}/{max_retries})...")
                time.sleep(wait_time)
                
                snapshots = self.client.get_market_data_snapshot(
                    conids,
                    fields=['31', '84', '86', '87', '7308', '7309', '7310', '7311', '7633']
                )
                
                if not snapshots:
                    logger.debug(f"{symbol}: No market data returned on retry")
                    return []
            
            # Create a map of conid -> snapshot for quick lookup
            snapshot_map = {s.get('conid'): s for s in snapshots}
            
            # Helper function
            def to_float(val):
                if val is None:
                    return None
                try:
                    if isinstance(val, str):
                        val = val.lstrip('CH').replace(',', '')
                    return float(val)
                except (ValueError, TypeError):
                    return None
            
            # Build options with market data and group by expiry
            options_by_expiry = {}
            
            for opt_info in options_list:
                conid = opt_info['conid']
                strike = opt_info['strike']
                expiry_str = opt_info['expiry']
                
                # Find snapshot for this conid
                snapshot = snapshot_map.get(conid)
                if not snapshot:
                    continue
                
                delta = to_float(snapshot.get('7308'))
                if delta is None:
                    continue
                
                bid = to_float(snapshot.get('84'))
                ask = to_float(snapshot.get('86'))
                
                # Calculate mid price
                mid_price = None
                if bid is not None and ask is not None:
                    mid_price = (bid + ask) / 2
                
                option_data = {
                    'strike': strike,
                    'expiry': expiry_str,
                    'right': 'P',  # From the fetch parameter
                    'delta': delta,
                    'bid': bid,
                    'ask': ask,
                    'mid': mid_price,
                    'implied_vol': to_float(snapshot.get('7633')),
                    'gamma': to_float(snapshot.get('7309')),
                    'theta': to_float(snapshot.get('7311')),
                    'vega': to_float(snapshot.get('7310')),
                    'volume': snapshot.get('87'),
                    'conid': conid,
                    'days_to_expiry': opt_info['days_to_expiry']
                }
                
                # Group by expiry
                if expiry_str not in options_by_expiry:
                    options_by_expiry[expiry_str] = []
                options_by_expiry[expiry_str].append(option_data)
            
            # Build chain data for each expiry
            all_chains = []
            for expiry_str, options in options_by_expiry.items():
                if options:
                    chain_data = {
                        'symbol': symbol,
                        'stock_price': stock_price,
                        'stock_conid': stock_conid,
                        'stock_ivhv': stock_ivhv,
                        'expiry': expiry_str,
                        'timestamp': datetime.now().isoformat(),
                        'options': options
                    }
                    all_chains.append(chain_data)
                    logger.debug(f"{symbol}: Chain for {expiry_str} has {len(options)} options")
            
            logger.debug(f"{symbol}: Built {len(all_chains)} chains with total {sum(len(c['options']) for c in all_chains)} options")
            
            return all_chains
            
        except Exception as e:
            logger.error(f"Error fetching option chains for {symbol}: {e}", exc_info=True)
            return []
    
    def save_option_chain(self, chain_data: Dict[str, Any]):
        """
        Save option chain data to cache file with file locking.
        
        Args:
            chain_data: Option chain data dictionary
        """
        if not chain_data:
            return
        
        symbol = chain_data['symbol']
        expiry_date = datetime.strptime(chain_data['expiry'], '%Y%m%d').date()
        filepath = self.get_cache_filepath(symbol, expiry_date)
        
        # Use file locking to prevent concurrent writes
        with open(filepath, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(chain_data, f, indent=2)
                logger.debug(f"Saved option chain for {symbol} to {filepath}")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def load_option_chain(self, symbol: str, expiry_date: date) -> Optional[Dict[str, Any]]:
        """
        Load option chain data from cache file with file locking.
        
        Args:
            symbol: Stock symbol
            expiry_date: Expiry date
            
        Returns:
            Option chain data dict or None if not found
        """
        filepath = self.get_cache_filepath(symbol, expiry_date)
        
        if not os.path.exists(filepath):
            return None
        
        # Use file locking to prevent reading while writing
        with open(filepath, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
                return data
            except Exception as e:
                logger.error(f"Error loading option chain for {symbol}: {e}")
                return None
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def get_cache_age(self, symbol: str, expiry_date: date) -> Optional[float]:
        """
        Get the age of cached data in seconds.
        
        Args:
            symbol: Stock symbol
            expiry_date: Expiry date
            
        Returns:
            Age in seconds or None if cache doesn't exist
        """
        data = self.load_option_chain(symbol, expiry_date)
        if not data:
            return None
        
        timestamp = datetime.fromisoformat(data['timestamp'])
        age = (datetime.now() - timestamp).total_seconds()
        return age
    
    def update_all_option_chains(self):
        """
        Fetch and update option chains for all configured indices and expiries.
        """
        logger.info("Starting option chain update...")
        
        # Get all symbols from configured indices
        all_symbols = set()
        for index in self.config.indices:
            symbols = IndexConstituents.get_constituents(index)
            if symbols:
                all_symbols.update(symbols)
                logger.info(f"Added {len(symbols)} symbols from {index}")
        
        if not all_symbols:
            logger.warning("No symbols to update")
            return
        
        logger.info(f"Total unique symbols to update: {len(all_symbols)}")
        
        # Get stock data for all symbols
        stock_data = self.get_stock_data_batch(list(all_symbols))
        
        logger.info(f"Got stock data for {len(stock_data)} symbols")
        
        # Fetch option chains for each symbol (multiple expiries per symbol)
        for symbol, data in stock_data.items():
            try:
                chain_data_list = self.fetch_option_chains_for_symbol(
                    symbol=symbol,
                    stock_conid=data['conid'],
                    stock_price=data['price'],
                    stock_ivhv=data['ivhv']
                )
                
                if chain_data_list:
                    total_options = 0
                    for chain_data in chain_data_list:
                        self.save_option_chain(chain_data)
                        total_options += len(chain_data['options'])
                    
                    logger.info(f"✓ {symbol}: Updated {total_options} options across {len(chain_data_list)} expiries")
                else:
                    logger.warning(f"✗ {symbol}: No option data")
                    
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
                continue
        
        logger.info("Option chain update complete")