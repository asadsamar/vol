import requests
import json
from typing import Dict, Optional, List, Any, Union
import logging
import configparser
import os
import time
from datetime import date, datetime, timedelta
from threading import Lock
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter to respect IBKR API pacing limitations.
    
    Rate limits are configured per endpoint category.
    """
    
    def __init__(self, rate_limits: Dict[str, float]):
        """
        Initialize rate limiter with specified limits.
        
        Args:
            rate_limits: Dictionary mapping endpoint categories to requests per second
        """
        self.locks = defaultdict(Lock)
        self.last_call_time = defaultdict(float)
        self.rate_limits = rate_limits
        
        logger.debug("Rate limiter initialized")
    
    def _get_endpoint_category(self, url: str) -> str:
        """
        Determine the rate limit category based on URL.
        
        Args:
            url: The API endpoint URL
            
        Returns:
            Category name for rate limiting
        """
        if '/portfolio/' in url:
            return 'portfolio'
        elif '/iserver/account' in url:
            return 'account'
        elif '/iserver/auth' in url:
            return 'auth'
        elif '/iserver/marketdata' in url or '/md/' in url:
            return 'market_data'
        elif '/tickle' in url:
            return 'tickle'
        else:
            return 'default'
    
    def wait_if_needed(self, url: str) -> None:
        """
        Wait if necessary to respect rate limits.
        
        Args:
            url: The API endpoint URL
        """
        category = self._get_endpoint_category(url)
        
        # Use default if category not found
        if category not in self.rate_limits:
            category = 'default'
        
        with self.locks[category]:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time[category]
            min_interval = 1.0 / self.rate_limits[category]
            
            if time_since_last_call < min_interval:
                sleep_time = min_interval - time_since_last_call
                logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s for {category}")
                time.sleep(sleep_time)
            
            self.last_call_time[category] = time.time()


class IBWebAPIClient:
    """
    Interactive Brokers Client Portal Web API Client with rate limiting.
    
    Prerequisites:
    1. Download and run the Client Portal Gateway from:
       https://www.interactivebrokers.com/en/trading/cpgw.php
    2. Start the gateway (usually runs on https://localhost:5000)
    """
    
    def __init__(self, config_file: str = "ibkr.conf"):
        """
        Initialize the IB Web API Client.
        
        Args:
            config_file: Path to configuration file (default: ibkr.conf)
        """
        self.config = self._load_config(config_file)
        self.base_url = self.config.get('api', 'base_url')
        self.session = requests.Session()
        # Disable SSL verification for localhost (gateway uses self-signed cert)
        self.session.verify = False
        # Suppress SSL warnings
        requests.packages.urllib3.disable_warnings()
        
        # Set logging level from config
        log_level = self.config.get('logging', 'level', fallback='INFO')
        logger.setLevel(getattr(logging, log_level))
        
        # Account setup
        self.account_id = None
        self.available_accounts = []
        
        # Initialize rate limiter with limits from config
        rate_limits = self._load_rate_limits()
        self.rate_limiter = RateLimiter(rate_limits)
        
        # Pre-flight setup: Call required endpoints and wait
        logger.debug("Performing pre-flight API setup...")
        self._preflight_setup()
    
    def _preflight_setup(self):
        """
        Perform pre-flight setup by calling required endpoints.
        This ensures the API is ready for market data requests.
        
        Per IBKR API docs:
        - /iserver/accounts must be called before /iserver/marketdata/snapshot
        - A small delay helps ensure data streams are ready
        """
        try:
            # Step 1: Call /iserver/accounts (required before market data)
            logger.debug("Fetching accounts...")
            accounts = self._fetch_accounts()
            
            if accounts:
                self.available_accounts = accounts
                
                # Auto-setup account
                account_ids = [acc['accountId'] for acc in accounts]
                configured_account = self.config.get('account', 'account_id', fallback='').strip()
                
                if configured_account and configured_account in account_ids:
                    self.account_id = configured_account
                elif account_ids:
                    self.account_id = account_ids[0]
            
            # Step 2: Make a test market data subscription (helps initialize data streams)
            logger.debug("Initializing market data streams...")
            test_contracts = self.search_contracts('SPY')
            if test_contracts:
                spy_conid = test_contracts[0].get('conid')
                if spy_conid:
                    # Make a test market data call (pre-flight request)
                    self.get_market_data_snapshot(spy_conid, fields=['31'])
            
            # Step 3: Wait for API to stabilize
            logger.debug("Waiting for API to stabilize...")
            time.sleep(5)
            
            logger.debug("Pre-flight setup complete")
            
        except Exception as e:
            logger.error(f"Error during pre-flight setup: {e}")
            logger.warning("Continuing anyway, but market data may not work immediately")
    
    def _load_rate_limits(self) -> Dict[str, float]:
        """
        Load rate limits from configuration.
        
        Returns:
            Dictionary mapping endpoint categories to requests per second
        """
        rate_limits = {}
        
        if self.config.has_section('rate_limits'):
            for key in self.config.options('rate_limits'):
                try:
                    rate_limits[key] = float(self.config.get('rate_limits', key))
                except ValueError:
                    logger.warning(f"Invalid rate limit value for {key}, using default")
        
        # Set defaults for any missing values
        defaults = {
            'default': 1.0,
            'portfolio': 1.0,
            'account': 1.0,
            'auth': 1.0,
            'market_data': 5.0,
            'tickle': 0.016,
        }
        
        for key, value in defaults.items():
            if key not in rate_limits:
                rate_limits[key] = value
        
        return rate_limits
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make an HTTP request with rate limiting.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL for the request
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response object
        """
        # Apply rate limiting
        self.rate_limiter.wait_if_needed(url)
        
        # Make the request
        if method.upper() == 'GET':
            return self.session.get(url, **kwargs)
        elif method.upper() == 'POST':
            return self.session.post(url, **kwargs)
        elif method.upper() == 'DELETE':
            return self.session.delete(url, **kwargs)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    
    def _load_config(self, config_file: str) -> configparser.ConfigParser:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            ConfigParser object with loaded configuration
        """
        config = configparser.ConfigParser()
        
        # Try to find config file in current directory or same directory as script
        config_paths = [
            config_file,
            os.path.join(os.path.dirname(__file__), config_file)
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                config.read(path)
                logger.info(f"Loaded configuration from {path}")
                return config
        
        logger.warning(f"Configuration file not found, using defaults")
        # Set defaults if config file not found
        config.add_section('api')
        config.set('api', 'base_url', 'https://localhost:5000/v1/api')
        config.set('api', 'host', 'localhost')
        config.set('api', 'port', '5000')
        config.add_section('logging')
        config.set('logging', 'level', 'INFO')
        config.add_section('session')
        config.set('session', 'tickle_interval', '60')
        config.add_section('account')
        config.set('account', 'account_id', '')
        config.add_section('rate_limits')
        config.set('rate_limits', 'default', '1.0')
        config.set('rate_limits', 'portfolio', '1.0')
        config.set('rate_limits', 'account', '1.0')
        config.set('rate_limits', 'auth', '1.0')
        config.set('rate_limits', 'market_data', '5.0')
        config.set('rate_limits', 'tickle', '0.016')
        
        return config
        
    def authenticate(self) -> bool:
        """
        Check authentication status and initiate login if needed.
        The Client Portal Gateway handles the actual authentication via web browser.
        """
        try:
            # Check authentication status
            response = self._make_request('GET', f"{self.base_url}/iserver/auth/status")
            response.raise_for_status()
            
            auth_status = response.json()
            
            if auth_status.get('authenticated', False):
                logger.debug("Authenticated to IBKR")
                return True
            else:
                logger.warning("Not authenticated. Please authenticate via the Client Portal Gateway web interface.")
                host = self.config.get('api', 'host')
                port = self.config.get('api', 'port')
                logger.info(f"Open https://{host}:{port} in your browser to authenticate")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication check failed: {e}")
            return False
    
    def setup_account(self) -> bool:
        """
        Setup the account to use based on configuration.
        
        Note: This is now mostly handled in _preflight_setup() during __init__.
        This method is kept for backwards compatibility.
        
        Returns:
            True if account setup successful, False otherwise
        """
        if self.account_id:
            logger.debug(f"Account already setup: {self.account_id}")
            return True
        
        # If account wasn't set up during init, try again
        logger.debug("Account not set up, fetching accounts...")
        accounts_response = self._fetch_accounts()
        
        if not accounts_response:
            logger.error("No accounts found")
            return False
        
        # Extract account IDs and store full account info
        self.available_accounts = accounts_response
        account_ids = [acc['accountId'] for acc in accounts_response]
        
        logger.debug(f"Found {len(account_ids)} account(s)")
        
        # Get configured account ID
        configured_account = self.config.get('account', 'account_id', fallback='').strip()
        
        if configured_account:
            if configured_account in account_ids:
                self.account_id = configured_account
                logger.debug(f"Using configured account: {configured_account}")
                return True
            else:
                logger.error(f"Configured account '{configured_account}' not found in available accounts")
                return False
        else:
            # Use first account if none configured
            self.account_id = account_ids[0]
            logger.debug(f"No account configured, using first available: {self.account_id}")
            return True
    
    def _fetch_accounts(self) -> Optional[List[Dict]]:
        """
        Internal method to fetch available accounts.
        
        Returns:
            List of account dictionaries
        """
        try:
            response = self._make_request('GET', f"{self.base_url}/portfolio/accounts")
            response.raise_for_status()
            accounts = response.json()
            return accounts
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get accounts: {e}")
            return None
    
    def tickle(self) -> bool:
        """
        Keep the session alive by calling the tickle endpoint.
        Should be called periodically to maintain connection.
        """
        try:
            response = self._make_request('POST', f"{self.base_url}/tickle")
            response.raise_for_status()
            result = response.json()
            logger.info(f"Session tickle: {result}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Tickle failed: {e}")
            return False
    
    def get_account_balance(self, account_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get account balance and summary information.
        
        Args:
            account_id: The account ID to query (uses configured account if not provided)
            
        Returns:
            Dictionary containing account balance information
        """
        if account_id is None:
            account_id = self.account_id
        
        if not account_id:
            logger.error("No account ID provided and no account configured")
            return None
        
        try:
            # Get account summary
            response = self._make_request(
                'GET',
                f"{self.base_url}/portfolio/{account_id}/summary"
            )
            response.raise_for_status()
            balance = response.json()
            logger.info(f"Account balance retrieved successfully for {account_id}")
            return balance
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get account balance: {e}")
            return None
    
    def get_account_ledger(self, account_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get detailed account ledger information including balances.
        
        Args:
            account_id: The account ID to query (uses configured account if not provided)
            
        Returns:
            Dictionary containing detailed ledger information
        """
        if account_id is None:
            account_id = self.account_id
        
        if not account_id:
            logger.error("No account ID provided and no account configured")
            return None
        
        try:
            response = self._make_request(
                'GET',
                f"{self.base_url}/portfolio/{account_id}/ledger"
            )
            response.raise_for_status()
            ledger = response.json()
            logger.info(f"Account ledger retrieved successfully for {account_id}")
            return ledger
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get account ledger: {e}")
            return None
    
    def get_account_cash_summary(self, account_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get a summary of account cash metrics including settled cash, buying power, and net liquidation value.
        
        Args:
            account_id: The account ID to query (uses configured account if not provided)
            
        Returns:
            Dictionary with keys: settled_cash, buying_power, net_liq_value, currency
            Returns None if error
        """
        if account_id is None:
            account_id = self.account_id
        
        if not account_id:
            logger.error("No account ID provided and no account configured")
            return None
        
        # Get ledger for cash balance
        ledger = self.get_account_ledger(account_id)
        if not ledger:
            return None
        
        # Get account summary for buying power
        summary = self.get_account_balance(account_id)
        if not summary:
            return None
        
        # Extract values from ledger (USD section)
        settled_cash = 0.0
        net_liq_value = 0.0
        currency = 'USD'
        
        if 'USD' in ledger:
            usd_data = ledger['USD']
            settled_cash = float(usd_data.get('settledcash', 0))
            net_liq_value = float(usd_data.get('netliquidationvalue', 0))
            currency = usd_data.get('currency', 'USD')
        
        # Extract buying power from summary
        # Summary format: {"buyingpower": {"amount": 1219857.0, ...}, ...}
        buying_power = 0.0
        
        if isinstance(summary, dict):
            # Look for buyingpower key
            if 'buyingpower' in summary:
                bp_data = summary['buyingpower']
                if isinstance(bp_data, dict) and 'amount' in bp_data:
                    buying_power = float(bp_data['amount'])
            
            # Alternative: look for availablefunds
            if buying_power == 0.0 and 'availablefunds' in summary:
                af_data = summary['availablefunds']
                if isinstance(af_data, dict) and 'amount' in af_data:
                    buying_power = float(af_data['amount'])
        
        result = {
            'settled_cash': settled_cash,
            'buying_power': buying_power,
            'net_liq_value': net_liq_value,
            'currency': currency
        }
        
        logger.info(f"Account cash summary for {account_id}:")
        logger.info(f"  Settled Cash: ${settled_cash:,.2f} {currency}")
        logger.info(f"  Buying Power: ${buying_power:,.2f} {currency}")
        logger.info(f"  Net Liquidation Value: ${net_liq_value:,.2f} {currency}")
        
        return result
    
    def get_positions(self, account_id: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Get all current positions for the account.
        
        Args:
            account_id: The account ID to query (uses configured account if not provided)
            
        Returns:
            List of dictionaries containing position information
        """
        if account_id is None:
            account_id = self.account_id
        
        if not account_id:
            logger.error("No account ID provided and no account configured")
            return None
        
        try:
            response = self._make_request(
                'GET',
                f"{self.base_url}/portfolio/{account_id}/positions/0"
            )
            response.raise_for_status()
            positions = response.json()
            logger.info(f"Positions retrieved successfully for {account_id}")
            logger.info(f"Found {len(positions)} position(s)")
            return positions
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get positions: {e}")
            return None
    
    def get_position_summary(self, account_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get a summary of positions with key information formatted nicely.
        
        Args:
            account_id: The account ID to query (uses configured account if not provided)
            
        Returns:
            Dictionary with summarized position information
        """
        positions = self.get_positions(account_id)
        
        if not positions:
            return None
        
        summary = {
            'total_positions': len(positions),
            'positions': []
        }
        
        for pos in positions:
            position_info = {
                'symbol': pos.get('contractDesc', 'N/A'),
                'ticker': pos.get('ticker', 'N/A'),
                'position': pos.get('position', 0),
                'market_price': pos.get('mktPrice', 0),
                'market_value': pos.get('mktValue', 0),
                'average_cost': pos.get('avgCost', 0),
                'unrealized_pnl': pos.get('unrealizedPnl', 0),
                'realized_pnl': pos.get('realizedPnl', 0),
                'currency': pos.get('currency', 'N/A'),
                'asset_class': pos.get('assetClass', 'N/A')
            }
            summary['positions'].append(position_info)
        
        return summary
    
    def get_stock_price(
        self,
        symbol: str,
        exchange: str = 'SMART'
    ) -> Optional[float]:
        """
        Get current stock price for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            exchange: Exchange (default: SMART for best execution)
            
        Returns:
            Current stock price or None if not available
        """
        try:
            logger.info(f"Getting price for {symbol}...")
            
            # Ensure account is set up
            if not self.account_id:
                logger.debug("Setting up account before market data request...")
                if not self.setup_account():
                    logger.warning("Failed to setup account")
                    return None
            
            # Search for the contract (required pre-flight for market data)
            contracts = self.search_contracts(symbol)
            if not contracts:
                logger.warning(f"No contracts found for {symbol}")
                return None
            
            logger.info(f"Found {len(contracts)} contracts for {symbol}")
            
            # Find the primary stock contract
            stock_contract = None
            for contract in contracts:
                sections = contract.get('sections', [])
                has_stock = any(section.get('secType') == 'STK' for section in sections)
                
                if has_stock and contract.get('symbol') == symbol:
                    description = contract.get('description', '')
                    # Prefer NYSE or NASDAQ
                    if 'NYSE' in description or 'NASDAQ' in description:
                        stock_contract = contract
                        logger.info(f"Found primary stock contract: {description}")
                        break
                    elif not stock_contract:
                        stock_contract = contract
            
            if not stock_contract:
                logger.warning(f"No stock contract found for {symbol}")
                return None
            
            conid = stock_contract.get('conid')
            if not conid:
                logger.warning(f"No conid found for {symbol}")
                return None
            
            logger.info(f"Getting market data for {symbol} (conid: {conid}, exchange: {exchange})")
            
            # Small delay after contract search (helps with pre-flight)
            time.sleep(0.5)
            
            # Get market data - may need retry for pre-flight
            snapshot = None
            for attempt in range(3):
                snapshot = self.get_market_data_snapshot(
                    conid, 
                    fields=['31', '84', '86', '87'],
                    ensure_preflight=False  # Already ensured above
                )
                
                logger.debug(f"Attempt {attempt + 1}: snapshot type = {type(snapshot)}, value = {snapshot}")
                
                if snapshot:
                    # Check the type - if it's a list, something went wrong
                    if isinstance(snapshot, list):
                        logger.error(f"ERROR: get_market_data_snapshot returned a list when single conid was passed!")
                        logger.error(f"  conid passed: {conid} (type: {type(conid)})")
                        logger.error(f"  snapshot received: {snapshot}")
                        # Take first element if it's a list
                        if len(snapshot) > 0:
                            snapshot = snapshot[0]
                        else:
                            snapshot = None
                            time.sleep(1.0)
                            continue
                    
                    # Check if we have actual price data (not just metadata)
                    metadata_keys = {'conid', 'conidEx', '_updated', 'server_id', '6119', '6509'}
                    has_price_data = any(key not in metadata_keys for key in snapshot.keys())
                    
                    if has_price_data:
                        break
                    else:
                        logger.debug(f"Pre-flight response, retrying... (attempt {attempt + 1}/3)")
                        time.sleep(1.0)
                else:
                    logger.debug(f"No snapshot, retrying... (attempt {attempt + 1}/3)")
                    time.sleep(1.0)
            
            if not snapshot:
                logger.warning(f"No market data snapshot for {symbol} after retries")
                return None
            
            # Try to extract price from various fields
            # Priority: last price (31) > bid (84)
            last_price = snapshot.get('31')
            bid = snapshot.get('84')
            
            def to_float(val):
                if val is None:
                    return None
                try:
                    if isinstance(val, str):
                        # Remove 'C' prefix and commas
                        val = val.lstrip('CH').replace(',', '')
                    return float(val)
                except (ValueError, TypeError):
                    return None
            
            price = to_float(last_price)
            if price is None:
                price = to_float(bid)
            
            if price is not None:
                logger.info(f"Got price for {symbol}: ${price:.2f} (last/close)")
                return price
            else:
                logger.warning(f"No valid price data for {symbol}")
                logger.warning(f"Snapshot contained: {snapshot}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}", exc_info=True)
            return None
    
    def search_contracts(self, symbol: str) -> Optional[List[Dict]]:
        """
        Search for contracts by symbol.
        
        Args:
            symbol: The symbol to search for
            
        Returns:
            List of matching contracts
        """
        try:
            params = {'symbol': symbol}
            response = self._make_request(
                'GET',
                f"{self.base_url}/iserver/secdef/search",
                params=params
            )
            response.raise_for_status()
            contracts = response.json()
            
            if not contracts:
                logger.debug(f"No contracts found for symbol: {symbol}")
                return []
            
            logger.debug(f"Found {len(contracts)} contract(s) for {symbol}")
            return contracts
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search contracts for {symbol}: {e}")
            return None
    
    def get_market_data_snapshot(
        self,
        conid: Union[int, List[int]],
        fields: Optional[List[str]] = None,
        ensure_preflight: bool = True
    ) -> Optional[Union[Dict, List[Dict]]]:
        """
        Get market data snapshot for one or more contracts.
        
        Note: Per IBKR API requirements:
        - /iserver/accounts must be called prior to /iserver/marketdata/snapshot
        - For derivative contracts, /iserver/secdef/search must be called first
        - These are per-contract requirements
        
        Args:
            conid: Contract ID or list of contract IDs
            fields: List of field IDs to request (e.g., ['84', '86'] for bid/ask)
            ensure_preflight: If True, ensure account is setup before request
            
        Returns:
            Dictionary of field values (single conid) or list of dicts (multiple conids)
            None if failed
        """
        # Ensure accounts have been fetched (pre-flight requirement)
        if ensure_preflight and not self.account_id:
            logger.debug("Account not set up, fetching accounts first...")
            if not self.setup_account():
                logger.warning("Failed to setup account before market data request")
                return None
        
        url = f"{self.base_url}/iserver/marketdata/snapshot"
        
        # Handle single conid or list - convert string to int if needed
        if isinstance(conid, str):
            conid = int(conid)
        
        is_single = isinstance(conid, int)
        conids = [conid] if is_single else list(conid)
        
        logger.debug(f"get_market_data_snapshot called with conid={conid} (type={type(conid)}), is_single={is_single}")
        
        # Build fields parameter
        if fields:
            fields_str = ','.join(str(f) for f in fields)
        else:
            # Default fields if none specified
            fields_str = '31,84,86,87'  # Last, Bid, Ask, Volume
        
        params = {
            'conids': ','.join(str(c) for c in conids),
            'fields': fields_str
        }
        
        logger.debug(f"Request params: {params}")
        
        try:
            response = self._make_request('GET', url, params=params)
            
            if response.status_code != 200:
                logger.warning(f"Market data request failed: {response.status_code}")
                return None
            
            data = response.json()
            
            if not data or not isinstance(data, list) or len(data) == 0:
                logger.debug(f"No data in market snapshot response")
                return None
            
            # Check if this is just a pre-flight response (only metadata)
            metadata_keys = {'conid', 'conidEx', '_updated', 'server_id', '6119', '6509'}
            
            for snapshot in data:
                has_data = any(key not in metadata_keys for key in snapshot.keys())
                if not has_data:
                    logger.debug(f"Received pre-flight response for conid {snapshot.get('conid')}, data may populate on next call")
            
            # Return single dict if single conid was requested, otherwise return list
            logger.debug(f"Returning: is_single={is_single}, data type={type(data)}, len={len(data)}")
            if is_single:
                logger.debug(f"Returning single dict: {type(data[0])}")
                return data[0]
            else:
                logger.debug(f"Returning list of dicts")
                return data
                
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def get_option_data(
        self,
        stock_conid: int,
        strike: float,
        expiry_date: date,
        right: str = 'P',
        exchange: str = 'SMART'
    ) -> Optional[Dict]:
        """
        Get option market data for a specific strike and expiry.
        
        This method handles all IBKR-specific pre-flight requirements:
        1. Calls /iserver/secdef/info to get option contract
        2. Calls /iserver/secdef/search for market data subscription (pre-flight)
        3. Waits for subscription to initialize
        4. Fetches market data snapshot with retry logic
        
        Args:
            stock_conid: Stock contract ID
            strike: Strike price
            expiry_date: Expiration date
            right: 'P' for put, 'C' for call (default: 'P')
            exchange: Exchange (default: 'SMART')
            
        Returns:
            Dictionary with option data:
            {
                'strike': float,
                'bid': float,
                'ask': float,
                'last': float,
                'delta': float,
                'gamma': float,
                'vega': float,
                'theta': float,
                'implied_vol': float,
                'volume': float
            }
        """
        try:
            # Step 1: Get the specific option contract
            month_format = expiry_date.strftime('%Y%m')
            option_params = {
                'conid': stock_conid,
                'sectype': 'OPT',
                'month': month_format,
                'exchange': exchange,
                'strike': str(strike),
                'right': right
            }
            
            secdef_url = f"{self.base_url}/iserver/secdef/info"
            secdef_response = self._make_request('GET', secdef_url, params=option_params)
            
            if secdef_response.status_code != 200:
                logger.debug(f"Failed to get option contract: status {secdef_response.status_code}")
                return None
            
            option_data = secdef_response.json()
            
            # Find contract with matching expiry
            option_conid = None
            target_expiry_str = expiry_date.strftime('%Y%m%d');
            
            if isinstance(option_data, list):
                for contract in option_data:
                    contract_expiry = contract.get('maturityDate')
                    if contract_expiry == target_expiry_str:
                        option_conid = contract.get('conid')
                        break
                
                if not option_conid and len(option_data) > 0:
                    return None
                    
            elif isinstance(option_data, dict):
                contract_expiry = option_data.get('maturityDate')
                if contract_expiry == target_expiry_str:
                    option_conid = option_data.get('conid')
                else:
                    return None
            
            if not option_conid:
                return None
            
            logger.debug(f"Found option conid {option_conid} for ${strike:.0f}{right}, calling secdef for pre-flight...")
            
            # Step 2: Call secdef/search for this specific option (required pre-flight for derivatives)
            # This subscribes to market data for this contract
            secdef_search_url = f"{self.base_url}/iserver/secdef/search"
            search_params = {'symbol': str(option_conid)}
            search_response = self._make_request('GET', secdef_search_url, params=search_params)
            
            if search_response.status_code == 200:
                logger.debug(f"Pre-flight secdef/search complete for option conid {option_conid}")
            else:
                logger.debug(f"Pre-flight secdef/search failed for option conid {option_conid}: {search_response.status_code}")
            
            # Small delay to allow subscription to initialize
            time.sleep(0.3)
            
            # Step 3: Get market data for the option (should now have data subscribed)
            option_snapshot = self.get_market_data_snapshot(
                option_conid,
                fields=[
                    '31',    # Last price
                    '84',    # Bid
                    '86',    # Ask
                    '87',    # Volume
                    '7308',  # Delta
                    '7309',  # Gamma
                    '7310',  # Vega
                    '7311',  # Theta
                    '7633',  # Implied Volatility % for this strike
                ],
                ensure_preflight=False  # We already did pre-flight
            )
            
            if not option_snapshot:
                logger.debug(f"No market data snapshot received for ${strike:.0f}{right}")
                return None
            
            # Check if we got actual data or just metadata
            metadata_keys = {'conid', 'conidEx', '_updated', 'server_id', '6119', '6509'}
            has_data = any(key not in metadata_keys for key in option_snapshot.keys())
            
            if not has_data:
                logger.debug(f"Only metadata received for ${strike:.0f}{right}, retrying once...")
                time.sleep(1.0)
                option_snapshot = self.get_market_data_snapshot(
                    option_conid,
                    fields=['31', '84', '86', '87', '7308', '7309', '7310', '7311', '7633'],
                    ensure_preflight=False
                )
                
                if not option_snapshot:
                    return None
                
                has_data = any(key not in metadata_keys for key in option_snapshot.keys())
                if not has_data:
                    logger.debug(f"Still no data for ${strike:.0f}{right} after retry")
                    return None
            
            logger.debug(f"Option data for ${strike:.0f}{right}: {option_snapshot}")
            
            # Parse option data
            def to_float(val):
                if val is None:
                    return None
                try:
                    if isinstance(val, str):
                        # Remove % sign, commas, and 'C' prefix
                        val = val.replace('%', '').replace(',', '').lstrip('CH')
                    return float(val)
                except (ValueError, TypeError):
                    return None
            
            # Get IV for this specific strike - field 7633 returns percentage like "19.9%"
            strike_iv = to_float(option_snapshot.get('7633'))
            if strike_iv and strike_iv > 5:
                strike_iv = strike_iv / 100.0
            
            logger.debug(f"Parsed: Strike IV={strike_iv}, Delta={option_snapshot.get('7308')}")
            
            result = {
                'strike': strike,
                'bid': to_float(option_snapshot.get('84')),
                'ask': to_float(option_snapshot.get('86')),
                'last': to_float(option_snapshot.get('31')),
                'delta': to_float(option_snapshot.get('7308')),
                'gamma': to_float(option_snapshot.get('7309')),
                'vega': to_float(option_snapshot.get('7310')),
                'theta': to_float(option_snapshot.get('7311')),
                'implied_vol': strike_iv,
                'volume': to_float(option_snapshot.get('87'))
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error getting option data for strike {strike}: {e}")
            return None
    
    def get_option_chain(
        self,
        stock_conid: int,
        expiry_date: date,
        right: str = 'P',
        exchange: str = 'SMART'
    ) -> Optional[List[Dict]]:
        """
        Get all available option strikes for a given expiry.
        
        Uses /iserver/secdef/info to efficiently retrieve all strikes.
        
        Args:
            stock_conid: Stock contract ID
            expiry_date: Expiration date
            right: 'P' for put, 'C' for call
            exchange: Exchange (default: 'SMART')
            
        Returns:
            List of option contract dicts with keys:
            - conid: Option contract ID
            - strike: Strike price
            - maturityDate: Expiry in YYYYMMDD format
            - right: 'P' or 'C'
        """
        try:
            month_format = expiry_date.strftime('%Y%m')
            
            params = {
                'conid': stock_conid,
                'sectype': 'OPT',
                'month': month_format,
                'exchange': exchange,
                'right': right
            }
            
            url = f"{self.base_url}/iserver/secdef/info"
            response = self._make_request('GET', url, params=params)
            
            if response.status_code != 200:
                logger.warning(f"Failed to get option chain: status {response.status_code}")
                return None
            
            data = response.json()
            
            if not data:
                logger.debug("No option chain data returned")
                return None
            
            # Filter to exact expiry date
            target_expiry_str = expiry_date.strftime('%Y%m%d')
            
            if isinstance(data, list):
                matching_options = [
                    opt for opt in data 
                    if opt.get('maturityDate') == target_expiry_str
                ]
            elif isinstance(data, dict):
                if data.get('maturityDate') == target_expiry_str:
                    matching_options = [data]
                else:
                    matching_options = []
            else:
                matching_options = []
            
            logger.info(f"Found {len(matching_options)} {right} options for {expiry_date.strftime('%Y-%m-%d')}")
            return matching_options
            
        except Exception as e:
            logger.error(f"Error getting option chain: {e}")
            return None
    
    def get_option_strikes(
        self,
        stock_conid: int,
        expiry_date: date,
        sectype: str = 'OPT'
    ) -> Optional[Dict]:
        """
        Get available option strikes for a contract using the proper 3-step procedure.
        
        Step 1: Call /iserver/secdef/search for underlying (always required)
        Step 2: Call /iserver/secdef/strikes with month
        
        Args:
            stock_conid: Stock contract ID
            expiry_date: Expiration date to get strikes for
            sectype: Security type (default: 'OPT')
            
        Returns:
            Dict with available strikes by type:
            {
                'call': [strike1, strike2, ...],
                'put': [strike1, strike2, ...],
                etc.
            }
        """
        try:
            month_format = expiry_date.strftime('%Y%m')
            
            # Step 1: Search for underlying symbol (always required before building option chain)
            logger.debug(f"Step 1: Searching for underlying conid {stock_conid}")
            search_url = f"{self.base_url}/iserver/secdef/search"
            search_params = {'symbol': str(stock_conid)}
            
            search_response = self._make_request('GET', search_url, params=search_params)
            
            if search_response.status_code != 200:
                logger.warning(f"Step 1 failed: status {search_response.status_code}")
                return None
            
            logger.debug(f"Step 1 complete: searched underlying")
            
            # Small delay between steps
            time.sleep(0.2)
            
            # Step 2: Get strikes using conid and month
            logger.debug(f"Step 2: Getting strikes for conid {stock_conid}, month {month_format}")
            strikes_url = f"{self.base_url}/iserver/secdef/strikes"
            strikes_params = {
                'conid': stock_conid,
                'sectype': sectype,
                'month': month_format
            }
            
            strikes_response = self._make_request('GET', strikes_url, params=strikes_params)
            
            if strikes_response.status_code != 200:
                logger.warning(f"Step 2 failed: status {strikes_response.status_code}")
                return None
            
            strikes_data = strikes_response.json()
            logger.debug(f"Step 2 complete: Retrieved strikes data")
            
            if isinstance(strikes_data, dict):
                if 'call' in strikes_data:
                    call_count = len(strikes_data['call']) if isinstance(strikes_data['call'], list) else 0
                    logger.debug(f"Found {call_count} call strikes")
                if 'put' in strikes_data:
                    put_count = len(strikes_data['put']) if isinstance(strikes_data['put'], list) else 0
                    logger.debug(f"Found {put_count} put strikes")
            
            return strikes_data
            
        except Exception as e:
            logger.error(f"Error getting strikes: {e}", exc_info=True)
            return None

    def get_options_near_strike(
        self,
        stock_conid: int,
        target_strike: float,
        expiry_date: date,
        right: str = 'P',
        num_strikes: int = 10
    ) -> List[Dict]:
        """
        Get strikes near a target price for an option chain.
        
        Args:
            stock_conid: Stock contract ID
            target_strike: Target strike price
            expiry_date: Expiration date
            right: 'P' for put, 'C' for call
            num_strikes: Number of strikes to return (closest to target)
            
        Returns:
            List of strike dicts with 'strike' and 'distance' keys, sorted by distance
        """
        # Get all available strikes
        strikes_data = self.get_option_strikes(stock_conid, expiry_date)
        
        if not strikes_data:
            return []
        
        # Get strikes for the requested right
        right_key = 'put' if right == 'P' else 'call'
        available_strikes = strikes_data.get(right_key, [])
        
        if not available_strikes:
            logger.debug(f"No {right_key} strikes available")
            return []
        
        logger.debug(f"Found {len(available_strikes)} available {right} strikes")
        
        # Calculate distance from target and sort
        strike_distances = []
        for strike in available_strikes:
            try:
                strike_val = float(strike)
                distance = abs(strike_val - target_strike)
                strike_distances.append({
                    'strike': strike_val,
                    'distance': distance
                })
            except (ValueError, TypeError):
                continue
        
        # Sort by distance and return closest N strikes
        strike_distances.sort(key=lambda x: x['distance'])
        closest_strikes = strike_distances[:num_strikes]
        
        logger.debug(f"Returning {len(closest_strikes)} strikes closest to ${target_strike:.2f}")
        
        return closest_strikes

    def get_options_data_batch(
        self,
        options_specs: List[Dict],
        right: str = 'P',
        exchange: str = 'SMART',
        skip_preflight: bool = False
    ) -> List[Optional[Dict]]:
        """
        Get option market data for multiple options in batch.
        Much faster than calling get_option_data() individually.
        
        Uses the proper option chain building procedure:
        Step 1: Search underlying + Get strikes (pre-flight) - optional if already done
        Step 2: Get all option conids for the strikes we want
        Step 3: Get market data for ALL options in ONE batch call
        
        Args:
            options_specs: List of dicts with keys:
                - stock_conid: Stock contract ID
                - strike: Strike price
                - expiry_date: Expiration date
            right: 'P' for put, 'C' for call (default: 'P')
            exchange: Exchange (default: 'SMART')
            skip_preflight: If True, skip the search+strikes pre-flight (already done)
            
        Returns:
            List of option data dicts (same order as input), None for failed requests
        """
        if not options_specs:
            return []
        
        logger.debug(f"Batch requesting {len(options_specs)} option contracts...")
        
        # Only do pre-flight if not already done
        if not skip_preflight:
            # Group by stock_conid and expiry for efficient querying
            specs_by_underlying = {}
            for i, spec in enumerate(options_specs):
                key = (spec['stock_conid'], spec['expiry_date'].strftime('%Y%m'))
                if key not in specs_by_underlying:
                    specs_by_underlying[key] = []
                specs_by_underlying[key].append((i, spec))
            
            logger.debug(f"Grouped into {len(specs_by_underlying)} underlying/expiry combinations")
            
            # Step 1: For each underlying/expiry, do pre-flight (search + get strikes)
            # This is required once per underlying/expiry combination
            for (stock_conid, month), specs in specs_by_underlying.items():
                logger.debug(f"Pre-flight for conid {stock_conid}, month {month}")
                expiry_date = specs[0][1]['expiry_date']
                # This does the 2-step pre-flight: search underlying + get strikes
                strikes_data = self.get_option_strikes(stock_conid, expiry_date)
                if strikes_data:
                    logger.debug(f"  Pre-flight complete for {stock_conid}")
            
            # Small delay after pre-flights
            time.sleep(0.3)
        else:
            logger.debug("Skipping pre-flight (already done)")
        
        # Step 2: Get option conids for ALL strikes we're interested in
        option_conids = []
        option_map = {}  # Map conid to original index
        
        logger.debug("Step 2: Getting option conids for all strikes...")
        for i, spec in enumerate(options_specs):
            try:
                stock_conid = spec['stock_conid']
                strike = spec['strike']
                expiry_date = spec['expiry_date']
                month_format = expiry_date.strftime('%Y%m')
                
                # Get specific option contract conid
                option_params = {
                    'conid': stock_conid,
                    'sectype': 'OPT',
                    'month': month_format,
                    'exchange': exchange,
                    'strike': str(strike),
                    'right': right
                }
                
                secdef_url = f"{self.base_url}/iserver/secdef/info"
                secdef_response = self._make_request('GET', secdef_url, params=option_params)
                
                if secdef_response.status_code != 200:
                    logger.debug(f"Failed to get option contract #{i} (strike ${strike}): status {secdef_response.status_code}")
                    option_conids.append(None)
                    continue
                
                option_data = secdef_response.json()
                target_expiry_str = expiry_date.strftime('%Y%m%d');
                
                option_conid = None
                if isinstance(option_data, list):
                    for contract in option_data:
                        if contract.get('maturityDate') == target_expiry_str:
                            option_conid = contract.get('conid')
                            break
                elif isinstance(option_data, dict):
                    if option_data.get('maturityDate') == target_expiry_str:
                        option_conid = option_data.get('conid')
                
                if option_conid:
                    option_conids.append(option_conid)
                    option_map[option_conid] = i
                    logger.debug(f"  Strike ${strike}: conid {option_conid}")
                else:
                    option_conids.append(None)
                    logger.debug(f"  Strike ${strike}: not found")
                    
            except Exception as e:
                logger.warning(f"Error getting option contract #{i}: {e}")
                option_conids.append(None)
        
        valid_conids = [c for c in option_conids if c is not None]
        logger.debug(f"Found {len(valid_conids)}/{len(options_specs)} valid option contracts")
        
        if not valid_conids:
            return [None] * len(options_specs)
        
        # Step 3: Get market data for ALL options in ONE batch call
        logger.debug(f"Step 3: Requesting market data for ALL {len(valid_conids)} options in ONE batch call...")
        
        # Small delay to allow data to populate
        time.sleep(0.5)
        
        snapshots = self.get_market_data_snapshot(
            valid_conids,  # Pass list of all conids
            fields=['31', '84', '86', '87', '7308', '7309', '7310', '7311', '7633'],
            ensure_preflight=False
        )
        
        if not snapshots:
            logger.warning("No market data returned from batch request")
            return [None] * len(options_specs)
        
        logger.debug(f"Received {len(snapshots)} snapshots from batch request")
        
        # Parse and map results back to original order
        results = [None] * len(options_specs)
        
        for snapshot in snapshots:
            conid = snapshot.get('conid')
            if conid not in option_map:
                continue
            
            original_idx = option_map[conid]
            spec = options_specs[original_idx]
            
            # Parse option data
            def to_float(val):
                if val is None:
                    return None
                try:
                    if isinstance(val, str):
                        val = val.replace('%', '').replace(',', '').lstrip('CH')
                    return float(val)
                except (ValueError, TypeError):
                    return None
            
            strike_iv = to_float(snapshot.get('7633'))
            if strike_iv and strike_iv > 5:
                strike_iv = strike_iv / 100.0
            
            results[original_idx] = {
                'strike': spec['strike'],
                'bid': to_float(snapshot.get('84')),
                'ask': to_float(snapshot.get('86')),
                'last': to_float(snapshot.get('31')),
                'delta': to_float(snapshot.get('7308')),
                'gamma': to_float(snapshot.get('7309')),
                'vega': to_float(snapshot.get('7310')),
                'theta': to_float(snapshot.get('7311')),
                'implied_vol': strike_iv,
                'volume': to_float(snapshot.get('87'))
            }
        
        success_count = sum(1 for r in results if r is not None)
        logger.debug(f"Successfully parsed {success_count}/{len(options_specs)} results")
        
        return results
    
    def get_stock_prices_batch(
        self,
        symbols: List[str],
        exchange: str = 'SMART'
    ) -> Dict[str, Optional[float]]:
        """
        Get current stock prices for multiple symbols in batch.
        Much faster than calling get_stock_price() individually.
        
        Args:
            symbols: List of stock ticker symbols
            exchange: Exchange (default: SMART for best execution)
            
        Returns:
            Dictionary mapping symbol to price (None if failed)
        """
        if not symbols:
            return {}
        
        logger.info(f"Batch requesting prices for {len(symbols)} symbols...")
        
        # Ensure account is set up
        if not self.account_id:
            logger.debug("Setting up account before batch market data request...")
            if not self.setup_account():
                logger.warning("Failed to setup account")
                return {symbol: None for symbol in symbols}
        
        # Step 1: Search for all contracts
        symbol_to_conid = {}
        conid_to_symbol = {}
        
        for symbol in symbols:
            try:
                contracts = self.search_contracts(symbol)
                if not contracts:
                    logger.debug(f"No contracts found for {symbol}")
                    symbol_to_conid[symbol] = None
                    continue
                
                # Find the primary stock contract
                stock_contract = None
                for contract in contracts:
                    sections = contract.get('sections', [])
                    has_stock = any(section.get('secType') == 'STK' for section in sections)
                    
                    if has_stock and contract.get('symbol') == symbol:
                        description = contract.get('description', '')
                        # Prefer NYSE or NASDAQ
                        if 'NYSE' in description or 'NASDAQ' in description:
                            stock_contract = contract
                            break
                        elif not stock_contract:
                            stock_contract = contract
            
                if stock_contract:
                    conid = int(stock_contract.get('conid'))
                    symbol_to_conid[symbol] = conid
                    conid_to_symbol[conid] = symbol
                    logger.debug(f"Found conid {conid} for {symbol}")
                else:
                    symbol_to_conid[symbol] = None
                    
            except Exception as e:
                logger.warning(f"Error searching for {symbol}: {e}")
                symbol_to_conid[symbol] = None
        
        valid_conids = [conid for conid in symbol_to_conid.values() if conid is not None]
        logger.info(f"Found {len(valid_conids)}/{len(symbols)} valid stock contracts")
        
        if not valid_conids:
            return {symbol: None for symbol in symbols}
        
        # Small delay after contract searches
        time.sleep(0.5)
        
        # Step 2: Batch request market data for all stocks
        logger.info(f"Requesting market data for {len(valid_conids)} stocks in batch...")
        
        # Try with retry logic
        snapshots = None
        for attempt in range(3):
            snapshots = self.get_market_data_snapshot(
                valid_conids,
                fields=['31', '84', '86', '87'],
                ensure_preflight=False  # Already ensured above
            )
            
            if snapshots:
                # Check if we have actual price data (not just metadata)
                metadata_keys = {'conid', 'conidEx', '_updated', 'server_id', '6119', '6509'}
                has_data_count = sum(
                    1 for snapshot in snapshots 
                    if any(key not in metadata_keys for key in snapshot.keys())
                )
                
                if has_data_count > 0:
                    logger.debug(f"Got data for {has_data_count}/{len(snapshots)} stocks on attempt {attempt + 1}")
                    break
                else:
                    logger.debug(f"Pre-flight response, retrying... (attempt {attempt + 1}/3)")
                    time.sleep(1.0)
            else:
                logger.debug(f"No snapshots, retrying... (attempt {attempt + 1}/3)")
                time.sleep(1.0)
        
        if not snapshots:
            logger.warning("No market data returned from batch request")
            return {symbol: None for symbol in symbols}
        
        # Step 3: Parse results and map back to symbols
        results = {}
        
        def to_float(val):
            if val is None:
                return None
            try:
                if isinstance(val, str):
                    val = val.lstrip('CH').replace(',', '')
                return float(val)
            except (ValueError, TypeError):
                return None
        
        for snapshot in snapshots:
            conid = snapshot.get('conid')
            if conid not in conid_to_symbol:
                continue
            
            symbol = conid_to_symbol[conid]
            
            # Try to extract price from various fields
            # Priority: last price (31) > bid (84)
            last_price = snapshot.get('31')
            bid = snapshot.get('84')
            
            price = to_float(last_price)
            if price is None:
                price = to_float(bid)
            
            results[symbol] = price
            
            if price:
                logger.debug(f"{symbol}: ${price:.2f}")
            else:
                logger.debug(f"{symbol}: No price data in snapshot")
        
        # Fill in None for symbols that weren't found
        for symbol in symbols:
            if symbol not in results:
                results[symbol] = None
        
        success_count = sum(1 for price in results.values() if price is not None)
        logger.info(f"Successfully got prices for {success_count}/{len(symbols)} symbols")
        
        return results