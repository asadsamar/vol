import requests
import json
from typing import Dict, Optional, List
import logging
import configparser
import os
import time
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
        
        logger.info("Rate limiter initialized with limits:")
        for category, limit in rate_limits.items():
            logger.info(f"  {category}: {limit} req/sec ({1.0/limit:.2f}s interval)")
    
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
            response = self.session.get(f"{self.base_url}/iserver/auth/status")
            response.raise_for_status()
            
            auth_status = response.json()
            logger.info(f"Authentication status: {auth_status}")
            
            if auth_status.get('authenticated', False):
                logger.info("Already authenticated")
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
        Fetches available accounts and selects the configured one or the first available.
        
        Returns:
            True if account setup successful, False otherwise
        """
        # Get available accounts
        accounts_response = self._fetch_accounts()
        
        if not accounts_response:
            logger.error("No accounts found")
            return False
        
        # Extract account IDs and store full account info
        self.available_accounts = accounts_response
        account_ids = [acc['accountId'] for acc in accounts_response]
        
        logger.info(f"Found {len(account_ids)} account(s):")
        for acc in accounts_response:
            logger.info(f"  - {acc['accountId']}: {acc['accountTitle']} ({acc['tradingType']})")
        
        # Get configured account ID
        configured_account = self.config.get('account', 'account_id', fallback='').strip()
        
        if configured_account:
            if configured_account in account_ids:
                self.account_id = configured_account
                logger.info(f"Using configured account: {configured_account}")
                return True
            else:
                logger.error(f"Configured account '{configured_account}' not found in available accounts")
                logger.error(f"Available accounts: {account_ids}")
                return False
        else:
            # Use first account if none configured
            self.account_id = account_ids[0]
            logger.info(f"No account configured, using first available: {self.account_id}")
            return True
    
    def _fetch_accounts(self) -> Optional[List[Dict]]:
        """
        Internal method to fetch available accounts.
        
        Returns:
            List of account dictionaries
        """
        try:
            response = self.session.get(f"{self.base_url}/portfolio/accounts")
            response.raise_for_status()
            accounts = response.json()
            return accounts
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get accounts: {e}")
            return None
    
    def get_accounts(self) -> Optional[List[str]]:
        """
        Get list of available account IDs.
        
        Returns:
            List of account ID strings
        """
        if self.available_accounts:
            return [acc['accountId'] for acc in self.available_accounts]
        
        accounts_response = self._fetch_accounts()
        if accounts_response:
            self.available_accounts = accounts_response
            return [acc['accountId'] for acc in accounts_response]
        
        return None
    
    def get_account_info(self, account_id: Optional[str] = None) -> Optional[Dict]:
        """
        Get detailed information about an account.
        
        Args:
            account_id: Account ID (uses configured account if not provided)
            
        Returns:
            Dictionary with account information
        """
        if account_id is None:
            account_id = self.account_id
        
        if not account_id:
            logger.error("No account ID provided and no account configured")
            return None
        
        for acc in self.available_accounts:
            if acc['accountId'] == account_id:
                return acc
        
        return None
    
    def tickle(self) -> bool:
        """
        Keep the session alive by calling the tickle endpoint.
        Should be called periodically to maintain connection.
        """
        try:
            response = self.session.post(f"{self.base_url}/tickle")
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
            response = self.session.get(
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
            response = self.session.get(
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


def main():
    """
    Example usage of the IB Web API Client
    """
    # Initialize client
    client = IBWebAPIClient()
    
    print("=" * 60)
    print("Interactive Brokers Web API - Account Balance Example")
    print("=" * 60)
    
    # Step 1: Check authentication
    print("\n1. Checking authentication...")
    if not client.authenticate():
        print("\nPlease complete these steps:")
        print("1. Download Client Portal Gateway from:")
        print("   https://www.interactivebrokers.com/en/trading/cpgw.php")
        print("2. Start the gateway application")
        print("3. Open https://localhost:5000 in your browser")
        print("4. Log in with your IB credentials")
        print("5. Run this script again")
        return
    
    # Step 2: Keep session alive
    print("\n2. Keeping session alive...")
    client.tickle()
    
    # Step 3: Get available accounts
    print("\n3. Retrieving accounts...")
    accounts = client.get_accounts()
    
    if not accounts:
        print("No accounts found or error occurred")
        return
    
    print(f"\nFound {len(accounts)} account(s)")
    
    # Step 4: Get balance for each account
    for account_id in accounts:
        print(f"\n{'=' * 60}")
        print(f"Account: {account_id}")
        print(f"{'=' * 60}")
        
        # Get account summary
        print("\nAccount Summary:")
        balance = client.get_account_balance(account_id)
        if balance:
            print(json.dumps(balance, indent=2))
        
        # Get detailed ledger
        print("\nAccount Ledger:")
        ledger = client.get_account_ledger(account_id)
        if ledger:
            print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()