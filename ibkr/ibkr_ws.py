import websocket
import json
import threading
import logging
import configparser
import requests
import time
from typing import Dict, Optional, Callable, List, Tuple
from collections import defaultdict

from ibkr.fields import IBKRMarketDataFields
from utils.book import Book

logger = logging.getLogger(__name__)


class IBWebSocketClient:
    """
    Interactive Brokers WebSocket API Client for streaming data.
    
    Supports subscribing to:
    - Market data (quotes, trades, bars)
    - Account updates
    - Order updates
    - Portfolio updates
    
    Prerequisites:
    1. Client Portal Gateway must be running
    2. Must be authenticated via REST API first (session token required)
    3. WebSocket support must be enabled in Gateway (v10.19 or later)
    
    Note: WebSocket support in IBKR Client Portal Gateway is OPTIONAL.
    Many Gateway versions do not include WebSocket support.
    """
    
    def __init__(self, config_file: str = "ibkr/ibkr.conf", rest_client: Optional['IBWebAPIClient'] = None):
        """
        Initialize WebSocket client.
        
        Args:
            config_file: Path to configuration file
            rest_client: Optional REST API client for session management
        """
        self.config = configparser.ConfigParser()
        files_read = self.config.read(config_file)
        
        if not files_read:
            logger.warning(f"Could not read config file: {config_file}")
        else:
            logger.info(f"Read config file: {config_file}")
        
        # Read from config file - use 'api' section, not 'ibkr'
        self.base_url = self.config.get('api', 'base_url', fallback='https://localhost:5000/v1/api')
        
        # Check if ws_url is explicitly set in config
        if self.config.has_option('api', 'ws_url'):
            self.ws_url = self.config.get('api', 'ws_url')
            logger.info(f"Using WebSocket URL from config: {self.ws_url}")
        else:
            # Derive from base_url
            self.ws_url = self.base_url.replace('https://', 'wss://').replace('http://', 'ws://').replace('/v1/api', '') + '/v1/api/ws'
            logger.info(f"Derived WebSocket URL from base_url: {self.ws_url}")
        
        self.ws = None
        self.ws_thread = None
        self.is_connected = False
        self.callbacks = defaultdict(list)
        self.market_data_books = {}  # Map conid -> Book object
        self.rest_client = rest_client
        
        # Connection management
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        self.connection_timeout = 10
        self.should_reconnect = True
        
        # Session for REST API calls (if we need to check session status)
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification for self-signed certs
        
        self._connection_lock = threading.Lock()
        
        logger.info(f"IBWebSocketClient initialized")
        logger.info(f"  Config file: {config_file}")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  WebSocket URL: {self.ws_url}")
    
    def _load_config(self, config_file: str) -> configparser.ConfigParser:
        """Load configuration from file."""
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
    
    def _check_websocket_support(self) -> bool:
        """
        Check if the Gateway supports WebSocket.
        This is done by checking the server version.
        
        Returns:
            True if WebSocket is likely supported
        """
        try:
            # Get iserver status which includes version info
            response = self.session.get(
                f"{self.base_url}/iserver/auth/status",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"iServer status: {data}")
                
                # Check if authenticated
                if data.get('authenticated'):
                    server_info = data.get('serverInfo', {})
                    server_version = server_info.get('serverVersion', '')
                    logger.info(f"Gateway version: {server_version}")
                    
                    # WebSocket support was added in version 10.19
                    # Version string format: "Build 10.XX.X..."
                    if 'Build' in server_version:
                        try:
                            version_parts = server_version.split('Build')[1].strip().split('.')
                            major = int(version_parts[0])
                            minor = int(version_parts[1].split(',')[0])
                            
                            if major >= 10 and minor >= 19:
                                logger.info(f"Gateway version {major}.{minor} supports WebSocket")
                                return True
                            else:
                                logger.warning(f"Gateway version {major}.{minor} may not support WebSocket (requires 10.19+)")
                                return False
                        except Exception as e:
                            logger.warning(f"Could not parse version string: {e}")
                            return False
                    
                return False
            else:
                logger.warning(f"Could not check Gateway version: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking WebSocket support: {e}")
            return False
    
    def _get_session_token(self) -> Optional[str]:
        """
        Get session token from REST API.
        This validates that we have an active authenticated session.
        
        Returns:
            Session token if available, None otherwise
        """
        try:
            logger.info(f"Checking session at: {self.base_url}/tickle")
            
            # Check if session is valid by calling tickle endpoint
            response = self.session.post(
                f"{self.base_url}/tickle",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Session validated: {data.get('session', 'unknown')}")
                
                # Get cookies/session info
                cookies = self.session.cookies.get_dict()
                logger.debug(f"Session cookies: {cookies}")
                
                return "authenticated"  # Session is valid
            else:
                logger.error(f"Session validation failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting session token: {e}", exc_info=True)
            return None
    
    def connect(self) -> bool:
        """
        Establish WebSocket connection.
        
        According to IBKR docs:
        1. Must have an authenticated REST session first
        2. WebSocket inherits the session authentication
        3. Connection should be automatic if REST session is valid
        4. WebSocket support requires Gateway version 10.19 or later
        
        Returns:
            True if connection successful, False otherwise
        """
        if self.connection_attempts >= self.max_connection_attempts:
            logger.warning("Max WebSocket connection attempts reached")
            return False
        
        self.connection_attempts += 1
        
        try:
            # Check if WebSocket is supported
            logger.info("Checking if Gateway supports WebSocket...")
            ws_supported = self._check_websocket_support()
            
            if not ws_supported:
                logger.warning("=" * 80)
                logger.warning("WebSocket may not be supported by this Gateway version")
                logger.warning("WebSocket requires Client Portal Gateway v10.19 or later")
                logger.warning("Current functionality will continue using REST API only")
                logger.warning("=" * 80)
                return False
            
            # Verify we have an authenticated REST session
            logger.info("Verifying REST API session...")
            session_token = self._get_session_token()
            
            if not session_token:
                logger.error("No valid REST API session found")
                logger.error("Please authenticate via REST API first")
                return False
            
            logger.info(f"REST session valid, connecting to WebSocket: {self.ws_url}")
            
            # Get cookies from session
            cookies = self.session.cookies.get_dict()
            cookie_str = "; ".join([f"{k}={v}" for k, v in cookies.items()])
            logger.debug(f"Using cookies: {cookie_str}")
            
            # Create WebSocket connection with SSL disabled for self-signed certs
            sslopt = {"cert_reqs": 0}  # Disable SSL certificate verification
            
            # Create header with cookies
            headers = []
            if cookie_str:
                headers.append(f"Cookie: {cookie_str}")
            
            # Enable debug output for troubleshooting
            websocket.enableTrace(False)
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                header=headers,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Run WebSocket in separate thread with SSL verification disabled
            self.ws_thread = threading.Thread(
                target=self.ws.run_forever,
                kwargs={'sslopt': sslopt}
            )
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection with timeout
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < self.connection_timeout:
                time.sleep(0.1)
            
            if self.is_connected:
                logger.info("✓ WebSocket connected successfully")
                return True
            else:
                logger.warning(f"✗ WebSocket connection failed after {self.connection_timeout}s")
                logger.warning("This is normal if:")
                logger.warning("  - Your Gateway version doesn't support WebSocket")
                logger.warning("  - WebSocket is disabled in Gateway settings")
                logger.warning("  - You're using an older Client Portal Gateway")
                logger.warning("")
                logger.warning("The application will continue using REST API polling instead.")
                return False
                
        except Exception as e:
            logger.warning(f"WebSocket connection error: {e}")
            logger.warning("Continuing without WebSocket support (REST API only)")
            return False
    
    def disconnect(self):
        """Disconnect from WebSocket."""
        if self.ws:
            self.should_reconnect = False
            self.ws.close()
            self.is_connected = False
            logger.info("WebSocket disconnected")
    
    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        self.is_connected = True
        logger.info("WebSocket connection opened")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            # Extract topic
            topic = data.get('topic', '')
            
            # Handle different message types
            if topic == 'system':
                self._handle_system(data)
            elif topic == 'sts':
                self._handle_server_status(data)
            elif topic.startswith('smd+'):
                # Market data with conid
                self._handle_market_data(data)
            elif topic == 'smd':
                # This is the error case - log the full message
                logger.error(f"❌ IBKR WebSocket Error on 'smd' topic (no conid):")
                logger.error(f"  Error: {data.get('error')}")
                logger.error(f"  Code: {data.get('code')}")
                logger.error(f"  Full message: {json.dumps(data, indent=2)}")
            else:
                # Generic topic handler
                if topic in self.callbacks:
                    for callback in self.callbacks[topic]:
                        callback(data)
                else:
                    logger.debug(f"Unhandled WebSocket message on topic '{topic}': {data}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
            logger.error(f"Raw message: {message}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}", exc_info=True)
            logger.error(f"Message: {message}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        # Downgrade to warning since WebSocket is optional
        logger.warning(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed."""
        self.is_connected = False
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        
        # Don't attempt reconnection - WebSocket is likely not supported
        if self.should_reconnect and self.connection_attempts < self.max_connection_attempts:
            logger.info("Attempting to reconnect WebSocket in 5 seconds...")
            time.sleep(5)
            self.connect()
    
    def subscribe(self, topic: str, callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to a WebSocket topic.
        
        This is the core subscription method that handles all WebSocket subscriptions.
        
        Args:
            topic: Full subscription message to send (e.g., 'smd+265598+{"fields":["31","84"]}')
            callback: Optional callback function to handle messages for this topic
            
        Returns:
            True if subscription successful
        """
        try:
            if not self.is_connected:
                logger.warning("Cannot subscribe - WebSocket not connected")
                return False
            
            # Extract the topic key for callback registration
            # Topic format: smd+{conid}+{fields} -> extract smd+{conid}
            topic_key = topic.split('+')[0] + '+' + topic.split('+')[1] if '+' in topic else topic
            
            # Register callback if provided
            if callback:
                self.callbacks[topic_key].append(callback)
            
            # Send subscription message
            self.ws.send(topic)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to topic '{topic}': {e}", exc_info=True)
            return False
    
    def unsubscribe_market_data(self, conid: int) -> bool:
        """
        Unsubscribe from streaming market data for a contract.
        
        Args:
            conid: Contract ID to unsubscribe from
            
        Returns:
            True if unsubscription successful
        """
        try:
            # Remove the book reference
            if conid in self.market_data_books:
                del self.market_data_books[conid]
            
            # Use the generic unsubscribe method
            topic = f'smd+{conid}'
            logger.info(f"Unsubscribing from market data for conid {conid}")
            
            return self.unsubscribe(topic)
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from market data for conid {conid}: {e}", exc_info=True)
            return False
    
    def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            
        Returns:
            True if unsubscription successful
        """
        try:
            if not self.is_connected:
                logger.debug("Cannot unsubscribe - WebSocket not connected")
                return False
            
            # Clear callbacks for this topic
            if topic in self.callbacks:
                del self.callbacks[topic]
            
            # Send unsubscription message for market data topics
            if topic.startswith('smd+'):
                # Extract conid from topic (format: smd+{conid} or smd+{conid}+{fields})
                conid = topic.split('+')[1]
                unsubscribe_msg = f"umd+{conid}"
                self.ws.send(unsubscribe_msg)
                logger.info(f"Sent unsubscription message: {unsubscribe_msg}")
            elif topic.startswith('sor+'):
                conid = topic.split('+')[1]
                unsubscribe_msg = f"uor+{conid}"
                self.ws.send(unsubscribe_msg)
                logger.info(f"Sent unsubscription message: {unsubscribe_msg}")
            else:
                logger.debug(f"No unsubscription message needed for topic: {topic}")
            
            logger.info(f"Unsubscribed from topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from topic '{topic}': {e}", exc_info=True)
            return False
    
    def subscribe_market_data(self, conid: int, option: 'Option', is_option: bool = True) -> bool:
        """
        Subscribe to streaming market data for a contract.
        
        Automatically subscribes to appropriate fields based on instrument type.
        For options: includes Greeks, IV, and underlying price.
        For stocks: basic quotes only.
        
        Args:
            conid: Contract ID to subscribe to
            option: Option object to update with market data (for Greeks) or Book object for non-options
            is_option: Whether this is an option contract (default: True)
            
        Returns:
            True if subscription successful
        """
        try:
            # Store the option reference for this conid
            self.market_data_books[conid] = option
            
            # Use appropriate field set based on instrument type
            if is_option:
                fields = IBKRMarketDataFields.OPTION_QUOTES
            else:
                fields = IBKRMarketDataFields.FULL_QUOTES
            
            # Format exactly as shown in IBKR docs:
            # ws.send('smd+'+conid+'+{"fields":["31","84","86"]}')
            fields_json = '{"fields":' + json.dumps(fields) + '}'
            topic = f'smd+{conid}+{fields_json}'
            
            # Use the generic subscribe() method
            return self.subscribe(topic, callback=None)
            
        except Exception as e:
            logger.error(f"Failed to subscribe to market data for conid {conid}: {e}", exc_info=True)
            return False
    
    def _handle_system(self, data: Dict):
        """Handle system messages."""
        # Filter out heartbeat messages
        if 'hb' in data:
            return
        
        logger.info(f"System message: {data}")
        
        # Notify callbacks
        if 'system' in self.callbacks:
            for callback in self.callbacks['system']:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in system callback: {e}", exc_info=True)
    
    def _handle_server_status(self, data: Dict):
        """Handle server status messages."""
        logger.info(f"Server status: {data}")
        
        # Notify callbacks
        if 'sts' in self.callbacks:
            for callback in self.callbacks['sts']:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in server status callback: {e}", exc_info=True)
    
    def _handle_market_data(self, data: Dict):
        """
        Handle streaming market data updates.
        Automatically updates the Book and Option objects associated with the conid.
        """
        try:
            conid = data.get('conid')
            if not conid:
                logger.warning("Market data update missing conid")
                return
            
            # Helper function to safely convert string to float
            def safe_float(value):
                if value is None:
                    return None
                try:
                    if isinstance(value, str):
                        value = value.replace(',', '')
                    return float(value)
                except (ValueError, TypeError):
                    return None
            
            # Helper function to safely convert to int
            def safe_int(value):
                if value is None:
                    return None
                try:
                    if isinstance(value, str):
                        value = value.replace(',', '')
                    return int(float(value))
                except (ValueError, TypeError):
                    return None
            
            # Extract price fields
            bid = safe_float(data.get(IBKRMarketDataFields.BID_PRICE))
            ask = safe_float(data.get(IBKRMarketDataFields.ASK_PRICE))
            last = safe_float(data.get(IBKRMarketDataFields.LAST_PRICE))
            bid_size = safe_int(data.get(IBKRMarketDataFields.BID_SIZE))
            ask_size = safe_int(data.get(IBKRMarketDataFields.ASK_SIZE))
            last_size = safe_int(data.get(IBKRMarketDataFields.LAST_SIZE))
            
            # Extract Greeks
            delta = safe_float(data.get(IBKRMarketDataFields.DELTA))
            gamma = safe_float(data.get(IBKRMarketDataFields.GAMMA))
            vega = safe_float(data.get(IBKRMarketDataFields.VEGA))
            theta = safe_float(data.get(IBKRMarketDataFields.THETA))
            
            # Extract volatility fields
            implied_vol = safe_float(data.get(IBKRMarketDataFields.IMPLIED_VOL))
            hist_vol = safe_float(data.get(IBKRMarketDataFields.HIST_VOL))
            underlying_price = safe_float(data.get(IBKRMarketDataFields.UNDERLYING_PRICE))
            
            timestamp = data.get('_updated', time.time() * 1000) / 1000.0
            
            # Check if we have actual price data (not just metadata)
            has_price_data = bid is not None or ask is not None or last is not None
            
            if not has_price_data:
                return
            
            # Find the option object for this conid and update it
            if conid in self.market_data_books:
                option = self.market_data_books[conid]
                
                # Update the option's book with prices
                option.option_book.update(
                    bid=bid,
                    ask=ask,
                    last=last,
                    bid_size=bid_size,
                    ask_size=ask_size,
                    last_size=last_size,
                    timestamp=timestamp
                )
                
                # Update the option's Greeks
                option.update_greeks(
                    delta=delta,
                    gamma=gamma,
                    vega=vega,
                    theta=theta,
                    implied_vol=implied_vol,
                    hist_vol=hist_vol,
                    underlying_price=underlying_price,
                    timestamp=timestamp
                )
            
        except Exception as e:
            logger.error(f"Error handling market data: {e}", exc_info=True)
            logger.error(f"Raw data: {data}")
    
    def subscribe_account_updates(self, callback: Callable) -> bool:
        """
        Subscribe to account ledger and summary updates.
        
        Args:
            callback: Function to call when account data changes
            
        Returns:
            True if subscription successful
        """
        try:
            # Subscribe to account ledger (spl)
            if not self.subscribe('spl', callback):
                return False
            
            # Subscribe to account summary (ssd)
            if not self.subscribe('ssd', callback):
                return False
            
            logger.info("Subscribed to account updates")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to account updates: {e}", exc_info=True)
            return False
    
    def subscribe_portfolio_updates(self, callback: Callable) -> bool:
        """
        Subscribe to portfolio/position updates.
        
        Args:
            callback: Function to call when portfolio changes
            
        Returns:
            True if subscription successful
        """
        try:
            # Subscribe to portfolio ledger
            if not self.subscribe('spl', callback):
                return False
            
            logger.info("Subscribed to portfolio updates")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to portfolio updates: {e}", exc_info=True)
            return False
    
    def subscribe_order_updates(self, callback: Callable) -> bool:
        """
        Subscribe to order status updates.
        
        Args:
            callback: Function to call when orders change
            
        Returns:
            True if subscription successful
        """
        try:
            # Subscribe to streaming orders (sor)
            if not self.subscribe('sor', callback):
                return False
            
            logger.info("Subscribed to order updates")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to order updates: {e}", exc_info=True)
            return False
    
    def subscribe_all_position_market_data(self, options: List) -> Tuple[int, int]:
        """
        Subscribe to market data for a list of option positions.
        
        Args:
            options: List of Option objects with conid attribute
            
        Returns:
            Tuple of (successful_subscriptions, failed_subscriptions)
        """
        subscribed = 0
        failed = 0
        
        for opt in options:
            if not opt.conid:
                logger.warning(f"Skipping {opt.symbol} - no conid")
                failed += 1
                continue
            
            logger.info(f"Subscribing to market data for {opt.symbol} (conid={opt.conid})...")
            
            # Pass the entire Option object, not just the book
            if self.subscribe_market_data(opt.conid, opt, is_option=True):
                subscribed += 1
                logger.info(f"  ✓ Subscribed successfully")
            else:
                failed += 1
                logger.warning(f"  ✗ Subscription failed")
            
            time.sleep(0.1)  # Rate limiting
        
        logger.info(f"Market data subscription summary: {subscribed} succeeded, {failed} failed")
        return subscribed, failed
