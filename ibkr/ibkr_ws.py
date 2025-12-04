import websocket
import json
import threading
import time
import logging
from typing import Dict, Optional, Callable, List
import configparser
import os
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
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
    2. Must be authenticated via REST API first
    """
    
    def __init__(self, config_file: str = "ibkr.conf"):
        """
        Initialize the WebSocket client.
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.host = self.config.get('api', 'host', fallback='localhost')
        self.port = self.config.get('api', 'port', fallback='5001')
        self.ws_url = f"wss://{self.host}:{self.port}/v1/api/ws"
        
        # WebSocket connection
        self.ws = None
        self.ws_thread = None
        self.is_connected = False
        self.should_reconnect = True
        
        # Subscriptions and callbacks
        self.subscriptions = defaultdict(list)  # topic -> list of callback functions
        self.active_subscriptions = set()  # set of active subscription IDs
        
        # Message handling
        self.message_handlers = {
            'smd': self._handle_market_data,
            'act': self._handle_account_update,
            'ord': self._handle_order_update,
            'pnl': self._handle_pnl_update,
            'error': self._handle_error,
        }
        
        # Set logging level
        log_level = self.config.get('logging', 'level', fallback='INFO')
        logger.setLevel(getattr(logging, log_level))
    
    def _load_config(self, config_file: str) -> configparser.ConfigParser:
        """Load configuration from file."""
        config = configparser.ConfigParser()
        
        config_paths = [
            config_file,
            os.path.join(os.path.dirname(__file__), config_file)
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                config.read(path)
                logger.info(f"Loaded configuration from {path}")
                return config
        
        logger.warning("Configuration file not found, using defaults")
        config.add_section('api')
        config.set('api', 'host', 'localhost')
        config.set('api', 'port', '5001')
        config.add_section('logging')
        config.set('logging', 'level', 'INFO')
        
        return config
    
    def connect(self) -> bool:
        """
        Establish WebSocket connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to WebSocket: {self.ws_url}")
            
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Run WebSocket in separate thread
            self.ws_thread = threading.Thread(
                target=self.ws.run_forever,
                kwargs={'sslopt': {'cert_reqs': 0}}  # Disable SSL verification for localhost
            )
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.is_connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if self.is_connected:
                logger.info("WebSocket connected successfully")
                return True
            else:
                logger.error("WebSocket connection timeout")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            return False
    
    def disconnect(self):
        """Close WebSocket connection."""
        self.should_reconnect = False
        if self.ws:
            self.ws.close()
        self.is_connected = False
        logger.info("WebSocket disconnected")
    
    def _on_open(self, ws):
        """Called when WebSocket connection is established."""
        self.is_connected = True
        logger.info("WebSocket connection opened")
    
    def _on_message(self, ws, message):
        """
        Called when a message is received from WebSocket.
        
        Args:
            ws: WebSocket instance
            message: Raw message string
        """
        try:
            data = json.loads(message)
            logger.debug(f"Received message: {data}")
            
            # Determine message type and route to appropriate handler
            topic = data.get('topic', '')
            
            # Call registered callbacks for this topic
            if topic in self.subscriptions:
                for callback in self.subscriptions[topic]:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in callback for topic {topic}: {e}")
            
            # Also call built-in handlers
            for prefix, handler in self.message_handlers.items():
                if topic.startswith(prefix):
                    handler(data)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _on_error(self, ws, error):
        """Called when WebSocket error occurs."""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Called when WebSocket connection is closed."""
        self.is_connected = False
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        # Attempt reconnection if desired
        if self.should_reconnect:
            logger.info("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            self.connect()
    
    def subscribe(self, topic: str, callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to a topic.
        
        Args:
            topic: Topic to subscribe to (e.g., 'smd+265598+{"fields":["31","84","85","86"]}')
            callback: Optional callback function to handle messages
            
        Returns:
            True if subscription successful, False otherwise
        """
        if not self.is_connected:
            logger.error("Cannot subscribe: WebSocket not connected")
            return False
        
        try:
            # Send subscription message
            sub_msg = json.dumps({
                "action": "subscribe",
                "topic": topic
            })
            
            self.ws.send(sub_msg)
            self.active_subscriptions.add(topic)
            
            if callback:
                self.subscriptions[topic].append(callback)
            
            logger.info(f"Subscribed to topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {topic}: {e}")
            return False
    
    def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from a topic.
        
        Args:
            topic: Topic to unsubscribe from
            
        Returns:
            True if unsubscription successful, False otherwise
        """
        if not self.is_connected:
            logger.error("Cannot unsubscribe: WebSocket not connected")
            return False
        
        try:
            # Send unsubscribe message
            unsub_msg = json.dumps({
                "action": "unsubscribe",
                "topic": topic
            })
            
            self.ws.send(unsub_msg)
            self.active_subscriptions.discard(topic)
            
            # Remove callbacks
            if topic in self.subscriptions:
                del self.subscriptions[topic]
            
            logger.info(f"Unsubscribed from topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {topic}: {e}")
            return False
    
    def subscribe_market_data(self, conid: int, fields: List[str], callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to market data for a contract.
        
        Args:
            conid: Contract ID
            fields: List of field IDs to subscribe to
                   Common fields: "31" (last price), "84" (bid), "86" (ask), "85" (bid size), "88" (ask size)
            callback: Optional callback function
            
        Returns:
            True if subscription successful
        """
        topic = f'smd+{conid}+{json.dumps({"fields": fields})}'
        return self.subscribe(topic, callback)
    
    def subscribe_orders(self, callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to order updates.
        
        Args:
            callback: Optional callback function
            
        Returns:
            True if subscription successful
        """
        return self.subscribe('ord', callback)
    
    def subscribe_pnl(self, account_id: str, callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to P&L updates.
        
        Args:
            account_id: Account ID
            callback: Optional callback function
            
        Returns:
            True if subscription successful
        """
        topic = f'pnl+{account_id}'
        return self.subscribe(topic, callback)
    
    def subscribe_account(self, callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to account updates.
        
        Args:
            callback: Optional callback function
            
        Returns:
            True if subscription successful
        """
        return self.subscribe('act', callback)
    
    def _handle_market_data(self, data: Dict):
        """Handle market data updates."""
        logger.debug(f"Market data: {data}")
    
    def _handle_account_update(self, data: Dict):
        """Handle account updates."""
        logger.debug(f"Account update: {data}")
    
    def _handle_order_update(self, data: Dict):
        """Handle order updates."""
        logger.info(f"Order update: {data}")
    
    def _handle_pnl_update(self, data: Dict):
        """Handle P&L updates."""
        logger.debug(f"P&L update: {data}")
    
    def _handle_error(self, data: Dict):
        """Handle error messages."""
        logger.error(f"Error from server: {data}")
    
    def get_active_subscriptions(self) -> List[str]:
        """
        Get list of active subscriptions.
        
        Returns:
            List of active topic strings
        """
        return list(self.active_subscriptions)