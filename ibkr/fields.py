"""
IBKR WebSocket field definitions and constants.
"""

class IBKRMarketDataFields:
    """
    IBKR WebSocket streaming market data field IDs.
    
    Reference: https://www.interactivebrokers.com/campus/ibkr-api-page/webapi-doc/
    """
    # Price fields
    LAST_PRICE = "31"
    BID_PRICE = "84"
    ASK_PRICE = "86"
    
    # Size fields
    LAST_SIZE = "7295"
    BID_SIZE = "7296"
    ASK_SIZE = "7297"
    
    # Volume and other
    VOLUME = "88"
    SYMBOL = "7219"
    EXCHANGE = "6119"
    
    # Common field sets
    BASIC_QUOTES = [BID_PRICE, ASK_PRICE, LAST_PRICE]
    FULL_QUOTES = [LAST_PRICE, BID_PRICE, ASK_PRICE, VOLUME, LAST_SIZE, BID_SIZE, ASK_SIZE, SYMBOL]
    ALL_FIELDS = [LAST_PRICE, BID_PRICE, ASK_PRICE, VOLUME, LAST_SIZE, BID_SIZE, ASK_SIZE, SYMBOL, EXCHANGE]


class IBKRAccountFields:
    """IBKR account/position field definitions."""
    # Add account field definitions as needed
    pass