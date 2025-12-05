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
    CLOSE_PRICE = "7295"
    
    # Size fields
    LAST_SIZE = "7295"
    BID_SIZE = "7296"
    ASK_SIZE = "7297"
    
    # Volume and other
    VOLUME = "88"
    SYMBOL = "7219"
    EXCHANGE = "6119"
    
    # Greeks
    DELTA = "7308"          # Option Delta
    GAMMA = "7309"          # Option Gamma
    VEGA = "7310"           # Option Vega
    THETA = "7311"          # Option Theta
    
    # Implied Volatility
    IMPLIED_VOL = "7633"    # Implied Volatility
    
    # Option-specific fields
    OPTION_VOLUME = "7762"  # Option Volume
    UNDERLYING_PRICE = "7635"  # Underlying/Stock Price for options
    SPX_DELTA = "7696"      # SPX Delta (normalized delta relative to SPX)
    
    # Historical Volatility
    HIST_VOL = "7084"       # Historical Volatility
    HIST_VOL_CLOSE = "7087" # Historical Volatility (Close)
    
    # Common field sets
    BASIC_QUOTES = [BID_PRICE, ASK_PRICE, LAST_PRICE]
    
    FULL_QUOTES = [
        LAST_PRICE, BID_PRICE, ASK_PRICE, 
        VOLUME, LAST_SIZE, BID_SIZE, ASK_SIZE, 
        SYMBOL
    ]
    
    OPTION_QUOTES = [
        LAST_PRICE, BID_PRICE, ASK_PRICE,
        BID_SIZE, ASK_SIZE, LAST_SIZE,
        VOLUME, SYMBOL,
        DELTA, GAMMA, VEGA, THETA,
        IMPLIED_VOL, UNDERLYING_PRICE,
        HIST_VOL, SPX_DELTA
    ]
    
    ALL_FIELDS = [
        LAST_PRICE, BID_PRICE, ASK_PRICE, 
        VOLUME, LAST_SIZE, BID_SIZE, ASK_SIZE, 
        SYMBOL, EXCHANGE,
        DELTA, GAMMA, VEGA, THETA,
        IMPLIED_VOL, UNDERLYING_PRICE,
        HIST_VOL, HIST_VOL_CLOSE, OPTION_VOLUME,
        SPX_DELTA
    ]


class IBKRAccountFields:
    """IBKR account/position field definitions."""
    # Add account field definitions as needed
    pass