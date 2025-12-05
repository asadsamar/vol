import re
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_option_symbol(symbol: str) -> Optional[Dict]:
    """
    Parse IBKR option symbol format.
    
    IBKR option format: [SYMBOL YYMMDDP/CSTRIKE MULTIPLIER]
    Example: [INTC  251205P00040500 100]
    - INTC: underlying symbol
    - 251205: expiry date (Dec 5, 2025)
    - P: Put (or C for Call)
    - 00040500: strike price (40.50 with padding)
    - 100: multiplier
    
    Args:
        symbol: Option symbol string
        
    Returns:
        Dictionary with parsed components or None if parse fails
    """
    try:
        # Pattern to match option symbol
        # Underlying can have spaces, followed by YYMMDD, P/C, strike (8 digits), multiplier
        pattern = r'([A-Z]+)\s+(\d{6})([PC])(\d{8})\s+(\d+)'
        match = re.search(pattern, symbol)
        
        if not match:
            return None
        
        underlying = match.group(1).strip()
        date_str = match.group(2)
        right = match.group(3)
        strike_str = match.group(4)
        multiplier = int(match.group(5))
        
        # Parse date
        year = int('20' + date_str[0:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        expiry_date = datetime(year, month, day)
        
        # Parse strike (format: 00040500 = 40.50)
        strike = float(strike_str) / 1000.0
        
        return {
            'underlying': underlying,
            'expiry_date': expiry_date,
            'expiry_str': expiry_date.strftime('%Y-%m-%d'),
            'right': 'PUT' if right == 'P' else 'CALL',
            'strike': strike,
            'multiplier': multiplier
        }
        
    except Exception as e:
        logger.error(f"Failed to parse option symbol '{symbol}': {e}")
        return None


def is_short_put(position: Dict) -> bool:
    """
    Determine if a position is a short put.
    
    Args:
        position: Position dictionary from IBKR API
        
    Returns:
        True if position is a short put, False otherwise
    """
    # Check if it's an option
    asset_class = position.get('assetClass', '')
    if asset_class != 'OPT':
        return False
    
    # Check if it's a short position (negative quantity)
    quantity = float(position.get('position', 0))
    if quantity >= 0:
        return False
    
    # Check if it's a put by looking for pattern YYMMDDP in contract description
    contract_desc = position.get('contractDesc', '')
    
    # Pattern: 6 digits followed by P
    pattern = r'\d{6}P'
    if re.search(pattern, contract_desc):
        return True
    
    return False


def is_short_call(position: Dict) -> bool:
    """
    Determine if a position is a short call.
    
    Args:
        position: Position dictionary from IBKR API
        
    Returns:
        True if position is a short call, False otherwise
    """
    # Check if it's an option
    asset_class = position.get('assetClass', '')
    if asset_class != 'OPT':
        return False
    
    # Check if it's a short position (negative quantity)
    quantity = float(position.get('position', 0))
    if quantity >= 0:
        return False
    
    # Check if it's a call by looking for pattern YYMMDDC in contract description
    contract_desc = position.get('contractDesc', '')
    
    # Pattern: 6 digits followed by C
    pattern = r'\d{6}C'
    if re.search(pattern, contract_desc):
        return True
    
    return False


def get_short_puts(positions: List[Dict]) -> List[Dict]:
    """
    Filter positions to get only short puts.
    
    Args:
        positions: List of position dictionaries
        
    Returns:
        List of short put positions
    """
    return [pos for pos in positions if is_short_put(pos)]


def get_short_calls(positions: List[Dict]) -> List[Dict]:
    """
    Filter positions to get only short calls.
    
    Args:
        positions: List of position dictionaries
        
    Returns:
        List of short call positions
    """
    return [pos for pos in positions if is_short_call(pos)]


def get_option_positions(positions: List[Dict]) -> List[Dict]:
    """
    Filter positions to get only options.
    
    Args:
        positions: List of position dictionaries
        
    Returns:
        List of option positions
    """
    return [pos for pos in positions if pos.get('assetClass') == 'OPT']


def calculate_put_cash_requirement(position: Dict) -> float:
    """
    Calculate cash required if a short put is exercised.
    
    Args:
        position: Position dictionary for a short put
        
    Returns:
        Cash requirement (strike × multiplier × contracts)
    """
    strike = float(position.get('strike', 0))
    quantity = abs(float(position.get('position', 0)))
    multiplier = float(position.get('multiplier', 100))
    
    return strike * multiplier * quantity


def format_option_symbol(position: Dict) -> str:
    """
    Format an option position into a readable string.
    
    Args:
        position: Position dictionary
        
    Returns:
        Formatted string like "INTC 251205 P40.50"
    """
    contract_desc = position.get('contractDesc', '')
    parsed = parse_option_symbol(contract_desc)
    
    if parsed:
        return f"{parsed['underlying']} {parsed['expiry_str']} {parsed['right'][0]}{parsed['strike']:.2f}"
    
    return contract_desc


def get_days_to_expiry(position: Dict) -> Optional[int]:
    """
    Calculate days until option expiration.
    
    Args:
        position: Position dictionary
        
    Returns:
        Number of days to expiry or None if cannot parse
    """
    contract_desc = position.get('contractDesc', '')
    parsed = parse_option_symbol(contract_desc)
    
    if parsed:
        today = datetime.now()
        days = (parsed['expiry_date'] - today).days
        return days
    
    return None