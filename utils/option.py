"""
Option contract class for representing individual option positions.
"""
from typing import Optional, Dict
from datetime import datetime, date
from utils.book import Book
import logging

logger = logging.getLogger(__name__)


class Option:
    """
    Represents an option contract (call or put).
    
    Attributes:
        underlyer: Underlying symbol (e.g., 'SPY', 'AAPL')
        strike: Strike price
        expiry: Expiration date
        option_type: 'CALL' or 'PUT'
        quantity: Number of contracts (negative for short positions)
        conid: IBKR contract ID
        multiplier: Contract multiplier (typically 100)
        currency: Currency (e.g., 'USD')
        option_book: Market data book for the option
        underlyer_book: Market data book for the underlying
    """
    
    def __init__(
        self,
        underlyer: str,
        strike: float,
        expiry: date,
        option_type: str = 'CALL',
        quantity: float = 0,
        conid: Optional[int] = None,
        multiplier: int = 100,
        currency: str = 'USD'
    ):
        self.underlyer = underlyer
        self.strike = strike
        self.expiry = expiry
        self.option_type = option_type.upper()
        self.quantity = quantity
        self.conid = conid
        self.multiplier = multiplier
        self.currency = currency
        
        # Market data books (only prices and sizes)
        self.option_book = Book()
        self.underlyer_book = Book()
        
        # Greeks and volatility (option-specific data)
        self.delta: Optional[float] = None
        self.gamma: Optional[float] = None
        self.vega: Optional[float] = None
        self.theta: Optional[float] = None
        self.implied_vol: Optional[float] = None
        self.hist_vol: Optional[float] = None
        self.underlying_price: Optional[float] = None
        self.spx_delta: Optional[float] = None  # SPX-normalized delta
        self.greeks_timestamp: Optional[float] = None
    
    def update_option_book(
        self,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        last: Optional[float] = None,
        bid_size: Optional[int] = None,
        ask_size: Optional[int] = None,
        last_size: Optional[int] = None,
        timestamp: Optional[float] = None
    ):
        """Update option market data book."""
        self.option_book.update(
            bid=bid,
            ask=ask,
            last=last,
            bid_size=bid_size,
            ask_size=ask_size,
            last_size=last_size,
            timestamp=timestamp
        )
    
    def update_underlyer_book(
        self,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        last: Optional[float] = None,
        bid_size: Optional[int] = None,
        ask_size: Optional[int] = None,
        last_size: Optional[int] = None,
        timestamp: Optional[float] = None
    ):
        """Update underlying market data book."""
        self.underlyer_book.update(
            bid=bid,
            ask=ask,
            last=last,
            bid_size=bid_size,
            ask_size=ask_size,
            last_size=last_size,
            timestamp=timestamp
        )
    
    def update_greeks(
        self,
        delta: Optional[float] = None,
        gamma: Optional[float] = None,
        vega: Optional[float] = None,
        theta: Optional[float] = None,
        implied_vol: Optional[float] = None,
        hist_vol: Optional[float] = None,
        underlying_price: Optional[float] = None,
        spx_delta: Optional[float] = None,
        timestamp: Optional[float] = None
    ):
        """Update option Greeks and volatility data."""
        if delta is not None:
            self.delta = delta
        if gamma is not None:
            self.gamma = gamma
        if vega is not None:
            self.vega = vega
        if theta is not None:
            self.theta = theta
        if implied_vol is not None:
            self.implied_vol = implied_vol
        if hist_vol is not None:
            self.hist_vol = hist_vol
        if underlying_price is not None:
            self.underlying_price = underlying_price
        if spx_delta is not None:
            self.spx_delta = spx_delta
        if timestamp is not None:
            self.greeks_timestamp = timestamp
    
    def get_exercise_probability(self) -> Optional[float]:
        """
        Calculate probability of exercise based on delta.
        
        For puts: delta is negative, so probability = abs(delta)
        For calls: delta is positive, so probability = delta
        
        Delta approximates the probability of the option being in-the-money at expiration.
        
        Returns:
            Probability (0.0 to 1.0) or None if delta not available
        """
        if self.delta is None:
            return None
        
        # For puts, delta is negative (e.g., -0.30 means ~30% chance of being ITM)
        # For calls, delta is positive (e.g., 0.30 means ~30% chance of being ITM)
        return abs(self.delta)
    
    def get_exercise_risk_amount(self) -> Optional[float]:
        """
        Calculate the dollar amount at risk if option is exercised.
        
        For short puts: strike * multiplier * abs(quantity) * probability
        For short calls: strike * multiplier * abs(quantity) * probability
        
        Returns:
            Dollar amount at risk, or None if probability can't be calculated
        """
        prob = self.get_exercise_probability()
        if prob is None:
            return None
        
        # For short positions, quantity is negative
        contracts = abs(self.quantity)
        
        # Exercise amount = strike * multiplier * contracts
        exercise_amount = self.strike * self.multiplier * contracts
        
        # Risk-adjusted amount = exercise_amount * probability
        risk_adjusted_amount = exercise_amount * prob
        
        return risk_adjusted_amount
    
    @classmethod
    def from_ibkr_position(cls, position: Dict) -> Optional['Option']:
        """
        Create an Option from IBKR position data.
        
        Args:
            position: IBKR position dictionary
            
        Returns:
            Option instance or None if not an option position
        """
        try:
            # Check if this is an option
            asset_class = position.get('assetClass', '').upper()
            if asset_class != 'OPT':
                return None
            
            # Debug: Print the raw position data
            logger.debug(f"Parsing option position: {position}")
            
            # Parse contract description
            # Format: "GLD    DEC2025 379 P [GLD   251212P00379000 100]"
            # or: "SYMBOL MMYYYY STRIKE C/P [OCC_SYMBOL SIZE]"
            contract_desc = position.get('contractDesc', '')
            
            # Try to extract from OCC symbol in brackets first (more reliable)
            # OCC format: SYMBOL 6-digit-date C/P 8-digit-strike
            # Example: GLD   251212P00379000
            if '[' in contract_desc and ']' in contract_desc:
                occ_part = contract_desc.split('[')[1].split(']')[0].strip()
                occ_parts = occ_part.split()
                if len(occ_parts) >= 2:
                    occ_symbol = occ_parts[0].strip()
                    # OCC format: SYMBOL YYMMDD[C/P]STRIKE
                    # Look for C or P in the second part
                    if 'P' in occ_parts[1]:
                        option_type = 'PUT'
                    elif 'C' in occ_parts[1]:
                        option_type = 'CALL'
                    else:
                        logger.warning(f"Could not determine option type from OCC: {occ_parts[1]}")
                        logger.debug(f"Full contract desc: {contract_desc}")
                        return None
                else:
                    # Fallback: look at end of contract desc
                    if contract_desc.strip().endswith('P'):
                        option_type = 'PUT'
                    elif contract_desc.strip().endswith('C'):
                        option_type = 'CALL'
                    else:
                        logger.warning(f"Could not determine option type from: {contract_desc}")
                        return None
            else:
                # No brackets, try simple parsing
                if contract_desc.endswith('P'):
                    option_type = 'PUT'
                elif contract_desc.endswith('C'):
                    option_type = 'CALL'
                else:
                    logger.warning(f"Could not determine option type from: {contract_desc}")
                    return None
            
            # Get underlyer from ticker or parse from description
            underlyer = position.get('ticker', '')
            if not underlyer:
                # Try to extract from contract description (first word)
                parts = contract_desc.split()
                if parts:
                    underlyer = parts[0]
            
            # Get strike from position data (more reliable than parsing)
            strike = float(position.get('strike', 0))
            
            # Get expiry from position data
            expiry_str = position.get('expiry')
            if expiry_str:
                # IBKR format: YYYYMMDD
                expiry = datetime.strptime(expiry_str, '%Y%m%d').date()
            else:
                logger.warning(f"No expiry found for position: {contract_desc}")
                return None
            
            # Get quantity (negative for short positions)
            quantity = float(position.get('position', 0))
            
            # Get conid
            conid = position.get('conid')
            
            # Get multiplier
            multiplier = int(position.get('multiplier', 100))
            
            # Get currency
            currency = position.get('currency', 'USD')
            
            # Create appropriate subclass
            if option_type == 'PUT':
                from utils.put import Put
                option = Put(
                    underlyer=underlyer,
                    strike=strike,
                    expiry=expiry,
                    quantity=quantity,
                    conid=conid,
                    multiplier=multiplier,
                    currency=currency
                )
            else:
                option = cls(
                    underlyer=underlyer,
                    strike=strike,
                    expiry=expiry,
                    option_type=option_type,
                    quantity=quantity,
                    conid=conid,
                    multiplier=multiplier,
                    currency=currency
                )
            
            # Add market data from position if available
            option.option_book.update(
                bid=position.get('bid'),
                ask=position.get('ask'),
                last=position.get('mktPrice')
            )
            
            logger.info(f"✓ Created {option_type} option: {option}")
            
            return option
            
        except Exception as e:
            logger.error(f"Error creating Option from IBKR position: {e}", exc_info=True)
            logger.error(f"Position data: {position}")
            return None
    
    @property
    def days_to_expiry(self) -> int:
        """Calculate days until expiration."""
        today = date.today()
        delta = self.expiry - today
        return delta.days
    
    @property
    def is_expired(self) -> bool:
        """Check if option is expired."""
        return self.days_to_expiry < 0
    
    @property
    def is_short(self) -> bool:
        """Check if this is a short position."""
        return self.quantity < 0
    
    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.quantity > 0
    
    @property
    def notional_value(self) -> float:
        """
        Calculate notional value of the position.
        Uses mid price if available, otherwise last price.
        """
        price = self.option_book.best_price
        if price is None:
            return 0.0
        return abs(self.quantity) * price * self.multiplier
    
    @property
    def symbol(self) -> str:
        """
        Generate option symbol in standard format.
        Format: UNDERLYER YYMMDD C/P STRIKE
        Example: AAPL 241220 P 150.00
        """
        expiry_str = self.expiry.strftime('%y%m%d')
        option_char = 'C' if self.option_type == 'CALL' else 'P'
        return f"{self.underlyer} {expiry_str} {option_char} {self.strike:.2f}"
    
    def __str__(self) -> str:
        """String representation."""
        qty_str = f"{self.quantity:+.0f}" if self.quantity != 0 else "0"
        return f"{qty_str} {self.symbol}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"Option(underlyer='{self.underlyer}', strike={self.strike}, "
                f"expiry={self.expiry}, type={self.option_type}, quantity={self.quantity})")