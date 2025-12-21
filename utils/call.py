"""
Call option class with exercise risk calculations.
"""
from typing import Optional
from datetime import date
import logging
from utils.option import Option

logger = logging.getLogger(__name__)


class Call(Option):
    """
    Call option with delivery calculations for short positions.
    
    For short call positions, calculates the stock delivery requirement if exercised
    (i.e., forced to sell the underlying at strike price).
    """
    
    def __init__(
        self,
        underlyer: str,
        strike: float,
        expiry: date,
        quantity: float = 0.0,
        conid: Optional[int] = None,
        multiplier: int = 100,
        currency: str = 'USD'
    ):
        """
        Initialize a call option.
        
        Args:
            underlyer: Underlying symbol
            strike: Strike price
            expiry: Expiration date
            quantity: Position size (negative for short, positive for long)
            conid: IBKR contract ID
            multiplier: Contract multiplier
            currency: Currency
        """
        super().__init__(
            underlyer=underlyer,
            strike=strike,
            expiry=expiry,
            option_type='CALL',
            quantity=quantity,
            conid=conid,
            multiplier=multiplier,
            currency=currency
        )
    
    @property
    def exercise_stock_requirement(self) -> float:
        """
        Calculate stock shares required if call is exercised (for short positions).
        
        When you're short a call and it gets exercised:
        - You are OBLIGATED to SELL the underlying at the strike price
        - Stock requirement = Multiplier × |Quantity|
        
        For long calls, there's no stock requirement (you have the right, not obligation).
        
        Returns:
            Shares required if exercised (0 for long positions)
        """
        if self.is_long:
            # Long calls have no stock requirement (you have a right, not obligation)
            return 0.0
        
        # Short calls: must deliver stock at strike if exercised
        # Requirement = multiplier × number of contracts
        return self.multiplier * abs(self.quantity)
    
    @property
    def margin_requirement(self) -> float:
        """
        Estimate margin requirement for naked short call.
        
        IBKR's margin requirement for naked short calls is approximately:
        - Greater of:
          1. 20% of underlying value + out-of-money amount (if any)
          2. 10% of underlying value
        
        This is a simplified calculation. Actual IBKR margin may vary.
        
        Returns:
            Estimated margin requirement
        """
        if self.is_long:
            # Long calls require only the premium paid (which is already debited)
            return 0.0
        
        if not self.is_short:
            return 0.0
        
        # Get underlying price (use best available price or strike as fallback)
        underlying_price = self.underlyer_book.best_price or self.strike
        
        # Calculate how out-of-money the call is
        # Call is OTM when strike > underlying price
        otm_amount = max(0, self.strike - underlying_price)
        
        # Method 1: 20% of underlying + OTM amount
        method1 = (0.20 * underlying_price + otm_amount) * self.multiplier * abs(self.quantity)
        
        # Method 2: 10% of underlying value
        method2 = 0.10 * underlying_price * self.multiplier * abs(self.quantity)
        
        # Return the greater of the two
        return max(method1, method2)
    
    @property
    def is_itm(self) -> bool:
        """
        Check if call is in-the-money.
        Call is ITM when strike < underlying price.
        """
        underlying_price = self.underlyer_book.best_price
        if underlying_price is None:
            return False
        return self.strike < underlying_price
    
    @property
    def is_otm(self) -> bool:
        """
        Check if call is out-of-the-money.
        Call is OTM when strike > underlying price.
        """
        underlying_price = self.underlyer_book.best_price
        if underlying_price is None:
            return False
        return self.strike > underlying_price
    
    @property
    def is_atm(self) -> bool:
        """
        Check if call is at-the-money.
        Call is ATM when strike ≈ underlying price (within 2%).
        """
        underlying_price = self.underlyer_book.best_price
        if underlying_price is None:
            return False
        
        pct_diff = abs(self.strike - underlying_price) / underlying_price
        return pct_diff < 0.02  # Within 2%
    
    @property
    def intrinsic_value(self) -> float:
        """
        Calculate intrinsic value.
        For calls: max(0, Underlying Price - Strike)
        """
        underlying_price = self.underlyer_book.best_price
        if underlying_price is None:
            return 0.0
        return max(0, underlying_price - self.strike)
    
    @property
    def extrinsic_value(self) -> Optional[float]:
        """
        Calculate extrinsic (time) value.
        Extrinsic = Option Price - Intrinsic Value
        """
        option_price = self.option_book.best_price
        if option_price is None:
            return None
        return option_price - self.intrinsic_value
    
    @property
    def moneyness_description(self) -> str:
        """Get human-readable moneyness description."""
        if self.is_itm:
            return "ITM"
        elif self.is_otm:
            return "OTM"
        elif self.is_atm:
            return "ATM"
        return "Unknown"
    
    def get_risk_summary(self) -> dict:
        """
        Get comprehensive risk summary for this call position.
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            'symbol': self.symbol,
            'underlyer': self.underlyer,
            'strike': self.strike,
            'expiry': self.expiry,
            'days_to_expiry': self.days_to_expiry,
            'quantity': self.quantity,
            'is_short': self.is_short,
            'moneyness': self.moneyness_description,
            'exercise_stock_requirement': self.exercise_stock_requirement,
            'margin_requirement': self.margin_requirement,
            'notional_value': self.notional_value,
            'intrinsic_value': self.intrinsic_value,
            'extrinsic_value': self.extrinsic_value,
            'option_bid': self.option_book.bid,
            'option_ask': self.option_book.ask,
            'option_mid': self.option_book.mid,
            'underlyer_price': self.underlyer_book.best_price,
        }
    
    def __str__(self) -> str:
        """String representation with moneyness."""
        qty_str = f"{self.quantity:+.0f}" if self.quantity != 0 else "0"
        moneyness = self.moneyness_description
        return f"{qty_str} {self.symbol} [{moneyness}]"