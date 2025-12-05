"""
Option risk calculations for various strategies.

This module provides functions to calculate cash requirements and risk metrics
for different option strategies.
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PositionRiskManager:
    """
    Manages risk for option positions using portfolio-based analysis.
    """
    
    def __init__(self, settled_cash: float, buying_power: float):
        """
        Initialize the risk manager.
        
        Args:
            settled_cash: Available settled cash
            buying_power: Available buying power
        """
        self.settled_cash = settled_cash
        self.buying_power = buying_power
        self.portfolio = None
        logger.info(f"Risk manager initialized with cash=${settled_cash:,.2f}, BP=${buying_power:,.2f}")
    
    def set_portfolio(self, portfolio):
        """
        Set the portfolio to manage.
        
        Args:
            portfolio: OptionPortfolio instance
        """
        from utils.portfolio import OptionPortfolio
        
        if not isinstance(portfolio, OptionPortfolio):
            raise TypeError("portfolio must be an OptionPortfolio instance")
        
        self.portfolio = portfolio
        total_requirement = portfolio.get_total_exercise_risk()
        logger.info(f"Portfolio set: {len(portfolio)} positions, requirement=${total_requirement:,.2f}")
    
    def update_balances(self, settled_cash: float, buying_power: float):
        """Update account balances."""
        self.settled_cash = settled_cash
        self.buying_power = buying_power
        logger.info(f"Balances updated: cash=${settled_cash:,.2f}, BP=${buying_power:,.2f}")
    
    def get_total_requirement(self) -> float:
        """Get total cash requirement for all positions."""
        if not self.portfolio:
            return 0.0
        return self.portfolio.get_total_exercise_risk()
    
    def get_cash_available(self) -> float:
        """Calculate available cash after all positions."""
        return self.settled_cash - self.get_total_requirement()
    
    def check_new_put_order(self, strike: float, quantity: int, use_cash_limit: bool = True) -> Dict:
        """
        Check if a new short put order is within risk limits.
        
        Args:
            strike: Strike price
            quantity: Number of contracts (positive)
            use_cash_limit: If True, check against settled cash; if False, check against buying power
            
        Returns:
            Dictionary with approval status and details
        """
        from utils.put import Put
        from datetime import date, timedelta
        
        # Calculate requirement for new position
        new_requirement = strike * 100 * abs(quantity)
        
        # Get current total requirement
        current_requirement = self.get_total_requirement()
        total_requirement = current_requirement + new_requirement
        
        # Choose limit based on parameter
        limit = self.settled_cash if use_cash_limit else self.buying_power
        limit_name = "Settled Cash" if use_cash_limit else "Buying Power"
        
        # Check if within limits
        approved = total_requirement <= limit
        available_after = limit - total_requirement
        
        result = {
            'approved': approved,
            'strike': strike,
            'quantity': quantity,
            'new_requirement': new_requirement,
            'current_requirement': current_requirement,
            'total_requirement': total_requirement,
            'limit': limit,
            'limit_name': limit_name,
            'available_after': available_after,
            'utilization_pct': (total_requirement / limit * 100) if limit > 0 else 0
        }
        
        if approved:
            result['message'] = (
                f"✓ ORDER APPROVED\n"
                f"  New Position: {quantity} contracts @ ${strike:.2f}\n"
                f"  New Requirement: ${new_requirement:,.2f}\n"
                f"  Total After Order: ${total_requirement:,.2f}\n"
                f"  {limit_name} Limit: ${limit:,.2f}\n"
                f"  Available After: ${available_after:,.2f}\n"
                f"  Utilization: {result['utilization_pct']:.1f}%"
            )
        else:
            shortfall = total_requirement - limit
            result['shortfall'] = shortfall
            result['message'] = (
                f"✗ ORDER REJECTED\n"
                f"  New Position: {quantity} contracts @ ${strike:.2f}\n"
                f"  New Requirement: ${new_requirement:,.2f}\n"
                f"  Total After Order: ${total_requirement:,.2f}\n"
                f"  {limit_name} Limit: ${limit:,.2f}\n"
                f"  Shortfall: ${shortfall:,.2f}\n"
                f"  Utilization: {result['utilization_pct']:.1f}%\n\n"
                f"  REASON: Insufficient {limit_name.lower()} to cover assignment risk"
            )
        
        return result
    
    def get_max_contracts(self, strike: float, use_cash_limit: bool = True) -> int:
        """
        Calculate maximum number of contracts that can be sold at given strike.
        
        Args:
            strike: Strike price
            use_cash_limit: If True, use settled cash; if False, use buying power
            
        Returns:
            Maximum number of contracts
        """
        current_requirement = self.get_total_requirement()
        limit = self.settled_cash if use_cash_limit else self.buying_power
        available = limit - current_requirement
        
        if available <= 0:
            return 0
        
        requirement_per_contract = strike * 100
        max_contracts = int(available / requirement_per_contract)
        
        return max(0, max_contracts)
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary."""
        current_requirement = self.get_total_requirement()
        cash_available = self.get_cash_available()
        
        return {
            'settled_cash': self.settled_cash,
            'buying_power': self.buying_power,
            'current_requirement': current_requirement,
            'cash_available': cash_available,
            'cash_utilization_pct': (current_requirement / self.settled_cash * 100) if self.settled_cash > 0 else 0,
            'bp_utilization_pct': (current_requirement / self.buying_power * 100) if self.buying_power > 0 else 0,
            'num_positions': len(self.portfolio) if self.portfolio else 0,
        }


def calculate_naked_put_requirement(strike: float, quantity: float, multiplier: float = 100.0) -> float:
    """
    Calculate cash requirement for a naked (cash-secured) put position.
    
    For a naked put, the seller must have enough cash to purchase the underlying
    if the option is exercised.
    
    Args:
        strike: Strike price of the put option
        quantity: Number of contracts (positive number)
        multiplier: Contract multiplier (typically 100 for equity options)
        
    Returns:
        Total cash requirement
    """
    return strike * multiplier * quantity


def calculate_naked_put_requirement_from_position(position: Dict) -> float:
    """
    Calculate cash requirement for a naked put from a position dictionary.
    
    Args:
        position: Position dictionary from IBKR API
        
    Returns:
        Cash requirement
    """
    strike = float(position.get('strike', 0))
    quantity = abs(float(position.get('position', 0)))
    multiplier = float(position.get('multiplier', 100))
    
    return calculate_naked_put_requirement(strike, quantity, multiplier)


def calculate_total_put_requirement(positions: List[Dict]) -> float:
    """
    Calculate total cash requirement for multiple put positions.
    
    Args:
        positions: List of position dictionaries
        
    Returns:
        Total cash requirement across all positions
    """
    total = 0.0
    for position in positions:
        total += calculate_naked_put_requirement_from_position(position)
    return total


def calculate_covered_call_requirement(position: Dict, underlying_shares: int = 0) -> Dict:
    """
    Calculate requirement for a covered call position.
    
    A covered call is secured by owning the underlying shares.
    
    Args:
        position: Position dictionary for the call option
        underlying_shares: Number of shares owned (positive)
        
    Returns:
        Dictionary with requirement details
    """
    contracts = abs(float(position.get('position', 0)))
    multiplier = float(position.get('multiplier', 100))
    shares_needed = contracts * multiplier
    shares_covered = min(underlying_shares, shares_needed)
    shares_uncovered = max(0, shares_needed - underlying_shares)
    
    return {
        'contracts': contracts,
        'shares_needed': shares_needed,
        'shares_covered': shares_covered,
        'shares_uncovered': shares_uncovered,
        'is_fully_covered': shares_uncovered == 0
    }


def calculate_vertical_spread_requirement(long_position: Dict, short_position: Dict) -> Dict:
    """
    Calculate max risk for a vertical spread (e.g., bull put spread, bear call spread).
    
    Args:
        long_position: Position dictionary for the long leg
        short_position: Position dictionary for the short leg
        
    Returns:
        Dictionary with spread risk metrics
    """
    long_strike = float(long_position.get('strike', 0))
    short_strike = float(short_position.get('strike', 0))
    quantity = abs(float(short_position.get('position', 0)))
    multiplier = float(short_position.get('multiplier', 100))
    
    # Max risk is the difference in strikes
    strike_diff = abs(short_strike - long_strike)
    max_risk = strike_diff * multiplier * quantity
    
    return {
        'long_strike': long_strike,
        'short_strike': short_strike,
        'strike_width': strike_diff,
        'contracts': quantity,
        'max_risk': max_risk
    }


def calculate_iron_condor_requirement(positions: List[Dict]) -> Dict:
    """
    Calculate max risk for an iron condor (4 legs: 2 vertical spreads).
    
    Args:
        positions: List of 4 position dictionaries (bull put + bear call)
        
    Returns:
        Dictionary with iron condor risk metrics
    """
    if len(positions) != 4:
        raise ValueError("Iron condor requires exactly 4 positions")
    
    # Sort positions by strike
    sorted_positions = sorted(positions, key=lambda x: float(x.get('strike', 0)))
    
    # Identify the two spreads
    # Typically: long put, short put, short call, long call (by strike)
    put_spread_max = abs(float(sorted_positions[1].get('strike', 0)) - 
                         float(sorted_positions[0].get('strike', 0)))
    call_spread_max = abs(float(sorted_positions[3].get('strike', 0)) - 
                          float(sorted_positions[2].get('strike', 0)))
    
    multiplier = float(sorted_positions[0].get('multiplier', 100))
    quantity = abs(float(sorted_positions[1].get('position', 0)))
    
    # Max risk is the larger of the two spread widths
    max_risk = max(put_spread_max, call_spread_max) * multiplier * quantity
    
    return {
        'put_spread_width': put_spread_max,
        'call_spread_width': call_spread_max,
        'max_risk_per_contract': max(put_spread_max, call_spread_max) * multiplier,
        'contracts': quantity,
        'total_max_risk': max_risk
    }


def calculate_portfolio_margin_requirement(positions: List[Dict]) -> Dict:
    """
    Estimate portfolio margin requirement (simplified calculation).
    
    Note: Actual portfolio margin calculations by brokers are complex and
    scenario-based. This is a simplified approximation.
    
    Args:
        positions: List of all option positions
        
    Returns:
        Dictionary with estimated margin requirements
    """
    # Simplified: assume 15% of notional value for naked options
    total_notional = 0.0
    
    for position in positions:
        strike = float(position.get('strike', 0))
        quantity = abs(float(position.get('position', 0)))
        multiplier = float(position.get('multiplier', 100))
        notional = strike * multiplier * quantity
        total_notional += notional
    
    estimated_margin = total_notional * 0.15  # 15% approximation
    
    return {
        'total_notional': total_notional,
        'estimated_margin_requirement': estimated_margin,
        'margin_percentage': 0.15
    }


def calculate_buying_power_effect(position: Dict, strategy_type: str = 'naked_put') -> float:
    """
    Calculate the buying power effect of a position.
    
    Args:
        position: Position dictionary
        strategy_type: Type of strategy ('naked_put', 'covered_call', etc.)
        
    Returns:
        Buying power reduction amount
    """
    if strategy_type == 'naked_put':
        return calculate_naked_put_requirement_from_position(position)
    elif strategy_type == 'covered_call':
        # Covered calls typically don't require additional buying power
        return 0.0
    else:
        # Default to naked put calculation for conservative estimate
        return calculate_naked_put_requirement_from_position(position)


def analyze_risk_coverage(cash_available: float, buying_power: float, 
                          total_requirement: float) -> Dict:
    """
    Analyze whether available funds cover the risk requirement.
    
    Args:
        cash_available: Settled cash available
        buying_power: Total buying power available
        total_requirement: Total cash requirement for all positions
        
    Returns:
        Dictionary with coverage analysis
    """
    cash_coverage_pct = (cash_available / total_requirement * 100) if total_requirement > 0 else 0
    bp_coverage_pct = (buying_power / total_requirement * 100) if total_requirement > 0 else 0
    
    if cash_available >= total_requirement:
        status = 'FULLY_SECURED'
        risk_level = 'LOW'
        excess_cash = cash_available - total_requirement
        shortfall = 0.0
    elif buying_power >= total_requirement:
        status = 'MARGIN_SECURED'
        risk_level = 'MEDIUM'
        excess_cash = 0.0
        shortfall = total_requirement - cash_available
    else:
        status = 'INSUFFICIENT'
        risk_level = 'HIGH'
        excess_cash = 0.0
        shortfall = total_requirement - buying_power
    
    return {
        'status': status,
        'risk_level': risk_level,
        'cash_available': cash_available,
        'buying_power': buying_power,
        'total_requirement': total_requirement,
        'cash_coverage_pct': cash_coverage_pct,
        'buying_power_coverage_pct': bp_coverage_pct,
        'excess_cash': excess_cash,
        'shortfall': shortfall
    }