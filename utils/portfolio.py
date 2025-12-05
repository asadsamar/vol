"""
Portfolio management with option position tracking.
"""
from typing import List, Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class OptionPortfolio:
    """
    Manages a portfolio of option positions.
    """
    
    def __init__(self):
        """Initialize empty portfolio."""
        self.options: List['Option'] = []  # Use forward reference
    
    def add_option(self, option: 'Option'):
        """Add an option to the portfolio."""
        self.options.append(option)
    
    def load_from_ibkr_positions(self, positions: List[Dict]):
        """
        Load portfolio from IBKR position data.
        
        Args:
            positions: List of IBKR position dictionaries
        """
        # Import here to avoid circular import
        from utils.option import Option
        
        self.options = []
        
        for pos in positions:
            option = Option.from_ibkr_position(pos)
            if option:
                self.options.append(option)
                logger.debug(f"Loaded option: {option}")
    
    @property
    def short_puts(self) -> List['Put']:
        """Get all short put positions."""
        from utils.put import Put
        return [opt for opt in self.options 
                if isinstance(opt, Put) and opt.is_short]
    
    @property
    def long_puts(self) -> List['Put']:
        """Get all long put positions."""
        from utils.put import Put
        return [opt for opt in self.options 
                if isinstance(opt, Put) and opt.is_long]
    
    @property
    def all_puts(self) -> List['Put']:
        """Get all put positions."""
        from utils.put import Put
        return [opt for opt in self.options if isinstance(opt, Put)]
    
    def get_positions_by_underlyer(self) -> Dict[str, List['Option']]:
        """Group positions by underlying symbol."""
        by_underlyer = defaultdict(list)
        for opt in self.options:
            by_underlyer[opt.underlyer].append(opt)
        return dict(by_underlyer)
    
    def get_total_exercise_risk(self) -> float:
        """
        Calculate total cash required if all short puts are exercised.
        
        Returns:
            Total cash requirement
        """
        total = 0.0
        for put in self.short_puts:
            total += put.exercise_cash_requirement
        return total
    
    def get_total_margin_requirement(self) -> float:
        """
        Calculate total estimated margin requirement for all short options.
        
        Returns:
            Total margin requirement
        """
        total = 0.0
        for opt in self.options:
            if hasattr(opt, 'margin_requirement'):
                total += opt.margin_requirement
        return total
    
    def get_expiry_timeline(self) -> List[Dict]:
        """
        Get cash requirement timeline by expiry date.
        
        Returns:
            List of expiry data sorted by date
        """
        expiry_data = defaultdict(lambda: {
            'expiry': None,
            'days': 0,
            'positions': [],
            'cash_requirement': 0.0
        })
        
        for put in self.short_puts:
            expiry_key = put.expiry.isoformat()
            
            if expiry_data[expiry_key]['expiry'] is None:
                expiry_data[expiry_key]['expiry'] = put.expiry
                expiry_data[expiry_key]['days'] = put.days_to_expiry
            
            expiry_data[expiry_key]['positions'].append(put)
            expiry_data[expiry_key]['cash_requirement'] += put.exercise_cash_requirement
        
        # Convert to sorted list
        timeline = sorted(expiry_data.values(), key=lambda x: x['expiry'])
        
        # Add cumulative
        cumulative = 0.0
        for item in timeline:
            cumulative += item['cash_requirement']
            item['cumulative'] = cumulative
        
        return timeline
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get comprehensive portfolio summary.
        
        Returns:
            Dictionary with portfolio metrics
        """
        short_puts = self.short_puts
        long_puts = self.long_puts
        
        return {
            'total_positions': len(self.options),
            'total_puts': len(self.all_puts),
            'short_puts': len(short_puts),
            'long_puts': len(long_puts),
            'underlyers': len(self.get_positions_by_underlyer()),
            'total_exercise_risk': self.get_total_exercise_risk(),
            'total_margin_requirement': self.get_total_margin_requirement(),
            'total_notional': sum(opt.notional_value for opt in self.options),
        }
    
    def print_summary(self):
        """Print portfolio summary."""
        summary = self.get_portfolio_summary()
        
        print("\n" + "=" * 80)
        print("OPTION PORTFOLIO SUMMARY")
        print("=" * 80)
        print(f"Total Positions:        {summary['total_positions']}")
        print(f"  - Short Puts:         {summary['short_puts']}")
        print(f"  - Long Puts:          {summary['long_puts']}")
        print(f"Unique Underlyers:      {summary['underlyers']}")
        print(f"\nRisk Metrics:")
        print(f"  Exercise Risk:        ${summary['total_exercise_risk']:,.2f}")
        print(f"  Margin Requirement:   ${summary['total_margin_requirement']:,.2f}")
        print(f"  Total Notional:       ${summary['total_notional']:,.2f}")
        print("=" * 80)
    
    def __len__(self) -> int:
        """Return number of positions."""
        return len(self.options)
    
    def __iter__(self):
        """Iterate over options."""
        return iter(self.options)