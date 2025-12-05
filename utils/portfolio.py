"""
Portfolio management with option position tracking.
"""
from typing import List, Dict, Optional, Tuple
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
        
        # Calculate risk-adjusted exercise requirements
        total_exercise_risk = 0
        total_max_exercise = 0
        positions_with_delta = 0
        
        # Calculate portfolio-level SPX delta
        # Note: IBKR sends position-level SPX delta (already multiplied by quantity)
        portfolio_spx_delta = 0
        positions_with_spx_delta = 0
        
        for opt in self.options:
            # Max exercise requirement (100% probability)
            max_exercise = abs(opt.quantity) * opt.strike * opt.multiplier
            total_max_exercise += max_exercise
            
            # Risk-adjusted exercise requirement
            risk_amount = opt.get_exercise_risk_amount()
            if risk_amount is not None:
                total_exercise_risk += risk_amount
                positions_with_delta += 1
            
            # Portfolio SPX delta - IBKR already accounts for position quantity
            # Just sum them up directly
            if opt.spx_delta is not None:
                portfolio_spx_delta += opt.spx_delta
                positions_with_spx_delta += 1
        
        # Group by underlyer
        by_underlyer = {}
        for opt in self.options:
            if opt.underlyer not in by_underlyer:
                by_underlyer[opt.underlyer] = {
                    'positions': 0,
                    'notional': 0,
                    'max_exercise': 0,
                    'risk_adjusted_exercise': 0,
                    'avg_delta': [],
                    'portfolio_spx_delta': 0
                }
            
            by_underlyer[opt.underlyer]['positions'] += 1
            by_underlyer[opt.underlyer]['notional'] += abs(opt.quantity) * opt.strike * opt.multiplier
            by_underlyer[opt.underlyer]['max_exercise'] += abs(opt.quantity) * opt.strike * opt.multiplier
            
            risk_amount = opt.get_exercise_risk_amount()
            if risk_amount is not None:
                by_underlyer[opt.underlyer]['risk_adjusted_exercise'] += risk_amount
                if opt.delta is not None:
                    by_underlyer[opt.underlyer]['avg_delta'].append(abs(opt.delta))
            
            # Add to underlyer SPX delta - already position-level from IBKR
            if opt.spx_delta is not None:
                by_underlyer[opt.underlyer]['portfolio_spx_delta'] += opt.spx_delta
        
        # Calculate average deltas per underlyer
        for underlyer_data in by_underlyer.values():
            if underlyer_data['avg_delta']:
                underlyer_data['avg_delta'] = sum(underlyer_data['avg_delta']) / len(underlyer_data['avg_delta'])
            else:
                underlyer_data['avg_delta'] = None
        
        return {
            'total_positions': len(self.options),
            'total_puts': len(self.all_puts),
            'short_puts': len(short_puts),
            'long_puts': len(long_puts),
            'underlyers': len(self.get_positions_by_underlyer()),
            'total_exercise_risk': self.get_total_exercise_risk(),
            'total_margin_requirement': self.get_total_margin_requirement(),
            'total_notional': sum(opt.notional_value for opt in self.options),
            'total_max_exercise': total_max_exercise,
            'total_risk_adjusted_exercise': total_exercise_risk,
            'positions_with_delta': positions_with_delta,
            'positions_with_spx_delta': positions_with_spx_delta,
            'risk_percentage': (total_exercise_risk / total_max_exercise * 100) if total_max_exercise > 0 else 0,
            'portfolio_spx_delta': portfolio_spx_delta,
            'by_underlyer': by_underlyer,
        }
    
    def print_summary(self):
        """Print portfolio summary with risk-adjusted exercise requirements."""
        summary = self.get_portfolio_summary()
        
        print("\n" + "=" * 100)
        print("OPTION PORTFOLIO SUMMARY - RISK-ADJUSTED EXERCISE ANALYSIS")
        print("=" * 100)
        
        print(f"\n📊 POSITIONS:")
        print(f"  Total Positions:        {summary['total_positions']}")
        print(f"  - Short Puts:           {summary['short_puts']}")
        print(f"  - Long Puts:            {summary['long_puts']}")
        print(f"  Unique Underlyers:      {summary['underlyers']}")
        print(f"  Positions w/ Delta:     {summary['positions_with_delta']} / {summary['total_positions']}")
        
        print(f"\n💰 EXERCISE RISK ANALYSIS:")
        print(f"  Max Exercise Req:       ${summary['total_max_exercise']:>15,.2f}  (100% probability)")
        print(f"  Risk-Adj Exercise:      ${summary['total_risk_adjusted_exercise']:>15,.2f}  ({summary['risk_percentage']:.1f}% of max)")
        print(f"  Traditional Risk:       ${summary['total_exercise_risk']:>15,.2f}  (legacy calc)")
        
        print(f"\n📈 OTHER METRICS:")
        print(f"  Margin Requirement:     ${summary['total_margin_requirement']:>15,.2f}")
        print(f"  Total Notional:         ${summary['total_notional']:>15,.2f}")
        
        # Show by underlyer
        if summary['by_underlyer']:
            print(f"\n📋 BREAKDOWN BY UNDERLYER:")
            print(f"  {'Symbol':<10} {'Positions':<10} {'Notional':<15} {'Avg Delta':<12} {'Risk-Adj Req':<15}")
            print(f"  {'-'*10} {'-'*10} {'-'*15} {'-'*12} {'-'*15}")
            
            for underlyer, data in sorted(summary['by_underlyer'].items()):
                avg_delta_str = f"{data['avg_delta']:.3f}" if data['avg_delta'] is not None else "N/A"
                print(f"  {underlyer:<10} {data['positions']:<10} ${data['notional']:>13,.2f} {avg_delta_str:<12} ${data['risk_adjusted_exercise']:>13,.2f}")
        
        print("\n" + "=" * 100)
    
    def print_one_line_summary(self, cash_info: Tuple[float, float, float]):
        """
        Print a single-line portfolio summary.
        
        Args:
            cash_info: Tuple of (settled_cash, buying_power, net_liquidation)
        """
        try:
            settled_cash, buying_power, net_liq = cash_info
            summary = self.get_portfolio_summary()
            
            # Calculate coverage ratio
            coverage_ratio = 0
            if summary['total_risk_adjusted_exercise'] > 0:
                coverage_ratio = settled_cash / summary['total_risk_adjusted_exercise']
            
            # Build status indicator
            if coverage_ratio < 1.0:
                status = "⚠️  LOW"
            elif coverage_ratio < 1.5:
                status = "⚡ MED"
            else:
                status = "✅ GOOD"
            
            # Format SPX delta
            spx_delta = summary['portfolio_spx_delta']
            
            # Print single line with key metrics
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp} | Pos: {summary['total_positions']:>3} ({summary['positions_with_delta']:>3} w/Δ) | "
                  f"SPX-Δ: {spx_delta:>8,.2f} ({summary['positions_with_spx_delta']:>2}) | "
                  f"RiskEx: ${summary['total_risk_adjusted_exercise']:>11,.0f} ({summary['risk_percentage']:>4.1f}%) | "
                  f"Cash: ${settled_cash:>11,.0f} | Cov: {coverage_ratio:>4.2f}x {status}")
            
        except Exception as e:
            logger.error(f"Error printing one-line summary: {e}", exc_info=True)
    
    def __len__(self) -> int:
        """Return the number of positions in the portfolio."""
        return len(self.options)