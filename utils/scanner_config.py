"""
Configuration for put scanner strategy.
"""

import configparser
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ScannerConfig:
    """Configuration for put scanner strategy."""
    indices: List[str]
    min_ivr: float
    strike_pct_below: float
    max_delta: float
    check_200d_mavg: bool
    max_symbols: int = 0  # 0 = no limit
    max_days_to_expiry: Optional[int] = None
    min_premium: Optional[float] = None
    min_volume: Optional[int] = None
    min_annualized_return: Optional[float] = None
    top_n: int = 20
    num_strikes: int = 10
    output_file: str = 'results/put_scan_results.csv'
    test_symbols: Optional[List[str]] = None 
    
    @classmethod
    def from_file(cls, config_file: str) -> 'ScannerConfig':
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            ScannerConfig object
        """
        config = configparser.ConfigParser()
        config.read(config_file)
        
        scan_section = config['SCANNER']
        
        # Parse indices (comma-separated)
        indices_str = scan_section.get('indices', 'XLE')
        indices = [idx.strip().upper() for idx in indices_str.split(',')]
        
        # Required parameters
        min_ivr = scan_section.getfloat('min_ivr', 50.0)
        strike_pct_below = scan_section.getfloat('strike_pct_below', 5.0)
        max_delta = scan_section.getfloat('max_delta', 0.20)
        check_200d_mavg = scan_section.getboolean('check_200d_mavg', False)
        max_symbols = scan_section.getint('max_symbols', 0)
        
        # Optional parameters
        max_days_to_expiry = scan_section.getint('max_days_to_expiry', fallback=None)
        min_premium = scan_section.getfloat('min_premium', fallback=None)
        min_volume = scan_section.getint('min_volume', fallback=None)
        min_annualized_return = scan_section.getfloat('min_annualized_return', fallback=None)
        top_n = scan_section.getint('top_n', 20)
        num_strikes = scan_section.getint('num_strikes', 10)
        output_file = scan_section.get('output_file', 'results/put_scan_results.csv')
        
        # Parse test_symbols (comma-separated)
        test_symbols_str = scan_section.get('test_symbols', fallback=None)
        test_symbols = None
        if test_symbols_str:
            test_symbols = [s.strip().upper() for s in test_symbols_str.split(',')]
        
        return cls(
            indices=indices,
            min_ivr=min_ivr,
            strike_pct_below=strike_pct_below,
            max_delta=max_delta,
            check_200d_mavg=check_200d_mavg,
            max_symbols=max_symbols,
            max_days_to_expiry=max_days_to_expiry,
            min_premium=min_premium,
            min_volume=min_volume,
            min_annualized_return=min_annualized_return,
            top_n=top_n,
            num_strikes=num_strikes,
            output_file=output_file,
            test_symbols=test_symbols
        )
    
    def __str__(self) -> str:
        """String representation of config."""
        max_sym_str = f"{self.max_symbols}" if self.max_symbols > 0 else "unlimited"
        lines = [
            "Scanner Config:",
        ]
        
        # NEW: Show test mode if enabled
        if self.test_symbols:
            lines.append(f"  TEST MODE: Single symbol(s) = {', '.join(self.test_symbols)}")
        else:
            lines.append(f"  Indices: {', '.join(self.indices)}")
            lines.append(f"  Max Symbols: {max_sym_str}")
        
        lines.extend([
            f"  Min IVR: {self.min_ivr:.1f}%",
            f"  Strike: {self.strike_pct_below:.1f}% below current price",
            f"  Max Delta: {self.max_delta:.2f}",
            f"  Check 200D MA: {self.check_200d_mavg}",
            f"  Num Strikes: {self.num_strikes}",
        ])
        
        if self.max_days_to_expiry:
            lines.append(f"  Max Days to Expiry: {self.max_days_to_expiry}")
        
        if self.min_premium:
            lines.append(f"  Min Premium: ${self.min_premium:.2f}")
        if self.min_volume:
            lines.append(f"  Min Volume: {self.min_volume}")
        if self.min_annualized_return:
            lines.append(f"  Min Annualized Return: {self.min_annualized_return:.1f}%")
        
        lines.append(f"  Top N: {self.top_n}")
        lines.append(f"  Output File: {self.output_file}")
        
        return "\n".join(lines)