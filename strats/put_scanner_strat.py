"""
High IVR Put Scanner Strategy

Scans indices for high IVR put selling opportunities and ranks them by IVR.
Focuses on finding the best premium collection opportunities based on volatility rank.
"""

import os
import sys
import logging
import configparser
from datetime import date, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass
import time


# Add vol directory to path - MUST BE FIRST
vol_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Remove the strats directory if it's in the path
strats_dir = os.path.dirname(os.path.abspath(__file__))
if strats_dir in sys.path:
    sys.path.remove(strats_dir)

# Insert vol_dir at the very beginning
if vol_dir in sys.path:
    sys.path.remove(vol_dir)
sys.path.insert(0, vol_dir)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from ibkr.ibkr import IBWebAPIClient
from utils.put_scanner import PutScanner, PutScanResult
from utils.scanner_config import ScannerConfig  # ADD this line

class PutScannerStrategy:
    """
    Strategy to find and rank high IVR put selling opportunities.
    """
    
    def __init__(self, config_file: str = "strats/put_scanner.conf", ibkr_config: str = "ibkr/ibkr.conf"):
        """
        Initialize the put scanner strategy.
        
        Args:
            config_file: Path to scanner configuration file
            ibkr_config: Path to IBKR configuration file
        """
        self.config_file = config_file
        self.ibkr_config = ibkr_config
        self.scan_config = None
        self.client = None
        self.scanner = None
        
    def load_config(self) -> bool:
        """
        Load scanner configuration.
        
        Returns:
            True if config loaded successfully
        """
        try:
            self.scan_config = ScannerConfig.from_file(self.config_file)
            logger.info("✓ Configuration loaded")
            logger.info(str(self.scan_config))
            return True
        except Exception as e:
            logger.error(f"Failed to load config: {e}", exc_info=True)
            return False
        
    def connect(self) -> bool:
        """
        Connect to IBKR.
        
        Returns:
            True if connection successful
        """
        try:
            logger.info("Connecting to IBKR...")
            self.client = IBWebAPIClient(self.ibkr_config)
            
            # Check authentication
            if not self.client.authenticate():
                logger.error("Failed to authenticate with IBKR")
                logger.error("Please ensure Client Portal Gateway is running and you are logged in")
                return False
            
            # Setup account
            if not self.client.setup_account():
                logger.error("Failed to setup account")
                return False
            
            logger.info(f"✓ Connected to IBKR (Account: {self.client.account_id})")
            
            # Initialize scanner
            self.scanner = PutScanner(self.client, self.scan_config)
            logger.info("✓ Scanner initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}", exc_info=True)
            return False
    
    def _apply_filters(self, result: PutScanResult) -> bool:
        """
        Apply additional filters to scan result.
        
        Args:
            result: PutScanResult to filter
            
        Returns:
            True if result passes all filters
        """
        # Min premium filter
        if self.scan_config.min_premium is not None:
            if result.mid is None or result.mid < self.scan_config.min_premium:
                logger.debug(f"  Filtered out {result.symbol}: Premium ${result.mid} < ${self.scan_config.min_premium}")
                return False
        
        # Min annualized return filter
        if self.scan_config.min_annualized_return is not None:
            if result.annualized_return is None or result.annualized_return < self.scan_config.min_annualized_return:
                ann_ret_str = f"{result.annualized_return:.1f}%" if result.annualized_return else "N/A"
                logger.debug(f"  Filtered out {result.symbol}: Ann. Return {ann_ret_str} < {self.scan_config.min_annualized_return:.1f}%")
                return False
        
        # TODO: Add volume filter when we get volume data
        # if self.scan_config.min_volume is not None:
        #     if result.volume < self.scan_config.min_volume:
        #         return False
        
        # TODO: Add 200d MA filter when we get historical data
        # if self.scan_config.check_200d_mavg:
        #     if result.current_price < result.ma_200d:
        #         return False
        
        return True
    
    def scan_index(self, index: str) -> List[PutScanResult]:
        """
        Scan an index for high IVR put opportunities.
        
        Args:
            index: Index to scan
            
        Returns:
            List of PutScanResult objects ranked by annualized return
        """
        if not self.scanner or not self.scan_config:
            logger.error("Scanner not initialized. Call connect() and load_config() first.")
            return []
        
        logger.info("=" * 100)
        # NEW: Show test mode in header
        if self.scan_config.test_symbol:
            logger.info(f"TEST MODE: SCANNING {self.scan_config.test_symbol}")
        else:
            logger.info(f"SCANNING {index.upper()} FOR HIGH IVR PUT OPPORTUNITIES")
        logger.info("=" * 100)
        
        # Calculate expiry date
        expiry_date = None
        if self.scan_config.max_days_to_expiry is not None:
            expiry_date = date.today() + timedelta(days=self.scan_config.max_days_to_expiry)
        
        # Scan the index using the scanner's scan method
        results = self.scanner.scan(
            index=index,
            strike_pct_below=self.scan_config.strike_pct_below,
            min_ivr=self.scan_config.min_ivr,
            max_delta=self.scan_config.max_delta,
            expiry_date=expiry_date,
            max_symbols=self.scan_config.max_symbols,
            min_premium=self.scan_config.min_premium,
            min_annualized_return=self.scan_config.min_annualized_return,
            output_file=self.scan_config.output_file,
            test_symbol=self.scan_config.test_symbol  # NEW: pass test_symbol
        )
        
        return results
    
    def scan_all_indices(self) -> Dict[str, List[PutScanResult]]:
        """
        Scan all configured indices.
        
        Returns:
            Dictionary mapping index name to list of results
        """
        all_results = {}
        
        for index in self.scan_config.indices:
            logger.info(f"\n{'='*100}")
            logger.info(f"Scanning {index.upper()}...")
            logger.info('='*100)
            
            results = self.scan_index(index)
            all_results[index] = results
        
        # Print combined top opportunities
        if len(self.scan_config.indices) > 1:
            self._print_combined_results(all_results, self.scan_config.top_n)
        
        return all_results
    
    def _print_summary(self, results: List[PutScanResult]):
        """Print summary statistics for scan results."""
        if not results:
            return
        
        # Calculate statistics
        avg_ivr = sum(r.ivr for r in results if r.ivr) / len([r for r in results if r.ivr])
        max_ivr = max((r.ivr for r in results if r.ivr), default=0)
        min_ivr = min((r.ivr for r in results if r.ivr), default=0)
        
        avg_delta = sum(abs(r.delta) for r in results if r.delta) / len([r for r in results if r.delta])
        
        avg_premium = sum(r.mid for r in results if r.mid) / len([r for r in results if r.mid])
        
        print("\n" + "=" * 100)
        print("SCAN SUMMARY")
        print("=" * 100)
        print(f"Total opportunities found: {len(results)}")
        print(f"IVR Range: {min_ivr:.1f} - {max_ivr:.1f} (avg: {avg_ivr:.1f})")
        print(f"Average Delta: {avg_delta:.3f}")
        print(f"Average Premium: ${avg_premium:.2f}")
        print("=" * 100)
    
    def _print_combined_results(self, all_results: Dict[str, List[PutScanResult]], top_n: int = 20):
        """Print combined top results from multiple indices."""
        # Flatten all results
        combined = []
        for index, results in all_results.items():
            for result in results:
                combined.append((index, result))
        
        # Sort by IVR
        combined.sort(key=lambda x: x[1].ivr if x[1].ivr else -1, reverse=True)
        
        print("\n" + "=" * 140)
        print(f"TOP {min(top_n, len(combined))} OPPORTUNITIES ACROSS ALL INDICES")
        print("=" * 140)
        print(f"{'Index':<6} | {'Symbol':<6} | {'Price':<8} | {'Strike':<8} | {'OTM%':<6} | "
              f"{'DTE':<4} | {'IVR':<5} | {'IV':<7} | {'Delta':<7} | {'Mid':<8} | {'Ann.Ret':<8}")
        print("-" * 140)
        
        for i, (index, result) in enumerate(combined[:top_n]):
            ivr_str = f"{result.ivr:.1f}" if result.ivr is not None else "N/A"
            iv_str = f"{result.implied_vol:.1%}" if result.implied_vol is not None else "N/A"
            delta_str = f"{result.delta:.3f}" if result.delta is not None else "N/A"
            mid_str = f"${result.mid:.2f}" if result.mid is not None else "N/A"
            ann_ret_str = f"{result.annualized_return:.1f}%" if result.annualized_return is not None else "N/A"
            
            print(f"{index:<6} | {result.symbol:<6} | ${result.current_price:>6.2f} | "
                  f"${result.strike:>6.2f} | {result.strike_pct_otm:>5.1f}% | "
                  f"{result.days_to_expiry:>3} | {ivr_str:>5} | {iv_str:>7} | "
                  f"{delta_str:>7} | {mid_str:>8} | {ann_ret_str:>8}")
        
        print("=" * 140)
    
    def run(self):
        """Main execution method."""
        try:
            # Load configuration
            if not self.load_config():
                logger.error("Failed to load configuration")
                return
            
            # Connect to IBKR
            if not self.connect():
                logger.error("Failed to connect to IBKR")
                return
            
            # Scan all configured indices
            logger.info("\n" + "="*100)
            logger.info("PUT SCANNER STRATEGY")
            logger.info("="*100)
            
            all_results = self.scan_all_indices()
            
            # Print final summary
            total_opportunities = sum(len(results) for results in all_results.values())
            if total_opportunities > 0:
                logger.info(f"\n✓ Strategy complete. Found {total_opportunities} total opportunities.")
                
                # Show top 5 across all indices
                all_combined = []
                for index, results in all_results.items():
                    for result in results:
                        all_combined.append((index, result))
                all_combined.sort(key=lambda x: x[1].ivr if x[1].ivr else -1, reverse=True)
                
                # Print top opportunities
                logger.info("\nTop 5 opportunities by IVR:")
                for i, (index, result) in enumerate(all_combined[:5], 1):  # Unpack the tuple here
                    # Handle None values in formatting
                    ivr_str = f"{result.ivr:.1f}%" if result.ivr is not None else "N/A"
                    iv_str = f"{result.implied_vol:.1%}" if result.implied_vol is not None else "N/A"
                    delta_str = f"{abs(result.delta):.3f}" if result.delta is not None else "N/A"
                    mid_str = f"${result.mid:.2f}" if result.mid is not None else "N/A"
                    
                    logger.info(
                        f"  {i}. [{index}] {result.symbol} ${result.strike:.0f}P - "
                        f"IVR: {ivr_str}, IV: {iv_str}, Δ: {delta_str}, "
                        f"Premium: {mid_str}, DTE: {result.days_to_expiry}"
                    )
            else:
                logger.warning("No opportunities found matching criteria")
            
        except KeyboardInterrupt:
            logger.info("\nStrategy interrupted by user")
        except Exception as e:
            logger.error(f"Strategy error: {e}", exc_info=True)
        finally:
            logger.info("Strategy execution complete")


def main():
    """
    Main entry point for put scanner strategy.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python put_scanner_strat.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Load configuration
    config = ScannerConfig.from_file(config_file)
    
    logger.info("PUT SCANNER")
    logger.info("=" * 80)
    
    # Connect to IBKR
    client = IBWebAPIClient('ibkr/ibkr.conf')
    
    if not client.authenticate():
        logger.error("Failed to authenticate to IBKR")
        sys.exit(1)
    
    if not client.setup_account():
        logger.error("Failed to setup account")
        sys.exit(1)
    
    logger.info(f"Connected to IBKR (Account: {client.account_id})")
    
    # Initialize scanner
    scanner = PutScanner(client, config)
    
    # Determine expiry date
    expiry_date = None
    if config.max_days_to_expiry:  # Changed from config.days_to_expiry
        expiry_date = date.today() + timedelta(days=config.max_days_to_expiry)
    
    # Scan each index
    all_results = []
    
    for index in config.indices:
        logger.info(f"\nScanning {index}...")
        logger.info("-" * 80)
        
        results = scanner.scan(
            index=index,
            strike_pct_below=config.strike_pct_below,
            min_ivr=config.min_ivr,
            max_delta=config.max_delta,
            expiry_date=expiry_date,
            max_symbols=config.max_symbols,
            min_premium=config.min_premium,
            min_annualized_return=config.min_annualized_return,
            max_days_to_expiry=config.max_days_to_expiry,
            output_file=config.output_file,
            test_symbols=config.test_symbols  # CHANGED from test_symbol
        )
        
        all_results.extend(results)
    
    # Display top results
    if all_results:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TOP {config.top_n} OPPORTUNITIES (Sorted by Annualized Return)")
        logger.info(f"{'=' * 80}")
        logger.info(f"{'Symbol':<6} {'Expiry':<10} {'Strike':<7} {'Delta':<7} {'IVR':<6} {'AnnRet':<7} {'Premium':<8}")
        logger.info("-" * 80)
        
        # Sort and display top N
        top_results = all_results[:config.top_n]
        
        for result in top_results:
            ivr_str = f"{result.ivr:.1f}%" if result.ivr else "N/A"
            ann_ret_str = f"{result.annualized_return:.1f}%" if result.annualized_return else "N/A"
            delta_str = f"{abs(result.delta):.3f}" if result.delta else "N/A"
            premium_str = f"${result.mid:.2f}" if result.mid else "N/A"
            
            logger.info(
                f"{result.symbol:<6} {result.expiry.strftime('%Y-%m-%d'):<10} "
                f"${result.strike:<6.0f} {delta_str:<7} {ivr_str:<6} {ann_ret_str:<7} {premium_str:<8}"
            )
        
        logger.info("=" * 80)
    else:
        logger.info("\nNo opportunities found matching criteria")


if __name__ == "__main__":
    main()