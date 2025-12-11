"""
Put option scanner for finding high IVR opportunities across indices.
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
import logging
from dataclasses import dataclass
import time
import configparser
import csv
from pathlib import Path

from utils.index_constituents import IndexConstituents

logger = logging.getLogger(__name__)


@dataclass
class PutScanResult:
    """Result from put scanner."""
    symbol: str
    stock_price: float
    strike: float
    expiry: date
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    vega: Optional[float]
    theta: Optional[float]
    implied_vol: Optional[float]
    ivr: Optional[float]
    volume: Optional[float]
    
    @property
    def current_price(self) -> float:
        """Alias for stock_price to match usage."""
        return self.stock_price
    
    @property
    def days_to_expiry(self) -> int:
        """Calculate days to expiry."""
        return (self.expiry - date.today()).days
    
    @property
    def strike_pct_otm(self) -> float:
        """Calculate percentage strike is OTM."""
        return ((self.stock_price - self.strike) / self.stock_price) * 100
    
    @property
    def annualized_return(self) -> Optional[float]:
        """
        Calculate annualized return for cash-secured put.
        
        Return = (Premium / Strike) * (365 / DTE) * 100
        
        This assumes:
        - Cash-secured put (collateral = strike price)
        - Premium is profit if not assigned
        - Annualized for comparison across different DTEs
        """
        if self.mid is None or self.strike is None:
            return None
        
        dte = self.days_to_expiry
        if dte <= 0:
            return None
        
        # Return on collateral (strike price)
        return_pct = (self.mid / self.strike) * 100
        
        # Annualize
        annualized = return_pct * (365 / dte)
        
        return annualized
    
    def __str__(self) -> str:
        """String representation."""
        parts = [
            f"{self.symbol} ${self.strike:.0f}P",
            f"expires {self.expiry.strftime('%Y-%m-%d')} ({self.days_to_expiry}d)"
        ]
        
        if self.bid is not None and self.ask is not None:
            parts.append(f"${self.bid:.2f}/${self.ask:.2f}")
        if self.mid is not None:
            parts.append(f"mid=${self.mid:.2f}")
        if self.delta is not None:
            parts.append(f"Δ={self.delta:.3f}")
        if self.implied_vol is not None:
            parts.append(f"IV={self.implied_vol:.1%}")
        if self.ivr is not None:
            parts.append(f"IVR={self.ivr:.1f}%")
        if self.annualized_return is not None:
            parts.append(f"Ann.Ret={self.annualized_return:.1f}%")
        
        return " | ".join(parts)


class PutScanner:
    """Scanner for finding put selling opportunities."""

    def __init__(self, client: 'IBWebAPIClient', config: Optional[configparser.ConfigParser] = None):
        """
        Initialize the put scanner.
        
        Args:
            client: IBWebAPIClient instance
            config: Optional configuration (for num_strikes setting)
        """
        self.client = client
        self.config = config
    
    def scan(
        self,
        index: str,
        strike_pct_below: float = 5.0,
        min_ivr: Optional[float] = None,
        max_delta: Optional[float] = None,
        expiry_date: Optional[date] = None,
        max_symbols: int = 0,
        min_premium: Optional[float] = None,
        min_annualized_return: Optional[float] = None,
        max_days_to_expiry: Optional[int] = None,
        output_file: Optional[str] = None,
        test_symbol: Optional[str] = None,  # NEW parameter
        test_symbols: Optional[List[str]] = None  # NEW parameter for multiple test symbols
    ) -> List[PutScanResult]:
        """
        Scan for put selling opportunities across an index.
        
        Args:
            index: Index name (e.g., 'SPX', 'SP500', 'NDX', 'NASDAQ100')
            strike_pct_below: Percentage below current price to target (default: 5%)
            min_ivr: Minimum IVR to filter (optional)
            max_delta: Maximum delta (absolute value) - only consider options with delta <= this
            expiry_date: Target expiry date (optional, defaults to next Friday)
            max_symbols: DEPRECATED - scans entire index (kept for backwards compatibility)
            min_premium: Minimum premium in dollars (optional)
            min_annualized_return: Minimum annualized return percentage (optional)
            max_days_to_expiry: Maximum days to expiry filter (optional)
            output_file: Path to save results CSV (optional)
            test_symbol: If provided, scan only this symbol instead of entire index (optional)
            test_symbols: If provided, scan only these symbols instead of entire index (optional)
            
        Returns:
            List of PutScanResult objects sorted by annualized return
        """
        # NEW: If test_symbol is provided, use it instead of fetching index constituents
        if test_symbol:
            symbols = [test_symbol.upper()]
            logger.info(f"TEST MODE: Scanning single symbol {test_symbol.upper()}")
        # If test_symbols are provided, use them instead of fetching index constituents
        if test_symbols:
            symbols = test_symbols
            logger.info(f"TEST MODE: Scanning {len(test_symbols)} symbols: {', '.join(test_symbols)}")
        else:
            # Get symbols for the index using IndexConstituents utility
            symbols = IndexConstituents.get_constituents(index)
            
            if not symbols:
                logger.error(f"Could not get constituents for index: {index}")
                return []
        
        # Limit symbols if max_symbols specified
        if max_symbols > 0:
            symbols = symbols[:max_symbols]
        
        # Determine expiry date window
        max_expiry = None
        if max_days_to_expiry is not None:
            max_expiry = date.today() + timedelta(days=max_days_to_expiry)
            logger.info(f"Looking for options expiring within {max_days_to_expiry} days (by {max_expiry.strftime('%Y-%m-%d')})")
        elif expiry_date is not None:
            # If specific expiry date provided, use it
            max_expiry = expiry_date
            logger.info(f"Looking for options expiring on {expiry_date.strftime('%Y-%m-%d')}")
        else:
            # Default to next Friday (7 days max)
            max_expiry = date.today() + timedelta(days=7)
            logger.info(f"No expiry specified, looking within 7 days (by {max_expiry.strftime('%Y-%m-%d')})")
        
        days_to_expiry = (max_expiry - date.today()).days
        
        logger.info(f"Scanning {len(symbols)} symbols (Expiry: {max_expiry.strftime('%Y-%m-%d')}, {days_to_expiry}d)")
        
        # STEP 1: Search for all symbols and get conids
        symbol_to_conid = {}
        
        for symbol in symbols:
            try:
                contracts = self.client.search_contracts(symbol)
                
                if not contracts:
                    logger.info(f"  ✗ {symbol}: No contracts found")
                    continue
                
                stock_conid = None
                for contract in contracts:
                    if contract.get('symbol') == symbol:
                        sections = contract.get('sections', [])
                        if any(s.get('secType') == 'STK' for s in sections):
                            stock_conid = int(contract.get('conid'))
                            break
                
                if stock_conid:
                    symbol_to_conid[symbol] = stock_conid
                else:
                    logger.info(f"  ✗ {symbol}: No stock contract found")
                    
            except Exception as e:
                logger.info(f"  ✗ {symbol}: Error searching - {e}")
                continue
        
        if not symbol_to_conid:
            logger.info("No valid contracts found")
            return []
        
        # STEP 2: Get market data for all stocks
        logger.info(f"Initializing market data for {len(symbol_to_conid)} stocks...")
        
        # Get conids list (already obtained in STEP 1)
        conids = list(symbol_to_conid.values())
        
        # Wait before requesting market data
        time.sleep(5)
        
        # Get market data for all stocks in one batch call
        snapshots = self.client.get_market_data_snapshot(
            conids,
            fields=['31', '84', '86', '7283', '7088']
        )
        
        if not snapshots:
            logger.warning("No market data returned")
            return []
        
        # Check if we got preflight responses and retry if needed
        metadata_keys = {'conid', 'conidEx', '_updated', 'server_id', '6119', '6509'}
        has_data = any(
            any(key not in metadata_keys for key in snapshot.keys())
            for snapshot in snapshots
        )
        
        if not has_data:
            logger.debug("Preflight response received, waiting and retrying...")
            time.sleep(2)
            snapshots = self.client.get_market_data_snapshot(
                conids,
                fields=['31', '84', '86', '7283', '7088']
            )
        
        # STEP 3: Parse stock data (remove duplicate batch call)
        # The next section "STEP 3: Batch get market data" should be deleted
        # Just continue with parsing the snapshots we already have
        conid_to_symbol = {v: k for k, v in symbol_to_conid.items()}
        stock_data = {}
        
        def to_float(val):
            if val is None:
                return None
            try:
                if isinstance(val, str):
                    val = val.lstrip('CH').replace(',', '')
                return float(val)
            except (ValueError, TypeError):
                return None
        
        def parse_pct(val):
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                return float(val.replace('%', '').strip())
            return None
        
        for snapshot in snapshots:
            conid = snapshot.get('conid')
            if conid not in conid_to_symbol:
                continue
            
            symbol = conid_to_symbol[conid]
            
            # Extract stock price
            stock_price = to_float(snapshot.get('31'))
            if stock_price is None:
                stock_price = to_float(snapshot.get('84'))
            
            if not stock_price:
                logger.info(f"  ✗ {symbol}: No price data")
                continue
            
            # Extract IV/HV
            iv_str = snapshot.get('7283')
            hv_str = snapshot.get('7088')
            
            stock_ivhv = None
            if iv_str and hv_str:
                iv = parse_pct(iv_str)
                hv = parse_pct(hv_str)
                
                if iv is not None and hv is not None and hv != 0:
                    stock_ivhv = (iv / hv) * 100
            
            # EARLY FILTER: Check IVR before getting option chain
            if min_ivr is not None:
                if stock_ivhv is None:
                    logger.info(f"  ✗ {symbol}: No IV/HV data")
                    continue
                elif stock_ivhv < min_ivr:
                    logger.info(f"  ✗ {symbol}: IVR {stock_ivhv:.1f}% < min {min_ivr:.1f}%")
                    continue
            
            stock_data[symbol] = {
                'price': stock_price,
                'conid': conid,
                'ivhv': stock_ivhv
            }
        
        # STEP 4: Find options
        logger.info(f"\n{'Symbol':<6} {'Expiry':<10} {'Strike':<7} {'Delta':<7} {'IVR':<6} {'AnnRet':<7} Status")
        logger.info("-" * 80)
        
        results = []
        
        for symbol, data in stock_data.items():
            try:
                # Calculate target strike
                target_strike = data['price'] * (1 - strike_pct_below / 100)
                
                # Find put option
                option_data = self._find_put_option_with_conid(
                    symbol=symbol,
                    stock_conid=data['conid'],
                    stock_price=data['price'],
                    target_strike=target_strike,
                    expiry_date=max_expiry,
                    target_delta=max_delta,
                    stock_ivhv=data['ivhv']
                )
                logger.debug(f"{symbol}: Option data found: {option_data}")

                if not option_data:
                    # Log which filters were active
                    active_filters = []
                    if max_delta is not None:
                        active_filters.append(f"max_delta={max_delta:.3f}")
                    if min_premium is not None:
                        active_filters.append(f"min_premium=${min_premium:.2f}")
                    if min_annualized_return is not None:
                        active_filters.append(f"min_ann_ret={min_annualized_return:.1f}%")
                    if max_days_to_expiry is not None:
                        active_filters.append(f"max_DTE={max_days_to_expiry}")
                    
                    filter_str = f" (filters: {', '.join(active_filters)})" if active_filters else ""
                    logger.info(f"{symbol:<6} - No suitable option found{filter_str}")
                    continue
                
                # Calculate mid price
                mid_price = None
                if option_data['bid'] is not None and option_data['ask'] is not None:
                    mid_price = (option_data['bid'] + option_data['ask']) / 2
                
                # Calculate annualized return
                ann_return = None
                if mid_price is not None and days_to_expiry > 0:
                    pct_return = (mid_price / option_data['strike']) * 100
                    ann_return = (pct_return / days_to_expiry) * 365
                
                # LOG THE OPTION DATA BEFORE FILTERING (DEBUG LEVEL)
                delta_val = abs(option_data['delta']) if option_data.get('delta') is not None else None
                delta_str = f"{delta_val:.3f}" if delta_val is not None else "N/A"
                mid_str = f"${mid_price:.2f}" if mid_price is not None else "N/A"
                ann_str = f"{ann_return:.1f}%" if ann_return is not None else "N/A"
                
                logger.debug(f"  ${option_data['strike']:<6.0f}P: "
                           f"delta={delta_str}, "
                           f"premium={mid_str}, "
                           f"ann.ret={ann_str}, "
                           f"DTE={days_to_expiry}")
                
                # Apply filters and collect reasons if they fail
                filter_reasons = []
                
                # Apply premium filter
                if min_premium is not None and mid_price is not None:
                    if mid_price < min_premium:
                        filter_reasons.append(f"premium ${mid_price:.2f} < min ${min_premium:.2f}")
                
                # Apply delta filter (from option_data)
                if max_delta is not None and option_data.get('delta') is not None:
                    abs_delta = abs(option_data['delta'])
                    if abs_delta > max_delta:
                        filter_reasons.append(f"delta {abs_delta:.3f} > max {max_delta:.3f}")
                
                # Apply annualized return filter
                if min_annualized_return is not None and ann_return is not None:
                    if ann_return < min_annualized_return:
                        filter_reasons.append(f"ann.ret {ann_return:.1f}% < min {min_annualized_return:.1f}%")
                
                # Apply max DTE filter
                if max_days_to_expiry is not None:
                    if days_to_expiry > max_days_to_expiry:
                        filter_reasons.append(f"DTE {days_to_expiry} > max {max_days_to_expiry}")
                
                # If any filters failed, log why and skip (DEBUG LEVEL)
                if filter_reasons:
                    logger.debug(f"    FILTERED: {', '.join(filter_reasons)}")
                    continue
                else:
                    logger.debug(f"    ✓ PASSED all filters")
                
                # Create result
                result = PutScanResult(
                    symbol=symbol,
                    stock_price=data['price'],
                    strike=option_data['strike'],
                    expiry=max_expiry,
                    bid=option_data['bid'],
                    ask=option_data['ask'],
                    mid=mid_price,
                    delta=option_data['delta'],
                    gamma=option_data.get('gamma'),
                    vega=option_data.get('vega'),
                    theta=option_data.get('theta'),
                    implied_vol=option_data['implied_vol'],
                    ivr=option_data['ivr'],
                    volume=option_data.get('volume')
                )
                
                results.append(result)
                
                # Print single-line summary
                ivr_str = f"{result['ivr']:.1f}%" if result.get('ivr') else "N/A"
                ann_ret_str = f"{ann_return:.1f}%" if ann_return else "N/A"
                delta_str = f"{abs(result['delta']):.3f}" if result.get('delta') else "N/A"
                
                expiry_str = max_expiry.strftime('%Y-%m-%d')
                logger.info("Expiry is: " + expiry_str)
                logger.info(f"{symbol:<6} {expiry_str:<10} "
                          f"${result['strike']:<6.0f} {delta_str:<7} {ivr_str:<6} {ann_ret_str:<7} ✓")
                logger.info("ALL DONE")
            except Exception as e:
                logger.info(f"{symbol:<6} - Error: {str(e)[:40]}")
                continue
        
        # Sort by annualized return
        results.sort(key=lambda r: (
            (r.mid / r.strike) * 365 / days_to_expiry * 100 
            if r.mid and days_to_expiry > 0 
            else (r.ivr if r.ivr else -1)
        ), reverse=True)
        
        logger.info("-" * 80)
        logger.info(f"Found {len(results)} opportunities")
        
        # Save results to CSV if output file specified
        if output_file and results:
            scan_params = {
                'Index': index.upper(),
                'Target Expiry': max_expiry.strftime('%Y-%m-%d'),
                'Days to Expiry': days_to_expiry,
                'Strike %': f"{strike_pct_below}%",
                'Target Delta': max_delta if max_delta else 'N/A',
                'Min IVR': f"{min_ivr}%" if min_ivr else 'N/A',
                'Min Premium': f"${min_premium}" if min_premium else 'N/A',
                'Min Ann. Return': f"{min_annualized_return}%" if min_annualized_return else 'N/A',
                'Total Scanned': len(symbols),
                'Passed Filters': len(results)
            }
            self.save_results_to_csv(results, output_file, scan_params)
        
        return results
    
    def _find_put_option_with_conid(
        self,
        symbol: str,
        stock_conid: int,
        stock_price: float,
        target_strike: float,
        expiry_date: date,
        target_delta: Optional[float] = None,
        stock_ivhv: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Find a put option near the target strike and expiry using existing conid.
        Simplified to avoid verbose logging.
        
        Args:
            symbol: Stock symbol
            stock_conid: Stock contract ID (already obtained)
            stock_price: Current stock price
            target_strike: Target strike price
            expiry_date: Target expiration date
            target_delta: Target delta (absolute value) - if provided, finds closest delta
            stock_ivhv: Stock IV/HV ratio (already calculated)
            
        Returns:
            Dictionary with option data including strike, bid, ask, greeks, IVR
        """
        try:
            # Get configured number of strikes to evaluate
            num_strikes = 10  # default
            if hasattr(self.config, 'num_strikes'):
                num_strikes = self.config.num_strikes
            
            logger.debug(f"{symbol}: Getting {num_strikes} strikes near ${target_strike:.2f}")
            
            # Get options near target strike using the option chain
            nearby_strikes = self.client.get_options_near_strike(
                stock_conid=stock_conid,
                target_strike=target_strike,
                expiry_date=expiry_date,
                right='P',
                num_strikes=num_strikes
            )
            
            if not nearby_strikes:
                logger.debug(f"{symbol}: No strikes returned from get_options_near_strike")
                return None
            
            logger.debug(f"{symbol}: Found {len(nearby_strikes)} strikes to evaluate")
            
            if target_delta:
                # Use delta-based selection
                # Prepare batch specs for all strikes
                batch_specs = []
                for strike_info in nearby_strikes:
                    batch_specs.append({
                        'stock_conid': stock_conid,
                        'strike': strike_info['strike'],
                        'expiry_date': expiry_date
                    })
                
                logger.debug(f"{symbol}: Batch requesting option data for {len(batch_specs)} strikes...")
                
                # Batch request all option data
                batch_results = self.client.get_options_data_batch(
                    batch_specs, 
                    right='P',
                    skip_preflight=True  # Pre-flight already done in get_options_near_strike()
                )
                
                logger.debug(f"{symbol}: Received {len(batch_results)} option data results")
                
                # Find best delta match
                best_option = None
                best_delta_diff = float('inf')
                
                for strike_info, option_result in zip(nearby_strikes, batch_results):
                    # LOG THE RAW OPTION RESULT
                    logger.debug(f"{symbol}: ${strike_info['strike']:.0f}P - Raw option_result: {option_result}")
                    
                    if option_result and option_result.get('delta') is not None:
                        # Add stock-level IV/HV to the result
                        option_result['ivr'] = stock_ivhv
                        
                        abs_delta = abs(option_result['delta'])
                        delta_diff = abs(abs_delta - target_delta)
                        
                        logger.debug(f"{symbol}: ${strike_info['strike']:.0f}P - delta={abs_delta:.3f}, diff from target={delta_diff:.3f}")
                        
                        if delta_diff < best_delta_diff:
                            best_delta_diff = delta_diff
                            best_option = option_result
                    else:
                        logger.debug(f"{symbol}: ${strike_info['strike']:.0f}P - No option data or delta")
                
                if best_option:
                    logger.debug(f"{symbol}: Best option found: ${best_option['strike']:.0f}P with delta={abs(best_option['delta']):.3f}")
                else:
                    logger.debug(f"{symbol}: No options with valid delta data")
                
                return best_option
            
            else:
                # Original behavior: use strike closest to target price
                closest_strike_info = nearby_strikes[0]  # Already sorted by distance
                
                logger.debug(f"{symbol}: Using closest strike ${closest_strike_info['strike']:.0f}")
                
                result = self.client.get_option_data(
                    stock_conid=stock_conid,
                    strike=closest_strike_info['strike'],
                    expiry_date=expiry_date,
                    right='P'
                )
                
                if result:
                    # Add stock-level IV/HV to the result
                    result['ivr'] = stock_ivhv
                    logger.debug(f"{symbol}: Got option data for ${closest_strike_info['strike']:.0f}P")
                    return result
                else:
                    logger.debug(f"{symbol}: No option data returned for ${closest_strike_info['strike']:.0f}P")
                    return None
                    
        except Exception as e:
            logger.debug(f"{symbol}: Exception in _find_put_option_with_conid: {e}")
            return None

    def _get_stock_price(self, symbol: str) -> Optional[float]:
        """Get current stock price using IBKR API."""
        try:
            price = self.client.get_stock_price(symbol)
            if price is None:
                logger.debug(f"No price available for {symbol}")
            return price
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def _find_put_option(
        self,
        symbol: str,
        stock_price: float,
        target_strike: float,
        expiry_date: date,
        target_delta: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Find a put option near the target strike and expiry.
        If target_delta is provided, finds option with delta closest to target.
        
        Args:
            symbol: Stock symbol
            stock_price: Current stock price
            target_strike: Target strike price
            expiry_date: Target expiration date
            target_delta: Target delta (absolute value) - if provided, finds closest delta
            
        Returns:
            Dictionary with option data including strike, bid, ask, greeks, IVR
        """
        try:
            # Get stock contract ID (only search once)
            contracts = self.client.search_contracts(symbol)
            if not contracts:
                logger.debug(f"  Reason: No contracts found for {symbol}")
                return None
            
            stock_conid = None
            for contract in contracts:
                if contract.get('symbol') == symbol:
                    sections = contract.get('sections', [])
                    if any(s.get('secType') == 'STK' for s in sections):
                        stock_conid = int(contract.get('conid'))
                        break
            
            if not stock_conid:
                logger.debug(f"  Reason: No stock contract found for {symbol}")
                return None
            
            # Get IV/HV ratio from the underlying stock (pass conid to avoid duplicate search)
            stock_ivhv = self._get_stock_ivhv(stock_conid, symbol)
            if stock_ivhv is None:
                logger.debug(f"  Reason: Could not get IV/HV ratio for {symbol}")
            else:
                logger.debug(f"Stock IV/HV ratio for {symbol}: {stock_ivhv:.1f}%")
            
            # Get configured number of strikes to evaluate
            num_strikes = 10  # default
            if hasattr(self.config, 'num_strikes'):
                num_strikes = self.config.num_strikes
            
            # Get options near target strike using the option chain
            logger.info(f"  Getting option chain for {symbol}...")
            nearby_strikes = self.client.get_options_near_strike(
                stock_conid=stock_conid,
                target_strike=target_strike,
                expiry_date=expiry_date,
                right='P',
                num_strikes=num_strikes
            )
            
            if not nearby_strikes:
                logger.debug(f"  Reason: No options found near strike ${target_strike:.2f}")
                return None
            
            logger.info(f"  Found {len(nearby_strikes)} strikes near ${target_strike:.2f}")
            
            if target_delta:
                # Use delta-based selection
                logger.info(f"  Finding option closest to target delta {target_delta:.2f}...")
                
                # Prepare batch specs for all strikes
                batch_specs = []
                for strike_info in nearby_strikes:
                    batch_specs.append({
                        'stock_conid': stock_conid,
                        'strike': strike_info['strike'],
                        'expiry_date': expiry_date
                    })
                
                # Batch request all option data
                logger.info(f"  Batch requesting data for {len(batch_specs)} strikes...")
                batch_results = self.client.get_options_data_batch(
                    batch_specs, 
                    right='P',
                    skip_preflight=True  # Pre-flight already done in get_options_near_strike()
                )
                
                # Find best delta match
                best_option = None
                best_delta_diff = float('inf')
                tested_count = 0
                
                for strike_info, option_result in zip(nearby_strikes, batch_results):
                    if option_result and option_result.get('delta') is not None:
                        # Add stock-level IV/HV to the result
                        option_result['ivr'] = stock_ivhv
                        
                        abs_delta = abs(option_result['delta'])
                        delta_diff = abs_delta - target_delta
                        tested_count += 1
                        
                        logger.info(f"    ${strike_info['strike']:.0f}: Δ={abs_delta:.3f} (diff: {delta_diff:.3f})")
                        
                        if delta_diff < best_delta_diff:
                            best_delta_diff = delta_diff
                            best_option = option_result
                
                if best_option:
                    logger.info(f"  Best match: ${best_option['strike']:.0f} with Δ={abs(best_option['delta']):.3f}")
                    return best_option
                else:
                    logger.debug(f"  Reason: could not find option with valid delta data")
                    return None
            
            else:
                # Original behavior: use strike closest to target price
                closest_strike_info = nearby_strikes[0]  # Already sorted by distance
                
                result = self.client.get_option_data(
                    stock_conid=stock_conid,
                    strike=closest_strike_info['strike'],
                    expiry_date=expiry_date,
                    right='P'
                )
                
                if result:
                    # Add stock-level IV/HV to the result
                    result['ivr'] = stock_ivhv
                    return result
                else:
                    logger.debug(f"  Reason: could not get option data for strike ${closest_strike_info['strike']:.0f}")
                    return None
                    
        except Exception as e:
            logger.debug(f"  Reason: Exception - {e}")
            return None

    def _get_stock_ivhv(self, stock_conid: int, symbol: str) -> Optional[float]:
        """
        Get IV/HV ratio for a stock using its conid.
        
        Args:
            stock_conid: Stock contract ID
            symbol: Stock symbol (for logging)
            
        Returns:
            IV/HV ratio as percentage, or None if unavailable
        """
        try:
            # Get volatility data snapshot (fields 7283=IV, 7088=HV)
            logger.debug(f"Getting IV/HV for {symbol} (conid: {stock_conid})")
            
            snapshots = self.client.get_market_data_snapshot(
                [stock_conid],
                fields=['7283', '7088'],  # IV and Historical Volatility
                ensure_preflight=False
            )
            
            if not snapshots or len(snapshots) == 0:
                logger.debug(f"No volatility data returned for {symbol}")
                return None
            
            snapshot = snapshots[0]
            logger.debug(f"{symbol} volatility snapshot: {snapshot}")
            
            # Extract IV and HV
            iv_str = snapshot.get('7283')  # Implied Volatility
            hv_str = snapshot.get('7088')  # Historical Volatility
            
            logger.debug(f"{symbol}: Current IV (7283)={iv_str}, Historical Vol (7088)={hv_str}")
            
            if not iv_str or not hv_str:
                logger.debug(f"Missing IV or HV data for {symbol}")
                return None
            
            # Parse percentages (may be like "20.5%" or "20.5")
            def parse_pct(val):
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    return float(val.replace('%', '').strip())
                return None
            
            iv = parse_pct(iv_str)
            hv = parse_pct(hv_str)
            
            if iv is None or hv is None or hv == 0:
                logger.debug(f"Invalid IV ({iv}) or HV ({hv}) for {symbol}")
                return None
            
            # Calculate IV/HV ratio
            ivr = (iv / hv) * 100
            
            logger.info(f"{symbol} IV/HV: {ivr:.1f}% (IV={iv:.1f}% / HV={hv:.1f}%)")
            
            return ivr
            
        except Exception as e:
            logger.warning(f"Error getting IV/HV for {symbol}: {e}")
            return None
            
    def _get_next_friday(self) -> date:
        """Get the next Friday's date."""
        today = date.today()
        days_until_friday = (4 - today.weekday()) % 7  # Friday is 4
        if days_until_friday == 0:
            days_until_friday = 7  # If today is Friday, get next Friday
        return today + timedelta(days=days_until_friday)
    
    def print_results(self, results: List[PutScanResult], top_n: int = 20):
        """
        Print scan results in a formatted table.
        
        Args:
            results: List of PutScanResult objects
            top_n: Number of top results to display
        """
        if not results:
            logger.info("No results to display")
            return
        
        print("\n" + "=" * 140)
        print(f"TOP {min(top_n, len(results))} PUT SELLING OPPORTUNITIES")
        print("=" * 140)
        print(f"{'Symbol':<6} | {'Price':<8} | {'Strike':<8} | {'OTM%':<6} | "
              f"{'DTE':<4} | {'IVR':<5} | {'IV':<7} | {'Delta':<7} | {'Mid':<8} | {'Ann.Ret':<8}")
        print("-" * 140)
        
        for result in results[:top_n]:
            ivr_str = f"{result.ivr:.1f}" if result.ivr is not None else "N/A"
            iv_str = f"{result.implied_vol:.1%}" if result.implied_vol is not None else "N/A"
            delta_str = f"{result.delta:.3f}" if result.delta is not None else "N/A"
            mid_str = f"${result.mid:.2f}" if result.mid is not None else "N/A"
            ann_ret_str = f"{result.annualized_return:.1f}%" if result.annualized_return is not None else "N/A"
            
            print(f"{result.symbol:<6} | ${result.current_price:>6.2f} | "
                  f"${result.strike:>6.2f} | {result.strike_pct_otm:>5.1f}% | "
                  f"{result.days_to_expiry:>3} | {ivr_str:>5} | {iv_str:>7} | "
                  f"{delta_str:>7} | {mid_str:>8} | {ann_ret_str:>8}")
        
        print("=" * 140)

    def scan_symbols(
        self,
        symbols: List[str],
        target_delta: float = 0.20,
        expiry_date: Optional[date] = None,
        strike_pct_below: float = 5.0
    ) -> List[PutScanResult]:
        """
        Scan multiple symbols for put selling opportunities.
        
        Args:
            symbols: List of stock symbols to scan
            target_delta: Target delta (absolute value) for put options
            expiry_date: Target expiration date (defaults to next Friday)
            strike_pct_below: Strike as percentage below current price
            
        Returns:
            List of PutScanResult objects
        """
        if not symbols:
            return []
        
        if expiry_date is None:
            expiry_date = self._get_next_friday()
        
        logger.info(f"Scanning {len(symbols)} symbols for puts expiring {expiry_date.strftime('%Y-%m-%d')}...")
        logger.info(f"Target: Δ~{target_delta:.2f}, Strike ~{strike_pct_below:.1f}% below current price")
        
        # Step 1: Get all stock prices in batch
        logger.info("Step 1: Getting stock prices...")
        stock_prices = self.client.get_stock_prices_batch(symbols)
        
        results = []
        valid_symbols = []
        
        # Filter out symbols with no price
        for symbol in symbols:
            price = stock_prices.get(symbol)
            if price:
                valid_symbols.append((symbol, price))
                logger.info(f"  ✓ {symbol}: ${price:.2f}")
            else:
                logger.info(f"  ✗ {symbol}: Could not get stock price")
        
        if not valid_symbols:
            logger.info("No valid stock prices found")
            return []
        
        # Step 2: Find options for each symbol
        logger.info(f"\nStep 2: Finding put options for {len(valid_symbols)} symbols...")
        
        for i, (symbol, stock_price) in enumerate(valid_symbols, 1):
            logger.info(f"Scanning {i}/{len(valid_symbols)}: {symbol}...")
            
            # Calculate target strike
            target_strike = stock_price * (1 - strike_pct_below / 100)
            
            # Find the put option
            option_data = self._find_put_option(
                symbol=symbol,
                stock_price=stock_price,
                target_strike=target_strike,
                expiry_date=expiry_date,
                target_delta=target_delta
            )
            
            if option_data:
                # Create result object
                result = PutScanResult(
                    symbol=symbol,
                    stock_price=stock_price,
                    strike=option_data['strike'],
                    expiry=expiry_date,
                    bid=option_data.get('bid'),
                    ask=option_data.get('ask'),
                    mid=(option_data['bid'] + option_data['ask']) / 2 
                        if option_data.get('bid') and option_data.get('ask') else None,
                    delta=option_data.get('delta'),
                    gamma=option_data.get('gamma'),
                    vega=option_data.get('vega'),
                    theta=option_data.get('theta'),
                    implied_vol=option_data.get('implied_vol'),
                    ivr=option_data.get('ivr'),
                    volume=option_data.get('volume')
                )
                results.append(result)
                logger.info(f"  ✓ {symbol}: Found suitable put option")
            else:
                logger.info(f"  ✗ {symbol}: No suitable put option found")
        
        logger.info(f"\nFound {len(results)} total opportunities")
        return results

    def save_results_to_csv(self, results: List[PutScanResult], output_file: str, scan_params: Optional[Dict] = None):
        """
        Save scan results to CSV file.
        
        Args:
            results: List of PutScanResult objects
            output_file: Path to output CSV file
            scan_params: Optional dict with scan parameters to include as header comments
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', newline='') as csvfile:
                # Write header comments with scan parameters
                if scan_params:
                    csvfile.write(f"# Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    for key, value in scan_params.items():
                        csvfile.write(f"# {key}: {value}\n")
                    csvfile.write("#\n")
                
                # Define CSV columns - only essential data
                fieldnames = [
                    'symbol', 'stock_price', 'strike', 'strike_pct_otm', 'expiry', 
                    'days_to_expiry', 'bid', 'ask', 'mid', 'delta', 
                    'implied_vol', 'ivr', 'annualized_return'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write results
                for result in results:
                    writer.writerow({
                        'symbol': result.symbol,
                        'stock_price': f"{result.stock_price:.2f}",
                        'strike': f"{result.strike:.2f}",
                        'strike_pct_otm': f"{result.strike_pct_otm:.2f}",
                        'expiry': result.expiry.strftime('%Y-%m-%d'),
                        'days_to_expiry': result.days_to_expiry,
                        'bid': f"{result.bid:.2f}" if result.bid else '',
                        'ask': f"{result.ask:.2f}" if result.ask else '',
                        'mid': f"{result.mid:.2f}" if result.mid else '',
                        'delta': f"{result.delta:.4f}" if result.delta else '',
                        'implied_vol': f"{result.implied_vol:.4f}" if result.implied_vol else '',
                        'ivr': f"{result.ivr:.2f}" if result.ivr else '',
                        'annualized_return': f"{result.annualized_return:.2f}" if result.annualized_return else ''
                    })
            
            logger.info(f"Results saved to: {output_file}")
            logger.info(f"  Total records: {len(results)}")
            
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}", exc_info=True)

