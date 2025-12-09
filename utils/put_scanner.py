"""
Put option scanner for finding high IVR opportunities across indices.
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
import logging
from dataclasses import dataclass
import time
import configparser

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
        
        # Cache for index constituents
        self._symbol_cache = {}
        
        # Common indices and their symbols
        self.INDICES = {
            'SPX': self._get_spx_symbols,
            'SP500': self._get_spx_symbols,  # Alias
            'NDX': self._get_ndx_symbols,
            'NASDAQ100': self._get_ndx_symbols,  # Alias
            'DJI': self._get_dji_symbols,
            'DOW': self._get_dji_symbols,  # Alias
            'XLE': self._get_xle_symbols,  # Energy
            'XLF': self._get_xlf_symbols,  # Financials
            'XLK': self._get_xlk_symbols,  # Technology
            'XLV': self._get_xlv_symbols,  # Healthcare
            'XLI': self._get_xli_symbols,  # Industrials
            'XLP': self._get_xlp_symbols,  # Consumer Staples
            'XLY': self._get_xly_symbols,  # Consumer Discretionary
            'XLU': self._get_xlu_symbols,  # Utilities
            'XLB': self._get_xlb_symbols,  # Materials
            'XLRE': self._get_xlre_symbols,  # Real Estate
        }
    
    def scan(
        self,
        index: str,
        strike_pct_below: float = 5.0,
        min_ivr: Optional[float] = None,
        target_delta: Optional[float] = None,
        expiry_date: Optional[date] = None,
        max_symbols: int = 0
    ) -> List[PutScanResult]:
        """
        Scan for put selling opportunities across an index.
        
        Args:
            index: Index name (e.g., 'SPX', 'SP500', 'NDX', 'NASDAQ100')
            strike_pct_below: Percentage below current price to target (default: 5%)
            min_ivr: Minimum IVR to filter (optional)
            target_delta: Target delta to match (optional, e.g., 0.20 for ~20% probability)
            expiry_date: Target expiry date (defaults to next Friday)
            max_symbols: Maximum number of symbols to scan (0 = no limit)
            
        Returns:
            List of PutScanResult objects sorted by IVR (highest first)
        """
        # Get symbols for the index
        if index.upper() not in self.INDICES:
            logger.error(f"Unknown index: {index}. Available: {list(self.INDICES.keys())}")
            return []
        
        all_symbols = self.INDICES[index.upper()]()
        
        # Limit symbols if max_symbols specified
        if max_symbols > 0:
            symbols = all_symbols[:max_symbols]
            logger.info(f"Limiting to first {max_symbols} of {len(all_symbols)} symbols")
        else:
            symbols = all_symbols
        
        logger.info(f"Scanning {len(symbols)} symbols in {index.upper()}...")
        
        # Determine target expiry (next Friday if not specified)
        if expiry_date is None:
            expiry_date = self._get_next_friday()
        
        logger.info(f"Target expiry: {expiry_date.strftime('%Y-%m-%d')}")
        logger.info(f"Target strike: {strike_pct_below:.1f}% below current price")
        if target_delta:
            logger.info(f"Target delta: {target_delta:.2f}")
        if min_ivr:
            logger.info(f"Minimum IV/HV ratio: {min_ivr:.1f}%")
        
        # Scan each symbol
        results = []
        filtered_count = 0
        no_price_count = 0
        no_option_count = 0
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Scanning {i+1}/{len(symbols)}: {symbol}...")
                
                # Add small delay to avoid overwhelming the API
                if i > 0:
                    time.sleep(0.5)
                
                result = self._scan_symbol(
                    symbol=symbol,
                    strike_pct_below=strike_pct_below,
                    expiry_date=expiry_date,
                    target_delta=target_delta
                )
                
                if result:
                    # Apply IVR filter
                    if min_ivr is not None:
                        if result.ivr is None:
                            logger.info(f"  ✗ {symbol}: No IV/HV ratio available")
                            filtered_count += 1
                            continue
                        elif result.ivr < min_ivr:
                            logger.info(f"  ✗ {symbol}: IV/HV {result.ivr:.1f}% < minimum {min_ivr:.1f}%")
                            filtered_count += 1
                            continue
                    
                    results.append(result)
                    logger.info(f"  ✓ {result}")
                else:
                    # Already logged in _scan_symbol
                    no_option_count += 1
                
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}", exc_info=True)
                continue
        
        # Sort by IVR (highest first)
        results.sort(key=lambda r: r.ivr if r.ivr is not None else -1, reverse=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Scan Summary:")
        logger.info(f"  Total symbols scanned: {len(symbols)}")
        logger.info(f"  Opportunities found: {len(results)}")
        logger.info(f"  Filtered by IVR: {filtered_count}")
        logger.info(f"  No suitable option: {no_option_count}")
        logger.info(f"{'='*60}")
        
        return results
    
    def _scan_symbol(
        self,
        symbol: str,
        strike_pct_below: float = 5.0,
        expiry_date: Optional[date] = None,
        target_delta: Optional[float] = None
    ) -> Optional[PutScanResult]:
        """
        Scan a single symbol for put selling opportunity.
        
        Args:
            symbol: Stock ticker symbol
            strike_pct_below: Percentage below current price to target
            expiry_date: Target expiry date (defaults to next Friday)
            target_delta: Target delta to match (optional)
            
        Returns:
            PutScanResult object or None
        """
        try:
            # Get stock contract ID and all market data in one go
            contracts = self.client.search_contracts(symbol)
            if not contracts:
                logger.info(f"  ✗ {symbol}: Could not find contracts")
                return None
            
            stock_conid = None
            for contract in contracts:
                if contract.get('symbol') == symbol:
                    sections = contract.get('sections', [])
                    if any(s.get('secType') == 'STK' for s in sections):
                        stock_conid = int(contract.get('conid'))
                        break
            
            if not stock_conid:
                logger.info(f"  ✗ {symbol}: No stock contract found")
                return None
            
            # Get ALL data we need in ONE snapshot call: price (31), IV (7283), HV (7088)
            snapshots = self.client.get_market_data_snapshot(
                [stock_conid],
                fields=['31', '84', '86', '7283', '7088'],  # last, bid, ask, IV, HV
                ensure_preflight=False
            )
            
            if not snapshots or len(snapshots) == 0:
                logger.info(f"  ✗ {symbol}: No market data available")
                return None
            
            snapshot = snapshots[0]
            
            # Extract stock price
            stock_price = None
            last_price = snapshot.get('31')
            bid = snapshot.get('84')
            
            def to_float(val):
                if val is None:
                    return None
                try:
                    if isinstance(val, str):
                        val = val.lstrip('CH').replace(',', '')
                    return float(val)
                except (ValueError, TypeError):
                    return None
            
            stock_price = to_float(last_price)
            if stock_price is None:
                stock_price = to_float(bid)
            
            if not stock_price:
                logger.info(f"  ✗ {symbol}: Could not get stock price")
                return None
            
            # Extract and calculate IV/HV ratio
            iv_str = snapshot.get('7283')
            hv_str = snapshot.get('7088')
            
            stock_ivhv = None
            if iv_str and hv_str:
                def parse_pct(val):
                    if isinstance(val, (int, float)):
                        return float(val)
                    if isinstance(val, str):
                        return float(val.replace('%', '').strip())
                    return None
                
                iv = parse_pct(iv_str)
                hv = parse_pct(hv_str)
                
                if iv is not None and hv is not None and hv != 0:
                    stock_ivhv = (iv / hv) * 100
                    logger.debug(f"{symbol} IV/HV: {stock_ivhv:.1f}% (IV={iv:.1f}% / HV={hv:.1f}%)")
            
            # Calculate target strike
            target_strike = stock_price * (1 - strike_pct_below / 100)
            
            # Use next Friday if no expiry specified
            if expiry_date is None:
                expiry_date = self._get_next_friday()
            
            # Find put option (pass conid to avoid duplicate search)
            option_data = self._find_put_option_with_conid(
                symbol=symbol,
                stock_conid=stock_conid,
                stock_price=stock_price,
                target_strike=target_strike,
                expiry_date=expiry_date,
                target_delta=target_delta,
                stock_ivhv=stock_ivhv
            )
            
            if not option_data:
                logger.info(f"  ✗ {symbol}: No suitable put option found")
                return None
            
            # Calculate mid price
            mid_price = None
            if option_data['bid'] is not None and option_data['ask'] is not None:
                mid_price = (option_data['bid'] + option_data['ask']) / 2
            
            # Create result
            result = PutScanResult(
                symbol=symbol,
                stock_price=stock_price,
                strike=option_data['strike'],
                expiry=expiry_date,
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
            
            return result
            
        except Exception as e:
            logger.error(f"  ✗ {symbol}: Error - {e}", exc_info=True)
            return None

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
                        delta_diff = abs(abs_delta - target_delta)
                        tested_count += 1
                        
                        logger.info(f"    ${strike_info['strike']:.0f}: Δ={abs_delta:.3f} (diff: {delta_diff:.3f})")
                        
                        if delta_diff < best_delta_diff:
                            best_delta_diff = delta_diff
                            best_option = option_result
                
                if best_option:
                    logger.info(f"  Best match: ${best_option['strike']:.0f} with Δ={abs(best_option['delta']):.3f}")
                    return best_option
                else:
                    logger.debug(f"  Reason: Could not find option with valid delta data")
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
                    logger.debug(f"  Reason: Could not get option data for strike ${closest_strike_info['strike']:.0f}")
                    return None
                    
        except Exception as e:
            logger.debug(f"  Reason: Exception - {e}")
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
                        delta_diff = abs(abs_delta - target_delta)
                        tested_count += 1
                        
                        logger.info(f"    ${strike_info['strike']:.0f}: Δ={abs_delta:.3f} (diff: {delta_diff:.3f})")
                        
                        if delta_diff < best_delta_diff:
                            best_delta_diff = delta_diff
                            best_option = option_result
                
                if best_option:
                    logger.info(f"  Best match: ${best_option['strike']:.0f} with Δ={abs(best_option['delta']):.3f}")
                    return best_option
                else:
                    logger.debug(f"  Reason: Could not find option with valid delta data")
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
                    logger.debug(f"  Reason: Could not get option data for strike ${closest_strike_info['strike']:.0f}")
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
    
    def _get_spx_symbols(self) -> List[str]:
        """Get S&P 500 constituent symbols from Yahoo Finance."""
        if 'SPX' in self._symbol_cache:
            return self._symbol_cache['SPX']
        
        try:
            import yahoo_fin.stock_info as si
            logger.info("Fetching S&P 500 constituents from Yahoo Finance...")
            symbols = si.tickers_sp500()
            
            # Clean up symbols (remove any with special characters that might cause issues)
            symbols = [s for s in symbols if '.' not in s and '^' not in s]
            
            logger.info(f"Loaded {len(symbols)} S&P 500 symbols")
            self._symbol_cache['SPX'] = symbols
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 symbols: {e}")
            logger.warning("Falling back to hardcoded liquid names")
            # Fallback to liquid names
            return [
                'SPY', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA',
                'BRK.B', 'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX',
                'MRK', 'ABBV', 'KO', 'AVGO', 'PEP', 'COST', 'WMT', 'MCD', 'CSCO'
            ]
    
    def _get_ndx_symbols(self) -> List[str]:
        """Get NASDAQ-100 constituent symbols from Yahoo Finance."""
        if 'NDX' in self._symbol_cache:
            return self._symbol_cache['NDX']
        
        try:
            import yahoo_fin.stock_info as si
            logger.info("Fetching NASDAQ-100 constituents from Yahoo Finance...")
            symbols = si.tickers_nasdaq()
            
            # NASDAQ-100 is a subset - for now use the full NASDAQ list
            # or we can manually maintain the list
            # Clean up symbols
            symbols = [s for s in symbols if '.' not in s and '^' not in s]
            
            # Limit to most liquid/recognizable names for now
            # TODO: Get actual NASDAQ-100 list
            liquid_nasdaq = [
                'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO',
                'COST', 'NFLX', 'ADBE', 'CSCO', 'PEP', 'AMD', 'INTC', 'INTU',
                'QCOM', 'AMAT', 'SBUX', 'ISRG', 'TXN', 'MU', 'ADI', 'PYPL',
                'BKNG', 'CMCSA', 'TMUS', 'AMGN', 'VRTX', 'MDLZ', 'GILD', 'ADP',
                'LRCX', 'REGN', 'PANW', 'ABNB', 'SNPS', 'CDNS', 'KLAC', 'MELI',
                'MAR', 'ORLY', 'ASML', 'FTNT', 'NXPI', 'WDAY', 'DASH', 'TEAM'
            ]
            
            logger.info(f"Using {len(liquid_nasdaq)} liquid NASDAQ symbols")
            self._symbol_cache['NDX'] = liquid_nasdaq
            return liquid_nasdaq
            
        except Exception as e:
            logger.error(f"Failed to fetch NASDAQ symbols: {e}")
            logger.warning("Falling back to hardcoded list")
            return [
                'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AVGO',
                'COST', 'NFLX', 'ADBE', 'CSCO', 'AMD', 'INTC', 'INTU', 'QCOM'
            ]
    
    def _get_dji_symbols(self) -> List[str]:
        """Get Dow Jones Industrial Average constituent symbols from Yahoo Finance."""
        if 'DJI' in self._symbol_cache:
            return self._symbol_cache['DJI']
        
        try:
            import yahoo_fin.stock_info as si
            logger.info("Fetching Dow Jones constituents from Yahoo Finance...")
            symbols = si.tickers_dow()
            
            logger.info(f"Loaded {len(symbols)} Dow Jones symbols")
            self._symbol_cache['DJI'] = symbols
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to fetch Dow Jones symbols: {e}")
            logger.warning("Falling back to hardcoded list")
            return [
                'AAPL', 'MSFT', 'UNH', 'GS', 'HD', 'CAT', 'MCD', 'AMGN', 'V',
                'BA', 'TRV', 'AXP', 'JPM', 'IBM', 'JNJ', 'PG', 'CVX', 'MRK',
                'WMT', 'DIS', 'NKE', 'MMM', 'KO', 'DOW', 'CSCO', 'VZ', 'INTC',
                'WBA', 'HON', 'CRM'
            ]
    
    def _get_xle_symbols(self) -> List[str]:
        """Get XLE (Energy Select Sector) holdings."""
        if 'XLE' in self._symbol_cache:
            return self._symbol_cache['XLE']
        
        # Major XLE holdings
        symbols = [
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO',
            'OXY', 'WMB', 'HES', 'KMI', 'HAL', 'DVN', 'FANG', 'BKR',
            'LNG', 'TRGP', 'EQT', 'OKE', 'CTRA', 'MRO', 'APA'
        ]
        
        logger.info(f"Loaded {len(symbols)} XLE (Energy) symbols")
        self._symbol_cache['XLE'] = symbols
        return symbols
    
    def _get_xlf_symbols(self) -> List[str]:
        """Get XLF (Financial Select Sector) holdings."""
        if 'XLF' in self._symbol_cache:
            return self._symbol_cache['XLF']
        
        symbols = [
            'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI',
            'AXP', 'C', 'BLK', 'SCHW', 'CB', 'MMC', 'PGR', 'AON', 'USB',
            'TFC', 'AIG', 'MET', 'AFL', 'ALL', 'PRU', 'AJG', 'COF'
        ]
        
        logger.info(f"Loaded {len(symbols)} XLF (Financial) symbols")
        self._symbol_cache['XLF'] = symbols
        return symbols
    
    def _get_xlk_symbols(self) -> List[str]:
        """Get XLK (Technology Select Sector) holdings."""
        if 'XLK' in self._symbol_cache:
            return self._symbol_cache['XLK']
        
        symbols = [
            'AAPL', 'MSFT', 'NVDA', 'AVGO', 'CSCO', 'ADBE', 'CRM', 'ACN',
            'AMD', 'INTC', 'QCOM', 'INTU', 'TXN', 'AMAT', 'NOW', 'ORCL',
            'IBM', 'PANW', 'ADI', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS',
            'PLTR', 'ANET', 'APH', 'ADSK', 'MSI', 'FTNT'
        ]
        
        logger.info(f"Loaded {len(symbols)} XLK (Technology) symbols")
        self._symbol_cache['XLK'] = symbols
        return symbols
    
    def _get_xlv_symbols(self) -> List[str]:
        """Get XLV (Healthcare Select Sector) holdings."""
        if 'XLV' in self._symbol_cache:
            return self._symbol_cache['XLV']
        
        symbols = [
            'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'AMGN', 'DHR',
            'PFE', 'BMY', 'CVS', 'VRTX', 'GILD', 'ELV', 'CI', 'REGN', 'MCK',
            'ZTS', 'BSX', 'MDT', 'SYK', 'ISRG', 'HCA', 'BDX', 'EW'
        ]
        
        logger.info(f"Loaded {len(symbols)} XLV (Healthcare) symbols")
        self._symbol_cache['XLV'] = symbols
        return symbols
    
    def _get_xli_symbols(self) -> List[str]:
        """Get XLI (Industrial Select Sector) holdings."""
        if 'XLI' in self._symbol_cache:
            return self._symbol_cache['XLI']
        
        symbols = [
            'GE', 'CAT', 'RTX', 'HON', 'UNP', 'BA', 'UPS', 'ADP', 'DE',
            'LMT', 'TT', 'GD', 'MMM', 'NOC', 'ETN', 'EMR', 'ITW', 'CSX',
            'WM', 'PH', 'NSC', 'CARR', 'PCAR', 'FDX', 'JCI', 'ODFL'
        ]
        
        logger.info(f"Loaded {len(symbols)} XLI (Industrial) symbols")
        self._symbol_cache['XLI'] = symbols
        return symbols
    
    def _get_xlp_symbols(self) -> List[str]:
        """Get XLP (Consumer Staples Select Sector) holdings."""
        if 'XLP' in self._symbol_cache:
            return self._symbol_cache['XLP']
        
        symbols = [
            'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL',
            'MNST', 'KMB', 'GIS', 'STZ', 'KHC', 'SYY', 'HSY', 'K', 'CHD',
            'TSN', 'CAG', 'MKC', 'CPB', 'HRL', 'SJM'
        ]
        
        logger.info(f"Loaded {len(symbols)} XLP (Consumer Staples) symbols")
        self._symbol_cache['XLP'] = symbols
        return symbols
    
    def _get_xly_symbols(self) -> List[str]:
        """Get XLY (Consumer Discretionary Select Sector) holdings."""
        if 'XLY' in self._symbol_cache:
            return self._symbol_cache['XLY']
        
        symbols = [
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG',
            'CMG', 'MAR', 'ABNB', 'GM', 'F', 'DHI', 'LEN', 'YUM', 'ORLY',
            'DG', 'ROST', 'AZO', 'GRMN', 'DECK', 'ULTA', 'TPR'
        ]
        
        logger.info(f"Loaded {len(symbols)} XLY (Consumer Discretionary) symbols")
        self._symbol_cache['XLY'] = symbols
        return symbols
    
    def _get_xlu_symbols(self) -> List[str]:
        """Get XLU (Utilities Select Sector) holdings."""
        if 'XLU' in self._symbol_cache:
            return self._symbol_cache['XLU']
        
        symbols = [
            'NEE', 'SO', 'DUK', 'CEG', 'SRE', 'AEP', 'D', 'PEG', 'VST',
            'EXC', 'XEL', 'ED', 'ETR', 'WEC', 'ES', 'AWK', 'FE', 'AEE',
            'DTE', 'PPL', 'ATO', 'CMS', 'NI', 'LNT'
        ]
        
        logger.info(f"Loaded {len(symbols)} XLU (Utilities) symbols")
        self._symbol_cache['XLU'] = symbols
        return symbols
    
    def _get_xlb_symbols(self) -> List[str]:
        """Get XLB (Materials Select Sector) holdings."""
        if 'XLB' in self._symbol_cache:
            return self._symbol_cache['XLB']
        
        symbols = [
            'LIN', 'APD', 'SHW', 'FCX', 'ECL', 'NEM', 'CTVA', 'DOW', 'DD',
            'NUE', 'VMC', 'MLM', 'PPG', 'IFF', 'STLD', 'IP', 'PKG', 'BALL',
            'AVY', 'CF', 'MOS', 'FMC', 'ALB'
        ]
        
        logger.info(f"Loaded {len(symbols)} XLB (Materials) symbols")
        self._symbol_cache['XLB'] = symbols
        return symbols
    
    def _get_xlre_symbols(self) -> List[str]:
        """Get XLRE (Real Estate Select Sector) holdings."""
        if 'XLRE' in self._symbol_cache:
            return self._symbol_cache['XLRE']
        
        symbols = [
            'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'WELL', 'DLR', 'O', 'CBRE',
            'SPG', 'EXR', 'IRM', 'AVB', 'VICI', 'VTR', 'EQR', 'SBAC', 'WY',
            'INVH', 'MAA', 'ARE', 'ESS', 'KIM', 'DOC', 'UDR'
        ]
        
        logger.info(f"Loaded {len(symbols)} XLRE (Real Estate) symbols")
        self._symbol_cache['XLRE'] = symbols
        return symbols
    
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

