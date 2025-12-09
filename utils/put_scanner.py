"""
Put option scanner for finding high IVR opportunities across indices.
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class PutScanResult:
    """Represents a scan result for a put option."""
    symbol: str
    stock_price: float  # Changed from current_price
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
    def days_to_expiry(self) -> int:
        """Calculate days to expiry."""
        return (self.expiry - date.today()).days
    
    @property
    def strike_pct_otm(self) -> float:
        """Calculate percentage OTM."""
        return ((self.stock_price - self.strike) / self.stock_price) * 100
    
    def __str__(self) -> str:
        ivr_str = f"{self.ivr:.1f}" if self.ivr is not None else "N/A"
        iv_str = f"{self.implied_vol:.1%}" if self.implied_vol is not None else "N/A"
        delta_str = f"{abs(self.delta):.3f}" if self.delta is not None else "N/A"
        mid_str = f"${self.mid:.2f}" if self.mid is not None else "N/A"
        
        return (f"{self.symbol:<6} | Price: ${self.stock_price:>7.2f} | "
                f"Strike: ${self.strike:>7.2f} ({self.strike_pct_otm:>5.1f}% OTM) | "
                f"DTE: {self.days_to_expiry:>2} | IVR: {ivr_str:>5} | "
                f"IV: {iv_str:>6} | Δ: {delta_str:>6} | Mid: {mid_str:>7}")


class PutScanner:
    """
    Scanner for finding high IVR put selling opportunities across indices.
    """
    
    def __init__(self, ibkr_client):
        """
        Initialize the put scanner.
        
        Args:
            ibkr_client: IBWebAPIClient instance for market data
        """
        self.client = ibkr_client
        
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
            # Get current stock price
            stock_price = self._get_stock_price(symbol)
            if not stock_price:
                logger.info(f"  ✗ {symbol}: Could not get stock price")
                return None
            
            # Calculate target strike
            target_strike = stock_price * (1 - strike_pct_below / 100)
            
            # Use next Friday if no expiry specified
            if expiry_date is None:
                expiry_date = self._get_next_friday()
            
            # Find put option
            option_data = self._find_put_option(symbol, stock_price, target_strike, expiry_date, target_delta)
            
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
            # Get stock contract
            contracts = self.client.search_contracts(symbol)
            if not contracts:
                logger.debug(f"  Reason: No contracts found in search")
                return None
            
            # Find stock contract
            stock_contract = None
            for contract in contracts:
                sections = contract.get('sections', [])
                has_stock = any(section.get('secType') == 'STK' for section in sections)
                
                if has_stock and contract.get('symbol') == symbol:
                    description = contract.get('description', '')
                    if 'NYSE' in description or 'NASDAQ' in description or description == 'SMART':
                        stock_contract = contract
                        break
                    elif not stock_contract:
                        stock_contract = contract
            
            if not stock_contract:
                logger.debug(f"  Reason: No stock contract found")
                return None
            
            stock_conid = stock_contract.get('conid')
            if not stock_conid:
                logger.debug(f"  Reason: No conid for stock")
                return None
            
            # Get IV/HV ratio from the underlying stock
            stock_ivhv = self._get_stock_ivhv(stock_conid, symbol)  # Add symbol parameter
            if stock_ivhv is None:
                logger.debug(f"  Reason: Could not get IV/HV ratio for {symbol}")
            else:
                logger.debug(f"Stock IV/HV ratio for {symbol}: {stock_ivhv:.1f}%")
            
            # Get option strikes using YYYYMM format
            month_format = expiry_date.strftime('%Y%m')
            
            url = f"{self.client.base_url}/iserver/secdef/strikes"
            params = {
                'conid': stock_conid,
                'sectype': 'OPT',
                'month': month_format,
                'exchange': 'SMART'
            }
            
            response = self.client._make_request('GET', url, params=params)
            
            if response.status_code != 200:
                logger.debug(f"  Reason: Failed to get strikes (HTTP {response.status_code})")
                return None
            
            data = response.json()
            
            if not data or not isinstance(data, dict):
                logger.debug(f"  Reason: Invalid strikes response")
                return None
            
            put_strikes = data.get('put', [])
            
            if not put_strikes:
                logger.debug(f"  Reason: No put strikes available for {month_format}")
                return None
            
            # Get available strikes
            float_strikes = [float(s) for s in put_strikes]
            
            logger.debug(f"Found {len(float_strikes)} strikes for {symbol}")
            
            # If target_delta provided, find closest delta match
            if target_delta is not None:
                sorted_strikes = sorted(float_strikes, reverse=True)
                
                # Find strikes below current stock price (OTM puts)
                candidates = []
                for strike in sorted_strikes:
                    if strike < stock_price:
                        candidates.append(strike)
                
                if not candidates:
                    logger.debug(f"  Reason: No OTM strikes below ${stock_price:.2f}")
                    return None
                
                logger.info(f"  Testing strikes from ${candidates[0]:.0f} down for Δ~{target_delta:.2f}")
                
                # Get option data for each candidate
                best_option = None
                best_delta_diff = float('inf')
                prev_delta_diff = float('inf')
                tested_count = 0
                
                for i, strike in enumerate(candidates[:20]):
                    option_result = self._get_option_data(
                        stock_conid, month_format, strike, expiry_date, stock_ivhv  # Pass stock_ivhv here
                    )
                    
                    if option_result and option_result.get('delta') is not None:
                        abs_delta = abs(option_result['delta'])
                        delta_diff = abs(abs_delta - target_delta)
                        tested_count += 1
                        
                        logger.info(f"    ${strike:.0f}: Δ={abs_delta:.3f} (diff: {delta_diff:.3f})")
                        
                        if delta_diff < best_delta_diff:
                            best_delta_diff = delta_diff
                            best_option = option_result
                        
                        # Stop if delta difference starts increasing
                        if delta_diff > prev_delta_diff:
                            logger.info(f"    Delta diff increasing, stopping search")
                            break
                        
                        prev_delta_diff = delta_diff
                    
                    if i < 19:
                        time.sleep(0.3)
                
                if best_option:
                    abs_best_delta = abs(best_option['delta'])
                    logger.info(f"  Best: ${best_option['strike']:.0f} Δ={abs_best_delta:.3f} (diff: {best_delta_diff:.3f})")
                    
                    # Log final result
                    log_parts = [f"{symbol} ${best_option['strike']:.0f}P"]
                    
                    if best_option['bid'] is not None:
                        log_parts.append(f"bid=${best_option['bid']:.2f}")
                    if best_option['ask'] is not None:
                        log_parts.append(f"ask=${best_option['ask']:.2f}")
                    if best_option['delta'] is not None:
                        log_parts.append(f"Δ={abs_best_delta:.3f}")
                    if best_option['implied_vol'] is not None:
                        log_parts.append(f"IV={best_option['implied_vol']:.1%}")
                    if best_option['ivr'] is not None:
                        log_parts.append(f"IV/HV={best_option['ivr']:.1f}%")
                    
                    logger.debug(f"✓ {' '.join(log_parts)}")
                    return best_option
                else:
                    logger.debug(f"  Reason: No valid delta found after testing {tested_count} strikes")
                    return None
            else:
                # Original behavior: use strike closest to target price
                closest_strike = min(float_strikes, key=lambda x: abs(x - target_strike))
                result = self._get_option_data(stock_conid, month_format, closest_strike, expiry_date, stock_ivhv)  # Pass stock_ivhv here too
                if not result:
                    logger.debug(f"  Reason: Could not get option data for strike ${closest_strike:.0f}")
                return result
            
        except Exception as e:
            logger.debug(f"  Reason: Exception - {str(e)}")
            return None
    
    def _get_option_data(
        self,
        stock_conid: int,
        month_format: str,
        strike: float,
        expiry_date: date,
        stock_ivhv: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Get option market data for a specific strike.
        
        Args:
            stock_conid: Stock contract ID
            month_format: Month format string for API
            strike: Strike price
            expiry_date: Expiration date
            stock_ivhv: IV/HV ratio from underlying stock (optional)
            
        Returns:
            Dictionary with option data
        """
        try:
            # Get the specific option contract
            option_params = {
                'conid': stock_conid,
                'sectype': 'OPT',
                'month': month_format,
                'exchange': 'SMART',
                'strike': str(strike),
                'right': 'P'
            }
            
            secdef_url = f"{self.client.base_url}/iserver/secdef/info"
            secdef_response = self.client._make_request('GET', secdef_url, params=option_params)
            
            if secdef_response.status_code != 200:
                logger.debug(f"Failed to get option contract: status {secdef_response.status_code}")
                return None
            
            option_data = secdef_response.json()
            
            # Find contract with matching expiry
            option_conid = None
            target_expiry_str = expiry_date.strftime('%Y%m%d')
            
            if isinstance(option_data, list):
                for contract in option_data:
                    contract_expiry = contract.get('maturityDate')
                    if contract_expiry == target_expiry_str:
                        option_conid = contract.get('conid')
                        break
                
                if not option_conid and len(option_data) > 0:
                    return None
                    
            elif isinstance(option_data, dict):
                contract_expiry = option_data.get('maturityDate')
                if contract_expiry == target_expiry_str:
                    option_conid = option_data.get('conid')
                else:
                    return None
            
            if not option_conid:
                return None
            
            # Get market data for the option
            option_snapshot = self.client.get_market_data_snapshot(
                option_conid,
                fields=[
                    '31',    # Last price
                    '84',    # Bid
                    '86',    # Ask
                    '87',    # Volume
                    '7308',  # Delta
                    '7309',  # Gamma
                    '7310',  # Vega
                    '7311',  # Theta
                    '7633',  # Implied Volatility % for this strike
                ]
            )
            
            if not option_snapshot:
                logger.debug(f"No market data snapshot received")
                return None
            
            logger.debug(f"Option data for ${strike:.0f}P: {option_snapshot}")
            
            # Parse option data
            def to_float(val):
                if val is None:
                    return None
                try:
                    if isinstance(val, str):
                        # Remove % sign, commas, and 'C' prefix
                        val = val.replace('%', '').replace(',', '').lstrip('CH')
                    return float(val)
                except (ValueError, TypeError):
                    return None
            
            # Get IV for this specific strike - field 7633 returns percentage like "19.9%"
            strike_iv = to_float(option_snapshot.get('7633'))
            if strike_iv and strike_iv > 5:
                strike_iv = strike_iv / 100.0
            
            logger.debug(f"Parsed: Strike IV={strike_iv}, Stock IV/HV={stock_ivhv}, Delta={option_snapshot.get('7308')}")
            
            result = {
                'strike': strike,
                'bid': to_float(option_snapshot.get('84')),
                'ask': to_float(option_snapshot.get('86')),
                'last': to_float(option_snapshot.get('31')),
                'delta': to_float(option_snapshot.get('7308')),
                'gamma': to_float(option_snapshot.get('7309')),
                'vega': to_float(option_snapshot.get('7310')),
                'theta': to_float(option_snapshot.get('7311')),
                'implied_vol': strike_iv,  # Use strike-specific IV
                'volume': to_float(option_snapshot.get('87')),
                'ivr': stock_ivhv  # Use calculated stock-level IV/HV ratio
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error getting option data for strike {strike}: {e}")
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
    
    def _get_stock_ivhv(self, stock_conid: int, symbol: str) -> Optional[float]:
        """
        Get IV/HV ratio for the underlying stock as a proxy for IVR.
        
        Args:
            stock_conid: Stock contract ID
            symbol: Stock symbol (for logging)
            
        Returns:
            IV/HV ratio as a percentage, or None
        """
        try:
            # Get current IV and historical volatility
            snapshot = self.client.get_market_data_snapshot(
                stock_conid,
                fields=['7283', '7088']  # 30-day IV, Historical Vol
            )
            
            if not snapshot:
                logger.debug(f"{symbol}: No volatility data available")
                return None
            
            logger.debug(f"{symbol} volatility snapshot: {snapshot}")
            
            def to_float(val):
                if val is None:
                    return None
                try:
                    if isinstance(val, str):
                        val = val.replace('%', '').replace(',', '').lstrip('CH')
                    return float(val)
                except (ValueError, TypeError):
                    return None
            
            current_iv = to_float(snapshot.get('7283'))
            historical_vol = to_float(snapshot.get('7088'))
            
            logger.debug(f"{symbol}: Current IV (7283)={current_iv}, Historical Vol (7088)={historical_vol}")
            
            if current_iv is not None and historical_vol is not None and historical_vol > 0:
                iv_hv_ratio = (current_iv / historical_vol) * 100
                logger.info(f"{symbol} IV/HV: {iv_hv_ratio:.1f}% (IV={current_iv:.1f}% / HV={historical_vol:.1f}%)")
                return iv_hv_ratio
            else:
                logger.debug(f"{symbol}: Insufficient data for IV/HV calculation")
                return None
            
        except Exception as e:
            logger.debug(f"Error getting IV/HV for {symbol}: {e}", exc_info=True)
            return None
    
    def print_results(self, results: List[PutScanResult], top_n: int = 20):
        """
        Print scan results in a formatted table.
        
        Args:
            results: List of PutScanResult objects
            top_n: Number of top results to print (default: 20)
        """
        if not results:
            print("No results found.")
            return
        
        print("\n" + "=" * 120)
        print(f"TOP {min(top_n, len(results))} HIGH IVR PUT SELLING OPPORTUNITIES")
        print("=" * 120)
        print(f"{'Symbol':<6} | {'Price':<8} | {'Strike':<8} | {'OTM%':<6} | "
              f"{'DTE':<4} | {'IVR':<5} | {'IV':<7} | {'Delta':<7} | {'Mid':<8}")
        print("-" * 120)
        
        for i, result in enumerate(results[:top_n]):
            ivr_str = f"{result.ivr:.1f}" if result.ivr is not None else "N/A"
            iv_str = f"{result.implied_vol:.1%}" if result.implied_vol is not None else "N/A"
            delta_str = f"{abs(result.delta):.3f}" if result.delta is not None else "N/A"
            mid_str = f"${result.mid:.2f}" if result.mid is not None else "N/A"
            
            print(f"{result.symbol:<6} | ${result.stock_price:>6.2f} | "
                  f"${result.strike:>6.2f} | {result.strike_pct_otm:>5.1f}% | "
                  f"{result.days_to_expiry:>3} | {ivr_str:>5} | {iv_str:>7} | "
                  f"{delta_str:>7} | {mid_str:>8}")
        
        print("=" * 120)

