"""
Utility for fetching and caching index constituent symbols.
"""

import logging
from typing import List, Dict
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class IndexConstituents:
    """Manages index constituent symbols with caching."""
    
    # Class-level in-memory cache for symbols
    _memory_cache: Dict[str, List[str]] = {}
    
    # Disk cache directory
    _CACHE_DIR = Path('data_cache/indices')
    
    # User-Agent header to avoid Wikipedia blocking
    _HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    @classmethod
    def get_constituents(cls, index: str, refresh: bool = False) -> List[str]:
        """
        Get constituent symbols for an index.
        
        Args:
            index: Index ticker (e.g., 'SPX', 'NDX', 'XLE')
            refresh: If True, fetch fresh data and update cache. If False, use cache if available.
            
        Returns:
            List of constituent ticker symbols
        """
        index_upper = index.upper()
        
        # Check in-memory cache first (unless refreshing)
        if not refresh and index_upper in cls._memory_cache:
            logger.debug(f"Using in-memory cached symbols for {index_upper}")
            return cls._memory_cache[index_upper]
        
        # Check disk cache (unless refreshing)
        if not refresh:
            cached_symbols = cls._load_from_disk_cache(index_upper)
            if cached_symbols:
                logger.info(f"Loaded {len(cached_symbols)} symbols for {index_upper} from disk cache")
                cls._memory_cache[index_upper] = cached_symbols
                return cached_symbols
        
        # Map index to fetcher method
        fetchers = {
            'SPX': cls._get_spx,
            'SP500': cls._get_spx,
            'NDX': cls._get_ndx,
            'NASDAQ100': cls._get_ndx,
            'DJI': cls._get_dji,
            'DOW': cls._get_dji,
            'XLE': cls._get_xle,
            'XLF': cls._get_xlf,
            'XLK': cls._get_xlk,
            'XLV': cls._get_xlv,
            'XLI': cls._get_xli,
            'XLP': cls._get_xlp,
            'XLY': cls._get_xly,
            'XLU': cls._get_xlu,
            'XLB': cls._get_xlb,
            'XLRE': cls._get_xlre,
        }
        
        fetcher = fetchers.get(index_upper)
        if not fetcher:
            logger.error(f"Unknown index: {index_upper}")
            logger.info(f"Available indices: {', '.join(fetchers.keys())}")
            return []
        
        # Fetch fresh data
        logger.info(f"Fetching fresh data for {index_upper}" + (" (refresh requested)" if refresh else ""))
        symbols = fetcher()
        
        if not symbols:
            logger.error(f"Failed to fetch symbols for {index_upper}")
            return []
        
        # Save to both caches
        cls._memory_cache[index_upper] = symbols
        cls._save_to_disk_cache(index_upper, symbols)
        
        return symbols
    
    @classmethod
    def clear_cache(cls, index: str = None):
        """
        Clear the symbol cache.
        
        Args:
            index: If specified, clear only this index. If None, clear all.
        """
        if index:
            index_upper = index.upper()
            # Clear from memory
            if index_upper in cls._memory_cache:
                del cls._memory_cache[index_upper]
                logger.info(f"Cleared in-memory cache for {index_upper}")
            
            # Clear from disk
            cache_file = cls._get_cache_file_path(index_upper)
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Deleted disk cache file for {index_upper}")
        else:
            # Clear all
            cls._memory_cache.clear()
            logger.info("Cleared all in-memory cache")
            
            # Clear all disk cache files
            if cls._CACHE_DIR.exists():
                for cache_file in cls._CACHE_DIR.glob('*.txt'):
                    cache_file.unlink()
                    logger.info(f"Deleted {cache_file.name}")
    
    @classmethod
    def _get_cache_file_path(cls, index: str) -> Path:
        """Get the cache file path for an index."""
        return cls._CACHE_DIR / f"{index}.txt"
    
    @classmethod
    def _load_from_disk_cache(cls, index: str) -> List[str]:
        """
        Load symbols from disk cache.
        
        Args:
            index: Index name
            
        Returns:
            List of symbols, or empty list if cache doesn't exist
        """
        cache_file = cls._get_cache_file_path(index)
        
        if not cache_file.exists():
            logger.debug(f"No disk cache found for {index}")
            return []
        
        try:
            with open(cache_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            
            logger.debug(f"Loaded {len(symbols)} symbols from {cache_file}")
            return symbols
            
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_file}: {e}")
            return []
    
    @classmethod
    def _save_to_disk_cache(cls, index: str, symbols: List[str]):
        """
        Save symbols to disk cache.
        
        Args:
            index: Index name
            symbols: List of symbols to save
        """
        # Create cache directory if it doesn't exist
        cls._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        cache_file = cls._get_cache_file_path(index)
        
        try:
            with open(cache_file, 'w') as f:
                f.write('\n'.join(symbols))
            
            logger.info(f"Saved {len(symbols)} symbols to {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_file}: {e}")
    
    @staticmethod
    def _get_spx() -> List[str]:
        """Get S&P 500 constituent symbols from Wikipedia."""
        logger.info("Fetching S&P 500 constituents from Wikipedia...")
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        try:
            tables = pd.read_html(
                url,
                attrs={'id': 'constituents'},
                storage_options=IndexConstituents._HEADERS
            )
            df = tables[0]
            
            # Extract symbols from the 'Symbol' column
            symbols = df['Symbol'].tolist()
            
            # Clean up symbols - replace periods with hyphens for IBKR compatibility
            symbols = [str(s).strip().replace('.', '-') for s in symbols]
            
            logger.info(f"Fetched {len(symbols)} S&P 500 symbols from Wikipedia")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 from Wikipedia: {e}")
            return []
    
    @staticmethod
    def _get_ndx() -> List[str]:
        """Get NASDAQ-100 constituent symbols from Wikipedia."""
        logger.info("Fetching NASDAQ-100 constituents from Wikipedia...")
        
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        
        try:
            tables = pd.read_html(url, storage_options=IndexConstituents._HEADERS)
            
            # Find the constituents table (usually has 'Ticker' or 'Symbol' column)
            for table in tables:
                if 'Ticker' in table.columns:
                    symbols = table['Ticker'].tolist()
                    symbols = [str(s).strip() for s in symbols if pd.notna(s)]
                    
                    logger.info(f"Fetched {len(symbols)} NASDAQ-100 symbols from Wikipedia")
                    return symbols
            
            logger.error("Could not find NASDAQ-100 table on Wikipedia")
            return []
            
        except Exception as e:
            logger.error(f"Failed to fetch NASDAQ-100 from Wikipedia: {e}")
            return []
    
    @staticmethod
    def _get_dji() -> List[str]:
        """Get Dow Jones Industrial Average constituent symbols from Wikipedia."""
        logger.info("Fetching Dow Jones constituents from Wikipedia...")
        
        url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
        
        try:
            tables = pd.read_html(url, storage_options=IndexConstituents._HEADERS)
            
            # Find the constituents table (usually has 'Symbol' column and exactly 30 rows)
            for table in tables:
                if 'Symbol' in table.columns and len(table) == 30:
                    symbols = table['Symbol'].tolist()
                    symbols = [str(s).strip() for s in symbols if pd.notna(s)]
                    
                    logger.info(f"Fetched {len(symbols)} Dow Jones symbols from Wikipedia")
                    return symbols
            
            logger.error("Could not find Dow Jones table on Wikipedia")
            return []
            
        except Exception as e:
            logger.error(f"Failed to fetch Dow Jones from Wikipedia: {e}")
            return []
    
    @staticmethod
    def _get_xle() -> List[str]:
        """Get XLE (Energy Select Sector) holdings from Wikipedia."""
        return IndexConstituents._get_sector_etf_holdings('XLE', 'Energy')
    
    @staticmethod
    def _get_xlf() -> List[str]:
        """Get XLF (Financial Select Sector) holdings from Wikipedia."""
        return IndexConstituents._get_sector_etf_holdings('XLF', 'Financial')
    
    @staticmethod
    def _get_xlk() -> List[str]:
        """Get XLK (Technology Select Sector) holdings from Wikipedia."""
        return IndexConstituents._get_sector_etf_holdings('XLK', 'Technology')
    
    @staticmethod
    def _get_xlv() -> List[str]:
        """Get XLV (Healthcare Select Sector) holdings from Wikipedia."""
        return IndexConstituents._get_sector_etf_holdings('XLV', 'Health_Care')
    
    @staticmethod
    def _get_xli() -> List[str]:
        """Get XLI (Industrial Select Sector) holdings from Wikipedia."""
        return IndexConstituents._get_sector_etf_holdings('XLI', 'Industrial')
    
    @staticmethod
    def _get_xlp() -> List[str]:
        """Get XLP (Consumer Staples Select Sector) holdings from Wikipedia."""
        return IndexConstituents._get_sector_etf_holdings('XLP', 'Consumer_Staples')
    
    @staticmethod
    def _get_xly() -> List[str]:
        """Get XLY (Consumer Discretionary Select Sector) holdings from Wikipedia."""
        return IndexConstituents._get_sector_etf_holdings('XLY', 'Consumer_Discretionary')
    
    @staticmethod
    def _get_xlu() -> List[str]:
        """Get XLU (Utilities Select Sector) holdings from Wikipedia."""
        return IndexConstituents._get_sector_etf_holdings('XLU', 'Utilities')
    
    @staticmethod
    def _get_xlb() -> List[str]:
        """Get XLB (Materials Select Sector) holdings from Wikipedia."""
        return IndexConstituents._get_sector_etf_holdings('XLB', 'Materials')
    
    @staticmethod
    def _get_xlre() -> List[str]:
        """Get XLRE (Real Estate Select Sector) holdings from Wikipedia."""
        return IndexConstituents._get_sector_etf_holdings('XLRE', 'Real_Estate')
    
    @staticmethod
    def _get_sector_etf_holdings(etf_symbol: str, sector_name: str) -> List[str]:
        """
        Get holdings for a sector ETF from Wikipedia S&P 500 page.
        
        Args:
            etf_symbol: ETF symbol (e.g., 'XLE')
            sector_name: Sector name on Wikipedia (e.g., 'Energy')
        
        Returns:
            List of symbols in the sector
        """
        logger.info(f"Fetching {etf_symbol} holdings from S&P 500 sector data...")
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        try:
            tables = pd.read_html(
                url,
                attrs={'id': 'constituents'},
                storage_options=IndexConstituents._HEADERS
            )
            df = tables[0]
            
            # Filter by GICS Sector column
            if 'GICS Sector' in df.columns:
                # Match sector name (case-insensitive, handle underscores)
                sector_filter = df['GICS Sector'].str.replace(' ', '_').str.lower() == sector_name.lower()
                sector_df = df[sector_filter]
                
                symbols = sector_df['Symbol'].tolist()
                symbols = [str(s).strip().replace('.', '-') for s in symbols]
                
                logger.info(f"Fetched {len(symbols)} {etf_symbol} symbols from S&P 500 {sector_name} sector")
                return symbols
            
            logger.error(f"Could not find GICS Sector column for {etf_symbol}")
            return []
            
        except Exception as e:
            logger.error(f"Failed to fetch {etf_symbol} from Wikipedia: {e}")
            return []


if __name__ == "__main__":
    """Test the IndexConstituents class."""
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("INDEX CONSTITUENTS TEST")
    print("=" * 80)
    
    # Parse command line args
    refresh = '--refresh' in sys.argv
    
    if refresh:
        print("\n⚠️  REFRESH MODE: Will fetch fresh data and update cache")
        sys.argv.remove('--refresh')
    
    # Test all available indices or specific one from command line
    test_indices = [
        'SPX', 'NDX', 'DJI', 'XLE', 'XLF', 'XLK', 'XLV', 
        'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE'
    ]
    
    # If command line arg provided, test only that index
    if len(sys.argv) > 1:
        test_indices = [sys.argv[1].upper()]
    
    results = {}
    
    for index in test_indices:
        print(f"\n{'-' * 80}")
        print(f"Testing: {index}")
        print(f"{'-' * 80}")
        
        try:
            symbols = IndexConstituents.get_constituents(index, refresh=refresh)
            
            if symbols:
                results[index] = len(symbols)
                print(f"✓ {index}: Found {len(symbols)} symbols")
                
                # Show first 10 symbols
                print(f"  First 10: {', '.join(symbols[:10])}")
                
                # Test caching (should use cache on second call unless refresh=True)
                print(f"  Testing cache...")
                cached_symbols = IndexConstituents.get_constituents(index, refresh=False)
                if cached_symbols == symbols:
                    print(f"  ✓ Cache working correctly")
                else:
                    print(f"  ✗ Cache mismatch!")
            else:
                print(f"✗ {index}: No symbols found")
                results[index] = 0
                
        except Exception as e:
            print(f"✗ {index}: Error - {e}")
            results[index] = -1
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Index':<15} {'Symbols':<10} {'Status':<10}")
    print(f"{'-' * 80}")
    
    for index, count in results.items():
        if count > 0:
            status = "✓ OK"
        elif count == 0:
            status = "✗ EMPTY"
        else:
            status = "✗ ERROR"
        
        print(f"{index:<15} {count:<10} {status:<10}")
    
    # Show cache files
    print(f"\n{'-' * 80}")
    print("DISK CACHE FILES")
    print(f"{'-' * 80}")
    cache_dir = IndexConstituents._CACHE_DIR
    if cache_dir.exists():
        cache_files = list(cache_dir.glob('*.txt'))
        if cache_files:
            for cache_file in sorted(cache_files):
                size = cache_file.stat().st_size
                print(f"  {cache_file.name:<20} ({size} bytes)")
        else:
            print("  No cache files found")
    else:
        print(f"  Cache directory does not exist: {cache_dir}")
    
    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}")
    print("\nUsage:")
    print("  python3 utils/index_constituents.py              # Test all indices (use cache)")
    print("  python3 utils/index_constituents.py SPX          # Test specific index (use cache)")
    print("  python3 utils/index_constituents.py --refresh    # Refresh all indices")
    print("  python3 utils/index_constituents.py SPX --refresh # Refresh specific index")