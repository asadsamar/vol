"""
Market data book (top of book / level 1 quotes).
"""
from typing import Optional
from dataclasses import dataclass


@dataclass
class Book:
    """
    Top of book (level 1) market data.
    
    Represents the best bid and ask prices with sizes.
    """
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    last: Optional[float] = None
    last_size: Optional[int] = None
    timestamp: Optional[float] = None
    
    @property
    def mid(self) -> Optional[float]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None
    
    @property
    def spread_pct(self) -> Optional[float]:
        """Calculate bid-ask spread as percentage of mid."""
        mid = self.mid
        spread = self.spread
        if mid and spread and mid > 0:
            return (spread / mid) * 100.0
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Calculate bid-ask spread in basis points."""
        spread_pct = self.spread_pct
        if spread_pct is not None:
            return spread_pct * 100.0
        return None
    
    @property
    def has_quotes(self) -> bool:
        """Check if we have valid bid/ask quotes."""
        return self.bid is not None and self.ask is not None
    
    @property
    def has_trade(self) -> bool:
        """Check if we have a last trade."""
        return self.last is not None
    
    @property
    def best_price(self) -> Optional[float]:
        """Get best available price (mid, last, or None)."""
        return self.mid or self.last
    
    def update(
        self,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        last: Optional[float] = None,
        bid_size: Optional[int] = None,
        ask_size: Optional[int] = None,
        last_size: Optional[int] = None,
        timestamp: Optional[float] = None
    ):
        """
        Update book with new data.
        Only updates fields that are provided (not None).
        """
        if bid is not None:
            self.bid = float(bid)
        if ask is not None:
            self.ask = float(ask)
        if last is not None:
            self.last = float(last)
        if bid_size is not None:
            self.bid_size = int(bid_size)
        if ask_size is not None:
            self.ask_size = int(ask_size)
        if last_size is not None:
            self.last_size = int(last_size)
        if timestamp is not None:
            self.timestamp = float(timestamp)
    
    def __str__(self) -> str:
        """String representation."""
        if self.has_quotes:
            parts = [f"Bid: ${self.bid:.2f}"]
            if self.bid_size:
                parts[0] += f" x {self.bid_size}"
            
            parts.append(f"Ask: ${self.ask:.2f}")
            if self.ask_size:
                parts[1] += f" x {self.ask_size}"
            
            parts.append(f"Mid: ${self.mid:.2f}")
            
            if self.spread is not None:
                parts.append(f"Spread: ${self.spread:.4f} ({self.spread_bps:.0f}bps)")
            
            return " | ".join(parts)
        elif self.has_trade:
            result = f"Last: ${self.last:.2f}"
            if self.last_size:
                result += f" x {self.last_size}"
            return result
        return "No quotes available"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"Book(bid={self.bid}, ask={self.ask}, last={self.last}, "
                f"bid_size={self.bid_size}, ask_size={self.ask_size})")