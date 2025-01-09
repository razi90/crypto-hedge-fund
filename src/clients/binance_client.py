from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
from typing import Dict, Optional
import urllib3

logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class BinanceClient:
    """Binance API client wrapper."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.client = Client(api_key, api_secret)

    async def get_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            # Convert symbol to Binance format (e.g., "BTCUSDT")
            binance_symbol = f"{symbol}USDT"
            ticker = self.client.get_symbol_ticker(symbol=binance_symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Binance API error for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            raise

    async def get_market_depth(self, symbol: str) -> Dict:
        """Get market depth data."""
        try:
            binance_symbol = f"{symbol}USDT"
            depth = self.client.get_order_book(symbol=binance_symbol)
            return {
                'bids': depth['bids'],
                'asks': depth['asks']
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting depth for {symbol}: {e}")
            raise

    async def get_24h_stats(self, symbol: str) -> Dict:
        """Get 24-hour statistics."""
        try:
            binance_symbol = f"{symbol}USDT"
            stats = self.client.get_ticker(symbol=binance_symbol)
            return {
                'volume': float(stats['volume']),
                'price_change': float(stats['priceChange']),
                'price_change_percent': float(stats['priceChangePercent']),
                'weighted_avg_price': float(stats['weightedAvgPrice']),
                'last_price': float(stats['lastPrice']),
                'last_qty': float(stats['lastQty']),
                'open_price': float(stats['openPrice']),
                'high_price': float(stats['highPrice']),
                'low_price': float(stats['lowPrice']),
                'volume': float(stats['volume']),
                'quote_volume': float(stats['quoteVolume'])
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting stats for {symbol}: {e}")
            raise

    async def get_historical_prices(self, symbol: str, interval: str = '1d', limit: int = 100) -> list:
        """Get historical kline/candlestick data."""
        try:
            binance_symbol = f"{symbol}USDT"
            klines = self.client.get_klines(
                symbol=binance_symbol,
                interval=interval,
                limit=limit
            )
            return [
                {
                    'timestamp': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': k[6],
                    'quote_volume': float(k[7]),
                    'trades': k[8],
                    'taker_buy_base': float(k[9]),
                    'taker_buy_quote': float(k[10])
                }
                for k in klines
            ]
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting klines for {symbol}: {e}")
            raise

    async def close(self):
        """Close any open connections."""
        pass  # Binance client doesn't need explicit cleanup