# src/agents/hedge_fund.py
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .base import BaseAgent
from tools import CryptoDataTools, get_prices
from executors.jupiter_client import JupiterClient
from sentiment.market_sentiment import MarketSentimentAnalyzer

logger = logging.getLogger(__name__)

DEFAULT_MARKET_DATA = {
    'price': 0.0,
    'volume': 0.0,
    'liquidity': 0.0,
    'holders': 0,
    'transactions': 0,
    'error': 'Using default data due to API error'
}

class HedgeFundAgent(BaseAgent):
    """Autonomous hedge fund agent."""

    def __init__(
        self,
        initial_capital: float,
        trading_pairs: List[str],
        risk_tolerance: float = 0.7,
        llm_config: Optional[Dict] = None,
        sentiment_config: Optional[Dict] = None
    ):
        super().__init__(llm_config)
        self.initial_capital = initial_capital
        self.trading_pairs = trading_pairs
        self.risk_tolerance = risk_tolerance

        # Initialize components
        self.data_tools = CryptoDataTools()
        self.jupiter = JupiterClient()
        self.sentiment_analyzer = MarketSentimentAnalyzer(sentiment_config or {})

        # Portfolio state
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},
            'total_value': initial_capital
        }

    async def analyze_market(
        self,
        tokens: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """Analyze market conditions for given tokens."""
        # Set default time range if not provided
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(hours=24))

        market_data = {}

        for token in tokens:
            try:
                # Get price data
                metrics = await get_prices(
                    token,
                    start_date=start_date,
                    end_date=end_date
                )

                # Get sentiment data separately
                sentiment = await self.sentiment_analyzer.get_sentiment(token)

                market_data[token] = {
                    'price': metrics.price,
                    'volume': metrics.volume,
                    'liquidity': metrics.liquidity,
                    'sentiment': sentiment,
                    'time_range': {
                        'start': start_date.isoformat(),
                        'end': end_date.isoformat()
                    }
                }
            except Exception as e:
                logger.error(f"Error getting metrics for {token}: {e}")
                market_data[token] = DEFAULT_MARKET_DATA.copy()

        # Generate thought about market conditions
        analysis = await self.think({
            'type': 'market_analysis',
            'data': market_data,
            'portfolio': self.portfolio,
            'timestamp': datetime.now().isoformat()
        })

        result = {
            'market_data': market_data,
            'analysis': analysis.get('thought', ''),
            'timestamp': datetime.now().isoformat(),
            'trades': []  # Initialize empty trades list
        }

        # Try to generate trades from analysis
        try:
            trades = await self.generate_trades_from_analysis(result)
            result['trades'] = trades
        except Exception as e:
            logger.error(f"Error generating trades: {e}")
            result['trades'] = []

        return result

    async def generate_trades_from_analysis(self, analysis: Dict) -> List[Dict]:
        """Generate trades based on market analysis."""
        trades = []

        for token, data in analysis['market_data'].items():
            if data.get('error'):
                continue  # Skip tokens with errors

            # Default conservative position size (1% of portfolio)
            position_size = self.portfolio['total_value'] * 0.01

            # Simple trading logic if LLM is not available
            if data['price'] > 0:
                price_24h_change = data.get('price_change_24h', 0)

                if price_24h_change > 5:  # 5% up
                    trades.append({
                        'token': token,
                        'action': 'sell',
                        'amount': position_size,
                        'confidence': 0.6,
                        'reasoning': f"Price up {price_24h_change}% in 24h"
                    })
                elif price_24h_change < -5:  # 5% down
                    trades.append({
                        'token': token,
                        'action': 'buy',
                        'amount': position_size,
                        'confidence': 0.6,
                        'reasoning': f"Price down {price_24h_change}% in 24h"
                    })

        return trades

    async def generate_trades(self, analysis: Dict) -> List[Dict]:
        """Generate trading decisions based on analysis."""
        trades = []

        for token in self.trading_pairs:
            token_data = analysis['market_data'].get(token, {})
            if 'error' not in token_data:
                # Get trade decision
                decision = await self.think({
                    'type': 'trade_decision',
                    'token': token,
                    'data': token_data,
                    'analysis': analysis['analysis'],
                    'portfolio': self.portfolio
                })

                if decision.get('action') in ['buy', 'sell']:
                    trades.append({
                        'token': token,
                        'action': decision['action'],
                        'amount': self.calculate_trade_size(
                            token,
                            decision.get('confidence', 0.5),
                            token_data
                        ),
                        'confidence': decision.get('confidence', 0.5),
                        'reasoning': decision.get('reasoning', '')
                    })

        return trades

    def calculate_trade_size(
        self,
        token: str,
        confidence: float,
        market_data: Dict
    ) -> float:
        """Calculate trade size based on multiple factors."""
        # Base position size (% of portfolio)
        max_position = self.portfolio['total_value'] * 0.2  # 20% max position

        # Scale by confidence
        position_size = max_position * confidence

        # Scale by liquidity
        liquidity = market_data.get('liquidity', 0)
        if liquidity > 0:
            # Limit to 10% of available liquidity
            position_size = min(position_size, liquidity * 0.1)

        # Ensure we have enough cash for buys
        if position_size > self.portfolio['cash']:
            position_size = self.portfolio['cash']

        return position_size

    async def execute_trades(self, trades: List[Dict]) -> Dict:
        """Execute validated trades."""
        results = {}

        for trade in trades:
            try:
                # Execute trade through Jupiter
                result = await self.jupiter.execute_trade(
                    input_token=trade['token'],
                    output_token='USDC',
                    amount=trade['amount'],
                    exact_out=trade['action'] == 'sell'
                )

                if result['success']:
                    # Update portfolio
                    self.update_portfolio(trade, result)

                results[trade['token']] = result

            except Exception as e:
                logger.error(f"Trade execution error: {e}")
                results[trade['token']] = {
                    'success': False,
                    'error': str(e)
                }

        return results

    def update_portfolio(self, trade: Dict, result: Dict):
        """Update portfolio after successful trade."""
        token = trade['token']
        amount = float(trade['amount'])
        price = float(result['executed_price'])

        if trade['action'] == 'buy':
            self.portfolio['cash'] -= amount * price
            self.portfolio['positions'][token] = \
                self.portfolio['positions'].get(token, 0) + amount
        else:
            self.portfolio['cash'] += amount * price
            self.portfolio['positions'][token] = \
                self.portfolio['positions'].get(token, 0) - amount

        # Update total value
        self.calculate_total_value()

    def calculate_total_value(self):
        """Calculate total portfolio value."""
        total = self.portfolio['cash']

        for token, amount in self.portfolio['positions'].items():
            # Get current price
            try:
                price = float(self.get_current_price(token))
                total += amount * price
            except Exception as e:
                logger.error(f"Error getting price for {token}: {e}")

        self.portfolio['total_value'] = total

    async def get_current_price(self, token: str) -> float:
        """Get current token price."""
        try:
            price = await self.jupiter.get_price(token)
            return float(price) if price else 0.0
        except Exception as e:
            logger.error(f"Error getting price for {token}: {e}")
            return 0.0