import sys
import os
import argparse
import asyncio
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, Sequence, TypedDict, List, Tuple

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.tools import calculate_bollinger_bands, calculate_intrinsic_value, calculate_macd, calculate_obv, calculate_rsi, search_line_items, get_financial_metrics, get_insider_trades, get_market_cap, get_prices, prices_to_df
from src.agents.base import BaseAgent

class TradingAgent(BaseAgent):
    def __init__(self, capital: float, trading_pairs: List[str], risk_factor: float, dry_run: bool, interval: int, show_reasoning: bool):
        super().__init__()  # Initialize the base agent with LLM capabilities
        self.capital = capital
        self.trading_pairs = trading_pairs
        self.risk_factor = risk_factor
        self.dry_run = dry_run
        self.interval = interval
        self.show_reasoning = show_reasoning

    async def run(self):
        print(f"Starting trading agent with {self.capital} USDC")

        while True:
            # Get market state for all trading pairs
            market_state = {}
            for token in self.trading_pairs:
                # Get current time range
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=24)

                try:
                    # Get current metrics
                    metrics = await get_prices(
                        token,
                        start_date=start_time,
                        end_date=end_time
                    )

                    # Get lookback data
                    lookback_data = await get_prices(
                        token,
                        start_date=start_time,
                        end_date=end_time
                    )

                    market_state[token] = {
                        "price": metrics.price,
                        "volume": metrics.volume,
                        "liquidity": metrics.liquidity,
                        "history": lookback_data
                    }
                except Exception as e:
                    print(f"Error getting data for {token}: {e}")
                    continue

            # Current portfolio state
            portfolio = {
                "cash": self.capital,
                "tokens": {pair: 0 for pair in self.trading_pairs}  # You'll need to track actual balances
            }

            # Get trading decisions
            decisions = await self.generate_trading_signals(
                market_state=market_state,
                portfolio=portfolio
            )

            # Execute trades (if not dry run)
            if not self.dry_run:
                for token, decision in decisions.items():
                    print(f"Would execute: {decision['action']} {decision['quantity']} {token}")
                    # TODO: Implement actual trade execution

            await asyncio.sleep(self.interval)

    async def generate_trading_signals(self, market_state: Dict, portfolio: Dict) -> Dict:
        """Generate trading signals based on market state and LLM analysis."""
        # Get LLM analysis of market conditions
        analysis = await self.think({
            'type': 'market_analysis',
            'market_state': market_state,
            'portfolio': portfolio,
            'risk_factor': self.risk_factor,
            'timestamp': datetime.now().isoformat()
        })

        decisions = {}

        for token in self.trading_pairs:
            token_data = market_state.get(token, {})

            if token_data.get('price', 0) > 0:
                position_size = self.capital * 0.01 * self.risk_factor

                # Combine simple price analysis with LLM insights
                price_change = token_data.get('price_change_24h', 0)

                if price_change > 5:  # 5% up
                    decisions[token] = {
                        "action": "sell",
                        "quantity": position_size,
                        "reasoning": f"Price up {price_change}% in 24h\nLLM Analysis: {analysis.get('thought', '')}"
                    }
                elif price_change < -5:  # 5% down
                    decisions[token] = {
                        "action": "buy",
                        "quantity": position_size,
                        "reasoning": f"Price down {price_change}% in 24h\nLLM Analysis: {analysis.get('thought', '')}"
                    }

        if self.show_reasoning:
            print(f"\nLLM Market Analysis:\n{analysis.get('thought', '')}")
            for token, decision in decisions.items():
                print(f"\nAnalysis for {token}:")
                print(f"Action: {decision['action']}")
                print(f"Quantity: {decision['quantity']}")
                print(f"Reasoning: {decision['reasoning']}")

        return decisions

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the trading agent."""
    parser = argparse.ArgumentParser(description='Crypto Trading Agent')

    # Required arguments
    parser.add_argument(
        '--pairs',
        nargs='+',
        required=True,
        help='Trading pairs to monitor (e.g., SOL BONK JUP)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        required=True,
        help='Initial capital in USDC'
    )

    # Optional arguments
    parser.add_argument(
        '--risk',
        type=float,
        default=0.5,
        help='Risk factor (0.0-1.0)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Trading interval in seconds'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in simulation mode without real trades'
    )

    parser.add_argument(
        '--show-reasoning',
        action='store_true',
        help='Show AI reasoning for trades'
    )

    # Validate arguments
    args = parser.parse_args()
    validate_args(args)

    return args

def validate_args(args: argparse.Namespace):
    """Validate parsed arguments."""
    if args.capital <= 0:
        raise ValueError("Capital must be positive")

    if not 0 <= args.risk <= 1:
        raise ValueError("Risk must be between 0 and 1")

    if args.interval < 10:
        raise ValueError("Interval must be at least 10 seconds")

    for pair in args.pairs:
        if pair not in ['SOL', 'BONK', 'JUP']:
            raise ValueError(f"Unsupported trading pair: {pair}")

if __name__ == "__main__":
    args = parse_args()
    trading_agent = TradingAgent(
        capital=args.capital,
        trading_pairs=args.pairs,
        risk_factor=args.risk,
        dry_run=args.dry_run,
        interval=args.interval,
        show_reasoning=args.show_reasoning
    )
    asyncio.run(trading_agent.run())