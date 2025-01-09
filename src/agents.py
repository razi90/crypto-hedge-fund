import sys
import os
import argparse
import asyncio
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, Sequence, TypedDict, List, Tuple
import logging

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

logger = logging.getLogger(__name__)

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

        # Configure analysis interval (in seconds)
        ANALYSIS_INTERVAL = 60  # Analyze every minute

        while True:
            try:
                # Set time range for analysis
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=24)

                # Analyze market for all trading pairs
                analysis_result = await self.analyze_market(
                    self.trading_pairs,
                    start_date=start_time,
                    end_date=end_time
                )

                # Generate potential trades based on analysis
                if 'trades' in analysis_result:
                    for trade in analysis_result['trades']:
                        try:
                            # Execute trades that meet confidence threshold
                            if trade['confidence'] > 0.7:  # Configurable threshold
                                result = await self.execute_trades([trade])
                                logger.info(f"Trade executed: {result}")

                                # Update portfolio after successful trade
                                if result.get(trade['token'], {}).get('success'):
                                    self.update_portfolio(trade, result[trade['token']])
                        except Exception as e:
                            logger.error(f"Error executing trade: {e}")

                # Log analysis results
                logger.info(f"Market Analysis: {analysis_result.get('analysis', '')}")
                logger.info(f"Current Portfolio: {self.portfolio}")

                # Wait for next analysis interval
                print(f"Waiting {ANALYSIS_INTERVAL} seconds until next analysis...")
                await asyncio.sleep(ANALYSIS_INTERVAL)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)  # Wait a bit before retrying on error

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