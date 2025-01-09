# src/agents/base.py
from typing import Dict, List, Optional
from datetime import datetime
import logging
import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base agent with core capabilities."""

    def __init__(
        self,
        llm_config: Optional[Dict] = None,
        memory_size: int = 1000,
        objectives: List[str] = None
    ):
        """Initialize base agent.

        Args:
            llm_config: Configuration for LLM. If None, uses defaults.
        """
        self.llm_config = llm_config or {}
        self.client = AsyncOpenAI(api_key=self.llm_config.get('api_key'))
        self.memory = []
        self.objectives = objectives or []
        self.last_thought = None

    async def think(self, context: Dict) -> Dict:
        """Core thinking process."""
        try:
            # Format messages for LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert crypto trading AI assistant.
                    Analyze market data and provide clear, actionable insights focused on:
                    - Technical analysis
                    - Risk assessment
                    - Market sentiment
                    - Trading opportunities"""
                },
                {
                    "role": "user",
                    "content": f"Analyze the following market context and provide insights:\n{context}"
                }
            ]

            # Get LLM response
            response = await self.client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper responses
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )

            thought = response.choices[0].message.content
            return {
                "thought": thought,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in thinking process: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def close(self):
        """Cleanup resources."""
        pass

    async def learn(self, experience: Dict):
        """Learn from experience and update memory."""
        self.memory.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'experience',
            'data': experience
        })

        if len(self.memory) > 1000:  # Keep memory size limited
            self.memory.pop(0)

        # Analyze experience for learning
        analysis = await self.think({
            'type': 'learning',
            'experience': experience
        })

        # Update objectives if needed
        if analysis.get('update_objectives'):
            self.objectives = self._update_objectives(analysis['update_objectives'])

    def _update_objectives(self, updates: List[str]) -> List[str]:
        """Update agent objectives based on learning."""
        current = set(self.objectives)
        new = set(updates)

        # Keep important objectives, add new ones
        return list(current.union(new))[:5]  # Keep top 5 objectives