"""Claude Sonnet agent implementation for ToolBlind benchmark."""

import time
from typing import List, Optional

import anthropic

from toolblind.agents.base import (
    AgentAction,
    BaseAgent,
    build_step_prompt,
    build_system_prompt,
    parse_agent_response,
)
from toolblind.dataset.tasks import Tool
from toolblind.environment.simulator import ToolUnavailableError
from toolblind.environment.trajectory import CompletedStep
from toolblind.utils.cache import get_cache
from toolblind.utils.config import get_config
from toolblind.utils.logging import get_logger

logger = get_logger("agent.claude")


class ClaudeAgent(BaseAgent):
    """Agent powered by Claude Sonnet via Anthropic SDK."""

    def __init__(self, cot: bool = True, model: str = "claude-sonnet-4-20250514"):
        """Initialize the Claude agent.

        Args:
            cot: Whether to use chain-of-thought prompting.
            model: Model identifier to use.
        """
        self._cot = cot
        self._model = model
        self._config = get_config()
        self._client: Optional[anthropic.Anthropic] = None
        self._cache = get_cache()

    def _get_client(self) -> anthropic.Anthropic:
        """Lazily initialize the Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._config.anthropic_api_key)
        return self._client

    def name(self) -> str:
        """Return the agent's display name."""
        mode = "cot" if self._cot else "direct"
        # Strip date suffix for cleaner display (e.g. claude-opus-4-5-20250415 -> claude-opus-4-5)
        display = self._model.rsplit("-", 1)[0] if self._model.count("-") > 3 else self._model
        return f"{display}-{mode}"

    def plan_step(
        self,
        goal: str,
        available_tools: List[Tool],
        trajectory_so_far: List[CompletedStep],
        current_step_description: str,
        tool_error: Optional[ToolUnavailableError] = None,
    ) -> AgentAction:
        """Plan the next action using Claude Sonnet."""
        system_prompt = build_system_prompt(available_tools, cot=self._cot)
        user_prompt = build_step_prompt(goal, trajectory_so_far, current_step_description, tool_error)
        prompt_key = system_prompt + "|||" + user_prompt

        # Check cache
        cached = self._cache.get(self._model, prompt_key)
        if cached is not None:
            logger.debug("Using cached response")
            return AgentAction.from_dict(cached)

        # Call API with retries
        response_text = ""
        tokens_used = 0
        for attempt in range(self._config.max_retries):
            try:
                client = self._get_client()
                response = client.messages.create(
                    model=self._model,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                response_text = response.content[0].text
                tokens_used = response.usage.input_tokens + response.usage.output_tokens
                break
            except anthropic.RateLimitError:
                delay = self._config.retry_base_delay * (2 ** attempt)
                logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1})")
                time.sleep(delay)
            except anthropic.APIError as e:
                logger.error(f"API error: {e}")
                if attempt == self._config.max_retries - 1:
                    return AgentAction(
                        action_type="HALT",
                        reasoning=f"API error after {self._config.max_retries} attempts: {e}",
                        halt_reason=f"API error: {e}",
                        tokens_used=0,
                    )
                delay = self._config.retry_base_delay * (2 ** attempt)
                time.sleep(delay)

        action = parse_agent_response(response_text)
        action.tokens_used = tokens_used

        # Cache the result
        self._cache.put(self._model, prompt_key, action.to_dict())

        return action
