"""GPT-4o agent implementation for ToolBlind benchmark."""

import time
from typing import List, Optional

import openai

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

logger = get_logger("agent.openai")


class OpenAIAgent(BaseAgent):
    """Agent powered by GPT-4o via OpenAI SDK."""

    def __init__(self, cot: bool = True, model: str = "gpt-4o"):
        """Initialize the OpenAI agent.

        Args:
            cot: Whether to use chain-of-thought prompting.
            model: Model identifier to use.
        """
        self._cot = cot
        self._model = model
        self._config = get_config()
        self._client: Optional[openai.OpenAI] = None
        self._cache = get_cache()

    def _get_client(self) -> openai.OpenAI:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            self._client = openai.OpenAI(api_key=self._config.openai_api_key)
        return self._client

    def name(self) -> str:
        """Return the agent's display name."""
        mode = "cot" if self._cot else "direct"
        return f"{self._model}-{mode}"

    def plan_step(
        self,
        goal: str,
        available_tools: List[Tool],
        trajectory_so_far: List[CompletedStep],
        current_step_description: str,
        tool_error: Optional[ToolUnavailableError] = None,
    ) -> AgentAction:
        """Plan the next action using GPT-4o."""
        system_prompt = build_system_prompt(available_tools, cot=self._cot)
        user_prompt = build_step_prompt(goal, trajectory_so_far, current_step_description, tool_error)
        prompt_key = system_prompt + "|||" + user_prompt

        # Check cache
        cached = self._cache.get(self._model, prompt_key)
        if cached is not None:
            logger.debug("Using cached response")
            return AgentAction.from_dict(cached)

        response_text = ""
        tokens_used = 0
        for attempt in range(self._config.max_retries):
            try:
                client = self._get_client()
                # Newer models (gpt-5.x, o3, o4) require max_completion_tokens
                token_param = {}
                if any(self._model.startswith(p) for p in ("gpt-5", "o3", "o4")):
                    token_param["max_completion_tokens"] = 2048
                else:
                    token_param["max_tokens"] = 2048
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    **token_param,
                )
                response_text = response.choices[0].message.content or ""
                usage = response.usage
                if usage:
                    tokens_used = usage.prompt_tokens + usage.completion_tokens
                break
            except openai.RateLimitError:
                delay = self._config.retry_base_delay * (2 ** attempt)
                logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1})")
                time.sleep(delay)
            except openai.APIError as e:
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

        self._cache.put(self._model, prompt_key, action.to_dict())
        return action
