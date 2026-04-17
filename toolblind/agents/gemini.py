"""Gemini 1.5 Pro agent implementation for ToolBlind benchmark."""

import time
from typing import List, Optional

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

logger = get_logger("agent.gemini")


class GeminiAgent(BaseAgent):
    """Agent powered by Gemini 1.5 Pro via Google GenerativeAI SDK."""

    def __init__(self, cot: bool = True, model: str = "gemini-1.5-pro"):
        """Initialize the Gemini agent.

        Args:
            cot: Whether to use chain-of-thought prompting.
            model: Model identifier to use.
        """
        self._cot = cot
        self._model = model
        self._config = get_config()
        self._genai = None
        self._gen_model = None
        self._cache = get_cache()

    def _get_model(self):
        """Lazily initialize the Gemini model."""
        if self._gen_model is None:
            import google.generativeai as genai
            genai.configure(api_key=self._config.google_api_key)
            self._genai = genai
            self._gen_model = genai.GenerativeModel(self._model)
        return self._gen_model

    def name(self) -> str:
        """Return the agent's display name."""
        mode = "cot" if self._cot else "direct"
        return f"gemini-1.5-pro-{mode}"

    def plan_step(
        self,
        goal: str,
        available_tools: List[Tool],
        trajectory_so_far: List[CompletedStep],
        current_step_description: str,
        tool_error: Optional[ToolUnavailableError] = None,
    ) -> AgentAction:
        """Plan the next action using Gemini 1.5 Pro."""
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
                model = self._get_model()
                full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
                response = model.generate_content(
                    full_prompt,
                    generation_config={"max_output_tokens": 2048, "temperature": 0.1},
                )
                response_text = response.text or ""
                # Gemini SDK provides usage metadata differently
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    tokens_used = getattr(um, "prompt_token_count", 0) + getattr(um, "candidates_token_count", 0)
                break
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str or "quota" in error_str:
                    delay = self._config.retry_base_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                else:
                    logger.error(f"Gemini API error: {e}")
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
