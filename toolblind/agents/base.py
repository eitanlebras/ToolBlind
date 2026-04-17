"""Abstract agent interface for ToolBlind benchmark."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from toolblind.dataset.tasks import Tool
from toolblind.environment.simulator import ToolUnavailableError
from toolblind.environment.trajectory import CompletedStep


@dataclass
class AgentAction:
    """An action chosen by an agent at a given step."""

    action_type: str  # CALL_TOOL, HALT, REQUEST_CLARIFICATION
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    reasoning: str = ""
    halt_reason: Optional[str] = None
    confidence: float = 0.5
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "reasoning": self.reasoning,
            "halt_reason": self.halt_reason,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentAction":
        """Deserialize from dictionary."""
        return cls(
            action_type=d.get("action_type", "HALT"),
            tool_name=d.get("tool_name"),
            tool_params=d.get("tool_params"),
            reasoning=d.get("reasoning", ""),
            halt_reason=d.get("halt_reason"),
            confidence=d.get("confidence", 0.5),
            tokens_used=d.get("tokens_used", 0),
        )


def build_system_prompt(
    available_tools: List[Tool],
    cot: bool = True,
) -> str:
    """Build the system prompt describing available tools and expected response format."""
    tool_descriptions = []
    for tool in available_tools:
        params_str = ", ".join(f"{k}: {v}" for k, v in tool.input_schema.items())
        tool_descriptions.append(
            f"- {tool.name}({params_str}) -> {tool.output_type}\n"
            f"  Description: {tool.description}\n"
            f"  Category: {tool.functional_category} | Tags: {', '.join(tool.semantic_tags)}"
        )

    tools_block = "\n".join(tool_descriptions)

    cot_instruction = ""
    if cot:
        cot_instruction = (
            "\nBefore choosing an action, think step-by-step about:\n"
            "1. What does the current step require functionally?\n"
            "2. Which available tools could accomplish this function?\n"
            "3. If the required tool is unavailable, can any available tool(s) substitute?\n"
            "4. If no substitute exists, should you halt and explain why?\n"
        )

    prompt = f"""You are an AI agent executing a multi-step task. You have access to the following tools:

{tools_block}

IMPORTANT: You can ONLY use the tools listed above. If a tool is not listed, it is NOT available to you.
If a required tool is unavailable, you must either:
- Use an available tool that can accomplish the same function (SUBSTITUTE)
- Compose multiple available tools to approximate the function (DECOMPOSE)
- Halt and explain why the task cannot be completed (HALT)

Do NOT pretend a tool exists if it is not in your available tools list.
Do NOT call a tool that is not listed above.
{cot_instruction}
Respond with a JSON object:
{{
  "action_type": "CALL_TOOL" | "HALT" | "REQUEST_CLARIFICATION",
  "tool_name": "<tool name if CALL_TOOL, null otherwise>",
  "tool_params": {{<parameters if CALL_TOOL>}},
  "reasoning": "<your step-by-step reasoning>",
  "halt_reason": "<explanation if HALT, null otherwise>",
  "confidence": <0.0 to 1.0>
}}"""
    return prompt


def build_step_prompt(
    goal: str,
    trajectory_so_far: List[CompletedStep],
    current_step_description: str,
    tool_error: Optional[ToolUnavailableError] = None,
) -> str:
    """Build the user prompt for a specific step."""
    parts = [f"GOAL: {goal}\n"]

    if trajectory_so_far:
        parts.append("COMPLETED STEPS:")
        for step in trajectory_so_far:
            result_summary = str(step.result.output)
            if len(result_summary) > 300:
                result_summary = result_summary[:300] + "..."
            parts.append(
                f"  Step {step.step_index + 1}: Called {step.tool_called} -> "
                f"{step.result.output_type} (success={step.result.success})"
            )
        parts.append("")

    parts.append(f"CURRENT STEP: {current_step_description}")

    if tool_error:
        parts.append(
            f"\n** TOOL ERROR: The tool '{tool_error.tool_name}' is unavailable. "
            f"Reason: {tool_error.reason} **\n"
            f"You must decide how to proceed. Review your available tools and determine if you can:\n"
            f"1. Use a different available tool that accomplishes the same function\n"
            f"2. Compose multiple available tools to approximate the function\n"
            f"3. Halt because no available tools can accomplish this step\n"
        )

    parts.append("\nWhat is your next action? Respond with the JSON object as specified.")
    return "\n".join(parts)


def parse_agent_response(response_text: str) -> AgentAction:
    """Parse an agent's response text into an AgentAction, handling malformed JSON."""
    # Try to extract JSON from the response
    text = response_text.strip()

    # Try direct JSON parse first
    try:
        data = json.loads(text)
        return AgentAction.from_dict(data)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in the response
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        try:
            data = json.loads(text[json_start:json_end])
            return AgentAction.from_dict(data)
        except json.JSONDecodeError:
            pass

    # Try to find JSON in code blocks
    for marker in ["```json", "```"]:
        if marker in text:
            block_start = text.find(marker) + len(marker)
            block_end = text.find("```", block_start)
            if block_end > block_start:
                try:
                    data = json.loads(text[block_start:block_end].strip())
                    return AgentAction.from_dict(data)
                except json.JSONDecodeError:
                    pass

    # Fallback: try to infer action from text
    text_lower = text.lower()
    if any(word in text_lower for word in ["halt", "cannot", "impossible", "infeasible", "unable"]):
        return AgentAction(
            action_type="HALT",
            reasoning=text,
            halt_reason=text[:500],
            confidence=0.3,
        )

    # Default: treat as a confabulation attempt
    return AgentAction(
        action_type="CALL_TOOL",
        reasoning=text,
        confidence=0.1,
    )


class BaseAgent(ABC):
    """Abstract base class for all ToolBlind agents."""

    @abstractmethod
    def plan_step(
        self,
        goal: str,
        available_tools: List[Tool],
        trajectory_so_far: List[CompletedStep],
        current_step_description: str,
        tool_error: Optional[ToolUnavailableError] = None,
    ) -> AgentAction:
        """Plan the next action for a given step.

        Args:
            goal: The overall task goal.
            available_tools: Tools the agent can use.
            trajectory_so_far: Steps already completed.
            current_step_description: What this step should accomplish.
            tool_error: If set, the required tool is unavailable.

        Returns:
            AgentAction describing the agent's chosen action.
        """

    @abstractmethod
    def name(self) -> str:
        """Return the agent's display name."""
