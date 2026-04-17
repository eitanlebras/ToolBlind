"""ReAct prompting wrapper for ToolBlind agents."""

from typing import List, Optional

from toolblind.agents.base import AgentAction, BaseAgent
from toolblind.dataset.tasks import Tool
from toolblind.environment.simulator import ToolUnavailableError
from toolblind.environment.trajectory import CompletedStep
from toolblind.utils.logging import get_logger

logger = get_logger("agent.react")


class ReActWrapper(BaseAgent):
    """Wraps any BaseAgent in ReAct (Thought/Action/Observation) prompting format."""

    def __init__(self, inner_agent: BaseAgent):
        """Initialize the ReAct wrapper around an inner agent.

        Args:
            inner_agent: The base agent to wrap with ReAct prompting.
        """
        self._inner = inner_agent

    def name(self) -> str:
        """Return the agent's display name with ReAct prefix."""
        return f"react-{self._inner.name()}"

    def plan_step(
        self,
        goal: str,
        available_tools: List[Tool],
        trajectory_so_far: List[CompletedStep],
        current_step_description: str,
        tool_error: Optional[ToolUnavailableError] = None,
    ) -> AgentAction:
        """Plan the next action using ReAct format.

        Wraps the inner agent's prompt with Thought/Action/Observation structure.
        """
        # Build ReAct-augmented step description
        react_description = self._build_react_description(
            goal, trajectory_so_far, current_step_description, tool_error
        )

        # Call inner agent with the ReAct-formatted prompt
        action = self._inner.plan_step(
            goal=goal,
            available_tools=available_tools,
            trajectory_so_far=trajectory_so_far,
            current_step_description=react_description,
            tool_error=tool_error,
        )

        # Augment reasoning with ReAct format
        action.reasoning = self._format_react_reasoning(action, current_step_description, tool_error)

        return action

    def _build_react_description(
        self,
        goal: str,
        trajectory_so_far: List[CompletedStep],
        current_step_description: str,
        tool_error: Optional[ToolUnavailableError],
    ) -> str:
        """Build a ReAct-formatted step description."""
        parts = [
            "Use the following format for your reasoning:",
            "",
            "Thought: <analyze what needs to be done and what tools are available>",
            "Action: <the tool to call or HALT>",
            "Action Input: <parameters for the tool>",
            "",
        ]

        # Add observation history from prior steps
        if trajectory_so_far:
            parts.append("Previous steps:")
            for step in trajectory_so_far:
                parts.append(f"Thought: I need to {step.agent_reasoning[:100] if step.agent_reasoning else 'execute this step'}")
                parts.append(f"Action: {step.tool_called}")
                result_str = str(step.result.output)[:200]
                parts.append(f"Observation: {result_str}")
                parts.append("")

        parts.append(f"Current step: {current_step_description}")

        if tool_error:
            parts.append(
                f"\nObservation: ERROR - Tool '{tool_error.tool_name}' is unavailable. "
                f"Reason: {tool_error.reason}"
            )
            parts.append(
                "Thought: The required tool is unavailable. I need to think about "
                "whether I can use an alternative tool, compose multiple tools, or halt."
            )

        return "\n".join(parts)

    def _format_react_reasoning(
        self,
        action: AgentAction,
        step_description: str,
        tool_error: Optional[ToolUnavailableError],
    ) -> str:
        """Format the agent's action into ReAct Thought/Action/Observation structure."""
        parts = []

        # Thought
        thought = action.reasoning or "Analyzing the current step requirements."
        parts.append(f"Thought: {thought}")

        # Action
        if action.action_type == "HALT":
            parts.append("Action: HALT")
            parts.append(f"Action Input: {action.halt_reason or 'Task cannot be completed'}")
        elif action.action_type == "CALL_TOOL":
            parts.append(f"Action: {action.tool_name}")
            params_str = str(action.tool_params) if action.tool_params else "{}"
            parts.append(f"Action Input: {params_str}")
        else:
            parts.append(f"Action: {action.action_type}")

        return "\n".join(parts)
