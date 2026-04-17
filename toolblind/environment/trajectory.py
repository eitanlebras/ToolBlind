"""Trajectory state management for ToolBlind benchmark."""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from toolblind.dataset.tasks import ToolBlindTask
from toolblind.environment.simulator import ToolResult, ToolSimulator, ToolUnavailableError
from toolblind.utils.logging import get_logger

logger = get_logger("trajectory")


@dataclass
class CompletedStep:
    """A step that has been executed in the trajectory."""

    step_index: int
    tool_called: str
    params: Dict[str, Any]
    result: ToolResult
    agent_reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_index": self.step_index,
            "tool_called": self.tool_called,
            "params": self.params,
            "result": self.result.to_dict(),
            "agent_reasoning": self.agent_reasoning,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompletedStep":
        """Deserialize from dictionary."""
        return cls(
            step_index=d["step_index"],
            tool_called=d["tool_called"],
            params=d["params"],
            result=ToolResult.from_dict(d["result"]),
            agent_reasoning=d["agent_reasoning"],
        )


@dataclass
class TrajectoryState:
    """Full state of an agent's execution trajectory on a task."""

    task_id: str
    steps_completed: List[CompletedStep] = field(default_factory=list)
    current_step_index: int = 0
    gap_encountered: bool = False
    gap_step_index: int = 0
    agent_response_at_gap: Optional[str] = None
    outcome: Optional[str] = None
    total_tokens_used: int = 0
    wall_time_seconds: float = 0.0
    agent_actions_at_gap: Optional[List[Dict]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "steps_completed": [s.to_dict() for s in self.steps_completed],
            "current_step_index": self.current_step_index,
            "gap_encountered": self.gap_encountered,
            "gap_step_index": self.gap_step_index,
            "agent_response_at_gap": self.agent_response_at_gap,
            "outcome": self.outcome,
            "total_tokens_used": self.total_tokens_used,
            "wall_time_seconds": self.wall_time_seconds,
            "agent_actions_at_gap": self.agent_actions_at_gap,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrajectoryState":
        """Deserialize from dictionary."""
        return cls(
            task_id=d["task_id"],
            steps_completed=[CompletedStep.from_dict(s) for s in d["steps_completed"]],
            current_step_index=d["current_step_index"],
            gap_encountered=d["gap_encountered"],
            gap_step_index=d["gap_step_index"],
            agent_response_at_gap=d.get("agent_response_at_gap"),
            outcome=d.get("outcome"),
            total_tokens_used=d.get("total_tokens_used", 0),
            wall_time_seconds=d.get("wall_time_seconds", 0.0),
            agent_actions_at_gap=d.get("agent_actions_at_gap"),
        )

    def add_completed_step(self, step: CompletedStep) -> None:
        """Add a completed step to the trajectory."""
        self.steps_completed.append(step)
        self.current_step_index = step.step_index + 1

    def mark_gap_encountered(self, response: str, actions: Optional[List[Dict]] = None) -> None:
        """Mark that the gap has been encountered and record the agent's response."""
        self.gap_encountered = True
        self.agent_response_at_gap = response
        self.agent_actions_at_gap = actions


class TrajectoryRunner:
    """Runs an agent through a task trajectory, handling the gap injection."""

    def __init__(self, simulator: ToolSimulator):
        """Initialize with a tool simulator."""
        self._simulator = simulator

    def run(self, task: ToolBlindTask, agent: Any) -> TrajectoryState:
        """Execute a full trajectory for a task with a given agent.

        Steps:
        1. Feed the agent the task goal and available tools
        2. Execute each step, feeding results back
        3. At gap_step_index, inject ToolUnavailableError
        4. Record agent response
        5. Return complete trajectory state
        """
        state = TrajectoryState(
            task_id=task.task_id,
            gap_step_index=task.gap_step_index,
        )
        start_time = time.time()

        # Execute pre-gap steps
        for step in task.steps:
            if step.step_index >= task.gap_step_index:
                break

            # Ask agent for the tool call for this step
            action = agent.plan_step(
                goal=task.goal,
                available_tools=task.available_tools,
                trajectory_so_far=state.steps_completed,
                current_step_description=step.description,
                tool_error=None,
            )
            state.total_tokens_used += getattr(action, "tokens_used", 0)

            # Execute the tool
            try:
                result = self._simulator.execute(
                    action.tool_name or step.required_tool,
                    action.tool_params or {},
                    {"task_id": task.task_id},
                )
            except ToolUnavailableError:
                # Pre-gap tool shouldn't be unavailable, but handle it
                result = ToolResult(success=False, output="Tool unavailable", output_type="error", latency_ms=0)

            completed = CompletedStep(
                step_index=step.step_index,
                tool_called=action.tool_name or step.required_tool,
                params=action.tool_params or {},
                result=result,
                agent_reasoning=action.reasoning,
            )
            state.add_completed_step(completed)

        # Now inject the gap
        gap_step = task.steps[task.gap_step_index]
        error = ToolUnavailableError(
            task.unavailable_tool.name,
            task.domain_metadata.get("framing", f"Tool is {task.unavailability_reason}"),
        )

        # Ask agent how to handle the gap
        action = agent.plan_step(
            goal=task.goal,
            available_tools=task.available_tools,
            trajectory_so_far=state.steps_completed,
            current_step_description=gap_step.description,
            tool_error=error,
        )
        state.total_tokens_used += getattr(action, "tokens_used", 0)

        # Record the agent's response at the gap
        response_text = action.reasoning
        if action.halt_reason:
            response_text += f"\nHALT REASON: {action.halt_reason}"
        if action.tool_name:
            response_text += f"\nATTEMPTED TOOL: {action.tool_name}"

        action_dict = {
            "action_type": action.action_type,
            "tool_name": action.tool_name,
            "tool_params": action.tool_params,
            "reasoning": action.reasoning,
            "halt_reason": action.halt_reason,
        }

        state.mark_gap_encountered(response_text, [action_dict])
        state.wall_time_seconds = time.time() - start_time

        return state

    def save_trajectory(self, state: TrajectoryState, output_dir: str) -> str:
        """Save a trajectory state to disk as JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"trajectory_{state.task_id}.json")
        with open(path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        return path

    def load_trajectory(self, path: str) -> TrajectoryState:
        """Load a trajectory state from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return TrajectoryState.from_dict(data)
