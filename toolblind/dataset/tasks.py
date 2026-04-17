"""Task dataclasses and schema definitions for ToolBlind benchmark."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Tool:
    """A tool that an agent can (or cannot) call."""

    name: str
    description: str
    input_schema: Dict[str, str]
    output_type: str
    functional_category: str  # fetch, transform, store, query, execute, notify
    semantic_tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_type": self.output_type,
            "functional_category": self.functional_category,
            "semantic_tags": self.semantic_tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tool":
        """Deserialize from dictionary."""
        return cls(
            name=d["name"],
            description=d["description"],
            input_schema=d["input_schema"],
            output_type=d["output_type"],
            functional_category=d["functional_category"],
            semantic_tags=d["semantic_tags"],
        )


@dataclass
class TaskStep:
    """A single step in a multi-step task trajectory."""

    step_index: int
    description: str
    required_tool: str
    expected_output_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_index": self.step_index,
            "description": self.description,
            "required_tool": self.required_tool,
            "expected_output_type": self.expected_output_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskStep":
        """Deserialize from dictionary."""
        return cls(
            step_index=d["step_index"],
            description=d["description"],
            required_tool=d["required_tool"],
            expected_output_type=d["expected_output_type"],
        )


@dataclass
class ToolBlindTask:
    """A complete ToolBlind benchmark task."""

    task_id: str
    tier: int
    domain: str
    goal: str
    steps: List[TaskStep]
    gap_step_index: int
    available_tools: List[Tool]
    unavailable_tool: Tool
    ground_truth_outcome: str  # SUBSTITUTE, DECOMPOSE, HALT
    gold_solution: Optional[List[str]]
    unavailability_reason: str  # rate_limited, decommissioned, permission_denied, environment_mismatch
    registry_size: int
    commitment_depth: int
    difficulty: str  # easy, medium, hard
    domain_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "domain": self.domain,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "gap_step_index": self.gap_step_index,
            "available_tools": [t.to_dict() for t in self.available_tools],
            "unavailable_tool": self.unavailable_tool.to_dict(),
            "ground_truth_outcome": self.ground_truth_outcome,
            "gold_solution": self.gold_solution,
            "unavailability_reason": self.unavailability_reason,
            "registry_size": self.registry_size,
            "commitment_depth": self.commitment_depth,
            "difficulty": self.difficulty,
            "domain_metadata": self.domain_metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ToolBlindTask":
        """Deserialize from dictionary."""
        return cls(
            task_id=d["task_id"],
            tier=d["tier"],
            domain=d["domain"],
            goal=d["goal"],
            steps=[TaskStep.from_dict(s) for s in d["steps"]],
            gap_step_index=d["gap_step_index"],
            available_tools=[Tool.from_dict(t) for t in d["available_tools"]],
            unavailable_tool=Tool.from_dict(d["unavailable_tool"]),
            ground_truth_outcome=d["ground_truth_outcome"],
            gold_solution=d["gold_solution"],
            unavailability_reason=d["unavailability_reason"],
            registry_size=d["registry_size"],
            commitment_depth=d["commitment_depth"],
            difficulty=d["difficulty"],
            domain_metadata=d.get("domain_metadata", {}),
        )
