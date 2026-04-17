#!/usr/bin/env python3
"""ToolBlind Test API — explore the benchmark without API keys.

Run with: uvicorn api:app --reload
Docs at:  http://localhost:8000/docs
"""

import random
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from toolblind.agents.base import AgentAction, BaseAgent
from toolblind.dataset.generator import generate_dataset
from toolblind.dataset.tasks import Tool, ToolBlindTask
from toolblind.environment.simulator import ToolSimulator, ToolUnavailableError
from toolblind.environment.trajectory import CompletedStep, TrajectoryRunner, TrajectoryState
from toolblind.evaluation.metrics import compute_all_metrics
from toolblind.evaluation.scorer import OutcomeScorer

# ---------------------------------------------------------------------------
# Stub agent — deterministic, no API keys needed
# ---------------------------------------------------------------------------

class StubSmartAgent(BaseAgent):
    """A stub agent that uses heuristics to pick reasonable actions."""

    def __init__(self, strategy: str = "smart"):
        """Initialize with a strategy: smart, always_halt, always_confabulate."""
        self._strategy = strategy

    def name(self) -> str:
        """Return agent name."""
        return f"stub-{self._strategy}"

    def plan_step(
        self,
        goal: str,
        available_tools: List[Tool],
        trajectory_so_far: List[CompletedStep],
        current_step_description: str,
        tool_error: Optional[ToolUnavailableError] = None,
    ) -> AgentAction:
        """Pick an action based on strategy."""
        if tool_error is None:
            tool_name = available_tools[0].name if available_tools else "unknown"
            return AgentAction(
                action_type="CALL_TOOL",
                tool_name=tool_name,
                tool_params={},
                reasoning=f"Executing step: {current_step_description}",
                confidence=0.9,
            )

        if self._strategy == "always_confabulate":
            return AgentAction(
                action_type="CALL_TOOL",
                tool_name=tool_error.tool_name,
                tool_params={},
                reasoning="Proceeding with the tool anyway",
                confidence=0.9,
            )

        if self._strategy == "always_halt":
            return AgentAction(
                action_type="HALT",
                reasoning=f"Tool '{tool_error.tool_name}' is unavailable: {tool_error.reason}",
                halt_reason=f"Cannot proceed: {tool_error.tool_name} is unavailable and no substitute exists",
                confidence=0.8,
            )

        # "smart" strategy: try to find a substitute
        missing_output = tool_error.tool_name
        for tool in available_tools:
            if tool.name != tool_error.tool_name:
                return AgentAction(
                    action_type="CALL_TOOL",
                    tool_name=tool.name,
                    tool_params={},
                    reasoning=f"Tool '{tool_error.tool_name}' unavailable. Using '{tool.name}' as alternative.",
                    confidence=0.6,
                )

        return AgentAction(
            action_type="HALT",
            reasoning="No available tools can substitute for the missing one",
            halt_reason="No functional substitute available in the current toolset",
            confidence=0.8,
        )


# ---------------------------------------------------------------------------
# Generate dataset at startup
# ---------------------------------------------------------------------------

DATASET = generate_dataset(seed=42)
TASK_INDEX = {t.task_id: t for t in DATASET}

# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class TaskSummary(BaseModel):
    """Summary of a task for listing."""
    task_id: str
    tier: int
    domain: str
    goal: str
    ground_truth_outcome: str
    commitment_depth: int
    difficulty: str
    unavailable_tool: str
    num_steps: int


class TaskDetail(BaseModel):
    """Full task detail."""
    task_id: str
    tier: int
    domain: str
    goal: str
    steps: List[Dict[str, Any]]
    gap_step_index: int
    available_tools: List[Dict[str, Any]]
    unavailable_tool: Dict[str, Any]
    ground_truth_outcome: str
    gold_solution: Optional[List[str]]
    unavailability_reason: str
    registry_size: int
    commitment_depth: int
    difficulty: str


class RunResult(BaseModel):
    """Result of running a single task."""
    task_id: str
    tier: int
    domain: str
    strategy: str
    outcome: str
    ground_truth: str
    correct: bool
    agent_reasoning: str
    confidence: float
    functional_reasoning_score: int
    steps_completed: int
    wall_time_seconds: float


class BatchResult(BaseModel):
    """Result of running a batch of tasks."""
    strategy: str
    total_tasks: int
    results: List[RunResult]
    metrics: Dict[str, Any]


class DatasetStats(BaseModel):
    """Dataset statistics."""
    total_tasks: int
    tier_distribution: Dict[str, int]
    domain_distribution: Dict[str, int]
    outcome_distribution: Dict[str, int]
    difficulty_distribution: Dict[str, int]
    avg_steps_per_task: float
    avg_registry_size: float


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ToolBlind Test API",
    description=(
        "Explore the ToolBlind benchmark — 500 tasks testing AI agent reasoning "
        "under tool absence with trajectory commitment. No API keys required. "
        "Uses deterministic stub agents for demonstration."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CORRECT_OUTCOME = {1: "SUBSTITUTE", 2: "DECOMPOSE", 3: "HALT"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["info"])
def root():
    """API overview."""
    return {
        "name": "ToolBlind Test API",
        "version": "1.0.0",
        "description": "Benchmark for AI agent reasoning under tool absence",
        "docs": "/docs",
        "endpoints": {
            "/stats": "Dataset statistics",
            "/tasks": "List and filter tasks",
            "/tasks/{task_id}": "Get full task detail",
            "/run/{task_id}": "Run a single task with a stub agent",
            "/run/batch": "Run multiple tasks and get metrics",
        },
    }


@app.get("/stats", response_model=DatasetStats, tags=["dataset"])
def get_stats():
    """Get dataset statistics."""
    from collections import Counter

    tiers = Counter(t.tier for t in DATASET)
    domains = Counter(t.domain for t in DATASET)
    outcomes = Counter(t.ground_truth_outcome for t in DATASET)
    diffs = Counter(t.difficulty for t in DATASET)

    return DatasetStats(
        total_tasks=len(DATASET),
        tier_distribution={f"tier_{k}": v for k, v in sorted(tiers.items())},
        domain_distribution=dict(sorted(domains.items())),
        outcome_distribution=dict(sorted(outcomes.items())),
        difficulty_distribution=dict(sorted(diffs.items())),
        avg_steps_per_task=sum(len(t.steps) for t in DATASET) / len(DATASET),
        avg_registry_size=sum(t.registry_size for t in DATASET) / len(DATASET),
    )


@app.get("/tasks", response_model=List[TaskSummary], tags=["dataset"])
def list_tasks(
    tier: Optional[int] = Query(None, description="Filter by tier (1, 2, or 3)"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    outcome: Optional[str] = Query(None, description="Filter by ground truth outcome"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Skip N results"),
):
    """List tasks with optional filtering."""
    filtered = DATASET
    if tier is not None:
        filtered = [t for t in filtered if t.tier == tier]
    if domain is not None:
        filtered = [t for t in filtered if t.domain == domain]
    if outcome is not None:
        filtered = [t for t in filtered if t.ground_truth_outcome == outcome.upper()]

    page = filtered[offset : offset + limit]
    return [
        TaskSummary(
            task_id=t.task_id,
            tier=t.tier,
            domain=t.domain,
            goal=t.goal,
            ground_truth_outcome=t.ground_truth_outcome,
            commitment_depth=t.commitment_depth,
            difficulty=t.difficulty,
            unavailable_tool=t.unavailable_tool.name,
            num_steps=len(t.steps),
        )
        for t in page
    ]


@app.get("/tasks/{task_id}", response_model=TaskDetail, tags=["dataset"])
def get_task(task_id: str):
    """Get full details for a specific task."""
    task = TASK_INDEX.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    return TaskDetail(
        task_id=task.task_id,
        tier=task.tier,
        domain=task.domain,
        goal=task.goal,
        steps=[s.to_dict() for s in task.steps],
        gap_step_index=task.gap_step_index,
        available_tools=[t.to_dict() for t in task.available_tools],
        unavailable_tool=task.unavailable_tool.to_dict(),
        ground_truth_outcome=task.ground_truth_outcome,
        gold_solution=task.gold_solution,
        unavailability_reason=task.unavailability_reason,
        registry_size=task.registry_size,
        commitment_depth=task.commitment_depth,
        difficulty=task.difficulty,
    )


@app.get("/run/{task_id}", response_model=RunResult, tags=["run"])
def run_task(
    task_id: str,
    strategy: str = Query("smart", description="Agent strategy: smart, always_halt, always_confabulate"),
):
    """Run a single task with a stub agent and return the scored result."""
    task = TASK_INDEX.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    if strategy not in ("smart", "always_halt", "always_confabulate"):
        raise HTTPException(status_code=400, detail="Strategy must be: smart, always_halt, always_confabulate")

    agent = StubSmartAgent(strategy=strategy)
    simulator = ToolSimulator(
        unavailable_tools={task.unavailable_tool.name},
        unavailability_reasons={
            task.unavailable_tool.name: task.domain_metadata.get(
                "framing", f"Tool is {task.unavailability_reason}"
            )
        },
    )
    runner = TrajectoryRunner(simulator)
    trajectory = runner.run(task, agent)

    scorer = OutcomeScorer(use_llm_judge=False)
    classification = scorer.classify(task, trajectory)

    outcome = classification["outcome"]
    correct = (task.tier in CORRECT_OUTCOME and outcome == CORRECT_OUTCOME[task.tier])

    reasoning = trajectory.agent_response_at_gap or ""

    return RunResult(
        task_id=task.task_id,
        tier=task.tier,
        domain=task.domain,
        strategy=strategy,
        outcome=outcome,
        ground_truth=task.ground_truth_outcome,
        correct=correct,
        agent_reasoning=reasoning[:500],
        confidence=classification.get("confidence", 0.5),
        functional_reasoning_score=classification.get("functional_reasoning_score", 0),
        steps_completed=len(trajectory.steps_completed),
        wall_time_seconds=round(trajectory.wall_time_seconds, 4),
    )


@app.post("/run/batch", response_model=BatchResult, tags=["run"])
def run_batch(
    strategy: str = Query("smart", description="Agent strategy"),
    tier: Optional[int] = Query(None, description="Filter by tier"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    sample: int = Query(30, ge=1, le=500, description="Number of tasks to run"),
    seed: int = Query(42, description="Random seed for sampling"),
):
    """Run a batch of tasks and return aggregate metrics."""
    if strategy not in ("smart", "always_halt", "always_confabulate"):
        raise HTTPException(status_code=400, detail="Strategy must be: smart, always_halt, always_confabulate")

    filtered = DATASET
    if tier is not None:
        filtered = [t for t in filtered if t.tier == tier]
    if domain is not None:
        filtered = [t for t in filtered if t.domain == domain]

    rng = random.Random(seed)
    tasks = list(filtered)
    rng.shuffle(tasks)
    tasks = tasks[:sample]

    agent = StubSmartAgent(strategy=strategy)
    scorer = OutcomeScorer(use_llm_judge=False)
    results = []
    metrics_input = []

    for task in tasks:
        simulator = ToolSimulator(
            unavailable_tools={task.unavailable_tool.name},
            unavailability_reasons={
                task.unavailable_tool.name: task.domain_metadata.get(
                    "framing", f"Tool is {task.unavailability_reason}"
                )
            },
        )
        runner = TrajectoryRunner(simulator)
        trajectory = runner.run(task, agent)
        classification = scorer.classify(task, trajectory)

        outcome = classification["outcome"]
        correct = (task.tier in CORRECT_OUTCOME and outcome == CORRECT_OUTCOME[task.tier])

        result = RunResult(
            task_id=task.task_id,
            tier=task.tier,
            domain=task.domain,
            strategy=strategy,
            outcome=outcome,
            ground_truth=task.ground_truth_outcome,
            correct=correct,
            agent_reasoning=(trajectory.agent_response_at_gap or "")[:300],
            confidence=classification.get("confidence", 0.5),
            functional_reasoning_score=classification.get("functional_reasoning_score", 0),
            steps_completed=len(trajectory.steps_completed),
            wall_time_seconds=round(trajectory.wall_time_seconds, 4),
        )
        results.append(result)

        metrics_input.append({
            "tier": task.tier,
            "outcome": outcome,
            "confidence": classification.get("confidence", 0.5),
            "functional_reasoning_score": classification.get("functional_reasoning_score", 0),
            "commitment_depth": task.commitment_depth,
        })

    metrics = compute_all_metrics(metrics_input)

    return BatchResult(
        strategy=strategy,
        total_tasks=len(results),
        results=results,
        metrics=metrics,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
