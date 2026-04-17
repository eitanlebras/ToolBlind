"""Experiment 1: Baseline evaluation across tiers for all models."""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from toolblind.agents.base import BaseAgent
from toolblind.dataset.tasks import ToolBlindTask
from toolblind.environment.simulator import ToolSimulator
from toolblind.environment.trajectory import TrajectoryRunner
from toolblind.evaluation.analyzer import ResultsAnalyzer
from toolblind.evaluation.scorer import OutcomeScorer
from toolblind.utils.config import get_config
from toolblind.utils.logging import console, get_logger

logger = get_logger("experiment.baseline")


def run_baseline(
    tasks: List[ToolBlindTask],
    agents: List[BaseAgent],
    sample_size: Optional[int] = None,
    seed: int = 42,
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    """Run Experiment 1: baseline across tiers.

    Runs all agents on all tasks (or a sample) and computes TBS, CR, FSR per model per tier.

    Args:
        tasks: The full task dataset.
        agents: List of agents to evaluate.
        sample_size: If set, sample N tasks per tier for faster evaluation.
        seed: Random seed for sampling.
        use_llm_judge: Whether to use LLM judge for scoring.

    Returns:
        Dict with all results and metrics.
    """
    rng = random.Random(seed)
    config = get_config()

    # Sample tasks if requested
    eval_tasks = tasks
    if sample_size is not None:
        by_tier: Dict[int, List[ToolBlindTask]] = {}
        for t in tasks:
            by_tier.setdefault(t.tier, []).append(t)
        eval_tasks = []
        for tier in sorted(by_tier.keys()):
            pool = by_tier[tier]
            rng.shuffle(pool)
            eval_tasks.extend(pool[:sample_size])
        logger.info(f"Sampled {len(eval_tasks)} tasks ({sample_size} per tier)")

    scorer = OutcomeScorer(use_llm_judge=use_llm_judge)
    all_results: List[Dict[str, Any]] = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for agent in agents:
        agent_name = agent.name()
        logger.info(f"Running {agent_name} on {len(eval_tasks)} tasks")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task_bar = progress.add_task(f"[cyan]{agent_name}", total=len(eval_tasks))

            for task in eval_tasks:
                # Create simulator with the unavailable tool
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

                # Classify outcome
                classification = scorer.classify(task, trajectory)

                result = {
                    "task_id": task.task_id,
                    "tier": task.tier,
                    "domain": task.domain,
                    "model": agent_name,
                    "outcome": classification["outcome"],
                    "confidence": classification.get("confidence", 0.5),
                    "ground_truth": task.ground_truth_outcome,
                    "commitment_depth": task.commitment_depth,
                    "registry_size": task.registry_size,
                    "unavailability_reason": task.unavailability_reason,
                    "functional_reasoning_score": classification.get("functional_reasoning_score", 0),
                    "reasoning": classification.get("reasoning", ""),
                    "tokens_used": trajectory.total_tokens_used,
                    "wall_time": trajectory.wall_time_seconds,
                }
                all_results.append(result)
                progress.advance(task_bar)

    # Analyze and save
    analyzer = ResultsAnalyzer(all_results)
    analyzer.print_main_table()

    output = {
        "experiment": "baseline",
        "timestamp": timestamp,
        "total_tasks": len(eval_tasks),
        "agents": [a.name() for a in agents],
        "sample_size": sample_size,
        "results": all_results,
        "metrics": analyzer.summary(),
        "by_model": analyzer.breakdown_by_model(),
    }

    # Save results
    results_dir = config.results_dir
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"baseline_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    return output
