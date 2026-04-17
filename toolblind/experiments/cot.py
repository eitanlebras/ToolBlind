"""Experiment 5: Chain-of-thought vs direct answer ablation."""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from toolblind.agents.base import BaseAgent
from toolblind.dataset.tasks import ToolBlindTask
from toolblind.environment.simulator import ToolSimulator
from toolblind.environment.trajectory import TrajectoryRunner
from toolblind.evaluation.metrics import compute_all_metrics
from toolblind.evaluation.scorer import OutcomeScorer
from toolblind.utils.config import get_config
from toolblind.utils.logging import console, get_logger

logger = get_logger("experiment.cot")


def run_cot_experiment(
    tasks: List[ToolBlindTask],
    cot_agents: List[BaseAgent],
    direct_agents: List[BaseAgent],
    sample_size: Optional[int] = None,
    seed: int = 42,
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    """Run Experiment 5: CoT vs direct answer.

    Runs all models on the full 500 tasks twice: cot=True and cot=False.

    Args:
        tasks: The full task dataset.
        cot_agents: Agents with chain-of-thought enabled.
        direct_agents: Agents with direct (no CoT) mode.
        sample_size: If set, sample N tasks per tier.
        seed: Random seed.
        use_llm_judge: Whether to use LLM judge.

    Returns:
        Dict with results and CoT comparison analysis.
    """
    config = get_config()
    rng = random.Random(seed)

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

    scorer = OutcomeScorer(use_llm_judge=use_llm_judge)
    all_results: List[Dict[str, Any]] = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    all_agents = [(True, cot_agents), (False, direct_agents)]

    for cot_mode, agents in all_agents:
        for agent in agents:
            agent_name = agent.name()
            logger.info(f"Running {agent_name} (cot={cot_mode}) on {len(eval_tasks)} tasks")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task_bar = progress.add_task(f"[cyan]{agent_name}", total=len(eval_tasks))

                for task in eval_tasks:
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

                    result = {
                        "task_id": task.task_id,
                        "tier": task.tier,
                        "domain": task.domain,
                        "model": agent_name,
                        "cot": cot_mode,
                        "outcome": classification["outcome"],
                        "confidence": classification.get("confidence", 0.5),
                        "ground_truth": task.ground_truth_outcome,
                        "commitment_depth": task.commitment_depth,
                        "registry_size": task.registry_size,
                        "unavailability_reason": task.unavailability_reason,
                        "functional_reasoning_score": classification.get("functional_reasoning_score", 0),
                    }
                    all_results.append(result)
                    progress.advance(task_bar)

    # Analyze CoT effect
    cot_results = [r for r in all_results if r.get("cot", False)]
    direct_results = [r for r in all_results if not r.get("cot", False)]

    cot_metrics = compute_all_metrics(cot_results) if cot_results else {}
    direct_metrics = compute_all_metrics(direct_results) if direct_results else {}

    # Per-tier CoT comparison
    tier_comparison: Dict[str, Dict[str, Any]] = {}
    for tier in [1, 2, 3]:
        cot_tier = [r for r in cot_results if r.get("tier") == tier]
        direct_tier = [r for r in direct_results if r.get("tier") == tier]
        tier_comparison[str(tier)] = {
            "cot": compute_all_metrics(cot_tier) if cot_tier else {},
            "direct": compute_all_metrics(direct_tier) if direct_tier else {},
        }

    # Print comparison table
    _print_cot_table(cot_metrics, direct_metrics, tier_comparison)

    output = {
        "experiment": "cot",
        "timestamp": timestamp,
        "total_tasks": len(eval_tasks),
        "agents_cot": [a.name() for a in cot_agents],
        "agents_direct": [a.name() for a in direct_agents],
        "results": all_results,
        "cot_metrics": cot_metrics,
        "direct_metrics": direct_metrics,
        "tier_comparison": tier_comparison,
    }

    results_dir = config.results_dir
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"cot_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    return output


def _print_cot_table(
    cot_metrics: Dict[str, Any],
    direct_metrics: Dict[str, Any],
    tier_comparison: Dict[str, Dict[str, Any]],
) -> None:
    """Print the CoT vs Direct comparison table."""
    table = Table(title="CoT vs Direct Answer Comparison")
    table.add_column("Metric", style="bold")
    table.add_column("CoT", justify="right")
    table.add_column("Direct", justify="right")
    table.add_column("Delta", justify="right")

    if cot_metrics and direct_metrics:
        for metric_name, display in [
            ("toolblind_score", "TBS"),
            ("confabulation_rate", "CR (overall)"),
            ("confabulation_rate_t1", "CR (Tier 1)"),
            ("confabulation_rate_t2", "CR (Tier 2)"),
            ("confabulation_rate_t3", "CR (Tier 3)"),
            ("false_substitution_rate", "FSR"),
            ("functional_reasoning_score", "FRS"),
        ]:
            cot_val = cot_metrics.get(metric_name, 0)
            direct_val = direct_metrics.get(metric_name, 0)
            delta = cot_val - direct_val
            delta_str = f"{delta:+.3f}"
            table.add_row(display, f"{cot_val:.3f}", f"{direct_val:.3f}", delta_str)

    console.print(table)
