"""Experiment 2: Trajectory commitment depth effect on confabulation."""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from toolblind.agents.base import BaseAgent
from toolblind.dataset.generator import generate_commitment_variants
from toolblind.dataset.tasks import ToolBlindTask
from toolblind.environment.simulator import ToolSimulator
from toolblind.environment.trajectory import TrajectoryRunner
from toolblind.evaluation.analyzer import ResultsAnalyzer
from toolblind.evaluation.metrics import compute_commitment_effect
from toolblind.evaluation.scorer import OutcomeScorer
from toolblind.utils.config import get_config
from toolblind.utils.logging import console, get_logger

logger = get_logger("experiment.commitment")


def run_commitment_experiment(
    tasks: List[ToolBlindTask],
    agents: List[BaseAgent],
    sample_size: Optional[int] = None,
    seed: int = 42,
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    """Run Experiment 2: trajectory commitment depth effect.

    Takes a 150-task stratified subset, generates 5 depth variants each (0-4),
    and measures confabulation rate as a function of commitment depth.

    Args:
        tasks: The full task dataset.
        agents: List of agents to evaluate.
        sample_size: If set, limit the number of base tasks per domain.
        seed: Random seed.
        use_llm_judge: Whether to use LLM judge for scoring.

    Returns:
        Dict with results, commitment effect analysis, and regression stats.
    """
    config = get_config()

    # Generate commitment depth variants
    variants = generate_commitment_variants(tasks, seed=seed)
    logger.info(f"Generated {len(variants)} commitment depth variants")

    if sample_size is not None:
        rng = random.Random(seed)
        rng.shuffle(variants)
        variants = variants[:sample_size * 5]  # 5 depths per base task
        logger.info(f"Sampled down to {len(variants)} variants")

    scorer = OutcomeScorer(use_llm_judge=use_llm_judge)
    all_results: List[Dict[str, Any]] = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for agent in agents:
        agent_name = agent.name()
        logger.info(f"Running {agent_name} on {len(variants)} commitment variants")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task_bar = progress.add_task(f"[cyan]{agent_name}", total=len(variants))

            for task in variants:
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

    # Analyze commitment effect
    commitment_analysis = compute_commitment_effect(all_results)

    # Per-model commitment analysis
    by_model: Dict[str, Any] = {}
    models = set(r["model"] for r in all_results)
    for model in models:
        model_results = [r for r in all_results if r["model"] == model]
        by_model[model] = compute_commitment_effect(model_results)

    analyzer = ResultsAnalyzer(all_results)
    analyzer.print_commitment_table()

    output = {
        "experiment": "commitment",
        "timestamp": timestamp,
        "total_variants": len(variants),
        "agents": [a.name() for a in agents],
        "results": all_results,
        "commitment_effect": commitment_analysis,
        "commitment_effect_by_model": by_model,
    }

    results_dir = config.results_dir
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"commitment_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    return output
