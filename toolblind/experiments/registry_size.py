"""Experiment 4: Registry size effect on confabulation."""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from toolblind.agents.base import BaseAgent
from toolblind.dataset.generator import generate_ablation_subset
from toolblind.dataset.tasks import ToolBlindTask
from toolblind.environment.simulator import ToolSimulator
from toolblind.environment.trajectory import TrajectoryRunner
from toolblind.evaluation.analyzer import ResultsAnalyzer
from toolblind.evaluation.scorer import OutcomeScorer
from toolblind.utils.config import get_config
from toolblind.utils.logging import console, get_logger

logger = get_logger("experiment.registry_size")

REGISTRY_SIZES = [5, 10, 15, 25]


def run_registry_size_experiment(
    tasks: List[ToolBlindTask],
    agents: List[BaseAgent],
    sample_size: Optional[int] = None,
    seed: int = 42,
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    """Run Experiment 4: registry size effect.

    Tests the 100-task ablation subset at registry sizes {5, 10, 15, 25}.

    Args:
        tasks: The full task dataset.
        agents: List of agents to evaluate.
        sample_size: If set, limit the number of variants per registry size.
        seed: Random seed.
        use_llm_judge: Whether to use LLM judge for scoring.

    Returns:
        Dict with results and registry size analysis.
    """
    config = get_config()

    ablation = generate_ablation_subset(tasks, seed=seed)
    registry_variants = ablation["registry_variants"]
    logger.info(f"Generated {len(registry_variants)} registry size variants")

    if sample_size is not None:
        rng = random.Random(seed)
        by_size: Dict[int, List[ToolBlindTask]] = {}
        for t in registry_variants:
            target_size = t.domain_metadata.get("target_registry_size", t.registry_size)
            by_size.setdefault(target_size, []).append(t)
        registry_variants = []
        for size in REGISTRY_SIZES:
            pool = by_size.get(size, [])
            rng.shuffle(pool)
            registry_variants.extend(pool[:sample_size])

    scorer = OutcomeScorer(use_llm_judge=use_llm_judge)
    all_results: List[Dict[str, Any]] = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for agent in agents:
        agent_name = agent.name()
        logger.info(f"Running {agent_name} on {len(registry_variants)} registry variants")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task_bar = progress.add_task(f"[cyan]{agent_name}", total=len(registry_variants))

            for task in registry_variants:
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
                    "target_registry_size": task.domain_metadata.get("target_registry_size", task.registry_size),
                    "unavailability_reason": task.unavailability_reason,
                    "functional_reasoning_score": classification.get("functional_reasoning_score", 0),
                }
                all_results.append(result)
                progress.advance(task_bar)

    # Analyze registry size effect
    analyzer = ResultsAnalyzer(all_results)
    registry_analysis = analyzer.registry_size_analysis()

    # Per-model analysis
    by_model: Dict[str, Dict[int, Dict[str, float]]] = {}
    models = set(r["model"] for r in all_results)
    for model in models:
        model_results = [r for r in all_results if r["model"] == model]
        model_analyzer = ResultsAnalyzer(model_results)
        by_model[model] = model_analyzer.registry_size_analysis()

    output = {
        "experiment": "registry_size",
        "timestamp": timestamp,
        "total_variants": len(registry_variants),
        "agents": [a.name() for a in agents],
        "results": all_results,
        "registry_analysis": {str(k): v for k, v in registry_analysis.items()},
        "registry_analysis_by_model": {
            model: {str(k): v for k, v in analysis.items()}
            for model, analysis in by_model.items()
        },
    }

    results_dir = config.results_dir
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"registry_size_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    return output
