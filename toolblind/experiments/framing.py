"""Experiment 3: Unavailability framing ablation."""

import json
import os
import random
import time
from typing import Any, Dict, List, Optional

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from scipy import stats

from toolblind.agents.base import BaseAgent
from toolblind.dataset.generator import generate_ablation_subset
from toolblind.dataset.tasks import ToolBlindTask
from toolblind.environment.simulator import ToolSimulator
from toolblind.environment.trajectory import TrajectoryRunner
from toolblind.evaluation.analyzer import ResultsAnalyzer
from toolblind.evaluation.scorer import OutcomeScorer
from toolblind.utils.config import get_config
from toolblind.utils.logging import console, get_logger

logger = get_logger("experiment.framing")

FRAMING_CONDITIONS = ["rate_limited", "decommissioned", "permission_denied", "environment_mismatch"]


def run_framing_experiment(
    tasks: List[ToolBlindTask],
    agents: List[BaseAgent],
    sample_size: Optional[int] = None,
    seed: int = 42,
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    """Run Experiment 3: unavailability framing ablation.

    Tests all 4 framing conditions on a 100-task ablation subset per model.

    Args:
        tasks: The full task dataset.
        agents: List of agents to evaluate.
        sample_size: If set, limit the ablation subset size.
        seed: Random seed.
        use_llm_judge: Whether to use LLM judge for scoring.

    Returns:
        Dict with results, framing effect analysis, and statistical tests.
    """
    config = get_config()

    # Generate ablation subset with framing variants
    ablation = generate_ablation_subset(tasks, seed=seed)
    framing_variants = ablation["framing_variants"]
    logger.info(f"Generated {len(framing_variants)} framing variants")

    if sample_size is not None:
        rng = random.Random(seed)
        # Sample proportionally across framings
        by_framing: Dict[str, List[ToolBlindTask]] = {}
        for t in framing_variants:
            by_framing.setdefault(t.unavailability_reason, []).append(t)
        framing_variants = []
        for reason in FRAMING_CONDITIONS:
            pool = by_framing.get(reason, [])
            rng.shuffle(pool)
            framing_variants.extend(pool[:sample_size])

    scorer = OutcomeScorer(use_llm_judge=use_llm_judge)
    all_results: List[Dict[str, Any]] = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for agent in agents:
        agent_name = agent.name()
        logger.info(f"Running {agent_name} on {len(framing_variants)} framing variants")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task_bar = progress.add_task(f"[cyan]{agent_name}", total=len(framing_variants))

            for task in framing_variants:
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
                    "unavailability_reason": task.unavailability_reason,
                    "functional_reasoning_score": classification.get("functional_reasoning_score", 0),
                }
                all_results.append(result)
                progress.advance(task_bar)

    # Analyze framing effect
    analyzer = ResultsAnalyzer(all_results)
    framing_analysis = analyzer.framing_analysis()
    analyzer.print_framing_table()

    # Statistical test: paired Wilcoxon signed-rank between framing conditions
    statistical_tests = _run_statistical_tests(all_results)

    output = {
        "experiment": "framing",
        "timestamp": timestamp,
        "total_variants": len(framing_variants),
        "agents": [a.name() for a in agents],
        "results": all_results,
        "framing_analysis": framing_analysis,
        "statistical_tests": statistical_tests,
    }

    results_dir = config.results_dir
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"framing_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    return output


def _run_statistical_tests(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run paired Wilcoxon signed-rank tests between framing conditions."""
    tests: Dict[str, Any] = {}

    # Group by parent task (strip framing suffix) and model
    by_parent: Dict[str, Dict[str, Dict[str, int]]] = {}
    for r in results:
        task_id = r["task_id"]
        # Extract parent task id (before _f_ suffix)
        parent = task_id.rsplit("_f_", 1)[0] if "_f_" in task_id else task_id
        model = r["model"]
        reason = r["unavailability_reason"]
        is_confab = 1 if r["outcome"] == "CONFABULATE" else 0
        key = f"{parent}:{model}"
        by_parent.setdefault(key, {})[reason] = is_confab

    # Compare each pair of framings
    for i, f1 in enumerate(FRAMING_CONDITIONS):
        for f2 in FRAMING_CONDITIONS[i + 1:]:
            paired_a = []
            paired_b = []
            for key, framings in by_parent.items():
                if f1 in framings and f2 in framings:
                    paired_a.append(framings[f1])
                    paired_b.append(framings[f2])

            if len(paired_a) >= 5:
                try:
                    stat, p_value = stats.wilcoxon(paired_a, paired_b, zero_method="zsplit")
                    tests[f"{f1}_vs_{f2}"] = {
                        "statistic": float(stat),
                        "p_value": float(p_value),
                        "n_pairs": len(paired_a),
                        "significant": p_value < 0.05,
                    }
                except ValueError:
                    tests[f"{f1}_vs_{f2}"] = {
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "n_pairs": len(paired_a),
                        "significant": False,
                        "note": "All differences are zero",
                    }
            else:
                tests[f"{f1}_vs_{f2}"] = {
                    "statistic": None,
                    "p_value": None,
                    "n_pairs": len(paired_a),
                    "significant": False,
                    "note": "Insufficient paired data",
                }

    return tests
