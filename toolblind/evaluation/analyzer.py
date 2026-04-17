"""Results analysis and breakdown for ToolBlind benchmark."""

import json
import os
from typing import Any, Dict, List

import pandas as pd
from rich.console import Console
from rich.table import Table

from toolblind.evaluation.metrics import (
    compute_all_metrics,
    compute_commitment_effect,
    compute_confabulation_rate,
    compute_toolblind_score,
)
from toolblind.utils.logging import get_logger

logger = get_logger("analyzer")
console = Console()


class ResultsAnalyzer:
    """Analyze and display experiment results."""

    def __init__(self, results: List[Dict[str, Any]]):
        """Initialize with a list of result dictionaries."""
        self._results = results
        self._df = pd.DataFrame(results) if results else pd.DataFrame()

    @classmethod
    def from_file(cls, path: str) -> "ResultsAnalyzer":
        """Load results from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        results = data if isinstance(data, list) else data.get("results", [])
        return cls(results)

    def summary(self) -> Dict[str, Any]:
        """Compute a full summary of metrics."""
        return compute_all_metrics(self._results)

    def breakdown_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Break down metrics by model/agent name."""
        models: Dict[str, List[Dict[str, Any]]] = {}
        for r in self._results:
            model = r.get("model", "unknown")
            models.setdefault(model, []).append(r)

        return {model: compute_all_metrics(results) for model, results in models.items()}

    def breakdown_by_tier(self) -> Dict[int, Dict[str, Any]]:
        """Break down metrics by tier."""
        tiers: Dict[int, List[Dict[str, Any]]] = {}
        for r in self._results:
            tier = r.get("tier", 0)
            tiers.setdefault(tier, []).append(r)

        return {tier: compute_all_metrics(results) for tier, results in sorted(tiers.items())}

    def breakdown_by_domain(self) -> Dict[str, Dict[str, Any]]:
        """Break down metrics by domain."""
        domains: Dict[str, List[Dict[str, Any]]] = {}
        for r in self._results:
            domain = r.get("domain", "unknown")
            domains.setdefault(domain, []).append(r)

        return {domain: compute_all_metrics(results) for domain, results in sorted(domains.items())}

    def commitment_analysis(self) -> Dict[str, Any]:
        """Analyze the commitment depth effect."""
        return compute_commitment_effect(self._results)

    def framing_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze the effect of unavailability framing."""
        framings: Dict[str, List[Dict[str, Any]]] = {}
        for r in self._results:
            reason = r.get("unavailability_reason", "unknown")
            framings.setdefault(reason, []).append(r)

        return {
            reason: {
                "confabulation_rate": compute_confabulation_rate(results),
                "toolblind_score": compute_toolblind_score(results),
                "count": len(results),
            }
            for reason, results in sorted(framings.items())
        }

    def registry_size_analysis(self) -> Dict[int, Dict[str, float]]:
        """Analyze the effect of registry size."""
        sizes: Dict[int, List[Dict[str, Any]]] = {}
        for r in self._results:
            size = r.get("registry_size", 10)
            sizes.setdefault(size, []).append(r)

        return {
            size: {
                "confabulation_rate": compute_confabulation_rate(results),
                "toolblind_score": compute_toolblind_score(results),
                "count": len(results),
            }
            for size, results in sorted(sizes.items())
        }

    def cot_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare CoT vs direct answer results."""
        cot_results = [r for r in self._results if r.get("cot", False)]
        direct_results = [r for r in self._results if not r.get("cot", False)]

        return {
            "cot": compute_all_metrics(cot_results) if cot_results else {},
            "direct": compute_all_metrics(direct_results) if direct_results else {},
        }

    def print_main_table(self) -> None:
        """Print the main results table (models x tiers x metrics)."""
        model_breakdown = self.breakdown_by_model()

        table = Table(title="ToolBlind Benchmark Results")
        table.add_column("Model", style="bold cyan")
        table.add_column("TBS", justify="right")
        table.add_column("CR", justify="right")
        table.add_column("CR-T1", justify="right")
        table.add_column("CR-T2", justify="right")
        table.add_column("CR-T3", justify="right")
        table.add_column("FSR", justify="right")
        table.add_column("FRS", justify="right")
        table.add_column("N", justify="right")

        for model, metrics in sorted(model_breakdown.items()):
            table.add_row(
                model,
                f"{metrics['toolblind_score']:.3f}",
                f"{metrics['confabulation_rate']:.3f}",
                f"{metrics['confabulation_rate_t1']:.3f}",
                f"{metrics['confabulation_rate_t2']:.3f}",
                f"{metrics['confabulation_rate_t3']:.3f}",
                f"{metrics['false_substitution_rate']:.3f}",
                f"{metrics['functional_reasoning_score']:.2f}",
                str(metrics['total_tasks']),
            )

        console.print(table)

    def print_commitment_table(self) -> None:
        """Print the commitment depth analysis table."""
        analysis = self.commitment_analysis()

        table = Table(title="Commitment Depth Effect on Confabulation Rate")
        table.add_column("Depth", justify="center")
        table.add_column("CR", justify="right")

        for d in analysis["depths"]:
            cr = analysis["cr_by_depth"][d]
            table.add_row(str(d), f"{cr:.3f}")

        table.add_section()
        table.add_row("Slope", f"{analysis['slope']:.4f}")
        table.add_row("p-value", f"{analysis['p_value']:.4f}")
        table.add_row("R²", f"{analysis['r_squared']:.4f}")

        console.print(table)

    def print_framing_table(self) -> None:
        """Print the framing analysis table."""
        analysis = self.framing_analysis()

        table = Table(title="Unavailability Framing Effect")
        table.add_column("Framing", style="bold")
        table.add_column("CR", justify="right")
        table.add_column("TBS", justify="right")
        table.add_column("N", justify="right")

        for reason, metrics in analysis.items():
            table.add_row(
                reason,
                f"{metrics['confabulation_rate']:.3f}",
                f"{metrics['toolblind_score']:.3f}",
                str(metrics['count']),
            )

        console.print(table)

    def save(self, path: str) -> None:
        """Save the full analysis to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        analysis = {
            "summary": self.summary(),
            "by_model": self.breakdown_by_model(),
            "by_tier": {str(k): v for k, v in self.breakdown_by_tier().items()},
            "by_domain": self.breakdown_by_domain(),
            "commitment_effect": self.commitment_analysis(),
            "framing_effect": self.framing_analysis(),
        }
        with open(path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info(f"Analysis saved to {path}")
