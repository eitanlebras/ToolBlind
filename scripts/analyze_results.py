#!/usr/bin/env python3
"""Analyze ToolBlind experiment results."""

import argparse
import glob
import os

from rich.console import Console

from toolblind.evaluation.analyzer import ResultsAnalyzer
from toolblind.utils.config import get_config
from toolblind.utils.logging import setup_logging

console = Console()


def main() -> None:
    """Analyze and display experiment results."""
    parser = argparse.ArgumentParser(description="Analyze ToolBlind results")
    parser.add_argument("results_file", nargs="?", default=None, help="Path to results JSON file")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory containing results files")
    parser.add_argument("--latest", action="store_true", help="Analyze the most recent results file")
    parser.add_argument("--save", type=str, default=None, help="Save analysis to JSON file")
    args = parser.parse_args()

    config = get_config()
    setup_logging(config.log_level)
    results_dir = args.results_dir or config.results_dir

    # Determine which file to analyze
    if args.results_file:
        results_path = args.results_file
    elif args.latest:
        pattern = os.path.join(results_dir, "*.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not files:
            console.print(f"[red]No results files found in {results_dir}[/]")
            return
        results_path = files[-1]
        console.print(f"Using latest: {results_path}")
    else:
        console.print("[red]Specify a results file or use --latest[/]")
        return

    console.print("[bold cyan]ToolBlind Results Analysis[/]")
    console.print(f"File: {results_path}")
    console.print()

    analyzer = ResultsAnalyzer.from_file(results_path)

    # Print all available tables
    analyzer.print_main_table()
    console.print()
    analyzer.print_commitment_table()
    console.print()
    analyzer.print_framing_table()

    # Print summary metrics
    summary = analyzer.summary()
    console.print()
    console.print("[bold]Summary Metrics:[/]")
    console.print(f"  ToolBlind Score: {summary.get('toolblind_score', 0):.4f}")
    console.print(f"  Confabulation Rate: {summary.get('confabulation_rate', 0):.4f}")
    console.print(f"  False Substitution Rate: {summary.get('false_substitution_rate', 0):.4f}")
    console.print(f"  Functional Reasoning Score: {summary.get('functional_reasoning_score', 0):.2f}")
    cal = summary.get("calibration", {})
    console.print(f"  Expected Calibration Error: {cal.get('ece', 0):.4f}")

    if args.save:
        analyzer.save(args.save)
        console.print(f"\nAnalysis saved to {args.save}")


if __name__ == "__main__":
    main()
