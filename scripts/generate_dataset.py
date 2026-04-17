#!/usr/bin/env python3
"""Generate the complete ToolBlind dataset."""

import argparse
import sys
from collections import Counter

from rich.console import Console
from rich.table import Table

from toolblind.dataset.generator import (
    generate_ablation_subset,
    generate_commitment_variants,
    generate_dataset,
    save_ablation,
    save_commitment_variants,
    save_dataset,
)
from toolblind.dataset.validator import validate_dataset
from toolblind.utils.config import get_config
from toolblind.utils.logging import setup_logging

console = Console()


def main() -> None:
    """Generate the complete ToolBlind benchmark dataset."""
    parser = argparse.ArgumentParser(description="Generate the ToolBlind dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for tasks")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation checks")
    args = parser.parse_args()

    config = get_config()
    setup_logging(config.log_level)
    output_dir = args.output_dir or config.tasks_dir

    console.print("[bold cyan]ToolBlind Dataset Generator[/]")
    console.print(f"Seed: {args.seed}")
    console.print(f"Output: {output_dir}")
    console.print()

    # Step 1: Generate main dataset
    console.print("[bold]Step 1:[/] Generating 500 tasks...")
    tasks = generate_dataset(seed=args.seed)
    console.print(f"  Generated {len(tasks)} tasks")

    # Step 2: Print statistics
    tier_counts = Counter(t.tier for t in tasks)
    domain_counts = Counter(t.domain for t in tasks)
    outcome_counts = Counter(t.ground_truth_outcome for t in tasks)

    stats_table = Table(title="Dataset Statistics")
    stats_table.add_column("Category", style="bold")
    stats_table.add_column("Value", justify="right")

    stats_table.add_row("Total tasks", str(len(tasks)))
    stats_table.add_section()
    for tier in sorted(tier_counts.keys()):
        stats_table.add_row(f"Tier {tier}", str(tier_counts[tier]))
    stats_table.add_section()
    for domain in sorted(domain_counts.keys()):
        stats_table.add_row(f"Domain: {domain}", str(domain_counts[domain]))
    stats_table.add_section()
    for outcome in sorted(outcome_counts.keys()):
        stats_table.add_row(f"Outcome: {outcome}", str(outcome_counts[outcome]))

    console.print(stats_table)
    console.print()

    # Step 3: Validate
    if not args.skip_validation:
        console.print("[bold]Step 2:[/] Validating dataset...")
        is_valid = validate_dataset(tasks, strict=True)
        if not is_valid:
            console.print("[bold red]Validation FAILED. Aborting.[/]")
            sys.exit(1)
        console.print("  [green]Validation passed[/]")
        console.print()

    # Step 4: Save main dataset
    console.print("[bold]Step 3:[/] Saving main dataset...")
    path = save_dataset(tasks, output_dir)
    console.print(f"  Saved to {path}")

    # Step 5: Generate and save ablation subset
    console.print("[bold]Step 4:[/] Generating ablation subset...")
    ablation = generate_ablation_subset(tasks, seed=args.seed)
    ablation_path = save_ablation(ablation, output_dir)
    console.print(f"  Base subset: {len(ablation['base_subset'])} tasks")
    console.print(f"  Framing variants: {len(ablation['framing_variants'])} tasks")
    console.print(f"  Registry variants: {len(ablation['registry_variants'])} tasks")
    console.print(f"  Saved to {ablation_path}")

    # Step 6: Generate and save commitment variants
    console.print("[bold]Step 5:[/] Generating commitment depth variants...")
    commitment = generate_commitment_variants(tasks, seed=args.seed)
    commitment_path = save_commitment_variants(commitment, output_dir)
    console.print(f"  Generated {len(commitment)} variants")
    console.print(f"  Saved to {commitment_path}")

    console.print()
    console.print("[bold green]Dataset generation complete![/]")


if __name__ == "__main__":
    main()
