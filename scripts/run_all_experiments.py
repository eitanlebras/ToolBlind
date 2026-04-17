#!/usr/bin/env python3
"""Run all ToolBlind experiments sequentially."""

import argparse

from rich.console import Console

from toolblind.agents.claude import ClaudeAgent
from toolblind.agents.gemini import GeminiAgent
from toolblind.agents.openai import OpenAIAgent
from toolblind.dataset.generator import load_dataset
from toolblind.experiments.baseline import run_baseline
from toolblind.experiments.commitment import run_commitment_experiment
from toolblind.experiments.cot import run_cot_experiment
from toolblind.experiments.framing import run_framing_experiment
from toolblind.experiments.registry_size import run_registry_size_experiment
from toolblind.utils.config import get_config
from toolblind.utils.logging import setup_logging

console = Console()


def main() -> None:
    """Run all five ToolBlind experiments."""
    parser = argparse.ArgumentParser(description="Run all ToolBlind experiments")
    parser.add_argument("--sample", type=int, default=None, help="Sample N tasks per tier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tasks-dir", type=str, default=None, help="Tasks directory")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge")
    parser.add_argument("--models", nargs="+", default=["claude", "openai", "gemini"],
                        help="Models to run")
    args = parser.parse_args()

    config = get_config()
    setup_logging(config.log_level)
    tasks_dir = args.tasks_dir or config.tasks_dir
    use_judge = not args.no_judge

    console.print("[bold cyan]ToolBlind: Running All Experiments[/]")
    console.print(f"Models: {', '.join(args.models)}")
    console.print(f"Sample: {args.sample or 'full'}")
    console.print()

    tasks = load_dataset(tasks_dir)
    console.print(f"Loaded {len(tasks)} tasks")

    # Build agents
    cot_agents = []
    direct_agents = []
    for name in args.models:
        if name == "claude":
            cot_agents.append(ClaudeAgent(cot=True))
            direct_agents.append(ClaudeAgent(cot=False))
        elif name == "openai":
            cot_agents.append(OpenAIAgent(cot=True))
            direct_agents.append(OpenAIAgent(cot=False))
        elif name == "gemini":
            cot_agents.append(GeminiAgent(cot=True))
            direct_agents.append(GeminiAgent(cot=False))

    # Experiment 1: Baseline
    console.rule("[bold]Experiment 1: Baseline Across Tiers")
    run_baseline(tasks, cot_agents, sample_size=args.sample, seed=args.seed, use_llm_judge=use_judge)

    # Experiment 2: Commitment Depth
    console.rule("[bold]Experiment 2: Trajectory Commitment Effect")
    run_commitment_experiment(tasks, cot_agents, sample_size=args.sample, seed=args.seed, use_llm_judge=use_judge)

    # Experiment 3: Framing Ablation
    console.rule("[bold]Experiment 3: Unavailability Framing")
    run_framing_experiment(tasks, cot_agents, sample_size=args.sample, seed=args.seed, use_llm_judge=use_judge)

    # Experiment 4: Registry Size
    console.rule("[bold]Experiment 4: Registry Size Effect")
    run_registry_size_experiment(tasks, cot_agents, sample_size=args.sample, seed=args.seed, use_llm_judge=use_judge)

    # Experiment 5: CoT vs Direct
    console.rule("[bold]Experiment 5: CoT vs Direct Answer")
    run_cot_experiment(tasks, cot_agents, direct_agents, sample_size=args.sample, seed=args.seed, use_llm_judge=use_judge)

    console.print()
    console.print("[bold green]All experiments complete![/]")


if __name__ == "__main__":
    main()
