#!/usr/bin/env python3
"""Run a single ToolBlind experiment."""

import argparse
import sys
from typing import List

from rich.console import Console

from toolblind.agents.base import BaseAgent
from toolblind.agents.claude import ClaudeAgent
from toolblind.agents.gemini import GeminiAgent
from toolblind.agents.openai import OpenAIAgent
from toolblind.agents.react import ReActWrapper
from toolblind.dataset.generator import load_dataset
from toolblind.experiments.baseline import run_baseline
from toolblind.experiments.commitment import run_commitment_experiment
from toolblind.experiments.cot import run_cot_experiment
from toolblind.experiments.framing import run_framing_experiment
from toolblind.experiments.registry_size import run_registry_size_experiment
from toolblind.utils.config import get_config
from toolblind.utils.logging import setup_logging

console = Console()

# Mapping from CLI shorthand to (AgentClass, model_id)
MODEL_REGISTRY = {
    # OpenAI models
    "openai": (OpenAIAgent, "gpt-4o"),
    "gpt-4o": (OpenAIAgent, "gpt-4o"),
    "gpt-5.4": (OpenAIAgent, "gpt-5.4"),
    "gpt-5.4-mini": (OpenAIAgent, "gpt-5.4-mini"),
    "o3": (OpenAIAgent, "o3"),
    "o4-mini": (OpenAIAgent, "o4-mini"),
    # Anthropic models
    "claude": (ClaudeAgent, "claude-sonnet-4-20250514"),
    "claude-sonnet": (ClaudeAgent, "claude-sonnet-4-20250514"),
    "claude-opus-4-5": (ClaudeAgent, "claude-opus-4-5-20250415"),
    "claude-sonnet-4-5": (ClaudeAgent, "claude-sonnet-4-5-20250414"),
    "claude-haiku": (ClaudeAgent, "claude-haiku-4-5-20251001"),
    # Google models
    "gemini": (GeminiAgent, "gemini-1.5-pro"),
    "gemini-2.5-pro": (GeminiAgent, "gemini-2.5-pro"),
    "gemini-2.5-flash": (GeminiAgent, "gemini-2.5-flash"),
}


def _build_agents(model_names: List[str], cot: bool = True) -> List[BaseAgent]:
    """Build agent instances from model name strings."""
    agents: List[BaseAgent] = []
    for name in model_names:
        # Check if it's a react wrapper
        is_react = name.startswith("react-")
        base_name = name[6:] if is_react else name

        if base_name in MODEL_REGISTRY:
            agent_cls, model_id = MODEL_REGISTRY[base_name]
            agent = agent_cls(cot=cot, model=model_id)
        else:
            console.print(f"[red]Unknown model: {name}[/]")
            console.print(f"Available: {', '.join(sorted(MODEL_REGISTRY.keys()))}")
            sys.exit(1)

        if is_react:
            agent = ReActWrapper(agent)

        agents.append(agent)
    return agents


def main() -> None:
    """Run a specified ToolBlind experiment."""
    parser = argparse.ArgumentParser(description="Run a ToolBlind experiment")
    parser.add_argument(
        "experiment",
        choices=["baseline", "commitment", "framing", "registry_size", "cot"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude"],
        help=f"Models to evaluate. Available: {', '.join(sorted(MODEL_REGISTRY.keys()))}. "
             "Prefix with 'react-' for ReAct wrapper (e.g. react-gpt-4o).",
    )
    parser.add_argument("--sample", type=int, default=None, help="Sample N tasks per tier (for development)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tasks-dir", type=str, default=None, help="Directory containing task JSON files")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge (rule-based only)")
    args = parser.parse_args()

    config = get_config()
    setup_logging(config.log_level)
    tasks_dir = args.tasks_dir or config.tasks_dir

    console.print(f"[bold cyan]ToolBlind Experiment: {args.experiment}[/]")
    console.print(f"Models: {', '.join(args.models)}")
    console.print(f"Sample size: {args.sample or 'full'}")
    console.print()

    # Load dataset
    console.print("Loading dataset...")
    tasks = load_dataset(tasks_dir)
    console.print(f"Loaded {len(tasks)} tasks")

    use_judge = not args.no_judge

    if args.experiment == "baseline":
        agents = _build_agents(args.models, cot=True)
        run_baseline(tasks, agents, sample_size=args.sample, seed=args.seed, use_llm_judge=use_judge)

    elif args.experiment == "commitment":
        agents = _build_agents(args.models, cot=True)
        run_commitment_experiment(tasks, agents, sample_size=args.sample, seed=args.seed, use_llm_judge=use_judge)

    elif args.experiment == "framing":
        agents = _build_agents(args.models, cot=True)
        run_framing_experiment(tasks, agents, sample_size=args.sample, seed=args.seed, use_llm_judge=use_judge)

    elif args.experiment == "registry_size":
        agents = _build_agents(args.models, cot=True)
        run_registry_size_experiment(tasks, agents, sample_size=args.sample, seed=args.seed, use_llm_judge=use_judge)

    elif args.experiment == "cot":
        cot_agents = _build_agents(args.models, cot=True)
        direct_agents = _build_agents(args.models, cot=False)
        run_cot_experiment(tasks, cot_agents, direct_agents, sample_size=args.sample, seed=args.seed, use_llm_judge=use_judge)

    console.print("[bold green]Experiment complete![/]")


if __name__ == "__main__":
    main()
