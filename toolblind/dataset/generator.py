"""Dataset generation pipeline for ToolBlind benchmark."""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

from toolblind.dataset.catalog import (
    AVAILABLE_TOOL_NAMES,
    DISTRACTOR_TOOLS,
    get_domain_tools,
    get_templates,
)
from toolblind.dataset.tasks import TaskStep, Tool, ToolBlindTask
from toolblind.utils.logging import get_logger

logger = get_logger("generator")

DOMAINS = ["web", "code", "file", "api", "database"]
TIERS = [1, 2, 3]
TIER_COUNTS = {1: 175, 2: 175, 3: 150}
TASKS_PER_DOMAIN_PER_TIER = {
    1: 35,  # 35 * 5 = 175
    2: 35,  # 35 * 5 = 175
    3: 30,  # 30 * 5 = 150
}
UNAVAILABILITY_REASONS = ["rate_limited", "decommissioned", "permission_denied", "environment_mismatch"]
DIFFICULTIES = {1: "easy", 2: "medium", 3: "hard"}
FRAMING_MESSAGES = {
    "rate_limited": "This tool is currently rate-limited and unavailable",
    "decommissioned": "This tool has been permanently decommissioned",
    "permission_denied": "This tool requires elevated permissions you do not have",
    "environment_mismatch": "This tool is not available in your current environment",
}


def _build_available_tools(domain: str, missing_tool_name: str, rng: random.Random,
                           registry_size: int = 10) -> List[Tool]:
    """Build the available tool list for a task, excluding the missing tool."""
    all_tools = get_domain_tools(domain)
    available_names = AVAILABLE_TOOL_NAMES[domain]
    available = [all_tools[name] for name in available_names if name != missing_tool_name]

    # Pad with distractors if we need more tools to reach registry_size
    if len(available) < registry_size:
        distractors = DISTRACTOR_TOOLS.get(domain, [])
        rng.shuffle(distractors)
        for d in distractors:
            if len(available) >= registry_size:
                break
            if d.name != missing_tool_name:
                available.append(d)

    # Trim to registry_size if we have too many
    if len(available) > registry_size:
        # Keep essential tools (those used in gold solutions) and trim others
        available = available[:registry_size]

    return available


def _generate_task_from_template(
    template: Tuple,
    domain: str,
    tier: int,
    task_index: int,
    rng: random.Random,
    registry_size: int = 10,
    unavailability_reason: Optional[str] = None,
) -> ToolBlindTask:
    """Generate a single task from a template."""
    goal, step_specs, gap_idx, missing_tool_name, ground_truth, gold_solution = template
    all_tools = get_domain_tools(domain)

    # Build steps
    steps = []
    for i, (desc, tool_name, out_type) in enumerate(step_specs):
        steps.append(TaskStep(
            step_index=i,
            description=desc,
            required_tool=tool_name,
            expected_output_type=out_type,
        ))

    # Get the unavailable tool
    unavailable_tool = all_tools.get(missing_tool_name)
    if unavailable_tool is None:
        # Tool might be in available pool for cross-domain tasks; create a placeholder
        unavailable_tool = Tool(
            name=missing_tool_name,
            description=f"Tool '{missing_tool_name}' — unavailable",
            input_schema={},
            output_type="unknown",
            functional_category="execute",
            semantic_tags=[domain],
        )

    # Build available tools
    available = _build_available_tools(domain, missing_tool_name, rng, registry_size)

    # Ensure gold solution tools are in available set
    if gold_solution:
        for gs_name in gold_solution:
            if not any(t.name == gs_name for t in available):
                if gs_name in all_tools:
                    available.append(all_tools[gs_name])

    # Also ensure any step tools (other than the missing one) are available
    for step in steps:
        if step.required_tool != missing_tool_name:
            if not any(t.name == step.required_tool for t in available):
                if step.required_tool in all_tools:
                    available.append(all_tools[step.required_tool])

    if unavailability_reason is None:
        unavailability_reason = rng.choice(UNAVAILABILITY_REASONS)

    task_id = f"tb_t{tier}_{domain}_{task_index:04d}"

    return ToolBlindTask(
        task_id=task_id,
        tier=tier,
        domain=domain,
        goal=goal,
        steps=steps,
        gap_step_index=gap_idx,
        available_tools=available,
        unavailable_tool=unavailable_tool,
        ground_truth_outcome=ground_truth,
        gold_solution=gold_solution if gold_solution else None,
        unavailability_reason=unavailability_reason,
        registry_size=len(available),
        commitment_depth=gap_idx,
        difficulty=DIFFICULTIES[tier],
        domain_metadata={"domain": domain, "framing": FRAMING_MESSAGES[unavailability_reason]},
    )


def _vary_template(template: Tuple, rng: random.Random, variation: int) -> Tuple:
    """Create a variation of a template by modifying the goal text."""
    goal, step_specs, gap_idx, missing_tool, ground_truth, gold_sol = template

    # Variations: rephrase the goal slightly
    prefixes = [
        "",
        "Please ",
        "I need to ",
        "Help me ",
        "Your task is to ",
    ]
    suffixes = [
        "",
        " and report the results",
        " as efficiently as possible",
        " for further analysis",
        " and verify the output",
    ]
    prefix = prefixes[variation % len(prefixes)]
    suffix = suffixes[variation % len(suffixes)]

    # Lowercase the first char of goal if adding a prefix
    varied_goal = goal
    if prefix:
        varied_goal = goal[0].lower() + goal[1:]
    varied_goal = prefix + varied_goal + suffix

    return (varied_goal, step_specs, gap_idx, missing_tool, ground_truth, gold_sol)


def generate_dataset(seed: int = 42) -> List[ToolBlindTask]:
    """Generate the complete 500-task dataset."""
    rng = random.Random(seed)
    tasks: List[ToolBlindTask] = []
    task_counter = 0

    for domain in DOMAINS:
        for tier in TIERS:
            target_count = TASKS_PER_DOMAIN_PER_TIER[tier]
            templates = get_templates(domain, tier)
            generated = 0

            # Cycle through templates with variations to reach target count
            variation = 0
            while generated < target_count:
                for tmpl in templates:
                    if generated >= target_count:
                        break
                    varied = _vary_template(tmpl, rng, variation)
                    task = _generate_task_from_template(
                        varied, domain, tier, task_counter, rng
                    )
                    tasks.append(task)
                    task_counter += 1
                    generated += 1
                variation += 1

    logger.info(f"Generated {len(tasks)} tasks")
    return tasks


def generate_ablation_subset(
    tasks: List[ToolBlindTask], seed: int = 42
) -> Dict[str, List[ToolBlindTask]]:
    """Generate a 100-task stratified ablation subset with framing and registry-size variants."""
    rng = random.Random(seed)

    # Stratified sampling: 20 per domain
    by_domain: Dict[str, List[ToolBlindTask]] = {d: [] for d in DOMAINS}
    for t in tasks:
        by_domain[t.domain].append(t)

    subset: List[ToolBlindTask] = []
    for domain in DOMAINS:
        pool = by_domain[domain]
        rng.shuffle(pool)
        subset.extend(pool[:20])

    # Generate framing variants
    framing_variants: List[ToolBlindTask] = []
    for task in subset:
        for reason in UNAVAILABILITY_REASONS:
            variant = ToolBlindTask(
                task_id=f"{task.task_id}_f_{reason}",
                tier=task.tier,
                domain=task.domain,
                goal=task.goal,
                steps=task.steps,
                gap_step_index=task.gap_step_index,
                available_tools=task.available_tools,
                unavailable_tool=task.unavailable_tool,
                ground_truth_outcome=task.ground_truth_outcome,
                gold_solution=task.gold_solution,
                unavailability_reason=reason,
                registry_size=task.registry_size,
                commitment_depth=task.commitment_depth,
                difficulty=task.difficulty,
                domain_metadata={
                    **task.domain_metadata,
                    "framing": FRAMING_MESSAGES[reason],
                    "parent_task_id": task.task_id,
                },
            )
            framing_variants.append(variant)

    # Generate registry-size variants
    registry_variants: List[ToolBlindTask] = []
    for task in subset:
        for reg_size in [5, 10, 15, 25]:
            available = _build_available_tools(
                task.domain, task.unavailable_tool.name, rng, reg_size
            )
            # Ensure gold solution tools remain
            all_tools = get_domain_tools(task.domain)
            if task.gold_solution:
                for gs_name in task.gold_solution:
                    if not any(t.name == gs_name for t in available):
                        if gs_name in all_tools:
                            available.append(all_tools[gs_name])
            for step in task.steps:
                if step.required_tool != task.unavailable_tool.name:
                    if not any(t.name == step.required_tool for t in available):
                        if step.required_tool in all_tools:
                            available.append(all_tools[step.required_tool])

            variant = ToolBlindTask(
                task_id=f"{task.task_id}_r_{reg_size}",
                tier=task.tier,
                domain=task.domain,
                goal=task.goal,
                steps=task.steps,
                gap_step_index=task.gap_step_index,
                available_tools=available,
                unavailable_tool=task.unavailable_tool,
                ground_truth_outcome=task.ground_truth_outcome,
                gold_solution=task.gold_solution,
                unavailability_reason=task.unavailability_reason,
                registry_size=len(available),
                commitment_depth=task.commitment_depth,
                difficulty=task.difficulty,
                domain_metadata={
                    **task.domain_metadata,
                    "target_registry_size": reg_size,
                    "parent_task_id": task.task_id,
                },
            )
            registry_variants.append(variant)

    return {
        "base_subset": subset,
        "framing_variants": framing_variants,
        "registry_variants": registry_variants,
    }


def generate_commitment_variants(
    tasks: List[ToolBlindTask], seed: int = 42
) -> List[ToolBlindTask]:
    """Generate commitment depth variants for Experiment 2.

    For a 150-task stratified subset (30 per domain), create 5 depth variants each.
    """
    rng = random.Random(seed + 1)  # Different seed than ablation

    by_domain: Dict[str, List[ToolBlindTask]] = {d: [] for d in DOMAINS}
    for t in tasks:
        by_domain[t.domain].append(t)

    subset: List[ToolBlindTask] = []
    for domain in DOMAINS:
        pool = by_domain[domain]
        rng.shuffle(pool)
        subset.extend(pool[:30])

    variants: List[ToolBlindTask] = []
    all_domain_tools_cache: Dict[str, Dict[str, Tool]] = {}

    for task in subset:
        if task.domain not in all_domain_tools_cache:
            all_domain_tools_cache[task.domain] = get_domain_tools(task.domain)
        all_tools = all_domain_tools_cache[task.domain]
        available_names = AVAILABLE_TOOL_NAMES[task.domain]

        for depth in range(5):  # depths 0, 1, 2, 3, 4
            # Build new steps with the gap at position `depth`
            # We need at least depth+2 steps (depth completed + gap + at least 1 after)
            total_steps_needed = max(len(task.steps), depth + 2)

            new_steps = []
            # Pre-gap steps use available tools
            avail_pool = [n for n in available_names if n != task.unavailable_tool.name]
            rng.shuffle(avail_pool)
            for i in range(depth):
                tool_name = avail_pool[i % len(avail_pool)]
                tool_obj = all_tools.get(tool_name)
                out_type = tool_obj.output_type if tool_obj else "result"
                new_steps.append(TaskStep(
                    step_index=i,
                    description=f"Step {i+1}: execute {tool_name} operation",
                    required_tool=tool_name,
                    expected_output_type=out_type,
                ))

            # Gap step
            new_steps.append(TaskStep(
                step_index=depth,
                description=f"Step {depth+1}: perform {task.unavailable_tool.name} operation",
                required_tool=task.unavailable_tool.name,
                expected_output_type=task.unavailable_tool.output_type,
            ))

            # Post-gap steps
            for i in range(depth + 1, total_steps_needed):
                tool_name = avail_pool[i % len(avail_pool)]
                tool_obj = all_tools.get(tool_name)
                out_type = tool_obj.output_type if tool_obj else "result"
                new_steps.append(TaskStep(
                    step_index=i,
                    description=f"Step {i+1}: execute {tool_name} operation",
                    required_tool=tool_name,
                    expected_output_type=out_type,
                ))

            variant = ToolBlindTask(
                task_id=f"{task.task_id}_d_{depth}",
                tier=task.tier,
                domain=task.domain,
                goal=task.goal,
                steps=new_steps,
                gap_step_index=depth,
                available_tools=task.available_tools,
                unavailable_tool=task.unavailable_tool,
                ground_truth_outcome=task.ground_truth_outcome,
                gold_solution=task.gold_solution,
                unavailability_reason=task.unavailability_reason,
                registry_size=task.registry_size,
                commitment_depth=depth,
                difficulty=task.difficulty,
                domain_metadata={
                    **task.domain_metadata,
                    "commitment_depth_variant": depth,
                    "parent_task_id": task.task_id,
                },
            )
            variants.append(variant)

    return variants


def save_dataset(tasks: List[ToolBlindTask], output_dir: str) -> str:
    """Save the dataset to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "tasks.json")
    data = [t.to_dict() for t in tasks]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(tasks)} tasks to {path}")
    return path


def save_ablation(ablation: Dict[str, List[ToolBlindTask]], output_dir: str) -> str:
    """Save ablation variants to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ablation_tasks.json")
    data = {
        key: [t.to_dict() for t in task_list]
        for key, task_list in ablation.items()
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved ablation data to {path}")
    return path


def save_commitment_variants(variants: List[ToolBlindTask], output_dir: str) -> str:
    """Save commitment depth variants to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "commitment_variants.json")
    data = [t.to_dict() for t in variants]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(variants)} commitment variants to {path}")
    return path


def load_dataset(tasks_dir: str) -> List[ToolBlindTask]:
    """Load the dataset from JSON file."""
    path = os.path.join(tasks_dir, "tasks.json")
    with open(path) as f:
        data = json.load(f)
    return [ToolBlindTask.from_dict(d) for d in data]


def load_ablation(tasks_dir: str) -> Dict[str, List[ToolBlindTask]]:
    """Load ablation variants from JSON file."""
    path = os.path.join(tasks_dir, "ablation_tasks.json")
    with open(path) as f:
        data = json.load(f)
    return {
        key: [ToolBlindTask.from_dict(d) for d in task_list]
        for key, task_list in data.items()
    }


def load_commitment_variants(tasks_dir: str) -> List[ToolBlindTask]:
    """Load commitment depth variants from JSON file."""
    path = os.path.join(tasks_dir, "commitment_variants.json")
    with open(path) as f:
        data = json.load(f)
    return [ToolBlindTask.from_dict(d) for d in data]
