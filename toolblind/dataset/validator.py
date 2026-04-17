"""Dataset validation and quality checks for ToolBlind."""

from collections import Counter
from typing import Dict, List, Tuple

from toolblind.dataset.tasks import ToolBlindTask
from toolblind.utils.logging import get_logger

logger = get_logger("validator")

VALID_TIERS = {1, 2, 3}
VALID_DOMAINS = {"web", "code", "file", "api", "database"}
VALID_OUTCOMES = {"SUBSTITUTE", "DECOMPOSE", "HALT"}
VALID_REASONS = {"rate_limited", "decommissioned", "permission_denied", "environment_mismatch"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def validate_schema(task: ToolBlindTask) -> List[str]:
    """Validate a single task has all required fields with valid values."""
    errors: List[str] = []

    if not task.task_id:
        errors.append("task_id is empty")
    if task.tier not in VALID_TIERS:
        errors.append(f"invalid tier: {task.tier}")
    if task.domain not in VALID_DOMAINS:
        errors.append(f"invalid domain: {task.domain}")
    if not task.goal or len(task.goal) < 10:
        errors.append("goal is too short or empty")
    if not task.steps or len(task.steps) < 2:
        errors.append("must have at least 2 steps")
    if task.gap_step_index < 0 or task.gap_step_index >= len(task.steps):
        errors.append(f"gap_step_index {task.gap_step_index} out of range [0, {len(task.steps)})")
    if not task.available_tools:
        errors.append("available_tools is empty")
    if not task.unavailable_tool:
        errors.append("unavailable_tool is missing")
    if task.ground_truth_outcome not in VALID_OUTCOMES:
        errors.append(f"invalid ground_truth_outcome: {task.ground_truth_outcome}")
    if task.unavailability_reason not in VALID_REASONS:
        errors.append(f"invalid unavailability_reason: {task.unavailability_reason}")
    if task.registry_size < 1:
        errors.append(f"invalid registry_size: {task.registry_size}")
    if task.commitment_depth < 0:
        errors.append(f"invalid commitment_depth: {task.commitment_depth}")
    if task.difficulty not in VALID_DIFFICULTIES:
        errors.append(f"invalid difficulty: {task.difficulty}")

    # Validate steps
    for i, step in enumerate(task.steps):
        if step.step_index != i:
            errors.append(f"step {i} has wrong step_index {step.step_index}")
        if not step.description:
            errors.append(f"step {i} has empty description")
        if not step.required_tool:
            errors.append(f"step {i} has empty required_tool")

    return errors


def validate_gold_solutions(task: ToolBlindTask) -> List[str]:
    """Validate gold solution feasibility."""
    errors: List[str] = []
    available_names = {t.name for t in task.available_tools}

    if task.ground_truth_outcome in ("SUBSTITUTE", "DECOMPOSE"):
        if task.gold_solution is None or len(task.gold_solution) == 0:
            errors.append(f"{task.task_id}: {task.ground_truth_outcome} task must have a gold_solution")
        else:
            for tool_name in task.gold_solution:
                if tool_name not in available_names:
                    errors.append(
                        f"{task.task_id}: gold_solution tool '{tool_name}' not in available_tools"
                    )
    elif task.ground_truth_outcome == "HALT":
        if task.gold_solution is not None:
            errors.append(f"{task.task_id}: HALT task should have null gold_solution")

    return errors


def validate_no_trivial(task: ToolBlindTask) -> List[str]:
    """Ensure the missing tool name does not appear in available tools."""
    errors: List[str] = []
    available_names = {t.name for t in task.available_tools}
    if task.unavailable_tool.name in available_names:
        errors.append(
            f"{task.task_id}: unavailable tool '{task.unavailable_tool.name}' is in available_tools"
        )
    return errors


def validate_commitment_depth(task: ToolBlindTask) -> List[str]:
    """Validate commitment_depth == gap_step_index."""
    errors: List[str] = []
    if task.commitment_depth != task.gap_step_index:
        errors.append(
            f"{task.task_id}: commitment_depth ({task.commitment_depth}) != "
            f"gap_step_index ({task.gap_step_index})"
        )
    return errors


def validate_infeasibility(task: ToolBlindTask) -> List[str]:
    """For HALT tasks, verify no available tool has the same output_type as the missing tool."""
    errors: List[str] = []
    if task.ground_truth_outcome != "HALT":
        return errors

    missing_output = task.unavailable_tool.output_type
    missing_tags = set(task.unavailable_tool.semantic_tags)

    for avail in task.available_tools:
        # A tool is a potential substitute if it shares output_type AND has >50% tag overlap
        if avail.output_type == missing_output:
            tag_overlap = len(set(avail.semantic_tags) & missing_tags) / max(len(missing_tags), 1)
            if tag_overlap > 0.5:
                errors.append(
                    f"{task.task_id}: HALT task but '{avail.name}' may substitute "
                    f"(same output_type='{missing_output}', tag overlap={tag_overlap:.0%})"
                )
    return errors


def validate_no_duplicates(tasks: List[ToolBlindTask]) -> List[str]:
    """Ensure no duplicate task IDs."""
    errors: List[str] = []
    seen: Dict[str, int] = {}
    for t in tasks:
        if t.task_id in seen:
            errors.append(f"duplicate task_id: {t.task_id}")
        seen[t.task_id] = seen.get(t.task_id, 0) + 1
    return errors


def validate_distribution(tasks: List[ToolBlindTask]) -> Tuple[Dict[str, int], Dict[str, int], List[str]]:
    """Validate tier and domain distributions."""
    tier_counts = Counter(t.tier for t in tasks)
    domain_counts = Counter(t.domain for t in tasks)
    warnings: List[str] = []

    # Check tier distribution (allow some flexibility from variants)
    expected_tiers = {1: 175, 2: 175, 3: 150}
    for tier, expected in expected_tiers.items():
        actual = tier_counts.get(tier, 0)
        if actual != expected:
            warnings.append(f"Tier {tier}: expected {expected}, got {actual}")

    # Check domain distribution
    for domain in VALID_DOMAINS:
        actual = domain_counts.get(domain, 0)
        if actual != 100:
            warnings.append(f"Domain '{domain}': expected 100, got {actual}")

    return dict(tier_counts), dict(domain_counts), warnings


def validate_dataset(tasks: List[ToolBlindTask], strict: bool = True) -> bool:
    """Run all validations on the dataset. Returns True if valid."""
    all_errors: List[str] = []
    all_warnings: List[str] = []

    # Per-task validations
    for task in tasks:
        all_errors.extend(validate_schema(task))
        all_errors.extend(validate_gold_solutions(task))
        all_errors.extend(validate_no_trivial(task))
        all_errors.extend(validate_commitment_depth(task))
        # Infeasibility check is a warning, not a hard error
        infeasibility_issues = validate_infeasibility(task)
        all_warnings.extend(infeasibility_issues)

    # Dataset-level validations
    all_errors.extend(validate_no_duplicates(tasks))
    _, _, dist_warnings = validate_distribution(tasks)
    all_warnings.extend(dist_warnings)

    if all_warnings:
        for w in all_warnings:
            logger.warning(f"Validation warning: {w}")

    if all_errors:
        for e in all_errors[:50]:  # Limit output
            logger.error(f"Validation error: {e}")
        if len(all_errors) > 50:
            logger.error(f"... and {len(all_errors) - 50} more errors")
        if strict:
            return False

    if not all_errors:
        logger.info(f"Dataset validation passed: {len(tasks)} tasks, 0 errors, {len(all_warnings)} warnings")

    return len(all_errors) == 0
