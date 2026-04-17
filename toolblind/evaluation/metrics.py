"""Metrics computation for ToolBlind benchmark."""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from toolblind.utils.logging import get_logger

logger = get_logger("metrics")

# Tier weights for ToolBlind Score
TIER_WEIGHTS = {1: 0.25, 2: 0.35, 3: 0.40}

# What counts as "correct" per tier
CORRECT_OUTCOME = {1: "SUBSTITUTE", 2: "DECOMPOSE", 3: "HALT"}


def compute_toolblind_score(results: List[Dict[str, Any]]) -> float:
    """Compute the primary ToolBlind Score (TBS).

    TBS = (w1 * correct_T1 + w2 * correct_T2 + w3 * correct_T3) / total
    """
    if not results:
        return 0.0

    tier_correct = {1: 0, 2: 0, 3: 0}
    total = len(results)

    for r in results:
        tier = r.get("tier", 0)
        outcome = r.get("outcome", "")
        if tier in CORRECT_OUTCOME and outcome == CORRECT_OUTCOME[tier]:
            tier_correct[tier] += 1

    weighted_sum = sum(TIER_WEIGHTS[t] * tier_correct[t] for t in TIER_WEIGHTS)
    return weighted_sum / total


def compute_confabulation_rate(results: List[Dict[str, Any]], tier: Optional[int] = None) -> float:
    """Compute the confabulation rate, optionally filtered by tier.

    CR = confabulations / total_tasks
    """
    if tier is not None:
        filtered = [r for r in results if r.get("tier") == tier]
    else:
        filtered = results

    if not filtered:
        return 0.0

    confab_count = sum(1 for r in filtered if r.get("outcome") == "CONFABULATE")
    return confab_count / len(filtered)


def compute_false_substitution_rate(results: List[Dict[str, Any]]) -> float:
    """Compute the false substitution rate.

    FSR = false_substitutions / total_tasks
    """
    if not results:
        return 0.0

    fs_count = sum(1 for r in results if r.get("outcome") == "FALSE_SUBSTITUTE")
    return fs_count / len(results)


def compute_functional_reasoning_score(results: List[Dict[str, Any]]) -> float:
    """Compute average Functional Reasoning Score for non-confabulating responses.

    Score 0-3, averaged across non-CONFABULATE tasks.
    """
    non_confab = [r for r in results if r.get("outcome") != "CONFABULATE"]
    if not non_confab:
        return 0.0

    scores = [r.get("functional_reasoning_score", 0) for r in non_confab]
    return float(np.mean(scores))


def compute_commitment_effect(
    results: List[Dict[str, Any]], depths: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Compute confabulation rate as a function of commitment depth.

    Returns:
        Dict with CR(d) values, regression slope, p-value.
    """
    if depths is None:
        depths = [0, 1, 2, 3, 4]

    cr_by_depth: Dict[int, float] = {}
    for d in depths:
        depth_results = [r for r in results if r.get("commitment_depth") == d]
        if depth_results:
            cr_by_depth[d] = compute_confabulation_rate(depth_results)
        else:
            cr_by_depth[d] = 0.0

    # Fit linear regression if we have enough data points
    x_vals = sorted(cr_by_depth.keys())
    y_vals = [cr_by_depth[x] for x in x_vals]

    slope = 0.0
    p_value = 1.0
    r_squared = 0.0

    if len(x_vals) >= 2 and any(y != y_vals[0] for y in y_vals):
        result = stats.linregress(x_vals, y_vals)
        slope = result.slope
        p_value = result.pvalue
        r_squared = result.rvalue ** 2

    return {
        "cr_by_depth": cr_by_depth,
        "slope": slope,
        "p_value": p_value,
        "r_squared": r_squared,
        "depths": x_vals,
        "confabulation_rates": y_vals,
    }


def compute_calibration_error(results: List[Dict[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
    """Compute Expected Calibration Error (ECE).

    Treats agent confidence as a probability and measures whether it
    tracks actual correctness.
    """
    if not results:
        return {"ece": 0.0, "bins": [], "accuracy_per_bin": [], "confidence_per_bin": [], "count_per_bin": []}

    confidences = []
    correct_flags = []
    for r in results:
        conf = r.get("confidence", 0.5)
        tier = r.get("tier", 0)
        outcome = r.get("outcome", "")
        is_correct = (tier in CORRECT_OUTCOME and outcome == CORRECT_OUTCOME[tier])
        confidences.append(conf)
        correct_flags.append(float(is_correct))

    confidences = np.array(confidences)
    correct_flags = np.array(correct_flags)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    accuracy_per_bin = []
    confidence_per_bin = []
    count_per_bin = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask = (confidences >= lower) & (confidences < upper)
        if i == n_bins - 1:  # Include upper bound for last bin
            mask = (confidences >= lower) & (confidences <= upper)

        bin_count = mask.sum()
        if bin_count == 0:
            bins.append((lower, upper))
            accuracy_per_bin.append(0.0)
            confidence_per_bin.append(0.0)
            count_per_bin.append(0)
            continue

        bin_accuracy = correct_flags[mask].mean()
        bin_confidence = confidences[mask].mean()

        ece += (bin_count / len(results)) * abs(bin_accuracy - bin_confidence)

        bins.append((lower, upper))
        accuracy_per_bin.append(float(bin_accuracy))
        confidence_per_bin.append(float(bin_confidence))
        count_per_bin.append(int(bin_count))

    return {
        "ece": float(ece),
        "bins": bins,
        "accuracy_per_bin": accuracy_per_bin,
        "confidence_per_bin": confidence_per_bin,
        "count_per_bin": count_per_bin,
    }


def compute_all_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all metrics for a set of results."""
    return {
        "toolblind_score": compute_toolblind_score(results),
        "confabulation_rate": compute_confabulation_rate(results),
        "confabulation_rate_t1": compute_confabulation_rate(results, tier=1),
        "confabulation_rate_t2": compute_confabulation_rate(results, tier=2),
        "confabulation_rate_t3": compute_confabulation_rate(results, tier=3),
        "false_substitution_rate": compute_false_substitution_rate(results),
        "functional_reasoning_score": compute_functional_reasoning_score(results),
        "calibration": compute_calibration_error(results),
        "total_tasks": len(results),
        "outcome_distribution": _outcome_distribution(results),
    }


def _outcome_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count outcomes by type."""
    dist: Dict[str, int] = {}
    for r in results:
        outcome = r.get("outcome", "UNKNOWN")
        dist[outcome] = dist.get(outcome, 0) + 1
    return dist
