"""Tests for ToolBlind scoring and metrics."""


from toolblind.agents.base import AgentAction
from toolblind.dataset.tasks import TaskStep, Tool, ToolBlindTask
from toolblind.environment.trajectory import TrajectoryState
from toolblind.evaluation.metrics import (
    compute_calibration_error,
    compute_commitment_effect,
    compute_confabulation_rate,
    compute_functional_reasoning_score,
    compute_toolblind_score,
)
from toolblind.evaluation.scorer import OutcomeScorer


def _make_tool(name: str, output_type: str = "test_output", tags: list = None) -> Tool:
    """Helper to create a Tool for testing."""
    return Tool(
        name=name,
        description=f"Test tool {name}",
        input_schema={"input": "string"},
        output_type=output_type,
        functional_category="fetch",
        semantic_tags=tags or ["test"],
    )


def _make_task(
    tier: int = 1,
    ground_truth: str = "SUBSTITUTE",
    gold_solution: list = None,
    available_tool_names: list = None,
    missing_tool_name: str = "missing_tool",
    missing_output_type: str = "missing_output",
) -> ToolBlindTask:
    """Helper to create a ToolBlindTask for testing."""
    avail_names = available_tool_names or ["tool_a", "tool_b"]
    available = [_make_tool(n) for n in avail_names]

    return ToolBlindTask(
        task_id="test_001",
        tier=tier,
        domain="web",
        goal="Test goal for the benchmark task",
        steps=[
            TaskStep(0, "Step 0", "tool_a", "test_output"),
            TaskStep(1, "Step 1", missing_tool_name, missing_output_type),
            TaskStep(2, "Step 2", "tool_b", "test_output"),
        ],
        gap_step_index=1,
        available_tools=available,
        unavailable_tool=_make_tool(missing_tool_name, missing_output_type, ["special"]),
        ground_truth_outcome=ground_truth,
        gold_solution=gold_solution,
        unavailability_reason="rate_limited",
        registry_size=len(available),
        commitment_depth=1,
        difficulty="easy" if tier == 1 else ("medium" if tier == 2 else "hard"),
    )


def _make_trajectory(action_type: str, tool_name: str = None, halt_reason: str = None) -> TrajectoryState:
    """Helper to create a TrajectoryState with a gap response."""
    state = TrajectoryState(task_id="test_001", gap_step_index=1)
    action = AgentAction(
        action_type=action_type,
        tool_name=tool_name,
        reasoning="Test reasoning about the tool situation",
        halt_reason=halt_reason,
        confidence=0.7,
    )
    state.mark_gap_encountered(
        f"reasoning: {action.reasoning}",
        [action.to_dict()],
    )
    return state


class TestConfabulationDetection:
    """Tests for confabulation detection."""

    def test_calling_missing_tool_is_confabulate(self):
        """Calling the unavailable tool's name = CONFABULATE."""
        task = _make_task(gold_solution=["tool_a"])
        trajectory = _make_trajectory("CALL_TOOL", tool_name="missing_tool")
        scorer = OutcomeScorer(use_llm_judge=False)
        result = scorer.classify(task, trajectory)
        assert result["outcome"] == "CONFABULATE"

    def test_calling_unknown_tool_is_confabulate(self):
        """Calling a tool not in available tools = CONFABULATE."""
        task = _make_task(gold_solution=["tool_a"])
        trajectory = _make_trajectory("CALL_TOOL", tool_name="nonexistent_tool")
        scorer = OutcomeScorer(use_llm_judge=False)
        result = scorer.classify(task, trajectory)
        assert result["outcome"] == "CONFABULATE"


class TestHaltDetection:
    """Tests for HALT detection."""

    def test_halt_with_reason(self):
        """HALT action with reason > 20 chars = HALT."""
        task = _make_task(tier=3, ground_truth="HALT", gold_solution=None)
        trajectory = _make_trajectory(
            "HALT",
            halt_reason="The required tool for LinkedIn API access is not available and cannot be approximated",
        )
        scorer = OutcomeScorer(use_llm_judge=False)
        result = scorer.classify(task, trajectory)
        assert result["outcome"] == "HALT"

    def test_halt_with_short_reason_still_halts(self):
        """HALT with a very short reason is still classified."""
        task = _make_task(tier=3, ground_truth="HALT", gold_solution=None)
        trajectory = _make_trajectory("HALT", halt_reason="Short")
        scorer = OutcomeScorer(use_llm_judge=False)
        result = scorer.classify(task, trajectory)
        # Short reason doesn't pass the >20 char check, falls through
        assert result["outcome"] in ("HALT", "CONFABULATE")


class TestFalseSubstituteDetection:
    """Tests for FALSE_SUBSTITUTE detection."""

    def test_wrong_output_type_is_false_substitute(self):
        """Using a tool with wrong output type = FALSE_SUBSTITUTE."""
        task = _make_task(
            tier=1,
            gold_solution=["tool_a"],
            available_tool_names=["tool_a", "tool_b"],
            missing_output_type="special_output",
        )
        # tool_a has output_type="test_output" but missing tool has "special_output"
        trajectory = _make_trajectory("CALL_TOOL", tool_name="tool_a")
        scorer = OutcomeScorer(use_llm_judge=False)
        result = scorer.classify(task, trajectory)
        assert result["outcome"] == "FALSE_SUBSTITUTE"


class TestToolBlindScore:
    """Tests for the ToolBlind Score formula."""

    def test_toolblind_score_weights(self):
        """TBS formula with known inputs produces correct result."""
        results = [
            {"tier": 1, "outcome": "SUBSTITUTE"},   # correct
            {"tier": 1, "outcome": "CONFABULATE"},   # wrong
            {"tier": 2, "outcome": "DECOMPOSE"},     # correct
            {"tier": 2, "outcome": "CONFABULATE"},   # wrong
            {"tier": 3, "outcome": "HALT"},           # correct
            {"tier": 3, "outcome": "CONFABULATE"},   # wrong
        ]
        score = compute_toolblind_score(results)
        # w1*1 + w2*1 + w3*1 = 0.25 + 0.35 + 0.40 = 1.0, / 6 = 0.1667
        expected = (0.25 * 1 + 0.35 * 1 + 0.40 * 1) / 6
        assert abs(score - expected) < 0.001

    def test_metrics_on_all_zeros(self):
        """All confabulations = TBS of 0."""
        results = [
            {"tier": 1, "outcome": "CONFABULATE"},
            {"tier": 2, "outcome": "CONFABULATE"},
            {"tier": 3, "outcome": "CONFABULATE"},
        ]
        score = compute_toolblind_score(results)
        assert score == 0.0

    def test_metrics_on_all_correct(self):
        """Perfect score when all outcomes match ground truth."""
        results = [
            {"tier": 1, "outcome": "SUBSTITUTE"},
            {"tier": 2, "outcome": "DECOMPOSE"},
            {"tier": 3, "outcome": "HALT"},
        ]
        score = compute_toolblind_score(results)
        # (0.25 + 0.35 + 0.40) / 3 = 1.0/3 ≈ 0.333
        expected = (0.25 + 0.35 + 0.40) / 3
        assert abs(score - expected) < 0.001

    def test_empty_results(self):
        """Empty results return 0."""
        assert compute_toolblind_score([]) == 0.0


class TestConfabulationRate:
    """Tests for confabulation rate computation."""

    def test_confabulation_rate(self):
        """CR computed correctly."""
        results = [
            {"outcome": "CONFABULATE", "tier": 1},
            {"outcome": "SUBSTITUTE", "tier": 1},
            {"outcome": "CONFABULATE", "tier": 2},
            {"outcome": "HALT", "tier": 3},
        ]
        assert compute_confabulation_rate(results) == 0.5

    def test_confabulation_rate_by_tier(self):
        """CR can be filtered by tier."""
        results = [
            {"outcome": "CONFABULATE", "tier": 1},
            {"outcome": "SUBSTITUTE", "tier": 1},
            {"outcome": "CONFABULATE", "tier": 2},
        ]
        assert compute_confabulation_rate(results, tier=1) == 0.5
        assert compute_confabulation_rate(results, tier=2) == 1.0
        assert compute_confabulation_rate(results, tier=3) == 0.0


class TestCommitmentEffect:
    """Tests for commitment depth effect analysis."""

    def test_commitment_effect_slope(self):
        """Slope computation works on synthetic data with increasing CR."""
        results = []
        # Create data where deeper commitment = higher confabulation
        for depth in range(5):
            for i in range(20):
                is_confab = i < (depth * 4)  # More confabulations at higher depths
                results.append({
                    "commitment_depth": depth,
                    "outcome": "CONFABULATE" if is_confab else "SUBSTITUTE",
                })

        effect = compute_commitment_effect(results)
        assert effect["slope"] > 0  # Positive slope: more depth = more confabulation
        assert effect["p_value"] < 0.1  # Should be statistically significant

    def test_commitment_effect_flat(self):
        """Flat CR across depths gives slope ~0."""
        results = []
        for depth in range(5):
            for _ in range(20):
                results.append({"commitment_depth": depth, "outcome": "SUBSTITUTE"})

        effect = compute_commitment_effect(results)
        assert abs(effect["slope"]) < 0.01


class TestCalibration:
    """Tests for calibration error."""

    def test_calibration_error_perfect(self):
        """Perfect calibration gives ECE close to 0."""
        results = [
            {"confidence": 1.0, "tier": 1, "outcome": "SUBSTITUTE"},
            {"confidence": 0.0, "tier": 1, "outcome": "CONFABULATE"},
        ]
        cal = compute_calibration_error(results)
        assert cal["ece"] >= 0

    def test_calibration_error_empty(self):
        """Empty results give ECE of 0."""
        cal = compute_calibration_error([])
        assert cal["ece"] == 0.0


class TestFunctionalReasoningScore:
    """Tests for functional reasoning score."""

    def test_frs_excludes_confabulations(self):
        """FRS only counts non-confabulating responses."""
        results = [
            {"outcome": "SUBSTITUTE", "functional_reasoning_score": 3},
            {"outcome": "CONFABULATE", "functional_reasoning_score": 0},
            {"outcome": "HALT", "functional_reasoning_score": 2},
        ]
        frs = compute_functional_reasoning_score(results)
        assert frs == 2.5  # (3 + 2) / 2

    def test_frs_all_confabulations(self):
        """FRS returns 0 when all responses are confabulations."""
        results = [
            {"outcome": "CONFABULATE", "functional_reasoning_score": 0},
        ]
        frs = compute_functional_reasoning_score(results)
        assert frs == 0.0
