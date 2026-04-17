"""Tests for ToolBlind dataset generation and validation."""

import tempfile
from collections import Counter

import pytest

from toolblind.dataset.generator import (
    generate_ablation_subset,
    generate_commitment_variants,
    generate_dataset,
    load_dataset,
    save_dataset,
)
from toolblind.dataset.tasks import TaskStep, Tool, ToolBlindTask
from toolblind.dataset.validator import (
    validate_commitment_depth,
    validate_dataset,
    validate_gold_solutions,
    validate_no_duplicates,
    validate_no_trivial,
    validate_schema,
)


@pytest.fixture(scope="module")
def dataset():
    """Generate the full dataset once for all tests in this module."""
    return generate_dataset(seed=42)


class TestTaskSchema:
    """Tests for task schema validation."""

    def test_task_schema_valid(self, dataset):
        """All tasks have required fields with valid values."""
        for task in dataset:
            errors = validate_schema(task)
            assert errors == [], f"Task {task.task_id} has schema errors: {errors}"

    def test_all_tasks_have_steps(self, dataset):
        """Every task has at least 2 steps."""
        for task in dataset:
            assert len(task.steps) >= 2, f"Task {task.task_id} has only {len(task.steps)} steps"

    def test_all_tasks_have_unavailable_tool(self, dataset):
        """Every task has a defined unavailable tool."""
        for task in dataset:
            assert task.unavailable_tool is not None
            assert task.unavailable_tool.name != ""


class TestTierDistribution:
    """Tests for tier distribution."""

    def test_tier_distribution(self, dataset):
        """Dataset has 175 Tier 1, 175 Tier 2, 150 Tier 3."""
        tier_counts = Counter(t.tier for t in dataset)
        assert tier_counts[1] == 175, f"Tier 1: expected 175, got {tier_counts[1]}"
        assert tier_counts[2] == 175, f"Tier 2: expected 175, got {tier_counts[2]}"
        assert tier_counts[3] == 150, f"Tier 3: expected 150, got {tier_counts[3]}"

    def test_total_tasks(self, dataset):
        """Dataset has exactly 500 tasks."""
        assert len(dataset) == 500


class TestDomainDistribution:
    """Tests for domain distribution."""

    def test_domain_distribution(self, dataset):
        """Each domain has exactly 100 tasks."""
        domain_counts = Counter(t.domain for t in dataset)
        for domain in ["web", "code", "file", "api", "database"]:
            assert domain_counts[domain] == 100, f"Domain '{domain}': expected 100, got {domain_counts[domain]}"


class TestGoldSolutions:
    """Tests for gold solution validity."""

    def test_gold_solutions_valid(self, dataset):
        """SUBSTITUTE/DECOMPOSE tasks have non-null gold solutions."""
        for task in dataset:
            if task.ground_truth_outcome in ("SUBSTITUTE", "DECOMPOSE"):
                assert task.gold_solution is not None, f"Task {task.task_id}: missing gold_solution"
                assert len(task.gold_solution) > 0, f"Task {task.task_id}: empty gold_solution"

    def test_halt_tasks_no_gold(self, dataset):
        """HALT tasks have null gold solutions."""
        for task in dataset:
            if task.ground_truth_outcome == "HALT":
                assert task.gold_solution is None, f"Task {task.task_id}: HALT should have null gold_solution"

    def test_gold_solution_tools_available(self, dataset):
        """Gold solution tools exist in the available tool set."""
        for task in dataset:
            errors = validate_gold_solutions(task)
            assert errors == [], f"Task {task.task_id}: {errors}"


class TestCommitmentDepth:
    """Tests for commitment depth consistency."""

    def test_commitment_depth_matches_gap_index(self, dataset):
        """commitment_depth == gap_step_index for all tasks."""
        for task in dataset:
            errors = validate_commitment_depth(task)
            assert errors == [], f"Task {task.task_id}: {errors}"


class TestNoTrivial:
    """Tests for non-trivial task construction."""

    def test_unavailable_tool_not_in_available(self, dataset):
        """The unavailable tool name does not appear in the available tools."""
        for task in dataset:
            errors = validate_no_trivial(task)
            assert errors == [], f"Task {task.task_id}: {errors}"


class TestNoDuplicates:
    """Tests for unique task IDs."""

    def test_no_duplicate_task_ids(self, dataset):
        """All task IDs are unique."""
        errors = validate_no_duplicates(dataset)
        assert errors == [], f"Duplicate task IDs found: {errors}"


class TestAblationSubset:
    """Tests for ablation subset generation."""

    def test_ablation_subset_exists(self, dataset):
        """100-task subset with 4 framing variants each is generated correctly."""
        ablation = generate_ablation_subset(dataset, seed=42)
        assert len(ablation["base_subset"]) == 100
        assert len(ablation["framing_variants"]) == 400  # 100 * 4 framings
        assert len(ablation["registry_variants"]) == 400  # 100 * 4 registry sizes

    def test_framing_variants_cover_all_reasons(self, dataset):
        """Framing variants cover all 4 unavailability reasons."""
        ablation = generate_ablation_subset(dataset, seed=42)
        reasons = set(t.unavailability_reason for t in ablation["framing_variants"])
        assert reasons == {"rate_limited", "decommissioned", "permission_denied", "environment_mismatch"}

    def test_ablation_domain_balance(self, dataset):
        """Ablation subset has 20 tasks per domain."""
        ablation = generate_ablation_subset(dataset, seed=42)
        domain_counts = Counter(t.domain for t in ablation["base_subset"])
        for domain in ["web", "code", "file", "api", "database"]:
            assert domain_counts[domain] == 20


class TestCommitmentVariants:
    """Tests for commitment depth variant generation."""

    def test_commitment_variants_generated(self, dataset):
        """Commitment variants are generated with 5 depths per task."""
        variants = generate_commitment_variants(dataset, seed=42)
        # 150 base tasks * 5 depths = 750
        assert len(variants) == 750

    def test_commitment_variant_depths(self, dataset):
        """Each base task has variants at depths 0-4."""
        variants = generate_commitment_variants(dataset, seed=42)
        depth_counts = Counter(t.commitment_depth for t in variants)
        assert set(depth_counts.keys()) == {0, 1, 2, 3, 4}
        for depth in range(5):
            assert depth_counts[depth] == 150  # 150 base tasks per depth


class TestSerialization:
    """Tests for dataset save/load."""

    def test_save_and_load(self, dataset):
        """Dataset can be saved to JSON and loaded back identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dataset(dataset, tmpdir)
            loaded = load_dataset(tmpdir)
            assert len(loaded) == len(dataset)
            for orig, loaded_task in zip(dataset, loaded):
                assert orig.task_id == loaded_task.task_id
                assert orig.tier == loaded_task.tier
                assert orig.domain == loaded_task.domain

    def test_task_to_dict_roundtrip(self):
        """A single task can be serialized and deserialized."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema={"input": "string"},
            output_type="test_output",
            functional_category="fetch",
            semantic_tags=["test"],
        )
        task = ToolBlindTask(
            task_id="test_001",
            tier=1,
            domain="web",
            goal="Test goal with enough length to pass validation",
            steps=[
                TaskStep(0, "Step 1 description", "test_tool", "test_output"),
                TaskStep(1, "Step 2 description", "other_tool", "other_output"),
            ],
            gap_step_index=0,
            available_tools=[tool],
            unavailable_tool=Tool("missing", "Missing tool", {}, "missing_output", "fetch", ["test"]),
            ground_truth_outcome="SUBSTITUTE",
            gold_solution=["test_tool"],
            unavailability_reason="rate_limited",
            registry_size=1,
            commitment_depth=0,
            difficulty="easy",
        )
        d = task.to_dict()
        restored = ToolBlindTask.from_dict(d)
        assert restored.task_id == task.task_id
        assert restored.tier == task.tier
        assert len(restored.steps) == len(task.steps)
        assert restored.unavailable_tool.name == task.unavailable_tool.name


class TestFullValidation:
    """Integration test for full validation."""

    def test_full_dataset_validates(self, dataset):
        """The complete generated dataset passes all validation checks."""
        assert validate_dataset(dataset, strict=True)
