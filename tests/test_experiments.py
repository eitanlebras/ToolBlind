"""Tests for ToolBlind experiment implementations."""

import json
from typing import List, Optional

import pytest

from toolblind.agents.base import AgentAction, BaseAgent
from toolblind.dataset.generator import generate_dataset, save_dataset
from toolblind.dataset.tasks import Tool
from toolblind.environment.simulator import ToolUnavailableError
from toolblind.environment.trajectory import CompletedStep
from toolblind.experiments.baseline import run_baseline
from toolblind.experiments.commitment import run_commitment_experiment
from toolblind.experiments.framing import run_framing_experiment
from toolblind.experiments.registry_size import run_registry_size_experiment
from toolblind.utils.config import Config, set_config


class StubAgent(BaseAgent):
    """A stub agent that returns deterministic actions for testing."""

    def __init__(self, default_action: str = "HALT"):
        """Initialize with a default action type."""
        self._default_action = default_action

    def name(self) -> str:
        """Return stub agent name."""
        return "stub-agent"

    def plan_step(
        self,
        goal: str,
        available_tools: List[Tool],
        trajectory_so_far: List[CompletedStep],
        current_step_description: str,
        tool_error: Optional[ToolUnavailableError] = None,
    ) -> AgentAction:
        """Return a deterministic action based on whether there's a tool error."""
        if tool_error is not None:
            if self._default_action == "HALT":
                return AgentAction(
                    action_type="HALT",
                    reasoning="The required tool is unavailable and no substitute exists in the current toolset",
                    halt_reason="Tool is unavailable and cannot be approximated by any available tools",
                    confidence=0.8,
                )
            elif self._default_action == "SUBSTITUTE":
                # Pick the first available tool
                tool_name = available_tools[0].name if available_tools else None
                return AgentAction(
                    action_type="CALL_TOOL",
                    tool_name=tool_name,
                    tool_params={},
                    reasoning="Using available tool as substitute",
                    confidence=0.7,
                )
            else:
                return AgentAction(
                    action_type="CALL_TOOL",
                    tool_name=tool_error.tool_name,  # Confabulate
                    reasoning="Proceeding with the tool",
                    confidence=0.9,
                )
        else:
            # Normal step — use the first available tool
            tool_name = available_tools[0].name if available_tools else "unknown"
            return AgentAction(
                action_type="CALL_TOOL",
                tool_name=tool_name,
                tool_params={},
                reasoning="Executing the step",
                confidence=0.9,
            )


@pytest.fixture
def test_config(tmp_path):
    """Create a test config with temporary directories."""
    config = Config(
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        google_api_key="test-key",
        cache_dir=str(tmp_path / "cache"),
        results_dir=str(tmp_path / "results"),
        tasks_dir=str(tmp_path / "tasks"),
        log_level="WARNING",
        max_retries=1,
    )
    set_config(config)
    return config


@pytest.fixture
def small_dataset(test_config):
    """Generate and save a small dataset for testing."""
    tasks = generate_dataset(seed=42)
    save_dataset(tasks, test_config.tasks_dir)
    return tasks


class TestExperiment1Baseline:
    """Tests for baseline experiment."""

    def test_experiment1_runs_on_sample(self, small_dataset, test_config):
        """Baseline experiment completes with --sample 2 without error."""
        agent = StubAgent(default_action="HALT")
        result = run_baseline(
            small_dataset, [agent], sample_size=2, seed=42, use_llm_judge=False
        )
        assert "results" in result
        assert len(result["results"]) > 0
        assert result["experiment"] == "baseline"

    def test_results_serializable(self, small_dataset, test_config):
        """Output JSON is valid and complete."""
        agent = StubAgent(default_action="HALT")
        result = run_baseline(
            small_dataset, [agent], sample_size=1, seed=42, use_llm_judge=False
        )
        # Verify JSON serializable
        json_str = json.dumps(result, default=str)
        loaded = json.loads(json_str)
        assert loaded["experiment"] == "baseline"
        assert "metrics" in loaded
        assert "results" in loaded

    def test_confabulating_agent(self, small_dataset, test_config):
        """Agent that confabulates has high confabulation rate."""
        agent = StubAgent(default_action="CONFABULATE")
        result = run_baseline(
            small_dataset, [agent], sample_size=2, seed=42, use_llm_judge=False
        )
        confab_count = sum(1 for r in result["results"] if r["outcome"] == "CONFABULATE")
        assert confab_count > 0


class TestExperiment2Commitment:
    """Tests for commitment depth experiment."""

    def test_experiment2_depth_variants(self, small_dataset, test_config):
        """5 depth variants are generated per task."""
        agent = StubAgent(default_action="HALT")
        result = run_commitment_experiment(
            small_dataset, [agent], sample_size=2, seed=42, use_llm_judge=False
        )
        assert "results" in result
        assert "commitment_effect" in result
        # Check that multiple depths are represented
        depths = set(r["commitment_depth"] for r in result["results"])
        assert len(depths) > 1


class TestExperiment3Framing:
    """Tests for framing ablation experiment."""

    def test_experiment3_four_framings(self, small_dataset, test_config):
        """4 framing conditions are tested."""
        agent = StubAgent(default_action="HALT")
        result = run_framing_experiment(
            small_dataset, [agent], sample_size=3, seed=42, use_llm_judge=False
        )
        assert "results" in result
        reasons = set(r["unavailability_reason"] for r in result["results"])
        assert len(reasons) >= 2  # At least some framing variety


class TestExperiment4RegistrySize:
    """Tests for registry size experiment."""

    def test_experiment4_registry_sizes(self, small_dataset, test_config):
        """Multiple registry sizes are tested."""
        agent = StubAgent(default_action="HALT")
        result = run_registry_size_experiment(
            small_dataset, [agent], sample_size=2, seed=42, use_llm_judge=False
        )
        assert "results" in result
        sizes = set(r["registry_size"] for r in result["results"])
        assert len(sizes) > 1
