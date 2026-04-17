"""Tests for ToolBlind environment simulation."""

import pytest

from toolblind.dataset.tasks import Tool
from toolblind.environment.registry import ToolRegistry
from toolblind.environment.simulator import ToolResult, ToolSimulator, ToolUnavailableError
from toolblind.environment.trajectory import CompletedStep, TrajectoryState


class TestToolSimulator:
    """Tests for the tool execution simulator."""

    def test_tool_execution_deterministic(self):
        """Same tool call with same params always returns the same result."""
        sim = ToolSimulator()
        r1 = sim.execute("fetch_url", {"url": "https://example.com"})
        r2 = sim.execute("fetch_url", {"url": "https://example.com"})
        assert r1.output == r2.output
        assert r1.latency_ms == r2.latency_ms
        assert r1.output_type == r2.output_type

    def test_different_params_different_output(self):
        """Different params produce different outputs."""
        sim = ToolSimulator()
        r1 = sim.execute("fetch_url", {"url": "https://example.com"})
        r2 = sim.execute("fetch_url", {"url": "https://other.com"})
        # Outputs should differ since URLs differ
        assert r1.output != r2.output

    def test_unavailable_tool_raises(self):
        """ToolUnavailableError is raised for unavailable tools."""
        sim = ToolSimulator(
            unavailable_tools={"fetch_url"},
            unavailability_reasons={"fetch_url": "This tool is rate-limited"},
        )
        with pytest.raises(ToolUnavailableError) as exc_info:
            sim.execute("fetch_url", {"url": "https://example.com"})
        assert "rate-limited" in str(exc_info.value)
        assert exc_info.value.tool_name == "fetch_url"

    def test_available_tool_works(self):
        """Available tools execute normally even when others are unavailable."""
        sim = ToolSimulator(unavailable_tools={"fetch_url"})
        result = sim.execute("search_web", {"query": "test", "num_results": 3})
        assert result.success is True
        assert result.output_type == "search_results"

    def test_mock_outputs_realistic(self):
        """Mock outputs are not placeholder strings."""
        sim = ToolSimulator()

        # Web domain
        result = sim.execute("fetch_url", {"url": "https://example.com"})
        assert "<!DOCTYPE html>" in result.output
        assert len(result.output) > 50

        # Database domain
        result = sim.execute("sql_query", {"query": "SELECT * FROM users", "params": []})
        assert "rows" in result.output
        assert len(result.output["rows"]) > 0
        assert "id" in result.output["rows"][0]

        # Code domain
        result = sim.execute("execute_python", {"code": "print('hello')", "timeout": 10})
        assert "stdout" in result.output
        assert "exit_code" in result.output

    def test_all_domain_tools_implemented(self):
        """Every tool in every domain pool has a working mock."""
        sim = ToolSimulator()
        domain_tools = {
            "web": ["fetch_url", "search_web", "parse_html", "extract_links", "extract_text",
                     "screenshot_page", "check_status", "download_file", "submit_form", "get_headers"],
            "code": ["read_file", "write_file", "execute_python", "execute_bash", "lint_code",
                      "format_code", "parse_ast", "git_commit", "install_package", "run_tests"],
            "file": ["read_file_content", "write_file_content", "list_directory", "copy_file", "delete_file",
                      "compress_file", "extract_archive", "get_metadata", "convert_format", "search_content"],
            "api": ["http_get", "http_post", "parse_json", "authenticate_oauth", "rate_limit_wait",
                     "cache_response", "validate_schema", "retry_request", "log_request", "transform_response"],
            "database": ["sql_query", "sql_insert", "sql_update", "connect_db", "list_tables",
                          "get_schema", "export_csv", "run_migration", "backup_table", "check_constraints"],
        }
        for domain, tools in domain_tools.items():
            for tool_name in tools:
                result = sim.execute(tool_name, {})
                assert result.success is True, f"Tool '{tool_name}' in domain '{domain}' failed"
                assert result.output is not None

    def test_call_log(self):
        """Simulator logs all calls."""
        sim = ToolSimulator()
        sim.execute("fetch_url", {"url": "https://example.com"})
        sim.execute("search_web", {"query": "test", "num_results": 3})
        log = sim.get_call_log()
        assert len(log) == 2
        assert log[0]["tool"] == "fetch_url"
        assert log[1]["tool"] == "search_web"


class TestTrajectoryState:
    """Tests for trajectory state management."""

    def test_trajectory_state_updates(self):
        """steps_completed increments correctly."""
        state = TrajectoryState(task_id="test_001", gap_step_index=2)
        assert len(state.steps_completed) == 0
        assert state.current_step_index == 0

        step = CompletedStep(
            step_index=0,
            tool_called="fetch_url",
            params={"url": "https://example.com"},
            result=ToolResult(True, "html content", "html_string", 100),
            agent_reasoning="I need to fetch the page",
        )
        state.add_completed_step(step)
        assert len(state.steps_completed) == 1
        assert state.current_step_index == 1

    def test_gap_encountered_marking(self):
        """Gap can be marked with response."""
        state = TrajectoryState(task_id="test_001", gap_step_index=1)
        assert state.gap_encountered is False

        state.mark_gap_encountered("I will try an alternative tool", [{"action_type": "CALL_TOOL"}])
        assert state.gap_encountered is True
        assert state.agent_response_at_gap == "I will try an alternative tool"

    def test_trajectory_serialization(self):
        """TrajectoryState can be serialized and deserialized."""
        state = TrajectoryState(task_id="test_002", gap_step_index=1)
        step = CompletedStep(
            step_index=0,
            tool_called="sql_query",
            params={"query": "SELECT 1"},
            result=ToolResult(True, {"rows": [{"a": 1}]}, "query_results", 50),
            agent_reasoning="Querying database",
        )
        state.add_completed_step(step)
        state.mark_gap_encountered("Halting because tool is missing")
        state.outcome = "HALT"

        d = state.to_dict()
        restored = TrajectoryState.from_dict(d)
        assert restored.task_id == "test_002"
        assert len(restored.steps_completed) == 1
        assert restored.gap_encountered is True
        assert restored.outcome == "HALT"


class TestToolRegistry:
    """Tests for the tool registry."""

    def test_functional_equivalence(self):
        """Functional equivalence scoring works correctly."""
        tool_a = Tool("fetch_url", "Fetch URL", {"url": "string"}, "html_string", "fetch", ["web", "http"])
        tool_b = Tool("search_web", "Search web", {"query": "string"}, "search_results", "fetch", ["web", "search"])
        tool_c = Tool("download_file", "Download file", {"url": "string"}, "file_path", "fetch", ["web", "http", "file"])

        registry = ToolRegistry([tool_a, tool_b, tool_c])

        # Same category but different output
        equiv_ab = registry.check_functional_equivalence(tool_a, tool_b)
        assert 0.0 < equiv_ab < 1.0

        # Same category, overlapping tags
        equiv_ac = registry.check_functional_equivalence(tool_a, tool_c)
        assert equiv_ac > 0

    def test_find_by_output_type(self):
        """Can find tools by output type."""
        tools = [
            Tool("t1", "Tool 1", {}, "html_string", "fetch", ["web"]),
            Tool("t2", "Tool 2", {}, "html_string", "transform", ["web"]),
            Tool("t3", "Tool 3", {}, "json_response", "fetch", ["api"]),
        ]
        registry = ToolRegistry(tools)
        matches = registry.find_by_output_type("html_string")
        assert len(matches) == 2
