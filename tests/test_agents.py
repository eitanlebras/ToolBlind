"""Tests for ToolBlind agent implementations."""

import json

from toolblind.agents.base import AgentAction, BaseAgent, build_step_prompt, build_system_prompt, parse_agent_response
from toolblind.agents.react import ReActWrapper
from toolblind.dataset.tasks import Tool
from toolblind.environment.simulator import ToolUnavailableError


def _make_tool(name: str) -> Tool:
    """Helper to create a tool for testing."""
    return Tool(
        name=name,
        description=f"Test tool {name}",
        input_schema={"input": "string"},
        output_type="test_output",
        functional_category="fetch",
        semantic_tags=["test"],
    )


class TestAgentActionParsing:
    """Tests for agent response parsing."""

    def test_parse_valid_json(self):
        """Valid JSON response is parsed correctly."""
        response = json.dumps({
            "action_type": "CALL_TOOL",
            "tool_name": "fetch_url",
            "tool_params": {"url": "https://example.com"},
            "reasoning": "I need to fetch the page",
            "halt_reason": None,
            "confidence": 0.9,
        })
        action = parse_agent_response(response)
        assert action.action_type == "CALL_TOOL"
        assert action.tool_name == "fetch_url"
        assert action.confidence == 0.9

    def test_parse_json_in_markdown(self):
        """JSON inside markdown code blocks is extracted."""
        response = "Here is my response:\n```json\n" + json.dumps({
            "action_type": "HALT",
            "reasoning": "Cannot proceed",
            "halt_reason": "Tool is not available and no substitute exists",
            "confidence": 0.8,
        }) + "\n```"
        action = parse_agent_response(response)
        assert action.action_type == "HALT"
        assert action.halt_reason is not None

    def test_parse_json_with_surrounding_text(self):
        """JSON embedded in text is extracted."""
        response = 'Let me think about this. {"action_type": "CALL_TOOL", "tool_name": "search_web", "reasoning": "trying alternative"} That should work.'
        action = parse_agent_response(response)
        assert action.action_type == "CALL_TOOL"
        assert action.tool_name == "search_web"

    def test_parse_malformed_json(self):
        """Malformed JSON is handled gracefully with fallback."""
        response = "I cannot do this because the tool is missing. There is no way to proceed."
        action = parse_agent_response(response)
        # Should fall back to HALT detection based on keywords
        assert action.action_type == "HALT"
        assert action.reasoning == response

    def test_parse_empty_response(self):
        """Empty response produces a fallback action."""
        action = parse_agent_response("")
        assert action.action_type in ("CALL_TOOL", "HALT")

    def test_parse_preserves_all_fields(self):
        """All fields from JSON are preserved in the action."""
        data = {
            "action_type": "CALL_TOOL",
            "tool_name": "http_get",
            "tool_params": {"url": "https://api.example.com", "headers": {"auth": "token"}},
            "reasoning": "Fetching API data",
            "halt_reason": None,
            "confidence": 0.95,
        }
        action = parse_agent_response(json.dumps(data))
        assert action.tool_params == data["tool_params"]
        assert action.reasoning == data["reasoning"]


class TestSystemPrompt:
    """Tests for prompt building."""

    def test_cot_mode_difference(self):
        """CoT prompt includes explicit reasoning instructions that direct does not."""
        tools = [_make_tool("tool_a")]
        cot_prompt = build_system_prompt(tools, cot=True)
        direct_prompt = build_system_prompt(tools, cot=False)

        # CoT has explicit reasoning steps instruction
        assert "Before choosing an action" in cot_prompt
        assert "Before choosing an action" not in direct_prompt

    def test_prompt_includes_all_tools(self):
        """System prompt lists all available tools."""
        tools = [_make_tool("alpha"), _make_tool("beta"), _make_tool("gamma")]
        prompt = build_system_prompt(tools)
        for tool in tools:
            assert tool.name in prompt
            assert tool.description in prompt

    def test_step_prompt_includes_error(self):
        """Step prompt includes the tool error when present."""
        error = ToolUnavailableError("missing_tool", "This tool has been decommissioned")
        prompt = build_step_prompt(
            goal="Test goal",
            trajectory_so_far=[],
            current_step_description="Do the thing",
            tool_error=error,
        )
        assert "missing_tool" in prompt
        assert "decommissioned" in prompt
        assert "unavailable" in prompt.lower()


class TestReActWrapper:
    """Tests for ReAct prompting wrapper."""

    def test_react_wrapper_format(self):
        """ReAct wrapper produces Thought/Action/Observation structure."""

        class MockAgent(BaseAgent):
            """Mock agent for testing."""

            def name(self) -> str:
                """Return mock name."""
                return "mock"

            def plan_step(self, goal, available_tools, trajectory_so_far,
                          current_step_description, tool_error=None):
                """Return a mock action."""
                return AgentAction(
                    action_type="CALL_TOOL",
                    tool_name="search_web",
                    tool_params={"query": "test"},
                    reasoning="Looking for alternatives",
                    confidence=0.7,
                )

        mock = MockAgent()
        react = ReActWrapper(mock)
        assert react.name() == "react-mock"

        action = react.plan_step(
            goal="Test goal",
            available_tools=[_make_tool("search_web")],
            trajectory_so_far=[],
            current_step_description="Find information",
        )

        assert "Thought:" in action.reasoning
        assert "Action:" in action.reasoning
        assert "search_web" in action.reasoning

    def test_react_wrapper_with_error(self):
        """ReAct wrapper handles tool errors in the prompt."""

        class MockHaltAgent(BaseAgent):
            """Mock agent that halts."""

            def name(self) -> str:
                """Return mock name."""
                return "mock-halt"

            def plan_step(self, goal, available_tools, trajectory_so_far,
                          current_step_description, tool_error=None):
                """Return a halt action."""
                return AgentAction(
                    action_type="HALT",
                    reasoning="Tool is missing and no alternative exists",
                    halt_reason="No functional substitute available for the required capability",
                    confidence=0.8,
                )

        mock = MockHaltAgent()
        react = ReActWrapper(mock)
        error = ToolUnavailableError("missing_tool", "Decommissioned")

        action = react.plan_step(
            goal="Test goal",
            available_tools=[_make_tool("tool_a")],
            trajectory_so_far=[],
            current_step_description="Do the thing",
            tool_error=error,
        )

        assert action.action_type == "HALT"
        assert "Thought:" in action.reasoning
        assert "Action: HALT" in action.reasoning


class TestCacheAndRetry:
    """Tests for cache hit and retry logic."""

    def test_cache_hit(self):
        """Second identical call returns cached response."""
        import tempfile

        from toolblind.utils.cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=tmpdir)
            data = {"action_type": "HALT", "reasoning": "cached"}

            # First call - miss
            assert cache.get("test-model", "prompt1") is None

            # Store
            cache.put("test-model", "prompt1", data)

            # Second call - hit
            result = cache.get("test-model", "prompt1")
            assert result is not None
            assert result["reasoning"] == "cached"

            cache.close()

    def test_action_serialization_roundtrip(self):
        """AgentAction can be serialized and deserialized."""
        action = AgentAction(
            action_type="CALL_TOOL",
            tool_name="fetch_url",
            tool_params={"url": "https://example.com"},
            reasoning="Fetching page",
            halt_reason=None,
            confidence=0.85,
            tokens_used=500,
        )
        d = action.to_dict()
        restored = AgentAction.from_dict(d)
        assert restored.action_type == action.action_type
        assert restored.tool_name == action.tool_name
        assert restored.confidence == action.confidence
        assert restored.tokens_used == action.tokens_used
