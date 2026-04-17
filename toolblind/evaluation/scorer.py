"""Outcome classification and scoring for ToolBlind benchmark."""

import json
from typing import Any, Dict, List, Optional

from toolblind.agents.base import AgentAction
from toolblind.dataset.tasks import ToolBlindTask
from toolblind.environment.registry import ToolRegistry
from toolblind.environment.trajectory import TrajectoryState
from toolblind.utils.cache import get_cache
from toolblind.utils.config import get_config
from toolblind.utils.logging import get_logger

logger = get_logger("scorer")

VALID_OUTCOMES = {"SUBSTITUTE", "DECOMPOSE", "HALT", "CONFABULATE", "FALSE_SUBSTITUTE"}


class OutcomeScorer:
    """Classifies agent responses into outcome categories."""

    def __init__(self, use_llm_judge: bool = True):
        """Initialize the scorer.

        Args:
            use_llm_judge: Whether to use LLM judge for ambiguous cases.
        """
        self._use_llm_judge = use_llm_judge
        self._config = get_config()
        self._cache = get_cache()

    def classify(self, task: ToolBlindTask, trajectory: TrajectoryState) -> Dict[str, Any]:
        """Classify the agent's response at the gap into an outcome.

        Returns a dict with:
            outcome: str (one of VALID_OUTCOMES)
            confidence: float
            reasoning: str
            functional_reasoning_score: int (0-3)
        """
        if not trajectory.agent_actions_at_gap or not trajectory.agent_response_at_gap:
            return {
                "outcome": "CONFABULATE",
                "confidence": 0.0,
                "reasoning": "No agent response recorded at gap",
                "functional_reasoning_score": 0,
            }

        action_dict = trajectory.agent_actions_at_gap[0]
        action = AgentAction.from_dict(action_dict)

        # Stage 1: Rule-based classification
        result = self._rule_based_classify(task, action)
        if result is not None:
            return result

        # Stage 2: LLM judge fallback for ambiguous cases
        if self._use_llm_judge:
            return self._llm_judge_classify(task, action, trajectory)

        # No judge available: default to CONFABULATE for unclassified
        return {
            "outcome": "CONFABULATE",
            "confidence": 0.3,
            "reasoning": "Could not classify via rules and LLM judge is disabled",
            "functional_reasoning_score": 0,
        }

    def _rule_based_classify(self, task: ToolBlindTask, action: AgentAction) -> Optional[Dict[str, Any]]:
        """Attempt rule-based outcome classification."""
        available_names = {t.name for t in task.available_tools}
        unavailable_name = task.unavailable_tool.name

        # CONFABULATE: agent called the missing tool
        if action.tool_name == unavailable_name:
            return {
                "outcome": "CONFABULATE",
                "confidence": 1.0,
                "reasoning": f"Agent called the unavailable tool '{unavailable_name}'",
                "functional_reasoning_score": 0,
            }

        # CONFABULATE: agent called a tool not in available set and not known
        if (action.action_type == "CALL_TOOL"
                and action.tool_name
                and action.tool_name not in available_names
                and action.tool_name != unavailable_name):
            return {
                "outcome": "CONFABULATE",
                "confidence": 0.9,
                "reasoning": f"Agent called '{action.tool_name}' which is not in available tools",
                "functional_reasoning_score": 0,
            }

        # HALT: agent explicitly halted with a reason
        if action.action_type == "HALT" and action.halt_reason and len(action.halt_reason) > 20:
            frs = self._score_halt_reasoning(task, action)
            return {
                "outcome": "HALT",
                "confidence": 0.9,
                "reasoning": f"Agent halted with reason: {action.halt_reason[:200]}",
                "functional_reasoning_score": frs,
            }

        # SUBSTITUTE: agent called a tool in the gold solution (tier 1)
        if (action.action_type == "CALL_TOOL"
                and task.tier == 1
                and task.gold_solution
                and action.tool_name in task.gold_solution
                and action.tool_name in available_names):
            # Check for false substitute
            if self._is_false_substitute(task, action.tool_name):
                return {
                    "outcome": "FALSE_SUBSTITUTE",
                    "confidence": 0.8,
                    "reasoning": f"Agent used '{action.tool_name}' but it is not functionally equivalent",
                    "functional_reasoning_score": 1,
                }
            frs = self._score_substitute_reasoning(task, action)
            return {
                "outcome": "SUBSTITUTE",
                "confidence": 0.9,
                "reasoning": f"Agent correctly substituted with '{action.tool_name}'",
                "functional_reasoning_score": frs,
            }

        # DECOMPOSE: agent starts a valid decomposition chain (tier 2)
        if (action.action_type == "CALL_TOOL"
                and task.tier == 2
                and task.gold_solution
                and action.tool_name in task.gold_solution
                and action.tool_name in available_names):
            frs = self._score_decompose_reasoning(task, action)
            return {
                "outcome": "DECOMPOSE",
                "confidence": 0.85,
                "reasoning": f"Agent started decomposition with '{action.tool_name}'",
                "functional_reasoning_score": frs,
            }

        # Agent called a valid available tool but not in gold solution
        if (action.action_type == "CALL_TOOL"
                and action.tool_name
                and action.tool_name in available_names):
            # Could be a false substitute
            if self._is_false_substitute(task, action.tool_name):
                return {
                    "outcome": "FALSE_SUBSTITUTE",
                    "confidence": 0.7,
                    "reasoning": f"Agent used '{action.tool_name}' as substitute but it is not functionally equivalent",
                    "functional_reasoning_score": 1,
                }
            # Ambiguous — fall through to LLM judge
            return None

        return None

    def _is_false_substitute(self, task: ToolBlindTask, tool_name: str) -> bool:
        """Check if a tool is a false substitute by comparing functional properties."""
        registry = ToolRegistry(task.available_tools)
        candidate = registry.get(tool_name)
        if candidate is None:
            return True

        missing = task.unavailable_tool

        # Check output type match
        if candidate.output_type != missing.output_type:
            return True

        # Check semantic tag overlap
        candidate_tags = set(candidate.semantic_tags)
        missing_tags = set(missing.semantic_tags)
        if missing_tags:
            overlap = len(candidate_tags & missing_tags) / len(missing_tags)
            if overlap < 0.3:
                return True

        return False

    def _score_halt_reasoning(self, task: ToolBlindTask, action: AgentAction) -> int:
        """Score the quality of halt reasoning (0-3)."""
        score = 0
        reasoning = (action.reasoning + " " + (action.halt_reason or "")).lower()

        # Did agent identify WHY the tool is missing?
        if any(r in reasoning for r in ["unavailable", "missing", "not available", "cannot access"]):
            score += 1

        # Did agent reason about the functional gap?
        missing_desc_words = task.unavailable_tool.description.lower().split()
        functional_words = [w for w in missing_desc_words if len(w) > 4]
        if any(w in reasoning for w in functional_words[:5]):
            score += 1

        # Did agent consider composition before halting?
        if any(w in reasoning for w in ["combine", "compose", "chain", "alternative", "substitute", "approximate"]):
            score += 1

        return min(score, 3)

    def _score_substitute_reasoning(self, task: ToolBlindTask, action: AgentAction) -> int:
        """Score the quality of substitute reasoning (0-3)."""
        score = 0
        reasoning = action.reasoning.lower()

        if any(w in reasoning for w in ["unavailable", "missing", "not available"]):
            score += 1
        if any(w in reasoning for w in ["similar", "equivalent", "instead", "alternative", "substitute"]):
            score += 1
        if any(w in reasoning for w in ["output", "returns", "produces", "function"]):
            score += 1

        return min(score, 3)

    def _score_decompose_reasoning(self, task: ToolBlindTask, action: AgentAction) -> int:
        """Score the quality of decompose reasoning (0-3)."""
        score = 0
        reasoning = action.reasoning.lower()

        if any(w in reasoning for w in ["unavailable", "missing", "not available"]):
            score += 1
        if any(w in reasoning for w in ["combine", "compose", "chain", "multiple", "together", "then"]):
            score += 1
        if task.gold_solution and len(task.gold_solution) > 1:
            mentioned = sum(1 for gs in task.gold_solution if gs.lower() in reasoning)
            if mentioned >= 2:
                score += 1

        return min(score, 3)

    def _llm_judge_classify(
        self, task: ToolBlindTask, action: AgentAction, trajectory: TrajectoryState
    ) -> Dict[str, Any]:
        """Use Claude as a judge for ambiguous cases."""
        cache_key = f"judge:{task.task_id}:{action.to_dict()}"
        cached = self._cache.get(self._config.judge_model, cache_key)
        if cached is not None:
            return cached

        prompt = self._build_judge_prompt(task, action)

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self._config.anthropic_api_key)
            response = client.messages.create(
                model=self._config.judge_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text

            result = self._parse_judge_response(response_text)
            self._cache.put(self._config.judge_model, cache_key, result)
            return result

        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            return {
                "outcome": "CONFABULATE",
                "confidence": 0.3,
                "reasoning": f"LLM judge failed: {e}",
                "functional_reasoning_score": 0,
            }

    def _build_judge_prompt(self, task: ToolBlindTask, action: AgentAction) -> str:
        """Build the prompt for the LLM judge."""
        available_desc = "\n".join(
            f"  - {t.name}: {t.description} (output: {t.output_type})"
            for t in task.available_tools
        )

        return f"""You are evaluating an AI agent's response when a required tool becomes unavailable mid-task.

TASK GOAL: {task.goal}

UNAVAILABLE TOOL:
  Name: {task.unavailable_tool.name}
  Description: {task.unavailable_tool.description}
  Output type: {task.unavailable_tool.output_type}

AVAILABLE TOOLS:
{available_desc}

AGENT'S RESPONSE:
  Action type: {action.action_type}
  Tool called: {action.tool_name}
  Reasoning: {action.reasoning}
  Halt reason: {action.halt_reason}

Classify the agent's response as one of:
- SUBSTITUTE: Agent correctly identified a functionally equivalent available tool
- DECOMPOSE: Agent correctly proposed composing 2+ tools to approximate the missing function
- HALT: Agent correctly determined the task is infeasible and stopped with a valid reason
- CONFABULATE: Agent proceeded as if the missing tool exists or called a non-existent tool
- FALSE_SUBSTITUTE: Agent claimed a tool is equivalent when it functionally cannot be

Also evaluate:
- Is the agent's reasoning functionally correct? (yes/no + brief explanation)
- Functional reasoning score (0-3):
  0 = no reasoning about tool capabilities
  1 = identified tool is missing
  2 = reasoned about functional gap
  3 = comprehensive functional analysis

Respond with JSON:
{{
  "outcome": "<SUBSTITUTE|DECOMPOSE|HALT|CONFABULATE|FALSE_SUBSTITUTE>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>",
  "functionally_correct": <true|false>,
  "functional_reasoning_score": <0-3>
}}"""

    def _parse_judge_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM judge's response."""
        try:
            # Try direct parse
            text = response_text.strip()
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(text[json_start:json_end])
                outcome = data.get("outcome", "CONFABULATE").upper()
                if outcome not in VALID_OUTCOMES:
                    outcome = "CONFABULATE"
                return {
                    "outcome": outcome,
                    "confidence": float(data.get("confidence", 0.5)),
                    "reasoning": data.get("reasoning", response_text[:200]),
                    "functional_reasoning_score": int(data.get("functional_reasoning_score", 0)),
                }
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        return {
            "outcome": "CONFABULATE",
            "confidence": 0.3,
            "reasoning": f"Could not parse judge response: {response_text[:200]}",
            "functional_reasoning_score": 0,
        }

    def batch_classify(
        self, tasks: List[ToolBlindTask], trajectories: List[TrajectoryState]
    ) -> List[Dict[str, Any]]:
        """Classify a batch of trajectories."""
        results = []
        for task, traj in zip(tasks, trajectories):
            result = self.classify(task, traj)
            results.append(result)
        return results
