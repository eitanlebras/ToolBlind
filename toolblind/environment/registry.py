"""Tool registry with functional metadata for ToolBlind."""

from typing import Dict, List, Optional, Set

from toolblind.dataset.tasks import Tool


class ToolRegistry:
    """Registry of tools with functional metadata for equivalence checking."""

    def __init__(self, tools: List[Tool]):
        """Initialize registry from a list of tools."""
        self._tools: Dict[str, Tool] = {t.name: t for t in tools}

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool exists in the registry."""
        return name in self._tools

    def list_names(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def list_tools(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def find_by_category(self, category: str) -> List[Tool]:
        """Find all tools with a given functional category."""
        return [t for t in self._tools.values() if t.functional_category == category]

    def find_by_output_type(self, output_type: str) -> List[Tool]:
        """Find all tools that produce a given output type."""
        return [t for t in self._tools.values() if t.output_type == output_type]

    def find_by_tags(self, tags: Set[str], min_overlap: int = 1) -> List[Tool]:
        """Find tools with at least min_overlap matching semantic tags."""
        results = []
        for tool in self._tools.values():
            overlap = len(set(tool.semantic_tags) & tags)
            if overlap >= min_overlap:
                results.append(tool)
        return results

    def check_functional_equivalence(self, tool_a: Tool, tool_b: Tool) -> float:
        """Compute a functional equivalence score between two tools (0.0 to 1.0)."""
        score = 0.0

        # Same output type: +0.4
        if tool_a.output_type == tool_b.output_type:
            score += 0.4

        # Same functional category: +0.2
        if tool_a.functional_category == tool_b.functional_category:
            score += 0.2

        # Semantic tag overlap: up to +0.4
        tags_a = set(tool_a.semantic_tags)
        tags_b = set(tool_b.semantic_tags)
        if tags_a or tags_b:
            jaccard = len(tags_a & tags_b) / len(tags_a | tags_b)
            score += 0.4 * jaccard

        return min(score, 1.0)

    def find_substitutes(self, missing_tool: Tool, threshold: float = 0.5) -> List[Tool]:
        """Find potential substitutes for a missing tool above a similarity threshold."""
        candidates = []
        for tool in self._tools.values():
            if tool.name == missing_tool.name:
                continue
            equiv = self.check_functional_equivalence(missing_tool, tool)
            if equiv >= threshold:
                candidates.append(tool)
        candidates.sort(
            key=lambda t: self.check_functional_equivalence(missing_tool, t),
            reverse=True,
        )
        return candidates

    def __len__(self) -> int:
        """Return number of tools in registry."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool name is in registry."""
        return name in self._tools
