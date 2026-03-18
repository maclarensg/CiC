"""Smart routing: map task complexity to model names.

CiCRouter lets callers associate complexity levels with specific model
identifiers. The current complexity is set via ``set_complexity()`` and
used by ``CiCClient`` to select the appropriate model for each call.
"""

from __future__ import annotations

# Complexity levels understood by CiCRouter, from lightest to heaviest.
COMPLEXITY_LEVELS = ("simple", "moderate", "complex")

# Default routing used when no custom routing is provided.
# Maps the three public complexity levels and the internal levels
# from the Lilith consciousness loop to sensible Claude model names.
DEFAULT_ROUTING: dict[str, str] = {
    # Public levels
    "simple": "haiku",
    "moderate": "sonnet",
    "complex": "opus",
    # Internal levels used by the Lilith agent loop (kept for compatibility)
    "heartbeat": "haiku",
    "routine": "sonnet",
    "very_complex": "opus",
}


class CiCRouter:
    """Maps complexity levels to model names for smart routing.

    Usage::

        router = CiCRouter(simple="haiku", moderate="sonnet", complex="opus")
        model = router.model_for("moderate")  # "sonnet"

    Args:
        simple: Model for simple/fast tasks (default: ``"haiku"``).
        moderate: Model for balanced tasks (default: ``"sonnet"``).
        complex: Model for heavy reasoning (default: ``"opus"``).
        extra: Additional complexity → model mappings (merged on top).
    """

    def __init__(
        self,
        *,
        simple: str = "haiku",
        moderate: str = "sonnet",
        complex: str = "opus",
        extra: dict[str, str] | None = None,
    ) -> None:
        self._routing: dict[str, str] = {
            "simple": simple,
            "moderate": moderate,
            "complex": complex,
            # Keep internal levels mapped to keep compatibility
            "heartbeat": simple,
            "routine": moderate,
            "very_complex": complex,
        }
        if extra:
            self._routing.update(extra)

    @classmethod
    def from_dict(cls, mapping: dict[str, str]) -> "CiCRouter":
        """Build a router from a raw complexity → model mapping dict.

        The dict must contain at least one of the three public complexity
        levels (``"simple"``, ``"moderate"``, ``"complex"``). Missing levels
        fall back to defaults.

        Args:
            mapping: A dict like ``{"simple": "haiku", "complex": "opus"}``.

        Returns:
            A configured CiCRouter.
        """
        defaults = DEFAULT_ROUTING.copy()
        defaults.update(mapping)
        instance = cls.__new__(cls)
        instance._routing = defaults
        return instance

    def model_for(self, complexity: str) -> str:
        """Return the model name for the given complexity level.

        Falls back to ``"sonnet"`` if the complexity is not in the routing
        table — a sensible middle ground.

        Args:
            complexity: A complexity level string, e.g. ``"moderate"``.

        Returns:
            The model name to use, e.g. ``"sonnet"``.
        """
        return self._routing.get(complexity, "sonnet")

    def as_dict(self) -> dict[str, str]:
        """Return the raw routing table as a copy."""
        return dict(self._routing)

    def __repr__(self) -> str:
        public = {k: v for k, v in self._routing.items() if k in COMPLEXITY_LEVELS}
        return f"CiCRouter({public!r})"
