"""Tests for CiCRouter smart routing."""

import pytest

from cic.routing import CiCRouter, DEFAULT_ROUTING, COMPLEXITY_LEVELS


class TestCiCRouter:
    def test_default_routing(self):
        router = CiCRouter()
        assert router.model_for("simple") == "haiku"
        assert router.model_for("moderate") == "sonnet"
        assert router.model_for("complex") == "opus"

    def test_custom_routing(self):
        router = CiCRouter(simple="haiku", moderate="sonnet", complex="sonnet")
        assert router.model_for("complex") == "sonnet"

    def test_from_dict(self):
        router = CiCRouter.from_dict({"simple": "haiku", "complex": "opus"})
        assert router.model_for("simple") == "haiku"
        assert router.model_for("complex") == "opus"

    def test_from_dict_fills_missing_with_defaults(self):
        router = CiCRouter.from_dict({"complex": "opus"})
        # "moderate" and "simple" should come from DEFAULT_ROUTING
        assert router.model_for("moderate") == DEFAULT_ROUTING["moderate"]

    def test_unknown_complexity_falls_back_to_sonnet(self):
        router = CiCRouter()
        assert router.model_for("unknownthing") == "sonnet"

    def test_internal_levels_mapped(self):
        router = CiCRouter()
        assert router.model_for("heartbeat") == "haiku"
        assert router.model_for("routine") == "sonnet"
        assert router.model_for("very_complex") == "opus"

    def test_extra_mappings(self):
        router = CiCRouter(extra={"experimental": "claude-3-7-sonnet-20250219"})
        assert router.model_for("experimental") == "claude-3-7-sonnet-20250219"

    def test_as_dict_returns_copy(self):
        router = CiCRouter()
        d = router.as_dict()
        d["simple"] = "mutated"
        assert router.model_for("simple") == "haiku"  # original unaffected

    def test_repr_shows_public_levels(self):
        router = CiCRouter()
        r = repr(router)
        assert "simple" in r
        assert "moderate" in r
        assert "complex" in r

    def test_complexity_levels_constant(self):
        assert "simple" in COMPLEXITY_LEVELS
        assert "moderate" in COMPLEXITY_LEVELS
        assert "complex" in COMPLEXITY_LEVELS
