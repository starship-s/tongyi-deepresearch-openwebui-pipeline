"""Tests for visit tool formatting helpers."""

from __future__ import annotations

from tongyi_deepresearch_openwebui_pipeline.tools.deepresearch_visit_tool import Tools


class TestFmtVisitOk:
    """Tests for Tools._fmt_visit_ok."""

    def test_contains_all_fields(self) -> None:
        result = Tools._fmt_visit_ok(
            url="https://example.com",
            goal="find pricing info",
            evidence="Product costs $10/mo",
            summary="The product has a monthly plan at $10.",
        )
        assert "https://example.com" in result
        assert "find pricing info" in result
        assert "Product costs $10/mo" in result
        assert "The product has a monthly plan at $10." in result

    def test_output_structure(self) -> None:
        result = Tools._fmt_visit_ok("http://x.com", "goal", "ev", "sum")
        assert "Evidence in page:" in result
        assert "Summary:" in result


class TestFmtVisitError:
    """Tests for Tools._fmt_visit_error."""

    def test_contains_url_and_goal(self) -> None:
        result = Tools._fmt_visit_error("https://example.com", "find data", "timeout")
        assert "https://example.com" in result
        assert "find data" in result

    def test_error_message_structure(self) -> None:
        result = Tools._fmt_visit_error("http://x.com", "goal", "err")
        assert "could not be accessed" in result
        assert "could not be processed" in result

    def test_does_not_leak_raw_error(self) -> None:
        result = Tools._fmt_visit_error("http://x.com", "goal", "secret")
        assert "secret" not in result
