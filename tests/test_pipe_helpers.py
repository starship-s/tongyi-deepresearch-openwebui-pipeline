"""Tests for pipe static/pure helpers and _CostTracker."""

from __future__ import annotations

from tongyi_deepresearch_openwebui_pipeline.pipes.tongyi_deepresearch_pipe import (
    Pipe,
    _CostTracker,
    _fmt_visit_fallback,
)

# ------------------------------------------------------------------ #
#  _extract_tool_call
# ------------------------------------------------------------------ #


class TestExtractToolCall:
    """Tests for Pipe._extract_tool_call."""

    def test_valid_json(self) -> None:
        text = (
            "Some preamble\n<tool_call>\n"
            '{"name": "search", "arguments": {"query": ["python asyncio"]}}\n'
            "</tool_call>\ntrailing text"
        )
        result = Pipe._extract_tool_call(text)
        assert result is not None
        assert result["name"] == "search"
        assert result["arguments"] == {"query": ["python asyncio"]}

    def test_python_interpreter(self) -> None:
        text = (
            "<tool_call>\n"
            '{"name": "PythonInterpreter", "arguments": '
            '{"code": "<code>print(42)</code>"}}\n'
            "</tool_call>"
        )
        result = Pipe._extract_tool_call(text)
        assert result is not None
        assert result["name"] == "PythonInterpreter"
        assert result["arguments"]["code"] == "print(42)"

    def test_malformed_json_trailing_comma(self) -> None:
        text = (
            "<tool_call>\n"
            "{'name': 'visit', 'arguments': {'url': ['http://example.com'],}}\n"
            "</tool_call>"
        )
        result = Pipe._extract_tool_call(text)
        assert result is not None
        assert result["name"] == "visit"

    def test_no_match(self) -> None:
        assert Pipe._extract_tool_call("no xml here") is None

    def test_empty_tool_call_tags(self) -> None:
        assert Pipe._extract_tool_call("<tool_call>  </tool_call>") is None

    def test_nested_whitespace(self) -> None:
        text = (
            "<tool_call>\n\n"
            '  {"name": "search", "arguments": {"query": ["test"]}}  \n\n'
            "</tool_call>"
        )
        result = Pipe._extract_tool_call(text)
        assert result is not None
        assert result["name"] == "search"


# ------------------------------------------------------------------ #
#  _extract_answer
# ------------------------------------------------------------------ #


class TestExtractAnswer:
    """Tests for Pipe._extract_answer."""

    def test_closed_tags(self) -> None:
        text = "preamble\n<answer>\nHello world\n</answer>\ntrailing"
        assert Pipe._extract_answer(text) == "Hello world"

    def test_unclosed_tag(self) -> None:
        text = "<answer>\nPartial answer without closing tag"
        assert Pipe._extract_answer(text) == "Partial answer without closing tag"

    def test_no_answer(self) -> None:
        assert Pipe._extract_answer("just plain text") is None


# ------------------------------------------------------------------ #
#  _extract_thinking
# ------------------------------------------------------------------ #


class TestExtractThinking:
    """Tests for Pipe._extract_thinking."""

    def test_extracts_think_content(self) -> None:
        text = "<think>\nI should search for X\n</think>\nContent"
        assert Pipe._extract_thinking(text) == "I should search for X"

    def test_no_think_tags(self) -> None:
        assert Pipe._extract_thinking("no thinking here") is None


# ------------------------------------------------------------------ #
#  _strip_xml_for_display
# ------------------------------------------------------------------ #


class TestStripXmlForDisplay:
    """Tests for Pipe._strip_xml_for_display."""

    def test_removes_think_tags(self) -> None:
        text = "<think>reasoning</think> visible"
        assert Pipe._strip_xml_for_display(text) == "visible"

    def test_removes_tool_call_tags(self) -> None:
        text = 'Hello <tool_call>{"name":"search"}</tool_call> world'
        assert Pipe._strip_xml_for_display(text) == "Hello  world"

    def test_removes_answer_tags(self) -> None:
        text = "<answer>final answer</answer>"
        assert Pipe._strip_xml_for_display(text) == "final answer"

    def test_combined(self) -> None:
        text = (
            "<think>hmm</think>"
            '<tool_call>{"name":"x"}</tool_call>'
            "<answer>result</answer>"
        )
        assert Pipe._strip_xml_for_display(text) == "result"


# ------------------------------------------------------------------ #
#  _reconstruct_full_turn
# ------------------------------------------------------------------ #


class TestReconstructFullTurn:
    """Tests for Pipe._reconstruct_full_turn."""

    def test_wraps_reasoning(self) -> None:
        result = Pipe._reconstruct_full_turn("thinking", "content")
        assert result == "<think>\nthinking\n</think>\ncontent"

    def test_no_double_wrap(self) -> None:
        content = "<think>\nalready\n</think>\ncontent"
        result = Pipe._reconstruct_full_turn("thinking", content)
        assert result == content

    def test_empty_reasoning(self) -> None:
        result = Pipe._reconstruct_full_turn("", "content")
        assert result == "content"


# ------------------------------------------------------------------ #
#  _enabled_tool_names
# ------------------------------------------------------------------ #


class TestEnabledToolNames:
    """Tests for Pipe._enabled_tool_names."""

    def test_all_enabled(self) -> None:
        pipe = Pipe()
        pipe.valves.SEARCH_ENABLED = True
        pipe.valves.VISIT_ENABLED = True
        pipe.valves.SCHOLAR_ENABLED = True
        assert pipe._enabled_tool_names() == [
            "search",
            "visit",
            "google_scholar",
        ]

    def test_partial(self) -> None:
        pipe = Pipe()
        pipe.valves.SEARCH_ENABLED = True
        pipe.valves.VISIT_ENABLED = False
        pipe.valves.SCHOLAR_ENABLED = False
        assert pipe._enabled_tool_names() == ["search"]

    def test_none_enabled(self) -> None:
        pipe = Pipe()
        pipe.valves.SEARCH_ENABLED = False
        pipe.valves.VISIT_ENABLED = False
        pipe.valves.SCHOLAR_ENABLED = False
        assert pipe._enabled_tool_names() == []


# ------------------------------------------------------------------ #
#  _build_system_prompt
# ------------------------------------------------------------------ #


class TestBuildSystemPrompt:
    """Tests for Pipe._build_system_prompt."""

    def test_includes_tool_definitions(self) -> None:
        pipe = Pipe()
        prompt = pipe._build_system_prompt()
        assert '"name": "search"' in prompt
        assert '"name": "visit"' in prompt
        assert '"name": "google_scholar"' in prompt
        assert "<tools>" in prompt

    def test_excludes_disabled_tools(self) -> None:
        pipe = Pipe()
        pipe.valves.SCHOLAR_ENABLED = False
        prompt = pipe._build_system_prompt()
        assert '"name": "search"' in prompt
        assert '"name": "visit"' in prompt
        assert '"name": "google_scholar"' not in prompt

    def test_preamble_prepended(self) -> None:
        pipe = Pipe()
        pipe.valves.SYSTEM_PROMPT_PREAMBLE = "Always cite sources."
        prompt = pipe._build_system_prompt()
        assert prompt.startswith("Always cite sources.\n\n")

    def test_empty_preamble_not_prepended(self) -> None:
        pipe = Pipe()
        pipe.valves.SYSTEM_PROMPT_PREAMBLE = ""
        prompt = pipe._build_system_prompt()
        assert prompt.startswith("You are a deep research assistant.")


# ------------------------------------------------------------------ #
#  Overlay model ID helpers
# ------------------------------------------------------------------ #


class TestOverlayModelIdHelpers:
    """Tests for model metadata overlay helper methods."""

    def test_extract_model_id_from_obj_dict(self) -> None:
        assert (
            Pipe._extract_model_id_from_obj(
                {"id": "  tongyi_deepresearch_pipe.tongyi_deepresearch  "}
            )
            == "tongyi_deepresearch_pipe.tongyi_deepresearch"
        )
        assert (
            Pipe._extract_model_id_from_obj(
                {"model_id": "tongyi_deepresearch.tongyi_deepresearch"}
            )
            == "tongyi_deepresearch.tongyi_deepresearch"
        )

    def test_extract_model_id_from_obj_attributes(self) -> None:
        class _ModelObj:
            def __init__(self, *, model_id: str | None = None) -> None:
                self.model_id = model_id

        obj = _ModelObj(model_id="  tongyi_deepresearch.tongyi_deepresearch  ")
        assert (
            Pipe._extract_model_id_from_obj(obj)
            == "tongyi_deepresearch.tongyi_deepresearch"
        )

    def test_looks_like_our_pipe_id(self) -> None:
        assert Pipe._looks_like_our_pipe_id("tongyi_deepresearch.tongyi_deepresearch")
        assert Pipe._looks_like_our_pipe_id(
            "tongyi_deepresearch_pipe.tongyi_deepresearch"
        )
        assert not Pipe._looks_like_our_pipe_id("openai.gpt-4o")
        assert not Pipe._looks_like_our_pipe_id("tongyi_deepresearch")

    def test_extract_live_overlay_candidates_for_dict(self) -> None:
        model_map = {
            "tongyi_deepresearch.tongyi_deepresearch": {"id": "m1"},
            123: {"id": "m2"},
        }
        ids, models = Pipe._extract_live_overlay_candidates(model_map)
        assert ids == ["tongyi_deepresearch.tongyi_deepresearch"]
        assert models == [{"id": "m1"}, {"id": "m2"}]

    def test_extract_live_overlay_candidates_for_iterables(self) -> None:
        ids, models = Pipe._extract_live_overlay_candidates(("a", "b"))
        assert ids == []
        assert models == ["a", "b"]

        class _Iterable:
            def __iter__(self):
                return iter(["x", "y"])

        ids2, models2 = Pipe._extract_live_overlay_candidates(_Iterable())
        assert ids2 == []
        assert models2 == ["x", "y"]

    def test_extract_live_overlay_candidates_for_non_iterable(self) -> None:
        class _NonIterable:
            pass

        ids, models = Pipe._extract_live_overlay_candidates(_NonIterable())
        assert ids == []
        assert models == []

    def test_resolve_overlay_model_ids_includes_fallbacks(self) -> None:
        resolved = Pipe._resolve_overlay_model_ids()
        assert "tongyi_deepresearch.tongyi_deepresearch" in resolved
        assert "tongyi_deepresearch_pipe.tongyi_deepresearch" in resolved
        assert len(resolved) == len(set(resolved))


# ------------------------------------------------------------------ #
#  _CostTracker
# ------------------------------------------------------------------ #


class TestCostTracker:
    """Tests for _CostTracker."""

    def test_empty_summary(self) -> None:
        t = _CostTracker()
        assert t.summary() == "0 in / 0 out"

    def test_update_accumulates(self) -> None:
        t = _CostTracker()
        t.update({"prompt_tokens": 100, "completion_tokens": 50})
        t.update({"prompt_tokens": 200, "completion_tokens": 80})
        assert t.input_tokens == 300
        assert t.output_tokens == 130
        assert t.calls == 2

    def test_summary_with_tokens(self) -> None:
        t = _CostTracker()
        t.update({"prompt_tokens": 1000, "completion_tokens": 500})
        assert t.summary() == "1,000 in / 500 out"

    def test_summary_with_cost(self) -> None:
        t = _CostTracker()
        t.update({"prompt_tokens": 1000, "completion_tokens": 500, "cost": 0.05})
        assert "1,000 in / 500 out" in t.summary()
        assert "$0.05" in t.summary()

    def test_summary_small_cost_precision(self) -> None:
        t = _CostTracker()
        t.update({"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001})
        assert "$0.0010" in t.summary()

    def test_summary_prefix(self) -> None:
        t = _CostTracker()
        t.update({"prompt_tokens": 100, "completion_tokens": 50})
        assert t.summary("Done · ").startswith("Done · ")

    def test_update_none_usage(self) -> None:
        t = _CostTracker()
        t.update(None)
        assert t.calls == 0


# ------------------------------------------------------------------ #
#  _fmt_visit_fallback
# ------------------------------------------------------------------ #


class TestFmtVisitFallback:
    """Tests for module-level _fmt_visit_fallback."""

    def test_format(self) -> None:
        result = _fmt_visit_fallback(
            "https://example.com", "find info", "404 Not Found"
        )
        assert "https://example.com" in result
        assert "find info" in result
        assert "could not be accessed" in result

    def test_does_not_leak_error(self) -> None:
        result = _fmt_visit_fallback("http://x.com", "goal", "secret error")
        assert "secret error" not in result
