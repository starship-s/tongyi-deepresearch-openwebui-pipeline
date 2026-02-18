"""
id: tongyi_deepresearch_pipe
title: Tongyi DeepResearch
author: starship-s
author_url: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline
version: 0.2.4
license: MIT
description: Agentic deep-research pipe for Open WebUI, powered by Tongyi DeepResearch.
required_open_webui_version: 0.5.0
requirements: httpx
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import time
from collections.abc import (  # noqa: TC003 — runtime needed by Open WebUI
    Awaitable,
    Callable,
)
from datetime import date
from typing import ClassVar
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =========================================================================== #
#  Constants                                                                   #
# =========================================================================== #

HTTP_OK = 200
STATUS_PING_INTERVAL_S = 4
COST_DISPLAY_THRESHOLD = 0.01
THINK_OPEN_TAG = "<think>\n"
THINK_CLOSE_TAG = "\n</think>\n"

# =========================================================================== #
#  Tool definitions (injected into the system prompt as JSON)                  #
# =========================================================================== #

_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Perform web searches then returns"
            " a string of the top search results."
            " Accepts multiple queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "minItems": 1,
                    "description": ("The list of search queries."),
                }
            },
            "required": ["query"],
        },
    },
}

_VISIT_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "visit",
        "description": ("Visit webpage(s) and return the summary of the content."),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The URL(s) of the webpage(s) to"
                        " visit. Can be a single URL or an"
                        " array of URLs."
                    ),
                },
                "goal": {
                    "type": "string",
                    "description": (
                        "The specific information goal for visiting webpage(s)."
                    ),
                },
            },
            "required": ["url", "goal"],
        },
    },
}

_SCHOLAR_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "google_scholar",
        "description": (
            "Leverage Google Scholar to retrieve relevant"
            " information from academic publications."
            " Accepts multiple queries. This tool will"
            " also return results from google search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "minItems": 1,
                    "description": ("The list of search queries for Google Scholar."),
                }
            },
            "required": ["query"],
        },
    },
}

_TOOL_DEFINITIONS = [
    _SEARCH_TOOL_DEF,
    _VISIT_TOOL_DEF,
    _SCHOLAR_TOOL_DEF,
]

# =========================================================================== #
#  System prompt template                                                      #
# =========================================================================== #

_SYSTEM_PROMPT_TEMPLATE = """\
You are a deep research assistant. Today's date is \
{human_date} ({iso_date}). Your core function is to \
conduct thorough, multi-source investigations into \
any topic. You must handle both broad, open-domain \
inquiries and queries within specialized academic \
fields. For every request, synthesize information \
from credible, diverse sources to deliver a \
comprehensive, accurate, and objective response. \
When you have gathered sufficient information and \
are ready to provide the definitive response, you \
must enclose the entire final answer within \
<answer></answer> tags.

# Tools

You may call one or more functions to assist with \
the user query.

You are provided with function signatures within \
XML tags:
<tools>
{tool_definitions}
</tools>

For each function call, return a json object with \
function name and arguments within XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


# =========================================================================== #
#  Cost tracker                                                                #
# =========================================================================== #


class _CostTracker:
    """Accumulates token counts and cost across model calls."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.calls = 0

    def update(self, usage: dict | None) -> None:
        """Record usage from one model call."""
        if not usage:
            return
        self.calls += 1
        self.input_tokens += usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("completion_tokens", 0)
        if "cost" in usage:
            self.cost += usage["cost"]

    def summary(self, prefix: str = "") -> str:
        """Format a human-readable cost summary string."""
        tok = f"{self.input_tokens + self.output_tokens:,} tokens"
        if self.cost > 0:
            cost = (
                f"${self.cost:.4f}"
                if self.cost < COST_DISPLAY_THRESHOLD
                else f"${self.cost:.2f}"
            )
            return f"{prefix}{tok} · {cost}"
        return f"{prefix}{tok}"


# =========================================================================== #
#  Pipe                                                                        #
# =========================================================================== #


class Pipe:
    """Open WebUI Pipe implementing the Tongyi DeepResearch agentic loop.

    The model emits XML ``<tool_call>`` blocks.  This pipe intercepts
    them, translates batched search queries into sequential
    single-query calls using Open WebUI's built-in web search engine
    (whatever is configured in Admin -> Settings -> Web Search),
    fetches URLs via the built-in content loader, and feeds results
    back wrapped in ``<tool_response>`` tags until the model produces
    a ``<answer>`` block.
    """

    # ------------------------------------------------------------------ #
    #  Valves (user-configurable settings shown in Admin UI)
    # ------------------------------------------------------------------ #

    class Valves(BaseModel):
        """User-configurable settings shown in the Admin UI."""

        API_KEY: str = Field(
            default="",
            description="API key for the OpenAI-compatible endpoint",
        )
        API_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="OpenAI-compatible API base URL",
        )
        MODEL_ID: str = Field(
            default="alibaba/tongyi-deepresearch-30b-a3b",
            description="Model identifier on the API provider",
        )
        MAX_TOOL_ROUNDS: int = Field(
            default=30,
            ge=1,
            le=200,
            description=("Maximum number of agentic tool-calling rounds"),
        )
        SEARCH_RESULTS_PER_QUERY: int = Field(
            default=5,
            ge=1,
            le=20,
            description=("Number of search results returned per individual query"),
        )
        MAX_QUERIES_PER_SEARCH: int = Field(
            default=5,
            ge=1,
            le=10,
            description=(
                "Cap on how many queries are executed from a single search tool call"
            ),
        )
        MAX_PAGE_LENGTH: int = Field(
            default=50000,
            ge=5000,
            description=("Maximum characters kept from a fetched page"),
        )
        SEARCH_ENABLED: bool = Field(
            default=True,
            description=("Enable the search tool and include it in the system prompt."),
        )
        SCHOLAR_ENABLED: bool = Field(
            default=True,
            description=(
                "Enable the google_scholar tool and include it in the system prompt."
            ),
        )
        VISIT_ENABLED: bool = Field(
            default=True,
            description=("Enable the visit tool and include it in the system prompt."),
        )
        AUTO_INSTALL_TOOLS: bool = Field(
            default=True,
            description=(
                "Auto-install enabled tool modules into"
                " Open WebUI's tool registry on startup."
            ),
        )
        TEMPERATURE: float = Field(default=0.6, ge=0.0, le=2.0)
        TOP_P: float = Field(default=0.95, ge=0.0, le=1.0)
        PRESENCE_PENALTY: float = Field(default=1.1, ge=0.0, le=2.0)
        MAX_TOKENS: int = Field(
            default=16000,
            ge=1024,
            description="Max tokens per model generation call",
        )
        MAX_CONTEXT_CHARS: int = Field(
            default=400000,
            ge=50000,
            description=(
                "Approximate character budget for the whole"
                " conversation before the model is forced"
                " to wrap up (~100K tokens)"
            ),
        )
        SYSTEM_PROMPT_PREAMBLE: str = Field(
            default="",
            description=(
                "Optional preamble prepended to the built-in"
                " system prompt. Use this for custom"
                " instructions (e.g. citation style, tone)."
                " Leave empty to use only the built-in"
                " prompt."
            ),
        )
        EMIT_THINKING: bool = Field(
            default=True,
            description=(
                "Show model thinking as collapsible blocks"
                " interleaved with tool-call cards in the"
                " chat message"
            ),
        )
        SHOW_COST_TRACKING: bool = Field(
            default=True,
            description=(
                "Display running token count and cost"
                " estimate in the status bar after each"
                " model call"
            ),
        )

    # ------------------------------------------------------------------ #
    #  Init / metadata
    # ------------------------------------------------------------------ #

    def __init__(self) -> None:
        """Initialise valves with defaults."""
        self.valves = self.Valves()
        self._request: object | None = None
        self._user: object | None = None
        self._event_emitter: Callable[[dict], Awaitable[None]] | None = None

    def pipes(self) -> list:
        """Return the list of available pipe definitions."""
        if self.valves.AUTO_INSTALL_TOOLS:
            self._auto_install_tools()
        return [
            {
                "id": "tongyi_deepresearch",
                "name": "Tongyi DeepResearch",
                "description": (
                    "Agentic deep-research assistant"
                    " powered by Alibaba Tongyi"
                    " DeepResearch. Conducts multi-step"
                    " web searches and page visits to"
                    " deliver thorough, cited answers."
                ),
                "profile_image_url": (
                    "https://cdn-avatars.huggingface.co"
                    "/v1/production/uploads"
                    "/63fc4c00a3c067e62899d32b"
                    "/dfd_EcIfylvu3sdc2WMqX.png"
                ),
            }
        ]

    # ------------------------------------------------------------------ #
    #  Tool auto-install
    # ------------------------------------------------------------------ #

    _TOOL_REGISTRY: ClassVar[list[dict]] = [
        {
            "id": "deepresearch_search_tool",
            "name": "DeepResearch Search Tool",
            "module": "search_tool.py",
            "description": (
                "Searches the web via Open WebUI's"
                " built-in search engine, formatted to"
                " match DeepResearch training output."
            ),
            "valve": "SEARCH_ENABLED",
            "specs": [_SEARCH_TOOL_DEF],
        },
        {
            "id": "deepresearch_scholar_tool",
            "name": "DeepResearch Scholar Tool",
            "module": "scholar_tool.py",
            "description": (
                "Searches academic literature via Open"
                " WebUI's built-in search with an"
                " 'academic research' prefix, formatted"
                " to match DeepResearch training output."
            ),
            "valve": "SCHOLAR_ENABLED",
            "specs": [_SCHOLAR_TOOL_DEF],
        },
        {
            "id": "deepresearch_visit_tool",
            "name": "DeepResearch Visit Tool",
            "module": "visit_tool.py",
            "description": (
                "Visits URLs and extracts structured"
                " evidence via a dedicated extractor"
                " LLM call."
            ),
            "valve": "VISIT_ENABLED",
            "specs": [_VISIT_TOOL_DEF],
        },
    ]

    def _auto_install_tools(self) -> None:
        """Auto-install enabled tool modules into Open WebUI's tool DB."""
        for entry in self._TOOL_REGISTRY:
            if not getattr(self.valves, entry["valve"], False):
                continue
            try:
                self._ensure_tool_installed(entry)
            except Exception:
                logger.debug(
                    "Auto-install failed for %s",
                    entry["id"],
                    exc_info=True,
                )

    def _ensure_tool_installed(self, entry: dict) -> None:
        """Check if a tool exists in the DB; create/update if needed."""
        try:
            from open_webui.models.tools import (  # type: ignore[import-not-found]  # noqa: PLC0415
                ToolForm,
                ToolMeta,
            )
            from open_webui.models.tools import (  # type: ignore[import-not-found]  # noqa: PLC0415
                Tools as ToolsDB,
            )
        except ImportError:
            return

        tool_id: str = entry["id"]
        existing = ToolsDB.get_tool_by_id(tool_id)

        source = self._read_tool_source(entry["module"])
        if not source:
            return

        meta = ToolMeta(description=entry["description"])

        if existing is None:
            form = ToolForm(
                id=tool_id,
                name=entry["name"],
                content=source,
                meta=meta,
            )
            ToolsDB.insert_new_tool(
                user_id="",
                form_data=form,
                specs=entry["specs"],
            )
            logger.info("Auto-installed tool: %s", tool_id)
        elif existing.content != source:
            ToolsDB.update_tool_by_id(
                tool_id,
                {
                    "content": source,
                    "name": entry["name"],
                    "meta": meta.model_dump(),
                },
            )
            logger.info("Auto-updated tool: %s", tool_id)

    @staticmethod
    def _read_tool_source(module_name: str) -> str | None:
        """Read tool source code from the installed package."""
        try:
            from importlib.resources import (  # noqa: PLC0415
                files as _files,
            )

            resource = _files("tongyi_deepresearch_openwebui_pipeline.tools").joinpath(
                module_name
            )
            return resource.read_text(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            logger.debug(
                "Could not read tool source for %s",
                module_name,
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------ #
    #  System-prompt builder
    # ------------------------------------------------------------------ #

    def _build_system_prompt(self) -> str:
        today = date.today()  # noqa: DTZ011
        tool_defs: list[dict] = []
        if self.valves.SEARCH_ENABLED:
            tool_defs.append(_SEARCH_TOOL_DEF)
        if self.valves.VISIT_ENABLED:
            tool_defs.append(_VISIT_TOOL_DEF)
        if self.valves.SCHOLAR_ENABLED:
            tool_defs.append(_SCHOLAR_TOOL_DEF)
        if not tool_defs:
            tool_defs = list(_TOOL_DEFINITIONS)
        tools_json = "\n".join(json.dumps(t, ensure_ascii=False) for t in tool_defs)
        prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            human_date=today.strftime("%A, %B %d, %Y"),
            iso_date=today.isoformat(),
            tool_definitions=tools_json,
        )
        preamble = self.valves.SYSTEM_PROMPT_PREAMBLE.strip()
        if preamble:
            return preamble + "\n\n" + prompt
        return prompt

    # ================================================================== #
    #  Event emitters                                                      #
    # ================================================================== #

    async def _emit_status(self, desc: str, done: bool = False) -> None:
        if self._event_emitter:
            await self._event_emitter(
                {
                    "type": "status",
                    "data": {
                        "description": desc,
                        "done": done,
                    },
                }
            )

    async def _emit_message(self, text: str) -> None:
        if self._event_emitter:
            await self._event_emitter(
                {
                    "type": "message",
                    "data": {"content": text},
                }
            )

    # ================================================================== #
    #  LLM streaming client                                                #
    # ================================================================== #

    async def _call_llm(
        self,
        messages: list,
        max_retries: int = 3,
    ) -> tuple:
        """Stream a chat completion from the configured API endpoint.

        Returns:
            Tuple of (reasoning_text, content_text,
            usage_dict). *usage_dict* may be ``None``
            if the provider did not report usage.
        """
        url = f"{self.valves.API_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": (f"Bearer {self.valves.API_KEY}"),
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openwebui.com",
            "X-Title": "Open WebUI - DeepResearch",
        }
        payload = {
            "model": self.valves.MODEL_ID,
            "messages": messages,
            "stream": True,
            "temperature": self.valves.TEMPERATURE,
            "top_p": self.valves.TOP_P,
            "presence_penalty": self.valves.PRESENCE_PENALTY,
            "max_tokens": self.valves.MAX_TOKENS,
        }

        last_exc: Exception | None = None

        for attempt in range(max_retries):
            try:
                return await self._stream_completion(url, headers, payload)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait = 2**attempt
                    await self._emit_status(
                        f"API error, retrying in {wait}s\u2026 ({exc})"
                    )
                    await asyncio.sleep(wait)

        raise last_exc  # type: ignore[misc]

    async def _emit_reasoning_delta(
        self,
        reasoning: str,
        prev_len: int,
        content: str,
        state: dict,
    ) -> None:
        """Emit new reasoning tokens wrapped in ``<think>`` tags.

        *state* tracks ``open`` and ``closed`` booleans across calls.
        """
        new_text = reasoning[prev_len:]
        if new_text:
            if not state["open"]:
                await self._emit_message(THINK_OPEN_TAG)
                state["open"] = True
            await self._emit_message(new_text)

        if state["open"] and not state["closed"] and content:
            await self._emit_message(THINK_CLOSE_TAG)
            state["closed"] = True

    async def _stream_completion(
        self,
        url: str,
        headers: dict,
        payload: dict,
    ) -> tuple:
        """Execute the SSE stream and collect the response.

        Reasoning tokens are emitted in real-time via ``_emit_message``
        wrapped in ``<think>`` tags so they interleave correctly with
        the tool-call cards that follow each model turn.  Content
        tokens (which may contain ``<tool_call>`` or ``<answer>`` XML)
        are collected silently for post-processing.
        """
        reasoning = ""
        content = ""
        usage: dict | None = None
        last_ping = time.time()
        emit_thinking = self.valves.EMIT_THINKING
        think_state: dict[str, bool] = {"open": False, "closed": False}

        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client,
            client.stream("POST", url, json=payload, headers=headers) as resp,
        ):
            if resp.status_code != HTTP_OK:
                body = await resp.aread()
                raise RuntimeError(
                    f"API error {resp.status_code}: "
                    f"{body.decode(errors='replace')[:500]}"
                )

            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                prev_reasoning_len = len(reasoning)
                reasoning, content, usage = self._process_sse_chunk(
                    chunk, reasoning, content, usage
                )

                if emit_thinking:
                    await self._emit_reasoning_delta(
                        reasoning, prev_reasoning_len, content, think_state
                    )

                if time.time() - last_ping > STATUS_PING_INTERVAL_S:
                    n = len(reasoning) + len(content)
                    await self._emit_status(f"Generating\u2026 ({n:,} chars received)")
                    last_ping = time.time()

        if emit_thinking and think_state["open"] and not think_state["closed"]:
            await self._emit_message(THINK_CLOSE_TAG)

        return reasoning, content, usage

    @staticmethod
    def _process_sse_chunk(
        chunk: dict,
        reasoning: str,
        content: str,
        usage: dict | None,
    ) -> tuple[str, str, dict | None]:
        """Extract reasoning, content and usage from one SSE chunk."""
        chunk_usage = chunk.get("usage")
        if chunk_usage:
            usage = chunk_usage

        choices = chunk.get("choices")
        if not choices:
            return reasoning, content, usage
        delta = choices[0].get("delta", {})

        for rkey in ("reasoning", "reasoning_content"):
            val = delta.get(rkey)
            if val:
                reasoning += val

        cval = delta.get("content")
        if cval:
            content += cval

        return reasoning, content, usage

    # ================================================================== #
    #  XML / JSON parsing helpers                                          #
    # ================================================================== #

    @staticmethod
    def _extract_tool_call(text: str) -> dict | None:
        """Return the first parsed ``<tool_call>`` dict, or *None*."""
        m = re.search(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            text,
            re.DOTALL,
        )
        if not m:
            return None

        raw = m.group(1).strip()

        if "PythonInterpreter" in raw:
            code_m = re.search(r"<code>(.*?)</code>", raw, re.DOTALL)
            code = code_m.group(1).strip() if code_m else ""
            return {
                "name": "PythonInterpreter",
                "arguments": {"code": code},
            }

        for parser in (
            json.loads,
            lambda s: json.loads(
                re.sub(
                    r",\s*([}\]])",
                    r"\1",
                    s.replace("'", '"'),
                )
            ),
        ):
            try:
                obj = parser(raw)
                if isinstance(obj, dict) and "name" in obj:
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue

        return None

    @staticmethod
    def _extract_answer(text: str) -> str | None:
        """Extract content inside ``<answer>`` tags (handles unclosed tag)."""
        m = re.search(
            r"<answer>(.*?)(?:</answer>|$)",
            text,
            re.DOTALL,
        )
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_thinking(text: str) -> str | None:
        """Extract content inside ``<think>`` tags."""
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _strip_xml_for_display(text: str) -> str:
        """Remove helper XML tags so the user sees clean prose."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(
            r"<tool_call>.*?</tool_call>",
            "",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(r"</?answer>", "", text)
        return text.strip()

    @staticmethod
    def _reconstruct_full_turn(reasoning: str, content: str) -> str:
        """Merge reasoning and content into the full assistant turn."""
        if reasoning and "<think>" not in content:
            return f"<think>\n{reasoning}\n</think>\n{content}"
        return content

    # ================================================================== #
    #  Tool router                                                         #
    # ================================================================== #

    def _enabled_tool_names(self) -> list[str]:
        """Return the names of currently enabled tools."""
        names: list[str] = []
        if self.valves.SEARCH_ENABLED:
            names.append("search")
        if self.valves.VISIT_ENABLED:
            names.append("visit")
        if self.valves.SCHOLAR_ENABLED:
            names.append("google_scholar")
        return names

    _UNAVAILABLE_TOOLS: ClassVar[dict[str, str]] = {
        "PythonInterpreter": (
            "[PythonInterpreter] Code execution is not"
            " available in this environment. Please"
            " reason through the computation manually"
            " or reformulate your approach using"
            " search."
        ),
        "parse_file": (
            "[parse_file] File parsing is not available in this environment."
        ),
    }

    async def _execute_tool(
        self,
        name: str,
        arguments: dict,
    ) -> str:
        """Route a parsed tool call to the appropriate handler."""
        if name in self._UNAVAILABLE_TOOLS:
            return self._UNAVAILABLE_TOOLS[name]

        enabled = self._enabled_tool_names()
        if name in {"search", "visit", "google_scholar"} and name not in enabled:
            return (
                f"[{name}] This tool is not enabled."
                f" Available tools: {', '.join(enabled)}"
            )

        if name == "search":
            queries = arguments.get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            return await self._execute_search(queries)

        if name == "visit":
            urls = arguments.get("url", [])
            if isinstance(urls, str):
                urls = [urls]
            goal = arguments.get("goal", "Extract relevant information")
            return await self._execute_visit(urls, goal)

        if name == "google_scholar":
            queries = arguments.get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            return await self._execute_scholar(queries)

        return f"[Error] Unknown tool: {name}"

    # ---- visit tool -------------------------------------------------- #

    async def _execute_visit(self, urls: list[str], goal: str) -> str:
        """Execute the visit tool using the configured backend."""
        if self.valves.VISIT_ENABLED and self.valves.API_KEY:
            return await self._execute_visit_with_tool(urls, goal)
        return await self._execute_visit_builtin(urls, goal)

    @staticmethod
    def _resolve_visit_tools_class() -> type | None:
        """Locate the visit-tool ``Tools`` class across install methods.

        Tries four strategies in order:
        1. Package import (pip-installed).
        2. Direct module import (``visit_tool.py`` on ``sys.path``).
        3. Scan ``sys.modules`` for an Open WebUI-loaded tool module
           (stored as ``tool_{id}``).
        4. Load from the Open WebUI database via
           ``load_tool_module_by_id``.
        """
        import sys as _sys  # noqa: PLC0415

        try:
            from tongyi_deepresearch_openwebui_pipeline.tools.visit_tool import (  # noqa: PLC0415
                Tools,
            )

            return Tools  # type: ignore[no-any-return]
        except ImportError:
            pass

        try:
            from visit_tool import (  # type: ignore[import-not-found]  # noqa: PLC0415
                Tools,
            )

            return Tools  # type: ignore[no-any-return]
        except ImportError:
            pass

        for _name, mod in _sys.modules.items():
            if not _name.startswith("tool_"):
                continue
            cls = getattr(mod, "Tools", None)
            if cls is not None and callable(getattr(cls, "visit", None)):
                return cls  # type: ignore[no-any-return]

        try:
            from open_webui.models.tools import (  # type: ignore[import-not-found]  # noqa: PLC0415
                Tools as ToolsDB,
            )
            from open_webui.utils.plugin import (  # type: ignore[import-not-found]  # noqa: PLC0415
                load_tool_module_by_id,
            )

            for tool in ToolsDB.get_tools():
                if "class Tools" in tool.content and "def visit" in tool.content:
                    instance, _ = load_tool_module_by_id(tool.id)
                    return type(instance)
        except Exception:  # noqa: S110
            pass

        return None

    async def _execute_visit_with_tool(self, urls: list[str], goal: str) -> str:
        """Visit URLs using the standalone visit_tool module."""
        VisitTools = self._resolve_visit_tools_class()  # noqa: N806
        if VisitTools is None:
            return (
                "[visit] visit_tool module not found —"
                " disable VISIT_ENABLED or install"
                " visit_tool.py."
            )

        visit_tools = VisitTools()
        visit_tools.valves.SUMMARY_MODEL_API_KEY = self.valves.API_KEY
        visit_tools.valves.SUMMARY_MODEL_BASE_URL = self.valves.API_BASE_URL
        visit_tools.valves.MAX_PAGE_TOKENS = self.valves.MAX_PAGE_LENGTH

        pipe_self = self

        async def _visit_emitter(event: dict) -> None:
            d = event.get("data", {})
            await pipe_self._emit_status(
                d.get("description", ""),
                d.get("done", False),
            )

        return await visit_tools.visit(urls, goal, _visit_emitter)

    async def _execute_visit_builtin(self, urls: list[str], goal: str) -> str:
        """Visit URLs using Open WebUI's built-in content loader."""
        from open_webui.retrieval.utils import (  # type: ignore[import-not-found]  # noqa: PLC0415
            get_content_from_url,
        )

        if len(urls) == 1:
            await self._emit_status(f"Visiting: {urls[0]}")
        else:
            await self._emit_status(f"Visiting {len(urls)} pages concurrently\u2026")

        max_len = self.valves.MAX_PAGE_LENGTH

        async def _fetch_one(u: str) -> str:
            try:
                content, _title = await asyncio.to_thread(
                    get_content_from_url,
                    self._request,
                    u,
                )
                if not content or not content.strip():
                    return _fmt_visit_fallback(u, goal, "Empty page content")
                if len(content) > max_len:
                    content = content[:max_len] + "\n\u2026[content truncated]"
                return (
                    f"The useful information in {u} for"
                    f" user goal {goal} as follows: \n\n"
                    f"Evidence in page: \n{content}\n\n"
                    "Summary: \nRaw page content provided"
                    " above for analysis.\n\n"
                )
            except Exception as exc:
                return _fmt_visit_fallback(u, goal, str(exc))

        if len(urls) == 1:
            return await _fetch_one(urls[0])
        results = await asyncio.gather(*(_fetch_one(u) for u in urls))
        return "\n=======\n".join(results)

    # ================================================================== #
    #  Search tool                                                         #
    # ================================================================== #

    @staticmethod
    def _resolve_search_tools_class() -> type | None:
        """Locate the search-tool ``Tools`` class across install methods.

        Tries four strategies in order:
        1. Package import (pip-installed).
        2. Direct module import (``search_tool.py`` on ``sys.path``).
        3. Scan ``sys.modules`` for an Open WebUI-loaded tool module
           (stored as ``tool_{id}``).
        4. Load from the Open WebUI database via
           ``load_tool_module_by_id``.
        """
        import sys as _sys  # noqa: PLC0415

        try:
            from tongyi_deepresearch_openwebui_pipeline.tools.search_tool import (  # noqa: PLC0415
                Tools,
            )

            return Tools  # type: ignore[no-any-return]
        except ImportError:
            pass

        try:
            from search_tool import (  # type: ignore[import-not-found]  # noqa: PLC0415
                Tools,
            )

            return Tools  # type: ignore[no-any-return]
        except ImportError:
            pass

        for _name, mod in _sys.modules.items():
            if not _name.startswith("tool_"):
                continue
            cls = getattr(mod, "Tools", None)
            if cls is not None and callable(getattr(cls, "search", None)):
                return cls  # type: ignore[no-any-return]

        try:
            from open_webui.models.tools import (  # type: ignore[import-not-found]  # noqa: PLC0415
                Tools as ToolsDB,
            )
            from open_webui.utils.plugin import (  # type: ignore[import-not-found]  # noqa: PLC0415
                load_tool_module_by_id,
            )

            for tool in ToolsDB.get_tools():
                if "class Tools" in tool.content and "def search" in tool.content:
                    instance, _ = load_tool_module_by_id(tool.id)
                    return type(instance)
        except Exception:  # noqa: S110
            pass

        return None

    async def _execute_search(
        self,
        queries: list[str],
    ) -> str:
        """Execute search via the standalone search_tool module."""
        SearchTools = self._resolve_search_tools_class()  # noqa: N806
        if SearchTools is None:
            return "[search] search_tool module not found — install search_tool.py."

        search_tools = SearchTools()
        search_tools.request = self._request
        search_tools.user = self._user
        search_tools.valves.MAX_RESULTS_PER_QUERY = self.valves.SEARCH_RESULTS_PER_QUERY
        search_tools.valves.MAX_QUERIES_PER_SEARCH = self.valves.MAX_QUERIES_PER_SEARCH

        pipe_self = self

        async def _search_emitter(event: dict) -> None:
            d = event.get("data", {})
            await pipe_self._emit_status(
                d.get("description", ""),
                d.get("done", False),
            )

        return await search_tools.search(queries, _search_emitter)

    # ================================================================== #
    #  Scholar tool                                                        #
    # ================================================================== #

    @staticmethod
    def _resolve_scholar_tools_class() -> type | None:
        """Locate the scholar-tool ``Tools`` class across install methods.

        Tries four strategies in order:
        1. Package import (pip-installed).
        2. Direct module import (``scholar_tool.py`` on ``sys.path``).
        3. Scan ``sys.modules`` for an Open WebUI-loaded tool module
           (stored as ``tool_{id}``).
        4. Load from the Open WebUI database via
           ``load_tool_module_by_id``.
        """
        import sys as _sys  # noqa: PLC0415

        try:
            from tongyi_deepresearch_openwebui_pipeline.tools.scholar_tool import (  # noqa: PLC0415
                Tools,
            )

            return Tools  # type: ignore[no-any-return]
        except ImportError:
            pass

        try:
            from scholar_tool import (  # type: ignore[import-not-found]  # noqa: PLC0415
                Tools,
            )

            return Tools  # type: ignore[no-any-return]
        except ImportError:
            pass

        for _name, mod in _sys.modules.items():
            if not _name.startswith("tool_"):
                continue
            cls = getattr(mod, "Tools", None)
            if cls is not None and callable(getattr(cls, "google_scholar", None)):
                return cls  # type: ignore[no-any-return]

        try:
            from open_webui.models.tools import (  # type: ignore[import-not-found]  # noqa: PLC0415
                Tools as ToolsDB,
            )
            from open_webui.utils.plugin import (  # type: ignore[import-not-found]  # noqa: PLC0415
                load_tool_module_by_id,
            )

            for tool in ToolsDB.get_tools():
                if (
                    "class Tools" in tool.content
                    and "def google_scholar" in tool.content
                ):
                    instance, _ = load_tool_module_by_id(tool.id)
                    return type(instance)
        except Exception:  # noqa: S110
            pass

        return None

    async def _execute_scholar(
        self,
        queries: list[str],
    ) -> str:
        """Execute Google Scholar search via the standalone scholar_tool module."""
        ScholarTools = self._resolve_scholar_tools_class()  # noqa: N806
        if ScholarTools is None:
            return (
                "[google_scholar] scholar_tool module not found"
                " — install scholar_tool.py."
            )

        scholar_tools = ScholarTools()
        scholar_tools.request = self._request
        scholar_tools.user = self._user
        scholar_tools.valves.MAX_RESULTS_PER_QUERY = (
            self.valves.SEARCH_RESULTS_PER_QUERY
        )
        scholar_tools.valves.MAX_QUERIES_PER_SEARCH = self.valves.MAX_QUERIES_PER_SEARCH

        pipe_self = self

        async def _scholar_emitter(event: dict) -> None:
            d = event.get("data", {})
            await pipe_self._emit_status(
                d.get("description", ""),
                d.get("done", False),
            )

        return await scholar_tools.google_scholar(queries, _scholar_emitter)

    # ---- tool-call display card -------------------------------------- #

    @staticmethod
    def _build_tool_call_card(
        tool_name: str,
        tool_args: dict,
        result: str,
        *,
        done: bool = True,
        max_result_display: int = 4000,
    ) -> str:
        """Build an HTML ``<details>`` card for a tool call.

        Matches Open WebUI's native tool-call card format so
        the frontend renders a collapsible card with the tool
        name, arguments, and result.
        """
        call_id = f"tc_{uuid4().hex[:24]}"
        escaped_args = html.escape(json.dumps(tool_args, ensure_ascii=False))

        if done:
            display_result = result
            if len(display_result) > max_result_display:
                display_result = (
                    display_result[:max_result_display]
                    + "\n\u2026[truncated for display]"
                )
            escaped_result = html.escape(json.dumps(display_result, ensure_ascii=False))
            return (
                '<details type="tool_calls" done="true" '
                f'id="{call_id}"'
                f' name="{html.escape(tool_name)}" '
                f'arguments="{escaped_args}" '
                f'result="{escaped_result}">\n'
                "<summary>Tool Executed</summary>\n"
                "</details>\n"
            )

        return (
            '<details type="tool_calls" done="false" '
            f'id="{call_id}"'
            f' name="{html.escape(tool_name)}" '
            f'arguments="{escaped_args}">\n'
            "<summary>Executing\u2026</summary>\n"
            "</details>\n"
        )

    # ================================================================== #
    #  Main agentic loop                                                   #
    # ================================================================== #

    async def pipe(
        self,
        body: dict,
        __user__: dict | None = None,
        __event_emitter__: (Callable[[dict], Awaitable[None]] | None) = None,
        __request__: object | None = None,
    ) -> str:
        """Entry point called by Open WebUI for every user message."""
        self._store_request_context(__user__, __event_emitter__, __request__)

        error = self._preflight_check()
        if error:
            return error

        messages = self._build_initial_messages(body)
        tracker = _CostTracker()

        return await self._run_agentic_loop(messages, tracker)

    # ---- pipe helpers ------------------------------------------------ #

    def _store_request_context(
        self,
        __user__: dict | None,
        __event_emitter__: (Callable[[dict], Awaitable[None]] | None),
        __request__: object | None,
    ) -> None:
        """Store the Open WebUI request context for tool handlers."""
        self._request = __request__
        self._event_emitter = __event_emitter__
        self._user = None
        if __user__:
            try:
                from open_webui.models.users import (  # type: ignore[import-not-found]  # noqa: PLC0415
                    UserModel,
                )

                self._user = UserModel(**__user__)
            except Exception:
                logger.debug(
                    "Failed to parse __user__",
                    exc_info=True,
                )

    def _preflight_check(self) -> str | None:
        """Return an error message if configuration is invalid."""
        if not self.valves.API_KEY:
            return (
                "**Configuration error:** API key is not"
                " set. Go to *Admin Panel \u2192 Functions"
                " \u2192 Tongyi DeepResearch \u2192 Valves*"
                " to configure it."
            )
        if self._request is None:
            return (
                "**Configuration error:** This pipe"
                " requires access to the Open WebUI request"
                " context (`__request__`). Ensure you are"
                " running a compatible version of Open"
                " WebUI (0.5.0+)."
            )
        return None

    def _build_initial_messages(self, body: dict) -> list[dict]:
        """Build the initial message list from the user's request."""
        messages: list[dict] = [
            {
                "role": "system",
                "content": self._build_system_prompt(),
            }
        ]
        for msg in body.get("messages", []):
            role = msg.get("role", "")
            if role == "system":
                continue
            messages.append(
                {
                    "role": role,
                    "content": msg.get("content", ""),
                }
            )
        return messages

    # ---- agentic loop ------------------------------------------------ #

    async def _run_agentic_loop(
        self,
        messages: list[dict],
        tracker: _CostTracker,
    ) -> str:
        """Run the multi-turn ReAct loop until an answer is produced."""
        for round_num in range(1, self.valves.MAX_TOOL_ROUNDS + 1):
            await self._emit_status(f"Round {round_num} \u2014 calling model\u2026")
            await self._apply_context_guard(messages)

            model_result = await self._call_model_safe(messages, tracker, round_num)
            if isinstance(model_result, str):
                return model_result

            reasoning, _content, full = model_result
            messages.append({"role": "assistant", "content": full})

            if self.valves.EMIT_THINKING and not reasoning:
                thinking = self._extract_thinking(full)
                if thinking:
                    await self._emit_message(
                        THINK_OPEN_TAG + thinking + THINK_CLOSE_TAG
                    )

            action = await self._process_round(full, messages, tracker, round_num)
            if action is not None:
                return action

        return await self._force_final_answer(messages, tracker)

    async def _call_model_safe(
        self,
        messages: list[dict],
        tracker: _CostTracker,
        round_num: int,
    ) -> tuple | str:
        """Call the model, returning (reasoning, content, full) or an error string."""
        try:
            reasoning, content, usage = await self._call_llm(messages)
            tracker.update(usage)
        except Exception as exc:
            await self._emit_status(f"API error: {exc}", done=True)
            return f"**API error:** {exc}"

        if self.valves.SHOW_COST_TRACKING:
            await self._emit_status(f"Round {round_num} \u2014 {tracker.summary()}")

        full = self._reconstruct_full_turn(reasoning, content)
        return reasoning, content, full

    async def _apply_context_guard(self, messages: list[dict]) -> None:
        """Inject a wrap-up request when the context budget is nearly exhausted."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        if total_chars > self.valves.MAX_CONTEXT_CHARS:
            await self._emit_status(
                "Context limit approaching \u2014 requesting final answer\u2026"
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have now reached the maximum"
                        " context length you can handle."
                        " Stop making tool calls and,"
                        " based on all the information"
                        " above, think again and provide"
                        " what you consider the most"
                        " likely answer within"
                        " <answer></answer> tags."
                    ),
                }
            )

    async def _process_round(
        self,
        full: str,
        messages: list[dict],
        tracker: _CostTracker,
        round_num: int,
    ) -> str | None:
        """Process one round of model output.

        Returns a final answer string, or ``None`` to
        continue the loop.
        """
        tc = self._extract_tool_call(full)
        if tc:
            await self._handle_tool_call(tc, messages, round_num)
            return None

        answer = self._extract_answer(full)
        if answer:
            await self._emit_status(
                tracker.summary("Research complete \u00b7 "),
                done=True,
            )
            return answer

        clean = self._strip_xml_for_display(full)
        if clean:
            await self._emit_status(tracker.summary("Done \u00b7 "), done=True)
            return clean

        await self._emit_status(
            f"Round {round_num} \u2014 empty response, retrying\u2026"
        )
        return None

    async def _handle_tool_call(
        self,
        tc: dict,
        messages: list[dict],
        round_num: int,
    ) -> None:
        """Execute a tool call and inject the result into messages."""
        tool_name = tc.get("name", "unknown")
        tool_args = tc.get("arguments", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        await self._emit_status(f"Round {round_num} \u2014 executing {tool_name}\u2026")

        try:
            result = await self._execute_tool(tool_name, tool_args)
        except Exception as exc:
            result = f"[Tool Error] {tool_name} failed: {exc}"

        await self._emit_message(
            self._build_tool_call_card(tool_name, tool_args, result)
        )

        messages.append(
            {
                "role": "user",
                "content": (f"<tool_response>\n{result}\n</tool_response>"),
            }
        )

    async def _force_final_answer(
        self,
        messages: list[dict],
        tracker: _CostTracker,
    ) -> str:
        """Force a final answer after the round limit is exhausted."""
        await self._emit_status(
            "Maximum research rounds reached \u2014 forcing final answer\u2026"
        )

        messages.append(
            {
                "role": "user",
                "content": (
                    "You have used all available research"
                    " rounds. Based on everything gathered"
                    " above, provide your best answer now"
                    " within <answer></answer> tags."
                ),
            }
        )

        try:
            reasoning, content, usage = await self._call_llm(messages)
            tracker.update(usage)
            full = self._reconstruct_full_turn(reasoning, content)
            answer = self._extract_answer(full)
            if answer:
                await self._emit_status(
                    tracker.summary("Research complete \u00b7 "),
                    done=True,
                )
                return answer

            await self._emit_status(
                tracker.summary("Complete \u00b7 "),
                done=True,
            )
            return self._strip_xml_for_display(full) or content

        except Exception as exc:
            await self._emit_status(
                tracker.summary("Error in final round \u00b7 "),
                done=True,
            )
            return f"Research limit reached. Final attempt failed: {exc}"


# =========================================================================== #
#  Module-level helpers                                                        #
# =========================================================================== #


def _fmt_visit_fallback(url: str, goal: str, _error: str) -> str:
    """Format a visit-tool error in the expected convention."""
    return (
        f"The useful information in {url} for user goal "
        f"{goal} as follows: \n\n"
        "Evidence in page: \n"
        "The provided webpage content could not be"
        " accessed. Please check the URL or file"
        " format.\n\n"
        "Summary: \n"
        "The webpage content could not be processed,"
        " and therefore, no information is"
        " available.\n\n"
    )
