"""
title: Tongyi DeepResearch
author: open-webui-community
version: 0.2.0
license: MIT
description: >
    Agentic deep-research pipe that bridges Tongyi DeepResearch
    (alibaba/tongyi-deepresearch-30b-a3b via OpenRouter) with Open WebUI.
    Translates the model's XML-structured <tool_call> blocks into real web
    searches and URL fetches using Open WebUI's built-in web search and
    content loader, then feeds results back until the model produces a
    final <answer>.
required_open_webui_version: 0.4.0
requirements: httpx
"""

import html
import json
import re
import time
import asyncio
from datetime import date
from typing import Optional, List, Callable, Awaitable
from uuid import uuid4

from pydantic import BaseModel, Field

import httpx

# =========================================================================== #
#  System prompt — mirrors the model's training format exactly.               #
#  Tool definitions for search, visit, and google_scholar are included so     #
#  the model stays in its trained distribution.  PythonInterpreter and        #
#  parse_file are omitted because they cannot be executed here; if the model  #
#  still attempts them, the tool router returns a graceful error.             #
# =========================================================================== #

DEEPRESEARCH_SYSTEM_PROMPT_TEMPLATE = """\
You are a deep research assistant. Today's date is {human_date} ({iso_date}). \
Your core function is to conduct thorough, multi-source investigations into \
any topic. You must handle both broad, open-domain inquiries and queries \
within specialized academic fields. For every request, synthesize information \
from credible, diverse sources to deliver a comprehensive, accurate, and \
objective response. When you have gathered sufficient information and are \
ready to provide the definitive response, you must enclose the entire final \
answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within XML tags:
<tools>
{{"type": "function", "function": {{"name": "search", "description": "Perform Kagi web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {{"type": "object", "properties": {{"query": {{"type": "array", "items": {{"type": "string", "description": "The search query."}}, "minItems": 1, "description": "The list of search queries."}}}}, "required": ["query"]}}}}}}
{{"type": "function", "function": {{"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {{"type": "object", "properties": {{"url": {{"type": "array", "items": {{"type": "string"}}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}}, "goal": {{"type": "string", "description": "The specific information goal for visiting webpage(s)."}}}}, "required": ["url", "goal"]}}}}}}
{{"type": "function", "function": {{"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search.", "parameters": {{"type": "object", "properties": {{"query": {{"type": "array", "items": {{"type": "string", "description": "The search query."}}, "minItems": 1, "description": "The list of search queries for Google Scholar."}}}}, "required": ["query"]}}}}}}
</tools>

For each function call, return a json object with function name and arguments within XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


# =========================================================================== #
#  Pipe                                                                       #
# =========================================================================== #


class Pipe:
    """
    Open WebUI Pipe implementing the Tongyi DeepResearch agentic loop.

    The model emits XML ``<tool_call>`` blocks.  This pipe intercepts them,
    translates batched search queries into sequential single-query calls
    using Open WebUI's built-in web search engine (whatever is configured
    in Admin → Settings → Web Search), fetches URLs via the built-in
    content loader, and feeds results back wrapped in ``<tool_response>``
    tags until the model produces a ``<answer>`` block.
    """

    # ------------------------------------------------------------------ #
    #  Valves (user-configurable settings shown in Admin UI)
    # ------------------------------------------------------------------ #

    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = Field(
            default="",
            description="OpenRouter API key",
        )
        OPENROUTER_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="OpenRouter-compatible API base URL",
        )
        MODEL_ID: str = Field(
            default="alibaba/tongyi-deepresearch-30b-a3b",
            description="Model identifier on OpenRouter",
        )
        MAX_TOOL_ROUNDS: int = Field(
            default=30,
            ge=1,
            le=200,
            description="Maximum number of agentic tool-calling rounds",
        )
        SEARCH_RESULTS_PER_QUERY: int = Field(
            default=5,
            ge=1,
            le=20,
            description="Number of search results returned per individual query",
        )
        MAX_QUERIES_PER_SEARCH: int = Field(
            default=5,
            ge=1,
            le=10,
            description="Cap on how many queries are executed from a single search tool call",
        )
        MAX_PAGE_LENGTH: int = Field(
            default=50000,
            ge=5000,
            description="Maximum characters kept from a fetched page",
        )
        VISIT_TOOL_ENABLED: bool = Field(
            default=True,
            description=(
                "Use the standalone visit_tool for richer LLM-based extraction. "
                "When False, falls back to the built-in Open WebUI content loader."
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
                "Approximate character budget for the whole conversation "
                "before the model is forced to wrap up (~100K tokens)"
            ),
        )
        SYSTEM_PROMPT_PREAMBLE: str = Field(
            default="",
            description=(
                "Optional preamble prepended to the built-in system prompt. "
                "Use this for custom instructions (e.g. citation style, tone). "
                "Leave empty to use only the built-in prompt."
            ),
        )
        EMIT_THINKING: bool = Field(
            default=True,
            description="Show abbreviated model thinking in the status bar",
        )
        SHOW_COST_TRACKING: bool = Field(
            default=True,
            description=(
                "Display running token count and cost estimate in the "
                "status bar after each model call"
            ),
        )

    # ------------------------------------------------------------------ #
    #  Init / metadata
    # ------------------------------------------------------------------ #

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list:
        return [
            {
                "id": "tongyi-deepresearch",
                "name": "Tongyi DeepResearch",
                "description": (
                    "Agentic deep-research assistant powered by "
                    "Alibaba Tongyi DeepResearch. Conducts multi-step "
                    "web searches and page visits to deliver thorough, "
                    "cited answers."
                ),
                "profile_image_url": "https://cdn-avatars.huggingface.co/v1/production/uploads/63fc4c00a3c067e62899d32b/dfd_EcIfylvu3sdc2WMqX.png",
            }
        ]

    # ------------------------------------------------------------------ #
    #  System-prompt builder
    # ------------------------------------------------------------------ #

    def _build_system_prompt(self) -> str:
        today = date.today()
        prompt = DEEPRESEARCH_SYSTEM_PROMPT_TEMPLATE.format(
            human_date=today.strftime("%A, %B %d, %Y"),
            iso_date=today.isoformat(),
        )
        preamble = self.valves.SYSTEM_PROMPT_PREAMBLE.strip()
        if preamble:
            return preamble + "\n\n" + prompt
        return prompt

    # ================================================================== #
    #  OpenRouter streaming client                                        #
    # ================================================================== #

    async def _call_openrouter(
        self,
        messages: list,
        emit_status: Optional[Callable] = None,
        max_retries: int = 3,
    ) -> tuple:
        """
        Stream a chat completion from OpenRouter.

        Returns
        -------
        (reasoning_text, content_text, usage_dict)
            *reasoning_text* comes from the provider's dedicated reasoning
            field (if any).  *content_text* is the main assistant output and
            may itself contain ``<think>`` tags when reasoning is inlined.
            *usage_dict* contains token counts and cost from the final
            streaming chunk (keys: prompt_tokens, completion_tokens,
            total_tokens, cost).  May be ``None`` if the provider did not
            report usage.
        """
        url = f"{self.valves.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
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

        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            reasoning = ""
            content = ""
            usage: Optional[dict] = None
            last_ping = time.time()

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(600.0, connect=30.0)
                ) as client:
                    async with client.stream(
                        "POST", url, json=payload, headers=headers
                    ) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            raise RuntimeError(
                                f"OpenRouter {resp.status_code}: "
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

                            # Capture usage from the final chunk
                            chunk_usage = chunk.get("usage")
                            if chunk_usage:
                                usage = chunk_usage

                            choices = chunk.get("choices")
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})

                            for rkey in ("reasoning", "reasoning_content"):
                                val = delta.get(rkey)
                                if val:
                                    reasoning += val

                            cval = delta.get("content")
                            if cval:
                                content += cval

                            if emit_status and time.time() - last_ping > 4:
                                n = len(reasoning) + len(content)
                                await emit_status(f"Generating… ({n:,} chars received)")
                                last_ping = time.time()

                return reasoning, content, usage

            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait = 2**attempt
                    if emit_status:
                        await emit_status(f"API error, retrying in {wait}s… ({exc})")
                    await asyncio.sleep(wait)

        raise last_exc  # type: ignore[misc]

    # ================================================================== #
    #  XML / JSON parsing helpers                                         #
    # ================================================================== #

    @staticmethod
    def _extract_tool_call(text: str) -> Optional[dict]:
        """Return the first parsed ``<tool_call>`` dict, or *None*."""
        m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        if not m:
            return None

        raw = m.group(1).strip()

        # Handle PythonInterpreter which puts code in <code> blocks
        if "PythonInterpreter" in raw:
            code_m = re.search(r"<code>(.*?)</code>", raw, re.DOTALL)
            code = code_m.group(1).strip() if code_m else ""
            return {"name": "PythonInterpreter", "arguments": {"code": code}}

        # Standard JSON tool call
        for parser in (
            lambda s: json.loads(s),
            lambda s: json.loads(re.sub(r",\s*([}\]])", r"\1", s.replace("'", '"'))),
        ):
            try:
                obj = parser(raw)
                if isinstance(obj, dict) and "name" in obj:
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue

        return None

    @staticmethod
    def _extract_answer(text: str) -> Optional[str]:
        """Extract content inside ``<answer>`` tags (handles unclosed tag)."""
        m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_thinking(text: str) -> Optional[str]:
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _strip_xml_for_display(text: str) -> str:
        """Remove helper XML tags so the user sees clean prose."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
        text = re.sub(r"</?answer>", "", text)
        return text.strip()

    # ================================================================== #
    #  Tool router                                                        #
    # ================================================================== #

    async def _execute_tool(
        self,
        name: str,
        arguments: dict,
        emit_status: Callable,
    ) -> str:
        """Route a parsed tool call to the appropriate handler."""
        if name == "search":
            queries = arguments.get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            return await self._execute_search(queries, emit_status)

        if name == "visit":
            urls = arguments.get("url", [])
            if isinstance(urls, str):
                urls = [urls]
            goal = arguments.get("goal", "Extract relevant information")

            if self.valves.VISIT_TOOL_ENABLED and self.valves.OPENROUTER_API_KEY:
                try:
                    from visit_tool import Tools as VisitTools
                except ImportError:
                    return (
                        "[visit] visit_tool module not found — "
                        "disable VISIT_TOOL_ENABLED or install visit_tool.py."
                    )

                visit_tools = VisitTools()
                visit_tools.valves.SUMMARY_MODEL_API_KEY = self.valves.OPENROUTER_API_KEY
                visit_tools.valves.SUMMARY_MODEL_BASE_URL = self.valves.OPENROUTER_BASE_URL
                visit_tools.valves.MAX_PAGE_TOKENS = self.valves.MAX_PAGE_LENGTH

                async def _visit_emitter(event: dict):
                    d = event.get("data", {})
                    await emit_status(d.get("description", ""), d.get("done", False))

                return await visit_tools.visit(urls, goal, _visit_emitter)

            from open_webui.retrieval.utils import get_content_from_url

            if len(urls) == 1:
                await emit_status(f"Visiting: {urls[0]}")
            else:
                await emit_status(f"Visiting {len(urls)} pages concurrently…")

            max_len = self.valves.MAX_PAGE_LENGTH

            async def _fetch_one(u: str) -> str:
                try:
                    content, _title = await asyncio.to_thread(
                        get_content_from_url, self._request, u,
                    )
                    if not content or not content.strip():
                        return (
                            f"The useful information in {u} for user goal "
                            f"{goal} as follows:\n\n"
                            f"Evidence in page:\n"
                            f"The provided webpage content could not be accessed. "
                            f"Error: Empty page content\n\n"
                            f"Summary:\n"
                            f"The webpage content could not be processed.\n"
                        )
                    if len(content) > max_len:
                        content = content[:max_len] + "\n…[content truncated]"
                    return (
                        f"The useful information in {u} for user goal "
                        f"{goal} as follows:\n\n"
                        f"Evidence in page:\n{content}\n\n"
                        f"Summary:\nRaw page content provided above for analysis.\n"
                    )
                except Exception as exc:
                    return (
                        f"The useful information in {u} for user goal "
                        f"{goal} as follows:\n\n"
                        f"Evidence in page:\n"
                        f"The provided webpage content could not be accessed. "
                        f"Error: {exc}\n\n"
                        f"Summary:\n"
                        f"The webpage content could not be processed.\n"
                    )

            if len(urls) == 1:
                return await _fetch_one(urls[0])
            results = await asyncio.gather(*(_fetch_one(u) for u in urls))
            return "\n=======\n".join(results)

        if name == "google_scholar":
            queries = arguments.get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            scholarly = [f"academic research: {q}" for q in queries]
            return await self._execute_search(scholarly, emit_status)

        if name == "PythonInterpreter":
            return (
                "[PythonInterpreter] Code execution is not available in "
                "this environment. Please reason through the computation "
                "manually or reformulate your approach using search."
            )

        if name == "parse_file":
            return "[parse_file] File parsing is not available in this " "environment."

        return f"[Error] Unknown tool: {name}"

    # ================================================================== #
    #  Search tool — concurrent queries via Open WebUI built-in search   #
    # ================================================================== #

    async def _execute_search(
        self,
        queries: List[str],
        emit_status: Callable,
    ) -> str:
        """
        Execute web searches concurrently.  The model may send an array
        of queries in a single call; we fire them all at once using
        ``asyncio.gather`` (each runs in its own thread via
        ``asyncio.to_thread``), then combine the results with
        ``=======`` separators (matching the format the model was
        trained on).
        """
        queries = queries[: self.valves.MAX_QUERIES_PER_SEARCH]

        if len(queries) == 1:
            await emit_status(f"Searching: {queries[0]}")
            return await self._search_single_query(queries[0])

        await emit_status(f"Searching {len(queries)} queries concurrently…")
        results = await asyncio.gather(*(self._search_single_query(q) for q in queries))

        return "\n=======\n".join(results)

    async def _search_single_query(self, query: str) -> str:
        """Run a single query via Open WebUI's built-in search and format results."""
        try:
            from open_webui.routers.retrieval import (
                search_web as _owui_search,
            )

            engine = self._request.app.state.config.WEB_SEARCH_ENGINE
            results = await asyncio.to_thread(
                _owui_search,
                self._request,
                engine,
                query,
                self._user,
            )
            hits = results[: self.valves.SEARCH_RESULTS_PER_QUERY] if results else []
        except Exception as exc:
            return f"[Search Error] Failed to search for '{query}': {exc}"

        if not hits:
            return f"No results found for '{query}'. " f"Try with a more general query."

        snippets: List[str] = []
        for idx, r in enumerate(hits, 1):
            title = r.title or "Untitled"
            link = r.link or ""
            snippet = r.snippet or ""

            parts = [f"{idx}. [{title}]({link})"]
            if snippet:
                parts.append(snippet)
            snippets.append("\n".join(parts))

        header = (
            f"A search for '{query}' found "
            f"{len(snippets)} results:\n\n## Web Results\n"
        )
        return header + "\n\n".join(snippets)

    # ---- tool-call display card --------------------------------------- #

    @staticmethod
    def _build_tool_call_card(
        tool_name: str,
        tool_args: dict,
        result: str,
        *,
        done: bool = True,
        max_result_display: int = 4000,
    ) -> str:
        """
        Build an HTML ``<details>`` block matching Open WebUI's native
        tool-call card format so the frontend renders a collapsible card
        with the tool name, arguments, and result.
        """
        call_id = f"tc_{uuid4().hex[:24]}"
        escaped_args = html.escape(json.dumps(tool_args, ensure_ascii=False))

        if done:
            display_result = result
            if len(display_result) > max_result_display:
                display_result = (
                    display_result[:max_result_display] + "\n…[truncated for display]"
                )
            escaped_result = html.escape(json.dumps(display_result, ensure_ascii=False))
            return (
                f'<details type="tool_calls" done="true" '
                f'id="{call_id}" name="{html.escape(tool_name)}" '
                f'arguments="{escaped_args}" '
                f'result="{escaped_result}">\n'
                f"<summary>Tool Executed</summary>\n"
                f"</details>\n"
            )

        return (
            f'<details type="tool_calls" done="false" '
            f'id="{call_id}" name="{html.escape(tool_name)}" '
            f'arguments="{escaped_args}">\n'
            f"<summary>Executing…</summary>\n"
            f"</details>\n"
        )

    # ================================================================== #
    #  Main agentic loop                                                  #
    # ================================================================== #

    async def pipe(
        self,
        body: dict,
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __request__=None,
    ) -> str:
        """
        Entry point called by Open WebUI for every user message.

        Implements a multi-turn ReAct loop:
        1. Send messages to OpenRouter (Tongyi DeepResearch).
        2. Parse the response for ``<tool_call>`` XML blocks.
        3. If a tool call is found → execute it → inject ``<tool_response>``
           → go to 1.
        4. If an ``<answer>`` block is found → return it to the user.
        5. Repeat until the model answers or the round limit is reached.
        """

        # -- store request/user for tool handlers ----------------------- #

        self._request = __request__
        self._user = None
        if __user__:
            try:
                from open_webui.models.users import UserModel

                self._user = UserModel(**__user__)
            except Exception:
                pass

        # -- event-emitter helpers --------------------------------------- #

        async def emit_status(desc: str, done: bool = False):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": desc, "done": done},
                    }
                )

        async def emit_message(text: str):
            if __event_emitter__:
                await __event_emitter__({"type": "message", "data": {"content": text}})

        # -- pre-flight checks ------------------------------------------ #

        if not self.valves.OPENROUTER_API_KEY:
            return (
                "**Configuration error:** OpenRouter API key is not set. "
                "Go to *Admin Panel → Functions → Tongyi DeepResearch → "
                "Valves* to configure it."
            )
        if __request__ is None:
            return (
                "**Configuration error:** This pipe requires access to "
                "the Open WebUI request context (`__request__`). Ensure "
                "you are running a compatible version of Open WebUI "
                "(0.4.0+)."
            )

        # -- assemble initial messages ---------------------------------- #
        # Ignore any incoming system message (e.g. global default from user
        # settings) — the pipe builds its own from the template + Valve preamble.

        messages: List[dict] = [
            {"role": "system", "content": self._build_system_prompt()}
        ]
        for msg in body.get("messages", []):
            role = msg.get("role", "")
            if role == "system":
                continue
            messages.append({"role": role, "content": msg.get("content", "")})

        # -- cost tracking ---------------------------------------------- #

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        model_calls = 0

        def _update_cost(usage: Optional[dict]) -> None:
            nonlocal total_input_tokens, total_output_tokens
            nonlocal total_cost, model_calls
            if not usage:
                return
            model_calls += 1
            total_input_tokens += usage.get("prompt_tokens", 0)
            total_output_tokens += usage.get("completion_tokens", 0)
            if "cost" in usage:
                total_cost += usage["cost"]

        def _cost_summary(prefix: str = "") -> str:
            tok = f"{total_input_tokens + total_output_tokens:,} tokens"
            if total_cost > 0:
                cost = (
                    f"${total_cost:.4f}" if total_cost < 0.01 else f"${total_cost:.2f}"
                )
                return f"{prefix}{tok} · {cost}"
            return f"{prefix}{tok}"

        # -- agentic loop ----------------------------------------------- #

        for round_num in range(1, self.valves.MAX_TOOL_ROUNDS + 1):

            await emit_status(f"Round {round_num} — calling model…")

            # --- context-length guard ---------------------------------- #
            total_chars = sum(len(m.get("content", "")) for m in messages)
            if total_chars > self.valves.MAX_CONTEXT_CHARS:
                await emit_status(
                    "Context limit approaching — requesting final answer…"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have now reached the maximum context "
                            "length you can handle. Stop making tool "
                            "calls and, based on all the information "
                            "above, think again and provide what you "
                            "consider the most likely answer within "
                            "<answer></answer> tags."
                        ),
                    }
                )

            # --- call the model ---------------------------------------- #

            try:
                reasoning, content, usage = await self._call_openrouter(
                    messages, emit_status
                )
                _update_cost(usage)
            except Exception as exc:
                await emit_status(f"API error: {exc}", done=True)
                return f"**API error:** {exc}"

            if self.valves.SHOW_COST_TRACKING:
                await emit_status(f"Round {round_num} — {_cost_summary()}")

            # Reconstruct the full assistant turn.
            # OpenRouter may return reasoning in a separate field; if so,
            # we prepend it with <think> tags.  If the content already has
            # <think> tags (inlined reasoning), we leave it as-is.
            if reasoning and "<think>" not in content:
                full = f"<think>\n{reasoning}\n</think>\n{content}"
            else:
                full = content

            messages.append({"role": "assistant", "content": full})

            # --- show abbreviated thinking in status bar --------------- #

            if self.valves.EMIT_THINKING:
                thinking = self._extract_thinking(full) or reasoning
                if thinking:
                    preview = thinking[:300].replace("\n", " ").strip()
                    if len(thinking) > 300:
                        preview += "…"
                    await emit_status(f"Round {round_num} thinking: {preview}")

            # --- check for a tool call --------------------------------- #

            tc = self._extract_tool_call(full)
            if tc:
                tool_name = tc.get("name", "unknown")
                tool_args = tc.get("arguments", {})
                if not isinstance(tool_args, dict):
                    tool_args = {}

                await emit_status(f"Round {round_num} — executing {tool_name}…")

                try:
                    result = await self._execute_tool(tool_name, tool_args, emit_status)
                except Exception as exc:
                    result = f"[Tool Error] {tool_name} failed: {exc}"

                # Show a collapsible tool-call card in the chat UI
                await emit_message(
                    self._build_tool_call_card(tool_name, tool_args, result)
                )

                # Feed the observation back as a user message with the
                # <tool_response> wrapper the model expects.
                messages.append(
                    {
                        "role": "user",
                        "content": (f"<tool_response>\n{result}\n</tool_response>"),
                    }
                )
                continue  # → next round

            # --- check for a final answer ------------------------------ #

            answer = self._extract_answer(full)
            if answer:
                await emit_status(_cost_summary("Research complete · "), done=True)
                return answer

            # --- model stopped without tool call *or* answer ----------- #
            # This can happen when the model just responds conversationally
            # (e.g. for simple greetings or clarifications).

            clean = self._strip_xml_for_display(full)
            if clean:
                await emit_status(_cost_summary("Done · "), done=True)
                return clean

            # Truly empty response — retry the round
            await emit_status(f"Round {round_num} — empty response, retrying…")

        # -- max rounds exhausted --------------------------------------- #

        await emit_status("Maximum research rounds reached — forcing final answer…")

        messages.append(
            {
                "role": "user",
                "content": (
                    "You have used all available research rounds. Based on "
                    "everything gathered above, provide your best answer "
                    "now within <answer></answer> tags."
                ),
            }
        )

        try:
            reasoning, content, usage = await self._call_openrouter(
                messages, emit_status
            )
            _update_cost(usage)
            full = (
                f"<think>\n{reasoning}\n</think>\n{content}"
                if reasoning and "<think>" not in content
                else content
            )
            answer = self._extract_answer(full)
            if answer:
                await emit_status(_cost_summary("Research complete · "), done=True)
                return answer

            await emit_status(_cost_summary("Complete · "), done=True)
            return self._strip_xml_for_display(full) or content

        except Exception as exc:
            await emit_status(_cost_summary("Error in final round · "), done=True)
            return f"Research limit reached. Final attempt failed: {exc}"
