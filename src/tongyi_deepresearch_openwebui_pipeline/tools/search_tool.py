"""DeepResearch search tool for Open WebUI.

title: DeepResearch Search Tool
author: starship-s
version: 0.1.0
license: MIT
description: >
    Searches the web and extracts structured evidence via a
    dedicated extractor LLM call.
requirements: duckduckgo-search, httpx
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable  # noqa: TC003

import httpx
from duckduckgo_search import AsyncDDGS
from pydantic import BaseModel, Field

SEARCH_EXTRACTOR_PROMPT = (
    "Please process the following search result snippets"
    " and query to extract relevant"
    " information:\n"
    "\n"
    "## **Search Results**\n"
    "{search_results}\n"
    "\n"
    "## **Query**\n"
    "{query}\n"
    "\n"
    "## **Task Guidelines**\n"
    "1. **Content Scanning for Rationale**: Scan the"
    " search result snippets for information **directly"
    " relevant** to the query\n"
    "2. **Key Extraction for Evidence**: Identify"
    " and extract the **most relevant information**"
    " from the snippets, you never miss any"
    " important information, output the **full"
    " original context** of the content as far as"
    " possible, it can be more than three"
    " paragraphs.\n"
    "3. **Summary Output for Summary**: Organize"
    " into a concise paragraph with logical flow,"
    " prioritizing clarity and judge the"
    " contribution of the information to the"
    " query.\n"
    "\n"
    "**Final Output Format using JSON format has"
    ' "rational", "evidence", "summary"'
    " feilds**\n"
)


class Tools:
    """Web searcher that extracts structured evidence via an extractor LLM."""

    class Valves(BaseModel):
        """User-configurable settings for the search tool."""

        SEARCH_MODEL_API_KEY: str = Field(
            default="",
            description="API key for the extractor LLM",
        )
        SEARCH_MODEL_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="OpenAI-compatible base URL",
        )
        SEARCH_MODEL_NAME: str = Field(
            default="qwen/qwen-2.5-72b-instruct",
            description="Model used for extraction",
        )
        SEARCH_TEMPERATURE: float = Field(
            default=0.7,
            ge=0.0,
            le=2.0,
            description="Extractor LLM temperature",
        )
        MAX_RESULTS_PER_QUERY: int = Field(
            default=5,
            description="DuckDuckGo results per query",
        )
        MAX_QUERIES_PER_SEARCH: int = Field(
            default=3,
            description="Cap on concurrent queries per call",
        )
        MAX_RETRIES: int = Field(
            default=3,
            description="LLM retry attempts per query",
        )

    def __init__(self) -> None:
        """Initialise valves with defaults."""
        self.valves = self.Valves()

    # ---- private helpers --------------------------------------------- #

    async def _search_ddg(self, query: str) -> str:
        try:
            async with AsyncDDGS() as ddgs:
                hits = await ddgs.atext(
                    query,
                    max_results=self.valves.MAX_RESULTS_PER_QUERY,
                )
            if not hits:
                return ""
            lines = [
                f"{idx}. [{h['title']}]({h['href']})\n{h['body']}"
                for idx, h in enumerate(hits, 1)
            ]
            return f"Search results for '{query}':\n\n" + "\n\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            return f"[Search Error] {exc}"

    async def _call_extractor(self, results_text: str, query: str) -> dict:
        prompt = SEARCH_EXTRACTOR_PROMPT.format(
            search_results=results_text, query=query
        )
        payload = {
            "model": self.valves.SEARCH_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.valves.SEARCH_TEMPERATURE,
        }
        headers = {
            "Authorization": f"Bearer {self.valves.SEARCH_MODEL_API_KEY}",
            "Content-Type": "application/json",
        }
        url = (
            f"{self.valves.SEARCH_MODEL_BASE_URL.rstrip('/')}/chat/completions"
        )

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()

        body = resp.json()
        raw_text = body["choices"][0]["message"]["content"]

        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip())
        raw_text = re.sub(r"```\s*$", "", raw_text.strip())

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", raw_text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    async def _process_query(self, query: str) -> str:
        results_text = await self._search_ddg(query)

        if not results_text or results_text.startswith("[Search Error]"):
            return self._fmt_search_error(
                query, results_text or "No results returned"
            )

        last_exc: Exception | None = None
        for _ in range(self.valves.MAX_RETRIES):
            try:
                raw_dict = await self._call_extractor(results_text, query)
            except (
                httpx.HTTPError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                last_exc = exc
                results_text = results_text[: int(len(results_text) * 0.7)]
                continue

            evidence = raw_dict.get("evidence")
            summary = raw_dict.get("summary")
            if evidence and summary:
                return self._fmt_search_ok(query, str(evidence), str(summary))
            results_text = results_text[: int(len(results_text) * 0.7)]

        if last_exc is not None:
            return self._fmt_search_error(query, str(last_exc))
        return self._fmt_search_error(
            query,
            "Extractor LLM failed to return valid JSON after retries",
        )

    # ---- formatting helpers ------------------------------------------ #

    @staticmethod
    def _fmt_search_ok(query: str, evidence: str, summary: str) -> str:
        return (
            f"The useful information found for query '{query}'"
            " as follows:\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Summary:\n{summary}\n\n"
        )

    @staticmethod
    def _fmt_search_error(query: str, _error: str) -> str:
        return (
            f"The useful information found for query '{query}'"
            " as follows:\n\n"
            "Evidence:\n"
            "No results could be retrieved for this query.\n\n"
            "Summary:\n"
            "The search could not be completed; no information"
            " is available.\n\n"
        )

    # ---- public tool methods ----------------------------------------- #

    async def search(
        self,
        query: list[str],
        __event_emitter__: Callable[[dict], Awaitable[None]] | None = None,
    ) -> str:
        """Search the web and extract structured evidence.

        Returns a summary with supporting evidence for each
        query.
        """

        async def emit(msg: str, done: bool = False) -> None:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": msg,
                            "done": done,
                        },
                    }
                )

        if not self.valves.SEARCH_MODEL_API_KEY:
            await emit(
                "Error: SEARCH_MODEL_API_KEY is not configured.",
                done=True,
            )
            return (
                "Error: SEARCH_MODEL_API_KEY valve is not"
                " set. Configure it in the tool settings."
            )

        if isinstance(query, str):
            query = [query]

        query = query[: self.valves.MAX_QUERIES_PER_SEARCH]

        if len(query) == 1:
            await emit(f"Searching: {query[0]}")
            result = await self._process_query(query[0])
        else:
            await emit(f"Searching {len(query)} queries concurrently…")
            results = await asyncio.gather(
                *(self._process_query(q) for q in query)
            )
            result = "\n=======\n".join(results)

        await emit("Done.", done=True)
        return result.strip()

    async def google_scholar(
        self,
        query: list[str],
        __event_emitter__: Callable[[dict], Awaitable[None]] | None = None,
    ) -> str:
        """Search academic literature via Google Scholar.

        Prefixes each query with 'academic research: ' and
        delegates to ``search()``.
        """
        if isinstance(query, str):
            query = [query]
        prefixed = [f"academic research: {q}" for q in query]
        return await self.search(prefixed, __event_emitter__)
