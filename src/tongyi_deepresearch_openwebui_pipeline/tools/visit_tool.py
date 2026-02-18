"""
id: deepresearch_visit_tool
title: DeepResearch Visit Tool
author: starship-s
author_url: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline
version: 0.1.1
license: MIT
description: Visits URLs and extracts evidence via a dedicated extractor LLM call.
requirements: httpx
"""

from __future__ import annotations

import asyncio
import html
import json
import re
from collections.abc import Awaitable, Callable  # noqa: TC003

import httpx
from pydantic import BaseModel, Field

EXTRACTOR_PROMPT = (
    "Please process the following webpage content"
    " and user goal to extract relevant"
    " information:\n"
    "\n"
    "## **Webpage Content**\n"
    "{webpage_content}\n"
    "\n"
    "## **User Goal**\n"
    "{goal}\n"
    "\n"
    "## **Task Guidelines**\n"
    "1. **Content Scanning for Rationale**: Locate"
    " the **specific sections/data** directly"
    " related to the user's goal within the"
    " webpage content\n"
    "2. **Key Extraction for Evidence**: Identify"
    " and extract the **most relevant information**"
    " from the content, you never miss any"
    " important information, output the **full"
    " original context** of the content as far as"
    " possible, it can be more than three"
    " paragraphs.\n"
    "3. **Summary Output for Summary**: Organize"
    " into a concise paragraph with logical flow,"
    " prioritizing clarity and judge the"
    " contribution of the information to the"
    " goal.\n"
    "\n"
    "**Final Output Format using JSON format has"
    ' "rational", "evidence", "summary"'
    " feilds**\n"
)


class Tools:
    """URL visitor that extracts structured evidence via an extractor LLM."""

    class Valves(BaseModel):
        """User-configurable settings for the visit tool."""

        SUMMARY_MODEL_API_KEY: str = Field(
            default="",
            description="API key for the extractor LLM",
        )
        SUMMARY_MODEL_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="OpenAI-compatible base URL",
        )
        SUMMARY_MODEL_NAME: str = Field(
            default="qwen/qwen3-30b-a3b-instruct-2507",
            description="Model used for extraction",
        )
        SUMMARY_TEMPERATURE: float = Field(
            default=0.7,
            ge=0.0,
            le=2.0,
            description="Extractor LLM temperature",
        )
        MAX_PAGE_TOKENS: int = Field(
            default=120_000,
            description="Max characters kept from a fetched page",
        )
        MAX_RETRIES: int = Field(
            default=3,
            description="LLM retry attempts per URL",
        )

    def __init__(self) -> None:
        """Initialise valves with defaults."""
        self.valves = self.Valves()

    # ---- private helpers --------------------------------------------- #

    async def _fetch_and_clean(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            raw = resp.text

        text = re.sub(r"<[^>]+>", " ", raw)
        text = html.unescape(text)
        text = re.sub(r"\s{2,}", " ", text).strip()

        limit = self.valves.MAX_PAGE_TOKENS
        if len(text) > limit:
            text = text[:limit] + "\n…[content truncated]"
        return text

    async def _call_extractor(self, content: str, goal: str) -> dict:
        prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
        payload = {
            "model": self.valves.SUMMARY_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.valves.SUMMARY_TEMPERATURE,
        }
        headers = {
            "Authorization": (f"Bearer {self.valves.SUMMARY_MODEL_API_KEY}"),
            "Content-Type": "application/json",
        }
        url = f"{self.valves.SUMMARY_MODEL_BASE_URL.rstrip('/')}/chat/completions"

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

    async def _process_url(self, url: str, goal: str) -> str:
        try:
            content = await self._fetch_and_clean(url)
        except Exception as exc:
            return self._fmt_visit_error(url, goal, str(exc))

        if not content or not content.strip():
            return self._fmt_visit_error(url, goal, "Empty page content")

        last_exc: Exception | None = None
        for _ in range(self.valves.MAX_RETRIES):
            try:
                raw_dict = await self._call_extractor(content, goal)
            except (
                httpx.HTTPError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                last_exc = exc
                content = content[: int(len(content) * 0.7)]
                continue

            evidence = raw_dict.get("evidence")
            summary = raw_dict.get("summary")
            if evidence and summary:
                return self._fmt_visit_ok(
                    url,
                    goal,
                    str(evidence),
                    str(summary),
                )
            content = content[: int(len(content) * 0.7)]

        if last_exc is not None:
            return self._fmt_visit_error(url, goal, str(last_exc))
        return self._fmt_visit_error(
            url,
            goal,
            "Extractor LLM failed to return valid JSON after retries",
        )

    # ---- formatting helpers ------------------------------------------ #

    @staticmethod
    def _fmt_visit_ok(url: str, goal: str, evidence: str, summary: str) -> str:
        return (
            f"The useful information in {url} for user goal "
            f"{goal} as follows: \n\n"
            f"Evidence in page: \n{evidence}\n\n"
            f"Summary: \n{summary}\n\n"
        )

    @staticmethod
    def _fmt_visit_error(url: str, goal: str, _error: str) -> str:
        return (
            f"The useful information in {url} for user goal "
            f"{goal} as follows: \n\n"
            f"Evidence in page: \n"
            "The provided webpage content could not be"
            " accessed. Please check the URL or file"
            " format.\n\n"
            f"Summary: \n"
            "The webpage content could not be processed,"
            " and therefore, no information is"
            " available.\n\n"
        )

    # ---- public tool method ------------------------------------------ #

    async def visit(
        self,
        url: list[str],
        goal: str,
        __event_emitter__: (Callable[[dict], Awaitable[None]] | None) = None,
    ) -> str:
        """Visit webpages and extract structured evidence.

        Returns a summary with supporting evidence for each
        URL.
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

        if not self.valves.SUMMARY_MODEL_API_KEY:
            await emit(
                "Error: SUMMARY_MODEL_API_KEY is not configured.",
                done=True,
            )
            return (
                "Error: SUMMARY_MODEL_API_KEY valve is not"
                " set. Configure it in the tool settings."
            )

        if isinstance(url, str):
            url = [url]

        if len(url) == 1:
            await emit(f"Visiting: {url[0]}")
            result = await self._process_url(url[0], goal)
        else:
            await emit(f"Visiting {len(url)} pages concurrently…")
            results = await asyncio.gather(*(self._process_url(u, goal) for u in url))
            result = "\n=======\n".join(results)

        await emit("Done.", done=True)
        return result.strip()
