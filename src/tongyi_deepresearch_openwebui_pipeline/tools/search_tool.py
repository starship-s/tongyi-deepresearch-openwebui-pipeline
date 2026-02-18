"""
id: deepresearch_search_tool
title: DeepResearch Search Tool
author: starship-s
author_url: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline
version: 0.2.4
license: MIT
description: Web search tool returning raw snippets in DeepResearch training format.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable  # noqa: TC003

from pydantic import BaseModel, Field


class Tools:
    """Web searcher returning raw snippets in DeepResearch training format."""

    class Valves(BaseModel):
        """User-configurable settings for the search tool."""

        MAX_RESULTS_PER_QUERY: int = Field(
            default=10,
            description="Number of search results returned per query",
        )
        MAX_QUERIES_PER_SEARCH: int = Field(
            default=5,
            description="Cap on concurrent queries per call",
        )

    def __init__(self) -> None:
        """Initialise valves and request context placeholders."""
        self.valves = self.Valves()
        self.request: object | None = None
        self.user: object | None = None

    # ---- private helpers --------------------------------------------- #

    async def _search_single_query(self, query: str) -> str:
        """Run a single query via Open WebUI's built-in search."""
        try:
            from open_webui.routers.retrieval import (  # type: ignore[import-not-found]  # noqa: PLC0415
                search_web as _owui_search,
            )

            engine = self.request.app.state.config.WEB_SEARCH_ENGINE  # type: ignore[union-attr]
            results = await asyncio.to_thread(
                _owui_search,
                self.request,
                engine,
                query,
                self.user,
            )
            hits = results[: self.valves.MAX_RESULTS_PER_QUERY] if results else []
        except Exception as exc:
            return (
                f"No results found for '{query}'."
                " Try with a more general query,"
                f" or remove the year filter. ({exc})"
            )

        if not hits:
            return (
                f"No results found for '{query}'."
                " Try with a more general query,"
                " or remove the year filter."
            )

        snippets: list[str] = []
        for idx, r in enumerate(hits, 1):
            title = r.title or "Untitled"
            link = r.link or ""
            snippet = r.snippet or ""

            entry = f"{idx}. [{title}]({link})"
            if snippet:
                entry += f"\n{snippet}"
            snippets.append(entry)

        header = (
            f"A Google search for '{query}'"
            f" found {len(snippets)} results:"
            "\n\n## Web Results\n"
        )

        return header + "\n\n".join(snippets)

    # ---- public tool methods ----------------------------------------- #

    async def search(
        self,
        query: list[str],
        __event_emitter__: Callable[[dict], Awaitable[None]] | None = None,
    ) -> str:
        """Search the web and return raw result snippets.

        Returns formatted search results for each query,
        matching the upstream DeepResearch training format.
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

        if isinstance(query, str):
            query = [query]

        query = query[: self.valves.MAX_QUERIES_PER_SEARCH]

        if len(query) == 1:
            await emit(f"Searching: {query[0]}")
            result = await self._search_single_query(query[0])
        else:
            await emit(f"Searching {len(query)} queries concurrently\u2026")
            results = await asyncio.gather(
                *(self._search_single_query(q) for q in query)
            )
            result = "\n=======\n".join(results)

        await emit("Done.", done=True)
        return result.strip()
