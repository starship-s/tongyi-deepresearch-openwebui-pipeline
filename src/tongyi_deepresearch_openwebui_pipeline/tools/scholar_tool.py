"""
id: deepresearch_scholar_tool
title: DeepResearch Scholar Tool
author: starship-s
author_url: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline
version: 0.1.1
license: MIT
description: Academic literature search via Open WebUI for the DeepResearch pipe.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable  # noqa: TC003

from pydantic import BaseModel, Field


class Tools:
    """Scholar searcher returning raw snippets in DeepResearch training format."""

    class Valves(BaseModel):
        """User-configurable settings for the scholar tool."""

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
                f" Try with a more general query. ({exc})"
            )

        if not hits:
            return f"No results found for '{query}'. Try with a more general query."

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
            f"A Google scholar for '{query}'"
            f" found {len(snippets)} results:"
            "\n\n## Scholar Results\n"
        )

        return header + "\n\n".join(snippets)

    # ---- public tool method ------------------------------------------ #

    async def google_scholar(
        self,
        query: list[str],
        __event_emitter__: Callable[[dict], Awaitable[None]] | None = None,
    ) -> str:
        """Search academic literature via Google Scholar.

        Prefixes each query with 'academic research: ' and
        returns formatted results matching the upstream
        DeepResearch training format.
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
        prefixed = [f"academic research: {q}" for q in query]

        if len(prefixed) == 1:
            await emit(f"Scholar search: {query[0]}")
            result = await self._search_single_query(prefixed[0])
        else:
            await emit(f"Scholar searching {len(prefixed)} queries concurrently\u2026")
            results = await asyncio.gather(
                *(self._search_single_query(q) for q in prefixed)
            )
            result = "\n=======\n".join(results)

        await emit("Done.", done=True)
        return result.strip()
