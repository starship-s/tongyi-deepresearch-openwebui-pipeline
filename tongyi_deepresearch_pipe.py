"""Tongyi DeepResearch pipe for Open WebUI.

title: Tongyi DeepResearch
author: starship-s
version: 0.2.0
license: MIT
description: >
    Agentic deep-research pipe that bridges Tongyi DeepResearch
    (alibaba/tongyi-deepresearch-30b-a3b via OpenRouter) with
    Open WebUI.  Translates the model's XML-structured tool_call
    blocks into real web searches and URL fetches using Open
    WebUI's built-in web search and content loader, then feeds
    results back until the model produces a final answer.
required_open_webui_version: 0.4.0
requirements: httpx
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.append(str(_src))

from tongyi_deepresearch_openwebui_pipeline.pipes.pipe import Pipe  # noqa: E402

__all__ = ["Pipe"]
