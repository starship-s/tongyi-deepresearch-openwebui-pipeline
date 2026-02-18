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

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.append(str(_src))

from tongyi_deepresearch_openwebui_pipeline.tools.search_tool import Tools  # noqa: E402

__all__ = ["Tools"]
