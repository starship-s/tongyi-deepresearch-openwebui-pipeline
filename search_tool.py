"""
id: deepresearch-search
title: DeepResearch Search Tool
author: starship-s
author_url: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline
version: 0.2.2
license: MIT
description: >
    Searches the web via Open WebUI's built-in search engine and
    returns raw result snippets formatted to match the upstream
    DeepResearch training format.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.append(str(_src))

from tongyi_deepresearch_openwebui_pipeline.tools.search_tool import Tools  # noqa: E402

__all__ = ["Tools"]
