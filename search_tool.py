"""
id: deepresearch_search_tool
title: DeepResearch Search Tool
author: starship-s
author_url: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline
version: 0.2.4
license: MIT
description: Web search tool returning raw snippets in DeepResearch training format.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.append(str(_src))

from tongyi_deepresearch_openwebui_pipeline.tools.search_tool import Tools  # noqa: E402

__all__ = ["Tools"]
