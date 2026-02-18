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

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.append(str(_src))

from tongyi_deepresearch_openwebui_pipeline.tools.visit_tool import Tools  # noqa: E402

__all__ = ["Tools"]
