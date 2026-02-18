"""DeepResearch Google Scholar tool for Open WebUI.

title: DeepResearch Scholar Tool
author: starship-s
version: 0.1.0
license: MIT
description: >
    Searches academic literature via Open WebUI's built-in search
    engine with an 'academic research' prefix, returning raw result
    snippets formatted to match the upstream DeepResearch training
    format for google_scholar.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.append(str(_src))

from tongyi_deepresearch_openwebui_pipeline.tools.scholar_tool import Tools  # noqa: E402, I001

__all__ = ["Tools"]
