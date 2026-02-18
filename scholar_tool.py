"""
id: deepresearch_scholar_tool
title: DeepResearch Scholar Tool
author: starship-s
author_url: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline
version: 0.1.1
license: MIT
description: Academic literature search via Open WebUI for the DeepResearch pipe.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.append(str(_src))

from tongyi_deepresearch_openwebui_pipeline.tools.scholar_tool import Tools  # noqa: E402, I001

__all__ = ["Tools"]
