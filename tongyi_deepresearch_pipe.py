"""
id: tongyi_deepresearch_pipe
title: Tongyi DeepResearch
author: starship-s
author_url: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline
version: 0.2.4
license: MIT
description: Agentic deep-research pipe for Open WebUI, powered by Tongyi DeepResearch.
required_open_webui_version: 0.5.0
requirements: httpx
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.append(str(_src))

from tongyi_deepresearch_openwebui_pipeline.pipes.pipe import Pipe  # noqa: E402

__all__ = ["Pipe"]
