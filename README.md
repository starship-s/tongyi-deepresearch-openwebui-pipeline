# Tongyi DeepResearch — Open WebUI Pipeline

> Agentic deep-research pipe that bridges Tongyi DeepResearch
> (alibaba/tongyi-deepresearch-30b-a3b) with Open WebUI via any
> OpenAI-compatible API.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Open WebUI ≥ 0.5.0](https://img.shields.io/badge/Open%20WebUI-%E2%89%A5%200.5.0-2ea44f.svg)](https://openwebui.com/)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-supported-6C47FF.svg)](https://openrouter.ai/)

## Overview

This project bridges Alibaba's Tongyi DeepResearch model into Open WebUI's
agentic pipe system via OpenRouter (or any OpenAI-compatible
  provider hosting `alibaba/tongyi-deepresearch-30b-a3b`). The pipe implements a multi-turn ReAct loop
that intercepts the model's XML-structured `<tool_call>` blocks, translates
batched search queries into concurrent web searches using whatever search engine
is configured in Open WebUI, fetches and processes URLs, then feeds the results
back wrapped in `<tool_response>` tags until the model produces a final
`<answer>`.

What makes this pipeline distinct is its two-tier architecture: the outer ReAct
loop manages tool routing and conversation flow, while the standalone
`visit_tool` runs a dedicated LLM-based extraction pipeline driven by
`EXTRACTOR_PROMPT`. Instead of dumping raw page content into context, the visit
tool sends fetched HTML through a separate extractor model that returns
structured JSON with `rational`, `evidence`, and `summary` fields — keeping the
research model's context clean and focused.

## Architecture

```mermaid
sequenceDiagram
    participant U as User
    participant OW as Open WebUI
    participant P as Pipe (pipes/pipe.py)
    participant OR as OpenRouter
    participant VT as visit_tool.py
    participant WS as Web Search (built-in)

    U->>OW: Send message
    OW->>P: pipe(body, __event_emitter__)
    loop ReAct rounds
        P->>OR: chat/completions (stream)
        OR-->>P: <tool_call> or <answer>
        alt tool_call = search (SEARCH_ENABLED)
            P->>WS: search_web()
            WS-->>P: results
        else tool_call = google_scholar (SCHOLAR_ENABLED)
            P->>WS: search_web() (with "academic research:" prefix)
            WS-->>P: results
        else tool_call = visit (VISIT_ENABLED)
            P->>VT: visit(urls, goal)
            VT->>VT: _fetch_and_clean() + _call_extractor()
            VT-->>P: structured evidence
        end
        P->>OR: <tool_response>…</tool_response>
    end
    OR-->>P: <answer>…</answer>
    P-->>U: Final answer
```

## Features

- **Multi-step ReAct loop** with configurable round limit (`MAX_TOOL_ROUNDS`, up to 200)
- **Concurrent search queries** — the model can fire multiple queries in a single `search` call; they execute in parallel via `asyncio.gather`
- **LLM-based structured extraction** via `EXTRACTOR_PROMPT` — the visit tool sends page content through a dedicated extractor model that returns `rational`, `evidence`, and `summary` fields
- **Collapsible tool-call cards** rendered in the chat UI as native `<details>` blocks
- **Cost and token tracking** — running totals displayed in the status bar after each model call
- **Configurable context-length guard** (`MAX_CONTEXT_CHARS`) — forces a final answer before the model's context window is exhausted
- **Per-tool enable/disable valves** — individually control `search`, `google_scholar`, and `visit` via `SEARCH_ENABLED`, `SCHOLAR_ENABLED`, `VISIT_ENABLED`; disabled tools are excluded from the system prompt
- **Auto-install to Open WebUI** — when `AUTO_INSTALL_TOOLS=True`, enabled tool modules are automatically registered/updated in Open WebUI's tool registry on startup

## Installation

### Prerequisites

- Open WebUI ≥ 0.5.0
- An [OpenRouter](https://openrouter.ai/) API key (or any OpenAI-compatible
  provider hosting Tongyi DeepResearch)

### Quick Start (recommended)

1. Download the four `.py` files from the latest
   [GitHub Release](https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/releases/latest):
   - **`tongyi_deepresearch_pipe.py`** — the pipe (import as a Function)
   - **`search_tool.py`** — web search tool
   - **`scholar_tool.py`** — Google Scholar tool
   - **`visit_tool.py`** — URL visit / extraction tool
2. In Open WebUI, go to **Workspace → Functions**, click **Import** (or **+**
   → **Import**), and import `tongyi_deepresearch_pipe.py`.
3. Go to **Workspace → Tools** and import `search_tool.py`,
   `scholar_tool.py`, and `visit_tool.py`.
4. Open the pipe's **Valves** and set your `API_KEY`.
5. Enable a Web Search engine in **Admin → Settings → Web Search**.
6. Start a new chat and select **Tongyi DeepResearch** from the model picker.

> **Tip:** With `AUTO_INSTALL_TOOLS=True` (default), you only need to import
> the pipe — it will automatically register the three tool modules into Open
> WebUI's **Tools** panel on first load. You can skip step 3 if you prefer
> this approach.

### Development

```bash
git clone https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline.git
cd tongyi-deepresearch-openwebui-pipeline
pip install -e ".[dev]"
```

### Releases

Browse all releases on the [Releases](https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/releases) page.

## Pipe Valves Reference

| Valve | Type | Default | Description |
|---|---|---|---|
| `API_KEY` | `str` | `""` | API key for the OpenAI-compatible endpoint |
| `API_BASE_URL` | `str` | `https://openrouter.ai/api/v1` | OpenAI-compatible API base URL |
| `MODEL_ID` | `str` | `alibaba/tongyi-deepresearch-30b-a3b` | Model identifier on the API provider |
| `MAX_TOOL_ROUNDS` | `int` | `30` | Max agentic rounds (1–200) |
| `SEARCH_RESULTS_PER_QUERY` | `int` | `5` | Results per search query (1–20) |
| `MAX_QUERIES_PER_SEARCH` | `int` | `5` | Max queries per `search` call (1–10) |
| `MAX_PAGE_LENGTH` | `int` | `50000` | Max chars kept from a fetched page |
| `SEARCH_ENABLED` | `bool` | `True` | Enable the search tool and include it in the system prompt |
| `SCHOLAR_ENABLED` | `bool` | `True` | Enable the google_scholar tool and include it in the system prompt |
| `VISIT_ENABLED` | `bool` | `True` | Enable the visit tool and include it in the system prompt |
| `AUTO_INSTALL_TOOLS` | `bool` | `True` | Auto-install enabled tool modules into Open WebUI's tool registry on startup |
| `TEMPERATURE` | `float` | `0.6` | Sampling temperature (0–2) |
| `TOP_P` | `float` | `0.95` | Nucleus sampling (0–1) |
| `PRESENCE_PENALTY` | `float` | `1.1` | Presence penalty (0–2) |
| `MAX_TOKENS` | `int` | `16000` | Max tokens per generation call |
| `MAX_CONTEXT_CHARS` | `int` | `400000` | Context budget before forced wrap-up |
| `SYSTEM_PROMPT_PREAMBLE` | `str` | `""` | Custom instructions prepended to system prompt |
| `EMIT_THINKING` | `bool` | `True` | Show thinking preview in status bar |
| `SHOW_COST_TRACKING` | `bool` | `True` | Show token count and cost in status bar |

## Visit Tool Valves Reference

| Valve | Type | Default | Description |
|---|---|---|---|
| `SUMMARY_MODEL_API_KEY` | `str` | `""` | API key for the extractor LLM |
| `SUMMARY_MODEL_BASE_URL` | `str` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL |
| `SUMMARY_MODEL_NAME` | `str` | `qwen/qwen3-30b-a3b-instruct-2507` | Extractor model |
| `SUMMARY_TEMPERATURE` | `float` | `0.7` | Extractor temperature (0–2) |
| `MAX_PAGE_TOKENS` | `int` | `120000` | Max chars kept from a fetched page |
| `MAX_RETRIES` | `int` | `3` | LLM retry attempts per URL |
