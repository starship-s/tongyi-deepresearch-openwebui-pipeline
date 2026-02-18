# AGENTS.md

## Overview

This file is the authoritative guide for contributors who want to understand or
extend the agentic system powering the Tongyi DeepResearch Open WebUI pipeline.
It describes the ReAct loop that drives multi-turn research, the XML tool
contract the model uses to request actions, and a step-by-step walkthrough for
adding new tools. If you are modifying the pipe or adding capabilities, start
here.

## Project Layout

The project uses a standard `src/` layout so it is installable as a Python
package via `pip install -e ".[dev]"`. The root-level `tongyi_deepresearch_pipe.py`
and `visit_tool.py` are **thin shims** that re-export from the package — they
exist solely for Open WebUI compatibility (which loads pipes/tools as individual
files).

```
src/tongyi_deepresearch_openwebui_pipeline/   ← the real package
├── __init__.py
├── pipes/
│   └── pipe.py                               ← main pipe implementation
└── tools/
    ├── search_tool.py                        ← search tool (raw snippets)
    ├── scholar_tool.py                       ← Google Scholar tool
    └── visit_tool.py                         ← visit/extraction tool

tongyi_deepresearch_pipe.py                   ← shim: re-exports Pipe
search_tool.py                                ← shim: re-exports Tools
scholar_tool.py                               ← shim: re-exports Tools
visit_tool.py                                 ← shim: re-exports Tools
```

## ReAct Loop (`src/tongyi_deepresearch_openwebui_pipeline/pipes/pipe.py`)

The core agentic behaviour lives in `Pipe.pipe()`. Each user message triggers a
loop that alternates between model generation and tool execution:

1. **`_build_system_prompt()`** injects today's date and the tool definitions
   into `DEEPRESEARCH_SYSTEM_PROMPT_TEMPLATE`. If `SYSTEM_PROMPT_PREAMBLE` is
   set, it is prepended before the built-in prompt.

2. **`_call_llm()`** streams a chat completion from the configured API endpoint.
   Reasoning tokens arriving in a dedicated provider field are separated from
   content and wrapped in `<think>` tags so the model's chain-of-thought is
   preserved but hidden from the final display.

3. **`_extract_tool_call()`** parses the first `<tool_call>` JSON block from
   the assistant response. It handles standard JSON as well as the special
   `PythonInterpreter` format that wraps code in `<code>` tags.

4. **`_execute_tool()`** routes the parsed call to the appropriate handler —
   `_execute_search` for search, `_execute_scholar` for google_scholar,
   `_execute_visit` for visit, or a graceful error string for unexecutable
   tools like `PythonInterpreter` and `parse_file`. Each tool branch checks
   whether the tool is enabled via its valve before executing.

5. The result is injected back as a **`user` message** wrapped in
   `<tool_response>` tags, and the loop continues.

6. **`_extract_answer()`** detects `<answer>` tags in the model's output to
   terminate the loop and return the final answer to the user.

7. A **context-length guard** at `MAX_CONTEXT_CHARS` forces a final answer
   before the model's context window is exhausted. When the total character
   count of all messages exceeds this threshold, a special user message is
   appended asking the model to wrap up immediately.

## XML Tool Contract

The model communicates tool requests and receives results through XML tags.
These formats are baked into `DEEPRESEARCH_SYSTEM_PROMPT_TEMPLATE` and must be
respected by any new tool handler.

### Tool call (model → pipe)

```
<tool_call>
{"name": "<function-name>", "arguments": <args-json-object>}
</tool_call>
```

### Tool response (pipe → model)

```
<tool_response>
…result text…
</tool_response>
```

### Final answer (model → user)

```
<answer>
…answer text…
</answer>
```

## Tool Inventory

| Tool | Handler | Notes |
|---|---|---|
| `search` | `search_tool.Tools.search()` | Controlled by `SEARCH_ENABLED`; returns raw snippets matching upstream training format |
| `visit` | `visit_tool.Tools.visit()` or built-in `get_content_from_url()` | Controlled by `VISIT_ENABLED` |
| `google_scholar` | `scholar_tool.Tools.google_scholar()` | Controlled by `SCHOLAR_ENABLED`; self-contained tool with `"academic research: "` prefix |
| `PythonInterpreter` | Graceful error string | Not executable in this environment |
| `parse_file` | Graceful error string | Not executable in this environment |

## Search Tool Deep-Dive (`src/tongyi_deepresearch_openwebui_pipeline/tools/search_tool.py`)

The search tool returns **raw search snippets** with no LLM post-processing,
matching the output format the model was trained on in the upstream DeepResearch
repository (`tool_search.py`).

**Output format (per query):**

```
A Google search for '{query}' found {N} results:

## Web Results
1. [Title](link)
snippet text

2. [Title](link)
snippet text
```

Multiple queries are separated by `\n=======\n`.

The tool delegates to Open WebUI's built-in `search_web()` for the actual HTTP
requests. The pipe passes `request` and `user` objects so the tool can call
`search_web` without needing its own API credentials.

## Scholar Tool Deep-Dive (`src/tongyi_deepresearch_openwebui_pipeline/tools/scholar_tool.py`)

The scholar tool is a **self-contained** module (no imports from `search_tool`)
that duplicates the `_search_single_query` logic with scholar-specific formatting,
matching the upstream `tool_scholar.py`.

**Output format (per query):**

```
A Google scholar for '{query}' found {N} results:

## Scholar Results
1. [Title](link)
snippet text
```

Each query is automatically prefixed with `"academic research: "` before being
sent to Open WebUI's `search_web()`. Multiple queries are separated by
`\n=======\n`.

## Visit Tool Deep-Dive (`src/tongyi_deepresearch_openwebui_pipeline/tools/visit_tool.py`)

The visit tool implements a three-stage pipeline in `Tools._process_url()`:

1. **`_fetch_and_clean(url)`** — Uses `httpx.AsyncClient` to GET the page,
   strips HTML tags with a regex, runs `html.unescape`, collapses whitespace,
   and truncates to `MAX_PAGE_TOKENS` characters.

2. **`_call_extractor(content, goal)`** — Sends `EXTRACTOR_PROMPT` (with
   `{webpage_content}` and `{goal}` placeholders filled in) to the extractor
   LLM via an OpenAI-compatible chat completion call. Parses the JSON response
   for `rational`, `evidence`, and `summary` fields. On failure, retries up to
   `MAX_RETRIES` times, shrinking the content by 30% on each attempt to stay
   within the extractor model's context window.

3. **`_fmt_visit_ok()` / `_fmt_visit_error()`** — Formats the result into the
   string the pipe expects inside `<tool_response>` tags. The format matches
   the upstream DeepResearch visit output convention:
   `"The useful information in {url} for user goal {goal} as follows: …"`.

**Module resolution:** The static method `Pipe._resolve_visit_tools_class()`
locates the visit tool's `Tools` class using four strategies, tried in order:

1. **Package import** — `from tongyi_deepresearch_openwebui_pipeline.tools.visit_tool import Tools`.
   Works when the package is pip-installed.
2. **Direct module import** — `from visit_tool import Tools`. Works when
   `visit_tool.py` is on `sys.path`.
3. **`sys.modules` scan** — iterates modules whose name starts with `tool_`
   (the naming convention Open WebUI uses when it `exec()`s tool code from the
   database) and picks the first one containing a `Tools` class with a callable
   `visit` attribute. Zero-config; works when the tool has already been loaded
   by Open WebUI.
4. **Open WebUI database load** — uses `open_webui.models.tools.Tools.get_tools()`
   to find a tool whose source contains `class Tools` and `def visit`, then
   loads it via `load_tool_module_by_id`.

All four strategies are wrapped in `try`/`except` so the resolver degrades
gracefully in non-Open-WebUI environments.

**Note:** When `VISIT_ENABLED=True`, the pipe's `_execute_visit` method
automatically propagates `API_KEY` into the visit tool's
`SUMMARY_MODEL_API_KEY` valve, so users only need to configure one API key in
most setups. The search and scholar tools do not require an API key — they use
Open WebUI's built-in `search_web()` via the `request`/`user` context passed by
the pipe.

## Dynamic Tools and Auto-Install

Each tool can be individually enabled or disabled via pipe valves:
`SEARCH_ENABLED`, `SCHOLAR_ENABLED`, `VISIT_ENABLED`. When a tool is disabled:

1. Its definition is excluded from the system prompt's `<tools>` block.
2. If the model calls it anyway, `_execute_tool` returns an error listing
   available tools.

When `AUTO_INSTALL_TOOLS=True` (default), the pipe auto-installs enabled tool
modules into Open WebUI's tool registry on startup (`pipes()` entry point). It
reads tool source code from the installed package via `importlib.resources` and
uses `Tools.insert_new_tool()` / `Tools.update_tool_by_id()` from
`open_webui.models.tools`. Preferred tool IDs: `deepresearch-search`,
`deepresearch-scholar`, `deepresearch-visit`.

## Development Workflow

Install dev dependencies with `pip install -e ".[dev]"`. The three dev tools are
configured in `pyproject.toml` and must pass before any contribution is merged.

### Ruff (lint + format)

Ruff handles both linting and formatting (there is no separate `black` config).
Line length is 88 characters. Docstrings must follow the **Google** convention
(`[tool.ruff.lint.pydocstyle] convention = "google"`).

```bash
ruff check .          # lint
ruff check --fix .    # lint with auto-fix
ruff format .         # format
ruff format --check . # verify formatting without writing
```

The lint rule set is extensive — see `[tool.ruff.lint] select` in
`pyproject.toml` for the full list. Key rule groups: Pyflakes, pycodestyle,
isort, pydocstyle (Google), pyupgrade, flake8-bandit, flake8-bugbear, and
Pylint.

### Pyright (type checking)

Pyright runs in **standard** mode targeting Python 3.14.

```bash
pyright
```

### Pytest

```bash
pytest
```

### Releasing

1. Bump `version` in `pyproject.toml` and add an entry to `CHANGELOG.md`.
2. Commit, tag, and push:

   ```bash
   git tag v<version>
   git push origin v<version>
   ```

3. The `release.yml` workflow runs automatically, building the wheel and sdist
   via hatchling and publishing a GitHub Release with assets: the wheel,
   the sdist, `tongyi_deepresearch_pipe.py`, `search_tool.py`,
   `scholar_tool.py`, and `visit_tool.py`.

## Adding a New Tool — Step-by-Step Guide

1. In `DEEPRESEARCH_SYSTEM_PROMPT_TEMPLATE` (top of `pipes/pipe.py`), add a
   new `{"type": "function", "function": {...}}` JSON line inside the
   `<tools>` block, following the exact same format as `search` and `visit`.

2. In `Pipe._execute_tool()`, add a new `if name == "<your_tool>":` branch
   that calls your handler and returns a plain string result.

3. If the tool is complex, create
   `src/tongyi_deepresearch_openwebui_pipeline/tools/<your_tool>.py` with a
   `Tools` class and `Valves(BaseModel)`, mirroring the structure of
   `tools/visit_tool.py`.

4. Import and instantiate it inside the `if name == "<your_tool>":` branch,
   propagating the API key from `self.valves` as done for `visit_tool`.

5. Update this `AGENTS.md` Tool Inventory table and the `README.md` Features
   list.
