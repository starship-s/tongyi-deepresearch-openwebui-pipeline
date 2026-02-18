# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.12] - 2026-02-18

### Changed

- System prompt now injects today's date at the start of the assistant identity
  ("Today's date is YYYY-MM-DD (ISO)") instead of appending it at the end.

## [0.2.11] - 2026-02-18

### Changed

- Visit tool now uses Open WebUI's built-in `get_content_from_url()` when the pipe
  passes the request context, respecting the user's configured web loader engine
  (safe_web, playwright, firecrawl, tavily, external, or YouTube). Falls back to
  httpx when the request is unavailable (standalone tool usage).

## [0.2.10] - 2026-02-18

### Fixed

- `_auto_install_model_metadata()` now correctly stores icon and description in Open
  WebUI's model database. Replaced invalid `access_grants` with `access_control=None`,
  added broad exception handling around form construction and DB writes, and replaced
  the create-once guard with an update-if-different check so previously broken entries
  are repaired and future icon/description changes are applied.

## [0.2.9] - 2026-02-18

### Fixed

- Tool auto-install now works when the pipe is imported via GitHub URL. When
  `importlib.resources` cannot locate the tool source (package not pip-installed),
  `_read_tool_source()` fetches tool modules from the project's GitHub raw URL,
  enabling search, scholar, and visit tools to be installed automatically.

## [0.2.8] - 2026-02-18

### Added

- Auto-install model metadata (icon and description) into Open WebUI's model
  database on startup. Open WebUI's `get_function_models()` ignores
  `description` and `profile_image_url` from `pipes()`, so a Model DB entry
  is now created to overlay this metadata onto the pipe model.

### Changed

- Model icon now uses the upstream Tongyi logo from the
  [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)
  repository.
- Model description updated to match the upstream repo introduction.

## [0.2.7] - 2026-02-18

### Changed

- **Breaking:** Renamed source files so filenames match their Open WebUI IDs
  (used when importing directly into Open WebUI):
  - `pipes/pipe.py` → `pipes/tongyi_deepresearch_pipe.py`
  - `tools/search_tool.py` → `tools/deepresearch_search_tool.py`
  - `tools/scholar_tool.py` → `tools/deepresearch_scholar_tool.py`
  - `tools/visit_tool.py` → `tools/deepresearch_visit_tool.py`
- Bumped `required_open_webui_version` from 0.5.0 to **0.8.0** (streaming
  return path requires Open WebUI ≥ 0.8).

### Added

- SSE streaming return path (`_pipe_stream`) — when `stream=True`, reasoning
  cards and tool-call cards are yielded directly from the pipe return instead
  of being emitted via `__event_emitter__`, enabling correct rendering of
  collapsible thinking blocks and tool cards in Open WebUI ≥ 0.8.
- `_build_thinking_block()` — wraps reasoning in `<think>` tags for Open WebUI's
  native "Thought for x seconds" renderer.
- `_execute_tool_call_and_append()` — helper that runs a tool call, appends the
  `<tool_response>` message, and returns the display card markup.
- Tool-call cards now include `<div class="tool-result">` for full result
  content alongside the truncated `result` attribute for backward compatibility.

### Fixed

- `<think>` tags no longer leak into chat output when streaming is active.
- Tool-call cards now render correctly instead of appearing as raw HTML.

## [0.2.5] - 2026-02-18

### Removed

- Root-level shim files (`tongyi_deepresearch_pipe.py`, `search_tool.py`,
  `scholar_tool.py`, `visit_tool.py`). The nested source files under `src/`
  already contain Open WebUI metadata and can be imported directly.

### Changed

- Release assets now point to the source files under
  `src/tongyi_deepresearch_openwebui_pipeline/` instead of the deleted shims.

## [0.2.4] - 2026-02-18

### Fixed

- Frontmatter `description` collapsed to a single line so Open WebUI's
  `extractFrontmatter` parser captures the full value instead of just `>`.
- Frontmatter `id` fields now use underscores (matching Open WebUI's
  auto-derive logic from the `title`) so the ID is consistent when importing.
- Tool IDs in `_TOOL_REGISTRY` and `pipes()` updated to match the new
  frontmatter IDs.

## [0.2.3] - 2026-02-17

### Fixed

- Module docstrings now start metadata on the first line so Open WebUI correctly
  picks up `title`, `author`, `version`, and other fields.
- Corrected `required_open_webui_version` from 0.4.0 to 0.5.0 (the project uses
  0.5+ import paths and the `__request__` pipe parameter).
- Added missing `scholar_tool.py` to release assets.

### Changed

- Release workflow no longer builds wheel/sdist; releases contain only the four
  `.py` shim files needed for Open WebUI import.
- README rewritten with a Quick Start guide focused on importing release shims.
- Valve names updated to `API_KEY` / `API_BASE_URL` in documentation.
- Added ruff per-file-ignores for pydocstyle rules (D205, D212, D415) on files
  that use Open WebUI metadata docstrings.

## [0.2.0] - 2026-02-17

### Added

- `src/` package layout with `pipes/pipe.py` and `tools/visit_tool.py`.
- Standalone `visit_tool.py` Open WebUI tool with `EXTRACTOR_PROMPT` LLM extraction.
- `VISIT_TOOL_ENABLED` valve in the pipe to toggle between the standalone extraction
  pipeline and the built-in Open WebUI content loader.
- `README.md` and `AGENTS.md` documentation.
- CI and release GitHub Actions workflows.

### Changed

- Pipe delegates `visit` calls to the standalone tool when `VISIT_TOOL_ENABLED=True`.
- Project is now installable via `pip install` using hatchling.

[Unreleased]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.12...HEAD
[0.2.12]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.11...v0.2.12
[0.2.11]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.10...v0.2.11
[0.2.10]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.9...v0.2.10
[0.2.9]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.5...v0.2.7
[0.2.5]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.0...v0.2.3
[0.2.0]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/releases/tag/v0.2.0
