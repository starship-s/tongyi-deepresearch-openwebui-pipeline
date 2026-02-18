# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.4...HEAD
[0.2.4]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.0...v0.2.3
[0.2.0]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/releases/tag/v0.2.0
