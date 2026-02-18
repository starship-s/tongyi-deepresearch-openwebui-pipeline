# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-02-17

### Fixed

- Module docstrings now start metadata on the first line so Open WebUI correctly
  picks up `title`, `author`, `version`, and other fields.
- Removed `required_open_webui_version` from pipe metadata.

### Changed

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

[Unreleased]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/starship-s/tongyi-deepresearch-openwebui-pipeline/releases/tag/v0.2.0
