# CHANGELOG

## V3 - 2026-05-18

### Changed
- Updated Phase documentation to V3.
- Kept original Phase filenames as the current working specifications.
- Moved V2 documentation backups to `archive/v2_docs/`.
- Confirmed active references point to unversioned current Phase files.

### Notes
- Files inside `archive/v2_docs/` are historical backups only.
- Current development should use the original `PHASE1` through `PHASE6` Markdown files (including phase checklists: `PHASE2_checklist.md` … `PHASE5_checklist.md`).

### Documentation update policy
- Small code changes do **not** require updating every Phase Markdown file.
- Update Phase documentation only when there is a meaningful change to: system architecture, data pipeline, model logic, Telegram report behavior, scheduling behavior, risk controls, project requirements, public interfaces or commands, or deployment procedure.
- For normal implementation changes, update `CHANGELOG.md`.
- For major requirement or design changes, update the relevant Phase Markdown file and also update `CHANGELOG.md`.
- Do not update all Phase Markdown files after every small code change.
