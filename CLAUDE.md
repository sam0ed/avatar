# Avatar

Project memory lives in **[AGENTS.md](AGENTS.md)** — current stage, progress log, the reasoning behind every
technical decision, and a log of which Vast.ai hosts work and which are broken. Read it before starting work.
Nothing else records that history.

- Stage-by-stage plan and verification criteria: `.github/prompts/plan-digitalAvatarClone.prompt.md`
- Coding conventions: `.github/copilot-instructions.md` — uv (never pip/conda), Python 3.11, async for all I/O,
  msgpack over WebSocket, logging not `print()`
- Prior research behind the big choices: `docs/`

## Two habits

**When a stage moves, update AGENTS.md in the same commit as the work** — the `Current Stage` section and a row
in the Progress Log. Not afterwards. Every time this has been left for later it has been skipped, and the
record then asserts something false until someone re-derives the truth by reading git history.

**Say in the commit message what you actually ran.** "Verified end-to-end on Vast.ai", "tested locally only",
"not yet run" — any of these is fine. Silence is not. Stage 4 landed as a clean stage-completion commit with no
runtime caveat, and weeks later working code was indistinguishable from code that had never been executed once.
