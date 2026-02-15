# Project Conventions

## Python

- Use **uv** for all dependency management. Never use pip directly or conda.
- Each component (`server/`, `client/`) has its own `pyproject.toml`.
- Python 3.11. Use type hints on all function signatures.
- Use `async`/`await` for all I/O-bound code (network, audio, file).
- Use `asyncio` as the event loop. No threading unless absolutely necessary for hardware access.
- Prefer `pathlib.Path` over `os.path`.
- Use `logging` module, not `print()`, for operational output.

## Code Style

- Follow PEP 8. Max line length 120.
- Use double quotes for strings.
- Docstrings on all public functions (Google style).
- Group imports: stdlib → third-party → local. One blank line between groups.

## Architecture

- Local components (Windows): orchestrator, ASR, VAD, virtual camera/mic output.
- Remote components (Vast.ai Linux): LLM, TTS, face animation.
- Communication: WebSocket with binary/msgpack serialization.
- Server framework: FastAPI with WebSocket endpoints.
- Docker-based deployment for Vast.ai. Dockerfile in `server/` or `docker/`.

## Dependencies

- Do not vendor dependencies. Use `pyproject.toml` with version constraints.
- Pin major versions, allow minor updates (e.g., `torch>=2.0,<3`).
- Keep `models/` directory gitignored — model weights are downloaded at runtime or baked into Docker.

## Error Handling

- All network calls must have timeouts and retry logic.
- Graceful degradation: if Vast.ai server is unreachable, log error and wait for reconnection.
- Never crash silently. Log all exceptions with traceback.

## Git

- Keep secrets (API keys, Vast.ai keys) out of version control. Use `.env` files (gitignored).
- Commit working increments. Each stage completion should be a clean commit.
