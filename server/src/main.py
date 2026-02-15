"""Server entry point.

Run with: uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

import uvicorn


def main() -> None:
    """Start the Avatar server."""
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
