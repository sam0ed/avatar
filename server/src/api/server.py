"""Avatar server — FastAPI + WebSocket entry point.

Stage 0: Hello-world connectivity verification.
Later stages add LLM, TTS, and face animation endpoints.
"""

import logging
import time

import msgpack
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger("avatar.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Avatar Server", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Main WebSocket endpoint.

    Stage 0: Echoes messages and responds to ping with pong + server timestamp.
    Message format (msgpack): {"type": str, "data": any, "ts": float}
    """
    await ws.accept()
    client_host = ws.client.host if ws.client else "unknown"
    logger.info("Client connected: %s", client_host)

    try:
        while True:
            raw = await ws.receive_bytes()
            msg = msgpack.unpackb(raw, raw=False)
            msg_type = msg.get("type", "unknown")

            if msg_type == "ping":
                response = {
                    "type": "pong",
                    "client_ts": msg.get("ts", 0),
                    "server_ts": time.time(),
                }
                await ws.send_bytes(msgpack.packb(response))

            elif msg_type == "echo":
                response = {
                    "type": "echo_reply",
                    "data": msg.get("data", ""),
                    "server_ts": time.time(),
                }
                await ws.send_bytes(msgpack.packb(response))

            else:
                response = {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "server_ts": time.time(),
                }
                await ws.send_bytes(msgpack.packb(response))

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", client_host)
    except Exception:
        logger.exception("WebSocket error for %s", client_host)
        await ws.close(code=1011, reason="Internal server error")
