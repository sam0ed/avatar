"""Avatar server — FastAPI + WebSocket entry point.

Stage 0: Hello-world connectivity verification.
Stage 2: LLM + TTS conversational pipeline (streaming).
"""

import asyncio
import base64
import logging
import time
from collections.abc import AsyncIterator

import msgpack
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.llm.chunker import SentenceChunker
from src.llm.client import ChatSession, LLMClient
from src.tts.client import TTSClient

logger = logging.getLogger("avatar.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Avatar Server", version="0.2.0")

# Service clients — initialized once, reused across requests
llm_client = LLMClient()
tts_client = TTSClient()

# Per-connection chat sessions (keyed by client id)
_sessions: dict[str, ChatSession] = {}


def _get_client_id(ws: WebSocket) -> str:
    """Generate a unique client identifier."""
    host = ws.client.host if ws.client else "unknown"
    port = ws.client.port if ws.client else 0
    return f"{host}:{port}"


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint with service status."""
    llm_ok = await llm_client.health_check()
    tts_ok = await tts_client.health_check()
    return {
        "status": "ok" if (llm_ok and tts_ok) else "degraded",
        "version": "0.2.0",
        "llm": "ok" if llm_ok else "unavailable",
        "tts": "ok" if tts_ok else "unavailable",
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Main WebSocket endpoint.

    Message format (msgpack): {"type": str, "data": any, "ts": float}

    Supported message types:
      - ping → pong (heartbeat)
      - echo → echo_reply (connectivity test)
      - chat → streams back: chat_token, chat_audio, chat_done
    """
    await ws.accept()
    client_id = _get_client_id(ws)
    logger.info("Client connected: %s", client_id)

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

            elif msg_type == "chat":
                await _handle_chat(ws, client_id, msg)

            else:
                response = {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "server_ts": time.time(),
                }
                await ws.send_bytes(msgpack.packb(response))

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", client_id)
        _sessions.pop(client_id, None)
    except Exception:
        logger.exception("WebSocket error for %s", client_id)
        _sessions.pop(client_id, None)
        await ws.close(code=1011, reason="Internal server error")


async def _handle_chat(ws: WebSocket, client_id: str, msg: dict) -> None:
    """Handle a chat message: LLM streaming → sentence chunking → TTS.

    Sends back three types of messages:
      - chat_token: {"type": "chat_token", "data": str, "server_ts": float}
        — individual LLM tokens for real-time display
      - chat_audio: {"type": "chat_audio", "data": str (base64), "sentence": str, "server_ts": float}
        — synthesized audio for a complete sentence
      - chat_done: {"type": "chat_done", "full_text": str, "server_ts": float}
        — signals end of response

    Args:
        ws: WebSocket connection.
        client_id: Unique client identifier.
        msg: Parsed chat message with "data" field containing user text.
    """
    user_text = msg.get("data", "").strip()
    if not user_text:
        await ws.send_bytes(msgpack.packb({
            "type": "error",
            "message": "Empty chat message",
            "server_ts": time.time(),
        }))
        return

    logger.info("Chat from %s: '%s'", client_id, user_text[:100])

    # Get or create chat session for this client
    if client_id not in _sessions:
        _sessions[client_id] = ChatSession()
    session = _sessions[client_id]
    session.add_user_message(user_text)

    # Stream LLM → chunk into sentences → synthesize each sentence
    chunker = SentenceChunker()
    full_response: list[str] = []
    tts_tasks: list[asyncio.Task] = []

    try:
        async for token in llm_client.stream_chat(session):
            full_response.append(token)

            # Send token to client for real-time display
            await ws.send_bytes(msgpack.packb({
                "type": "chat_token",
                "data": token,
                "server_ts": time.time(),
            }))

            # Check for complete sentences
            sentences = chunker.add(token)
            for sentence in sentences:
                task = asyncio.create_task(
                    _synthesize_and_send(ws, sentence)
                )
                tts_tasks.append(task)

        # Flush remaining text from chunker
        remainder = chunker.flush()
        if remainder:
            task = asyncio.create_task(
                _synthesize_and_send(ws, remainder)
            )
            tts_tasks.append(task)

        # Wait for all TTS tasks to complete
        if tts_tasks:
            await asyncio.gather(*tts_tasks)

        # Store assistant response in session history
        assistant_text = "".join(full_response)
        session.add_assistant_message(assistant_text)

        # Signal completion
        await ws.send_bytes(msgpack.packb({
            "type": "chat_done",
            "full_text": assistant_text,
            "server_ts": time.time(),
        }))

        logger.info("Chat response to %s: %d chars, %d audio chunks", client_id, len(assistant_text), len(tts_tasks))

    except Exception:
        logger.exception("Chat pipeline error for %s", client_id)
        await ws.send_bytes(msgpack.packb({
            "type": "error",
            "message": "Chat pipeline error",
            "server_ts": time.time(),
        }))


async def _synthesize_and_send(ws: WebSocket, sentence: str) -> None:
    """Synthesize a sentence and send the audio to the client.

    Args:
        ws: WebSocket connection.
        sentence: Text to synthesize.
    """
    audio_data = await tts_client.synthesize(sentence)
    if audio_data:
        await ws.send_bytes(msgpack.packb({
            "type": "chat_audio",
            "data": base64.b64encode(audio_data).decode("ascii"),
            "sentence": sentence,
            "server_ts": time.time(),
        }))
    else:
        logger.warning("TTS synthesis failed for: '%s'", sentence[:50])
