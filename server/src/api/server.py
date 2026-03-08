"""Avatar server — FastAPI + WebSocket entry point.

Stage 0: Hello-world connectivity verification.
Stage 2: LLM + TTS conversational pipeline (streaming).
Stage 4: Face animation with decoupled A/V streaming.
"""

import asyncio
import base64
import contextlib
import logging
import os
import time
from pathlib import Path

import msgpack
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from starlette.requests import Request

from src.llm.chunker import SentenceChunker
from src.llm.client import ChatSession, LLMClient
from src.tts.client import TTSClient

logger = logging.getLogger("avatar.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Avatar Server", version="0.5.0")

# Service clients — initialized once, reused across requests
llm_client = LLMClient()
tts_client = TTSClient()
logger.info("Voice cloning disabled by default (upload refs via /voice/reference, enable via /voice/enable)")

# Face animation (optional, gated by FACE_ENABLED env var)
FACE_ENABLED = os.environ.get("FACE_ENABLED", "false").lower() == "true"
face_client = None
_active_avatar_id: str | None = None

if FACE_ENABLED:
    from src.face.client import FaceAnimationClient
    face_client = FaceAnimationClient()
    logger.info("Face animation ENABLED (FACE_BASE_URL=%s)", face_client.base_url)
else:
    logger.info("Face animation disabled (set FACE_ENABLED=true to enable)")

# Shared references directory — Fish Speech reads from here at inference time.
# Orchestrator runs from /app/orchestrator, TTS from /app.
REFERENCES_DIR = Path(os.environ.get("REFERENCES_DIR", "/app/references"))

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
    face_ok = False
    if face_client is not None:
        face_ok = await face_client.health_check()
    result: dict[str, str] = {
        "status": "ok" if (llm_ok and tts_ok) else "degraded",
        "version": "0.5.0",
        "llm": "ok" if llm_ok else "unavailable",
        "tts": "ok" if tts_ok else "unavailable",
    }
    if FACE_ENABLED:
        result["face"] = "ok" if face_ok else "unavailable"
        result["face_avatar"] = _active_avatar_id or "none"
    return result


# ── Voice cloning endpoints ──────────────────────────────────────────

@app.post("/voice/reference")
async def upload_voice_reference(
    audio: UploadFile = File(..., description="WAV audio file"),
    text: str = Form(..., description="Transcript of the audio"),
    ref_id: str = Form("my-voice", description="Reference ID (folder name)"),
) -> dict:
    """Upload a voice reference to the shared filesystem.

    Multiple files can be uploaded under the same ref_id — Fish Speech
    loads all audio+lab pairs from the folder at inference time.
    Each file is stored as <stem>.wav + <stem>.lab inside
    references/<ref_id>/.
    """
    audio_bytes = await audio.read()
    stem = (audio.filename or "sample").rsplit(".", 1)[0]

    ref_dir = REFERENCES_DIR / ref_id
    ref_dir.mkdir(parents=True, exist_ok=True)

    wav_path = ref_dir / f"{stem}.wav"
    lab_path = ref_dir / f"{stem}.lab"
    wav_path.write_bytes(audio_bytes)
    lab_path.write_text(text.strip(), encoding="utf-8")

    # Count total files in this reference folder
    audio_count = len(list(ref_dir.glob("*.wav")))
    logger.info(
        "Saved reference '%s/%s' (%d bytes, %d chars) — %d audio(s) in folder",
        ref_id, stem, len(audio_bytes), len(text), audio_count,
    )
    return {
        "message": f"Reference saved ({len(audio_bytes)} bytes)",
        "reference_id": ref_id,
        "file": f"{stem}.wav",
        "audio_count": audio_count,
    }


@app.get("/voice/references")
async def list_voice_references() -> dict:
    """List available reference IDs and their audio file counts."""
    refs: dict[str, int] = {}
    if REFERENCES_DIR.is_dir():
        for d in sorted(REFERENCES_DIR.iterdir()):
            if d.is_dir():
                count = len(list(d.glob("*.wav")))
                if count > 0:
                    refs[d.name] = count
    return {"references": refs}

@app.post("/voice/enable")
async def enable_voice(ref_id: str = "") -> dict:
    """Enable voice cloning with a server-side reference and run warmup.

    The reference folder must exist under /app/references/<ref_id>/
    with at least one .wav + .lab pair.
    """
    if not ref_id:
        return {"enabled": False, "error": "ref_id is required"}

    tts_client.set_reference_id(ref_id)
    warmup_ok = await tts_client.warmup()
    return {
        "enabled": True,
        "reference_id": tts_client.reference_id,
        "warmup": "ok" if warmup_ok else "failed",
    }


@app.post("/voice/disable")
async def disable_voice() -> dict:
    """Disable voice cloning (uses default TTS voice)."""
    tts_client.clear_reference_id()
    return {"enabled": False, "reference_id": None}


@app.get("/voice/status")
async def voice_status() -> dict:
    """Get current voice cloning status."""
    return {
        "enabled": tts_client.voice_enabled,
        "reference_id": tts_client.reference_id,
    }


# ── Face Animation Endpoints ────────────────────────────────────────


@app.post("/face/prepare")
async def prepare_face(request: Request) -> dict:
    """Upload a reference video for avatar preparation.

    Expects multipart form: video file + optional avatar_id.
    """
    if not FACE_ENABLED or face_client is None:
        return {"error": "Face animation is not enabled"}
    form = await request.form()
    video_file = form.get("video")
    if video_file is None:
        return {"error": "No video file provided"}
    avatar_id = form.get("avatar_id")
    if isinstance(avatar_id, str) and avatar_id.strip():
        avatar_id = avatar_id.strip()
    else:
        avatar_id = None
    video_bytes = await video_file.read()
    result = await face_client.prepare_avatar(video_bytes, avatar_id)
    return result


@app.post("/face/enable")
async def enable_face(avatar_id: str) -> dict:
    """Enable face animation with the specified avatar."""
    global _active_avatar_id
    if not FACE_ENABLED or face_client is None:
        return {"error": "Face animation is not enabled"}
    avatars = await face_client.list_avatars()
    if avatar_id not in avatars:
        return {"error": f"Avatar '{avatar_id}' not found"}
    _active_avatar_id = avatar_id
    return {"enabled": True, "avatar_id": avatar_id}


@app.post("/face/disable")
async def disable_face() -> dict:
    """Disable face animation."""
    global _active_avatar_id
    _active_avatar_id = None
    return {"enabled": False, "avatar_id": None}


@app.get("/face/status")
async def face_status() -> dict:
    """Get current face animation status."""
    return {
        "face_enabled": FACE_ENABLED,
        "face_available": face_client is not None,
        "active_avatar_id": _active_avatar_id,
    }


@app.get("/face/avatars")
async def list_face_avatars() -> dict:
    """List available avatar IDs."""
    if not FACE_ENABLED or face_client is None:
        return {"avatars": []}
    avatars = await face_client.list_avatars()
    return {"avatars": avatars}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Main WebSocket endpoint.

    Message format (msgpack): {"type": str, "data": any, "ts": float}

    Supported message types:
      - ping → pong (heartbeat)
      - echo → echo_reply (connectivity test)
      - chat → streams back: chat_token, chat_audio, chat_done
      - chat_cancel → cancels ongoing chat, sends chat_cancelled
    """
    await ws.accept()
    client_id = _get_client_id(ws)
    logger.info("Client connected: %s", client_id)

    chat_task: asyncio.Task | None = None

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
                # Cancel any existing chat before starting a new one
                if chat_task is not None and not chat_task.done():
                    chat_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await chat_task
                chat_task = asyncio.create_task(_handle_chat(ws, client_id, msg))

            elif msg_type == "chat_cancel":
                if chat_task is not None and not chat_task.done():
                    chat_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await chat_task
                    logger.info("Chat cancel completed for %s", client_id)
                elif chat_task is not None and chat_task.done():
                    # Task already finished (chat_done sent) — no extra message needed.
                    logger.info("Chat cancel for %s: task already done", client_id)
                else:
                    # No task exists — acknowledge immediately.
                    await ws.send_bytes(msgpack.packb({
                        "type": "chat_cancelled",
                        "server_ts": time.time(),
                    }))

            else:
                response = {
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                    "server_ts": time.time(),
                }
                await ws.send_bytes(msgpack.packb(response))

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", client_id)
        if chat_task is not None and not chat_task.done():
            chat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await chat_task
        _sessions.pop(client_id, None)
    except Exception:
        logger.exception("WebSocket error for %s", client_id)
        if chat_task is not None and not chat_task.done():
            chat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await chat_task
        _sessions.pop(client_id, None)
        with contextlib.suppress(Exception):
            await ws.close(code=1011, reason="Internal server error")


async def _handle_chat(ws: WebSocket, client_id: str, msg: dict) -> None:
    """Handle a chat message: LLM streaming → sentence chunking → TTS streaming.

    Sends back three types of messages:
      - chat_token: {"type": "chat_token", "data": str, "server_ts": float}
        — individual LLM tokens for real-time display
      - chat_audio: {"type": "chat_audio", "data": str (base64), "sentence": str, "server_ts": float}
        — synthesized audio chunk (complete WAV) for playback
      - chat_done: {"type": "chat_done", "full_text": str, "server_ts": float}
        — signals end of response

    Sentences are synthesized in order via a background consumer task.
    Each sentence’s audio is streamed as multiple small WAV chunks for
    lower first-audio latency.

    Args:
        ws: WebSocket connection.
        client_id: Unique client identifier.
        msg: Parsed chat message with "data" field containing user text.
    """
    user_text = msg.get("data", "").strip()
    chat_id = msg.get("chat_id", "")
    if not user_text:
        await ws.send_bytes(msgpack.packb({
            "type": "error",
            "message": "Empty chat message",
            "chat_id": chat_id,
            "server_ts": time.time(),
        }))
        return

    logger.info("Chat from %s [%s]: '%s'", client_id, chat_id, user_text[:100])

    # Get or create chat session for this client
    if client_id not in _sessions:
        _sessions[client_id] = ChatSession()
    session = _sessions[client_id]

    # If the previous response was interrupted, prepend context so the LLM
    # knows not to repeat itself.
    if session.was_interrupted:
        user_text = (
            "[System: your previous response was cut short because the user"
            " interrupted you. Don't repeat what you already said.]\n"
            + user_text
        )
        session.was_interrupted = False

    session.add_user_message(user_text)

    # Stream LLM → chunk into sentences → stream-synthesize each sentence
    chunker = SentenceChunker()
    full_response: list[str] = []
    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
    audio_chunk_count = 0

    # Video pipeline: PCM chunks are forked here for MuseTalk processing
    video_pcm_queue: asyncio.Queue[bytes | None] | None = None
    face_session_id: str | None = None

    if FACE_ENABLED and face_client is not None and _active_avatar_id is not None:
        video_pcm_queue = asyncio.Queue()
        try:
            face_session_id = await face_client.start_session(_active_avatar_id)
        except Exception:
            logger.warning("Failed to start face session, continuing audio-only")
            video_pcm_queue = None

    async def _tts_consumer() -> None:
        """Sequentially process sentences from the queue with streaming TTS.

        Sends chat_audio immediately. If face animation is active, forks
        raw PCM to the video_pcm_queue for parallel processing.
        """
        nonlocal audio_chunk_count
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:  # Poison pill — all sentences submitted
                break
            try:
                chunk_idx = 0
                async for wav_chunk in tts_client.synthesize_streaming(sentence):
                    # Send audio immediately (decoupled — never blocked by face)
                    await ws.send_bytes(msgpack.packb({
                        "type": "chat_audio",
                        "data": base64.b64encode(wav_chunk).decode("ascii"),
                        "sentence": sentence if chunk_idx == 0 else "",
                        "chat_id": chat_id,
                        "server_ts": time.time(),
                    }))

                    # Fork raw PCM to video pipeline (strip 44-byte WAV header)
                    if video_pcm_queue is not None and len(wav_chunk) > 44:
                        raw_pcm = wav_chunk[44:]
                        await video_pcm_queue.put(raw_pcm)

                    chunk_idx += 1
                audio_chunk_count += chunk_idx
                if chunk_idx == 0:
                    logger.warning("TTS streaming produced no audio: '%s'", sentence[:50])
            except Exception:
                logger.exception("TTS streaming error for: '%s'", sentence[:50])

        # Signal video pipeline that audio is done
        if video_pcm_queue is not None:
            await video_pcm_queue.put(None)

    async def _video_producer() -> None:
        """Consume PCM from video queue, feed to MuseTalk, send chat_video.

        Runs in parallel with _tts_consumer. Batches ~200-400ms of PCM
        before feeding MuseTalk for GPU efficiency.
        """
        if video_pcm_queue is None or face_session_id is None or face_client is None:
            return

        # Batch parameters: accumulate ~300ms of audio before feeding
        # 44100Hz * 2 bytes/sample * 0.3s = 26460 bytes
        batch_threshold = 26460
        pcm_batch = bytearray()
        frame_idx = 0

        async def _feed_and_send_frames(pcm_data: bytes) -> None:
            """Feed PCM to MuseTalk and send resulting JPEG frames."""
            nonlocal frame_idx
            try:
                jpeg_frames = await face_client.feed_audio(face_session_id, pcm_data)
                for jpeg_bytes in jpeg_frames:
                    await ws.send_bytes(msgpack.packb({
                        "type": "chat_video",
                        "frame": base64.b64encode(jpeg_bytes).decode("ascii"),
                        "frame_idx": frame_idx,
                        "fps": 25,
                        "chat_id": chat_id,
                        "server_ts": time.time(),
                    }))
                    frame_idx += 1
            except Exception:
                logger.warning("Failed to feed audio to face service")

        try:
            while True:
                pcm_chunk = await video_pcm_queue.get()
                if pcm_chunk is None:
                    # Flush remaining PCM
                    if pcm_batch:
                        await _feed_and_send_frames(bytes(pcm_batch))
                    break

                pcm_batch.extend(pcm_chunk)

                # Feed when batch is large enough
                if len(pcm_batch) >= batch_threshold:
                    await _feed_and_send_frames(bytes(pcm_batch))
                    pcm_batch.clear()

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Video producer error")
        finally:
            # End face session to flush final frames + release GPU memory
            try:
                final_frames = await face_client.end_session(face_session_id)
                for jpeg_bytes in final_frames:
                    await ws.send_bytes(msgpack.packb({
                        "type": "chat_video",
                        "frame": base64.b64encode(jpeg_bytes).decode("ascii"),
                        "frame_idx": frame_idx,
                        "fps": 25,
                        "chat_id": chat_id,
                        "server_ts": time.time(),
                    }))
                    frame_idx += 1
            except Exception:
                logger.warning("Failed to end face session cleanly")

    consumer_task = asyncio.create_task(_tts_consumer())
    video_task: asyncio.Task | None = None
    if video_pcm_queue is not None:
        video_task = asyncio.create_task(_video_producer())

    try:
        async for token in llm_client.stream_chat(session):
            full_response.append(token)

            # Send token to client for real-time display
            await ws.send_bytes(msgpack.packb({
                "type": "chat_token",
                "data": token,
                "chat_id": chat_id,
                "server_ts": time.time(),
            }))

            # Check for complete sentences
            sentences = chunker.add(token)
            for sentence in sentences:
                await sentence_queue.put(sentence)

        # Flush remaining text from chunker
        remainder = chunker.flush()
        if remainder:
            await sentence_queue.put(remainder)

        # Signal TTS consumer to stop after processing all sentences
        await sentence_queue.put(None)
        await consumer_task

        # Wait for video pipeline to finish (processes remaining audio)
        if video_task is not None:
            await video_task

        # Store assistant response in session history
        assistant_text = "".join(full_response)
        session.add_assistant_message(assistant_text)

        # Signal completion
        await ws.send_bytes(msgpack.packb({
            "type": "chat_done",
            "full_text": assistant_text,
            "chat_id": chat_id,
            "server_ts": time.time(),
        }))

        logger.info("Chat response to %s: %d chars, %d audio chunks", client_id, len(assistant_text), audio_chunk_count)

    except asyncio.CancelledError:
        logger.info("Chat pipeline cancelled for %s [%s]", client_id, chat_id)
        consumer_task.cancel()
        if video_task is not None:
            video_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer_task
        if video_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await video_task

        # ── Fix conversation history after cancellation ──
        # add_user_message() was called before streaming started, so the
        # session already contains the user turn.  Without an assistant
        # reply the history has consecutive user messages, which makes
        # Gemma 2 (and most chat models) produce 0 tokens.
        partial_text = "".join(full_response)
        if partial_text:
            # Save whatever was generated before the cancel
            session.add_assistant_message(partial_text)
            session.was_interrupted = True
            logger.info(
                "Saved partial response (%d chars) for %s [%s]",
                len(partial_text), client_id, chat_id,
            )
        else:
            # No tokens generated — roll back the user message
            if session.messages and session.messages[-1].role == "user":
                session.messages.pop()
                logger.info(
                    "Removed unanswered user message for %s [%s]",
                    client_id, chat_id,
                )

        try:
            await ws.send_bytes(msgpack.packb({
                "type": "chat_cancelled",
                "chat_id": chat_id,
                "server_ts": time.time(),
            }))
        except Exception:
            pass  # WebSocket may already be closed

    except Exception:
        logger.exception("Chat pipeline error for %s", client_id)
        consumer_task.cancel()
        if video_task is not None:
            video_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer_task
        if video_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await video_task
        try:
            await ws.send_bytes(msgpack.packb({
                "type": "error",
                "message": "Chat pipeline error",
                "chat_id": chat_id,
                "server_ts": time.time(),
            }))
        except Exception:
            pass  # WebSocket may already be closed
