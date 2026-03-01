"""Avatar client orchestrator — main pipeline coordinator.

Stage 0–3: WebSocket connection + ASR + voice conversation pipeline.

The orchestrator manages the high-level state machine:
    IDLE → LISTENING → PROCESSING → SPEAKING → LISTENING → ...

For voice conversation, use VoiceClient directly:
    uv run python src/voice_client.py ws://HOST:PORT/ws

This module provides the state machine and connection management
that will grow to encompass video/face animation in later stages.

Run with: uv run python src/orchestrator.py [server_url]
"""

import asyncio
import logging
import sys
import time
from enum import Enum, auto

import msgpack
import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger("avatar.orchestrator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DEFAULT_SERVER_URL = "ws://localhost:8000/ws"

# Connection settings
RECONNECT_DELAY = 3.0
HEARTBEAT_INTERVAL = 5.0


class PipelineState(Enum):
    """State machine for the conversation pipeline."""

    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


class AvatarOrchestrator:
    """Coordinates the local avatar pipeline.

    Manages WebSocket connection to the Vast.ai server,
    local audio capture/playback, and state transitions.
    """

    def __init__(self, server_url: str) -> None:
        """Initialize orchestrator.

        Args:
            server_url: WebSocket URL of the avatar server.
        """
        self.server_url = server_url
        self.state = PipelineState.IDLE
        self._ws: ClientConnection | None = None
        self._running = False

    async def connect(self) -> None:
        """Establish WebSocket connection with auto-reconnect."""
        while self._running:
            try:
                logger.info("Connecting to %s ...", self.server_url)
                self._ws = await websockets.connect(self.server_url)
                logger.info("Connected to server")
                self.state = PipelineState.LISTENING
                return
            except (OSError, websockets.exceptions.WebSocketException) as exc:
                logger.warning("Connection failed: %s — retrying in %.0fs", exc, RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)

    async def _heartbeat(self) -> None:
        """Send periodic heartbeat pings to keep connection alive."""
        while self._running and self._ws:
            try:
                msg = {"type": "ping", "ts": time.time()}
                await self._ws.send(msgpack.packb(msg))
                raw = await self._ws.recv()
                response = msgpack.unpackb(raw, raw=False)
                rtt = (time.time() - msg["ts"]) * 1000
                logger.debug("Heartbeat RTT: %.1f ms", rtt)
            except Exception:
                logger.warning("Heartbeat failed — reconnecting")
                break
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def run(self) -> None:
        """Main orchestrator loop."""
        self._running = True
        logger.info("Avatar orchestrator starting")

        while self._running:
            await self.connect()

            if not self._ws:
                continue

            try:
                # Run heartbeat alongside future pipeline tasks
                heartbeat_task = asyncio.create_task(self._heartbeat())

                # Stage 0: just keep the connection alive
                # Future stages will add:
                #   - Audio capture task (mic → VAD → ASR)
                #   - Server communication task (text → LLM → TTS → face)
                #   - Audio playback task (TTS audio → VB-Cable)
                #   - Video output task (face frames → pyvirtualcam)

                logger.info("Pipeline ready — state: %s", self.state.name)
                logger.info("Press Ctrl+C to stop")

                await heartbeat_task

            except asyncio.CancelledError:
                logger.info("Pipeline cancelled")
                break
            except Exception:
                logger.exception("Pipeline error — reconnecting")
            finally:
                if self._ws:
                    await self._ws.close()
                    self._ws = None

        logger.info("Orchestrator stopped")

    async def shutdown(self) -> None:
        """Gracefully shut down the orchestrator."""
        self._running = False
        if self._ws:
            await self._ws.close()


def main() -> None:
    """Entry point for the avatar orchestrator."""
    server_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERVER_URL

    orchestrator = AvatarOrchestrator(server_url)

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
