"""Voice conversation client — speak with the avatar using your microphone.

Captures speech via Moonshine Voice ASR, sends transcribed text to the
server over WebSocket, displays streaming LLM tokens, and plays
synthesized audio through speakers.

State machine:
    IDLE → LISTENING → PROCESSING → SPEAKING → LISTENING → ...

Barge-in: The microphone stays active during playback. If the user
starts speaking while the avatar is talking, playback is cancelled
immediately and the server pipeline is stopped. The laptop's built-in
echo cancellation prevents speaker output from being picked up as
user input.

Usage:
    uv run python src/voice_client.py ws://HOST:PORT/ws
    uv run python src/voice_client.py ws://HOST:PORT/ws --no-audio
    uv run python src/voice_client.py ws://HOST:PORT/ws --language en
"""

import asyncio
import base64
import contextlib
import logging
import re
import sys
import time
from pathlib import Path

# Ensure the client/src package is importable when running as a script
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import msgpack
import websockets
from websockets.asyncio.client import ClientConnection

from asr.transcriber import SpeechTranscriber
from audio.playback import AudioPlayer

logger = logging.getLogger("avatar.voice_client")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DEFAULT_SERVER_URL = "ws://localhost:8000/ws"

# Barge-in pre-filter settings
BACKCHANNEL_PATTERN = re.compile(
    r"^(mm-?hm|mhm|uh-?huh|yeah|yep|okay|ok|sure|right|hm+|oh|ah|got it|i see)\.?$",
    re.IGNORECASE,
)
MIN_INTERRUPTION_WORDS = 2
MIN_INTERRUPTION_DURATION = 0.5  # seconds


def _is_backchannel(text: str) -> bool:
    """Check if text is a backchannel/filler that should not interrupt."""
    return bool(BACKCHANNEL_PATTERN.match(text.strip()))


class VoiceClient:
    """Interactive voice conversation client.

    Captures speech from the microphone, sends it to the avatar server,
    and plays the synthesized audio response.

    Args:
        server_url: WebSocket URL of the avatar server.
        play_audio: Whether to play synthesized audio.
        language: ASR language code (e.g., "en").
        device: sounddevice input device index or name (None = default).
    """

    def __init__(
        self,
        server_url: str,
        play_audio: bool = True,
        language: str = "en",
        device: int | str | None = None,
        use_smart_turn: bool = True,
    ) -> None:
        """Initialize voice client.

        Args:
            server_url: WebSocket URL of the avatar server.
            play_audio: Whether to play synthesized audio.
            language: ASR language code (e.g., "en").
            device: sounddevice input device index or name (None = default).
            use_smart_turn: Whether to use Smart Turn end-of-turn detection.
        """
        self.server_url = server_url
        self.play_audio = play_audio
        self._ws: ClientConnection | None = None
        self._transcriber = SpeechTranscriber(language=language, device=device)
        self._player = AudioPlayer()
        self._running = False
        self._use_smart_turn = use_smart_turn
        self._smart_turn = None  # SmartTurnAnalyzer, initialized in run()
        self._chat_seq = 0  # Incrementing chat ID for message correlation

    async def connect(self) -> bool:
        """Establish WebSocket connection.

        Returns:
            True if connection succeeded.
        """
        try:
            logger.info("Connecting to %s ...", self.server_url)
            self._ws = await websockets.connect(self.server_url)
            logger.info("Connected to server")
            return True
        except Exception as e:
            logger.error("Connection failed: %s", e)
            return False

    async def _initialize_asr(self) -> bool:
        """Download model and initialize the speech transcriber.

        Returns:
            True if initialization succeeded.
        """
        try:
            await self._transcriber.initialize()
            return True
        except Exception as e:
            logger.error("ASR initialization failed: %s", e)
            return False

    async def _initialize_smart_turn(self) -> bool:
        """Initialize Smart Turn end-of-turn detection.

        Downloads the Smart Turn v3.2 ONNX model on first run and loads
        it for inference.  Failures are non-fatal — the client continues
        without end-of-turn detection.

        Returns:
            True if initialization succeeded or Smart Turn is disabled.
        """
        if not self._use_smart_turn:
            return True
        try:
            from asr.smart_turn import SmartTurnAnalyzer

            loop = asyncio.get_event_loop()
            self._smart_turn = await loop.run_in_executor(None, SmartTurnAnalyzer)
            logger.info("Smart Turn analyzer initialized")
            return True
        except ImportError:
            logger.warning(
                "Smart Turn unavailable (install transformers + onnxruntime). "
                "Continuing without end-of-turn detection."
            )
            return True
        except Exception as e:
            logger.warning("Smart Turn init failed: %s. Continuing without it.", e)
            return True

    def _should_filter_interruption(self, text: str) -> bool:
        """Check if a potential barge-in should be filtered out.

        Returns True for backchannels, very short utterances, or
        brief speech that doesn't warrant interrupting the avatar.
        """
        # Backchannel detection (e.g., "mhm", "yeah", "okay")
        if _is_backchannel(text):
            logger.info("Filtered backchannel: '%s'", text)
            return True

        # Minimum word count
        words = text.split()
        if len(words) < MIN_INTERRUPTION_WORDS:
            logger.info("Filtered short interruption (%d words): '%s'", len(words), text)
            return True

        # Minimum speech duration
        duration = self._transcriber.last_speech_duration
        if 0 < duration < MIN_INTERRUPTION_DURATION:
            logger.info("Filtered brief speech (%.2fs): '%s'", duration, text)
            return True

        return False

    async def _process_chat(self, text: str) -> str | None:
        """Send text to server and handle streaming response with barge-in.

        Displays LLM tokens in real-time and plays audio as it arrives.
        The mic stays active — if the user starts speaking, playback and
        the server pipeline are cancelled immediately.

        Args:
            text: Transcribed user speech to send.

        Returns:
            The barge-in transcription if the user interrupted, or None
            if the response completed normally.
        """
        if not self._ws:
            logger.error("Not connected")
            return None

        self._chat_seq += 1
        chat_id = str(self._chat_seq)

        msg = {
            "type": "chat",
            "data": text,
            "chat_id": chat_id,
            "ts": time.time(),
        }
        await self._ws.send(msgpack.packb(msg))

        # Timing metrics
        start_time = time.perf_counter()
        first_token_time: float | None = None
        first_audio_time: float | None = None
        token_count = 0

        # Start background audio player
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        player_task = (
            asyncio.create_task(self._player.play_queue(audio_queue))
            if self.play_audio
            else None
        )

        # Start barge-in detector — waits for user speech in parallel
        barge_in_text: str | None = None
        barge_in_task = asyncio.create_task(self._transcriber.get_next_line())

        print("\n\033[1mAvatar:\033[0m ", end="", flush=True)

        interrupted = False
        while True:
            try:
                # Race: next server message vs. user speech (barge-in)
                recv_task = asyncio.ensure_future(
                    asyncio.wait_for(self._ws.recv(), timeout=120.0)
                )

                done, _ = await asyncio.wait(
                    [recv_task, barge_in_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # ── Barge-in detected ──
                if barge_in_task in done:
                    recv_task.cancel()
                    barge_in_text = barge_in_task.result()

                    # Pre-filter: ignore backchannels, short/brief utterances
                    if self._should_filter_interruption(barge_in_text):
                        barge_in_task = asyncio.create_task(
                            self._transcriber.get_next_line()
                        )
                        continue

                    interrupted = True
                    print(f"\n\033[90m[interrupted]\033[0m")
                    logger.info("Barge-in detected: '%s'", barge_in_text)

                    # 1) Stop audio playback immediately
                    self._player.cancel()

                    # 2) Tell server to cancel LLM + TTS pipeline
                    if self._ws:
                        cancel_msg = {"type": "chat_cancel", "chat_id": chat_id, "ts": time.time()}
                        await self._ws.send(msgpack.packb(cancel_msg))

                    # 3) Drain remaining server messages until chat_done/chat_cancelled
                    await self._drain_server_response()
                    break

                # ── Normal server message ──
                if recv_task in done:
                    raw = recv_task.result()
                    response = msgpack.unpackb(raw, raw=False)

                    # Ignore stale messages from previous chat requests
                    msg_chat_id = response.get("chat_id", chat_id)
                    if msg_chat_id != chat_id:
                        logger.debug(
                            "Ignoring stale message (type=%s, chat_id=%s, expected=%s)",
                            response.get("type"), msg_chat_id, chat_id,
                        )
                        continue

                    msg_type = response.get("type", "")

                    if msg_type == "chat_token":
                        token = response.get("data", "")
                        print(token, end="", flush=True)
                        token_count += 1
                        if first_token_time is None:
                            first_token_time = time.perf_counter()

                    elif msg_type == "chat_audio":
                        audio_b64 = response.get("data", "")
                        if audio_b64:
                            audio_data = base64.b64decode(audio_b64)
                            if first_audio_time is None:
                                first_audio_time = time.perf_counter()
                            await audio_queue.put(audio_data)

                    elif msg_type == "chat_done":
                        print()  # newline after streaming tokens
                        break

                    elif msg_type == "chat_cancelled":
                        print("\n\033[90m[cancelled]\033[0m")
                        break

                    elif msg_type == "error":
                        print(f"\n\033[31mError: {response.get('message', 'Unknown error')}\033[0m")
                        break

            except asyncio.TimeoutError:
                print("\n\033[31mTimeout waiting for response\033[0m")
                break

        elapsed = time.perf_counter() - start_time

        # Print timing stats
        stats: list[str] = [f"Total: {elapsed:.1f}s"]
        if first_token_time is not None:
            stats.append(f"First token: {first_token_time - start_time:.2f}s")
        if first_audio_time is not None:
            stats.append(f"First audio: {first_audio_time - start_time:.2f}s")
        stats.append(f"Tokens: {token_count}")
        if interrupted:
            stats.append("INTERRUPTED")
        print(f"\033[90m[{' | '.join(stats)}]\033[0m")

        # Clean up audio player
        if player_task:
            if interrupted:
                # Player was already cancelled, just await to collect
                with contextlib.suppress(Exception):
                    await player_task
            else:
                await audio_queue.put(None)  # sentinel
                await player_task

        # Clean up barge-in task if response finished without interruption
        if not barge_in_task.done():
            barge_in_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await barge_in_task

        return barge_in_text

    async def _drain_server_response(self) -> None:
        """Drain remaining server messages after a barge-in cancel.

        Reads and discards messages until chat_done or chat_cancelled,
        so the WebSocket is clean for the next turn.
        """
        if not self._ws:
            return
        try:
            while True:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                response = msgpack.unpackb(raw, raw=False)
                msg_type = response.get("type", "")
                if msg_type in ("chat_done", "chat_cancelled"):
                    break
        except (asyncio.TimeoutError, Exception):
            pass  # Server may have already finished

    async def _wait_for_complete_turn(self, initial_text: str) -> str:
        """Accumulate speech segments until Smart Turn says turn is complete.

        Uses the Smart Turn v3.2 ONNX model to analyze audio prosody and
        determine if the user has finished their turn.  Falls back to a
        3-second silence timeout if the model says "incomplete".

        Args:
            initial_text: First transcribed line from Moonshine.

        Returns:
            Accumulated text from all speech segments in the turn.
        """
        assert self._smart_turn is not None
        accumulated = initial_text

        while self._running:
            audio = self._transcriber.get_audio_buffer()
            is_complete, prob = await asyncio.get_event_loop().run_in_executor(
                None, self._smart_turn.is_turn_complete, audio,
            )
            logger.info(
                "Smart Turn: complete=%s (p=%.2f) text='%s'",
                is_complete, prob, accumulated,
            )

            if is_complete:
                return accumulated

            # Wait for more speech with 3s fallback timeout
            try:
                next_line = await asyncio.wait_for(
                    self._transcriber.get_next_line(), timeout=3.0,
                )
                accumulated += " " + next_line
                print(f" {next_line}", end="", flush=True)
            except asyncio.TimeoutError:
                logger.info("Smart Turn: timeout, treating as complete")
                return accumulated

        return accumulated

    async def run(self) -> None:
        """Main voice conversation loop.

        1. Initialize ASR + connect to server
        2. Listen for speech
        3. Send transcription → receive response → play audio
        4. Repeat
        """
        # Initialize ASR
        print("\033[90m[Initializing speech recognition...]\033[0m")
        if not await self._initialize_asr():
            return

        # Initialize Smart Turn end-of-turn detection
        if self._use_smart_turn:
            print("\033[90m[Initializing Smart Turn...]\033[0m")
            await self._initialize_smart_turn()

        # Connect to server
        if not await self.connect():
            return

        self._running = True

        print("\n\033[1m=== Avatar Voice Chat ===\033[0m")
        print("Speak into your microphone. Press Ctrl+C to exit.")
        print("Type 'quit' and Enter to exit.\n")

        # Start mic capture
        self._transcriber.start()

        # Also start a background task for keyboard input (quit command)
        quit_event = asyncio.Event()
        quit_task = asyncio.create_task(self._quit_listener(quit_event))

        # If the user barge-in interrupts, their speech carries over
        pending_text: str | None = None

        try:
            while self._running and not quit_event.is_set():
                # LISTENING state — mic is always active
                print("\033[90m[Listening...]\033[0m", flush=True)

                # If we have a barge-in transcription from previous turn,
                # use it directly instead of waiting for new speech
                if pending_text is not None:
                    text = pending_text
                    pending_text = None
                else:
                    # Wait for either a speech line or quit
                    line_task = asyncio.create_task(self._transcriber.get_next_line())
                    done, _pending = await asyncio.wait(
                        [line_task, quit_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if quit_event.is_set() or line_task not in done:
                        line_task.cancel()
                        break

                    text = line_task.result()

                # Smart Turn: accumulate until turn is complete
                if self._smart_turn is not None:
                    print(f"\033[1mYou:\033[0m {text}", end="", flush=True)
                    text = await self._wait_for_complete_turn(text)
                    print()  # newline after accumulated text
                else:
                    print(f"\033[1mYou:\033[0m {text}")

                # PROCESSING / SPEAKING — mic stays active for barge-in
                barge_in_text = await self._process_chat(text)

                # If the user interrupted, their speech is already captured
                # — feed it straight into the next turn
                if barge_in_text:
                    pending_text = barge_in_text
                    # Clear any stale transcriptions from speaker echo
                    self._transcriber.clear_queue()
                    self._transcriber.clear_audio_buffer()

        except KeyboardInterrupt:
            print("\n")
        finally:
            self._running = False
            quit_task.cancel()
            self._transcriber.close()
            if self._ws:
                await self._ws.close()
            print("\033[90m[Disconnected]\033[0m")

    async def _quit_listener(self, quit_event: asyncio.Event) -> None:
        """Listen for 'quit' typed on stdin."""
        loop = asyncio.get_event_loop()
        try:
            while not quit_event.is_set():
                user_input = await loop.run_in_executor(
                    None, lambda: input()
                )
                if user_input.strip().lower() in ("quit", "exit", "q"):
                    quit_event.set()
                    return
        except (EOFError, KeyboardInterrupt):
            quit_event.set()


def main() -> None:
    """Entry point for the voice client."""
    import argparse

    parser = argparse.ArgumentParser(description="Avatar voice conversation client")
    parser.add_argument(
        "server_url",
        nargs="?",
        default=DEFAULT_SERVER_URL,
        help=f"WebSocket URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio playback (text-only mode)",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="ASR language code (default: en)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input audio device index (default: system default)",
    )
    parser.add_argument(
        "--no-smart-turn",
        action="store_true",
        help="Disable Smart Turn end-of-turn detection",
    )
    args = parser.parse_args()

    client = VoiceClient(
        server_url=args.server_url,
        play_audio=not args.no_audio,
        language=args.language,
        device=args.device,
        use_smart_turn=not args.no_smart_turn,
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
