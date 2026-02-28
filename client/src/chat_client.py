"""Terminal chat client — interactive conversation with the avatar.

Sends text to the server over WebSocket, displays streaming LLM tokens
in real-time, and plays synthesized audio through speakers.

Usage:
    uv run python src/chat_client.py ws://HOST:PORT/ws
    uv run python src/chat_client.py ws://HOST:PORT/ws --no-audio
"""

import asyncio
import base64
import io
import logging
import sys
import time
import wave

import msgpack
import sounddevice as sd
import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger("avatar.chat_client")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DEFAULT_SERVER_URL = "ws://localhost:8000/ws"


class ChatClient:
    """Interactive terminal chat client with audio playback.

    Connects to the avatar server over WebSocket, sends user text,
    and handles streaming responses (tokens + audio).
    """

    def __init__(self, server_url: str, play_audio: bool = True) -> None:
        """Initialize chat client.

        Args:
            server_url: WebSocket URL of the avatar server.
            play_audio: Whether to play synthesized audio.
        """
        self.server_url = server_url
        self.play_audio = play_audio
        self._ws: ClientConnection | None = None
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

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

    async def send_chat(self, text: str) -> None:
        """Send a chat message and process streaming response.

        Displays LLM tokens in real-time and plays audio as it arrives.

        Args:
            text: User message text.
        """
        if not self._ws:
            logger.error("Not connected")
            return

        msg = {
            "type": "chat",
            "data": text,
            "ts": time.time(),
        }
        await self._ws.send(msgpack.packb(msg))

        # Process streaming response
        start_time = time.perf_counter()
        first_token_time: float | None = None
        first_audio_time: float | None = None
        token_count = 0
        audio_count = 0

        # Start background audio player
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        player_task = asyncio.create_task(self._play_audio_queue(audio_queue)) if self.play_audio else None

        print("\n\033[1mAvatar:\033[0m ", end="", flush=True)

        while True:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=120.0)
                response = msgpack.unpackb(raw, raw=False)
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
                        audio_count += 1
                        if first_audio_time is None:
                            first_audio_time = time.perf_counter()
                        # Send to player immediately — don't wait
                        await audio_queue.put(audio_data)

                elif msg_type == "chat_done":
                    print()  # newline after streaming tokens
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
        stats.append(f"Audio chunks: {audio_count}")
        print(f"\033[90m[{' | '.join(stats)}]\033[0m")

        # Signal audio player to finish and wait for it
        if player_task:
            await audio_queue.put(None)  # sentinel
            await player_task

    async def _play_audio_queue(self, queue: asyncio.Queue[bytes | None]) -> None:
        """Play audio chunks continuously without gaps.

        Uses a single ``sd.RawOutputStream`` across all chunks so that
        back-to-back writes produce seamless audio.  Each chunk is a
        self-contained WAV; we strip the header and feed raw PCM into
        the stream.

        Args:
            queue: Queue of WAV audio bytes (None = stop).
        """
        stream: sd.RawOutputStream | None = None
        loop = asyncio.get_event_loop()

        try:
            while True:
                audio_data = await queue.get()
                if audio_data is None:
                    break

                try:
                    with wave.open(io.BytesIO(audio_data), "rb") as wf:
                        if stream is None:
                            stream = sd.RawOutputStream(
                                samplerate=wf.getframerate(),
                                channels=wf.getnchannels(),
                                dtype="int16",
                            )
                            stream.start()
                        pcm = wf.readframes(wf.getnframes())

                    # write() blocks until the device buffer has room —
                    # run in executor to avoid blocking the event loop.
                    await loop.run_in_executor(None, stream.write, pcm)
                except Exception as e:
                    logger.warning("Audio playback failed: %s", e)
        finally:
            if stream is not None:
                # Drain the remaining audio in the device buffer
                await loop.run_in_executor(None, stream.stop)
                stream.close()

    async def run(self) -> None:
        """Main interactive chat loop."""
        if not await self.connect():
            return

        print("\n\033[1m=== Avatar Chat ===\033[0m")
        print("Type your message and press Enter. Type 'quit' or Ctrl+C to exit.\n")

        try:
            while True:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("\033[1mYou:\033[0m ")
                    )
                except EOFError:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    break

                await self.send_chat(user_input)

        except KeyboardInterrupt:
            print("\n")
        finally:
            if self._ws:
                await self._ws.close()
            print("\033[90m[Disconnected]\033[0m")


def main() -> None:
    """Entry point for the chat client."""
    import argparse

    parser = argparse.ArgumentParser(description="Avatar terminal chat client")
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
    args = parser.parse_args()

    client = ChatClient(
        server_url=args.server_url,
        play_audio=not args.no_audio,
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
