"""Quick latency test — sends a message and prints timing stats."""

import asyncio
import time

import msgpack
import websockets


async def test(message: str = "Привіт, хто ти?") -> None:
    """Send a chat message and measure first-token / first-audio latency."""
    ws = await websockets.connect("ws://localhost:8000/ws")
    msg = {"type": "chat", "data": message, "ts": time.time()}
    await ws.send(msgpack.packb(msg))

    t0 = time.perf_counter()
    ft: float | None = None
    fa: float | None = None
    tc = 0
    ac = 0

    while True:
        r = msgpack.unpackb(await asyncio.wait_for(ws.recv(), 30), raw=False)
        if r["type"] == "chat_token":
            tc += 1
            print(r["data"], end="", flush=True)
            if ft is None:
                ft = time.perf_counter()
        elif r["type"] == "chat_audio":
            ac += 1
            if fa is None:
                fa = time.perf_counter()
        elif r["type"] == "chat_done":
            break

    print()
    e = time.perf_counter() - t0
    ft_s = f"{ft - t0:.3f}s" if ft else "N/A"
    fa_s = f"{fa - t0:.3f}s" if fa else "N/A"
    print(f"Total: {e:.2f}s | First token: {ft_s} | First audio: {fa_s} | Tokens: {tc} | Audio: {ac}")
    await ws.close()


if __name__ == "__main__":
    asyncio.run(test())
