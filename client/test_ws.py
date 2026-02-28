"""Quick WebSocket debug test."""
import asyncio
import msgpack
import time
import websockets
import httpx


async def test():
    # First test: direct LLM via external port
    print("--- Direct LLM test (external port) ---")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://31.154.161.115:16441/v1/chat/completions",
            json={
                "model": "mamaylm",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 10,
                "stream": True,
            },
            timeout=30.0,
        )
        print(f"LLM direct status: {resp.status_code}")
        print(f"LLM direct body: {resp.text[:200]}")

    # Second test: ping via websocket
    print("\n--- WebSocket ping test ---")
    ws = await websockets.connect("ws://31.154.161.115:16527/ws")
    ping_msg = {"type": "ping", "ts": time.time()}
    await ws.send(msgpack.packb(ping_msg))
    raw = await asyncio.wait_for(ws.recv(), timeout=10)
    r = msgpack.unpackb(raw, raw=False)
    print(f"Ping response: {r}")

    # Third test: chat
    print("\n--- Chat test ---")
    msg = {"type": "chat", "data": "Hello, how are you?", "ts": time.time()}
    await ws.send(msgpack.packb(msg))
    start = time.perf_counter()
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=120)
        r = msgpack.unpackb(raw, raw=False)
        elapsed = time.perf_counter() - start
        t = r.get("type", "")
        d = str(r.get("data", ""))[:80]
        ft = str(r.get("full_text", ""))[:120]
        m = str(r.get("message", ""))[:120]
        print(f"[{elapsed:.2f}s] type={t} data={d!r} full_text={ft!r} message={m!r}")
        if t in ("chat_done", "error"):
            break
    await ws.close()


asyncio.run(test())
