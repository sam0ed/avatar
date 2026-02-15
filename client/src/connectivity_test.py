"""Avatar client — WebSocket connectivity test.

Stage 0: Connects to the server, sends ping messages, measures RTT.
Run with: uv run python src/connectivity_test.py [server_url]
"""

import asyncio
import logging
import sys
import time

import msgpack
import websockets

logger = logging.getLogger("avatar.client")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DEFAULT_SERVER_URL = "ws://localhost:8000/ws"
PING_COUNT = 10
PING_INTERVAL = 0.5


async def measure_rtt(server_url: str, count: int = PING_COUNT) -> None:
    """Connect to the server and measure WebSocket round-trip time.

    Args:
        server_url: WebSocket URL of the avatar server.
        count: Number of ping messages to send.
    """
    logger.info("Connecting to %s ...", server_url)

    async with websockets.connect(server_url) as ws:
        logger.info("Connected! Sending %d pings...", count)

        rtts: list[float] = []

        for i in range(count):
            send_ts = time.time()
            msg = {"type": "ping", "ts": send_ts}
            await ws.send(msgpack.packb(msg))

            raw = await ws.recv()
            recv_ts = time.time()

            response = msgpack.unpackb(raw, raw=False)
            rtt_ms = (recv_ts - send_ts) * 1000
            rtts.append(rtt_ms)

            logger.info(
                "Ping %d/%d — RTT: %.1f ms (server_ts: %.3f)",
                i + 1,
                count,
                rtt_ms,
                response.get("server_ts", 0),
            )

            await asyncio.sleep(PING_INTERVAL)

        # Summary
        avg_rtt = sum(rtts) / len(rtts)
        min_rtt = min(rtts)
        max_rtt = max(rtts)

        logger.info("--- Ping Summary ---")
        logger.info("  Pings sent: %d", count)
        logger.info("  Avg RTT:    %.1f ms", avg_rtt)
        logger.info("  Min RTT:    %.1f ms", min_rtt)
        logger.info("  Max RTT:    %.1f ms", max_rtt)

        if avg_rtt < 100:
            logger.info("  ✓ RTT is within target (<100ms)")
        else:
            logger.warning("  ✗ RTT exceeds target (>100ms) — consider a closer datacenter")

        # Echo test
        echo_msg = {"type": "echo", "data": "Hello from Avatar client!"}
        await ws.send(msgpack.packb(echo_msg))
        raw = await ws.recv()
        echo_response = msgpack.unpackb(raw, raw=False)
        logger.info("Echo test: sent='%s' received='%s'", echo_msg["data"], echo_response.get("data", ""))

        if echo_response.get("data") == echo_msg["data"]:
            logger.info("  ✓ Echo test passed")
        else:
            logger.error("  ✗ Echo test failed")


def main() -> None:
    """Entry point for connectivity test."""
    server_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERVER_URL
    asyncio.run(measure_rtt(server_url))


if __name__ == "__main__":
    main()
