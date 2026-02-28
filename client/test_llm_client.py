"""Minimal LLM streaming debug."""
import asyncio
import sys
import os
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from src.llm.client import ChatSession, LLMClient

URL = "http://172.81.127.36:29488/v1/chat/completions"


async def main() -> None:
    # Build the exact same payload the orchestrator builds
    session = ChatSession()
    session.add_user_message("Hello, how are you?")
    
    client = LLMClient(base_url="http://172.81.127.36:29488")
    payload = {
        "model": client.model,
        "messages": session.to_api_format(),
        "stream": True,
        "temperature": client.temperature,
        "max_tokens": client.max_tokens,
        "top_p": client.top_p,
    }
    print(f"Messages: {len(payload['messages'])}, system prompt: {len(payload['messages'][0]['content'])} chars")

    # Test 1: non-streaming with system prompt (merged into user msg)
    print(f"\n=== API format (first msg) ===")
    api = session.to_api_format()
    print(f"Roles: {[m['role'] for m in api]}")
    print(f"First msg length: {len(api[0]['content'])} chars")

    payload = {
        "model": client.model,
        "messages": api,
        "stream": True,
        "temperature": client.temperature,
        "max_tokens": 50,
        "top_p": client.top_p,
    }

    # Test: streaming with merged system prompt
    print("\n=== Streaming with merged system prompt ===")
    async with httpx.AsyncClient(timeout=120) as c:
        async with c.stream("POST", URL, json=payload) as r:
            print(f"Status: {r.status_code}")
            count = 0
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        print("  [DONE]")
                        break
                    import json
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        count += 1
                        print(f"  token {count}: {content!r}")
            print(f"\nTotal tokens: {count}")


asyncio.run(main())
