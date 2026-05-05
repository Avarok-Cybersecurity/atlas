#!/usr/bin/env python3
"""Minimal Atlas client using the OpenAI Python SDK.

Atlas exposes an OpenAI-compatible /v1/chat/completions endpoint, so
the official `openai` Python package works unchanged. Just point it at
your Atlas server and use any model that's been loaded.

Install:
    pip install openai

Run (assuming Atlas is serving on localhost:8888):
    python3 examples/python/chat_completions.py
"""
import os

from openai import OpenAI

# Atlas doesn't enforce auth by default. The SDK still requires an api_key
# field, so we pass a placeholder. If you've enabled auth in front of Atlas
# (reverse proxy, etc.), set OPENAI_API_KEY in the environment.
client = OpenAI(
    base_url=os.environ.get("ATLAS_BASE_URL", "http://localhost:8888/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "atlas-no-auth"),
)

MODEL = os.environ.get(
    "ATLAS_MODEL",
    # Override via ATLAS_MODEL env var; this default matches the
    # model-id Atlas reports via /v1/models for the most common
    # NVFP4 35B-A3B deployment.
    "Sehyo/Qwen3.5-35B-A3B-NVFP4",
)


def blocking_example() -> None:
    """One-shot blocking completion."""
    print("=== Blocking ===")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Write a 3-sentence story about a cat."},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    print(response.choices[0].message.content)
    print()
    print(
        f"  prompt_tokens={response.usage.prompt_tokens} "
        f"completion_tokens={response.usage.completion_tokens}"
    )


def streaming_example() -> None:
    """Streaming completion (SSE)."""
    print("\n=== Streaming ===")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Count from 1 to 10, comma-separated."},
        ],
        max_tokens=64,
        temperature=0.0,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def tool_call_example() -> None:
    """Tool-calling — Atlas auto-detects the model's tool format."""
    print("\n=== Tool call ===")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "units": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        max_tokens=200,
        temperature=0.0,
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  → {tc.function.name}({tc.function.arguments})")
    else:
        print(f"  (no tool call) {msg.content}")


if __name__ == "__main__":
    blocking_example()
    streaming_example()
    tool_call_example()
