#!/usr/bin/env python3
"""
bench_marconi.py — Marconi SSM state caching benchmark.

Measures TTFT improvement from SSM snapshot caching on multi-turn conversations.
Marconi saves SSM state at the radix tree leaf after each request. When a new
request EXTENDS a previous one (same prefix + new tokens), the snapshot is
restored and only the new tokens are processed through SSM layers.

Test strategy:
  1. Send a short initial request (primes cache, saves snapshot)
  2. Send a longer request whose prefix includes the full initial request
     → Marconi should skip SSM computation for the shared prefix
  3. Repeat with growing conversation context to show compounding benefit
"""

import json, sys, time, statistics
from urllib.request import Request, urlopen

URL   = "http://localhost:8888"
MODEL = "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4"

SYSTEM_PROMPT = (
    "You are a highly knowledgeable assistant specializing in science, "
    "technology, engineering, and mathematics. You provide clear, concise, "
    "and accurate answers. When explaining complex topics, break them down "
    "into simple steps. Always cite relevant formulas or principles when "
    "applicable. If a question is ambiguous, ask for clarification before "
    "answering. You should respond in English only. Your responses should "
    "be well-structured with proper formatting."
)

def send_chat(messages, max_tokens=30):
    """Send a chat completion request. Returns (ttft_ms, prompt_tokens, content)."""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = Request(f"{URL}/v1/chat/completions",
                  data=payload,
                  headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    resp = urlopen(req, timeout=60)
    t1 = time.monotonic()
    data = json.loads(resp.read())
    usage = data.get("usage", {})
    content = data["choices"][0]["message"]["content"]
    ttft = usage.get("time_to_first_token_ms", (t1 - t0) * 1000)
    prompt_tokens = usage.get("prompt_tokens", 0)
    tps = usage.get("tokens_per_second", 0)
    return ttft, prompt_tokens, content, tps


print("=" * 70)
print("Marconi SSM State Caching — Multi-Turn TTFT Benchmark")
print("=" * 70)
print(f"  Model: {MODEL}")
print(f"  System prompt: ~{len(SYSTEM_PROMPT)} chars")
print()

# ── Phase 1: Baseline — cold start ──
print("Phase 1: Cold start (no prefix cache)")
msgs_turn1 = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What is the speed of light in vacuum?"},
]
ttft1, ptok1, content1, tps1 = send_chat(msgs_turn1, max_tokens=80)
print(f"  Turn 1: TTFT={ttft1:.1f}ms, {ptok1} prompt tokens, {tps1:.0f} tok/s")
print(f"  Content: {content1[:80]!r}")
print()

# ── Phase 2: Multi-turn extension (Marconi hit expected) ──
# Each "turn" appends assistant + user messages, so the prefix grows.
# The radix tree should find the previous turn's leaf and restore SSM state.
print("Phase 2: Multi-turn conversation (Marconi SSM cache hit expected)")
print("  Each turn extends the previous, reusing cached SSM state.")
print()

# Simulate multi-turn: we include the assistant reply from turn 1
# and add a new user question. The tokenized prefix of turn 2 includes
# all of turn 1's tokens plus the assistant reply.
conversation = list(msgs_turn1)
turns = [
    ("What is E=mc²?", content1),
    ("How does GPS use relativity?", None),
    ("What about gravitational lensing?", None),
    ("Explain the twin paradox.", None),
]

ttfts_cold = [ttft1]
ttfts_warm = []
prompt_tokens_log = [ptok1]

for i, (user_msg, prev_reply) in enumerate(turns, 2):
    # Add the previous assistant reply and new user message
    if prev_reply is not None:
        conversation.append({"role": "assistant", "content": prev_reply})
    else:
        # Use a placeholder reply for subsequent turns
        conversation.append({"role": "assistant", "content": "I understand. Let me explain."})
    conversation.append({"role": "user", "content": user_msg})

    ttft, ptok, content, tps = send_chat(conversation, max_tokens=80)
    ttfts_warm.append(ttft)
    prompt_tokens_log.append(ptok)
    print(f"  Turn {i}: TTFT={ttft:.1f}ms, {ptok} prompt tokens, {tps:.0f} tok/s")
    print(f"    Content: {content[:70]!r}")

print()

# ── Phase 3: Different conversation (cold, no Marconi hit) ──
print("Phase 3: Different conversation (cold, no Marconi hit)")
msgs_diff = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Tell me about the history of computers."},
]
ttft_diff, ptok_diff, content_diff, _ = send_chat(msgs_diff, max_tokens=30)
print(f"  TTFT={ttft_diff:.1f}ms, {ptok_diff} prompt tokens (should be ~cold TTFT)")
print()

# ── Phase 4: Same conversation as Phase 1 (KV prefix cache hit) ──
print("Phase 4: Repeat Phase 1 (KV prefix cache hit, no SSM benefit expected)")
ttft_repeat, ptok_repeat, content_repeat, _ = send_chat(msgs_turn1, max_tokens=30)
print(f"  TTFT={ttft_repeat:.1f}ms, {ptok_repeat} prompt tokens")
print()

# ── Summary ──
print("=" * 70)
print("Summary")
print("=" * 70)
print(f"  Cold TTFT (turn 1, no cache):   {ttft1:.1f}ms ({ptok1} tokens)")
if ttfts_warm:
    warm_avg = statistics.mean(ttfts_warm)
    print(f"  Warm TTFT (turns 2-5, avg):     {warm_avg:.1f}ms (avg {statistics.mean(prompt_tokens_log[1:]):.0f} tokens)")
    print(f"  Prompt tokens grew:             {prompt_tokens_log[0]} → {prompt_tokens_log[-1]}")
    if ttft1 > 0:
        print(f"  TTFT ratio (warm/cold):         {warm_avg/ttft1:.2f}x")
    print()
    print("  Per-turn breakdown:")
    for i, (ttft, ptok) in enumerate(zip([ttft1] + ttfts_warm, prompt_tokens_log)):
        marker = " ← cold" if i == 0 else " ← Marconi" if i > 0 else ""
        ms_per_tok = ttft / ptok if ptok > 0 else 0
        print(f"    Turn {i+1}: {ttft:7.1f}ms  {ptok:4d} tokens  {ms_per_tok:.2f} ms/tok{marker}")
print(f"  Different conversation (cold):  {ttft_diff:.1f}ms ({ptok_diff} tokens)")
print(f"  Repeated turn 1 (KV only):      {ttft_repeat:.1f}ms ({ptok_repeat} tokens)")
