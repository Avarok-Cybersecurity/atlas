#!/usr/bin/env python3
"""Deterministic multi-turn warm-vs-cold divergence repro for the Marconi
warm-hit corruption (2026-06-10).

Drives N chained "turns" against a single running Atlas server at temp 0:
each turn appends the previous assistant reply plus a new user message, so
turn k>=2 produces a Marconi warm hit (intermediate or leaf restore) when
prefix caching is enabled. Run once against a prefix-caching server and once
against a no-caching server (or use --label to tag runs); diff the saved
transcripts to find the first divergent turn.

Usage:
  python3 warm_cold_chain_repro.py --base http://localhost:8888/v1 \
      --turns 6 --out /tmp/chain_warm.json [--max-tokens 300]
  # restart server without --enable-prefix-caching, then:
  python3 warm_cold_chain_repro.py --base http://localhost:8888/v1 \
      --turns 6 --out /tmp/chain_cold.json
  diff <(jq -r '.turns[].reply' /tmp/chain_warm.json) \
       <(jq -r '.turns[].reply' /tmp/chain_cold.json)

The user prompts intentionally include path-like strings (the live failure
corrupts file paths) and enough text per turn to span checkpoint blocks.
"""

import argparse
import json
import sys
import urllib.request

FILLER = (
    "Consider the module layout under /tmp/harness-proj/src with files "
    "main.rs, routes.rs, state.rs and tests in /tmp/harness-proj/tests. "
)

PROMPTS = [
    "We are planning a Rust axum webserver in /tmp/harness-proj. List the "
    "files you would create, with full absolute paths, one per line.",
    "Now write the exact cargo command to create it and repeat the full "
    "path /tmp/harness-proj/src/main.rs back to me three times, once per line.",
    "Describe the ping endpoint handler. Then repeat this path exactly once: "
    "/tmp/harness-proj/src/routes.rs",
    "What port env var should we read? Repeat the path "
    "/tmp/harness-proj/tests/integration.rs exactly twice, one per line.",
    "Summarize every file path mentioned so far, one absolute path per line, "
    "no duplicates.",
    "Repeat the project root path /tmp/harness-proj five times separated by "
    "spaces, then say DONE.",
    "Spell the path /tmp/harness-proj/src/state.rs forwards, then repeat it "
    "unmodified on the next line.",
    "Final check: list all paths from this conversation again, one per line.",
]


def chat(base, messages, max_tokens, timeout=180):
    body = {
        "model": "Qwen/Qwen3.6-35B-A3B-FP8",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8888/v1")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Long-ish system prompt + filler pushes the conversation across the
    # 4096-token intermediate-checkpoint boundary within a few turns.
    system = (
        "You are a precise assistant. Follow instructions exactly. "
        "Never abbreviate file paths. " + FILLER * 320
    )
    messages = [{"role": "system", "content": system}]
    turns = []
    for k in range(min(args.turns, len(PROMPTS))):
        messages.append({"role": "user", "content": PROMPTS[k] + "\n" + FILLER * 10})
        reply = chat(args.base, messages, args.max_tokens)
        messages.append({"role": "assistant", "content": reply})
        corrupt = (
            "/tmp/harness-proj" not in reply
            and k > 0
            or "//" in reply.replace("http://", "")
        )
        turns.append({"turn": k + 1, "prompt": PROMPTS[k][:60], "reply": reply})
        print(f"--- turn {k + 1} ({len(reply)} chars){' [SUSPECT]' if corrupt else ''}")
        print(reply[:400])
        sys.stdout.flush()

    with open(args.out, "w") as f:
        json.dump({"turns": turns}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
