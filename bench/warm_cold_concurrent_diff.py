#!/usr/bin/env python3
"""Concurrent agentic multi-turn WARM-vs-COLD diff.

The serve.rs warning's symptom is ~1-in-3 agentic multi-turn runs under
concurrency. The async_chkpt.rs live-state-invariant note names "a second
concurrent request" as a non-verify successor that can read dirty live SSM
state after an MTP reject. This harness drives CONCURRENT multi-turn chats so
MTP verify-commit (secondary stream, no-wait) on one sequence interleaves with
Marconi snapshot reads/restores on another sharing the same batch.

Method:
  1. Establish a per-case COLD reference SERIALLY (no concurrency, fresh-ish
     cache) — the proven-correct greedy turn-2 answer.
  2. Replay all cases' turn-2 requests CONCURRENTLY (many in flight at once),
     forcing warm Marconi restores to overlap with other sequences' MTP
     commits. Compare each concurrent turn-2 answer to its serial COLD ref.
  Greedy ⇒ must be bit-identical. Divergence = the concurrency-coupled
  warm-restore corruption.
Repeats R rounds to catch the intermittent (~1/3) failure.
"""
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("ATLAS_URL", "http://localhost:8888")
MODEL = os.environ.get("ATLAS_MODEL", "model")
T1_MAX = int(os.environ.get("T1_MAX", "320"))
T2_MAX = int(os.environ.get("T2_MAX", "160"))
ROUNDS = int(os.environ.get("ROUNDS", "4"))
CONC = int(os.environ.get("CONC", "6"))

SYS = ("You are a helpful assistant. Answer thoroughly and in detail.")

# Reuse the chat cases (distinct prefixes → distinct sequences in the batch).
from warm_cold_chat_diff import CASES  # noqa: E402


def chat(messages, max_tokens):
    body = {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "logprobs": True, "top_logprobs": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    c = r["choices"][0]
    u = r.get("usage", {})
    lp = (c.get("logprobs") or {}).get("content") or []
    return {
        "text": c["message"].get("content") or "",
        "tokens": [e.get("token") for e in lp] if lp else None,
        "finish_reason": c.get("finish_reason"),
        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "prompt_tokens": u.get("prompt_tokens", 0),
    }


def first_div(a, b):
    if a.get("tokens") and b.get("tokens"):
        ta, tb = a["tokens"], b["tokens"]
        for i in range(min(len(ta), len(tb))):
            if ta[i] != tb[i]:
                return i
        return -1 if len(ta) == len(tb) else min(len(ta), len(tb))
    return -1 if a["text"] == b["text"] else 0


def build_t2(u1, u2):
    a1 = chat([{"role": "system", "content": SYS},
               {"role": "user", "content": u1}], T1_MAX)
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": a1["text"]},
            {"role": "user", "content": u2}]


def main():
    n = int(os.environ.get("N", str(len(CASES))))
    cases = (CASES * ((n // len(CASES)) + 1))[:n]
    print(f"=== Concurrent WARM-vs-COLD: {n} cases × {ROUNDS} rounds, "
          f"conc={CONC} (T1_MAX={T1_MAX}, T2_MAX={T2_MAX}) ===", flush=True)

    # Phase 1: build t2 message arrays + serial COLD references.
    t2s, cold = [], []
    for i, (u1, u2) in enumerate(cases):
        msgs = build_t2(u1, u2)
        t2s.append(msgs)
        c = chat(msgs, T2_MAX)   # first send = cold reference
        cold.append(c)
    print(f"  built {n} cold refs (cached avg "
          f"{sum(c['cached'] for c in cold)//max(n,1)})", flush=True)

    # Phase 2: concurrent warm replays, R rounds.
    fails = 0
    total = 0
    for rnd in range(ROUNDS):
        with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
            futs = {ex.submit(chat, t2s[i], T2_MAX): i for i in range(n)}
            warm = {}
            for fut in cf.as_completed(futs):
                warm[futs[fut]] = fut.result()
        for i in range(n):
            total += 1
            div = first_div(cold[i], warm[i])
            if div != -1:
                fails += 1
                print(f"  [FAIL] round{rnd} C{i}: div_idx={div} "
                      f"warm_cached={warm[i]['cached']}/{warm[i]['prompt_tokens']}",
                      flush=True)
                print(f"      COLD: {cold[i]['text']!r}", flush=True)
                print(f"      WARM: {warm[i]['text']!r}", flush=True)
        print(f"  round {rnd}: {n - sum(1 for i in range(n) if first_div(cold[i], warm[i]) != -1)}/{n} match",
              flush=True)
    print(f"\n=== {total - fails}/{total} PASS, {fails} FAIL ===", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
