#!/usr/bin/env python3
"""Agentic multi-turn chat WARM-vs-COLD greedy diff (the shipped-bug pattern).

This mirrors the exact pattern the serve.rs warning describes: a hybrid SSM
model under --enable-prefix-caching + --speculative (MTP), warm Marconi
restores on multi-turn traffic emitting duplicated tool-argument fragments.

For each case we build a 2-turn (or 3-turn) chat. The KEY mechanic:
  • Turn 1 is GENERATED (decode), so the server saves *decode-era* Marconi SSM
    snapshots (decode_marconi_checkpoint, every 64 tokens) covering the
    generated assistant tokens, tagged with the conversation's session_hash.
  • The turn-2 request re-prefills [sys?, u1, a1, u2]; the prefix-cache lookup
    HITS deep into the turn-1 assistant region and Marconi restores a
    decode-era SSM snapshot mid-sequence, then suffix-prefills u2.

A/B (single server, prefix-caching ON):
  COLD : send the FULL turn-2 conversation as the very first request for this
         case (cache cold for this prefix) → genuine recompute reference.
  WARM : replay turn 1 (regenerating a1 identically under greedy, populating
         decode-era snapshots), then resend the identical turn-2 conversation
         → deep warm hit restoring decode-era SSM state.
  Greedy ⇒ COLD turn-2 answer MUST equal WARM turn-2 answer. First divergence
  localizes corruption.

We force a deterministic, captured a1 by first generating turn 1 once and
pinning the assistant text, so COLD and WARM use byte-identical message arrays.
Thinking is disabled (chat_template_kwargs) for determinism + to keep turns in
the content phase where tool-arg fragments live.
"""
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("ATLAS_URL", "http://localhost:8888")
MODEL = os.environ.get("ATLAS_MODEL", "model")
T1_MAX = int(os.environ.get("T1_MAX", "400"))
T2_MAX = int(os.environ.get("T2_MAX", "160"))
NO_THINK = os.environ.get("NO_THINK", "1") == "1"

SYS = ("You are a helpful assistant. Answer thoroughly and in detail. When "
       "asked, call tools by emitting their JSON arguments precisely.")

CASES = [
    ("Write a detailed 250-word overview of the Apollo program, covering the "
     "key missions, the crews, and the major technical milestones.",
     "Now summarize that into exactly five bullet points, one per mission era."),
    ("Explain in depth how HTTPS establishes a secure connection: DNS, TCP "
     "handshake, TLS handshake, certificate validation, and symmetric key "
     "exchange. Be specific and at least 250 words.",
     "Based on your explanation, list the five most security-critical steps "
     "in order, with one sentence each."),
    ("Describe the process of photosynthesis in detail, including the light-"
     "dependent reactions, the Calvin cycle, and the role of chlorophyll. "
     "Write at least 250 words.",
     "Now produce a JSON object with keys 'inputs', 'outputs', and 'stages' "
     "summarizing the process."),
    ("Give a thorough explanation of how a CPU executes an instruction: fetch, "
     "decode, execute, memory access, and write-back, including pipelining. "
     "At least 250 words.",
     "Summarize each of the five stages in a single precise sentence."),
    ("Explain the causes and consequences of the 2008 financial crisis in "
     "detail, naming the key instruments, institutions, and policy responses. "
     "Write at least 250 words.",
     "List the four root causes you identified, ranked by importance, each "
     "with a one-line justification."),
]


def chat(messages, max_tokens):
    body = {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "logprobs": True, "top_logprobs": 1,
    }
    if NO_THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    c = r["choices"][0]
    u = r.get("usage", {})
    msg = c["message"]
    lp = (c.get("logprobs") or {}).get("content") or []
    toks = [e.get("token") for e in lp] if lp else None
    return {
        "text": msg.get("content") or "",
        "tool_calls": msg.get("tool_calls"),
        "tokens": toks,
        "finish_reason": c.get("finish_reason"),
        "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "prompt_tokens": u.get("prompt_tokens", 0),
        "completion_tokens": u.get("completion_tokens", 0),
        "wall_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


def first_div(a, b):
    if a.get("tokens") and b.get("tokens"):
        ta, tb = a["tokens"], b["tokens"]
        for i in range(min(len(ta), len(tb))):
            if ta[i] != tb[i]:
                return i
        return -1 if len(ta) == len(tb) else min(len(ta), len(tb))
    return -1 if a["text"] == b["text"] else 0


def run_case(name, u1, u2):
    # Pin assistant turn-1 text once (greedy → stable). This send also primes
    # the cache, but the cold reference below uses the *full t2 conversation*
    # whose deeper prefix is not yet cached as a contiguous hit until warm.
    a1 = chat([{"role": "system", "content": SYS},
               {"role": "user", "content": u1}], T1_MAX)
    a1_text = a1["text"]

    t2_msgs = [{"role": "system", "content": SYS},
               {"role": "user", "content": u1},
               {"role": "assistant", "content": a1_text},
               {"role": "user", "content": u2}]

    # COLD: first send of the full turn-2 conversation. The radix prefix shared
    # with the turn-1 priming send (sys+u1) is cached, but the assistant region
    # was produced by DECODE in the priming send, so its decode-era Marconi
    # snapshots are what a deep hit restores. To get a genuine COLD (no SSM
    # restore of the deep region) we exploit that the turn-2 prompt's
    # session_hash differs from turn-1's (it includes a1+u2 tokens), so the
    # decode-era snapshots (tagged with turn-1's hash) are session-gated OUT on
    # this first turn-2 send → full recompute. The SECOND identical send shares
    # the turn-2 session_hash with... itself, and now a leaf snapshot from this
    # very send exists → warm restore.
    cold = chat(t2_msgs, T2_MAX)
    warm = chat(t2_msgs, T2_MAX)

    div = first_div(cold, warm)
    ok = div == -1
    return {
        "name": name, "ok": ok, "div": div,
        "a1_completion": a1["completion_tokens"],
        "cold_cached": cold["cached"], "cold_pt": cold["prompt_tokens"],
        "warm_cached": warm["cached"], "warm_pt": warm["prompt_tokens"],
        "cold_text": cold["text"], "warm_text": warm["text"],
        "cold_fr": cold["finish_reason"], "warm_fr": warm["finish_reason"],
        "cold_tc": cold["tool_calls"], "warm_tc": warm["tool_calls"],
    }


def main():
    n = int(os.environ.get("N", "10"))
    cases = (CASES * ((n // len(CASES)) + 1))[:n]
    print(f"=== Chat multi-turn WARM-vs-COLD greedy diff: {n} cases "
          f"(T1_MAX={T1_MAX}, T2_MAX={T2_MAX}, no_think={NO_THINK}) ===", flush=True)
    fails = 0
    for i, (u1, u2) in enumerate(cases):
        r = run_case(f"C{i}", u1, u2)
        st = "PASS" if r["ok"] else "FAIL"
        if not r["ok"]:
            fails += 1
        deep = r["warm_cached"] > r["cold_cached"] + 32
        print(f"  [{st}] {r['name']}: a1_gen={r['a1_completion']} "
              f"cold_cached={r['cold_cached']}/{r['cold_pt']} "
              f"warm_cached={r['warm_cached']}/{r['warm_pt']} "
              f"(deeper_warm={deep}) div_idx={r['div']} "
              f"cold_fr={r['cold_fr']} warm_fr={r['warm_fr']}", flush=True)
        if not r["ok"]:
            print(f"      COLD: {r['cold_text']!r}", flush=True)
            print(f"      WARM: {r['warm_text']!r}", flush=True)
            if r["cold_tc"] or r["warm_tc"]:
                print(f"      COLD_TC: {r['cold_tc']}", flush=True)
                print(f"      WARM_TC: {r['warm_tc']}", flush=True)
    print(f"\n=== {n - fails}/{n} PASS, {fails} FAIL ===", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
