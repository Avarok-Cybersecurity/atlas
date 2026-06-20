#!/usr/bin/env python3
"""Deterministic single-server multi-turn PARTIAL-HIT prefix-cache repro.

Goal: isolate the residual `--enable-prefix-caching` regression (cache-ON
opencode webserver_ok ~23% vs cache-OFF ~65%) to the MULTI-TURN partial-hit
recompute path — restore SSM snapshot at snap_tok, recompute SSM over
[snap_tok, total), reuse cached attention KV for [0, snap_tok).

It needs ONLY ONE server (the cache-ON `atlas-gb10:pfxfix` build). It does
NOT compare head-vs-worker. Instead, per trial it compares the SAME final
turn fired:

  (a) COLD  — with an empty/non-matching cache (a unique nonce in the SYSTEM
              prompt makes turn-1's blocks distinct, so turn-2's prefix walk
              finds NOTHING to reuse — full recompute, the cache-OFF-equiv).
  (b) WARM  — turn-1 fired first to populate the radix tree + Marconi
              snapshots, THEN the identical turn-2 fired so its long prefix
              partial-hits turn-1's cached blocks at a block-aligned
              checkpoint < total.

At temp 0 the COLD and WARM turn-2 completions MUST be byte-identical.
Any divergence reproduces the bug — and the first differing token is the
forensic evidence.

Design choices that target the agentic path:
  * Turn-1 user message is LONG (configurable --filler-tokens, default ~3500
    words) so turn-2's prefix exceeds several KV blocks and several
    --max-prefill-tokens=2048 chunks → partial hit lands on an intermediate
    Marconi checkpoint, the c487bc4 path.
  * messages=[{system},{user1},{assistant1},{user2}] — a canned assistant1
    turn, exactly like opencode's transcript replay.
  * The cache is "cleared" for the COLD arm purely by changing the nonce
    (no container restart needed) — cold trials share no prefix with any
    prior warm trial. For an even stronger guarantee, pass --restart to
    bounce the container between arms (requires docker perms).

Usage:
  python3 prefix_partial_hit_repro.py --url http://localhost:8889 \
      --model Qwen/Qwen3.6-35B-A3B-FP8 --trials 5 --max-tokens 200

Exit code 0 = all trials identical (no repro); 1 = divergence found.
"""
import argparse
import json
import sys
import time
import urllib.request
import uuid


def chat(url, model, messages, max_tokens, seed=0):
    """Fire one /v1/chat/completions at temp 0. Returns (text, usage)."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    wall = (time.perf_counter() - t0) * 1000.0
    msg = r["choices"][0]["message"]
    text = msg.get("content") or ""
    u = r.get("usage", {}) or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return text, {
        "prompt_tokens": u.get("prompt_tokens"),
        "cached_tokens": cached,
        "wall_ms": round(wall, 1),
        "finish": r["choices"][0].get("finish_reason"),
        "completion_tokens": u.get("completion_tokens"),
    }


def make_filler(n_words):
    # Deterministic, content-rich filler so token boundaries are stable and
    # the prefix spans several blocks. Varied vocabulary avoids degenerate
    # repetition that the model might special-case.
    words = (
        "the inference engine schedules paged attention blocks while the "
        "recurrent state machine advances its convolutional window across "
        "every token in the hybrid transformer mamba stack producing keys "
        "values and gated deltas that feed the mixture of experts router "
        ).split()
    out = []
    i = 0
    while len(out) < n_words:
        out.append(words[i % len(words)])
        i += 1
    return " ".join(out)


def build_messages(nonce, filler, user2):
    return [
        {
            "role": "system",
            "content": (
                f"Session {nonce}. You are a precise engineering assistant. "
                "Answer concisely and deterministically."
            ),
        },
        {
            "role": "user",
            "content": (
                "Here is a long technical document to keep in mind:\n\n"
                f"{filler}\n\n"
                "First, briefly acknowledge you have read it."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Acknowledged. I have read the technical document describing "
                "the hybrid transformer-mamba inference engine, its paged "
                "attention, recurrent convolutional state, and mixture-of-"
                "experts routing. I am ready for your next question."
            ),
        },
        {"role": "user", "content": user2},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8889")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--filler-words", type=int, default=3500,
                    help="approx words in turn-1 user msg (drives prefix length)")
    ap.add_argument("--user2", default=(
        "Now: explain in 3 numbered steps how a partial prefix-cache hit "
        "recomputes the recurrent SSM state. Be specific and deterministic."))
    args = ap.parse_args()

    filler = make_filler(args.filler_words)
    divergences = 0

    for t in range(args.trials):
        nonce = uuid.uuid4().hex
        msgs = build_messages(nonce, filler, args.user2)

        # --- WARM arm: fire turn-1 (system+user1) to populate cache, then
        #     fire the full 4-message turn-2 so its prefix partial-hits. ---
        # Turn-1 priming request = first 2 messages only (no assistant yet),
        # which writes system+user1 blocks + intermediate SSM checkpoints.
        prime = msgs[:2]
        _ = chat(args.url, args.model, prime, max_tokens=8, seed=0)
        # Now the full transcript: shares the system+user1 prefix with the
        # primed cache → partial hit at a block-aligned checkpoint < total.
        warm_text, warm_u = chat(args.url, args.model, msgs, args.max_tokens, seed=0)

        # --- COLD arm: brand-new nonce so NOTHING in cache matches this
        #     trial's prefix → full recompute (cache-OFF equivalent). ---
        cold_nonce = uuid.uuid4().hex
        cold_msgs = build_messages(cold_nonce, filler, args.user2)
        cold_text, cold_u = chat(args.url, args.model, cold_msgs, args.max_tokens, seed=0)

        identical = warm_text == cold_text
        if not identical:
            divergences += 1
            # First diverging char index.
            k = 0
            while k < min(len(warm_text), len(cold_text)) and warm_text[k] == cold_text[k]:
                k += 1
            print(f"\n[trial {t}] DIVERGENCE at char {k}", flush=True)
            print(f"  warm cached={warm_u['cached_tokens']}/{warm_u['prompt_tokens']} "
                  f"finish={warm_u['finish']} ctoks={warm_u['completion_tokens']}", flush=True)
            print(f"  cold cached={cold_u['cached_tokens']}/{cold_u['prompt_tokens']} "
                  f"finish={cold_u['finish']} ctoks={cold_u['completion_tokens']}", flush=True)
            ctx = 60
            print(f"  COLD …{cold_text[max(0,k-ctx):k+ctx]!r}", flush=True)
            print(f"  WARM …{warm_text[max(0,k-ctx):k+ctx]!r}", flush=True)
        else:
            print(f"[trial {t}] identical "
                  f"(warm cached={warm_u['cached_tokens']}/{warm_u['prompt_tokens']}, "
                  f"cold cached={cold_u['cached_tokens']}/{cold_u['prompt_tokens']})",
                  flush=True)

    print(f"\n=== {args.trials - divergences}/{args.trials} identical; "
          f"{divergences} divergences ===", flush=True)
    sys.exit(1 if divergences else 0)


if __name__ == "__main__":
    main()
