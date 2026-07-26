#!/usr/bin/env python3
"""Measure the ACTUAL per-turn SSM replay distance on the real MLPerf-edge workload.

This settles the open question left by the TTFT law. Fitting the golden run gives
`TTFT = 879 ms + 1.753 ms/new_token`, and the dgx2 server-side law
`server_prefill ~= 170 + 2.0 x tokens_actually_prefilled` back-solves that 879 ms
intercept to ~251 tokens re-prefilled per turn even when the delta is ~0. Two-block
tail-checkpoint rounding (<=32 tokens) cannot explain 251.

The dgx2 root cause (Marconi restores at the previous turn's PROMPT end, leaving its
generated response uncheckpointed) was measured in CHAT mode, where the assistant's
generated tokens are echoed back verbatim. The MLPerf harness does NOT do that: it
drives a FLAT string prompt that is an exact prefix-extension, substituting its own
`[{"id":"functions.bash:0",...}]` JSON for the model's output (verified: 987/987
turns are exact prefix extensions). So the chat-mode explanation may not transfer,
and guessing is pointless when the server logs the answer.

Rather than synthesise a workload, this replays the REAL prompt sequence of a real
conversation straight out of the golden run's events.jsonl, in order, and reads the
server's own accounting:

    "Marconi intermediate hit: restored from checkpoint at token N
     (skipping S tokens, replaying R SSM tokens to reach M)"

R is the number that matters. If R is ~0-32 per turn, the re-prefill lever does not
exist on this workload and the 879 ms intercept is something else. If R is in the
hundreds, it does, and it is worth ~655 s.

Usage: replay_distance_probe.py <port> <events.jsonl> <conversation_id> <out.json>
                                [--max-turns 25]
"""
import argparse
import json
import statistics
import time
import urllib.request

MODEL = "centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf"


def load_turns(path, cid, limit):
    """Real prompts for one conversation, in turn order, from the harness log."""
    turns = {}
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("event_type") != "sample.issued":
                continue
            if d.get("conversation_id") != cid or d.get("turn") is None:
                continue
            data = d.get("data")
            if isinstance(data, list) and len(data) > 1:
                turns[d["turn"]] = data[1]
    return [turns[t] for t in sorted(turns)][:limit]


def send(port, prompt, max_tokens=16):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "seed": 42, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            s = raw.decode("utf-8", "ignore").strip()
            if not s.startswith("data:"):
                continue
            p = s[5:].strip()
            if p == "[DONE]":
                break
            try:
                d = json.loads(p)
            except Exception:
                continue
            if (d.get("choices") or [{}])[0].get("text") and ttft is None:
                ttft = (time.time() - t0) * 1000.0
    return ttft or 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port"); ap.add_argument("events"); ap.add_argument("cid")
    ap.add_argument("out"); ap.add_argument("--max-turns", type=int, default=25)
    a = ap.parse_args()

    prompts = load_turns(a.events, a.cid, a.max_turns)
    if not prompts:
        raise SystemExit(f"no turns found for conversation {a.cid}")
    print(f"replaying {len(prompts)} real turns of {a.cid}", flush=True)

    rows = []
    prev = None
    for i, p in enumerate(prompts):
        # Character-level delta vs the previous turn, i.e. exactly what the cache
        # should have to prefill if the restore lands at the previous prompt end.
        if prev is None:
            d_ch = len(p)
        else:
            n = min(len(prev), len(p)); j = 0
            while j < n and prev[j] == p[j]:
                j += 1
            d_ch = len(p) - j
        t = send(a.port, p)
        rows.append({"turn": i, "prompt_chars": len(p), "delta_chars": d_ch, "ttft": t})
        print(f"  turn {i:3d}  prompt={len(p):7d}ch  delta={d_ch:6d}ch  ttft={t:8.1f} ms",
              flush=True)
        prev = p

    warm = [r for r in rows if r["turn"] > 0]
    summary = {
        "conversation": a.cid, "turns": len(rows),
        "warm_ttft_p50": statistics.median([r["ttft"] for r in warm]) if warm else 0,
        "delta_ch_p50": statistics.median([r["delta_chars"] for r in warm]) if warm else 0,
        "rows": rows,
    }
    with open(a.out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwarm TTFT p50 = {summary['warm_ttft_p50']:.0f} ms, "
          f"delta p50 = {summary['delta_ch_p50']:.0f} chars")
    print("\nNow read the SERVER's own accounting from the container log:")
    print("  sudo docker logs <container> 2>&1 | grep -a 'Marconi'")
    print("The 'replaying R SSM tokens' field is the measurement this probe exists for.")


if __name__ == "__main__":
    main()
