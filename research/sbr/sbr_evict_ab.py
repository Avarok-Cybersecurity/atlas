#!/usr/bin/env python3
"""SBR M1 eviction-isolation A/B — does tail-pin keep warm-hit TTFT flat?

Uses FEW long-lived sessions (distinct prefixes -> distinct session_hash),
interleaved round-robin so each session's deep tail competes for snapshot slots
with the others. Avoids the many-short-session slot leak. With the snapshot pool
sized below the sum of all sessions' checkpoint chains, the eviction policy must
choose victims: baseline evicts the least-recently-used session's deep tail (so
its next resume replays far); tail-pin protects each session's deepest snapshot
(so every resume replays only ~one interval).

Run on dgx2 against a server already up:
  python3 sbr_evict_ab.py --label sbr_on --out sbr_on.json --sessions 4 --rounds 6
"""
import argparse, json, time, urllib.request

WORDS = ("the recurrent layer accumulates a decaying hidden state while attention "
         "reads the key value cache and each token updates the matrix in sequence "
         "according to a learned gate beta and delta rule producing the next output").split()


def filler(n_tokens, salt):
    n = int(n_tokens / 0.75)
    base = " ".join(WORDS[(i + salt) % len(WORDS)] for i in range(n))
    return f"[session {salt}] " + base + f" [unique-marker-{salt}]"


def chat_ttft(base_url, model, messages, max_tokens):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(base_url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, first, text = None, None, []
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                try:
                    d = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                c = d.get("choices", [{}])[0].get("delta", {}).get("content")
                if c:
                    if ttft is None:
                        ttft, first = time.perf_counter() - t0, c
                    text.append(c)
    return ttft, "".join(text), first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8888/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4")
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--base-tokens", type=int, default=3000)
    ap.add_argument("--resp-tokens", type=int, default=300)
    a = ap.parse_args()

    histories = [[{"role": "system", "content": filler(a.base_tokens, s)}]
                 for s in range(a.sessions)]
    rows = []
    for rnd in range(a.rounds):
        for s in range(a.sessions):
            histories[s].append({"role": "user", "content": f"R{rnd} S{s}: continue the analysis."})
            depth = sum(len(m["content"].split()) for m in histories[s])  # ~words≈tokens*0.75
            depth_tok = int(depth / 0.75)
            ttft, text, first = chat_ttft(a.base_url, a.model, histories[s], a.resp_tokens)
            histories[s].append({"role": "assistant", "content": text})
            rows.append({"round": rnd, "session": s, "depth_tok": depth_tok,
                         "ttft_s": ttft, "first_tok": first})
            print(f"[{a.label}] r{rnd} s{s} depth~{depth_tok:6d} TTFT={ttft:.3f}s", flush=True)

    json.dump({"label": a.label, "sessions": a.sessions, "rounds": a.rounds, "rows": rows},
              open(a.out, "w"), indent=2)
    # per-round resume TTFT (rounds>0 are warm resumes)
    warm = [r["ttft_s"] for r in rows if r["round"] > 0 and r["ttft_s"]]
    if warm:
        print(f"\n[{a.label}] warm-resume TTFT: mean={sum(warm)/len(warm):.3f}s "
              f"max={max(warm):.3f}s min={min(warm):.3f}s")


if __name__ == "__main__":
    main()
