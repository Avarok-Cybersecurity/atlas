#!/usr/bin/env python3
"""SBR M1 multi-conversation robustness — many REAL conversations competing.

Phase A (WARM): build M conversations sequentially to moderate depth, resuming
  each turn so every conversation becomes resumable (hit_count>0).
Phase B (COMPETE): round-robin resume all M conversations for R rounds; between
  a conversation's turns the other M-1 churn the snapshot pool. With K*M > slots
  the tail-pin fallback is exercised. Measures per-conversation resume TTFT.

Reports per-round resume TTFT distribution. Compare baseline vs tail-pin: tail-pin
should keep every resumable conversation's deep tail anchored (low, flat TTFT)
while baseline strands the least-recently-used ones (high-variance TTFT).
"""
import argparse, json, time, urllib.request

WORDS = ("the recurrent state integrates tokens through a gated delta rule while "
         "attention reads cached keys and values producing activations for the next "
         "transformer block across the growing context window of the conversation").split()


def filler(n_tokens, salt):
    n = int(n_tokens / 0.75)
    return f"[conv {salt}] " + " ".join(WORDS[(i + salt) % len(WORDS)] for i in range(n)) + f" #{salt}"


def chat(base_url, model, messages, max_tokens):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(base_url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft=None; out=[]
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            ln = raw.decode("utf-8","ignore").strip()
            if ln.startswith("data:") and ln[5:].strip() not in ("","[DONE]"):
                try: d=json.loads(ln[5:].strip())
                except json.JSONDecodeError: continue
                c=d.get("choices",[{}])[0].get("delta",{}).get("content")
                if c:
                    if ttft is None: ttft=time.perf_counter()-t0
                    out.append(c)
    return ttft, "".join(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8888/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4")
    ap.add_argument("--convs", type=int, default=8)
    ap.add_argument("--build-turns", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--base-tokens", type=int, default=2000)
    ap.add_argument("--chunk-tokens", type=int, default=1500)
    a=ap.parse_args()

    H=[[{"role":"system","content":"You analyze a document.\n"+filler(a.base_tokens, s)}]
       for s in range(a.convs)]
    rows=[]
    # Phase A: warm each conversation (sequential build, resumed -> resumable)
    for s in range(a.convs):
        for t in range(a.build_turns):
            H[s].append({"role":"user","content":f"Sec {t}: {filler(a.chunk_tokens, s*10+t)} Summarize."})
            ttft,text=chat(a.base_url,a.model,H[s],60)
            H[s].append({"role":"assistant","content":text})
        print(f"[{a.label}] warmed conv {s} depth~{int(sum(len(m['content'].split()) for m in H[s])/0.75)}",flush=True)

    # Phase B: round-robin resume under mutual pressure
    for rnd in range(a.rounds):
        for s in range(a.convs):
            H[s].append({"role":"user","content":f"R{rnd}: conclude given all sections."})
            depth=int(sum(len(m['content'].split()) for m in H[s])/0.75)
            ttft,text=chat(a.base_url,a.model,H[s],60)
            H[s].append({"role":"assistant","content":text})
            rows.append({"round":rnd,"conv":s,"depth":depth,"ttft_s":ttft})
            print(f"[{a.label}] r{rnd} conv{s} depth~{depth:6d} TTFT={ttft:.3f}s",flush=True)

    json.dump({"label":a.label,"convs":a.convs,"rounds":a.rounds,"rows":rows},open(a.out,"w"),indent=2)
    t=[r["ttft_s"] for r in rows if r.get("ttft_s")]
    import statistics
    print(f"\n[{a.label}] resume TTFT mean={statistics.mean(t):.3f}s p50={statistics.median(t):.3f}s "
          f"p90={sorted(t)[int(0.9*len(t))-1]:.3f}s max={max(t):.3f}s")


if __name__=="__main__":
    main()
