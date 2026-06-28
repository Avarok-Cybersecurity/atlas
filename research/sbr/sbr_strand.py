#!/usr/bin/env python3
"""SBR M1 stranding reproduction — the user's actual scenario.

Phase 1 (BUILD+WARM): grow ONE deep conversation A to ~target depth over several
  turns, resuming each turn -> A becomes "resumable" (snapshot hits register).
Phase 2 (PRESSURE): bursts of one-shot distractor conversations churn the
  snapshot pool while A is idle.
Phase 3 (RESUME): resume A after each pressure burst; measure TTFT + (from logs)
  replay distance. Baseline: A's deep checkpoint evicted -> far replay (slow).
  Tail-pin: A is resumable -> its deepest pinned -> survives -> fast.

Each A turn appends ~chunk_tokens of new context (fast depth growth) + short gen.
"""
import argparse, json, time, urllib.request

WORDS = ("the recurrent state integrates each token through a gated delta update "
         "while attention layers read the cached keys and values across the context "
         "window producing hidden activations consumed by the next transformer block").split()


def filler(n_tokens, salt=0):
    n = int(n_tokens / 0.75)
    return f"[ctx {salt}] " + " ".join(WORDS[(i + salt) % len(WORDS)] for i in range(n))


def chat(base_url, model, messages, max_tokens):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True}).encode()
    req = urllib.request.Request(base_url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft=None; first=None; out=[]
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            ln = raw.decode("utf-8","ignore").strip()
            if ln.startswith("data:") and ln[5:].strip() not in ("","[DONE]"):
                try: d=json.loads(ln[5:].strip())
                except json.JSONDecodeError: continue
                c=d.get("choices",[{}])[0].get("delta",{}).get("content")
                if c:
                    if ttft is None: ttft,first=time.perf_counter()-t0,c
                    out.append(c)
    return ttft, "".join(out), first


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8888/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4")
    ap.add_argument("--build-turns", type=int, default=8)
    ap.add_argument("--chunk-tokens", type=int, default=1800)
    ap.add_argument("--pressure", type=int, default=30, help="one-shot convs per burst")
    ap.add_argument("--pressure-tokens", type=int, default=2000)
    ap.add_argument("--resume-cycles", type=int, default=4)
    a=ap.parse_args()

    A=[{"role":"system","content":"You analyze a growing document.\n"+filler(3000)}]
    rows=[]
    # Phase 1: build + warm (A resumed each turn -> becomes resumable)
    for t in range(a.build_turns):
        A.append({"role":"user","content":f"Section {t}: {filler(a.chunk_tokens, t+1)}\nSummarize so far."})
        depth=int(sum(len(m['content'].split()) for m in A)/0.75)
        ttft,text,_=chat(a.base_url,a.model,A,80)
        A.append({"role":"assistant","content":text})
        rows.append({"phase":"build","t":t,"depth":depth,"ttft_s":ttft})
        print(f"[{a.label}] build t{t} depth~{depth:6d} TTFT={ttft:.3f}s",flush=True)

    # Phase 2+3: pressure burst then resume A
    for cyc in range(a.resume_cycles):
        for d in range(a.pressure):
            sysd=filler(a.pressure_tokens, 1000+cyc*100+d)+f" salt-{cyc}-{d}"
            try: chat(a.base_url,a.model,[{"role":"system","content":sysd},
                                          {"role":"user","content":"One-line summary."}],24)
            except Exception as e: print(f"  pressure err {e}")
        A.append({"role":"user","content":f"Resume {cyc}: given all sections, conclude."})
        depth=int(sum(len(m['content'].split()) for m in A)/0.75)
        ttft,text,_=chat(a.base_url,a.model,A,80)
        A.append({"role":"assistant","content":text})
        rows.append({"phase":"resume","cyc":cyc,"depth":depth,"ttft_s":ttft})
        print(f"[{a.label}] RESUME cyc{cyc} (after {a.pressure} pressure) depth~{depth:6d} TTFT={ttft:.3f}s",flush=True)

    json.dump({"label":a.label,"rows":rows},open(a.out,"w"),indent=2)
    res=[r["ttft_s"] for r in rows if r["phase"]=="resume" and r.get("ttft_s")]
    if res: print(f"\n[{a.label}] post-pressure RESUME TTFT: mean={sum(res)/len(res):.3f}s max={max(res):.3f}s")


if __name__=="__main__":
    main()
