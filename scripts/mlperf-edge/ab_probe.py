#!/usr/bin/env python3
"""Decode A/B probe: steady-state TPOT + byte-identical output capture.

Greedy (temp 0, seed 42). For a bit-identical kernel toggle, both legs MUST
emit byte-identical text; the TPOT comparison is only valid when they do
(spec decode is trajectory-dependent). Emits JSON: per-prompt ttft/tpot/ntok/
sha + a combined output sha for the leg.
"""
import hashlib
import json
import statistics
import sys
import time
import urllib.request

PORT, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
URL = f"http://0.0.0.0:{PORT}/v1/chat/completions"

# Long-decode prompts so steady-state TPOT dominates TTFT.
PROMPTS = [
    (
        "Explain, in detail and step by step, how paged attention manages KV cache "
        "blocks during autoregressive decoding. Cover block tables, copy-on-write "
        "for beam search, and fragmentation. Write at least 250 words."
    ),
    (
        "Describe how speculative decoding with a draft model changes the number of "
        "target forward passes, why acceptance rate matters, and how rejection "
        "sampling preserves the target distribution. At least 250 words."
    ),
    (
        "Walk through NVFP4 (E2M1 with FP8 block scales) quantization end to end: "
        "packing, block scale factors, dequant in a GEMM epilogue, and where "
        "accuracy is lost. At least 250 words."
    ),
]
NREPEAT = 3          # each prompt run NREPEAT times; first is warmup-ish, keep all
MAXTOK = 320


def stream_once(prompt):
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK, "temperature": 0, "seed": 42, "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ntok = 0
    chunks = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                if ttft is None:
                    ttft = (time.time() - t0) * 1000.0
                ntok += 1
                chunks.append(delta["content"])
    total = (time.time() - t0) * 1000.0
    tpot = (total - ttft) / max(ntok - 1, 1) if ttft and ntok > 1 else 0.0
    text = "".join(chunks)
    return {"ttft": ttft or 0.0, "tpot": tpot, "ntok": ntok,
            "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text}


runs = []
combined = hashlib.sha256()
for pi, p in enumerate(PROMPTS):
    for r in range(NREPEAT):
        res = stream_once(p)
        res["prompt"] = pi
        res["rep"] = r
        runs.append(res)
        combined.update(res["text"].encode())
        print(f"[{TAG}] p{pi} r{r}: ttft={res['ttft']:7.1f}ms "
              f"tpot={res['tpot']:6.2f}ms/tok ntok={res['ntok']:4d} sha={res['sha']}")

# Steady-state TPOT: drop the first rep of each prompt (cold-ish), median the rest.
warm = [x["tpot"] for x in runs if x["rep"] > 0 and x["tpot"] > 0]
tpots = [x["tpot"] for x in runs if x["tpot"] > 0]
summary = {
    "tag": TAG,
    "tpot_med_warm": statistics.median(warm) if warm else 0.0,
    "tpot_med_all": statistics.median(tpots) if tpots else 0.0,
    "ttft_med": statistics.median([x["ttft"] for x in runs if x["ttft"] > 0]),
    "combined_sha": combined.hexdigest()[:16],
    "runs": runs,
}
with open(OUT, "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"[{TAG}] TPOT warm-median={summary['tpot_med_warm']:.2f}ms  "
      f"combined_sha={summary['combined_sha']}")
