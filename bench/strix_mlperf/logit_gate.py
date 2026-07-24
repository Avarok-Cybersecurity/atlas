"""Prefill-numerics drift gate.

Uses /v1/completions with echo=true + logprobs=K. Prompt logprobs are produced
entirely by the PREFILL path, so this measures exactly what swapping the FFN
prefill GEMM (m128 -> M64) changes -- thousands of scored positions per prompt,
with no decode/MTP/sampling confound. Run once per binary/config, then diff.

usage: python3 logit_gate.py <out.json>
"""
import json, random, sys, urllib.request

URL = "http://localhost:8081/v1/completions"
MODEL = "nvidia/Qwen3.6-27B-NVFP4"
TOPK = 5

POOL = ("system module cache latency kernel tensor buffer schedule pipeline "
        "gradient matrix vector throughput register occupancy warp lane stride "
        "prefetch dispatch retire commit branch predictor allocator arena page "
        "descriptor fragment shader compute queue fence semaphore barrier atomic "
        "coherent snapshot replay checkpoint rollback session transcript token").split()

def filler(nwords, seed):
    rng = random.Random(seed)
    return " ".join(rng.choice(POOL) for _ in range(nwords))

# Mix of lengths. The long ones matter most: they drive M well past 64 so the
# BASELINE actually takes the m128 arm being replaced.
PROMPTS = [
    ("short_prose", "The capital of France is Paris, and the city is known for"),
    ("code", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n"),
    ("mid_2k", "[doc a91f] Log excerpt.\n" + filler(1600, 4242) + "\nSummarize the above."),
    ("long_6k", "[doc c17e] Log excerpt.\n" + filler(4800, 9090) + "\nSummarize the above."),
]

def score(prompt):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 1,
                       "temperature": 0, "echo": True, "logprobs": TOPK}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        o = json.loads(r.read())
    ch = o["choices"][0]
    lp = ch.get("logprobs") or {}
    return {
        "tokens": lp.get("tokens"),
        "token_logprobs": lp.get("token_logprobs"),
        "top_logprobs": lp.get("top_logprobs"),
        "text_tail": ch.get("text", "")[-200:],
    }

out = {}
for name, p in PROMPTS:
    out[name] = score(p)
    n = len(out[name]["token_logprobs"] or [])
    print("scored %-12s positions=%d" % (name, n), flush=True)

json.dump(out, open(sys.argv[1], "w"))
print("wrote", sys.argv[1])
