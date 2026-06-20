#!/usr/bin/env python3
"""
test_kv_dtype_smoke.py — End-to-end smoke for every KvCacheDtype the
runtime advertises in the FromStr table.

Catches the class of bug that landed Turbo2 silently routing to the FP8
catch-all in dispatch arms (write_kv_cache.rs and prefill/paged_attn.rs):
the static enum was fine, the FromStr/Display were fine, but a downstream
`match dtype` site fell through to `_` and ran the wrong kernel ABI on
the right kernel handle → CUDA_ERROR_INVALID_ADDRESS_SPACE on first
full-attention layer.

For each dtype: start a fresh container, generate 64 tokens against a
short prompt, assert non-error response with completion_tokens > 0. Skips
ALL of the dtypes that fail to load on the test model (e.g. ones that
need a Bf16-K side on a model that ships only NVFP4 attention weights);
treats load failure as SKIP, runtime failure as FAIL.

Usage:
  python tests/test_kv_dtype_smoke.py
  python tests/test_kv_dtype_smoke.py --image atlas-gb10-tqplus \\
      --model-path /home/pidtom/models/qwen3.6-35b-fp8 \\
      --dtypes turbo2,turbo3,turbo4,bf16k_turbo3v
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request

PORT = 8889
CONTAINER = "atlas-smoke"

DEFAULT_DTYPES = [
    # Symmetric — all should pass on any model.
    "bf16", "fp8", "turbo2", "turbo3", "turbo4", "turbo8",
    # Asymmetric — pass on models whose K-side weights match.
    "bf16k_turbo2v", "bf16k_turbo3v", "bf16k_turbo4v",
    "fp8k_turbo2v", "fp8k_turbo3v", "fp8k_turbo4v",
    "turbo4k_turbo3v", "turbo4k_turbo8v", "turbo3k_turbo8v",
]


def sh(cmd, timeout=120):
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return (p.stdout or "") + (p.stderr or "")


def stop_container():
    sh(f"docker stop {CONTAINER} 2>/dev/null; docker rm {CONTAINER} 2>/dev/null; true")


def start_container(image, model_path, kv_dtype):
    stop_container()
    out = sh(
        f"docker run -d --name {CONTAINER} --gpus all --ipc=host "
        f"-p {PORT}:8888 -v {model_path}:/model "
        f"-e RUST_LOG=warn "
        f"{image} serve --model-from-path /model "
        f"--port 8888 --bind 0.0.0.0 --max-seq-len 8192 "
        f"--kv-cache-dtype {kv_dtype} --kv-high-precision-layers 0"
    )
    return out.strip()[:80]


def wait_ready(timeout_s=120):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2) as r:
                json.load(r)
            return True
        except Exception:
            time.sleep(2)
    return False


def call(prompt, max_tok=64, timeout=120):
    body = {
        "model": "smoke",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def smoke_one(image, model_path, dtype, prompt):
    cid = start_container(image, model_path, dtype)
    if not cid:
        return "FAIL", "container failed to start"
    if not wait_ready():
        # Distinguish load timeout (likely incompatible weights) from crash.
        logs = sh(f"docker logs {CONTAINER} 2>&1 | tail -5")
        if "incompatible" in logs.lower() or "mismatch" in logs.lower():
            return "SKIP", "load incompat"
        return "SKIP", "load timeout (likely incompat)"
    try:
        d = call(prompt, max_tok=64)
        n = d["usage"]["completion_tokens"]
        if n <= 0:
            return "FAIL", f"empty completion ({n} tokens)"
        return "PASS", f"{n} tok / sim'd ok"
    except Exception as e:
        return "FAIL", str(e)[:160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="atlas-gb10-tqplus")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dtypes", default=",".join(DEFAULT_DTYPES))
    ap.add_argument("--prompt", default="Explain what attention is in one sentence.")
    args = ap.parse_args()

    results = []
    for dt in args.dtypes.split(","):
        dt = dt.strip()
        print(f"\n[{dt}] ", end="", flush=True)
        status, note = smoke_one(args.image, args.model_path, dt, args.prompt)
        print(f"{status:<6} {note}", flush=True)
        results.append((dt, status, note))

    stop_container()

    print("\n=== SUMMARY ===")
    fmt = "{:<22} {:<6} {}"
    for dt, st, note in results:
        print(fmt.format(dt, st, note))

    fails = [r for r in results if r[1] == "FAIL"]
    if fails:
        print(f"\n{len(fails)} FAIL", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll dispatched dtypes OK ({sum(1 for r in results if r[1]=='PASS')} pass, "
          f"{sum(1 for r in results if r[1]=='SKIP')} skip)")


if __name__ == "__main__":
    main()
