#!/usr/bin/env python3
"""35B NVFP4 A/B benchmark — isolate the pass-3 throughput gap.

Four variants, head DGX only (both Kbenkhaled and Sehyo are cached
there). Each variant loads the model, runs a 500-token essay prompt
three times, then stops. The goal is to pinpoint which config knob
accounts for the gap between pass-3's ~79 tok/s and the historic
~130 tok/s for 35B NVFP4 + MTP.

Run:
    python3 tests/ab_35b.py 2>&1 | tee /tmp/ab-35b.log
"""

import json
import subprocess
import time
import urllib.request

IMAGE = "atlas-gb10:alpha-2.11"
HF_CACHE = "/workspace/.cache/huggingface"
PORT = 8888
RESULT_PATH = "/workspace/atlas/tests/ab_35b_results.json"

VARIANTS = [
    {
        "label": "A — historic Kbenkhaled K=2 8k",
        "model": "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
        "extra": ["--num-drafts", "1",
                  "--max-seq-len", "8192",
                  "--gpu-memory-utilization", "0.88"],
    },
    {
        "label": "B — Sehyo K=2 8k",
        "model": "Sehyo/Qwen3.5-35B-A3B-NVFP4",
        "extra": ["--num-drafts", "1",
                  "--max-seq-len", "8192",
                  "--gpu-memory-utilization", "0.88"],
    },
    {
        "label": "C — Sehyo K=2 32k",
        "model": "Sehyo/Qwen3.5-35B-A3B-NVFP4",
        "extra": ["--num-drafts", "1",
                  "--max-seq-len", "32768",
                  "--gpu-memory-utilization", "0.88"],
    },
    {
        "label": "D — Sehyo K=3 32k (pass-3 config)",
        "model": "Sehyo/Qwen3.5-35B-A3B-NVFP4",
        "extra": ["--num-drafts", "2",
                  "--max-seq-len", "32768",
                  "--gpu-memory-utilization", "0.90"],
    },
]

PROMPT = {
    "messages": [{
        "role": "user",
        "content": (
            "Write a detailed essay on the history of computing from the "
            "abacus to modern GPUs, covering major milestones, key inventors, "
            "and the impact on society. Be thorough and use multiple "
            "paragraphs."
        ),
    }],
    "max_tokens": 500,
    "temperature": 0.3,
}


def sh(cmd, check=True, capture=False, timeout=None):
    if isinstance(cmd, str):
        cmd = ["bash", "-lc", cmd]
    return subprocess.run(cmd, check=check, capture_output=capture,
                          text=True, timeout=timeout)


def start_container(v: dict) -> str:
    name = f"ab-35b-{v['label'].split()[0].strip('—')}"
    sh(f"sudo docker rm -f {name} 2>/dev/null || true", check=False, capture=True)
    serve = [
        "serve", v["model"],
        "--port", str(PORT),
        "--scheduling-policy", "slai",
        "--kv-cache-dtype", "nvfp4",
        "--max-batch-size", "16",
        "--speculative",
        "--mtp-quantization", "nvfp4",
    ] + v["extra"]
    cmd = (
        f"sudo docker run -d --name {name} --gpus all --ipc=host "
        f"-p {PORT}:{PORT} "
        f"-v {HF_CACHE}:/root/.cache/huggingface "
        f"{IMAGE} " + " ".join(serve)
    )
    sh(cmd, capture=True)
    return name


def wait_listening(name: str, timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = sh(f"sudo docker ps -q -f name={name}", check=False, capture=True)
        if not r.stdout.strip():
            return False
        r = sh(f"sudo docker logs {name} 2>&1", check=False, capture=True)
        if "Listening on" in r.stdout:
            return True
        if "Error:" in r.stdout and "ERROR" in r.stdout:
            return False
        time.sleep(5)
    return False


def stop_container(name: str) -> None:
    sh(f"sudo docker stop {name}", check=False, capture=True, timeout=60)
    sh(f"sudo docker rm -f {name}", check=False, capture=True, timeout=30)


def one_request(model: str) -> dict:
    payload = dict(PROMPT, model=model)
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
    elapsed = time.time() - t0
    usage = body.get("usage", {})
    return {
        "elapsed": elapsed,
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "ttft_ms": usage.get("time_to_first_token_ms", 0),
        "wall_tps": (usage.get("completion_tokens", 0) / elapsed) if elapsed > 0 else 0,
        "server_tps": usage.get("response_token/s", 0),
    }


def run_variant(v: dict) -> dict:
    print(f"\n{'='*70}\n{v['label']}\n{'='*70}")
    name = start_container(v)
    print(f"  Starting {name} with {v['model']} and extras: {v['extra']}")
    if not wait_listening(name):
        print(f"  [FAIL] {name} did not reach Listening")
        r = sh(f"sudo docker logs --tail 30 {name} 2>&1", check=False, capture=True)
        print(r.stdout[-1500:])
        stop_container(name)
        return {"label": v["label"], "model": v["model"], "extra": v["extra"],
                "status": "start_fail"}
    print(f"  Ready, running 3 warm requests...")
    runs = []
    # One warmup, then 3 measured
    try:
        _ = one_request(v["model"])  # warmup
    except Exception as e:
        print(f"  [FAIL] warmup: {e}")
        stop_container(name)
        return {"label": v["label"], "model": v["model"], "extra": v["extra"],
                "status": f"warmup_fail: {e}"}
    for i in range(3):
        try:
            r = one_request(v["model"])
            runs.append(r)
            print(f"    run {i+1}: comp={r['completion_tokens']} "
                  f"wall_tps={r['wall_tps']:.1f} "
                  f"server_tps={r['server_tps']:.1f} "
                  f"ttft={r['ttft_ms']:.0f}ms "
                  f"elapsed={r['elapsed']:.2f}s")
        except Exception as e:
            print(f"    run {i+1} FAIL: {e}")
            runs.append({"error": str(e)})
    stop_container(name)
    # Aggregate
    good = [r for r in runs if "error" not in r]
    if not good:
        return {"label": v["label"], "model": v["model"], "extra": v["extra"],
                "status": "all_runs_failed", "runs": runs}
    avg_wall = sum(r["wall_tps"] for r in good) / len(good)
    avg_server = sum(r["server_tps"] for r in good) / len(good)
    avg_ttft = sum(r["ttft_ms"] for r in good) / len(good)
    avg_elapsed = sum(r["elapsed"] for r in good) / len(good)
    return {
        "label": v["label"],
        "model": v["model"],
        "extra": v["extra"],
        "status": "ok",
        "avg_wall_tps": avg_wall,
        "avg_server_tps": avg_server,
        "avg_ttft_ms": avg_ttft,
        "avg_elapsed": avg_elapsed,
        "runs": runs,
    }


def main():
    results = []
    for v in VARIANTS:
        try:
            results.append(run_variant(v))
        except Exception as e:
            print(f"  [FAIL] {v['label']}: {e}")
            results.append({"label": v["label"], "status": f"exception: {e}"})
        time.sleep(30)  # settle between variants

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\n{'='*70}\nAB 35B NVFP4 RESULTS\n{'='*70}")
    print(f"{'variant':<40} {'wall':>8} {'server':>8} {'ttft':>7}")
    print("-" * 70)
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['label']:<40} {'FAIL':>8} — {r.get('status','')[:30]}")
            continue
        print(f"{r['label']:<40} "
              f"{r['avg_wall_tps']:7.1f}  "
              f"{r['avg_server_tps']:7.1f}  "
              f"{r['avg_ttft_ms']:5.0f}ms")


if __name__ == "__main__":
    main()
