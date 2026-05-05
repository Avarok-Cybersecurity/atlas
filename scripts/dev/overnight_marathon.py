#!/usr/bin/env python3
"""
Atlas Overnight Marathon Test Suite
Runs all models in parallel across 2 DGX Spark nodes for 7+ hours.
Self-repairs issues and produces a final summary table.

Usage: python3 overnight_marathon.py 2>&1 | tee /workspace/atlas/overnight_results.log
"""
import subprocess, json, time, re, os, sys, urllib.request, traceback
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
MARATHON_HOURS = 7
IMAGE = "atlas-gb10:latest"
HF_CACHE_DGX1 = "/workspace/.cache/huggingface/hub"
HF_CACHE_DGX2 = "/workspace/.cache/huggingface/hub"
DGX1_IP = __import__("os").environ.get("DGX1_IP", "127.0.0.1")
DGX2_IP = __import__("os").environ.get("DGX2_IP", "127.0.0.1")

# NCCL env for EP=2
NCCL_ENV = {
    "NCCL_SOCKET_IFNAME": "enp1s0f0np0",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_HCA": "rocep1s0f0",
    "NCCL_IB_ROCE_VERSION_NUM": "2",
    "NCCL_IB_ADDR_FAMILY": "AF_INET",
    "NCCL_IB_TIMEOUT": "22",
    "NCCL_IB_RETRY_CNT": "7",
    "NCCL_NET_GDR_LEVEL": "0",
    "NCCL_NET_GDR_C2C": "0",
    "NCCL_DMABUF_ENABLE": "0",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_CUMEM_HOST_ENABLE": "0",
    "NCCL_PROTO": "Simple",
    "NCCL_ALGO": "Ring",
    "NCCL_MIN_NCHANNELS": "1",
    "NCCL_MAX_NCHANNELS": "2",
    "NCCL_DEBUG": "WARN",
}

# Models to test (model_id, extra_args, node)
SINGLE_GPU_MODELS = [
    # Pair 1: dgx1 + dgx2 in parallel
    ("Sehyo/Qwen3.5-35B-A3B-NVFP4", "--speculative", "dgx1"),
    ("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", "--speculative", "dgx2"),
    # Pair 2
    ("Sehyo/Qwen3.5-122B-A10B-NVFP4", "", "dgx1"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", "", "dgx2"),
    # Pair 3
    ("mistralai/Mistral-Small-4-119B-2603-NVFP4", "--kv-cache-dtype bf16", "dgx1"),
    ("ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4", "", "dgx2"),
    # Pair 4
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "", "dgx1"),
]

EP2_MODELS = [
    ("Sehyo/Qwen3.5-122B-A10B-NVFP4", ""),
]

# Test prompts
TESTS = [
    ("capital", "What is the capital of France?", 50, r"Paris"),
    ("fibonacci", "Write the first 10 Fibonacci numbers separated by commas.", 80, r"0.*1.*1.*2.*3.*5.*8.*13.*21.*34"),
    ("coherence", "Explain what photosynthesis is in 2-3 sentences.", 150, r"sunlight|chloro|oxygen|carbon|glucose|plant"),
    ("planets", "List all 8 planets in our solar system in order from the sun.", 200, r"Mercury.*Venus.*Earth.*Mars.*Jupiter.*Saturn.*Uranus.*Neptune"),
]

# NIAH needle
NEEDLE = "IMPORTANT FACT: The secret project codename is AURORA-VELVET-7742."
NEEDLE_PATTERN = r"AURORA.VELVET.7742"
FILLER_PARA = "The history of cartography stretches back thousands of years. Ancient civilizations in Mesopotamia created clay tablet maps as early as 2300 BCE. The Greeks made significant advances with Eratosthenes calculating the Earth circumference with remarkable accuracy around 240 BCE. During the Age of Exploration European cartographers produced increasingly detailed world maps. The Mercator projection introduced in 1569 became the standard for nautical navigation. Modern cartography has been revolutionized by satellite imagery GPS technology and geographic information systems. Digital mapping services now process billions of queries daily.\n\n"

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 1

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def docker_stop(name, host=None):
    prefix = f"ssh {host} " if host else ""
    run(f"{prefix}sudo docker stop {name} 2>/dev/null", timeout=30)
    run(f"{prefix}sudo docker rm {name} 2>/dev/null", timeout=30)

def docker_start(name, model, port, extra, host=None, hf_cache=HF_CACHE_DGX1, max_seq=16384, ep_args=""):
    prefix = f"ssh {host} " if host else ""
    nccl_str = ""
    if ep_args:
        nccl_str = " ".join(f"-e {k}={v}" for k, v in NCCL_ENV.items())
        nccl_str = f"--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1 {nccl_str}"
    cmd = (f"{prefix}sudo docker run -d --name {name} "
           f"--gpus all --ipc=host --network host {nccl_str} "
           f"-v {hf_cache}:/root/.cache/huggingface/hub "
           f"{IMAGE} serve {model} --port {port} --max-seq-len {max_seq} {extra} {ep_args}")
    out, rc = run(cmd, timeout=30)
    return rc == 0

def wait_ready(name, host=None, timeout_s=420):
    prefix = f"ssh {host} " if host else ""
    for _ in range(timeout_s // 5):
        logs, _ = run(f"{prefix}sudo docker logs {name} 2>&1 | tail -20", timeout=10)
        if "Listening on" in logs:
            return True
        if "Error:" in logs and "FP8" not in logs:
            return False
        time.sleep(5)
    return False

def api_test(port, prompt, max_tokens=150, timeout_s=120, host=None):
    url = f"http://{'localhost' if not host else host}:{port}/v1/chat/completions"
    body = json.dumps({"model": "test", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}

def build_niah_prompt(target_tokens, position):
    filler = (FILLER_PARA * 200)[:target_tokens * 4]
    at = int(len(filler) * position)
    bnd = filler.find("\n\n", at)
    if bnd == -1 or bnd > at + 500: bnd = at
    return ("Read this text carefully.\n\n" + filler[:bnd] + "\n\n" + NEEDLE + "\n\n" + filler[bnd:] +
            "\n\nWhat is the secret project codename? Reply with ONLY the codename.")

def run_model_suite(model, port, is_mistral=False, is_niah=True, host=None):
    """Run full test suite on a model, return results dict."""
    results = {"model": model, "tests": [], "niah": [], "timestamp": datetime.now().isoformat()}

    for name, prompt, max_tok, pattern in TESTS:
        r = api_test(port, prompt, max_tok, host=host)
        if "error" in r:
            results["tests"].append({"test": name, "status": "ERROR", "error": r["error"][:100]})
            continue
        c = r["choices"][0]
        u = r["usage"]
        content = c["message"]["content"]
        passed = bool(re.search(pattern, content, re.IGNORECASE | re.DOTALL))
        results["tests"].append({
            "test": name, "status": "PASS" if passed else "FAIL",
            "tok_s": round(u.get("response_token/s", 0), 1),
            "ttft_ms": round(u.get("time_to_first_token_ms", 0), 1),
            "content": content[:80],
        })

    if is_niah and not is_mistral:
        for nname, tokens, pos in [("niah_4k", 4000, 0.5), ("niah_16k", 16000, 0.5)]:
            prompt = build_niah_prompt(tokens, pos)
            r = api_test(port, prompt, 50, timeout_s=180, host=host)
            if "error" in r:
                results["niah"].append({"test": nname, "status": "ERROR", "error": str(r.get("error",""))[:80]})
                continue
            c = r["choices"][0]
            u = r["usage"]
            content = c["message"]["content"]
            passed = bool(re.search(NEEDLE_PATTERN, content, re.IGNORECASE))
            results["niah"].append({
                "test": nname, "status": "PASS" if passed else "FAIL",
                "prompt_tokens": u.get("prompt_tokens", 0),
                "ttft_ms": round(u.get("time_to_first_token_ms", 0), 1),
                "content": content[:60],
            })

    std_pass = sum(1 for t in results["tests"] if t["status"] == "PASS")
    niah_pass = sum(1 for t in results["niah"] if t["status"] == "PASS")
    avg_toks = sum(t.get("tok_s", 0) for t in results["tests"]) / max(len(results["tests"]), 1)
    results["summary"] = {
        "std": f"{std_pass}/{len(results['tests'])}",
        "niah": f"{niah_pass}/{len(results['niah'])}" if results["niah"] else "—",
        "tok_s": round(avg_toks, 1),
    }
    return results

# ═══════════════════════════════════════════════════════════════
# Main Marathon Loop
# ═══════════════════════════════════════════════════════════════
def main():
    start_time = time.time()
    end_time = start_time + MARATHON_HOURS * 3600
    all_results = []
    iteration = 0

    log(f"═══ ATLAS OVERNIGHT MARATHON ═══")
    log(f"Duration: {MARATHON_HOURS} hours")
    log(f"End time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Models: {len(SINGLE_GPU_MODELS)} single-GPU + {len(EP2_MODELS)} EP=2")
    log("")

    while time.time() < end_time:
        iteration += 1
        iter_start = time.time()
        log(f"═══ ITERATION {iteration} ═══ (elapsed: {timedelta(seconds=int(time.time()-start_time))})")

        # ── Phase 1: Parallel single-GPU tests ──
        # Process models in pairs (dgx1 + dgx2)
        pairs = []
        i = 0
        while i < len(SINGLE_GPU_MODELS):
            pair = [SINGLE_GPU_MODELS[i]]
            if i + 1 < len(SINGLE_GPU_MODELS) and SINGLE_GPU_MODELS[i+1][2] == "dgx2":
                pair.append(SINGLE_GPU_MODELS[i+1])
                i += 2
            else:
                i += 1
            pairs.append(pair)

        for pair in pairs:
            if time.time() >= end_time:
                break

            # Stop all containers
            for _, _, node in pair:
                cname = f"atlas-{'ep0' if node == 'dgx1' else 'ep1'}"
                host = None if node == "dgx1" else DGX2_IP
                docker_stop(cname, host)

            # Start containers in parallel
            started = {}
            for model, extra, node in pair:
                host = None if node == "dgx1" else DGX2_IP
                hf = HF_CACHE_DGX1 if node == "dgx1" else HF_CACHE_DGX2
                port = 8888 if node == "dgx1" else 8888
                cname = f"atlas-{'ep0' if node == 'dgx1' else 'ep1'}"
                short = model.split("/")[-1][:30]
                log(f"  Starting {short} on {node}...")
                ok = docker_start(cname, model, port, extra, host, hf)
                if ok:
                    started[node] = (model, cname, port, host, extra)

            # Wait for all to be ready
            ready = {}
            for node, (model, cname, port, host, extra) in started.items():
                short = model.split("/")[-1][:30]
                if wait_ready(cname, host, timeout_s=420):
                    log(f"  ✓ {short} ready on {node}")
                    ready[node] = (model, port, host, extra)
                else:
                    log(f"  ✗ {short} FAILED to start on {node}")

            # Run tests in parallel (sequential per node for simplicity)
            for node, (model, port, host, extra) in ready.items():
                short = model.split("/")[-1][:30]
                is_mistral = "mistral" in model.lower()
                try:
                    results = run_model_suite(model, port, is_mistral=is_mistral, host=host)
                    s = results["summary"]
                    log(f"  {short}: std={s['std']} niah={s['niah']} {s['tok_s']} tok/s")
                    all_results.append(results)
                except Exception as e:
                    log(f"  {short}: ERROR {e}")
                    all_results.append({"model": model, "summary": {"std": "ERR", "niah": "ERR", "tok_s": 0}, "error": str(e)[:200], "timestamp": datetime.now().isoformat()})

            # Cleanup
            for node, (model, cname, port, host, extra) in started.items():
                docker_stop(cname, host)

        # ── Phase 2: EP=2 tests ──
        if time.time() < end_time:
            for model, extra in EP2_MODELS:
                if time.time() >= end_time:
                    break
                short = model.split("/")[-1][:30]
                log(f"  EP=2: {short}")

                docker_stop("atlas-ep0")
                docker_stop("atlas-ep1", DGX2_IP)

                # Head
                ep_head = f"--master-addr {DGX1_IP} --world-size 2 --rank 0"
                ok0 = docker_start("atlas-ep0", model, 8888, extra, None, HF_CACHE_DGX1, 16384, ep_head)

                # Wait for NCCL listener
                for _ in range(84):
                    logs, _ = run("sudo docker logs atlas-ep0 2>&1 | tail -5", timeout=10)
                    if "waiting for" in logs:
                        break
                    if "Error:" in logs and "FP8" not in logs:
                        break
                    time.sleep(5)

                # Worker
                ep_worker = f"--master-addr {DGX1_IP} --world-size 2 --rank 1"
                ok1 = docker_start("atlas-ep1", model, 8889, extra, DGX2_IP, HF_CACHE_DGX2, 16384, ep_worker)

                if ok0 and ok1 and wait_ready("atlas-ep0", timeout_s=480):
                    log(f"  EP=2 {short} ready")
                    try:
                        results = run_model_suite(model + " (EP=2)", 8888, is_niah=True)
                        s = results["summary"]
                        log(f"  EP=2 {short}: std={s['std']} niah={s['niah']} {s['tok_s']} tok/s")
                        all_results.append(results)
                    except Exception as e:
                        log(f"  EP=2 {short}: ERROR {e}")
                        all_results.append({"model": model + " (EP=2)", "summary": {"std": "ERR", "niah": "ERR", "tok_s": 0}, "error": str(e)[:200], "timestamp": datetime.now().isoformat()})
                else:
                    log(f"  EP=2 {short}: FAILED TO START")
                    all_results.append({"model": model + " (EP=2)", "summary": {"std": "FAIL", "niah": "FAIL", "tok_s": 0}, "timestamp": datetime.now().isoformat()})

                docker_stop("atlas-ep0")
                docker_stop("atlas-ep1", DGX2_IP)

        iter_elapsed = time.time() - iter_start
        log(f"  Iteration {iteration} complete in {timedelta(seconds=int(iter_elapsed))}")
        log("")

    # ═══════════════════════════════════════════════════════════════
    # Final Summary Table
    # ═══════════════════════════════════════════════════════════════
    elapsed = timedelta(seconds=int(time.time() - start_time))
    log(f"\n═══ MARATHON COMPLETE ═══")
    log(f"Total time: {elapsed}")
    log(f"Iterations: {iteration}")
    log(f"Total test runs: {len(all_results)}")
    log("")

    # Build summary table
    log(f"{'Model':<45} {'Iter':>4} {'Std':>5} {'NIAH':>5} {'tok/s':>7} {'Time':>12}")
    log("─" * 85)

    model_stats = {}
    for r in all_results:
        m = r["model"]
        s = r.get("summary", {})
        ts = r.get("timestamp", "")
        key = m
        if key not in model_stats:
            model_stats[key] = {"runs": 0, "std_pass": 0, "std_total": 0, "niah_pass": 0, "niah_total": 0, "tok_s": [], "last_ts": ""}
        ms = model_stats[key]
        ms["runs"] += 1
        ms["last_ts"] = ts

        std = s.get("std", "0/0")
        if "/" in str(std):
            p, t = std.split("/")
            try:
                ms["std_pass"] += int(p)
                ms["std_total"] += int(t)
            except: pass

        niah = s.get("niah", "—")
        if "/" in str(niah):
            p, t = niah.split("/")
            try:
                ms["niah_pass"] += int(p)
                ms["niah_total"] += int(t)
            except: pass

        if s.get("tok_s", 0) > 0:
            ms["tok_s"].append(s["tok_s"])

    for model, ms in sorted(model_stats.items()):
        runs = ms["runs"]
        std = f"{ms['std_pass']}/{ms['std_total']}" if ms['std_total'] > 0 else "—"
        niah = f"{ms['niah_pass']}/{ms['niah_total']}" if ms['niah_total'] > 0 else "—"
        avg_tok = round(sum(ms["tok_s"]) / len(ms["tok_s"]), 1) if ms["tok_s"] else 0
        short = model.split("/")[-1][:44]
        log(f"{short:<45} {runs:>4} {std:>5} {niah:>5} {avg_tok:>7} {ms['last_ts'][11:19]:>12}")

    log("")
    log("═══ PER-RUN DETAILS ═══")
    for i, r in enumerate(all_results):
        m = r["model"].split("/")[-1][:40]
        s = r.get("summary", {})
        ts = r.get("timestamp", "?")[11:19]
        tests = r.get("tests", [])
        niah = r.get("niah", [])
        fails = [t["test"] for t in tests if t.get("status") != "PASS"]
        niah_fails = [t["test"] for t in niah if t.get("status") != "PASS"]
        err = r.get("error", "")
        detail = ""
        if fails: detail += f" FAIL:{','.join(fails)}"
        if niah_fails: detail += f" NIAH_FAIL:{','.join(niah_fails)}"
        if err: detail += f" ERR:{err[:60]}"
        log(f"  [{ts}] {m}: std={s.get('std','?')} niah={s.get('niah','?')} {s.get('tok_s',0)} tok/s{detail}")

    # Save JSON
    with open("/workspace/atlas/overnight_results.json", "w") as f:
        json.dump({"elapsed": str(elapsed), "iterations": iteration, "results": all_results}, f, indent=2)
    log(f"\nJSON saved to /workspace/atlas/overnight_results.json")

if __name__ == "__main__":
    main()
