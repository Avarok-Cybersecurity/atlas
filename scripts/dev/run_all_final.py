#!/usr/bin/env python3
"""Run coherence + fib + NIAH + concurrency on all models, then EP=2."""
import json, urllib.request, re, time, subprocess, threading, sys

import os

IMAGE = os.environ.get("ATLAS_IMAGE", "atlas-gb10:latest")
HF = os.environ.get(
    "ATLAS_HF_HUB",
    os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
)
# Rank-1 host for EP=2 (single-node default: localhost). Override with
# DGX2=<remote-ip> python3 scripts/dev/run_all_final.py for cross-node EP=2.
DGX2 = os.environ.get("DGX2", "127.0.0.1")
# Rank-0 NCCL bootstrap address. Default to DGX2 if EP_MASTER_ADDR isn't set.
EP_MASTER_ADDR = os.environ.get("EP_MASTER_ADDR", "127.0.0.1")
NEEDLE = "IMPORTANT FACT: The secret project codename is AURORA-VELVET-7742."
FILLER = "The history of cartography stretches back thousands of years. Ancient civilizations in Mesopotamia created clay tablet maps as early as 2300 BCE. The Greeks made significant advances with Eratosthenes calculating the Earth circumference. During the Age of Exploration European cartographers produced increasingly detailed world maps. The Mercator projection introduced in 1569 became the standard for nautical navigation. Modern cartography has been revolutionized by satellite imagery GPS technology and geographic information systems. Digital mapping services now process billions of queries daily.\n\n"

NCCL_ENV = "-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_TIMEOUT=22 -e NCCL_IB_RETRY_CNT=7 -e NCCL_NET_GDR_LEVEL=0 -e NCCL_NET_GDR_C2C=0 -e NCCL_DMABUF_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 -e NCCL_PROTO=Simple -e NCCL_ALGO=Ring -e NCCL_MIN_NCHANNELS=1 -e NCCL_MAX_NCHANNELS=2 -e NCCL_DEBUG=WARN"

def sh(cmd, to=60):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=to).stdout.strip()
    except: return ""

def stop(name, host=None):
    p = f"ssh {host} " if host else ""
    sh(f"{p}sudo docker stop {name} 2>/dev/null", 15); sh(f"{p}sudo docker rm {name} 2>/dev/null", 15)

def start(name, model, port, extra, host=None, hf=HF, maxseq=16384, ep=""):
    p = f"ssh {host} " if host else ""
    rdma = "--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1 " + NCCL_ENV if ep else ""
    sh(f"{p}sudo docker run -d --name {name} --gpus all --ipc=host --network host {rdma} -v {hf}:/root/.cache/huggingface/hub {IMAGE} serve {model} --port {port} --max-seq-len {maxseq} {extra} {ep}", 30)

def ready(name, host=None, secs=420):
    p = f"ssh {host} " if host else ""
    for _ in range(secs//5):
        l = sh(f"{p}sudo docker logs {name} 2>&1 | tail -5", 10)
        if "Listening on" in l: return True
        if "Error:" in l and "FP8" not in l: return False
        time.sleep(5)
    return False

def api(port, prompt, mt=150, to=300):
    body = json.dumps({"model":"t","messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0.0}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=to) as r: return json.loads(r.read())

def test_model(port, is_mistral=False):
    res = {}
    # Quality
    tests = [("capital","What is the capital of France?",50,r"Paris"),
             ("fibonacci","Write the first 10 Fibonacci numbers separated by commas.",80,r"0.*1.*1.*2.*3.*5.*8.*13.*21.*34")]
    if is_mistral:
        tests.append(("count","Count from one to ten using words, separated by commas.",60,r"one.*two.*three.*four.*five.*six.*seven.*eight.*nine.*ten"))
    else:
        tests.append(("count","Count from 1 to 20, separated by commas.",100,r"1.*2.*3.*4.*5.*6.*7.*8.*9.*10"))
    tests.append(("coherence","Explain photosynthesis in 2 sentences.",150,r"sunlight|oxygen|plant|chloro"))
    for name,prompt,mt,pat in tests:
        try:
            r = api(port, prompt, mt)
            c = r["choices"][0]["message"]["content"]
            ok = bool(re.search(pat, c, re.I|re.S))
            tps = r["usage"].get("response_token/s",0)
            res[name] = {"pass": ok, "tok_s": round(tps,1)}
        except Exception as e:
            res[name] = {"pass": False, "tok_s": 0, "err": str(e)[:60]}
    # NIAH
    for nname, tokens in [("niah_4k",4000),("niah_16k",16000)]:
        try:
            f = (FILLER*200)[:tokens*4]; at=int(len(f)*0.5); b=f.find("\n\n",at)
            if b==-1 or b>at+500: b=at
            p = "Read carefully.\n\n"+f[:b]+"\n\n"+NEEDLE+"\n\n"+f[b:]+"\n\nWhat is the secret project codename? Reply with ONLY the codename."
            r = api(port, p, 50)
            c = r["choices"][0]["message"]["content"]; u = r["usage"]
            ok = bool(re.search(r"AURORA.VELVET.7742",c,re.I))
            res[nname] = {"pass": ok, "tok_s": round(u.get("response_token/s",0),1), "ttft": round(u.get("time_to_first_token_ms",0),0), "prompt": u.get("prompt_tokens",0)}
        except Exception as e:
            res[nname] = {"pass": False, "err": str(e)[:60]}
    # Concurrency
    for conc in [1,4,16]:
        try:
            prompt = "Count from 1 upward, one number per line. " + "x "*100
            results = [None]*conc; barrier = threading.Barrier(conc)
            def _r(i): barrier.wait(); results[i] = api(port, prompt, 128)
            ts = [threading.Thread(target=_r, args=(i,)) for i in range(conc)]
            t0=time.perf_counter()
            for t in ts: t.start()
            for t in ts: t.join()
            wall=time.perf_counter()-t0
            total = sum(r["usage"].get("completion_tokens",0) for r in results if r)
            res[f"conc_{conc}x"] = round(total/wall, 1) if wall > 0 else 0
        except Exception as e:
            res[f"conc_{conc}x"] = 0
    return res

# ═══════════════════════════════════════════════════════════════
# Single-GPU models — parallel pairs (dgx1 + dgx2)
# ═══════════════════════════════════════════════════════════════
MODELS = [
    # (model, extra_args, node, short_name)
    ("Sehyo/Qwen3.5-35B-A3B-NVFP4", "--speculative --scheduling-policy slai --max-batch-size 16", "dgx1", "35B+MTP"),
    ("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", "--speculative", "dgx2", "80B+MTP"),
    ("Sehyo/Qwen3.5-122B-A10B-NVFP4", "", "dgx1", "122B"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", "", "dgx2", "Nano 30B"),
    ("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4", "", "dgx1", "Super 120B"),
    ("mistralai/Mistral-Small-4-119B-2603-NVFP4", "--kv-cache-dtype bf16", "dgx1", "Mistral 119B"),
    ("ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4", "", "dgx1", "VL 30B"),
]

all_results = {}
pairs = []
i = 0
while i < len(MODELS):
    pair = [MODELS[i]]
    if i+1 < len(MODELS) and MODELS[i+1][2] != MODELS[i][2]:
        pair.append(MODELS[i+1]); i += 2
    else:
        i += 1
    pairs.append(pair)

for pair in pairs:
    # Start
    for model, extra, node, short in pair:
        host = DGX2 if node == "dgx2" else None
        hf_path = HF
        cname = f"atlas-{node}"
        stop(cname, host)
        print(f"Starting {short} on {node}...", file=sys.stderr)
        start(cname, model, 8888, extra, host, hf_path)

    # Wait
    for model, extra, node, short in pair:
        host = DGX2 if node == "dgx2" else None
        cname = f"atlas-{node}"
        if ready(cname, host):
            print(f"  ✓ {short} ready", file=sys.stderr)
        else:
            print(f"  ✗ {short} FAILED", file=sys.stderr)

    # Test
    for model, extra, node, short in pair:
        host = DGX2 if node == "dgx2" else None
        port = 8888
        is_mistral = "mistral" in model.lower()
        try:
            # For dgx2, we need to route API calls through dgx2's IP
            if host:
                # Monkey-patch api to use dgx2 IP
                def api_remote(port, prompt, mt=150, to=300):
                    body = json.dumps({"model":"t","messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0.0}).encode()
                    req = urllib.request.Request(f"http://{DGX2}:{port}/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
                    with urllib.request.urlopen(req, timeout=to) as r: return json.loads(r.read())
                old_api = globals()['api']
                globals()['api'] = api_remote
                res = test_model(port, is_mistral)
                globals()['api'] = old_api
            else:
                res = test_model(port, is_mistral)
            all_results[short] = res
            std_pass = sum(1 for k in ["capital","fibonacci","count","coherence"] if res.get(k,{}).get("pass"))
            niah_pass = sum(1 for k in ["niah_4k","niah_16k"] if res.get(k,{}).get("pass"))
            avg_tok = sum(res.get(k,{}).get("tok_s",0) for k in ["capital","fibonacci","count","coherence"])/4
            print(f"  {short}: std={std_pass}/4 niah={niah_pass}/2 {avg_tok:.0f} tok/s conc16={res.get('conc_16x',0)} tok/s", file=sys.stderr)
        except Exception as e:
            print(f"  {short}: ERROR {e}", file=sys.stderr)
            all_results[short] = {"error": str(e)[:100]}

    # Cleanup
    for model, extra, node, short in pair:
        host = DGX2 if node == "dgx2" else None
        stop(f"atlas-{node}", host)

# ═══════════════════════════════════════════════════════════════
# EP=2: Qwen3.5-122B
# ═══════════════════════════════════════════════════════════════
print("\n=== EP=2: Qwen3.5-122B ===", file=sys.stderr)
stop("atlas-dgx1"); stop("atlas-dgx2", DGX2)

start("atlas-dgx1", "Sehyo/Qwen3.5-122B-A10B-NVFP4", 8888, "", None, HF, 16384,
      f"--master-addr {EP_MASTER_ADDR} --world-size 2 --rank 0")
# Wait for NCCL listener
for _ in range(84):
    l = sh("sudo docker logs atlas-dgx1 2>&1 | tail -5", 10)
    if "waiting for" in l: break
    if "Error:" in l and "FP8" not in l: break
    time.sleep(5)
start("atlas-dgx2", "Sehyo/Qwen3.5-122B-A10B-NVFP4", 8889, "", DGX2, HF, 16384,
      f"--master-addr {EP_MASTER_ADDR} --world-size 2 --rank 1")

if ready("atlas-dgx1", secs=480):
    print("  EP=2 ready", file=sys.stderr)
    res = test_model(8888)
    all_results["122B EP=2"] = res
    std_pass = sum(1 for k in ["capital","fibonacci","count","coherence"] if res.get(k,{}).get("pass"))
    niah_pass = sum(1 for k in ["niah_4k","niah_16k"] if res.get(k,{}).get("pass"))
    avg_tok = sum(res.get(k,{}).get("tok_s",0) for k in ["capital","fibonacci","count","coherence"])/4
    print(f"  122B EP=2: std={std_pass}/4 niah={niah_pass}/2 {avg_tok:.0f} tok/s", file=sys.stderr)
else:
    print("  EP=2 FAILED TO START", file=sys.stderr)
    all_results["122B EP=2"] = {"error": "failed to start"}

stop("atlas-dgx1"); stop("atlas-dgx2", DGX2)

# ═══════════════════════════════════════════════════════════════
# Final table
# ═══════════════════════════════════════════════════════════════
print(f"\n{'Model':<20} {'Std':>5} {'NIAH':>5} {'Avg tok/s':>10} {'1x':>7} {'4x':>7} {'16x':>7} {'NIAH 4K TTFT':>13} {'NIAH 16K TTFT':>14}")
print("─"*95)
for short in ["35B+MTP","80B+MTP","122B","Nano 30B","Super 120B","Mistral 119B","VL 30B","122B EP=2"]:
    r = all_results.get(short, {})
    if "error" in r:
        print(f"{short:<20} {'ERR':>5} {'ERR':>5} {0:>10} {0:>7} {0:>7} {0:>7} {'':>13} {'':>14}")
        continue
    std_p = sum(1 for k in ["capital","fibonacci","count","coherence"] if r.get(k,{}).get("pass"))
    niah_p = sum(1 for k in ["niah_4k","niah_16k"] if r.get(k,{}).get("pass"))
    avg = sum(r.get(k,{}).get("tok_s",0) for k in ["capital","fibonacci","count","coherence"])/4
    c1 = r.get("conc_1x", 0); c4 = r.get("conc_4x", 0); c16 = r.get("conc_16x", 0)
    t4k = r.get("niah_4k",{}).get("ttft","—"); t16k = r.get("niah_16k",{}).get("ttft","—")
    t4k_s = f"{t4k:.0f}ms" if isinstance(t4k,(int,float)) else str(t4k)
    t16k_s = f"{t16k:.0f}ms" if isinstance(t16k,(int,float)) else str(t16k)
    print(f"{short:<20} {std_p:>3}/4 {niah_p:>3}/2 {avg:>9.1f} {c1:>6.1f} {c4:>6.1f} {c16:>6.1f} {t4k_s:>13} {t16k_s:>14}")

# Save JSON
with open("/workspace/atlas/final_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print("\nJSON: /workspace/atlas/final_results.json")
