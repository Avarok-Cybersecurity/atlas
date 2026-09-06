#!/usr/bin/env python3
"""Concurrent streaming decode: launch C requests at once (distinct salted prompts of ~prompt_tokens),
report per-stream decode tok/s (server-attested completion_tokens / decode wall), aggregate tok/s, TTFT,
and sample host free memory + GPU memory every 5 s during the run (host swap pressure watch)."""
import json, sys, time, urllib.request, random, argparse, statistics, threading, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8899)
ap.add_argument("--model", default="qwen3.8-flash-next")
ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
ap.add_argument("--prompt-tokens", type=int, default=2000)
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--salt", type=int, default=7)
args = ap.parse_args()

WORDS = ("kernel stream tensor cache block schedule verify draft router expert layer norm gate highway "
         "carry snapshot commit rewind window hash gather slot arena page token prefix budget latency "
         "bandwidth roofline launch cooperative barrier reduction").split()

def prompt(n_tokens, salt):
    rng = random.Random(salt)
    body = " ".join(rng.choice(WORDS) for _ in range(int(n_tokens / 1.037)))
    return (f"Session salt {salt}. Here is a log:\n\n{body}\n\nWrite a complete Rust implementation of an LRU "
            f"cache with generic key and value types and unit tests, explaining each design decision in comments.")

def stream_one(p, out):
    body = {"model": args.model, "messages": [{"role": "user", "content": p}], "max_tokens": args.max_tokens,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); times = []; usage = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                if not line.startswith(b"data:"): continue
                payload = line[5:].strip()
                if payload == b"[DONE]": break
                d = json.loads(payload)
                if d.get("usage"): usage = d["usage"]
                ch = d.get("choices") or []
                if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")):
                    times.append(time.perf_counter())
    except Exception as e:
        out.append({"error": str(e)}); return
    if len(times) < 5: out.append({"error": f"only {len(times)} chunks"}); return
    ct = usage["completion_tokens"] if usage else len(times)
    out.append({"ttft_s": times[0] - t0, "decode_wall_s": times[-1] - times[0], "completion_tokens": ct,
                "prompt_tokens": usage.get("prompt_tokens") if usage else None,
                "tok_s": (ct - 1) / (times[-1] - times[0])})

def sample_mem(stop, samples):
    while not stop.is_set():
        try:
            free = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.split("\n")[1].split()
            avail = int(free[6])
            swap = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.split("\n")[2].split()
            samples.append((time.time(), avail, int(swap[2])))
        except Exception: pass
        stop.wait(5)

print(f"FINGERPRINT port={args.port} model={args.model} prompt_tokens~{args.prompt_tokens} max_tokens={args.max_tokens} "
      f"temp=0 effort=low stream=1 date={time.strftime('%Y-%m-%dT%H:%M:%S')}")
for c in args.concurrency:
    out = []; stop = threading.Event(); samples = []
    mon = threading.Thread(target=sample_mem, args=(stop, samples), daemon=True); mon.start()
    t0 = time.perf_counter()
    ths = [threading.Thread(target=stream_one, args=(prompt(args.prompt_tokens, args.salt * 100 + c * 10 + i), out)) for i in range(c)]
    for t in ths: t.start()
    for t in ths: t.join()
    wall = time.perf_counter() - t0; stop.set(); mon.join(timeout=6)
    ok = [o for o in out if "error" not in o]
    errs = [o for o in out if "error" in o]
    total_tokens = sum(o["completion_tokens"] for o in ok)
    per = [o["tok_s"] for o in ok]
    min_avail = min((s[1] for s in samples), default=-1); max_swap = max((s[2] for s in samples), default=-1)
    print(f"C={c}: streams_ok={len(ok)}/{c} errors={len(errs)} per_stream_tok_s={[round(x,1) for x in per]} "
          f"median_per_stream={statistics.median(per) if per else 0:.1f} aggregate_tok_s={total_tokens/wall:.1f} "
          f"ttft_s={[round(o['ttft_s'],1) for o in ok]} prompt_tokens={[o['prompt_tokens'] for o in ok][:1]} "
          f"wall={wall:.1f}s min_host_avail_GB={min_avail} max_swap_used_GB={max_swap}")
    for e in errs: print("  error:", e["error"][:200])
