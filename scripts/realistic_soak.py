#!/usr/bin/env python3
"""Realistic real-world soak / stability + perf test for a running Atlas server.

Models production traffic, not a synthetic prompt loop. Each request is drawn
from a realistic MIX:
  - long-context fact-extraction (text)
  - tool-calling agent step (sends a `tools` schema; model may emit tool_calls)
  - vision task (image + text)  [only if --vision; the model must support images]
crossed with a thinking on/off mix (per-request `chat_template_kwargs.enable_thinking`).

KV prefix-cache hit rate is TUNABLE via --prefix-hit-rate: each request gets a
shared, byte-identical, cacheable context prefix sized to that fraction of the
prompt, plus a unique fresh remainder. With prefix caching ON the shared prefix
hits cache after warmup, so the measured hit %% tracks the knob — letting you
benchmark across cache regimes (0 = all fresh / worst case; 0.8 = heavy reuse).

It is a CLIENT-side load generator: start the server however you normally do
(e.g. the cuda13.2 container with the Holo perf flags), then point this at it.

Reports (per 30s + final): requests by kind, error rate, thinking split,
tool_calls emitted, FRESH vs CACHED prefill tok/s (+ achieved vs target hit %%),
decode tok/s, reasoning-token share, and request-latency p50/p99. Exits non-zero
above --max-error-rate so it can gate a perf/stability pipeline.

Usage:
  # 2-minute default iteration run (start server on :8890 first):
  python3 scripts/realistic_soak.py --url http://127.0.0.1:8890 --model holo --vision

  # longer soak, heavy cache reuse, text-only:
  python3 scripts/realistic_soak.py --duration 1800 --prefix-hit-rate 0.6
"""
import argparse, base64, json, os, random, statistics, sys, threading, time, urllib.request

# committed test images (varied sizes) live alongside the repo; resolve from
# this script's location so --vision works regardless of CWD.
DEFAULT_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "images")

SIZES = [400, 1000, 2000, 4000, 8000]            # approx prompt tokens
SIZE_WEIGHTS = [30, 30, 22, 12, 6]
CHARS_PER_TOK = 4                                 # rough, for sizing the shared prefix

# One fixed, byte-identical shared "agent session context" — the cacheable
# prefix. Long enough to slice any --prefix-hit-rate fraction from.
_HIST = ("Earlier in this session the agent inspected the repository, ran the build, read the "
         "logs, and recorded that module Aurora-7 emits telemetry on channel G-12 with operator "
         "Lena Vasquez; prior steps confirmed the staging deploy and noted three open incidents. ")
SHARED_CTX = _HIST * 400                           # ~24k tokens of identical, cacheable prefix

TOOLS = [
    {"type": "function", "function": {"name": "run_query",
        "description": "Run a SQL query against the telemetry DB",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
    {"type": "function", "function": {"name": "get_metric",
        "description": "Fetch a named metric for a sector",
        "parameters": {"type": "object", "properties": {"sector": {"type": "string"}, "metric": {"type": "string"}},
                       "required": ["sector", "metric"]}}},
]
DECODE_LENS = [64, 128, 200, 300]


def build_request(rng, kinds, prefix_hit_rate, model, img_urls, think_rate):
    """Return (body, kind, think). body is a full chat-completions payload."""
    target = rng.choices(SIZES, weights=SIZE_WEIGHTS)[0]
    shared_tok = int(target * max(0.0, min(1.0, prefix_hit_rate)))
    unique_tok = max(20, target - shared_tok)
    prefix = SHARED_CTX[: shared_tok * CHARS_PER_TOK]            # identical leading prefix -> cacheable
    nonce = rng.randint(0, 10**9)
    topic = rng.choice(["sector G-12", "module Aurora-7", "channel 7", "operator Lena", "incident 44"])
    unique = f"[req {nonce}] " + (f"Fresh observation about {topic}: value {{}} shifted under load. "
                                  ).format(nonce) * (unique_tok // 16)

    kind = rng.choice(kinds)
    think = rng.random() < think_rate                 # thinking on/off mix
    text_q = {
        "fact": "\nExtract every entity, channel, and operator mentioned as a JSON array.",
        "tool": "\nUsing the tools, fetch the temperature metric for sector G-12, then query the DB for anomalies.",
        "img":  "\nDescribe this image in detail and list every visible UI element.",
    }[kind]
    text = prefix + unique + text_q

    if kind == "img":
        content = [{"type": "text", "text": text},
                   {"type": "image_url", "image_url": {"url": rng.choice(img_urls)}}]
    else:
        content = text
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": rng.choice(DECODE_LENS), "temperature": 0.0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": think}}
    if kind == "tool":
        body["tools"] = TOOLS
    return body, kind, think


def stream_chat(url, body, timeout):
    t0 = time.time(); ttft = None; pt = ct = cached = rtok = 0; toolcall = False; txt = []
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                ln = raw.decode("utf-8", "ignore").strip()
                if not ln.startswith("data:"):
                    continue
                pl = ln[5:].strip()
                if pl == "[DONE]":
                    break
                try:
                    o = json.loads(pl)
                except Exception:
                    continue
                if o.get("usage"):
                    u = o["usage"]; pt = u.get("prompt_tokens", 0); ct = u.get("completion_tokens", 0)
                    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                    rtok = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or u.get("reasoning_tokens", 0) or 0
                ch = o.get("choices") or []
                if ch:
                    d = ch[0].get("delta", {})
                    if (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")) and ttft is None:
                        ttft = time.time() - t0
                    if d.get("content"):
                        txt.append(d["content"])
                    if d.get("tool_calls"):
                        toolcall = True
        return {"ok": True, "ct": ct, "pt": pt, "cached": cached, "rtok": rtok, "txt": "".join(txt),
                "toolcall": toolcall, "el": time.time() - t0, "ttft": ttft or (time.time() - t0)}
    except Exception as e:
        return {"ok": False, "err": str(e)[:140], "el": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8890")
    ap.add_argument("--model", default="holo")
    ap.add_argument("--duration", type=int, default=120, help="soak seconds (default 2 min; override for long soaks)")
    ap.add_argument("--clients", type=int, default=6)
    ap.add_argument("--vision", action="store_true", help="include image requests (model MUST support vision)")
    ap.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR,
                    help="dir of test images cycled across vision requests (default: committed tests/fixtures/images)")
    ap.add_argument("--prefix-hit-rate", type=float, default=0.0,
                    help="target fraction (0..1) of each prompt that is a shared cacheable prefix")
    ap.add_argument("--think-rate", type=float, default=0.5, help="fraction of requests with thinking ON")
    ap.add_argument("--conv-rate", type=float, default=0.5,
                    help="fraction of clients that run GROWING multi-turn conversations (rest do one-shot mix)")
    ap.add_argument("--rolling-context", type=int, default=64000,
                    help="grow each conversation thread until ~this many tokens, then reset (adjustable)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-error-rate", type=float, default=2.0, help="fail (exit 1) above this %% errors")
    args = ap.parse_args()

    chat_url = args.url.rstrip("/") + "/v1/chat/completions"
    try:
        urllib.request.urlopen(args.url.rstrip("/") + "/v1/models", timeout=5)
    except Exception as e:
        print(f"server not reachable at {args.url}: {e}", file=sys.stderr); sys.exit(2)

    kinds = ["fact", "tool"]
    img_urls = []
    if args.vision:
        try:
            files = sorted(f for f in os.listdir(args.image_dir)
                           if f.lower().endswith((".png", ".jpg", ".jpeg")))
            for f in files:
                ext = f.rsplit(".", 1)[-1].lower(); mt = "jpeg" if ext in ("jpg", "jpeg") else ext
                with open(os.path.join(args.image_dir, f), "rb") as fh:
                    img_urls.append(f"data:image/{mt};base64," + base64.b64encode(fh.read()).decode())
            if img_urls:
                kinds.append("img")
                print(f"vision: {len(img_urls)} test images from {args.image_dir}", flush=True)
            else:
                print(f"--vision set but no images in {args.image_dir}; text-only", file=sys.stderr)
        except Exception as e:
            print(f"--vision set but image dir unavailable ({e}); text-only", file=sys.stderr)

    st = {"req": 0, "err": 0, "ct": 0, "pt": 0, "cached": 0, "rtok": 0, "tc": 0, "think": 0,
          "fact": 0, "tool": 0, "img": 0, "conv": 0, "conv_turns": 0, "max_thread_tok": 0}
    lat, lock, stop = [], threading.Lock(), time.time() + args.duration
    FOLLOWUPS = ["Continue your analysis.", "Elaborate on the most important point.",
                 "What did you miss? Add detail.", "Now consider the edge cases.",
                 "Summarize so far, then go deeper.", "Cross-check that against the context above."]

    def record(r, kind, think):
        with lock:
            st["req"] += 1; st[kind] += 1
            if think:
                st["think"] += 1
            if r["ok"]:
                st["ct"] += r["ct"]; st["pt"] += r["pt"]; st["cached"] += r["cached"]
                st["rtok"] += r["rtok"]; st["tc"] += int(r["toolcall"]); lat.append(r["el"])
            else:
                # Count failures so errs=/err_rate and the --max-error-rate exit-1
                # gate actually work (previously st["err"] was never incremented).
                st["err"] += 1
                print(f"  [err] {kind}: {r.get('err', 'unknown')}", file=sys.stderr, flush=True)
        return r["ok"]

    def oneshot_client(cid, rng):
        consec = 0
        while time.time() < stop:
            body, kind, think = build_request(rng, kinds, args.prefix_hit_rate, args.model, img_urls, args.think_rate)
            ok = record(stream_chat(chat_url, body, args.timeout), kind, think)
            consec = 0 if ok else consec + 1
            if consec:
                time.sleep(min(2.0, 0.2 * consec))

    def conversation_client(cid, rng):
        """A GROWING multi-turn thread: resend the whole conversation each turn so
        the prefix (all prior turns) is reused/cached, until it hits
        --rolling-context tokens, then reset. Models a real chat session."""
        consec = 0
        while time.time() < stop:
            # seed the thread with a shared cacheable preamble + opening question
            body, _, think = build_request(rng, ["fact"], args.prefix_hit_rate, args.model, img_urls, args.think_rate)
            msgs = body["messages"]
            thread_tok = 0; turns = 0
            with lock:
                st["conv"] += 1
            while time.time() < stop and thread_tok < args.rolling_context:
                turn_think = rng.random() < args.think_rate
                tb = {"model": args.model, "messages": msgs, "max_tokens": rng.choice(DECODE_LENS),
                      "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True},
                      "chat_template_kwargs": {"enable_thinking": turn_think}}
                r = stream_chat(chat_url, tb, args.timeout)
                ok = record(r, "fact", turn_think)
                if not ok:
                    consec += 1; time.sleep(min(2.0, 0.2 * consec)); break
                consec = 0; turns += 1
                thread_tok = r["pt"] + r["ct"]                       # thread size after this turn
                msgs = msgs + [{"role": "assistant", "content": r["txt"] or "(no content)"},
                               {"role": "user", "content": rng.choice(FOLLOWUPS)}]
            with lock:
                st["conv_turns"] += turns
                st["max_thread_tok"] = max(st["max_thread_tok"], thread_tok)

    n_conv = int(round(args.clients * max(0.0, min(1.0, args.conv_rate))))

    def client(cid):
        rng = random.Random(cid * 7919 + args.seed)
        (conversation_client if cid < n_conv else oneshot_client)(cid, rng)

    t0 = time.time()
    print(f"clients: {n_conv} conversation (roll to {args.rolling_context} tok) + "
          f"{args.clients - n_conv} one-shot; think-rate {args.think_rate}", flush=True)
    threads = [threading.Thread(target=client, args=(c,)) for c in range(args.clients)]
    for t in threads:
        t.start()
    while time.time() < stop:
        time.sleep(30)
        with lock:
            fresh = st["pt"] - st["cached"]
            print(f"  [{int(time.time()-t0)}s] reqs={st['req']} errs={st['err']} "
                  f"(fact={st['fact']} tool={st['tool']} img={st['img']}) think={st['think']} "
                  f"tool_calls={st['tc']} fresh_prefill={fresh} cached={st['cached']} gen={st['ct']}", flush=True)
    for t in threads:
        t.join()

    el = time.time() - t0
    fresh = st["pt"] - st["cached"]
    err_rate = 100 * st["err"] / max(1, st["req"])
    hit = 100 * st["cached"] / max(1, st["pt"])
    p50 = statistics.median(lat) if lat else 0
    p99 = sorted(lat)[int(len(lat) * 0.99)] if lat else 0
    print(f"\nSOAK DONE {el:.0f}s: {st['req']} reqs, {st['err']} errs ({err_rate:.1f}%) | "
          f"kinds fact={st['fact']} tool={st['tool']} img={st['img']} | "
          f"thinking {st['think']}/{st['req']} ({100*st['think']/max(1,st['req']):.0f}%) | "
          f"tool_calls emitted={st['tc']}", flush=True)
    print(f"  CONVERSATIONS: {st['conv']} threads, {st['conv_turns']} total turns, "
          f"deepest thread {st['max_thread_tok']} tok (roll @ {args.rolling_context})", flush=True)
    print(f"  PREFILL: {fresh} fresh tok ({st['cached']} cached, {hit:.0f}% hit vs "
          f"{100*args.prefix_hit_rate:.0f}% target) = {fresh/el:.0f} tok/s", flush=True)
    print(f"  DECODE:  {st['ct']} gen tok = {st['ct']/el:.0f} tok/s (reasoning {st['rtok']} tok, "
          f"{100*st['rtok']/max(1,st['ct']):.0f}%) ; latency p50={p50:.2f}s p99={p99:.2f}s", flush=True)

    if err_rate > args.max_error_rate:
        print(f"FAIL: error rate {err_rate:.1f}% > {args.max_error_rate}%", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
