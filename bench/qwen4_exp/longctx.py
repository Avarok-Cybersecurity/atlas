#!/usr/bin/env python3
"""Decode and prefill at REAL context length, streamed.

Every throughput number in the qwen4exp optimisation series was taken at 60-460
token prompts. This config serves 128K. Attention cost scales with context and
the MoE/projection costs do not, so the short-prompt profile says nothing about
the regime the config exists for — this measures it.

METHOD. Streaming, so prefill and decode are separated by measurement rather
than by arithmetic:

    TTFT          first SSE token chunk         -> prefill = prompt_tok / ttft
    decode rate   (n_tok - 1) / (t_last - t_first)

The first token is excluded from the decode window because it is the one TTFT
already paid for; including it would fold prefill into the decode rate at
exactly the long contexts where prefill dominates.

COLD BY CONSTRUCTION. Prefix caching is on (standing rule), so every prompt is
built from a DISTINCT random word stream with a distinct seed. Two prompts at
the same length share no prefix, and re-running does not serve a warm cache —
otherwise this would measure the radix tree, not the model.

Usage:  longctx.py [--ctx 4096,16384,32768,65536,100000] [--conc 1] [--gen 300]
"""
import argparse, json, os, random, sys, threading, time, urllib.request

PORT = int(os.environ.get("ATLAS_BENCH_PORT", "8888"))
MODEL = os.environ.get("ATLAS_BENCH_MODEL", "qwen4exp-nvfp4")

# Ordinary English-ish words: the tokenizer splits real words predictably,
# where random character noise would inflate tokens/word and make the target
# length unpredictable.
WORDS = """time person year way day thing man world life hand part child eye woman place work week
case point government company number group problem fact system program question night area money story
fact month lot right study book eye job word business issue side kind head house service friend father
power hour game line end member law car city community name president team minute idea kid body
information back parent face others level office door health person art war history party result change
morning reason research girl guy moment air teacher force education foot boy age policy process music
market sense nation plan college interest death experience effect use class control care field development
role effort rule image mind data method bank practice quality pressure answer source growth model""".split()


def make_prompt(target_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    # MEASURED on this tokenizer with this corpus: ~1.00 tokens per word, so
    # the word count IS the token target. (An earlier 1.35 multiplier — from
    # assuming 0.75 tok/word — overshot 100K into 135K and the server
    # correctly 400'd it against a 131072 cap.) The actual count is always
    # reported from the server's own usage, never assumed.
    n_words = int(target_tokens * 1.0)
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return (
        f"Reference log {seed:08x}. Read the following record, then answer.\n\n"
        f"{body}\n\n"
        "Summarise the record above in a few sentences, then explain your reasoning."
    )


def stream_one(prompt: str, gen: int, out: list, idx: int):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    n_tok = 0
    prompt_tok = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                u = d.get("usage") or {}
                if u.get("prompt_tokens"):
                    prompt_tok = u["prompt_tokens"]
                ch = d.get("choices") or []
                # Count BOTH fields: with thinking on, a long-context turn can
                # spend its whole budget in `reasoning_content`, and a decoded
                # token costs the same whichever field carries it. Counting
                # only `content` reported zero tokens for exactly those turns.
                delta = (ch[0].get("delta") or {}) if ch else {}
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_tok += 1
    except Exception as e:
        out[idx] = {"err": f"{type(e).__name__}: {e}"[:160]}
        return
    wall = time.perf_counter() - t0
    if t_first is None or n_tok < 2:
        out[idx] = {"err": f"only {n_tok} streamed tokens"}
        return
    ttft = t_first - t0
    dec_s = t_last - t_first
    out[idx] = {
        "prompt_tok": prompt_tok,
        "gen_tok": n_tok,
        "ttft_s": ttft,
        "prefill_tps": (prompt_tok / ttft) if prompt_tok and ttft > 0 else 0.0,
        # First token excluded: TTFT already paid for it.
        "decode_tps": (n_tok - 1) / dec_s if dec_s > 0 else 0.0,
        "wall_s": wall,
    }


def run(ctx: int, conc: int, gen: int, seed0: int):
    out = [None] * conc
    ths = [threading.Thread(target=stream_one,
                            args=(make_prompt(ctx, seed0 + i), gen, out, i))
           for i in range(conc)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in out if r and "err" not in r]
    for i, r in enumerate(out):
        if r and "err" in r:
            print(f"    req{i}: ERROR {r['err']}")
    if not ok:
        return None
    ptok = ok[0]["prompt_tok"]
    agg = sum(r["decode_tps"] for r in ok)
    print(f"  ctx~{ctx:>6}  prompt_tok={ptok:>6}  C={conc}  "
          f"TTFT {min(r['ttft_s'] for r in ok):6.1f}-{max(r['ttft_s'] for r in ok):6.1f}s  "
          f"prefill {sum(r['prefill_tps'] for r in ok):7.1f} tok/s  "
          f"decode/seq {agg/len(ok):6.2f}  aggregate {agg:6.2f} tok/s  "
          f"wall {wall:5.1f}s")
    return {"ctx": ctx, "prompt_tok": ptok, "conc": conc,
            "decode_per_seq": agg / len(ok), "decode_agg": agg,
            "prefill_tps": sum(r["prefill_tps"] for r in ok),
            "ttft_max": max(r["ttft_s"] for r in ok)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="4096,16384,32768,65536,100000")
    ap.add_argument("--conc", default="1")
    ap.add_argument("--gen", type=int, default=300)
    a = ap.parse_args()
    ctxs = [int(x) for x in a.ctx.split(",")]
    concs = [int(x) for x in a.conc.split(",")]
    print(f"long-context decode — port {PORT}, gen={a.gen}, cold prompts (unique seeds)")
    rows = []
    # Clock-seeded: a fixed seed replays a prompt the prefix cache still
    # holds, which turns a cold-prefill measurement into a warm restore.
    seed = int(time.time()) & 0xffffff
    for c in concs:
        print(f"\n--- C={c} ---")
        for ctx in ctxs:
            r = run(ctx, c, a.gen, seed)
            seed += 1000
            if r:
                rows.append(r)
    if rows:
        base = rows[0]["decode_per_seq"]
        print("\nctx_tokens  C  decode/seq  vs_shortest  aggregate  prefill_tok/s  TTFT_s")
        for r in rows:
            print(f"{r['prompt_tok']:>10}  {r['conc']}  {r['decode_per_seq']:10.2f}  "
                  f"{r['decode_per_seq']/base:10.2f}x  {r['decode_agg']:9.2f}  "
                  f"{r['prefill_tps']:13.1f}  {r['ttft_max']:6.1f}")
