#!/usr/bin/env python3
"""SBR M1 benchmark — warm-hit TTFT vs conversation depth under cache pressure.

Reproduces the "replay grows as slots fill" symptom and measures whether the
tail-pin eviction (ATLAS_SBR_TAIL_PIN, set on the SERVER) flattens it.

Workload:
  * ONE "main" multi-turn conversation whose shared prefix grows each turn
    (system document + accumulated turns). Each turn re-sends the full history
    -> prefix-cache + SSM-snapshot warm hit; replay = matched - nearest_snapshot.
  * Between main turns, fire DISTRACTORS distinct one-shot conversations to churn
    the snapshot LRU (pool default 16 slots). Baseline LRU evicts the main
    conversation's deep tail; tail-pin protects it.

Metric: TTFT (time-to-first-token, streaming) of each MAIN turn vs depth.
Also records the assistant's first-token id per turn for A/B exactness check.

Run (on dgx2, server already up on :8888):
  python3 sbr_bench.py --label tailpin_on --out tailpin_on.json --turns 16 --distractors 24
"""
import argparse, json, time, sys
import urllib.request

WORDS = ("the model integrates state across tokens while attention reads the "
         "key value cache and the recurrent layer accumulates a hidden matrix "
         "that decays per step according to a learned gate and beta factor "
         "producing an output that the next layer consumes in sequence").split()


def filler(n_tokens):
    # ~0.75 tokens/word -> words ~= tokens / 0.75; deterministic, varied by seed
    n_words = int(n_tokens / 0.75)
    return " ".join(WORDS[i % len(WORDS)] for i in range(n_words))


def chat_stream(base_url, model, messages, max_tokens):
    """POST streaming chat; return (ttft_s, total_s, text, first_tok)."""
    body = json.dumps({
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(base_url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    first_tok = None
    text = []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = ev.get("choices", [{}])[0].get("delta", {})
            chunk = delta.get("content")
            if chunk:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    first_tok = chunk
                text.append(chunk)
    total = time.perf_counter() - t0
    return ttft, total, "".join(text), first_tok


def run_distractors(base_url, model, n, base_tokens, idx):
    for d in range(n):
        doc = f"distractor {idx}_{d} " + filler(base_tokens) + f" unique salt {idx}-{d}-xyz"
        msgs = [{"role": "system", "content": doc},
                {"role": "user", "content": f"Summarize point {d} in one sentence."}]
        try:
            chat_stream(base_url, model, msgs, 32)
        except Exception as e:
            print(f"  distractor {idx}_{d} err: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8888/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4")
    ap.add_argument("--turns", type=int, default=16)
    ap.add_argument("--distractors", type=int, default=24)
    ap.add_argument("--base-tokens", type=int, default=6000)
    ap.add_argument("--distractor-tokens", type=int, default=400,
                    help="size of each distractor conv (just needs to occupy a snapshot slot)")
    ap.add_argument("--resp-tokens", type=int, default=200)
    args = ap.parse_args()

    system_doc = ("You are a careful assistant. Reference document follows.\n" +
                  filler(args.base_tokens))
    history = [{"role": "system", "content": system_doc}]
    rows = []
    questions = [
        "What does the recurrent layer accumulate?", "How does the gate affect state?",
        "Summarize the document in one line.", "What consumes the output?",
        "Explain the beta factor.", "What decays per step?",
        "Relate attention to the kv cache.", "Why is sequence order important?",
        "What is the hidden matrix?", "Describe state integration.",
        "What does the next layer read?", "How is the output produced?",
    ]

    # warm up the main prefix once (cold), then churn + measure resumes
    for turn in range(args.turns):
        q = questions[turn % len(questions)]
        history.append({"role": "user", "content": f"Turn {turn}: {q}"})
        # churn the snapshot LRU between resumes (skip before first turn)
        if turn > 0:
            run_distractors(args.base_url, args.model, args.distractors, args.distractor_tokens, turn)
        approx_depth = sum(len(m["content"].split()) for m in history) * 4 // 3
        ttft, total, text, first_tok = chat_stream(
            args.base_url, args.model, history, args.resp_tokens)
        history.append({"role": "assistant", "content": text})
        rows.append({"turn": turn, "approx_depth_tokens": approx_depth,
                     "ttft_s": ttft, "total_s": total, "first_tok": first_tok})
        print(f"[{args.label}] turn {turn:2d} depth~{approx_depth:6d} "
              f"TTFT={ttft:.3f}s total={total:.3f}s", flush=True)

    out = {"label": args.label, "turns": args.turns, "distractors": args.distractors,
           "base_tokens": args.base_tokens, "rows": rows}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    ttfts = [r["ttft_s"] for r in rows if r["ttft_s"]]
    if ttfts:
        print(f"\n[{args.label}] TTFT first={ttfts[0]:.3f}s last={ttfts[-1]:.3f}s "
              f"max={max(ttfts):.3f}s mean={sum(ttfts)/len(ttfts):.3f}s")


if __name__ == "__main__":
    main()
