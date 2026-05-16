#!/usr/bin/env python3
"""Multi-sample reasoning-boundary evaluation harness (the gate instrument).

The single methodological failure of all prior think->content boundary work
was n=1 A/B on a temp=0.6 stochastic decode -- every "X regressed / helped"
conclusion was statistically void. This harness makes each layer KNOWABLE:
N>=10 generations per config, scored deterministically, aggregated with
mean +/- stdev, and gated by a two-proportion z-test. No layer is trusted on
a single sample.

Subcommands
-----------
  run          Run N gens of the chess prompt (stream + blocking) + 3
               regression prompts; write a JSON results file + a one-line
               VERDICT to stdout.
  gate         Compare two `run` JSON files (baseline vs candidate) with the
               statistical gate; exit 0 iff candidate passes.
  determinism  DEER byte-equality check: two endpoints/configs, fixed seed,
               same prompt -> assert token/text identical (Layer A acceptance).

Stdlib only (urllib), mirrors bench/qwen36_correctness.py. SSE parsing
reuses the `data: ` line-reader used across bench/.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "chess_prompt.json")

# ---- scoring constants (SSOT — chat_stream/regen.rs::classify_done must mirror) ----
JS_MARKERS = (
    "function ", "=>", "addEventListener", "requestAnimationFrame",
    "THREE.", "const ", "class ",
)
ROLE_LEAK = ("\nassistant", "\nuser", "\ntool")
JS_SUBSTANCE_MIN = 8           # success threshold (CLI-tunable)
REPEAT_BLOCK_CHARS = 200       # repeated >=this-char block twice => degenerate


def post_blocking(url, body, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        r = json.loads(resp.read())
    ch = r["choices"][0]
    msg = ch.get("message", {})
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "finish_reason": ch.get("finish_reason"),
        "tool_calls": msg.get("tool_calls") or [],
        "usage": r.get("usage", {}),
    }


def post_streaming(url, body, timeout):
    body = dict(body, stream=True)
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    content, reasoning, finish, usage, tool_calls = [], [], None, {}, []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            ch = d["choices"][0] if d.get("choices") else {}
            delta = ch.get("delta", {})
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            if delta.get("tool_calls"):
                tool_calls.extend(delta["tool_calls"])
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            if d.get("usage"):
                usage = d["usage"]
    return {
        "content": "".join(content), "reasoning": "".join(reasoning),
        "finish_reason": finish, "tool_calls": tool_calls, "usage": usage,
    }


def _has_repeated_block(text):
    n = len(text)
    if n < 2 * REPEAT_BLOCK_CHARS:
        return False
    seen = set()
    step = 50
    for i in range(0, n - REPEAT_BLOCK_CHARS, step):
        blk = text[i:i + REPEAT_BLOCK_CHARS]
        if blk in seen:
            return True
        seen.add(blk)
    return False


def score(content, finish_reason):
    """Deterministic per-generation score. Mirrors regen.rs::classify_done."""
    low = content.lower()
    doctype = low.count("<!doctype")
    has_html_close = "</html>" in low
    script_close = low.count("</script>")
    js = sum(content.count(m) for m in JS_MARKERS)
    role_leak = any(r in content for r in ROLE_LEAK)
    esc_nl = "\\n" in content and content.count("\\n") > 5
    repeated = _has_repeated_block(content)
    degenerate = role_leak or esc_nl or repeated or doctype >= 2
    success = (
        doctype == 1 and has_html_close and script_close >= 1
        and js >= JS_SUBSTANCE_MIN and not degenerate
        and finish_reason in ("stop", "tool_calls")
    )
    return {
        "finish_reason": finish_reason, "doctype": doctype,
        "html_close": has_html_close, "script_close": script_close,
        "js_markers": js, "role_leak": role_leak, "esc_nl": esc_nl,
        "repeated_block": repeated, "degenerate": degenerate,
        "success": bool(success),
    }


def _agg(samples):
    if not samples:
        return {}
    n = len(samples)
    succ = sum(1 for s in samples if s["success"])
    js = [s["js_markers"] for s in samples]
    restart = sum(1 for s in samples if s["doctype"] >= 2)
    hist = {}
    for s in samples:
        hist[s["finish_reason"]] = hist.get(s["finish_reason"], 0) + 1
    return {
        "n": n, "success": succ, "success_rate": succ / n,
        "mean_js": statistics.mean(js),
        "stdev_js": statistics.pstdev(js) if n > 1 else 0.0,
        "restart_rate": restart / n,
        "finish_hist": hist,
        "samples": samples,
    }


def run(args):
    fx = json.load(open(FIXTURE))
    base = {"model": args.model, "temperature": args.temperature,
            "max_tokens": args.max_tokens}
    out = {"meta": {"url": args.url, "model": args.model, "n": args.n,
                    "temperature": args.temperature, "ts": time.time(),
                    "config": args.config}}
    cep = f"{args.url}/v1/chat/completions"

    # ---- chess prompt: N gens, stream AND blocking ----
    chess = fx["chess"]
    for mode, fn in (("stream", post_streaming), ("blocking", post_blocking)):
        samples = []
        for i in range(args.n):
            b = dict(base, messages=chess["messages"])
            if args.seed is not None:
                b["seed"] = args.seed + i
            try:
                r = fn(cep, b, args.timeout)
            except Exception as e:  # noqa: BLE001
                samples.append({"finish_reason": f"ERR:{type(e).__name__}",
                                "doctype": 0, "html_close": False,
                                "script_close": 0, "js_markers": 0,
                                "role_leak": False, "esc_nl": False,
                                "repeated_block": False, "degenerate": True,
                                "success": False})
                print(f"  [{args.config}/chess/{mode}] {i+1}/{args.n} ERROR {e}",
                      file=sys.stderr)
                continue
            sc = score(r["content"], r["finish_reason"])
            samples.append(sc)
            print(f"  [{args.config}/chess/{mode}] {i+1}/{args.n} "
                  f"succ={sc['success']} js={sc['js_markers']} "
                  f"doctype={sc['doctype']} fr={sc['finish_reason']}",
                  file=sys.stderr)
        out[f"chess_{mode}"] = _agg(samples)

    # ---- regression prompts: N gens, blocking (cheaper, deterministic enough) ----
    for key in ("qa_thinking", "tool_call", "non_thinking"):
        spec = fx[key]
        samples = []
        for i in range(args.n):
            b = dict(base, messages=spec["messages"])
            for k in ("tools", "tool_choice", "chat_template_kwargs"):
                if k in spec:
                    b[k] = spec[k]
            if args.seed is not None:
                b["seed"] = args.seed + i
            try:
                r = post_blocking(cep, b, args.timeout)
            except Exception as e:  # noqa: BLE001
                samples.append({"ok": False, "err": f"{type(e).__name__}"})
                continue
            # regression "ok" = produced something sane for its kind
            if key == "tool_call":
                ok = bool(r["tool_calls"]) or "get_weather" in (r["content"] or "")
            elif key == "non_thinking":
                ok = bool((r["content"] or "").strip()) and "<think>" not in (r["content"] or "")
            else:  # qa_thinking
                ok = bool((r["content"] or "").strip()) and r["finish_reason"] in ("stop", "length")
            samples.append({"ok": bool(ok), "finish_reason": r["finish_reason"]})
        ok_n = sum(1 for s in samples if s.get("ok"))
        out[f"reg_{key}"] = {"n": len(samples), "ok": ok_n,
                             "ok_rate": ok_n / max(1, len(samples)),
                             "samples": samples}

    json.dump(out, open(args.out, "w"), indent=2)
    cs, cb = out["chess_stream"], out["chess_blocking"]
    verdict = (
        f"VERDICT cfg={args.config} N={args.n} "
        f"chess_stream_succ={cs['success']}/{cs['n']} "
        f"chess_blocking_succ={cb['success']}/{cb['n']} "
        f"meanJS_stream={cs['mean_js']:.1f}±{cs['stdev_js']:.1f} "
        f"restart_stream={cs['restart_rate']:.2f} "
        f"reg_qa={out['reg_qa_thinking']['ok']}/{out['reg_qa_thinking']['n']} "
        f"reg_tool={out['reg_tool_call']['ok']}/{out['reg_tool_call']['n']} "
        f"reg_nothink={out['reg_non_thinking']['ok']}/{out['reg_non_thinking']['n']} "
        f"-> {args.out}"
    )
    print(verdict)
    return 0


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _ztest(k1, n1, k2, n2):
    """One-sided two-proportion z-test: is p1 (candidate) > p2 (baseline)?
    Returns (p_value, delta)."""
    if n1 == 0 or n2 == 0:
        return 1.0, 0.0
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0 if p1 > p2 else 1.0), p1 - p2
    z = (p1 - p2) / se
    return 1.0 - _norm_cdf(z), p1 - p2


def gate(args):
    base = json.load(open(args.baseline))
    cand = json.load(open(args.candidate))
    overall_pass = True
    for mode in ("chess_stream", "chess_blocking"):
        b, c = base[mode], cand[mode]
        pval, delta = _ztest(c["success"], c["n"], b["success"], b["n"])
        # near-ceiling fallback: meanJS improves >=1 stdev, non-overlapping
        js_gain = c["mean_js"] - b["mean_js"]
        js_pass = (js_gain >= max(b["stdev_js"], 1.0)
                   and (c["mean_js"] - c["stdev_js"]) > (b["mean_js"] + b["stdev_js"]))
        mode_pass = (pval < 0.05 and delta >= 0.20) or js_pass
        overall_pass &= mode_pass
        print(f"[gate] {mode}: base={b['success']}/{b['n']} "
              f"cand={c['success']}/{c['n']} dSucc={delta:+.2f} p={pval:.4f} "
              f"dMeanJS={js_gain:+.1f} -> {'PASS' if mode_pass else 'FAIL'}")
    # regressions: candidate must not significantly drop on any of the 3
    for key in ("reg_qa_thinking", "reg_tool_call", "reg_non_thinking"):
        b, c = base[key], cand[key]
        pval, delta = _ztest(b["ok"], b["n"], c["ok"], c["n"])  # base > cand?
        regressed = pval < 0.05 and delta >= 0.20
        if regressed:
            overall_pass = False
        print(f"[gate] {key}: base_ok={b['ok']}/{b['n']} "
              f"cand_ok={c['ok']}/{c['n']} -> "
              f"{'REGRESSED' if regressed else 'ok'}")
    print(f"[gate] OVERALL: {'PASS — keep layer' if overall_pass else 'FAIL — revert layer'}")
    return 0 if overall_pass else 1


def determinism(args):
    """DEER acceptance: same prompt+seed, two configs, byte-identical content.
    Run baseline endpoint and ATLAS_DEER_FORCE_ROLLBACK endpoint; assert equal."""
    fx = json.load(open(FIXTURE))
    msgs = fx["chess"]["messages"]
    body = {"model": args.model, "messages": msgs, "temperature": 0.0,
            "max_tokens": args.max_tokens, "seed": args.seed}
    a = post_blocking(f"{args.url_a}/v1/chat/completions", body, args.timeout)
    b = post_blocking(f"{args.url_b}/v1/chat/completions", body, args.timeout)
    same = a["content"] == b["content"] and a["reasoning"] == b["reasoning"]
    if same:
        print(f"[determinism] PASS — byte-identical ({len(a['content'])} chars)")
        return 0
    # locate first divergence
    ca, cb = a["content"], b["content"]
    i = next((j for j in range(min(len(ca), len(cb))) if ca[j] != cb[j]),
             min(len(ca), len(cb)))
    print(f"[determinism] FAIL — diverge at char {i}: "
          f"A={ca[i:i+60]!r} B={cb[i:i+60]!r}", file=sys.stderr)
    print("[determinism] DEER rollback is LOSSY — Layer A REJECTED",
          file=sys.stderr)
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--url", default="http://localhost:8888")
    r.add_argument("--model", default="qwen3.6-27b")
    r.add_argument("--n", type=int, default=10)
    r.add_argument("--temperature", type=float, default=0.6)
    r.add_argument("--max-tokens", type=int, default=16000)
    r.add_argument("--seed", type=int, default=None,
                   help="base seed; gen i uses seed+i (set for reproducibility)")
    r.add_argument("--timeout", type=float, default=360.0)
    r.add_argument("--config", default="unnamed",
                   help="config label, e.g. cf-complete-gate / deer-on")
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)

    g = sub.add_parser("gate")
    g.add_argument("baseline")
    g.add_argument("candidate")
    g.set_defaults(func=gate)

    d = sub.add_parser("determinism")
    d.add_argument("--url-a", required=True, help="baseline (enable_deer=false)")
    d.add_argument("--url-b", required=True,
                   help="enable_deer=true + ATLAS_DEER_FORCE_ROLLBACK=1")
    d.add_argument("--model", default="qwen3.6-27b")
    d.add_argument("--seed", type=int, default=12345)
    d.add_argument("--max-tokens", type=int, default=4000)
    d.add_argument("--timeout", type=float, default=360.0)
    d.set_defaults(func=determinism)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
