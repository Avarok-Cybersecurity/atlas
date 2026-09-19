#!/usr/bin/env python3
"""B.E.N.C.H P.A.R.I.T.Y O.R.A.C.L.E — are two engine invocations measuring the same thing?

Reads JSONL on stdin, one {"engine","command"[,"env"][,"harness"]} per line, and
adjudicates parity axis by axis.

★ AN UNEXAMINED AXIS IS NOT A PASSED AXIS. The oracle returns UNDETERMINED when
it cannot see an axis on both sides. That is deliberate: every unfair comparison
this repo has caught hid in an axis nobody looked at -- OSL 1024 vs 320 on one
checkpoint reads 478 vs 116 tok/s, and both runs were correct.
"""
import json, shlex, sys
from collections import OrderedDict

# Canonical axis -> {engine: (flag aliases, kind)}. kind: 'value' | 'bool' | 'presence'
SERVE_AXES = OrderedDict([
    ("checkpoint",      {"atlas": (["--model-name"], "value"), "vllm": (["--model"], "value"),
                         "sglang": (["--model-path"], "value"), "trtllm": (["--model"], "value")}),
    ("context_len",     {"atlas": (["--max-seq-len"], "value"), "vllm": (["--max-model-len"], "value"),
                         "sglang": (["--context-length"], "value"), "trtllm": (["--max_seq_len"], "value")}),
    ("batch_cap",       {"atlas": (["--max-batch-size"], "value"), "vllm": (["--max-num-seqs"], "value"),
                         "sglang": (["--max-running-requests"], "value"), "trtllm": (["--max_batch_size"], "value")}),
    ("kv_dtype",        {"atlas": (["--kv-cache-dtype"], "value"), "vllm": (["--kv-cache-dtype"], "value"),
                         "sglang": (["--kv-cache-dtype"], "value"), "trtllm": (["--kv_cache_dtype"], "value")}),
    ("gpu_mem_frac",    {"atlas": (["--gpu-memory-utilization"], "value"), "vllm": (["--gpu-memory-utilization"], "value"),
                         "sglang": (["--mem-fraction-static"], "value"), "trtllm": (["--free_gpu_memory_fraction"], "value")}),
    ("prefix_caching",  {"atlas": (["--enable-prefix-caching"], "bool"), "vllm": (["--enable-prefix-caching", "--no-enable-prefix-caching"], "bool"),
                         "sglang": (["--enable-prefix-caching", "--disable-radix-cache"], "bool"), "trtllm": ([], "value")}),
    ("speculation",     {"atlas": (["--num-drafts"], "value"), "vllm": (["--num-speculative-tokens", "--speculative-config"], "value"),
                         "sglang": (["--speculative-num-draft-tokens"], "value"), "trtllm": ([], "value")}),
    ("tensor_parallel", {"atlas": (["--tensor-parallel"], "value"), "vllm": (["--tensor-parallel-size"], "value"),
                         "sglang": (["--tp-size", "--tp"], "value"), "trtllm": (["--tp_size"], "value")}),
    ("scheduling",      {"atlas": (["--scheduling-policy"], "value"), "vllm": (["--scheduling-policy"], "value"),
                         "sglang": (["--schedule-policy"], "value"), "trtllm": ([], "value")}),
])

# Axes that live in the CLIENT, not the serve command. Absent `harness`, they are
# unexamined -- and they are the ones that have actually broken comparisons here.
HARNESS_AXES = ["isl", "osl", "concurrency", "reps", "temperature", "seed",
                "presence_penalty", "frequency_penalty", "thinking", "prompt_fixture"]

UNSET = object()


def tokens(cmd):
    try:
        return shlex.split(cmd)
    except ValueError as e:                     # unbalanced quotes -> say so, don't guess
        raise SystemExit(f"could not parse a command ({e}); fix the quoting rather than trusting a partial parse")


def extract(engine, cmd):
    """Canonical axis -> value, UNSET when the flag never appears."""
    toks = tokens(cmd)
    out = {}
    for axis, per_engine in SERVE_AXES.items():
        flags, kind = per_engine.get(engine, ([], "value"))
        val = UNSET
        for i, t in enumerate(toks):
            base = t.split("=", 1)[0]
            if base not in flags:
                continue
            if kind == "bool":
                val = "false" if base.startswith("--no-") or base.startswith("--disable-") else "true"
            elif "=" in t:
                val = t.split("=", 1)[1]
            elif i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                val = toks[i + 1]
            else:
                val = "true"
            break
        out[axis] = val
    # checkpoint often arrives positionally: first non-flag token after the subcommand
    if out.get("checkpoint") is UNSET:
        skip = {"vllm", "serve", "spark", "python", "python3", "-m", "sglang.launch_server", "trtllm-serve"}
        for t in toks:
            if t.startswith("-") or t in skip or "/" not in t:
                continue
            out["checkpoint"] = t
            break
    return out


def norm(axis, v):
    """Normalise so 0.85 == .85 and FP8 == fp8, without inventing equivalences."""
    if v is UNSET:
        return v
    s = str(v).strip().strip('"\'')
    if axis in ("gpu_mem_frac", "context_len", "batch_cap", "speculation", "tensor_parallel"):
        try:
            f = float(s)
            return f"{f:g}"
        except ValueError:
            return s.lower()
    return s.lower()


def main():
    legs = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            legs.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"line is not JSON ({e}); the oracle takes one object per line")
    if len(legs) < 2:
        raise SystemExit("need at least two legs to compare; got %d" % len(legs))

    want_json = "--json" in sys.argv
    parsed = []
    for leg in legs:
        eng = str(leg.get("engine", "")).lower()
        if not eng:
            raise SystemExit("every leg needs an `engine`")
        if "command" not in leg:
            raise SystemExit(f"leg {eng!r} has no `command`")
        if eng not in ("atlas", "vllm", "sglang", "trtllm"):
            print(f"note: engine {eng!r} is unknown to the oracle; its flags cannot be normalised",
                  file=sys.stderr)
        parsed.append((eng, extract(eng, leg["command"]), leg))

    rows, diffs, unknowns = [], [], []
    for axis in SERVE_AXES:
        vals = [(e, norm(axis, p.get(axis, UNSET))) for e, p, _ in parsed]
        shown = ["(unset)" if v is UNSET else v for _, v in vals]
        if any(v is UNSET for _, v in vals):
            state = "UNEXAMINED"
            unknowns.append(axis)
        elif len({v for _, v in vals}) == 1:
            state = "match"
        else:
            state = "DIFFER"
            diffs.append((axis, shown))
        rows.append((axis, state, shown))

    for axis in HARNESS_AXES:
        seen = [str((leg.get("harness") or {}).get(axis, UNSET)) for _, _, leg in parsed]
        if any(s == str(UNSET) for s in seen):
            rows.append((axis + " (client)", "UNEXAMINED", ["(not supplied)"] * len(parsed)))
            unknowns.append(axis)
        elif len(set(seen)) == 1:
            rows.append((axis + " (client)", "match", seen))
        else:
            rows.append((axis + " (client)", "DIFFER", seen))
            diffs.append((axis, seen))

    if diffs:
        verdict, why = "NOT IN PARITY", f"{len(diffs)} axis/axes differ"
    elif unknowns:
        verdict, why = "UNDETERMINED", f"{len(unknowns)} axis/axes unexamined: {', '.join(unknowns)}"
    else:
        verdict, why = "IN PARITY", "every known axis matches and none is unexamined"

    if want_json:
        print(json.dumps({"verdict": verdict, "why": why,
                          "engines": [e for e, _, _ in parsed],
                          "axes": [{"axis": a, "state": s, "values": v} for a, s, v in rows]}, indent=2))
    else:
        w = max(len(a) for a, _, _ in rows) + 2
        print(f"{'axis'.ljust(w)}{'state'.ljust(12)}" + "  ".join(e for e, _, _ in parsed))
        print("-" * (w + 12 + 24))
        for a, s, v in rows:
            print(f"{a.ljust(w)}{s.ljust(12)}" + "  ".join(v))
        print()
        print(f"VERDICT: {verdict} — {why}")
        if verdict == "UNDETERMINED":
            print("An unexamined axis is not a passed axis: supply `harness` for the client-side")
            print("axes, or state them explicitly, before calling this comparison fair.")
    return 0 if verdict == "IN PARITY" else 1


if __name__ == "__main__":
    sys.exit(main())
