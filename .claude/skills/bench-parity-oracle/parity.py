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

# ── speculation, and why it cannot be a plain flag comparison ────────────────
# The published ladder38 pair spells the SAME setting two ways:
#   atlas  --speculative --num-drafts 3 --mtp-quantization bf16
#   vllm   --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
# Compared as strings those differ, and the oracle said NOT IN PARITY on a pair
# that is in parity -- the failure mode that matters most here, because it is
# the one that would have us "fix" a fair comparison until it became unfair.
# So speculation normalises to "<method>:<k>", and any command the reader
# cannot resolve to both a method and a k stays UNSET, which forces
# UNDETERMINED rather than a guess.
# Every flag by which each engine can turn speculation on. If NONE of them is
# present the run observably has no speculation -- "none" is a measured state,
# not an unexamined one. The distinction is load-bearing: the published
# vllm-nospec baseline legitimately runs without speculation, and reporting
# that as UNEXAMINED would hide a real DIFFER behind an "I could not look".
SPEC_FLAGS = {
    "atlas":  ["--speculative", "--num-drafts", "--speculative-model", "--draft-model",
               "--mtp-quantization", "--mtp-gate"],
    "vllm":   ["--speculative-config", "--num-speculative-tokens", "--speculative-model",
               "--speculative-method"],
    "sglang": ["--speculative-algorithm", "--speculative-num-draft-tokens"],
    "trtllm": ["--speculative_config"],
}


def _spec(engine, toks):
    joined = " ".join(toks)
    bases = {t.split("=", 1)[0] for t in toks}
    if not (bases & set(SPEC_FLAGS.get(engine, []))):
        return "none"
    if engine == "atlas":
        # Atlas names the method by which family of flags is present; --num-drafts
        # alone is not enough, because it is also the draft width for a draft
        # MODEL. MTP is asserted by the mtp-* flags the recipe carries.
        k = _flag_value(toks, ["--num-drafts"])
        if k is UNSET:
            return UNSET
        if "--speculative-model" in joined or "--draft-model" in joined:
            return f"draft-model:{k}"
        if "--mtp-quantization" in joined or "--mtp-gate" in joined or "--speculative" in toks:
            return f"mtp:{k}"
        return UNSET
    if engine == "vllm":
        cfg = _flag_value(toks, ["--speculative-config"])
        if cfg is not UNSET:
            try:
                d = json.loads(cfg)
            except json.JSONDecodeError:
                return UNSET
            m = d.get("method"); k = d.get("num_speculative_tokens")
            if m is None or k is None:
                return UNSET
            return f"{str(m).lower()}:{k}"
        k = _flag_value(toks, ["--num-speculative-tokens"])
        if k is UNSET:
            return UNSET
        m = _flag_value(toks, ["--speculative-method"])
        if m is not UNSET:
            return f"{str(m).lower()}:{k}"
        if _flag_value(toks, ["--speculative-model"]) is not UNSET:
            return f"draft-model:{k}"
        return UNSET
    if engine == "sglang":
        k = _flag_value(toks, ["--speculative-num-draft-tokens"])
        m = _flag_value(toks, ["--speculative-algorithm"])
        if k is UNSET or m is UNSET:
            return UNSET
        return f"{str(m).lower()}:{k}"
    return UNSET


# ── defaults, declared rather than assumed ───────────────────────────────────
# Both legs leaving an axis unset is not the same as nobody looking at it: if
# both engines document the same default, the axis IS in parity. But a match
# that rests on a default is weaker evidence than a match on two observed
# values, so it is labelled in the table and counted separately in the verdict.
# Only axes whose default is documented and stable appear here.
ENGINE_DEFAULTS = {
    "tensor_parallel": {"atlas": "1", "vllm": "1", "sglang": "1", "trtllm": "1"},
    "scheduling":      {"vllm": "fcfs", "sglang": "lpm", "atlas": UNSET, "trtllm": UNSET},
}

# Two engines' spellings of one policy. Each entry is an equivalence this file
# is willing to defend, not a convenience.
VALUE_ALIASES = {
    "scheduling": {"fcfs": "fifo", "first_come_first_served": "fifo", "fifo": "fifo"},
}


def _flag_value(toks, flags):
    """The value of the first of `flags` that appears, else UNSET."""
    for i, t in enumerate(toks):
        base = t.split("=", 1)[0]
        if base not in flags:
            continue
        if "=" in t:
            return t.split("=", 1)[1]
        if i + 1 < len(toks):
            return toks[i + 1]
        return UNSET
    return UNSET



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
    out["speculation"] = _spec(engine, toks)
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


def _input_stream():
    """Where the JSONL comes from.

    ★ A PATH ARGUMENT MUST NOT HANG. The first real use of this oracle was
    `parity.py ladder38-mtp.jsonl`, which is the obvious way to call anything
    that takes a file -- and it blocked on an empty stdin for two minutes with
    no output at all, because the only reader was `sys.stdin`. A tool whose
    most natural invocation silently waits forever is a trap, so the path form
    is accepted, and a stdin read from an interactive terminal is refused with
    the usage instead of hanging.
    """
    paths = [a for a in sys.argv[1:] if not a.startswith("-")]
    if len(paths) > 1:
        raise SystemExit("one JSONL path at a time; got %d" % len(paths))
    if paths:
        try:
            return open(paths[0], encoding="utf-8")
        except OSError as e:
            raise SystemExit(f"cannot read {paths[0]}: {e}")
    if sys.stdin.isatty():
        raise SystemExit(
            "usage: parity.py [--json] <legs.jsonl>   (or pipe the JSONL on stdin)\n"
            "  one object per line: {\"engine\": \"atlas|vllm|sglang|trtllm\", "
            "\"command\": \"...\", \"harness\": {...}}"
        )
    return sys.stdin


def main():
    legs = []
    for line in _input_stream():
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

    rows, diffs, unknowns, defaulted = [], [], [], []
    for axis in SERVE_AXES:
        vals, shown, from_default = [], [], False
        for e, p, _ in parsed:
            raw = p.get(axis, UNSET)
            tag = ""
            if raw is UNSET:
                d = ENGINE_DEFAULTS.get(axis, {}).get(e, UNSET)
                if d is not UNSET:
                    raw, tag, from_default = d, " (default)", True
            v = norm(axis, raw)
            if v is not UNSET:
                aliased = VALUE_ALIASES.get(axis, {}).get(v)
                if aliased and aliased != v:
                    tag += f" \u2192 {aliased}"
                    v = aliased
            vals.append(v)
            shown.append("(unset)" if v is UNSET else f"{v}{tag}")
        if any(v is UNSET for v in vals):
            state = "UNEXAMINED"
            unknowns.append(axis)
        elif len(set(vals)) == 1:
            state = "match (default)" if from_default else "match"
            if from_default:
                defaulted.append(axis)
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
        verdict = "NOT IN PARITY"
        why = f"{len(diffs)} axis/axes differ: {', '.join(a for a, _ in diffs)}"
        # A DIFFER settles the verdict, but it must not swallow the news that
        # some other axis was never resolved -- fixing the differ would then
        # produce a green that still rests on an unexamined axis.
        if unknowns:
            why += f"; and {len(unknowns)} still unexamined: {', '.join(unknowns)}"
    elif unknowns:
        verdict, why = "UNDETERMINED", f"{len(unknowns)} axis/axes unexamined: {', '.join(unknowns)}"
    elif defaulted:
        verdict = "IN PARITY"
        why = ("every known axis matches and none is unexamined; "
               f"{len(defaulted)} rest(s) on a documented engine default rather than an "
               f"observed value: {', '.join(defaulted)}")
    else:
        verdict, why = "IN PARITY", "every known axis matches and none is unexamined"

    if want_json:
        print(json.dumps({"verdict": verdict, "why": why,
                          "engines": [e for e, _, _ in parsed],
                          "defaulted": defaulted,
                          "axes": [{"axis": a, "state": s, "values": v} for a, s, v in rows]}, indent=2))
    else:
        w = max(len(a) for a, _, _ in rows) + 2
        sw = max(len(st) for _, st, _ in rows) + 2
        print(f"{'axis'.ljust(w)}{'state'.ljust(sw)}" + "  ".join(e for e, _, _ in parsed))
        print("-" * (w + sw + 24))
        for a, st, v in rows:
            print(f"{a.ljust(w)}{st.ljust(sw)}" + "  ".join(v))
        print()
        print(f"VERDICT: {verdict} — {why}")
        if verdict == "UNDETERMINED":
            print("An unexamined axis is not a passed axis: supply `harness` for the client-side")
            print("axes, or state them explicitly, before calling this comparison fair.")
    return 0 if verdict == "IN PARITY" else 1


if __name__ == "__main__":
    sys.exit(main())
