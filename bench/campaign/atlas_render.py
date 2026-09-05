#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Render the Atlas serve side of a cell from bench/campaign/atlas_recipes.json.

The mirror image of vllm_render.py, and it exists for the same reason: the flags
a cell was served with are the difference between a result and an anecdote. What
it produces is the EXTRA_ARGS string for scripts/start-node-ep.sh, which appends
it verbatim to EVERY rank -- that is how the speculative flags stay identical
across ranks, which QUICKSTART.md:328-333 says they must be.

It deliberately does NOT emit topology or bind/port/warmup flags. The launcher
owns those (--rank/--world-size/--ep-size/--tp-size/--gpu-ordinal/--port/
--master-*/--no-tui, plus --bind on rank 0 and --warmup-prompt from
WARMUP_PROMPT), and its own header says not to put topology flags in EXTRA_ARGS.
Setting a flag from two places is how two ranks end up disagreeing.

Exit codes: 0 ok · 2 usage · 3 no Atlas recipe for the pair · 4 --spec on
against a model whose recipe declares no speculative support · 7 an entry
carries a flag outside the frozen vocabulary · 9 an excluded thinking mode.
"""

import argparse
import json
import pathlib
import shlex
import sys

from thinking_policy import E_THINK_POLICY, POLICY_PATH, refusal

E_USAGE = 2
E_NO_PROFILE = 3
E_NO_SPEC = 4
E_UNKNOWN_FLAG = 7


def find(doc, model_key, sku):
    for e in doc["entries"]:
        if e["model_key"] == model_key and e["sku"] == sku:
            return e
    return None


def apply_overrides(common, overrides):
    """Replace a common flag's value in place rather than appending a second copy.

    Appending relies on clap's last-wins behaviour for a flag that may or may not
    be declared `multiple`; replacing in place does not.
    """
    out = list(common)
    for flag, value in overrides.items():
        if flag not in out:
            raise KeyError(flag)
        out[out.index(flag) + 1] = value
    return out


def build_args(doc, entry, spec, think):
    args = apply_overrides(doc["common_args"], entry.get("overrides") or {})
    args += list(entry["extra_args"])
    if think == "off":
        args += list(doc["think_off_args"])
    if spec == "on":
        args += list(entry["spec_args"] or doc["spec_args"])
    return args


def flag_audit(tokens, known):
    return [t for t in tokens if t.startswith("--") and t not in known]


def selftest(doc):
    known = set(doc["known_flag_prefixes"])
    checks = 0
    fails = []

    def check(cond, what):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(what)

    for e in doc["entries"]:
        tag = f"{e['model_key']}/{e['sku']}"
        try:
            off = build_args(doc, e, "off", "on")
        except KeyError as exc:
            fails.append(f"{tag}: override {exc} names a flag not in common_args")
            checks += 1
            continue
        checks += 1

        check(not flag_audit(off, known),
              f"{tag}: not-in-recipe flags {flag_audit(off, known)}")
        check("--disable-thinking" not in off, f"{tag}: think=on must not disable thinking")
        check("--disable-thinking" in build_args(doc, e, "off", "off"),
              f"{tag}: think=off must add --disable-thinking")
        check("--speculative" not in off, f"{tag}: spec=off must not add --speculative")

        # No topology or launcher-owned flag may appear in EXTRA_ARGS.
        for owned in ("--rank", "--world-size", "--ep-size", "--tp-size",
                      "--gpu-ordinal", "--port", "--bind", "--master-addr",
                      "--master-port", "--no-tui", "--warmup-prompt"):
            check(owned not in off, f"{tag}: {owned} is the launcher's, not EXTRA_ARGS'")

        # The Atlas topology rule from resolve_topology, checked here so a bad
        # entry costs a selftest rather than N ranks that load weights and bail.
        check(e["tp_size"] * e["ep_size"] == e["ngpus"]
              or e["tp_size"] == e["ep_size"] == e["ngpus"],
              f"{tag}: world {e['ngpus']} is neither tp*ep nor tp==ep==world")

        if e["spec_supported"]:
            on = build_args(doc, e, "on", "on")
            spec_tokens = list(e["spec_args"] or doc["spec_args"])
            check(on == off + spec_tokens,
                  f"{tag}: --spec on must append exactly the speculative flags")
            check("--num-drafts" in spec_tokens, f"{tag}: spec flags name no draft count")

    # Pairs that must NOT exist, so run_cell.sh exits 3 rather than guessing.
    check(find(doc, "qwen3-next-80b-fp8", "h100") is None,
          "qwen3-next-80b-fp8/h100 must be absent (the PRD marks 2xH100 reconstructed)")
    for key in ("glm-5.3", "glm-5.3-flash", "glm-4.5-air-fp8", "minimax-m3",
                "kimi-k3", "qwen3.8-flash-next-fp8"):
        check(all(find(doc, key, s) is None for s in ("h100", "h200", "b200", "gb10")),
              f"{key} has no Atlas recipe fixture and must have no entry")

    # NEGATIVE: a hand-edited flag must be reported.
    bad = flag_audit(["--gpu-memory-utilization", "0.9", "--enable-sneaky-fastpath"], known)
    check(bad == ["--enable-sneaky-fastpath"],
          f"hand-edited unknown flag not reported as not-in-recipe (got {bad})")

    # NEGATIVE: an override naming a flag that is not in common_args must raise
    # rather than silently appending a second copy of it.
    victim = json.loads(json.dumps(doc["entries"][0]))
    victim["overrides"] = {"--max-thinking-budget": "0"}
    try:
        build_args(doc, victim, "off", "on")
        check(False, "an override outside common_args must be refused")
    except KeyError:
        check(True, "")

    for f in fails:
        print(f"FAIL: {f}")
    print(f"atlas_render selftest: {checks - len(fails)}/{checks} checks passed "
          f"over {len(doc['entries'])} entries")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipes", required=True)
    ap.add_argument("--model")
    ap.add_argument("--sku")
    ap.add_argument("--spec", choices=["on", "off"])
    ap.add_argument("--think", choices=["on", "off"])
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--field")
    ap.add_argument("--extra-args", action="store_true")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    doc = json.loads(pathlib.Path(args.recipes).read_text())
    if args.selftest:
        return selftest(doc)
    if not args.model or not args.sku:
        print("ERROR: --model and --sku are required", file=sys.stderr)
        return E_USAGE

    entry = find(doc, args.model, args.sku)
    if entry is None:
        print(f"no rendered profile for {args.model} on {args.sku}")
        return E_NO_PROFILE
    if args.probe:
        return 0
    if args.field:
        value = entry.get(args.field)
        print("" if value is None else (json.dumps(value) if isinstance(value, (dict, list))
                                        else value))
        return 0

    if not args.spec or not args.think:
        print("ERROR: --spec and --think are required to render args", file=sys.stderr)
        return E_USAGE
    error = refusal(json.loads(POLICY_PATH.read_text()), args.model, args.think)
    if error:
        print(error, file=sys.stderr)
        return E_THINK_POLICY
    if args.spec == "on" and not entry["spec_supported"]:
        print(f"ERROR: --spec on, but the Atlas recipe for {args.model} on {args.sku} "
              f"declares no speculative support.", file=sys.stderr)
        print(f"       {entry['notes']}", file=sys.stderr)
        return E_NO_SPEC

    try:
        rendered = build_args(doc, entry, args.spec, args.think)
    except KeyError as exc:
        print(f"ERROR: override {exc} names a flag that is not in common_args",
              file=sys.stderr)
        return E_UNKNOWN_FLAG

    unknown = flag_audit(rendered, set(doc["known_flag_prefixes"]))
    if unknown:
        print(f"ERROR: entry {args.model}/{args.sku} carries flags that are not in the "
              f"frozen recipe vocabulary: {' '.join(sorted(set(unknown)))}", file=sys.stderr)
        return E_UNKNOWN_FLAG

    if args.extra_args:
        print(" ".join(shlex.quote(a) for a in rendered))
        return 0

    print("=== Atlas serve recipe ===")
    print(f"model_key:   {entry['model_key']}")
    print(f"sku:         {entry['sku']}")
    print(f"hf_id:       {entry['hf_id']}")
    print(f"quant:       {entry['quant']}")
    print(f"topology:    NGPUS={entry['ngpus']} EP_SIZE={entry['ep_size']} "
          f"TP_SIZE={entry['tp_size']}")
    print(f"spec:        {args.spec}   think: {args.think}")
    print(f"pairable:    {entry['pairable']}")
    print(f"sources:     {'; '.join(entry['sources'])}")
    print(f"EXTRA_ARGS:  {' '.join(shlex.quote(a) for a in rendered)}")
    print(f"notes:       {entry['notes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
