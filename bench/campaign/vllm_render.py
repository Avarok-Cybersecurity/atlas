#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Renderer behind bench/campaign/vllm_control.sh.

Splitting the JSON arithmetic out of the shell is not tidiness. Three of the
captured renders carry a flag whose value is a JSON blob with spaces and double
quotes in it (`--reasoning-config '{"reasoning_parser":"deepseek_v4",...}'`,
`--attention-config '{"use_prefill_query_quantization":true,...}'`,
`--speculative-config '{"method":"dspark",...}'`). Round-tripping those through
a shell string is where a control leg quietly stops being verbatim, so the argv
never becomes a string: it is handed to the launcher NUL-separated.

Nothing here composes a command. Every token in `args` / `worker_args` /
`spec_args` comes from bench/campaign/vllm_recipes.json, which was transcribed
from the captured recipe evidence. The only tokens this file can add are the
`docker run` preamble and whatever the caller passed in `--extra`, and both are
labelled as such in the printed block.

Exit codes match vllm_control.sh: 0 ok · 2 usage · 3 no rendered profile ·
4 --spec on with no speculative profile · 7 an entry carries a flag that is not
in the frozen recipe vocabulary.
"""

import argparse
import json
import os
import pathlib
import re
import shlex
import sys

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

E_USAGE = 2
E_NO_PROFILE = 3
E_NO_SPEC = 4
E_UNKNOWN_FLAG = 7


def load(path):
    with open(path) as fh:
        return json.load(fh)


def find(doc, model_key, sku):
    for e in doc["entries"]:
        if e["model_key"] == model_key and e["sku"] == sku:
            return e
    return None


def groups(tokens):
    """Split an argv tail into [flag, value...] groups.

    A group starts at a `--flag` token and absorbs every following token that
    is not itself a flag. That is how `--speculative-config '{...}'` and the
    bare `--no-enable-chunked-prefill` that rides with it on Qwen3-Next end up
    as two removable units instead of one fragile subsequence.
    """
    out = []
    for tok in tokens:
        if tok.startswith("--") or not out:
            out.append([tok])
        else:
            out[-1].append(tok)
    return out


def remove_group(args, group):
    """Drop the first occurrence of `group` as a contiguous run of `args`."""
    n = len(group)
    for i in range(len(args) - n + 1):
        if args[i:i + n] == group:
            return args[:i] + args[i + n:]
    return list(args)


def apply_spec(args, spec_args, spec):
    """Return `args` with the recipe's speculative group(s) added or removed.

    The both-or-neither rule in bench/hopper_ab/workloads.json is only
    enforceable if the two renders differ by exactly this and nothing else, so
    the on-render is built from the off-render rather than from a second path.
    """
    off = list(args)
    for g in groups(spec_args or []):
        off = remove_group(off, g)
    if spec == "off":
        return off
    return off + list(spec_args or [])


def flag_audit(tokens, known):
    """Flags in `tokens` that appear nowhere in the frozen recipe vocabulary."""
    return [t for t in tokens if t.startswith("--") and t not in known]


def image_ref(image, digest):
    repo = image.split(":")[0] if "/" in image else image
    return f"{repo}@{digest}" if digest else image


def docker_argv(entry, vllm_args, image, digest, container, hf_cache, docker):
    argv = [docker, "run", "-d", "--name", container,
            "--gpus", "all", "--ipc=host", "--network", "host",
            "-v", f"{hf_cache}:/root/.cache/huggingface"]
    for key in sorted(entry.get("env") or {}):
        argv += ["-e", f"{key}={entry['env'][key]}"]
    # --entrypoint: the recipe renders `vllm serve ...` as the command, and the
    # published images have disagreed about their default entrypoint across
    # tags. Naming it makes the render independent of that, and is the one
    # orchestration adaptation in the launch line. Everything after the image
    # reference is the recipe's own argv with `vllm` consumed as the entrypoint.
    argv += ["--entrypoint", vllm_args[0], image_ref(image, digest)]
    return argv + list(vllm_args[1:])


def pasteable(argv):
    return " ".join(shlex.quote(a) for a in argv)


def describe(entry, spec, extra_tokens, image, digest, unknown_recipe, unknown_extra):
    top = entry["topology"]
    lines = [
        "=== vLLM control leg (recipe render, not a composed command) ===",
        f"model_key:   {entry['model_key']}",
        f"sku:         {entry['sku']}",
        f"hf_id:       {entry['hf_id']}",
        f"quant:       {entry['quant']}",
        f"topology:    gpus={entry['gpus']} nnodes={entry['nnodes']} "
        f"tp={top['tp']} pp={top['pp']} dp={top['dp']} ({entry['strategy']})",
        f"verdict:     {entry['verdict']}",
        f"source:      {entry['source_url']}",
        f"retrieved:   {entry['retrieved_utc']}  evidence sha256 {entry['evidence_sha256']}",
        f"image:       {image}",
        f"image_ref:   {image_ref(image, digest)}"
        + ("" if digest else "   (tag only -- a non-dry run will be REFUSED, exit 5)"),
    ]
    if entry["spec_args"]:
        lines.append(f"spec:        {spec}   (recipe renders spec "
                     f"{entry['spec_rendered_default']}; the switchable group is "
                     f"{pasteable(entry['spec_args'])})")
    else:
        lines.append(f"spec:        {spec}   (this recipe has NO speculative "
                     f"profile -- `--spec on` exits 4)")
    lines.append(f"extra:       {pasteable(extra_tokens) if extra_tokens else '(none)'}")
    lines.append(f"flag audit:  {len(unknown_recipe)} not-in-recipe in the entry, "
                 f"{len(unknown_extra)} not-in-recipe in --extra")
    return "\n".join(lines)


def cmd_probe(doc, args):
    entry = find(doc, args.model, args.sku)
    if entry is None:
        print(f"no rendered profile for {args.model} on {args.sku}")
        return E_NO_PROFILE
    return 0


def cmd_list(doc, _args):
    print(f"{'model_key':24s} {'sku':5s} {'gpus':>4s} {'nodes':>5s} {'spec':>5s}  image")
    for e in doc["entries"]:
        print(f"{e['model_key']:24s} {e['sku']:5s} {e['gpus']:4d} {e['nnodes']:5d} "
              f"{('yes' if e['spec_args'] else 'none'):>5s}  {e['image']}")
    return 0


def cmd_render(doc, args):
    known = set(doc["known_flag_prefixes"])
    entry = find(doc, args.model, args.sku)
    if entry is None:
        print(f"no rendered profile for {args.model} on {args.sku}")
        return E_NO_PROFILE

    if args.spec == "on" and not entry["spec_args"]:
        print(f"ERROR: --spec on, but the recipe for {args.model} on {args.sku} "
              f"renders no speculative profile.", file=sys.stderr)
        print("       Speculation is both-or-neither: run BOTH legs spec off, "
              "or pick a model whose recipe has one.", file=sys.stderr)
        return E_NO_SPEC

    all_recipe = list(entry["args"]) + list(entry["spec_args"] or [])
    for w in entry.get("worker_args") or []:
        all_recipe += list(w)
    unknown_recipe = flag_audit(all_recipe, known)
    if unknown_recipe:
        print(f"ERROR: entry {args.model}/{args.sku} carries flags that are not in "
              f"the frozen recipe vocabulary: {' '.join(sorted(set(unknown_recipe)))}",
              file=sys.stderr)
        print("       vllm_recipes.json is a transcription; a flag that is in an "
              "entry but not in known_flag_prefixes was hand-edited in.",
              file=sys.stderr)
        return E_UNKNOWN_FLAG

    extra_tokens = shlex.split(args.extra or "")
    unknown_extra = flag_audit(extra_tokens, known)

    image = args.image or entry["image"]
    digest = args.image_digest or ""
    if digest and not DIGEST_RE.match(digest):
        print(f"ERROR: VLLM_IMAGE_DIGEST must match sha256:<64 hex>, got '{digest}'",
              file=sys.stderr)
        return E_USAGE

    print(describe(entry, args.spec, extra_tokens, image, digest,
                   unknown_recipe, unknown_extra))
    if unknown_extra:
        print(f"WARNING: --extra introduces {len(unknown_extra)} flag(s) that are in "
              f"no recipe: {' '.join(sorted(set(unknown_extra)))}")
        print("         That is what --extra is for, but the artifact must record "
              "the cell as a recipe ADAPTATION, not as the recipe.")

    heads = [(0, apply_spec(entry["args"], entry["spec_args"], args.spec) + extra_tokens)]
    for i, w in enumerate(entry.get("worker_args") or []):
        heads.append((i + 1, apply_spec(w, entry["spec_args"], args.spec) + extra_tokens))

    stage = pathlib.Path(args.stage) if args.stage else None
    for rank, vargs in heads:
        name = args.container if rank == 0 else f"{args.container}-node{rank}"
        argv = docker_argv(entry, vargs, image, digest, name, args.hf_cache, args.docker)
        role = "head (node-rank 0, serves HTTP)" if rank == 0 else f"worker node-rank {rank} (--headless)"
        print("")
        print(f"# node {rank} -- {role}")
        if entry["nnodes"] > 1:
            print("# $HEAD_IP is a PLACEHOLDER left verbatim from the recipe: "
                  "resolve it on the booked cluster.")
        print(pasteable(argv))
        if stage:
            with open(stage / f"node{rank}.argv", "wb") as fh:
                fh.write(b"\0".join(a.encode() for a in argv) + b"\0")
            if rank == 0:
                with open(stage / "head.argv", "wb") as fh:
                    fh.write(b"\0".join(a.encode() for a in argv) + b"\0")
    if stage and entry["nnodes"] > 1:
        (stage / "multinode").write_text(f"{entry['nnodes']}\n")
    if stage:
        (stage / "container").write_text(f"{args.container}\n")
        (stage / "image_ref").write_text(image_ref(image, digest) + "\n")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


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
        off = apply_spec(e["args"], e["spec_args"], "off")

        # Every render must name the model it serves. The recipes place the HF
        # id positionally after `serve`; a --served-model-name alias is the
        # other accepted form.
        check("--served-model-name" in off or e["hf_id"] in off,
              f"{tag}: render names neither --served-model-name nor {e['hf_id']}")
        check(off[:2] == ["vllm", "serve"], f"{tag}: render does not start with 'vllm serve'")

        # No entry may smuggle a flag past the frozen vocabulary.
        allargs = list(e["args"]) + list(e["spec_args"] or [])
        for w in e.get("worker_args") or []:
            allargs += list(w)
        check(not flag_audit(allargs, known),
              f"{tag}: not-in-recipe flags {flag_audit(allargs, known)}")

        # Spec on vs off differ by EXACTLY the speculative group.
        if e["spec_args"]:
            on = apply_spec(e["args"], e["spec_args"], "on")
            check(sorted(set(on) - set(off)) == sorted(set(e["spec_args"]) - set(off)),
                  f"{tag}: --spec on adds more than the speculative flags")
            check(not (set(off) - set(on)),
                  f"{tag}: --spec off drops something --spec on keeps")
            check("--speculative-config" in on and "--speculative-config" not in off,
                  f"{tag}: --speculative-config not toggled by --spec")
            check(len(on) - len(off) == len(e["spec_args"]),
                  f"{tag}: token-count delta {len(on) - len(off)} != "
                  f"{len(e['spec_args'])} speculative tokens")
        else:
            check(e["spec_rendered_default"] == "off",
                  f"{tag}: no spec_args but rendered default claims spec on")

        # Multi-node renders must carry the head/worker split the campaign
        # prints placeholders for.
        if e["nnodes"] > 1:
            check(bool(e.get("worker_args")), f"{tag}: multi-node with no worker_args")
            for w in e["worker_args"]:
                check("--headless" in w, f"{tag}: worker command is not --headless")
                check("$HEAD_IP" in w, f"{tag}: worker command has no $HEAD_IP placeholder")

    # The pairs the campaign must be able to REFUSE.
    check(find(doc, "glm-4.5-air-fp8", "h100") is None,
          "glm-4.5-air-fp8/h100 must have no profile (both recipe twins 404'd)")
    check(find(doc, "kimi-k3", "b200") is not None, "kimi-k3/b200 profile missing")
    k3 = find(doc, "kimi-k3", "b200")
    check(k3["topology"]["tp"] == 8 and k3["topology"]["pp"] == 2 and k3["nnodes"] == 2,
          "kimi-k3/b200 must be the 2-node TP8+PP2 render")
    check(all(find(doc, m, "gb10") is None for m in
              {e["model_key"] for e in doc["entries"]}),
          "no gb10 vLLM profile was captured; none may be invented")

    # NEGATIVE: a hand-edited entry must be reported, not rendered.
    victim = json.loads(json.dumps(doc["entries"][0]))
    victim["args"] = list(victim["args"]) + ["--enable-sneaky-fastpath", "1"]
    bad = flag_audit(victim["args"], known)
    check(bad == ["--enable-sneaky-fastpath"],
          f"hand-edited unknown flag not reported as not-in-recipe (got {bad})")

    # NEGATIVE: --spec on against a recipe with no speculative profile.
    nano = find(doc, "nemotron-3-nano-fp8", "h100")
    check(nano is not None and not nano["spec_args"],
          "nemotron-3-nano-fp8/h100 must have no speculative profile "
          "(the recipe declares none; the Atlas fixture says 'No MTP support')")

    # NEGATIVE: a malformed digest is refused before anything launches.
    check(not DIGEST_RE.match("sha256:deadbeef"), "short digest must not validate")
    check(not DIGEST_RE.match("latest"), "a tag must not validate as a digest")
    check(bool(DIGEST_RE.match("sha256:" + "a" * 64)), "a real digest must validate")

    for f in fails:
        print(f"FAIL: {f}")
    print(f"vllm_control selftest: {checks - len(fails)}/{checks} checks passed "
          f"over {len(doc['entries'])} rendered entries")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipes", required=True)
    ap.add_argument("--model")
    ap.add_argument("--sku")
    ap.add_argument("--spec", choices=["on", "off"])
    ap.add_argument("--extra", default="")
    ap.add_argument("--stage")
    ap.add_argument("--container", default="vllm-control")
    ap.add_argument("--hf-cache", default=os.path.expanduser("~/.cache/huggingface"))
    ap.add_argument("--docker", default="docker")
    ap.add_argument("--image")
    ap.add_argument("--image-digest")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    doc = load(args.recipes)
    if args.selftest:
        return selftest(doc)
    if args.list:
        return cmd_list(doc, args)
    if not args.model or not args.sku:
        print("ERROR: --model and --sku are required", file=sys.stderr)
        return E_USAGE
    if args.probe:
        return cmd_probe(doc, args)
    if not args.spec:
        print("ERROR: --spec on|off is required", file=sys.stderr)
        return E_USAGE
    return cmd_render(doc, args)


if __name__ == "__main__":
    sys.exit(main())
