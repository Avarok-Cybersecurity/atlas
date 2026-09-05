#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Assemble one campaign cell artifact from the stage outputs run_cell.sh produced.

Every field here is either copied from a stage output, copied from a recipe data
file, or null. Nothing is computed from an assumption, and nothing is defaulted
into existence: a stage that did not run leaves its block null, which is what
makes "not measured" distinguishable from "measured zero" downstream.

Two reductions are worth naming because they are the ones SCHEMA-GAPS.md says
get misread:

  * The latency percentiles are the arithmetic MEAN of each rep's percentile.
    The ladder keeps one percentile per rep and discards the per-request
    samples, so a pooled cell percentile is not reconstructible from its JSON.
    percentile_method records exactly that, and the validator refuses a
    ladder-sourced artifact that claims pooled_requests.
  * The verdict is derived from gate EVIDENCE, not from the exit codes of the
    stages that produced it. Three of the measurement gates -- the vacuity
    floor, request errors, and throughput spread -- are conditions the ladder
    reports while exiting 0, and a paired artifact is only a pair if it is the
    same cell on the other engine within 24 h. See gate_blockers/pair_check.
  * engine_version is the ENGINE's identity and harness.git_sha is this
    checkout's. They were one field, and the harness SHA was written into it
    for either engine -- so an artifact named a revision nothing had verified
    while the digest of the image that ran and the hash of the binary that ran
    were both null. What run_cell can verify goes in engine_version; what it
    merely knows about itself goes in harness.
  * isl_measured_p50 is filled in ONLY when every rep observed a single prompt
    length. The ladder stores a sorted SET of prompt_tokens per rep, so
    multiplicities are gone and a median over the union would be a number with
    no defined meaning. Otherwise it stays null and the observed set goes in
    notes.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

VACUITY_FLOOR = 0.8  # PRD section 9 / crates/atlas-plugin/src/benchmarks/concurrency.rs
SPREAD_MAX_PCT = 10.0  # PRD section 4 latency-pack gate

HARD_STAGES = ("preflight", "serve", "boot", "coherency")

PAIR_WINDOW_S = 24 * 3600  # PRD section 4: the two legs must be a day apart at most

COHERENCY_GATES = ("determinism_ok", "toolcall_ok", "think_leak_ok", "known_answer_ok")


def parse_utc(text):
    """`2026-09-05T09:00:00Z` -> aware datetime, or None. No other format."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc)
    except ValueError:
        return None


def policy_matches(policy, think):
    """Require the gate's named mode and literal request boolean to agree."""
    if not isinstance(policy, dict) or policy.get("think") != think:
        return False
    kwargs = policy.get("chat_template_kwargs")
    return isinstance(kwargs, dict) and kwargs.get("enable_thinking") is (think == "on")


def gate_blockers(boot, coh, ladder, rung, metrics, think):
    """Every gate whose evidence is absent or negative, in stage order.

    This is the difference between "the stage exited 0" and "the gate passed".
    `run_cell.sh` sets --failing-stage from a stage's EXIT CODE, and three of
    the measurement gates -- the 80% vacuity floor, request errors, and the
    10% throughput spread -- are conditions the ladder reports while exiting
    successfully. They used to be appended to `notes` and nothing else, so a
    cell whose every timed request returned 32 of 256 tokens was CERTIFIED
    with the fact written in prose beside the verdict that contradicted it.
    A gate is a result or it is not a gate.

    Returns [(stage, reason)]; the first entry names the failing stage.
    """
    out = []
    if boot is None:
        out.append(("boot", "no boot gate JSON was recorded for this cell"))
    elif boot.get("passed") is not True:
        out.append(("boot", f"the boot gate did not pass "
                            f"(passed={boot.get('passed')!r}, status={boot.get('status')!r})"))

    if coh is None:
        out.append(("coherency", "no coherency gate JSON was recorded for this cell"))
    else:
        for key in COHERENCY_GATES:
            if coh.get(key) is not True:
                # null is "this gate did not report", which is not a pass: a
                # coherency JSON written before the known-answer probe existed
                # is evidence of three gates, not four.
                out.append(("coherency", f"coherency {key}={coh.get(key)!r}"))
        if "request_policy" not in coh and "check_request_policy" not in coh:
            # Before policy recording existed, every gate request hardcoded
            # enable_thinking=false. That historical evidence is off only.
            if think != "off":
                out.append(("coherency", "legacy request policy is think-off; "
                            "think-on requires explicit matching gate evidence"))
        else:
            if not policy_matches(coh.get("request_policy"), think):
                out.append(("coherency", f"coherency request policy does not match think={think}"))
            per_check = coh.get("check_request_policy")
            for key in COHERENCY_GATES:
                expected = "off" if key == "think_leak_ok" else think
                got = per_check.get(key) if isinstance(per_check, dict) else None
                if not policy_matches(got, expected):
                    out.append(("coherency", f"{key} request policy must be think={expected}"))

    if ladder is None:
        out.append(("ladder", "no ladder JSON was recorded for this cell"))
    elif rung is None:
        out.append(("ladder", "the ladder JSON carries no rung at this concurrency"))
    else:
        # Old headers without this field also predate thinking support and
        # represent off. A present malformed field is not historical proof.
        kwargs = ladder.get("chat_template_kwargs", {"enable_thinking": False})
        if (not isinstance(kwargs, dict)
                or kwargs.get("enable_thinking") is not (think == "on")):
            out.append(("ladder", f"ladder request policy does not match think={think}"))
        if metrics["vacuous"] is not False:
            out.append(("ladder",
                        "the 80% vacuity floor was not cleared"
                        if metrics["vacuous"] else
                        "the 80% vacuity floor could not be applied"))
        errors_total = rung.get("errors_total")
        if errors_total is None:
            out.append(("ladder", "the rung records no error count"))
        elif errors_total:
            out.append(("ladder", f"{errors_total} request error(s) in the measured reps"))
        spread = metrics["tok_s_spread_pct"]
        if spread is None:
            out.append(("ladder", "the rung records no throughput spread"))
        elif spread > SPREAD_MAX_PCT:
            out.append(("ladder", f"throughput spread {spread:.2f}% exceeds "
                                  f"the {SPREAD_MAX_PCT}% gate"))
        if metrics["tok_s_mean"] is None:
            out.append(("ladder", "the rung records no mean throughput"))
    return out


def pair_check(paired, args, timing):
    """(within_24h, reasons) for the other engine's cell.

    `within_24h` is the verdict of the WHOLE check, not the clock comparison
    alone: a cell measured four minutes ago on a different SKU is not a pair,
    and neither is `{}`. True only when the pair is the same cell on the other
    engine and both timestamps fall inside the window; False when something
    checkable is wrong; None when it could not be decided at all.
    """
    if paired is None:
        return None, ["no paired artifact was recorded"]
    if not isinstance(paired, dict) or not paired:
        return False, ["the paired artifact carries no cell (an empty JSON "
                       "object is not a measurement of anything)"]

    reasons = []

    def same(path, got, expect):
        if got != expect:
            reasons.append(f"paired {path}={got!r}, this cell has {expect!r}")

    same("engine", paired.get("engine"), "vllm" if args.engine == "atlas" else "atlas")
    same("model.model_key", (paired.get("model") or {}).get("model_key"), args.model_key)
    same("hardware.hardware_id", (paired.get("hardware") or {}).get("hardware_id"), args.sku)
    same("workload.name", (paired.get("workload") or {}).get("name"), args.workload)
    same("workload.concurrency", (paired.get("workload") or {}).get("concurrency"),
         args.concurrency)
    same("spec.on", (paired.get("spec") or {}).get("on"), args.spec == "on")
    same("think", paired.get("think"), args.think)

    theirs = paired.get("timing") or {}
    deltas, undecidable = [], False
    for field in ("started_utc", "finished_utc"):
        ours, other = parse_utc(timing.get(field)), parse_utc(theirs.get(field))
        if ours is None or other is None:
            undecidable = True
            missing = "this cell" if ours is None else "the paired cell"
            reasons.append(f"timing.{field}: {missing} has no usable timestamp")
            continue
        deltas.append((field, abs((ours - other).total_seconds())))
    for field, delta in deltas:
        if delta > PAIR_WINDOW_S:
            reasons.append(f"timing.{field}: the two legs are {delta / 3600:.1f} h "
                           f"apart, outside the {PAIR_WINDOW_S // 3600} h window")

    if undecidable:
        return None, reasons
    return not reasons, reasons


def read_json(path):
    if not path:
        return None
    p = pathlib.Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def sha256_file(path):
    p = pathlib.Path(path) if path else None
    if not p or not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_argv(path):
    p = pathlib.Path(path) if path else None
    if not p or not p.is_file():
        return []
    raw = p.read_bytes()
    return [t.decode() for t in raw.split(b"\0") if t]


def read_env(path):
    p = pathlib.Path(path) if path else None
    out = {}
    if not p or not p.is_file():
        return out
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def parse_smi_q(path):
    """Pull the few identity fields out of `nvidia-smi -q`, or nulls."""
    out = {"gpu": None, "driver": None, "cuda": None, "attached": None}
    p = pathlib.Path(path) if path else None
    if not p or not p.is_file():
        return out
    text = p.read_text(errors="replace")
    for key, pat in (("driver", r"Driver Version\s*:\s*(\S+)"),
                     ("cuda", r"CUDA Version\s*:\s*(\S+)"),
                     ("gpu", r"Product Name\s*:\s*(.+)"),
                     ("attached", r"Attached GPUs\s*:\s*(\d+)")):
        m = re.search(pat, text)
        if m:
            out[key] = m.group(1).strip()
    if out["attached"] is not None:
        out["attached"] = int(out["attached"])
    return out


def find_entry(doc, model_key, sku):
    for e in doc["entries"]:
        if e["model_key"] == model_key and e["sku"] == sku:
            return e
    return None


def reduce_ladder(ladder, concurrency, osl):
    """Rung -> metrics block, plus the notes the reduction owes the reader."""
    empty = {
        "percentile_method": "mean_of_rep_percentiles",
        "tok_s_series": [], "tok_s_mean": None, "tok_s_spread_pct": None,
        "ttft_p50_ms": None, "ttft_p99_ms": None, "tpot_p50_ms": None,
        "e2e_p50_s": None, "vacuous": None, "ladder_json_path": None,
    }
    if not ladder:
        return empty, None, ["ladder JSON absent: every metric is null, not zero"]

    rung = next((r for r in ladder.get("rungs", [])
                 if r.get("concurrency") == concurrency), None)
    if rung is None:
        return empty, None, [f"ladder JSON has no rung at C={concurrency}"]

    notes = []
    reps = rung.get("reps", [])

    def mean_of(field):
        vals = [r[field] for r in reps
                if isinstance(r.get(field), (int, float)) and r[field] > 0]
        return statistics.fmean(vals) if vals else None

    tpot = mean_of("tpot_p50_ms")
    if tpot is None:
        notes.append("tpot_p50_ms is null: the ladder excludes zero TPOT (one "
                     "content delta or a single token), which is an absence, "
                     "not a zero-time decode")

    # Vacuity: every timed request, not the rep aggregate.
    floor = VACUITY_FLOOR * osl
    short = [n for r in reps for n in r.get("completion_tokens_per_req", []) if n < floor]
    missing_counts = any("completion_tokens_per_req" not in r for r in reps)
    vacuous = None if missing_counts else bool(short)
    if missing_counts:
        notes.append("vacuous is null: at least one rep carries no per-request "
                     "completion counts, so the 80% floor could not be applied")
    elif short:
        notes.append(f"vacuous: {len(short)} timed request(s) returned under "
                     f"{floor:.0f} of {osl} output tokens")

    spread = rung.get("tok_s_spread_pct")
    if isinstance(spread, (int, float)) and spread > SPREAD_MAX_PCT:
        notes.append(f"throughput spread {spread:.2f}% exceeds the {SPREAD_MAX_PCT}% gate")

    errors_total = rung.get("errors_total")
    if errors_total:
        notes.append(f"{errors_total} request error(s) in the measured reps")

    metrics = {
        "percentile_method": "mean_of_rep_percentiles",
        "tok_s_series": rung.get("tok_s_series", []),
        "tok_s_mean": rung.get("tok_s_mean"),
        "tok_s_spread_pct": spread,
        "ttft_p50_ms": mean_of("ttft_p50_ms"),
        "ttft_p99_ms": mean_of("ttft_p99_ms"),
        "tpot_p50_ms": tpot,
        "e2e_p50_s": mean_of("e2e_p50_s"),
        "vacuous": vacuous,
        "ladder_json_path": None,
    }
    return metrics, rung, notes


def measured_isl(rung):
    """A single observed prompt length, or null (see the module docstring)."""
    if not rung:
        return None, None
    seen = set()
    for rep in rung.get("reps", []):
        seen.update(rep.get("prompt_tokens_per_req") or [])
    if len(seen) == 1:
        return seen.pop(), None
    if not seen:
        return None, "no prompt_tokens usage was recorded"
    return None, (f"observed prompt lengths {sorted(seen)}: the ladder stores a "
                  f"SET per rep, so multiplicities are gone and no p50 is defined")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", required=True, choices=["atlas", "vllm"])
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--sku", required=True)
    ap.add_argument("--workload", required=True, choices=["lat", "agent"])
    ap.add_argument("--concurrency", required=True, type=int)
    ap.add_argument("--spec", required=True, choices=["on", "off"])
    ap.add_argument("--think", required=True, choices=["on", "off"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workloads", required=True)
    ap.add_argument("--atlas-recipes", required=True)
    ap.add_argument("--vllm-recipes", required=True)
    ap.add_argument("--client", required=True, help="path to the ladder harness")
    ap.add_argument("--serve-argv")
    ap.add_argument("--serve-env")
    ap.add_argument("--nvidia-smi-q")
    ap.add_argument("--git-sha", help="the ENGINE's own revision, when something "
                                      "verified it (an image's revision label, a "
                                      "binary that prints one) -- never this "
                                      "checkout's HEAD")
    ap.add_argument("--harness-git-sha", help="the campaign checkout that drove the "
                                             "cell; provenance for the harness, not "
                                             "for the engine")
    ap.add_argument("--binary")
    ap.add_argument("--image-digest")
    ap.add_argument("--vllm-version")
    ap.add_argument("--boot-json")
    ap.add_argument("--coherency-json")
    ap.add_argument("--ladder-json")
    ap.add_argument("--ptx-receipt")
    ap.add_argument("--paired-artifact")
    ap.add_argument("--failing-stage", default="")
    ap.add_argument("--extra-note", default="")
    args = ap.parse_args()

    workloads = json.loads(pathlib.Path(args.workloads).read_text())
    shape = workloads["workloads"][args.workload]
    isl, osl = shape["isl"], shape["osl"]

    recipes = json.loads(pathlib.Path(
        args.atlas_recipes if args.engine == "atlas" else args.vllm_recipes).read_text())
    entry = find_entry(recipes, args.model_key, args.sku)
    if entry is None:
        print(f"no rendered profile for {args.model_key} on {args.sku}", file=sys.stderr)
        return 3

    if args.engine == "atlas":
        world = entry["ngpus"]
        top = {"tp": entry["tp_size"], "ep": entry["ep_size"], "pp": 1, "dp": 1,
               "world_size": world, "nnodes": 1}
        spec_supported = entry["spec_supported"]
        # The method describes what this cell RAN, not what the recipe could
        # run. `--spec off` against a spec-capable recipe used to record
        # method="mtp", which the validator correctly rejects (method must be
        # `none` while `on` is false) -- turning a valid speculation-off cell
        # into a validation failure. The vLLM branch below already does this.
        spec_method = "none"
        spec_k = None
        if args.spec == "on" and spec_supported:
            spec_method = "mtp"
            sargs = entry["spec_args"] or recipes["spec_args"]
            spec_k = int(sargs[sargs.index("--num-drafts") + 1])
        recipe_max = entry["recipe_max"]
    else:
        world = entry["gpus"]
        t = entry["topology"]
        top = {"tp": t["tp"], "ep": 1, "pp": t["pp"], "dp": t["dp"],
               "world_size": world, "nnodes": entry["nnodes"]}
        spec_supported = bool(entry["spec_args"])
        spec_method = "none"
        spec_k = None
        if args.spec == "on" and spec_supported:
            blob = entry["spec_args"][entry["spec_args"].index("--speculative-config") + 1]
            cfg = json.loads(blob)
            spec_method = cfg.get("method", "mtp")
            spec_k = cfg.get("num_speculative_tokens")
        # Every vLLM entry IS the rendered recipe profile; vllm_control.sh has
        # no way to step one down, so it is recipe-max by construction.
        recipe_max = True

    smi = parse_smi_q(args.nvidia_smi_q)
    boot = read_json(args.boot_json)
    coh = read_json(args.coherency_json)
    ladder = read_json(args.ladder_json)

    metrics, rung, mnotes = reduce_ladder(ladder, args.concurrency, osl)
    if args.ladder_json and pathlib.Path(args.ladder_json).is_file():
        metrics["ladder_json_path"] = args.ladder_json

    isl_obs, isl_note = measured_isl(rung)
    notes = list(mnotes)
    if isl_note:
        notes.append(f"isl_measured_p50 is null: {isl_note}")
    if args.think == "on":
        # The ladder must have been run with --enable-thinking; its header is
        # the proof. A think-on cell whose ladder header says false is a
        # mismatch, not a measurement.
        kwargs = (ladder or {}).get("chat_template_kwargs")
        sent = kwargs.get("enable_thinking") if isinstance(kwargs, dict) else None
        if sent is not True:
            notes.append(
                "think=on but the ladder header records chat_template_kwargs."
                f"enable_thinking={sent!r}: the client did not request thinking, "
                "so this cell is not evidence of thinking-enabled generation.")
    if args.extra_note:
        notes.append(args.extra_note)
    notes.append(entry.get("notes", ""))

    timing = {field: (ladder or {}).get(field) if isinstance(
                  (ladder or {}).get(field), str) else None
              for field in ("started_utc", "finished_utc")}

    paired_doc = read_json(args.paired_artifact)
    if args.paired_artifact and paired_doc is None:
        notes.append(f"the paired artifact {args.paired_artifact} could not be read "
                     "as JSON, so this cell has no pair")

    # CERTIFIED is a claim about evidence, and it is assembled from that
    # evidence here -- never from "a stage exited 0" or "a file was parseable".
    within_24h, pair_reasons = pair_check(paired_doc, args, timing)
    stage = args.failing_stage or None
    if stage is None:
        blockers = gate_blockers(ladder=ladder, boot=boot, coh=coh, rung=rung,
                                 metrics=metrics, think=args.think)
        if blockers:
            stage = blockers[0][0]
            notes.append("not CERTIFIED, gate evidence: "
                         + "; ".join(f"{st}: {why}" for st, why in blockers))
        elif within_24h is not True:
            stage = "pair"
            notes.append("gates all passed; not CERTIFIED because "
                         + ("; ".join(pair_reasons) if pair_reasons else
                            "the paired cell from the other engine is not recorded"))
    verdict = "CERTIFIED" if stage is None else (
        "NO-GO" if stage in HARD_STAGES else "PARTIAL")
    artifact = {
        "schema": 1,
        "campaign": workloads["campaign"],
        "cell_id": (f"{args.engine}.{args.model_key}.{args.sku}.{args.workload}."
                    f"c{args.concurrency}.spec{args.spec}.think{args.think}"),
        "engine": args.engine,
        "model": {
            "hf_id": entry["hf_id"],
            "model_key": args.model_key,
            "revision": None,
            "quant": entry["quant"],
        },
        "engine_version": {
            "git_sha": args.git_sha or None,
            "image_digest": args.image_digest or None,
            "binary_sha256": sha256_file(args.binary),
            "vllm_version": args.vllm_version or None,
        },
        # Separate from engine_version on purpose: this is the checkout that
        # RAN the cell, and it says nothing about which engine build served
        # the requests. The ladder client's hash is not repeated here -- it is
        # client.sha256, its single source.
        "harness": {"git_sha": args.harness_git_sha or None},
        "hardware": {
            "gpu": smi["gpu"],
            "gpu_count": world,
            "hardware_id": args.sku,
            "driver": smi["driver"],
            "cuda": smi["cuda"],
            "sm_clock_mhz": None,
            "clocks_locked": None,
            "nvidia_smi_q_sha256": sha256_file(args.nvidia_smi_q),
        },
        "topology": dict(top, matched=False, recipe_max=recipe_max),
        "serve_command": read_argv(args.serve_argv),
        "serve_env": read_env(args.serve_env),
        "workload": {"name": args.workload, "isl": isl, "osl": osl,
                     "concurrency": args.concurrency,
                     "source": "bench/hopper_ab/workloads.json"},
        "client": {
            "name": args.client,
            "sha256": sha256_file(args.client),
            "isl_nominal": isl,
            "isl_measured_p50": isl_obs,
            "osl": osl,
            "reps": workloads["reps"],
            "warmup": workloads["warmup"],
            "seed": workloads["sampling"]["seed"],
            "temperature": workloads["sampling"]["temperature"],
            "penalties": {
                "presence_penalty": workloads["sampling"]["presence_penalty"],
                "frequency_penalty": workloads["sampling"]["frequency_penalty"],
            },
            "prompt_mode": "essay",
        },
        # From the ladder's own header, so the pairing window is measured
        # across when the two legs were MEASURED, not when they were written.
        "timing": timing,
        "spec": {"on": args.spec == "on", "method": spec_method, "k": spec_k},
        "think": args.think,
        "boot": {
            "time_to_ready_s": (boot or {}).get("time_to_ready_s"),
            "first_token_s": (boot or {}).get("first_token_s"),
            "pass": (boot or {}).get("passed"),
            "timeout_s": int((boot or {}).get("timeout_s") or workloads["gates"]["boot_s_max"]),
        },
        "coherency": {
            "determinism_ok": (coh or {}).get("determinism_ok"),
            "toolcall_ok": (coh or {}).get("toolcall_ok"),
            "think_leak_ok": (coh or {}).get("think_leak_ok"),
            "known_answer_ok": (coh or {}).get("known_answer_ok"),
            "gate_json_path": args.coherency_json if coh is not None else None,
        },
        "metrics": metrics,
        "ptx_gate_receipt_sha256": sha256_file(args.ptx_receipt),
        "ptx_gate_not_applicable_reason": None,
        "paired_cell": {
            "cell_id": (paired_doc or {}).get("cell_id"),
            "artifact_path": args.paired_artifact or None,
            "within_24h": within_24h,
        },
        "verdict": verdict,
        "failing_stage": stage,
        "notes": " | ".join(n for n in notes if n),
    }

    if artifact["ptx_gate_receipt_sha256"] is None:
        artifact["ptx_gate_not_applicable_reason"] = (
            "the Atlas PTX gate does not apply to an official vLLM image"
            if args.engine == "vllm" else
            "no PTX gate receipt was passed to this cell (--ptx-receipt unset)")

    if args.spec == "on" and not spec_supported:
        artifact["spec"] = {"on": False, "method": "none", "k": None}
        artifact["notes"] += (" | --spec on was requested but this recipe declares "
                              "no speculative profile; recorded as spec off")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"# wrote {out} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
