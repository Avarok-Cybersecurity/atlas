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
  * isl_measured_p50 is filled in ONLY when every rep observed a single prompt
    length. The ladder stores a sorted SET of prompt_tokens per rep, so
    multiplicities are gone and a median over the union would be a number with
    no defined meaning. Otherwise it stays null and the observed set goes in
    notes.
"""

import argparse
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
    ap.add_argument("--git-sha")
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
        spec_method = "mtp" if spec_supported else "none"
        spec_k = None
        if args.spec == "on" and spec_supported:
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
        notes.append(
            "think=on is a SERVE-SIDE setting only: the ladder client hardcodes "
            "chat_template_kwargs.enable_thinking=false in every request body "
            "(harness_w55_conc_ladder.py), so this cell does not measure a "
            "thinking-enabled generation. Changing that needs a ladder owner.")
    if args.extra_note:
        notes.append(args.extra_note)
    notes.append(entry.get("notes", ""))

    stage = args.failing_stage or None
    if stage is None:
        # Every gate passed, but a cell is only CERTIFIED once its pair exists
        # within 24 h (PRD section 4). Without one the cell is complete and
        # uncertified, and `pair` is the stage that has not happened yet.
        paired = read_json(args.paired_artifact)
        if paired is None:
            stage = "pair"
            verdict = "PARTIAL"
            notes.append("gates all passed; awaiting the paired cell from the "
                         "other engine within 24 h before this can be CERTIFIED")
        else:
            verdict = "CERTIFIED"
    else:
        verdict = "NO-GO" if stage in HARD_STAGES else "PARTIAL"

    paired_doc = read_json(args.paired_artifact)
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
            "gate_json_path": args.coherency_json if coh is not None else None,
        },
        "metrics": metrics,
        "ptx_gate_receipt_sha256": sha256_file(args.ptx_receipt),
        "ptx_gate_not_applicable_reason": None,
        "paired_cell": {
            "cell_id": (paired_doc or {}).get("cell_id"),
            "artifact_path": args.paired_artifact or None,
            "within_24h": None,
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
