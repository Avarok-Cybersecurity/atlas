#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Assemble one GB10 rehearsal cell per ladder rung, without certification."""

import argparse
import copy
import hashlib
import json
import math
import pathlib
import re
import statistics
import sys

import compare


class InvalidInput(ValueError):
    """Input structure or recorded provenance disagrees."""


def require(condition, message):
    if not condition:
        raise InvalidInput(message)


def finite_json(value):
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite JSON number")
    elif isinstance(value, dict):
        for child in value.values():
            finite_json(child)
    elif isinstance(value, list):
        for child in value:
            finite_json(child)


def assemble(ladder, boot, coherency, provenance, workload, run_id, frozen):
    """Pure assembly; the CLI owns reading inputs and writing artifacts."""
    for name, source in (("ladder", ladder), ("boot", boot), ("coherency", coherency),
                         ("provenance", provenance)):
        require(isinstance(source, dict), f"{name} must be a JSON object")
        finite_json(source)
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id)), "unsafe run ID")
    require(workload in frozen["workloads"], "unknown frozen workload")
    require(provenance.get("engine") in ("atlas", "vllm"), "engine must be atlas or vllm")
    try:
        compare.assert_comparable(ladder, ladder)
        stats = compare.index_rungs(ladder)
    except (compare.Mismatch, KeyError, TypeError) as exc:
        raise InvalidInput(str(exc)) from exc
    expected = {**{key: frozen["workloads"][workload][key] for key in ("isl", "osl")},
                **{key: frozen[key] for key in ("reps", "warmup")},
                **{key: frozen["sampling"][key] for key in ("temperature", "seed", "chat_template_kwargs")}}
    for key, value in expected.items():
        require(ladder[key] == value, f"{key} disagrees with frozen {workload} workload")
    require(set(stats) <= set(frozen["concurrencies"]), "rung outside frozen concurrency set")
    for key in ("url", "model"):
        require(isinstance(ladder.get(key), str) and bool(ladder[key]), f"ladder {key} missing")
        for name, source in (("boot", boot), ("coherency", coherency)):
            require(source.get(key) == ladder[key], f"{name} {key} differs from ladder")
    require(boot.get("engine") == provenance["engine"], "boot engine differs from provenance")
    gaps = []

    def nullable_block(name, keys):
        block = provenance.get(name)
        require(block is None or isinstance(block, dict), f"{name} must be object or null")
        result = copy.deepcopy(block or {})
        for key in keys:
            if result.get(key) is None:
                result[key] = None
                gaps.append(f"{name}.{key}: not supplied by external provenance")
        return result

    engine_version = nullable_block("engine_version", ("git_sha", "image_digest", "binary_sha256"))
    model = nullable_block("model", ("hf_id", "revision", "quant"))
    hardware = nullable_block("hardware", ("gpu", "gpu_count", "driver", "cuda", "hardware_id",
                                           "sm_clock_mhz", "nvidia_smi_q_sha256"))
    require(hardware["hardware_id"] == "gb10", "only observed GB10 hardware is in this rehearsal")
    topology = nullable_block("topology", ("tp", "ep", "world_size", "matched"))
    client = nullable_block("client", ("git_sha", "file_git_revision", "python_version", "aiohttp_version",
                                       "invocation", "environment", "prefix_cache_control"))
    for value, recorded, label in ((model.get("served_model_name"), ladder["model"], "served model alias"),
                                    (client.get("driver_sha256"), ladder["driver_sha256"], "client hash"),
                                    (client.get("url"), ladder["url"], "client URL")):
        require(value is None or value == recorded, f"{label} mismatch")
    model["served_model_name"] = ladder["model"]
    client["driver_sha256"] = ladder["driver_sha256"]
    client["url"] = ladder["url"]
    command = provenance.get("serve_command")
    require(command is None or (isinstance(command, list) and bool(command)
            and all(isinstance(arg, str) for arg in command)), "serve_command must be argv or null")
    if command is None:
        gaps.append("serve_command: executed argv not supplied")
    environment = provenance.get("environment")
    require(environment is None or isinstance(environment, dict), "environment must be object or null")
    if environment is None:
        gaps.append("environment: server launch environment not supplied")
    external_workload = nullable_block("workload", ("spec", "presence_penalty", "frequency_penalty"))
    for key in ("presence_penalty", "frequency_penalty"):
        require(external_workload[key] is None or external_workload[key] == frozen["sampling"][key],
                f"{key} differs from frozen sampling")
    for key, value in expected.items():
        require(external_workload.get(key) is None or external_workload[key] == value,
                f"external workload {key} mismatch")

    boot_pass = boot.get("passed")
    require(boot_pass is None or type(boot_pass) is bool, "boot.passed must be boolean or null")
    if boot_pass is None:
        gaps.append("boot.passed: no recorded boot verdict")
    boot_issues = []
    boot_incomplete = False
    for key in ("time_to_ready_s", "first_token_s", "total_s"):
        value = boot.get(key)
        if value is None:
            gaps.append(f"boot.{key}: not emitted")
            if boot_pass is True:
                boot_issues.append(f"passed boot lacks {key}")
                boot_incomplete = True
        elif not compare.finite_number(value) or value < 0:
            raise InvalidInput(f"invalid boot.{key}")
    if compare.finite_number(boot.get("total_s")) and boot["total_s"] > 1800:
        boot_issues.append("boot exceeded the PRD 1800-second deadline")
    if boot_pass is True and boot.get("status") != "ready":
        raise InvalidInput("passed boot status is not ready")
    if compare.finite_number(boot.get("total_s")) and boot["total_s"] > 1800:
        boot_pass = False
    elif boot_incomplete:
        boot_pass = None
    check_names = ("determinism_ok", "toolcall_ok", "think_leak_ok")
    checks = {key: coherency.get(key) for key in check_names}
    for key, value in checks.items():
        require(value is None or type(value) is bool, f"coherency.{key} must be boolean or null")
        if value is None:
            gaps.append(f"coherency.{key}: not emitted")
    coherent = False if False in checks.values() else True if all(v is True for v in checks.values()) else None
    claimed = coherency.get("passed")
    require(claimed is None or type(claimed) is bool, "coherency.passed must be boolean or null")
    require(claimed is None or coherent is None or claimed == coherent, "inconsistent coherency gate verdict")
    gaps += ["workload.isl is nominal filler-word-based length; retain observed token usage",
             "warmup results are discarded by the ladder and unavailable",
             "scalar latency percentiles are arithmetic means of per-rep percentiles, not pooled",
             "clock_sample_at_rep_start is sampled before timing on the client host",
             "single-leg GB10 rehearsal cannot establish paired topology, A/B parity or Hopper certification"]
    ptx_hash = provenance.get("ptx_gate_ledger_sha256")
    if ptx_hash is None:
        gaps.append("ptx_gate_ledger_sha256: not applicable to vLLM" if provenance["engine"] == "vllm"
                    else "ptx_gate_ledger_sha256: not supplied for Atlas")
    results = []
    for rung in ladder["rungs"]:
        c = rung["concurrency"]
        issues = list(stats[c]["issues"])
        reps = rung.get("reps") or []
        rates = [rep.get("tok_s") for rep in reps]
        if rung.get("tok_s_series") != rates:
            issues.append("stored tok_s_series differs from raw reps")
        all_rates = bool(rates) and all(compare.finite_number(n) and n > 0 for n in rates)
        rate_mean = statistics.fmean(rates) if all_rates else None
        if rate_mean is not None and (not compare.finite_number(rung.get("tok_s_mean"))
                or not math.isclose(rung["tok_s_mean"], rate_mean, rel_tol=1e-9)):
            issues.append("stored tok_s_mean differs from raw reps")
        latency = {}
        latency_series = {}
        for key in ("ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "e2e_p50_s"):
            values = [rep.get(key) for rep in reps]
            valid = bool(values) and all(compare.finite_number(n) and n >= 0 for n in values)
            if not valid:
                issues.append(f"{key}: incomplete latency series")
            latency_series[key] = values
            latency[key] = statistics.fmean(values) if valid else None
        counts = [rep.get("completion_tokens_per_req") for rep in reps]
        complete_usage = bool(counts) and all(isinstance(values, list) and len(values) == c
                          and all(type(n) is int and n >= 0 for n in values) for values in counts)
        vacuous = any(n < .8 * ladder["osl"] for values in counts for n in values) if complete_usage else None
        measured_failure = boot_pass is False or coherent is False or bool(issues)
        rehearsal = "FAIL" if measured_failure else "PASS" if boot_pass and coherent else "INCOMPLETE"
        results.append({
            "schema": 1, "campaign": "hopper-atlas-vs-vllm-2026-09", "run_id": run_id,
            "artifact_type": "gb10-rehearsal-cell", "scope": "GB10 rehearsal only; no Hopper or Blackwell DC measurement",
            "engine": provenance["engine"], "engine_version": engine_version, "model": model,
            "hardware": hardware, "topology": topology, "serve_command": command,
            "environment": environment, "client": client,
            "workload": {**external_workload, "name": workload, "isl": ladder["isl"], "isl_kind": "nominal",
                         "osl": ladder["osl"], "concurrency": c, "reps": ladder["reps"], "warmup": ladder["warmup"],
                         "temperature": ladder["temperature"], "seed": ladder["seed"],
                         "enable_thinking": ladder["chat_template_kwargs"]["enable_thinking"]},
            "boot": {"time_to_ready_s": boot.get("time_to_ready_s"), "first_token_s": boot.get("first_token_s"),
                     "total_s": boot.get("total_s"), "pass": boot_pass,
                     "first_token_semantics": "non-streaming one-token response latency including framing"},
            "coherency": {**checks, "passed": coherent},
            "metrics": {"tok_s_series": rates, "tok_s_mean": rate_mean, **latency, "vacuous": vacuous,
                        "latency_percentile_series": latency_series,
                        "latency_aggregation": "arithmetic_mean_of_per_rep_percentiles",
                        "throughput_aggregation": "arithmetic_mean_of_per_rep_rates"},
            "gates": {"boot": boot_pass, "coherency": coherent, "latency_pack": not issues,
                      "issues": {"boot": boot_issues, "latency_pack": issues}, "ab": None},
            "ptx_gate_ledger_sha256": ptx_hash, "verdict": "NO-GO" if measured_failure else "PARTIAL",
            "rehearsal_verdict": rehearsal, "schema_gaps": list(gaps),
            "raw": {"ladder_header": {k: copy.deepcopy(v) for k, v in ladder.items() if k != "rungs"},
                    "rung": copy.deepcopy(rung), "boot": copy.deepcopy(boot),
                    "coherency": copy.deepcopy(coherency), "provenance": copy.deepcopy(provenance)},
            "notes": "Gate outcomes describe this GB10 leg. PARTIAL does not certify a paired campaign.",
        })
    return results


def fixture():
    reps = []
    for i, rate in enumerate((100.0, 101.0, 99.0)):
        reps.append({"rep": i, "wall_s": 256 / rate, "completion_tokens": 256,
                     "prompt_tokens": 1300, "prompt_tokens_per_req": [1300],
                     "tok_s": rate, "n_ok": 1, "n_err": 0, "errors": [],
                     "ttft_p50_ms": 10.0 + i, "ttft_p99_ms": 10.0 + i,
                     "tpot_p50_ms": 9.0 + i, "e2e_p50_s": 2.56,
                     "finish_reasons": ["length"], "completion_tokens_per_req": [256],
                     "clock_sample_at_rep_start": "208, 8.9"})
    ladder = {"label": "synthetic", "url": "http://fixture:8000", "model": "nano",
              "isl": 1024, "osl": 256, "reps": 3, "warmup": 1,
              "temperature": 0.0, "seed": 42,
              "chat_template_kwargs": {"enable_thinking": False},
              "driver_sha256": "a" * 64,
              "rungs": [{"concurrency": 1, "reps": reps, "tok_s_series": [100.0, 101.0, 99.0],
                         "tok_s_mean": 100.0, "errors_total": 0}]}
    boot = {"schema": 1, "engine": "vllm", "url": ladder["url"], "model": "nano",
            "status": "ready", "passed": True, "time_to_ready_s": 1.0,
            "first_token_s": 0.1, "total_s": 1.1, "timeout_s": 1800}
    coherency = {"schema": 1, "url": ladder["url"], "model": "nano",
                 "determinism_ok": True, "toolcall_ok": True, "think_leak_ok": True,
                 "passed": True, "details": {"fixture": "synthetic only"}}
    provenance = {"engine": "vllm", "hardware": {"hardware_id": "gb10"},
                  "client": {"driver_sha256": "a" * 64, "environment": {"W55_PROMPT_MODE": "essay"}},
                  "model": {"served_model_name": "nano"},
                  "environment": {},
                  "workload": {"spec": {"on": False, "k": 0}},
                  "serve_command": ["vllm", "serve", "fixture/model"]}
    return ladder, boot, coherency, provenance


def selftest():
    inputs = fixture()
    frozen = read_json(pathlib.Path(__file__).with_name("workloads.json"))[0]
    original = copy.deepcopy(inputs)
    cell = assemble(*inputs, "lat", "fixture", frozen)[0]
    assert cell["hardware"]["hardware_id"] == "gb10"
    assert cell["verdict"] == "PARTIAL" and cell["rehearsal_verdict"] == "PASS"
    assert cell["metrics"]["ttft_p50_ms"] == 11.0
    assert cell["raw"]["rung"] == inputs[0]["rungs"][0]
    assert cell["engine_version"]["image_digest"] is None and cell["schema_gaps"]
    assert inputs == original, "assembler mutated evidence inputs"
    print("PASS positive: raw reps retained, mean-per-rep latency labelled, unknown provenance null")
    multi = list(copy.deepcopy(inputs))
    second = copy.deepcopy(multi[0]["rungs"][0])
    second["concurrency"] = 16
    for rep in second["reps"]:
        rep.update(n_ok=16, completion_tokens=4096, prompt_tokens=20800,
                   completion_tokens_per_req=[256] * 16, wall_s=4096 / rep["tok_s"])
    multi[0]["rungs"].append(second)
    assert len(assemble(*multi, "lat", "fixture", frozen)) == 2
    print("PASS positive: one artifact per C=1 and C=16 rung")

    cases = []
    def changed(label, input_index, update):
        candidate = list(copy.deepcopy(inputs))
        update(candidate[input_index])
        cases.append((label, candidate))
    changed("frozen workload mismatch", 0, lambda d: d.update(osl=512))
    changed("client hash mismatch", 3, lambda d: d["client"].update(driver_sha256="b" * 64))
    changed("boot engine mismatch", 1, lambda d: d.update(engine="atlas"))
    changed("gate model mismatch", 2, lambda d: d.update(model="other"))
    changed("Hopper wrongly labelled GB10", 3, lambda d: d["hardware"].update(hardware_id="h100"))
    changed("nonfinite metric", 0, lambda d: d["rungs"][0]["reps"][0].update(tok_s=float("nan")))
    changed("inconsistent gate verdict", 2, lambda d: d.update(toolcall_ok=False))
    for label, candidate in cases:
        try:
            assemble(*candidate, "lat", "fixture", frozen)
        except InvalidInput:
            print(f"PASS negative: {label} rejected")
        else:
            raise AssertionError(f"accepted bad fixture: {label}")

    for label, update in (
        ("vacuous measured request", lambda d: d[0]["rungs"][0]["reps"][0].update(completion_tokens_per_req=[200])),
        ("missing completion usage", lambda d: d[0]["rungs"][0]["reps"][0].pop("completion_tokens_per_req")),
        ("missing TPOT", lambda d: d[0]["rungs"][0]["reps"][0].update(tpot_p50_ms=None)),
        ("stale throughput summary", lambda d: d[0]["rungs"][0].update(tok_s_mean=999.0)),
        ("boot exceeds campaign cap", lambda d: d[1].update(total_s=1801.0)),
        ("failed actual coherency gate", lambda d: d[2].update(passed=False, toolcall_ok=False)),
    ):
        candidate = list(copy.deepcopy(inputs))
        update(candidate)
        failed = assemble(*candidate, "lat", "fixture", frozen)[0]
        assert failed["rehearsal_verdict"] == "FAIL" and failed["verdict"] == "NO-GO", label
        assert failed["raw"]["rung"] == candidate[0]["rungs"][0], label
        if label == "missing TPOT":
            assert failed["metrics"]["tpot_p50_ms"] is None
        print(f"PASS negative: {label} retained as failed cell")
    print("SELFTEST OK: synthetic fixtures only; no engine or GPU used")


def read_json(path):
    raw = pathlib.Path(path).read_bytes()
    value = json.loads(raw)
    finite_json(value)
    return value, {"path": str(pathlib.Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    for flag in ("ladder", "boot", "coherency", "provenance", "workload", "run-id", "out-dir"):
        parser.add_argument("--" + flag)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not all(getattr(args, flag) for flag in ("ladder", "boot", "coherency", "provenance", "workload", "run_id", "out_dir")):
        parser.error("--ladder --boot --coherency --provenance --workload --run-id --out-dir are required")
    try:
        loaded = {key: read_json(getattr(args, key)) for key in ("ladder", "boot", "coherency", "provenance")}
        frozen, frozen_source = read_json(pathlib.Path(__file__).with_name("workloads.json"))
        cells = assemble(*(loaded[key][0] for key in ("ladder", "boot", "coherency", "provenance")),
                         args.workload, args.run_id, frozen)
        sources = {key: record[1] for key, record in loaded.items()}
        sources["workloads"] = frozen_source
        sources["assembler"] = {"path": str(pathlib.Path(__file__).resolve()),
                                "sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
        targets = [pathlib.Path(args.out_dir) / f"{args.run_id}-{cell['engine']}-{args.workload}-c{cell['workload']['concurrency']}.json"
                   for cell in cells]
        require(not any(path.exists() for path in targets), "output already exists; use a new run ID or directory")
        pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        for cell, target in zip(cells, targets):
            cell["source_artifacts"] = sources
            target.write_text(json.dumps(cell, indent=2, allow_nan=False) + "\n")
            print(f"{target}: {cell['rehearsal_verdict']} ({cell['verdict']})")
        return 1 if any(cell["rehearsal_verdict"] != "PASS" for cell in cells) else 0
    except (InvalidInput, OSError, json.JSONDecodeError) as exc:
        print(f"refusing assembly: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
