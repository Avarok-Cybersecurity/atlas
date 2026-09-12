#!/usr/bin/env python3
"""Turn two ladder JSONs into the campaign's Pareto table.

Input is two files written by `bench/ladder38/harness_w55_conc_ladder.py` --
one per engine. Output is a markdown table and a JSON twin: per concurrency,
tok/s, TTFT p50/p99, TPOT p50, the Atlas/vLLM ratio, and WIN, TIE or LOSS.

★ The fairness oracle is the reason this exists rather than a spreadsheet.
Before it compares anything it refuses two files whose workload axes disagree:
isl, osl, temperature, seed, chat_template_kwargs, reps, warmup and driver hash.
The hash is the ladder client source, including its penalty and nonce pins;
it is not the server's version. That failure is otherwise
SILENT -- two legs run days apart with one flag changed produce a table that
looks exactly like a valid one, and the only trace is a number that seems
surprising. `bench/ladder38/published.json` carries a whole `harness_shas`
block arguing after the fact that two legs really were equivalent; refusing up
front is cheaper than arguing later.

Usage:
  compare.py --atlas ATLAS.json --vllm VLLM.json [--out-md F] [--out-json F]
  compare.py --selftest
"""

import argparse
import copy
import json
import math
import pathlib
import statistics
import sys

# The workload axes that must agree for two runs to be comparable at all.
#
# Every one of these changes what was measured, not how well it went. Anything
# NOT in this list -- label, url, served-model alias, timestamps -- may differ.
# Model revision, server speculation and hardware require separate artifacts:
# the ladder does not emit them, and header equality cannot certify them.
PARITY_KEYS = ("isl", "osl", "temperature", "seed", "chat_template_kwargs",
               "reps", "warmup", "driver_sha256")


class Mismatch(Exception):
    """The two files do not describe the same workload."""


def load(path):
    d = json.loads(pathlib.Path(path).read_text())
    if "rungs" not in d:
        raise Mismatch(f"{path}: no 'rungs' -- not a harness_w55_conc_ladder.py output")
    return d


def valid_header_value(key, value):
    if key in ("isl", "osl", "reps", "warmup"):
        return type(value) is int and value >= (0 if key == "warmup" else 1)
    if key == "temperature":
        return finite_number(value) and value >= 0
    if key == "seed":
        return type(value) is int
    if key == "chat_template_kwargs":
        return isinstance(value, dict) and type(value.get("enable_thinking")) is bool
    if key == "driver_sha256":
        return (isinstance(value, str) and len(value) == 64
                and all(c in "0123456789abcdef" for c in value))
    raise ValueError(f"no validation for parity key {key}")


def assert_comparable(a, b, a_name="atlas", b_name="vllm"):
    """Refuse the comparison unless every parity axis matches.

    A MISSING key is a mismatch too, not a pass. An older harness that did not
    record `chat_template_kwargs` cannot be shown to have disabled thinking,
    and "we probably did" is exactly the claim this campaign cannot make.
    """
    problems = []
    for key in PARITY_KEYS:
        if key not in a or key not in b:
            problems.append(
                f"{key}: {a_name}={a.get(key, '<absent>')!r} {b_name}={b.get(key, '<absent>')!r} "
                "(absent is not a match -- the run cannot show it used this value)"
            )
        elif not valid_header_value(key, a[key]) or not valid_header_value(key, b[key]):
            problems.append(f"{key}: invalid header value: {a_name}={a[key]!r} {b_name}={b[key]!r}")
        elif a[key] != b[key]:
            problems.append(f"{key}: {a_name}={a[key]!r} vs {b_name}={b[key]!r}")
    if problems:
        raise Mismatch(
            "refusing to compare two runs that did not measure the same workload:\n  - "
            + "\n  - ".join(problems)
        )


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def rung_stats(rung, expected_reps, osl):
    """One rung's headline numbers, aggregated over its timed reps.

    tok/s comes from the harness's own mean over the rep series. The latency
    percentiles are per-rep, so they are averaged across reps -- a median of
    medians would hide a rep that was uniformly slow, which on a shared box is
    the reading that matters.
    """
    reps = rung.get("reps") or []
    if not isinstance(reps, list) or any(not isinstance(r, dict) for r in reps):
        raise Mismatch("rung reps must be a list of records")
    issues = []
    if not reps or len(reps) != expected_reps:
        issues.append(f"expected {expected_reps} timed reps, got {len(reps)}")
    if rung.get("errors_total") != 0:
        issues.append("errors_total is absent or nonzero")
    rates = []
    for i, rep in enumerate(reps):
        if rep.get("n_err") != 0 or rep.get("n_ok") != rung["concurrency"]:
            issues.append(f"rep {i}: request errors or incomplete request count")
        tokens = rep.get("completion_tokens_per_req")
        if (not isinstance(tokens, list) or len(tokens) != rung["concurrency"]
                or any(type(n) is not int or n < 0 for n in tokens)):
            issues.append(f"rep {i}: missing or invalid per-request completion usage")
        elif any(n < 0.8 * osl for n in tokens):
            issues.append(f"rep {i}: vacuity (a request returned <80% of OSL {osl})")
        rate = rep.get("tok_s")
        if not finite_number(rate) or rate <= 0:
            issues.append(f"rep {i}: throughput must be finite and positive")
        else:
            rates.append(rate)
    if rates and (max(rates) - min(rates)) / statistics.fmean(rates) > 0.10:
        issues.append("timed throughput spread exceeds 10%")
    throughput = rung.get("tok_s_mean")
    if not finite_number(throughput) or throughput <= 0:
        issues.append("tok_s_mean must be finite and positive")
        throughput = None

    def mean_of(key):
        vals = [r[key] for r in reps if finite_number(r.get(key)) and r[key] >= 0]
        if len(vals) != len(reps):
            issues.append(f"{key}: missing, negative or nonfinite latency")
        return statistics.fmean(vals) if vals else None

    return {
        "concurrency": rung["concurrency"],
        "tok_s": throughput,
        "ttft_p50_ms": mean_of("ttft_p50_ms"),
        "ttft_p99_ms": mean_of("ttft_p99_ms"),
        "tpot_p50_ms": mean_of("tpot_p50_ms"),
        "errors": rung.get("errors_total"),
        "issues": issues,
    }


def index_rungs(run):
    rungs = run.get("rungs")
    if not isinstance(rungs, list) or not rungs:
        raise Mismatch("rungs must be a nonempty list")
    by_c = {}
    for rung in rungs:
        c = rung.get("concurrency") if isinstance(rung, dict) else None
        if type(c) is not int or c <= 0 or c in by_c:
            raise Mismatch(f"invalid or duplicate concurrency: {c!r}")
        by_c[c] = rung_stats(rung, run["reps"], run["osl"])
    return by_c


def compare(atlas, vllm):
    assert_comparable(atlas, vllm)
    atlas_by_c = index_rungs(atlas)
    vllm_by_c = index_rungs(vllm)
    rows = []
    for concurrency in sorted(atlas_by_c.keys() | vllm_by_c.keys()):
        a = atlas_by_c.get(concurrency)
        b = vllm_by_c.get(concurrency)
        if a is None or b is None:
            # Reported, not dropped: a rung one engine ran and the other did
            # not is a hole in the campaign, and a table that silently omits it
            # reads as complete.
            rows.append({**{f"atlas_{k}": v for k, v in (a or {}).items()},
                         **{f"vllm_{k}": v for k, v in (b or {}).items()},
                         "concurrency": concurrency, "verdict": "NO-PAIR"})
            continue
        invalid = bool(a["issues"] or b["issues"])
        ratio = None if invalid else a["tok_s"] / b["tok_s"]
        rows.append({
            "concurrency": a["concurrency"],
            "atlas_tok_s": a["tok_s"], "vllm_tok_s": b["tok_s"],
            "atlas_ttft_p50_ms": a["ttft_p50_ms"], "vllm_ttft_p50_ms": b["ttft_p50_ms"],
            "atlas_ttft_p99_ms": a["ttft_p99_ms"], "vllm_ttft_p99_ms": b["ttft_p99_ms"],
            "atlas_tpot_p50_ms": a["tpot_p50_ms"], "vllm_tpot_p50_ms": b["tpot_p50_ms"],
            "atlas_errors": a["errors"], "vllm_errors": b["errors"],
            "atlas_issues": a["issues"], "vllm_issues": b["issues"],
            "ratio": ratio,
            # Throughput decides the label because that is what the campaign's
            # scoreboard is scored on; the latency columns sit beside it so a
            # win bought by a TTFT regression is visible in the same row rather
            # than in a footnote.
            "verdict": ("INVALID" if invalid else "TIE" if ratio == 1.0 else
                        "WIN" if ratio is not None and ratio > 1.0 else "LOSS"),
        })
    return {
        "schema": 1,
        "workload": {k: atlas[k] for k in PARITY_KEYS if k in atlas},
        "atlas_label": atlas.get("label"), "vllm_label": vllm.get("label"),
        "atlas_model": atlas.get("model"), "vllm_model": vllm.get("model"),
        "rows": rows,
        "rungs_won": sum(1 for r in rows if r["verdict"] == "WIN"),
        "rungs_compared": sum(1 for r in rows if r["verdict"] in ("WIN", "TIE", "LOSS")),
    }


def to_markdown(result):
    w = result["workload"]
    num = lambda v, p=2: "--" if v is None else f"{v:.{p}f}"
    lines = [
        f"### ISL {w.get('isl')} / OSL {w.get('osl')} "
        f"(temp {w.get('temperature')}, seed {w.get('seed')})",
        "",
        "| C | Atlas tok/s | vLLM tok/s | ratio | Atlas TTFT p50/p99 ms | vLLM TTFT p50/p99 ms "
        "| Atlas TPOT p50 ms | vLLM TPOT p50 ms | rung |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in result["rows"]:
        if r["verdict"] == "NO-PAIR":
            lines.append(f"| {r['concurrency']} | {num(r.get('atlas_tok_s'))} "
                         f"| {num(r.get('vllm_tok_s'))} | -- | -- "
                         "| -- | -- | -- | **NO-PAIR** |")
            continue
        lines.append(
            f"| {r['concurrency']} | {num(r['atlas_tok_s'])} | {num(r['vllm_tok_s'])} "
            f"| {num(r['ratio']) + 'x' if r['ratio'] is not None else '--'} "
            f"| {num(r['atlas_ttft_p50_ms'], 0)} / {num(r['atlas_ttft_p99_ms'], 0)} "
            f"| {num(r['vllm_ttft_p50_ms'], 0)} / {num(r['vllm_ttft_p99_ms'], 0)} "
            f"| {num(r['atlas_tpot_p50_ms'])} | {num(r['vllm_tpot_p50_ms'])} "
            f"| **{r['verdict']}** |"
        )
    lines += ["", f"{result['rungs_won']}/{result['rungs_compared']} rungs won.", ""]
    for r in result["rows"]:
        for side in ("atlas", "vllm"):
            if r.get(f"{side}_issues"):
                lines.append(f"C={r['concurrency']} {side}: " + "; ".join(r[f"{side}_issues"]))
    return "\n".join(lines)


def selftest():
    """Validate the instrument against three known cases.

    The matched pair must produce the arithmetic the fixtures were built to
    give (Atlas 2x at C=1, 0.5x at C=16 -- one WIN and one LOSS, so a table
    that reported everything as a win would be caught). The mismatched pair
    must be REFUSED: `fixtures/vllm_tiny_mismatched.json` differs from the
    Atlas fixture in exactly one axis, `osl`, which is the shape of the mistake
    that actually happens -- a re-run with one flag edited. And the third case
    is a file with no `chat_template_kwargs` at all, which must also be refused
    rather than assumed equivalent.
    """
    here = pathlib.Path(__file__).parent / "fixtures"
    atlas = load(here / "atlas_tiny.json")
    result = compare(atlas, load(here / "vllm_tiny.json"))
    print(to_markdown(result))
    assert result["rows"][0]["ratio"] == 2.0, result["rows"][0]
    assert result["rows"][0]["verdict"] == "WIN", result["rows"][0]
    assert result["rows"][1]["ratio"] == 0.5, result["rows"][1]
    assert result["rows"][1]["verdict"] == "LOSS", result["rows"][1]
    assert result["rungs_won"] == 1 and result["rungs_compared"] == 2, result
    assert result["rows"][0]["atlas_ttft_p50_ms"] == 100.0, result["rows"][0]

    for bad, expect in (("vllm_tiny_mismatched.json", "osl"),
                        ("vllm_tiny_no_thinking_key.json", "chat_template_kwargs")):
        try:
            compare(atlas, load(here / bad))
        except Mismatch as e:
            assert expect in str(e), f"{bad}: refusal did not name {expect}: {e}"
            print(f"REFUSED {bad}: {str(e).splitlines()[-1].strip()}")
        else:
            raise AssertionError(f"{bad} must be refused, not compared")
    # Arithmetic identity and rung-set union are independent of GPU timings.
    failures = []
    tied = compare(atlas, copy.deepcopy(atlas))
    if not (all(r["ratio"] == 1.0 and r["verdict"] == "TIE" for r in tied["rows"])
            and tied["rungs_compared"] == 2 and tied["rungs_won"] == 0):
        failures.append("identity oracle: A/A ratio 1.0 must be TIE and count as compared")
    for name, a, b in (("atlas-only", atlas, {**atlas, "rungs": atlas["rungs"][:1]}),
                       ("vllm-only", {**atlas, "rungs": atlas["rungs"][:1]}, atlas)):
        unpaired = compare(a, b)
        rows = {r["concurrency"]: r for r in unpaired["rows"]}
        if 16 not in rows or rows[16]["verdict"] != "NO-PAIR":
            failures.append(f"rung-union oracle: {name} C=16 must remain NO-PAIR")
        elif name == "vllm-only" and "200.00" not in to_markdown(unpaired):
            failures.append("rung-union oracle: vllm-only throughput must render in its column")
    # PRD sections 4/9: no errors, every request >=80% OSL, <=10% spread.
    # Mutate raw ladder fields, including a dishonest errors_total=0 summary.
    invalid_cases = [
        ("rung errors", lambda r: r.update(errors_total=1)),
        ("rep errors", lambda r: r["reps"][0].update(n_err=1)),
        ("missing errors", lambda r: r.pop("errors_total")),
        ("vacuity", lambda r: r["reps"][0].update(completion_tokens_per_req=[204])),
        ("missing usage", lambda r: r["reps"][0].pop("completion_tokens_per_req")),
        ("missing requests", lambda r: r["reps"][0].update(n_ok=0)),
        ("incomplete reps", lambda r: r.update(reps=r["reps"][:1])),
        ("empty reps", lambda r: r.update(reps=[])),
        ("zero throughput", lambda r: r.update(tok_s_mean=0)),
        ("nonfinite throughput", lambda r: r.update(tok_s_mean=float("nan"))),
        ("nonfinite latency", lambda r: r["reps"][0].update(ttft_p50_ms=float("inf"))),
        ("unstable reps", lambda r: r["reps"][0].update(tok_s=80.0)),
    ]
    for name, mutate in invalid_cases:
        bad = copy.deepcopy(atlas)
        mutate(bad["rungs"][0])
        for side, a, b in (("atlas", bad, atlas), ("vllm", atlas, bad)):
            result_bad = compare(a, b)
            row = result_bad["rows"][0]
            if (row["verdict"] != "INVALID" or row.get("ratio") is not None
                    or not row.get(f"{side}_issues") or result_bad["rungs_compared"] != 1):
                failures.append(f"validity oracle: {side} {name} must be INVALID, unscored, with reasons")
    # The 80% floor is per request, not an average; 205/256 passes.
    boundary = copy.deepcopy(atlas)
    boundary["rungs"][0]["reps"][0]["completion_tokens_per_req"] = [205]
    if compare(boundary, atlas)["rows"][0]["verdict"] != "TIE":
        failures.append("vacuity oracle: the first integer at or above 80% must pass")
    mixed = copy.deepcopy(atlas)
    mixed["rungs"][1]["reps"][0]["completion_tokens_per_req"][0] = 204
    if compare(mixed, atlas)["rows"][1]["verdict"] != "INVALID":
        failures.append("vacuity oracle: one short request at C=16 must fail despite a full average")
    for name, rungs in (("empty", []), ("duplicate", atlas["rungs"] * 2)):
        try:
            compare({**atlas, "rungs": rungs}, atlas)
        except Mismatch:
            pass
        else:
            failures.append(f"rung identity oracle: {name} rungs must be refused")
    # The ladder header pins rep counts and its full request-body source hash.
    for key, changed in (("reps", 3), ("warmup", 0), ("driver_sha256", "1" * 64)):
        for name, bad in (("changed", {**atlas, key: changed}),
                          ("absent", {k: v for k, v in atlas.items() if k != key})):
            try:
                compare(atlas, bad)
            except Mismatch as e:
                assert key in str(e), e
            except (KeyError, TypeError) as e:
                failures.append(f"header oracle: {name} {key} must be REFUSED, not crash: {e}")
            else:
                failures.append(f"header oracle: {name} {key} must be REFUSED")
    for key, value in (("isl", 0), ("osl", 0), ("reps", 0), ("warmup", -1),
                       ("temperature", None), ("seed", None),
                       ("chat_template_kwargs", None), ("driver_sha256", "")):
        bad = {**atlas, key: value}
        try:
            compare(bad, copy.deepcopy(bad))
        except Mismatch as e:
            assert key in str(e), e
        else:
            failures.append(f"header oracle: equal invalid {key} must not establish parity")
    for failure in failures:
        print("FAIL:", failure)
    assert not failures, "; ".join(failures)
    print("SELFTEST OK: arithmetic, parity refusal, A/A identity, symmetric rung union and validity exclusions")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--atlas")
    ap.add_argument("--vllm")
    ap.add_argument("--out-md")
    ap.add_argument("--out-json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return 0
    if not a.atlas or not a.vllm:
        ap.error("--atlas and --vllm are required (or pass --selftest)")

    try:
        result = compare(load(a.atlas), load(a.vllm))
    except Mismatch as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    md = to_markdown(result)
    print(md)
    if a.out_md:
        pathlib.Path(a.out_md).write_text(md + "\n")
    if a.out_json:
        pathlib.Path(a.out_json).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
