#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Gate and pair eligibility checks shared by campaign artifact assembly."""

import datetime

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
