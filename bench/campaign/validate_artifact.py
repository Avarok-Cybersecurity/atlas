#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Validate a campaign cell artifact against bench/campaign/artifact.schema.json.

Stdlib only, on purpose. This runs at the END of every cell, on a rented box,
after the engine has been torn down -- the moment where `pip install jsonschema`
is most likely to be the thing that fails and least likely to be worth the
minutes. So it implements exactly the draft-2020-12 keyword subset the schema
uses and refuses anything it does not understand, rather than silently passing
a constraint it cannot evaluate:

    type (including nullable as a type array), const, enum, required,
    properties, additionalProperties:false, items, minimum, pattern

An unknown keyword is a validator error, not a skipped check. A gate that
silently ignores a rule is worse than no gate, because it reports PASS.

Conditional rules -- "a null PTX receipt needs a reason", "CERTIFIED means no
failing stage" -- are the ones draft 2020-12 writes with if/then. They live in
the schema's `x-atlas-cross-checks` list by name and are implemented here by
the same names; a name in one and not the other is reported as a validator bug
before any artifact is looked at.

Usage:
  validate_artifact.py ARTIFACT.json [--schema PATH]
  validate_artifact.py --selftest [--schema PATH]

Exit: 0 valid · 1 invalid (every violation printed, path first) · 2 usage.
"""

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_SCHEMA = HERE / "artifact.schema.json"
FIXTURES = HERE / "fixtures"

SUPPORTED = {
    "$schema", "$id", "title", "description", "type", "const", "enum",
    "required", "properties", "additionalProperties", "items", "minimum",
    "pattern", "x-atlas-cross-checks", "x-atlas-cross-checks-note",
}

TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def type_ok(value, name):
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name in ("number", "integer"):
        # bool is an int in Python; a boolean is never a number here.
        if isinstance(value, bool):
            return False
        return isinstance(value, TYPES[name])
    return isinstance(value, TYPES[name])


def validate(value, schema, path, errors):
    for key in schema:
        if key not in SUPPORTED:
            errors.append(f"{path}: validator does not implement schema keyword "
                          f"'{key}' -- refusing to report PASS on a rule it cannot check")

    if "type" in schema:
        names = schema["type"]
        names = [names] if isinstance(names, str) else names
        if not any(type_ok(value, n) for n in names):
            errors.append(f"{path}: expected type {'|'.join(names)}, "
                          f"got {json_type(value)}")
            return

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected the constant {schema['const']!r}, got {value!r}")

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of "
                      f"{', '.join(repr(v) for v in schema['enum'])}")

    if "pattern" in schema and isinstance(value, str):
        if not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} does not match /{schema['pattern']}/")

    if "minimum" in schema and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")

    if isinstance(value, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name}: required property is missing")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in props:
                    errors.append(f"{path}.{name}: property is not allowed by the schema")
        for name, sub in props.items():
            if name in value:
                validate(value[name], sub, f"{path}.{name}", errors)

    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{i}]", errors)


def json_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


# ── cross-checks: the conditional rules, by name ─────────────────────────────


def _ptx_receipt_or_reason(doc, errors):
    """A null PTX receipt has to say why it is null."""
    if doc.get("ptx_gate_receipt_sha256") is None:
        reason = doc.get("ptx_gate_not_applicable_reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append("$.ptx_gate_not_applicable_reason: must be a non-empty "
                          "string when ptx_gate_receipt_sha256 is null")
    elif doc.get("ptx_gate_not_applicable_reason") is not None:
        errors.append("$.ptx_gate_not_applicable_reason: must be null when a PTX "
                      "receipt sha256 is recorded")


def _world_size_equals_gpu_count(doc, errors):
    """One rank per GPU, across every node of the deployment."""
    top, hw = doc.get("topology"), doc.get("hardware")
    if not isinstance(top, dict) or not isinstance(hw, dict):
        return
    if top.get("world_size") != hw.get("gpu_count"):
        errors.append(f"$.topology.world_size: {top.get('world_size')} does not equal "
                      f"$.hardware.gpu_count {hw.get('gpu_count')}")


def _spec_method_iff_on(doc, errors):
    """`none` and a null draft count mean off, and nothing else does."""
    spec = doc.get("spec")
    if not isinstance(spec, dict):
        return
    if spec.get("on") is True:
        if spec.get("method") == "none":
            errors.append("$.spec.method: cannot be 'none' while $.spec.on is true")
        if not isinstance(spec.get("k"), int) or isinstance(spec.get("k"), bool):
            errors.append("$.spec.k: must be an integer draft count while $.spec.on is true")
    elif spec.get("on") is False:
        if spec.get("method") != "none":
            errors.append(f"$.spec.method: must be 'none' while $.spec.on is false, "
                          f"got {spec.get('method')!r}")
        if spec.get("k") is not None:
            errors.append("$.spec.k: must be null while $.spec.on is false")


def _verdict_failing_stage(doc, errors):
    """CERTIFIED is the only verdict with nothing to name."""
    verdict, stage = doc.get("verdict"), doc.get("failing_stage")
    if verdict == "CERTIFIED" and stage is not None:
        errors.append(f"$.failing_stage: must be null when the verdict is CERTIFIED, "
                      f"got {stage!r}")
    if verdict in ("NO-GO", "PARTIAL") and stage is None:
        errors.append(f"$.failing_stage: must name a stage when the verdict is {verdict}")


def _ladder_cannot_pool(doc, errors):
    """The ladder discards per-request latencies, so it cannot pool.

    This is the SCHEMA-GAPS finding made mechanical: `harness_w55_conc_ladder.py`
    stores one percentile per rep and throws the samples away. Any artifact whose
    client is that harness and whose percentile_method claims pooled_requests is
    describing a reduction its own instrument cannot perform.
    """
    client = doc.get("client")
    metrics = doc.get("metrics")
    if not isinstance(client, dict) or not isinstance(metrics, dict):
        return
    name = str(client.get("name") or "")
    if "harness_w55_conc_ladder" in name and metrics.get("percentile_method") == "pooled_requests":
        errors.append("$.metrics.percentile_method: the ladder client keeps one "
                      "percentile per rep and discards per-request latencies, so it "
                      "cannot produce pooled_requests percentiles")


def _certified_needs_gate_evidence(doc, errors):
    """CERTIFIED is a claim about evidence, so the evidence has to be present.

    An artifact reached this validator CERTIFIED with every gate value null and
    a `{}` paired artifact, and it was accepted: the schema alone cannot say
    that a verdict must be supported, only what shape each field has. So the
    rule is written here. It duplicates no judgement -- cell_assemble.py
    decides the verdict from the same evidence -- but a verdict is exactly the
    field an artifact can be edited to claim, and this is the check that
    refuses the claim.
    """
    if doc.get("verdict") != "CERTIFIED":
        return
    boot = doc.get("boot")
    if not isinstance(boot, dict) or boot.get("pass") is not True:
        errors.append("$.boot.pass: must be true when the verdict is CERTIFIED")
    coh = doc.get("coherency") if isinstance(doc.get("coherency"), dict) else {}
    for key in ("determinism_ok", "toolcall_ok", "think_leak_ok", "known_answer_ok"):
        if coh.get(key) is not True:
            errors.append(f"$.coherency.{key}: must be true when the verdict is "
                          f"CERTIFIED, got {coh.get(key)!r}")
    metrics = doc.get("metrics") if isinstance(doc.get("metrics"), dict) else {}
    if metrics.get("vacuous") is not False:
        errors.append(f"$.metrics.vacuous: must be false when the verdict is "
                      f"CERTIFIED, got {metrics.get('vacuous')!r} -- null is "
                      f"'the floor could not be applied', not 'it was cleared'")
    if metrics.get("tok_s_mean") is None:
        errors.append("$.metrics.tok_s_mean: a CERTIFIED cell has a measured rate")
    pair = doc.get("paired_cell") if isinstance(doc.get("paired_cell"), dict) else {}
    if not isinstance(pair.get("cell_id"), str) or not pair["cell_id"].strip():
        errors.append("$.paired_cell.cell_id: a CERTIFIED cell names its pair")
    if pair.get("within_24h") is not True:
        errors.append(f"$.paired_cell.within_24h: must be true when the verdict is "
                      f"CERTIFIED, got {pair.get('within_24h')!r}")


def _certified_needs_engine_identity(doc, errors):
    """A CERTIFIED cell has to say WHICH engine build produced its numbers.

    `git_sha` used to be filled in with the CAMPAIGN checkout's HEAD for either
    engine, so an artifact could name a revision nothing had verified while the
    two fields that are actually verifiable -- the digest of the image that ran
    and the hash of the binary that ran -- were both null, and still certify.
    A revision that was read off the harness is not evidence about the engine,
    so CERTIFIED requires one of the two that is.
    """
    if doc.get("verdict") != "CERTIFIED":
        return
    ev = doc.get("engine_version") if isinstance(doc.get("engine_version"), dict) else {}
    if not isinstance(ev.get("image_digest"), str) and not isinstance(
            ev.get("binary_sha256"), str):
        errors.append("$.engine_version.image_digest: a CERTIFIED cell identifies the "
                      "engine that produced it -- image_digest (container) or "
                      "binary_sha256 (local binary) must be non-null; a harness "
                      "checkout SHA is not the engine's identity")


CROSS_CHECKS = {
    "ptx_receipt_or_reason": _ptx_receipt_or_reason,
    "world_size_equals_gpu_count": _world_size_equals_gpu_count,
    "spec_method_iff_on": _spec_method_iff_on,
    "verdict_failing_stage": _verdict_failing_stage,
    "ladder_cannot_pool": _ladder_cannot_pool,
    "certified_needs_gate_evidence": _certified_needs_gate_evidence,
    "certified_needs_engine_identity": _certified_needs_engine_identity,
}


def validate_document(doc, schema):
    errors = []
    declared = schema.get("x-atlas-cross-checks", [])
    for name in declared:
        if name not in CROSS_CHECKS:
            errors.append(f"$: schema declares cross-check '{name}' which "
                          f"validate_artifact.py does not implement")
    for name in CROSS_CHECKS:
        if name not in declared:
            errors.append(f"$: validate_artifact.py implements cross-check '{name}' "
                          f"which the schema does not declare")
    validate(doc, schema, "$", errors)
    for name in declared:
        if name in CROSS_CHECKS:
            CROSS_CHECKS[name](doc, errors)
    return errors


# ── selftest ─────────────────────────────────────────────────────────────────

# Each bad fixture names the ONE path its rejection must mention. A validator
# that rejects for the wrong reason is not a validator, it is a coin flip.
BAD_FIXTURES = {
    "bad_missing_required.json": "$.metrics.ladder_json_path",
    "bad_wrong_enum.json": "$.model.quant",
    "bad_sha_pattern.json": "$.engine_version.image_digest",
    "bad_percentile_method_missing.json": "$.metrics.percentile_method",
    "bad_ladder_claims_pooled.json": "$.metrics.percentile_method",
    "bad_certified_with_failing_stage.json": "$.failing_stage",
    "bad_certified_without_engine_identity.json": "$.engine_version.image_digest",
}


def selftest(schema):
    checks = 0
    fails = []

    def check(cond, what):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(what)

    good = FIXTURES / "good_atlas_cell.json"
    check(good.exists(), f"missing fixture {good}")
    if good.exists():
        errs = validate_document(json.loads(good.read_text()), schema)
        check(not errs, f"{good.name} must validate, got: {errs}")

    for name, path in BAD_FIXTURES.items():
        f = FIXTURES / name
        check(f.exists(), f"missing fixture {f}")
        if not f.exists():
            continue
        errs = validate_document(json.loads(f.read_text()), schema)
        check(bool(errs), f"{name} must be REJECTED but validated clean")
        check(any(e.startswith(path + ":") for e in errs),
              f"{name}: rejection must name {path}, got {errs}")

    # The validator must refuse a schema keyword it cannot evaluate rather than
    # reporting PASS on it.
    errs = []
    validate("x", {"type": "string", "multipleOf": 2}, "$", errs)
    check(any("multipleOf" in e for e in errs),
          "an unimplemented keyword must be reported, not skipped")

    # A boolean is not a number, and an integer is not a boolean.
    errs = []
    validate(True, {"type": "integer"}, "$.b", errs)
    check(bool(errs), "true must not validate as an integer")
    errs = []
    validate(1, {"type": "boolean"}, "$.i", errs)
    check(bool(errs), "1 must not validate as a boolean")

    for f in fails:
        print(f"FAIL: {f}")
    print(f"validate_artifact selftest: {checks - len(fails)}/{checks} checks passed "
          f"({len(BAD_FIXTURES)} known-bad fixtures)")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifact", nargs="?")
    ap.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    schema = json.loads(pathlib.Path(args.schema).read_text())
    if args.selftest:
        return selftest(schema)
    if not args.artifact:
        print("ERROR: an artifact path is required (or --selftest)", file=sys.stderr)
        return 2
    try:
        doc = json.loads(pathlib.Path(args.artifact).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read {args.artifact}: {exc}", file=sys.stderr)
        return 2

    errors = validate_document(doc, schema)
    for e in errors:
        print(e)
    if errors:
        print(f"INVALID: {args.artifact} ({len(errors)} violation(s))")
        return 1
    print(f"VALID: {args.artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
