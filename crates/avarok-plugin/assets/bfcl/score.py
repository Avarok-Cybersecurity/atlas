# SPDX-License-Identifier: AGPL-3.0-only
"""Score BFCL v4 single-turn responses with bfcl-eval's AST checker.

Ported from mlcommons/endpoints `inference_endpoint/evaluation/bfcl_v4_scorer.py`
(Apache-2.0, NVIDIA CORPORATION). The aggregation is reproduced exactly, because
`overall_accuracy` and `normalized_single_turn_score` are the numbers every
recorded baseline and the MLPerf-edge floor are expressed in:

  * live          — sample-weighted mean over its subsets
  * non_live      — HIERARCHICAL: the simple_* subsets collapse to one score
                    first, then an unweighted mean with the rest
  * hallucination — unweighted mean
  * normalized    — unweighted mean of the three category scores
  * overall       — plain mean over every sample

One deliberate difference from the reference. It decides "did the model make a
tool call?" for hallucination samples by re-parsing the serialized output, and
documents the false positive that follows when a refusal happens to be valid
JSON. We are the client, so we know: the caller passes `has_tool_calls`
directly. Same intent, no heuristic.

Input:  --dataset  the JSONL provision.py wrote (for ground truth)
        --responses  one JSON object per line:
            {"sample_id", "subset", "has_tool_calls", "tool_calls":[{"name","arguments"}]}
Output: one JSON object on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

CATEGORY_MAP = {
    "non_live": [
        "simple_python",
        "simple_java",
        "simple_javascript",
        "multiple",
        "parallel",
        "parallel_multiple",
    ],
    "live": ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"],
    "hallucination": ["irrelevance", "live_irrelevance"],
}
HALLUCINATION_SUBSETS = set(CATEGORY_MAP["hallucination"])
SIMPLE_AST_SUBSETS = [s for s in CATEGORY_MAP["non_live"] if s.startswith("simple_")]
SUBSET_LANGUAGE_NAMES = {"simple_java": "JAVA", "simple_javascript": "JAVASCRIPT"}
CATEGORY_AGGREGATION = {
    "live": "sample_weighted",
    "non_live": "hierarchical",
    "hallucination": "unweighted",
}
AST_CHECKER_MODEL_NAME = "gpt-4o-2024-11-20-FC"


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _score_ast(tool_calls, ground_truth, func_description, subset):
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    try:
        expected = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (json.JSONDecodeError, TypeError):
        return 0.0

    has_calls = bool(tool_calls)
    # An empty ground truth means "no call was the right answer".
    if not expected or expected in ({}, []):
        return 1.0 if not has_calls else 0.0

    # ast_checker wants [{"name": {...args}}], not [{"name":…, "arguments":…}].
    bfcl_output = []
    for call in tool_calls:
        name = call.get("name")
        if not name:
            continue
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        bfcl_output.append({name: args})

    language = Language[SUBSET_LANGUAGE_NAMES.get(subset, "PYTHON")]
    try:
        result = ast_checker(
            func_description=func_description or [],
            model_output=bfcl_output,
            possible_answer=expected,
            language=language,
            test_category=subset,
            model_name=AST_CHECKER_MODEL_NAME,
        )
    except Exception:
        # A checker crash on one malformed sample scores that sample zero; it
        # must not abandon the other 994.
        return 0.0
    return 1.0 if result.get("valid") else 0.0


def _read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _selftest() -> int:
    """Import what scoring imports, and nothing else.

    The AST checker is imported INSIDE `_score_ast`, so nothing at module
    scope touches `bfcl_eval` — `score.py --help` succeeds against a venv that
    cannot score at all. That is how a missing transitive dependency
    (`qwen_agent` -> `soundfile`) stayed invisible until after 995 responses
    had been generated: an hour and a half of inference, then an import error,
    and no gate record because the run ended Failed.

    Kept here rather than as a list of module names in the provisioner, so it
    cannot drift from what the scorer actually needs.
    """
    from bfcl_eval.constants.enums import Language  # noqa: F401
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: F401

    return 0


def main() -> int:
    # Handled before the parser is built, so `--dataset`/`--responses` keep
    # argparse's own `required=True` contract: a scoring invocation that omits
    # them must still fail exactly as it always has.
    if "--selftest" in sys.argv[1:]:
        return _selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--responses", required=True)
    args = ap.parse_args()

    dataset = {row["sample_id"]: row for row in _read_jsonl(args.dataset)}
    scores_by_subset = defaultdict(list)
    all_scores = []
    missing = 0

    for resp in _read_jsonl(args.responses):
        sample_id = resp.get("sample_id")
        row = dataset.get(sample_id)
        if row is None:
            missing += 1
            continue
        subset = row["subset"]
        if subset in HALLUCINATION_SUBSETS:
            # Correct behaviour is to NOT call a tool.
            score = 0.0 if resp.get("has_tool_calls") else 1.0
        else:
            try:
                func_description = json.loads(row.get("func_description") or "[]")
            except json.JSONDecodeError:
                func_description = []
            score = _score_ast(
                resp.get("tool_calls") or [],
                row.get("ground_truth", "[]"),
                func_description,
                subset,
            )
        scores_by_subset[subset].append(score)
        all_scores.append(score)

    if not all_scores:
        print("no responses matched the dataset", file=sys.stderr)
        return 2

    subset_results = {name: _mean(s) for name, s in scores_by_subset.items()}
    category_results = {}
    for category, subsets in CATEGORY_MAP.items():
        present = [s for s in subsets if s in subset_results]
        if not present:
            continue
        strategy = CATEGORY_AGGREGATION.get(category, "unweighted")
        if strategy == "sample_weighted":
            total = sum(len(scores_by_subset[s]) for s in present)
            category_results[category] = (
                sum(subset_results[s] * len(scores_by_subset[s]) for s in present) / total
            )
        elif strategy == "hierarchical":
            simple = [subset_results[s] for s in SIMPLE_AST_SUBSETS if s in subset_results]
            top = ([_mean(simple)] if simple else []) + [
                subset_results[s] for s in present if s not in SIMPLE_AST_SUBSETS
            ]
            if top:
                category_results[category] = _mean(top)
        else:
            category_results[category] = _mean([subset_results[s] for s in present])

    normalized = _mean(list(category_results.values())) if category_results else 0.0
    print(
        json.dumps(
            {
                "overall_accuracy": round(_mean(all_scores) * 100, 2),
                "normalized_single_turn_score": round(normalized * 100, 2),
                "category_scores": {k: round(v * 100, 2) for k, v in category_results.items()},
                "subset_scores": {k: round(v * 100, 2) for k, v in subset_results.items()},
                "total_samples": len(all_scores),
                "unmatched_responses": missing,
                # ADDITIVE, and the only reason this file changed. Every number
                # above is a MEAN, and means cannot be recombined across a
                # four-way shard split: `non_live` collapses the three simple_*
                # subsets into one term, `hallucination` weights its two subsets
                # equally regardless of size, and a subset absent from a shard
                # silently changes its category's divisor. Raw (hits, n) counts
                # recombine exactly -- `_score_ast` returns only 0.0 or 1.0, so
                # nothing is lost -- and the Rust aggregator applies the
                # weighting once, to the union. Nothing above is affected.
                "subset_totals": {
                    name: [int(sum(s)), len(s)] for name, s in scores_by_subset.items()
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
