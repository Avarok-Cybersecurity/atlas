# SPDX-License-Identifier: AGPL-3.0-only
"""Generate parity_fixtures.json from the UPSTREAM MLPerf inline scorer.

The Rust port in `benchmarks/mlperf_agentic/scoring.rs` re-implements
`AgenticInferenceInlineScorer` (mlcommons/endpoints, commit 7935df4,
`src/inference_endpoint/evaluation/scoring.py`). A port can diverge silently —
one regex flag, one alias entry — and a diverged scorer produces numbers that
look like MLPerf inline accuracy and are not. So the expected values in the
fixture file are never typed by hand: this script EXECUTES the upstream class
(its source sliced out verbatim, no retyping) over the case list below and
records what it returns. The Rust tests then assert the port agrees on every
case, including one generated case per alias-table entry and per shell wrapper,
so the whole ~60-entry table is pinned rather than spot-checked.

Not run at build or provision time. Re-run only when deliberately tracking an
upstream scorer change, then commit the regenerated JSON with the new commit id:

    python3 gen_parity_fixtures.py \
        --scoring-py <endpoints-checkout>/src/inference_endpoint/evaluation/scoring.py \
        --upstream-commit <sha> \
        --out parity_fixtures.json
"""

import argparse
import json
import re  # noqa: F401 — used by the exec'd upstream source
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar  # noqa: F401 — used by the exec'd upstream source


def load_upstream(scoring_py: Path):
    """Exec the upstream class body verbatim inside a stub-based shim.

    Only the pure comparison members are exercised (`_model_intent`,
    `_ground_truth_intents`, `_bash_actions`, the class-level regexes and
    tables); the heavyweight members (events.jsonl IO, pandas) are defined but
    never called, so their imports can be stubbed.
    """
    src = scoring_py.read_text()
    start = src.index("class AgenticInferenceInlineScorer")
    end = src.index("class ", src.index("\n", start + 1))
    while not src[end:].startswith("class LiveCodeBenchScorer"):
        end = src.index("class ", end + 1)
    block = src[start:end]
    header_end = block.index("\n")
    block = "class Upstream:" + block[header_end:]

    import os  # noqa: F401 — annotation on upstream __init__

    namespace = {
        "re": re,
        "json": json,
        "Counter": Counter,
        "ClassVar": ClassVar,
        "Any": Any,
        "os": __import__("os"),
        # Stubs for names that appear only in annotations / uncalled bodies.
        "Scorer": object,
        "Dataset": object,
        "Extractor": object,
        "AgenticInferenceDataset": object,
        "defaultdict": dict,
        "logger": None,
    }
    exec(compile(block, str(scoring_py), "exec"), namespace)  # noqa: S102
    cls = namespace["Upstream"]
    return cls, cls.__new__(cls)


# ── Case lists ───────────────────────────────────────────────────────────────
# Each case is an assistant-turn dict exactly as the scorer receives it. The
# comments record which upstream behaviour the case pins, including the
# ambiguous corners a porter would plausibly get wrong.

INTENT_CASES = [
    # The documented happy path.
    {"content": "intent: I042"},
    # Explicit form is preferred, reasoning_content searched before content.
    {"reasoning_content": "intent: I001", "content": "intent: I002"},
    # ★ The explicit pass runs over BOTH fields before the bare fallback runs
    # over either: an explicit match in content beats a bare token in
    # reasoning_content, even though reasoning_content is listed first.
    {"reasoning_content": "thinking about I111", "content": "final intent: I222"},
    # Explicit regex is IGNORECASE and the result is upper-cased.
    {"content": "Intent: i042"},
    # \s* between the colon and the code; Python \s is unicode-aware, so
    # NBSP (U+00A0) counts as whitespace here.
    {"content": "intent: I042"},
    # ★ No space is allowed before the colon and \b guards "intent", so the
    # EXPLICIT pattern fails on both of these — but the bare fallback still
    # finds the I042 token, so they score anyway. A port that only implements
    # the explicit pattern would return None here and diverge.
    {"content": "intent : I042"},
    {"content": "reintent: I042"},
    # \b after the third digit: a fourth digit kills the explicit match, AND
    # the bare fallback ("I0425" has no boundary after digit three either).
    {"content": "intent: I0425"},
    # Bare fallback: LAST bare token wins.
    {"content": "maybe I100, no, I200"},
    # ★ Bare fallback is case-SENSITIVE (no IGNORECASE on _BARE_INTENT_RE):
    # a lowercase bare code matches nothing.
    {"content": "maybe i123"},
    # \b before bare I: "AI123" is one word, no match.
    {"content": "the AI123 model"},
    # Bare needs exactly three digits with a boundary after.
    {"content": "I12 and I1234"},
    # Missing/None fields fall through without error.
    {"content": None},
    {},
    # Non-string content is skipped, reasoning still searched.
    {"content": 7, "reasoning_content": "intent: I300"},
]

GT_INTENT_CASES = [
    # Docstring example: lowercase upper-cased, None dropped.
    {"intent_codes": ["i001", "I002", None]},
    # Empty string dropped; non-list is no ground truth at all.
    {"intent_codes": ["", "I003"]},
    {"intent_codes": "I001"},
    {"intent_codes": None},
    {},
]

BASH_CASES = [
    # Upstream docstring example: env assignment + abs path + version alias.
    {"cmd": "CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 -m pytest"},
    # Pipes split stages; every stage's executable is normalized.
    {"cmd": "cat foo.py | grep bar | wc -l"},
    # `||` splits once (alternation order: \|\| before \|), `;` and newlines too.
    {"cmd": "make build || make clean; ls\nfind . -name x"},
    # ★ Executables NOT in the alias table vanish from the multiset entirely —
    # `echo` contributes nothing, on either side of the IoU.
    {"cmd": 'echo "done" && grep -r pattern src'},
    # Quoted spans are stripped BEFORE splitting: the | inside quotes is not a
    # separator, and quoted text cannot smuggle in an executable.
    {"cmd": 'grep "a | b" file.txt'},
    # Double-quote escapes: the escaped quote does not close the span.
    {"cmd": 'grep "say \\"hi\\" | there" f && ls'},
    # Backtick spans are stripped too.
    {"cmd": "diff `ls` other"},
    # Unterminated quote: no span matches, the quote char stays in the text.
    {"cmd": "grep 'unterminated && ls"},
    # Wrapper + env-assignment stripping is iterative and order-free.
    {"cmd": "sudo env FOO=1 time make test"},
    # An env-assignment-only stage contributes nothing.
    {"cmd": "FOO=1"},
    # `.` aliases to source; `./script.sh` basenames to script.sh → dropped.
    {"cmd": ". ./venv/bin/activate && ./run.sh"},
    # Version-suffix stripping: one trailing .N group...
    {"cmd": "python3.11 setup.py"},
    # ...and the regex takes at most two trailing groups in one match.
    {"cmd": "python3.1.2.3 x"},
    # ★ A single `&` is NOT a separator upstream (the regex has no bare &):
    # "sleep 5 &" keeps `&` as a token and `sleep` is not in the alias table.
    {"cmd": "sleep 5 & wait"},
    # Uppercase executables are lowercased before the table lookup.
    {"cmd": "GREP pattern file"},
    # "command" key preferred; empty string falls through to "cmd".
    {"command": "git status"},
    {"command": "", "cmd": "git diff"},
    # Repeats are preserved — this is a multiset, not a set.
    {"cmd": "grep a f1; grep b f2; grep c f3"},
]

# (gt_actions_cmd, model_turn) pairs scored end to end, IoU semantics included.
TURN_CASES = [
    # Multiset IoU: gt {python, pytest}, model {python} → 1/2.
    {
        "domain": "coding",
        "gt": {"tool_calls": [{"function": {"name": "bash", "arguments": {"cmd": "python x.py && pytest"}}}]},
        "model": {"tool_calls": [{"function": {"name": "bash", "arguments": {"cmd": "python y.py"}}}]},
    },
    # Duplicates count: gt {grep×2}, model {grep×1} → 1/2.
    {
        "domain": "coding",
        "gt": {"tool_calls": [{"function": {"name": "bash", "arguments": {"cmd": "grep a; grep b"}}}]},
        "model": {"tool_calls": [{"function": {"name": "bash", "arguments": {"cmd": "grep a"}}}]},
    },
    # Model produced no bash calls at all → 0, denominator intact.
    {
        "domain": "coding",
        "gt": {"tool_calls": [{"function": {"name": "bash", "arguments": {"cmd": "make"}}}]},
        "model": {"content": "I would run make here."},
    },
    # Non-bash tool calls are invisible to the scorer.
    {
        "domain": "coding",
        "gt": {"tool_calls": [{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}]},
        "model": {"tool_calls": [{"function": {"name": "edit_file", "arguments": {"cmd": "ls"}}}]},
    },
    # Arguments arrive as a JSON STRING over the wire; malformed JSON in one
    # call skips that call, not the turn.
    {
        "domain": "coding",
        "gt": {"tool_calls": [{"function": {"name": "bash", "arguments": '{"cmd": "git log"}'}}]},
        "model": {
            "tool_calls": [
                {"function": {"name": "bash", "arguments": "{not json"}},
                {"function": {"name": "bash", "arguments": '{"cmd": "git show"}'}},
            ]
        },
    },
    # Workflow: binary membership, gt codes upper-cased.
    {"domain": "workflow", "gt": {"intent_codes": ["i042"]}, "model": {"content": "intent: I042"}},
    {"domain": "workflow", "gt": {"intent_codes": ["I001", "I002"]}, "model": {"content": "intent: I003"}},
    # Workflow with no extractable intent → None ∉ codes → 0.
    {"domain": "workflow", "gt": {"intent_codes": ["I001"]}, "model": {"content": "no code here"}},
]

DOMAIN_CASES = [
    "sim_001",
    "sim_12345",
    "sim_",          # \d+ requires at least one digit
    "sim_1x",        # $ anchor: trailing junk is coding
    "Sim_001",       # case-sensitive
    "xsim_001",      # ^ anchor
    "django__django-12345",
    "task_991",
]


def coding_turn_score(obj, gt_turn, model_turn):
    """The exact IoU expression from upstream score()."""
    gt_counts = Counter(obj._bash_actions(gt_turn))
    model_counts = Counter(obj._bash_actions(model_turn))
    union = sum((gt_counts | model_counts).values())
    return sum((gt_counts & model_counts).values()) / union


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring-py", required=True, type=Path)
    ap.add_argument("--upstream-commit", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cls, obj = load_upstream(args.scoring_py)

    def bash_turn(case):
        return {"tool_calls": [{"function": {"name": "bash", "arguments": dict(case)}}]}

    out = {
        "upstream_commit": args.upstream_commit,
        "upstream_file": "src/inference_endpoint/evaluation/scoring.py",
        "upstream_class": "AgenticInferenceInlineScorer",
        "intent_cases": [
            {"turn": t, "expected": obj._model_intent(t)} for t in INTENT_CASES
        ],
        "gt_intent_cases": [
            {"turn": t, "expected": sorted(obj._ground_truth_intents(t))}
            for t in GT_INTENT_CASES
        ],
        "bash_cases": [
            {"arguments": c, "expected": obj._bash_actions(bash_turn(c))}
            for c in BASH_CASES
        ],
        # One case per alias entry and per wrapper: the WHOLE table is pinned,
        # generated from the upstream dict itself so nothing is retyped.
        "alias_cases": [
            {"arguments": {"cmd": f"{key} --arg"}, "expected": obj._bash_actions(bash_turn({"cmd": f"{key} --arg"}))}
            for key in cls._EXECUTABLE_ALIASES
        ],
        "wrapper_cases": [
            {"arguments": {"cmd": f"{w} ls -la"}, "expected": obj._bash_actions(bash_turn({"cmd": f"{w} ls -la"}))}
            for w in sorted(cls._SHELL_WRAPPERS)
        ],
        "turn_cases": [
            {
                **case,
                "expected": (
                    (1.0 if obj._model_intent(case["model"]) in obj._ground_truth_intents(case["gt"]) else 0.0)
                    if case["domain"] == "workflow"
                    else coding_turn_score(obj, case["gt"], case["model"])
                ),
            }
            for case in TURN_CASES
        ],
        "domain_cases": [
            {"conversation_id": cid, "workflow": bool(cls._WORKFLOW_CONVERSATION_RE.match(cid))}
            for cid in DOMAIN_CASES
        ],
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out} ({len(out['alias_cases'])} alias cases)")


if __name__ == "__main__":
    main()
