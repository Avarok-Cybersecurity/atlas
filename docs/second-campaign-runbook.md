# The second campaign: the tree that merges into `main`

Written 2026-08-28, while the first campaign (#754 into
`feat/longcat-ngram-readiness`) was still running, so the traps below are the
ones that actually cost time rather than the ones that sound plausible.

Companion to `docs/gate-queue-protocol.md`, which covers the serialized-landing
protocol itself; this covers the campaign that has to happen before it.

## Why a second campaign at all

Records are content-anchored over PERF_PATHS, not ancestry-anchored. The
campaign that just ran covers **#754's tree** (`1d30d5d5ff`). Merging
`feat/longcat-ngram-readiness` into `main` produces a *different* tree, because
`main` is **25 commits ahead** and two of them touch perf paths:

    6cd39ba02  feat(cli): spark sync-recipes            11 perf files
    8903c4659  stack: serve-options snapshot, RST ...  299 perf files

That interaction is unmeasured, and the gate refuses it by design. Verified
2026-08-28 with a DEEPENED clone — a shallow one reports the branch as 1 commit
from main with a site-only delta, which is wrong and would have skipped this
campaign entirely.

## Order (do not reorder)

1. **#754 lands** into `feat/longcat-ngram-readiness`. Until then this tree does
   not exist.
2. **Freeze**: merge current `main` into `feat/longcat-ngram-readiness` and push.
   Any later movement on `main` restarts the protocol — see
   `scripts/queue-perf-pr.sh`, which does exactly this step.
3. **Campaign** against that exact sha, on a box with the GPU to itself.
4. **Undraft #746** and queue it with no other perf PR ahead of it.

## The run

    export ATLAS_HOME=${ATLAS_HOME:-$HOME/.atlas}   # MUST be writable, or every
    # gate dies instantly with "recipe ... is not in the local index (0 cached)"
    # and nothing says why.

    cd <worktree at the frozen sha>
    export PATH=/usr/local/cuda/bin:$PATH CUDA_ROOT=/usr/local/cuda

    for g in decode-floor vision-fidelity video-fidelity \
             ssm-state-poisoning-gate concurrency-sweep agentic-webserver \
             ttft-cold-gate ttft-cold-gate ttft-warm-gate ttft-warm-gate \
             bfcl-subset bfcl-subset-echolp; do
      timeout 21600 ./target/release/spark benchmark run "$g" --pull-request-gate --yes
      rc=$?          # capture IMMEDIATELY: a $(date) on the next line resets it
      echo "$g rc=$rc"
    done

Cheap gates first, so a config mistake surfaces in minutes rather than after the
1.6h BFCL leg. TTFT gates listed **twice** each: on a box with no stored
baseline the first run only creates it and records `info`, which the gate does
not accept — that is how a ten-gate campaign silently comes back eight.

## Judge by the record, never by rc

    ./target/release/spark benchmark --pull-request-gate-check --pr 746

`rc` is not the evidence. On 2026-08-28 a BFCL run generated all 995 responses,
died in scoring, correctly wrote **no** record — and exited 0. Separately, the
gate-check command itself exits 0 while reporting `NONE` for a missing bench.
Read the words: refuse on `still need`, `NONE`, or `FAIL`.

## Before the BFCL legs, check the scorer imports

The failure that cost 1.6h was a transitive `soundfile` under `qwen_agent`,
imported lazily during scoring — so importing `bfcl_eval` alone does not prove
it. Run the two imports `score.py` actually performs:

    $ATLAS_HOME/artifacts/bfcl/venv/bin/python -c "
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker"

(#777 adds `score.py --selftest` and runs it at provision time, which makes this
step automatic — it is held behind its own benchmark gate.)

## Landing the records

A landing script is the pattern: it refuses unless the gate-check text
is clean, refuses if anything outside `.benchmarks/` is dirty, and never merges.
Point a copy at `--pr 746` and the `feat/longcat-ngram-readiness` branch.
