# D0 handoff — 2026-09-05

Step C is complete as a CPU-only assembly investigation. The [report](../../GB10_PREFILL_SHARED_MEMORY.md) and [receipt](../../receipts/gb10-prefill-shared-memory-2026-09-05.json) preserve their original source revision: upstream `567b5ebe7784ac3657a4ae97940f6783c8414393`. Rebasing the write-up onto `b2c17cf` does not turn those measurements into measurements of that newer tree. No driver-JIT or Hopper/B200 execution was observed.

| Step | Status | Evidence and limits |
|---|---|---|
| A2: Nano FP8-KV three-way diagnosis | Not started in the available execution record | No labelled bf16-KV, calibrated-FP8-KV, and NVFP4 diagnosis artifacts were found in the searched evidence roots. The earlier Atlas coherency failure is an initial rehearsal result, not this diagnosis. |
| B: perf-gate records at code tip | Not started | [Historical status](evidence/d0-handoff/step-b.status.json) records no build, kernel audit or gate run, with target ambiguity and an unowned Spark 2 workload as the recorded blockers. No new `.benchmarks` records exist in this contribution. |
| C: static shared-memory investigation | Done | 151/173 files assembled; 22 rejected modules contain 42 rejected entries. These are CPU compiler/assembler observations, not GPU correctness results. |

The lead subsequently clarified the current-tip target and tree-based gate coverage. Historical statements about the old target are retained as evidence of the prior stop, not as present blockers. Step D uses `b2c17cf` or the newer code tip before publishing.

The initial 40GB storage stop preceded the successful rehearsal after the cap became 70GB. It does not establish an A2-specific storage failure. [Status and provenance](evidence/d0-handoff/status.json) list the searched roots and hashes of the historical B evidence copied into this PR.

Oracle: count A2 complete only with all three diagnostic outputs, and B complete only with newly measured exact-tip gate records. Stopping rule: publish this handoff and the existing Step C evidence; do not create A2/B measurements during D0.
