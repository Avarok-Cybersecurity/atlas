# B200 K3 host projection improvement

The four-row CPU GEMV change improves this small packed K3 fixture while retaining
the reduction order of every output row. LatentMoE now uses the same helper instead
of a duplicate scalar implementation. CUDA kernel bodies and numeric tolerances
are unchanged. This does not move full-model projections onto the GPU.

## Controlled generation comparison

Each topology used the same two prompts, temperature zero, 128-token budget,
prefix caching off and 32-token prefill chunks. Each case had one warmup plus
three measured repeats. Every request actually generated 128 tokens. All 48
responses matched byte-for-byte, including finish reason and token counts,
across versions and TP1/2/4. These are development measurements, not certification.

| GPUs | Short prompt tok/s before → after | Longer prompt tok/s before → after |
| --- | --- | --- |
| 1 | 29.40 → 38.74 | 22.11 → 29.75 |
| 2 | 32.31 → 41.50 | 24.30 → 31.35 |
| 4 | 34.57 → 42.03 | 27.54 → 31.97 |

The short/long prompts tokenize to 8/65 tokens. Median decode improves by
16–35%; TTFT improves by 26–39%. Per-request values and receipt/binary hashes
are in `performance-comparison.json`. Collective diagnostic logging was off
equally for all timing runs. The earlier lifecycle run retained diagnostics.

The four GPUs are on one NVLink-connected host, not four network nodes. Four
GPUs help this tiny model only modestly: after the change, TP1 already reaches
38.74 tok/s versus TP4 42.03 tok/s on the short prompt. The host projection
improvement explains more of the gain than adding GPUs. Host/GPU round trips,
scalar/reference work elsewhere and communication remain. This is not a
prediction of full K3 throughput or a substitute for B300/TP8 validation.

Generation-phase sampled GPU memory was roughly 2.8–4.1 GiB per GPU; this is
not startup peak memory. The first TP1 baseline boot lacked the newly explicit
CUDA cache path; later launches used it. Do not compare total session duration
as an isolated startup or decoding benchmark. Warm request timing is reported
separately. Clocks were not locked and these are three-repeat development runs.

## Validation and limitation

- The new bit-exact scalar-reference test covers grouped rows, tails, empty
  dimensions and cancellation-prone input magnitudes. Both new tests pass.
- Release clippy passes for `avarok-core --lib --tests`.
- The broader K3 core suite reports 104 passed, 11 ignored and one existing
  optimized-x86 SiTU test failure: `situ_glu_matches_closed_form_vector` compares
  -9.068051 with -9.068053 at absolute tolerance 1e-6. Restoring both original
  GEMV source files and rerunning the same Cargo test reproduces the failure.
  No tolerance or threshold was relaxed; this remains a separate portability
  issue. A standalone compilation of the unchanged SiTU test passes, so codegen
  context matters and the Cargo reproduction should be retained.
- The final candidate also passed all 121 TP2 lifecycle checks across five
  cycles in 141.64 seconds, after 45 seconds idle. Full untrimmed rank logs match
  at 118,639 submissions per rank, and all owned processes exit zero. This adds
  streaming/cancellation/concurrency recovery to the 24 long-generation checks.
  See `final-lifecycle-tools.json`. Production-shape GPU numerical oracles
  preceded the CPU-only change; no CUDA kernel body changed.
