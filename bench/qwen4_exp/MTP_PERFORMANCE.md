# EXL3 staging and concurrency follow-up

This follows PR 834's wider-draft numerical fixes. Validation is in progress;
the final measured profile and results will be recorded before publication.

## Bounded optimization

`542886a49` combines routing staging and activation replication into one CUDA
launch. It preserves local/remote expert IDs, FP16 rounding, token/slot order,
and the existing expert GEMM launch and reduction geometry. The original two
launches remain available with `ATLAS_NO_EXL3_FUSED_INGRESS=1` for a same-binary
comparison.

The GPU parity leg checks all three staging buffers against the original
production wrappers and an independent conversion/index oracle, with guard
bytes around the output allocations. It covers rows 1–4, nonuniform routing
probabilities, remote expert slots, conversion edges and actual model shapes.
The existing serial/batched expert fixture is also exercised with both ingress
paths. Removing one launch alone is not evidence of a serving speedup.

## Concurrency boundaries under review

Qwen's hyperconnection verification requires its specialized per-sequence
forward/commit path. The generic multi-sequence verifier cannot be assumed to
preserve that layout. Live concurrency validation must also cover draft KV
block reuse, draft QSA cleanup, abandoned proposals, and diagnostic shadow
state ownership. Concurrency support and batched-forward acceleration are
separate claims.

## Benchmark profile

The agentic sanity profile explicitly sets both
`ATLAS_AGENTIC_SAMPLING=model-card` and `ATLAS_AGENTIC_PRESERVE_THINKING=1`,
with served `reasoning_effort=low`. Preserving previous-turn reasoning is
independent of `ATLAS_DFLASH_SPEC_THINK=1`, which enables speculation during
the current thinking span. Earlier agentic records without preserve-thinking
remain compatibility observations, not the intended workload baseline.

Performance comparisons use one binary, identical warmup and requests, and
three repetitions. Isolated CUDA-event timing is reported separately from
server-attested token timing and request wall time. Two-request tests start
with bounded context and record memory throughout; wider concurrency is not
inferred from a two-request pass.
