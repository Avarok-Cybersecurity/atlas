# Resident KDA state development rehearsal

The serving path previously copied the complete KDA convolution history and
recurrent matrix between CPU and GPU for every token. The existing CUDA resident
launch now owns those buffers for the lifetime of a sequence. Projections remain
on the host; the kernel and floating-point arithmetic are unchanged.

`K3CpuFallbackState` owns the device buffers. Prefix snapshots download the
current device state into the existing portable cache encoding; restoring a
snapshot releases the old buffers and seeds the next CUDA token from that
snapshot. The explicit CPU escape downloads current state before continuing.
The existing `release_state` hook frees allocations at sequence retirement,
including cancellation. Partial allocation/upload failure releases allocations.

Targeted tests cover authoritative snapshots, restore and reuse, malformed
snapshot refusal without discarding current state, idempotent release, allocation
failure cleanup, and the BoundLayer dispatch paths. The independent CUDA mixer
oracle covers nonzero recurrent history and restored continuation against the
CPU implementation.

This is development validation on the packed 0.40B twin. Full K3 memory,
B300 execution, TP8 collectives, and multi-host execution remain unvalidated.

The paired LatentMoE change groups selected experts' gate and up projections into
one existing E8M0 GEMM launch. Each row still selects the same weight pointer and
uses the same dot-product kernel. A real GPU oracle compares selected expert
orders `[3]`, `[2,0]`, and `[3,1,0]` against the previous separate-launch pipeline;
the production-shape GEMM CPU oracle remains independent.

The resident-only intermediate binary is retained separately from the earlier
CPU-GEMV baseline and the combined binary. Initial resident timings overlapped a
compiler and are not used for performance conclusions. The final comparison
rejects compiler overlap and samples compiler/linker processes every 0.5 seconds.
It uses the same two prompts, 128 actual generated tokens, one warmup and three
measured requests per prompt, with identical cache and diagnostics settings.
Residency alone has not demonstrated a tiny-model latency improvement; the
reason for it is to avoid transferring the full recurrent matrix at production width every token.

| TP | Previous CPU-GEMV binary, short / long tok/s | Resident KDA only | Resident KDA + batched gate/up |
|---|---:|---:|---:|
| 1 | 41.45 / 30.58 | 40.56 / 30.44 | 43.85 / 32.24 |
| 2 | 41.90 / 31.94 | 43.07 / 32.03 | 44.98 / 33.45 |

The combined change improves these median decode measurements by approximately
4.7–7.3%. These are three-repeat development measurements without locked clocks;
they do not predict full K3 throughput. All 48 responses agree exactly in text,
finish reason, prompt count and generated count. Every accepted timing session
has zero sampled compiler overlap. The first combined TP2 attempt was rejected
when a compiler appeared and was repeated after compilation finished.

Both resident-only and combined binaries passed the 121-check, five-cycle TP2
lifecycle suite after 45 seconds idle: plain/streaming parity, repeat requests,
cancellation recovery and two concurrent clients. All owned processes exited
zero; the complete collective logs agree at the expected 118,639 submissions
per rank. This does not prove immediate GPU cancellation or absence of leaks.
Compact receipts are [transfer-comparison.json](transfer-comparison.json) and
[resident-lifecycle.json](resident-lifecycle.json); bulk logs remain outside Git.

These timing/lifecycle results identify the pre-cleanup-fix binaries by SHA256.
A subsequent failure-path correction moves resident preparation inside existing
AttnRes error cleanup, so an allocation failure on a later layer cannot retain
that token's host stream. Its regression test seeds an earlier layer, forces the
next allocation to fail, and checks that the stream is removed. The separately
committed dense CUDA primitive is compiled but not connected to serving in this
change; no measurements above claim a dense GPU projection speedup.

The final failure-path build passed its targeted regression, release clippy
(`spark-model --lib --tests`, including the dense primitive), and a real TP2
canary with all 16 generated tokens, finish reason and counts exactly matching
the retained baseline. All owned processes exited zero. Its binary SHA256 is
`89ec58530084fe58828702e5a37d413e1368b63f3a300e61ba4254b660acf29b`;
see [resident-final-canary.json](resident-final-canary.json). The earlier
runtime slice passed 54 K3 unit tests and both real GPU MoE oracles; the final
cleanup regression adds one targeted check. Formatting and diff checks pass.
