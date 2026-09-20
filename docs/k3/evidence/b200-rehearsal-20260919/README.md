# B200 K3 bounded rehearsal — September 19, 2026

Four B200 GPUs on an x86 host, with NV18 topology between every pair.
CUDA 13.0.48 compiled all 182 selected `b200/kimi-k3/mxfp4` kernels.
The packed 0.40B derivative was regenerated from the pinned source; both source
and packed weight hashes matched the Spark artifacts. See `results.json`.

## Measured

- Production TP8-shape KDA and MLA GPU numerical oracles passed, including
  nonzero history, restored continuation and negative controls.
- E8M0 w1/w2/w3 and irregular GEMM cases matched independent CPU arithmetic
  exactly after BF16 rounding.
- TP1, TP2 and TP4 each generated the same complete 16-token aviation canary
  as the Spark reference, including whitespace, finish reason and token counts.
- Every owned rank exited zero and the GPUs were released afterward.
- CPU registration/inherited-target tests: 17 passed. Packed loader-to-binding
  test passed for TP2, TP4 and TP8. No second packed weight allocation occurred.

## Limits and next work

This was one generation canary per topology, not the full Spark soak suite,
benchmark certification, full K3 inference or a B300 execution receipt.
Production-shape kernel fixtures do not prove full-model memory fit or TP8
communication. The ~57–60-second initial canary runs include substantial CPU
JIT/startup; these are not throughput measurements.

The full-log collective checker **failed** at teardown: rank zero recorded one
extra final four-byte broadcast for TP2 and TP4. NCCL also logged
`driver shutting down` during simultaneous process cleanup. Generation and
zero exits passed, but clean collective shutdown is not established. Investigate
head-first graceful shutdown before claiming full submission agreement; do not
trim the tail and call the original full-log check successful. The pre-existing
eight small ownerless scratch allocations were reclaimed by backend teardown.

The tested source began at `32ebc80d5` plus this B200 target/test change.
After execution, E8M0 was isolated as a byte-identical B200-owned source to
remove an indirect cross-hardware symlink; MODEL.toml comments were clarified.
No CUDA kernel body or runtime behavior changed during that cleanup. The final
rebuild and 17 target tests passed, but its binary hash changed; both hashes are
recorded. The GPU receipts above name the earlier binary. A short final-artifact
canary remains necessary alongside the shutdown-harness fix.
Bulk rank logs remain outside the repository; compact receipts include their
probe hashes. This is development evidence, not a `.benchmarks` record.
