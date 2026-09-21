# Optional K3 resident GPU serving

This slice builds on the serving foundation. It preserves per-sequence KDA
conv/recurrent buffers and MLA KV between tokens, downloads authoritative
state when snapshotting or switching to the explicit CPU path, invalidates
device state on restore, and frees owned allocations on sequence release.
CUDA MLA requires `serve_max_seq_len` (`--max-seq-len`) to bound the device
KV; it appends one row per token instead of re-uploading `[0..T]`. Partial
allocation and upload failures clean up allocated buffers; a later-layer
failure also removes the in-progress AttnRes token stream.

`K3_CUDA_DENSE=1` selects the resident FP32/BF16 weight path for dense and shared
MLPs. Gate/up plus SiTU runs on device before the down projection. Matrix
weights are reused from the existing binding; weights are not duplicated.
FP32 activations still cross the host boundary. Unset or `0` retains host dense
execution; other values fail explicitly. This does not move every projection,
router, AttnRes or the full model graph onto GPU.

## Prerequisites

The review base is `extract/k3-serving-foundation`. The B200 K3 target #1177
provides `dense_f32io` kernels for this opt-in path; this Rust slice does not
silently add a target or enable dense execution elsewhere. The foundation's
separate communication, capacity, shutdown and tokenizer prerequisites remain.
Expert gate/up batching #1163 remains independent and is not duplicated here.

## Oracles and limits

The four real-CUDA oracle binaries are intentionally ignored by default and
require an explicitly selected idle device, matching compiled K3 target and an
external process timeout:

- `k3_dense_cuda_oracle`: FP32/BF16 dense/shared MLP against independent f64.
- `k3_mixers_cuda_oracle`: KDA/MLA callbacks against host references.
- `k3_mxfp4_cuda_oracle`: E8M0 grouped expert GEMM against independent scalar
  arithmetic, plus selected-expert pipeline against separate projections.
- `k3_tp4_composed_cuda`: synthetic official **rank-local TP4 dimensions**
  through two composed layers, reset and history checks. One process/rank,
  selected tensors, no NCCL, checkpoint loading or full-model quality claim.

For example, build with `AVAROK_TARGET_HW=b200 AVAROK_TARGET_MODEL=kimi-k3
AVAROK_TARGET_QUANT=mxfp4`, then run the desired test binary with
`K3_ORACLE_GPU_ORDINAL=0` under a 180-second external deadline and
`--ignored --nocapture --test-threads=1`. Use the same target settings for build
and test. Do not run all heavy tests concurrently on a shared device.

The source is extracted from #1150's tested development path, but these new
stack heads have no GPU certification. A Metal-feature model-library compile
check passes locally; that checks Rust interfaces, not CUDA behavior. Prior
B200 twin lifecycle and oracle receipts do not validate a reconstructed tree.
Run its own Linux/CUDA tests and bounded multi-rank lifecycle before landing.
Full official weights, B300 execution, TP8 and multi-host remain unvalidated.
