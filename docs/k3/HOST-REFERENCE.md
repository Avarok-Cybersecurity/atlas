# K3 host reference graph

This extraction provides checkpoint-to-host binding and the composed K3 CPU
forward path for the existing KDA, MLA, AttnRes and latent-MoE primitives.
It is a reference and bring-up foundation, not a GPU serving implementation.

The source comes from integration PR #1150, including the latest callback
boundaries and row-interleaved matrix-vector loops. Tensor-parallel checkpoint
allocation, device binding, serving dispatch and resident GPU callbacks are
separate changes. Existing main primitives remain unchanged.

Run without a CUDA device:

```sh
AVAROK_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p avarok-core --lib kimi_k3 --no-default-features
```

Extraction validation: 94 passed, 11 ignored. The ignored cases require the
small real checkpoint via `K3_TWIN`; they were not executed here. Synthetic
checks include greedy determinism, state-history/reset sensitivity, ablations
that deliberately break attention or routing, host callback equivalence,
packed-format admission, and simulated TP2 reductions. Simulation does not
prove NCCL or distributed execution.

The checked-in 0.40B golden token fixture is historical independent reference
evidence. No real-checkpoint inference, GPU measurement or certification was
rerun for this extraction. Clippy for the core crate and formatting passed.
Remaining product integration and GPU fallbacks are tracked in #1167-#1170.
