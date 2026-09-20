# K3 serving integration: host reference restoration

This slice restores the missing server binding and CPU reference from historical
PR #1053, commit `940bd4eebf3a049a34fa0f48942a26dd7d5e88d9`, onto current main.
It reuses main's config parser, KDA/MLA/SiTU/AttnRes math, TP tensor plan, and
CUDA launch modules. The previously extracted device-cache APIs remain intact.
The restored binding still uses host projections and host state round trips;
this is a correctness reference, not the B300 performance endpoint.

## Charter and oracle

Explore whether the newly registered K3 loader binds the BF16/FP32 twin to the
same graph already validated in the historical campaign. Compare reference
logits and tokens against the frozen HF twin goldens; independently exercise
prefill/decode equivalence, prefix state, AttnRes ablation, expert routing, and
TP shard reconstruction. The tiny and synthetic checks do not establish full
checkpoint or GPU inference.

## Observed checks on the development Mac

- `AVAROK_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p avarok-core --no-default-features --lib kimi_k3`:
  **90 passed, 11 explicitly ignored**. Those eleven need the real twin.
- `AVAROK_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy -p avarok-core --no-default-features --tests -- -D warnings`:
  passed.
- Known-bad instrument check: explicitly run
  `c1_all_prompts_first_generated_token -- --ignored` without `K3_TWIN`:
  **exit 101**, naming the missing checkpoint. Restored tests previously
  returned successfully when it was absent; they now show as ignored normally
  and fail if deliberately invoked without their inputs.
- `cargo check -p spark-model --tests` on macOS stops in existing
  `spark-storage` Linux-specific `posix_fallocate`, `posix_fadvise`, and
  `O_DIRECT` references. This is not a successful model compile or a K3 defect
  classification; validate the model crate on Linux.

## Required next checks

On Linux, run the K3 model tests and then the eleven real checkpoint tests with
`K3_TWIN` naming the 0.40B fixture, using `--ignored --test-threads=1`.
A CPU reference pass is followed by rebuilt-server inference, first TP1 and
then TP2. Observe actual generated token IDs through EOS, kernel dispatch,
and rank-kill behavior. Do not infer serving from an HTTP models listing.

Packed MXFP4 TP is deliberately refused until implemented and tested. A
regression test supplies a packed tensor to TP8 and requires refusal before
binding missing tensors. Production K3 remains unvalidated by this slice.
