# DFlash Option-B Paged Propose — Handoff (2026-06-06)

## Root cause — CONFIRMED

`inferspark_prefill_paged_indirect` compiled with **HDIM=256** (the default in
`prefill_paged_compute.cuh:36`). The DFlash drafter has **head_dim=128**. This
causes all smem tiles to be double-wide: Q/K/V loads read 256 elements per head
instead of 128, the QK^T loop runs 16 inner iterations instead of 8, and K/V
smem rows spill into adjacent-head data → garbage attn_out every layer → degenerate
draft tokens (unconditional frequency prior).

The identical bug was already fixed for the **non-paged** path: `from_weights.rs:150-159`
comments explain it and the non-paged kernel was switched to `inferspark_prefill_h128`
(a standalone HDIM=128 kernel in `common/`). The **paged indirect** path was never fixed.

## Evidence trail (from probes run today)

- `DFLASH OPTION_B DIAG`: cache write bit-perfect (src==cached). Slot mapping: OK.
- `DFLASH ATTN PROBE`: `kv_len=16`, `q_offset=0` correct; `q_buf[row0]` matches
  legacy (minor BF16 rounding); **`attn_out[row0]` completely wrong** vs legacy.
- All Rust-side suspects cleared. Bug is in-kernel.
- Commit `7c656d5` already documents HDIM=256 as a P1 root cause for the non-paged path.
- `from_weights.rs:150-159` has the comment explaining the exact corruption.

## Fix — ALREADY WRITTEN (one file created)

**File created:**
```
kernels/gb10/qwen3.6-27b/nvfp4/inferspark_prefill_paged_indirect.cu
```
Content (3 meaningful lines):
```c
#define HDIM 128
#include "../../common/inferspark_prefill_paged_indirect.cu"
```

The `collect_cu_files` mechanism in `crates/atlas-kernels/build.rs` automatically
shadows the common kernel when a model-specific file with the same stem exists.
No changes needed to `from_weights.rs`, `KERNEL.toml`, or any Rust code.

## What Ronald must do next

### 1. Delete stale PTX and touch files to force recompile
```bash
rm /home/rstesiak/code/atlas/target/release/build/atlas-kernels-*/out/t0__inferspark_prefill_paged_indirect.ptx
touch /home/rstesiak/code/atlas/kernels/gb10/qwen3.6-27b/nvfp4/inferspark_prefill_paged_indirect.cu
touch /home/rstesiak/code/atlas/crates/atlas-kernels/build.rs
```

### 2. Build
```bash
ATLAS_TARGET_MODEL=qwen3.6-27b cargo build --release -p spark-server
```

### 3. Verify the new PTX was actually compiled (CRITICAL — see NaN-fix pitfall)
```bash
# Should show the new file with today's timestamp:
ls -la /home/rstesiak/code/atlas/target/release/build/atlas-kernels-*/out/t0__inferspark_prefill_paged_indirect.ptx

# Confirm HDIM=128 is baked in — with HDIM=128, inner QK^T loop runs 8 times.
# Grep for the loop count signature (8 mma instructions in a row vs 16):
grep -c "mma.sync" /home/rstesiak/code/atlas/target/release/build/atlas-kernels-*/out/t0__inferspark_prefill_paged_indirect.ptx
# HDIM=128 → fewer mma blocks than HDIM=256. Compare against the inferspark_prefill_h128.ptx count.
```

### 4. Run and check for row-0 echo
```bash
ATLAS_DFLASH_OPTION_B_NO_CTX=1 bash ~/launch-dflash-gamma-nograph.sh
grep "TRACE drafts\|accepted=" ~/dflash_gamma_nograph_*.log | head -20
```

**Expected:** `drafts_pre_cap[0]` should now echo `token_in` (row-0 echo restored).
Accept should climb from ~0% toward the >50% seen on the non-batched K2 path.

### 5. If row-0 echoes, remove NO_CTX and run full context path
```bash
bash ~/launch-dflash-gamma-nograph.sh
grep "accepted=" ~/dflash_gamma_nograph_*.log | head -20
```

## State of uncommitted work

The following files are intentionally modified (do not discard):
- `crates/spark-model/src/layers/dflash_head/forward_block_layer_paged.rs` — ATTN PROBE added (can be removed after fix confirmed)
- `crates/spark-model/src/model/trait_impl/verify_d.rs` — per-layer hidden capture (keep)
- `crates/spark-server/src/scheduler/verify_dflash_step.rs` — verify trace (keep)
- `crates/spark-model/src/layers/ops/ssm_gdn_b.rs` + `trait_decode_batched_conv_gdn.rs` — GDN DUMP (harmless, env-gated, leave)

## After accept is healthy

The batched verify kernel (`wy16`/`wy17`) is a SEPARATE concern. Fix propose first,
then re-evaluate verify accept. See `dflash_gamma_on_ssm.md` for verify context.
