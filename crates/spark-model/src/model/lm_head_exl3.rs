// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 (QTIP trellis) LM head — `ATLAS_EXL3_NATIVE=1`.
//!
//! Serves the vocab projection straight from the checkpoint's packed trellis
//! codes through the fused `exl3_matmul` kernels instead of materializing a
//! BF16 `lm_head.weight` (~1.27 GB on Qwen3.8-Flash-Next vs ~325 MB packed at
//! K=4, and no double-quantization loss on the projection that picks tokens).
//!
//! Data path per call (all stream-ordered, no host syncs, no allocations):
//!
//! ```text
//!   normed BF16 [rows, H] --exl3_bf16_to_f16--> A fp16 (per-row scratch slab)
//!   A --cooperative exl3_gemv (rows<=8, K in 2..=4 only) / exl3_gemm (every
//!       other case: rows>8, the GEMV heuristic declining, or K in {5,6,8}
//!       which has gemm instances only — 4.05bpw ships lm_head at K=6)
//!       --> C fp16 [rows, N_pad]
//!       (in-kernel: A.*suh, H128 rotation into A_had=A, trellis decode,
//!        MMA with fp32 accumulate on sm_121a, H128 + svh epilogue)
//!   C --one pitched cudaMemcpy2DAsync--> dst rows narrowed to V columns
//!   dst --exl3_f16_to_bf16 IN PLACE--> BF16 logits (grid-stride elementwise,
//!       each index read-then-written once, so src==dst is safe)
//! ```
//!
//! `N_pad` is the trellis row count — the EXL3 export PADS the vocab to a
//! multiple of 128 (Qwen3.8-Flash-Next: 248077 -> 248320) because the trellis
//! format needs `out % 128 == 0`, and the kernels write the full padded row.
//! The C slab absorbs those rows; the pitched copy drops the pad columns while
//! narrowing each row to the logits arena's `V`-column stride.
//!
//! The fp16 C hop costs one extra rounding vs the f32-C kernel variants; at
//! logit magnitudes (|v| << 100) fp16 spacing is finer than BF16's, so the
//! final BF16 logits lose nothing vs a native-BF16 GEMM. The single-token
//! FP32-logits decode path (`use_fp32_logits`) runs the f32-C variant into a
//! dedicated one-row f32 slab and narrow-copies — no conversion rounding.
//!
//! **Scratch ownership / concurrency.** The fp16 `A` and `C` slabs are
//! allocated ONCE at construction, sized to the logits arena's row capacity,
//! and indexed by the DESTINATION logits row: co-dispatched prefill streams
//! each write their own logits row (`finalize_last`), so keying the scratch
//! by that row gives every concurrent stream disjoint slabs with zero
//! hot-path allocation (the 901 playbook: nothing here may alloc or sync —
//! although this path never runs under capture, see below). The `A_had`
//! argument aliases `A` (upstream-sanctioned: the kernel's rotation stage
//! fully writes A_had behind a grid sync before any reuse), so one slab
//! serves both. On Qwen3.8-Flash-Next at the default arena (160 rows):
//! A 0.8 MB + C 79.5 MB + f32 8 rows 7.9 MB ≈ 88 MB, against the ~950 MB the
//! BF16 materialized head would have cost.
//!
//! Every projection runs inside a dispatch SECTION of the model-shared
//! [`ops::Exl3LaunchState`] (the same locks buffer, host mutex and device
//! fence the MoE and dense GDN/attention arms use), so a head GEMV on the
//! decode stream can never be partially co-resident with a prefill chunk's
//! cooperative GEMM on the prefill stream — the deadlock class a private
//! locks buffer does NOT protect against (locks only cover the split-K
//! counters). The head is the last section of a step, never nested inside a
//! layer's section.
//!
//! **CUDA graphs.** Cooperative launches cannot be captured
//! (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`), so installing this head vetoes
//! decode-graph capture at both decision points (`decode_a.rs` single-seq,
//! `decode_a2.rs` multi-seq) — decode runs eagerly. Measured context: decode
//! graphs are speed-neutral on GB10 for this family.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::exl3::{Exl3Codebook, Exl3Weight};

use super::types::TransformerModel;
use crate::layers::ops;

/// Resident state for the natively-served EXL3 LM head. Built once in
/// `factory/build.rs` when `ATLAS_EXL3_NATIVE=1` kept `lm_head` packed,
/// installed via [`TransformerModel::set_lm_head_exl3`].
#[derive(Debug)]
pub(crate) struct Exl3LmHead {
    w: Exl3Weight,
    /// Kernel codebook index: 1 = MCG, 2 = MUL1 (cb0/"3inst" has no
    /// compiled instances and is rejected at construction).
    cb: u32,
    /// The model-shared launch state: locks buffer, SM count, and the
    /// dispatch-section mutex + device fence every cooperative EXL3 launch
    /// in the model (MoE, dense GDN/attention, this head) goes through, so
    /// no two cooperative kernels can be partially co-resident.
    launch: std::sync::Arc<ops::Exl3LaunchState>,
    /// fp16 activation slab `[max_rows, in_dim]` — raw-A AND A_had rotation
    /// scratch (aliased), indexed by destination logits row.
    a_f16: DevicePtr,
    /// fp16 C slab `[max_rows, out_dim]` (PADDED vocab columns), indexed by
    /// destination logits row like `a_f16`.
    c_f16: DevicePtr,
    /// One f32 row `[out_dim]` for the single-token FP32-logits path — its
    /// own slab (not two `c_f16` rows) so a concurrent co-dispatched BF16
    /// projection at another row can never overlap it. Sized to 8 rows: it
    /// is also the fp32-C target of the small-row GEMM at K > 4.
    c_f32_single: DevicePtr,
    /// Logical vocab (`config.vocab_size`) — the narrowed column count.
    vocab: usize,
    /// Row capacity reachable through the logits arena, equal to that arena's
    /// row capacity. Rows `0..max_rows` are keyed by the destination logits
    /// row; row `max_rows` is the RESERVED DRAFT ROW (see `draft_row`).
    max_rows: usize,
    /// Allocated rows of the fp16 slabs = `max_rows + 1`. The extra row is
    /// the qwen4_exp MTP draft head's, whose destination is its own PRIVATE
    /// arena and therefore has no logits-arena row to key on. Reserving a row
    /// instead of borrowing row 0 keeps the draft's rotation scratch disjoint
    /// from every co-dispatched prefill row.
    slab_rows: usize,
}

impl Exl3LmHead {
    /// Validate the tensor against the compiled kernel envelope and the
    /// model geometry, then allocate the launch state. `max_rows` must be
    /// the logits arena's row capacity — every projection destination is a
    /// row range of that arena, and the scratch is keyed by it.
    ///
    /// Allocations: locks 4,202,760 B + scratch `max_rows * hidden * 2` B
    /// (160 rows x 2560 = 819 KB on Qwen3.8-Flash-Next).
    pub(crate) fn new(
        gpu: &dyn GpuBackend,
        w: Exl3Weight,
        vocab: usize,
        hidden: usize,
        max_rows: usize,
    ) -> Result<Self> {
        ensure!(
            crate::weight_map::exl3_native_supported(&w),
            "EXL3 native lm_head: K={} cb={:?} [{}x{}] is outside the compiled \
             kernel envelope (K in {:?}, cb MCG/MUL1, dims %128)",
            w.k_bits,
            w.cb,
            w.in_dim,
            w.out_dim,
            crate::weight_map::EXL3_NATIVE_DENSE_K_BITS,
        );
        // The checkpoint's lm_head rows are PADDED past the logical vocab
        // (the HF embedding ships 248320 rows vs Atlas's vocab_size 248077 on
        // this family — same prefix property the native-FP8 share relies on):
        // the trellis holds `out_dim >= vocab` rows and the kernels write all
        // of them; the pad columns are dropped by the pitched narrow copy in
        // `project`. A large gap means the tensor is for a different model —
        // refuse rather than serve a truncated head.
        ensure!(
            w.in_dim == hidden && w.out_dim >= vocab && w.out_dim - vocab <= 4096,
            "EXL3 native lm_head: trellis geometry [{}x{}] does not match \
             hidden={hidden} vocab={vocab} (expect out = vocab plus row padding)",
            w.in_dim,
            w.out_dim,
        );
        ensure!(max_rows >= 1, "EXL3 native lm_head: zero-row logits arena");
        let cb = match w.cb {
            Exl3Codebook::Mcg => 1,
            Exl3Codebook::Mul1 => 2,
            Exl3Codebook::Inst3 => unreachable!("rejected by exl3_native_supported"),
        };
        // Fail at load, not mid-serve: resolve every instance `project` can
        // select for the head's geometry at this K — the GEMM shape the
        // Blackwell heuristic picks for [hidden -> vocab_pad] (shape 4 at
        // n=248320: n % 512 and n > 16384) plus the universal shape-2
        // fallback, both C dtypes, and the GEMV set only where K has one.
        for name in ops::exl3_dense_kernel_names(w.in_dim, w.out_dim, w.k_bits, cb)? {
            gpu.kernel("exl3_matmul", &name).with_context(|| {
                format!(
                    "EXL3 native lm_head needs exl3_matmul::{name} (gb10 targets only) — \
                     unset ATLAS_EXL3_NATIVE on this target"
                )
            })?;
        }
        let launch = ops::Exl3LaunchState::shared(gpu)?;
        let mut owned: Vec<DevicePtr> = Vec::new();
        let mut alloc = |bytes: usize| -> Result<DevicePtr> {
            match gpu.alloc(bytes) {
                Ok(p) => {
                    owned.push(p);
                    Ok(p)
                }
                Err(e) => {
                    for p in owned.drain(..) {
                        gpu.free(p).ok();
                    }
                    Err(e)
                }
            }
        };
        // One row past the logits arena: the reserved MTP-draft row.
        let slab_rows = max_rows + 1;
        let a_f16 = alloc(slab_rows * hidden * 2)?;
        let c_f16 = alloc(slab_rows * w.out_dim * 2)?;
        // 8 rows (the GEMV tier's cap): row 0 is the fp32-logits single-token
        // path; rows 0..8 are the fp32-C GEMM target for small-row
        // projections at K > 4, where no GEMV instance exists and the f16-C
        // GEMM would hand split-K partials between blocks in fp16.
        let c_f32_single = alloc(ops::EXL3_GEMV_MAX_M * w.out_dim * 4)?;
        tracing::info!(
            "EXL3 native lm_head installed: [{hidden} -> {vocab}] K={} cb={:?} \
             (trellis rows {}), scratch {max_rows} rows ({:.1} MB A + {:.1} MB C) \
             over the shared launch state; decode-graph capture disabled \
             (cooperative launches are not capturable)",
            w.k_bits,
            w.cb,
            w.out_dim,
            slab_rows as f64 * hidden as f64 * 2.0 / 1e6,
            slab_rows as f64 * w.out_dim as f64 * 2.0 / 1e6,
        );
        Ok(Self {
            w,
            cb,
            launch,
            a_f16,
            c_f16,
            c_f32_single,
            vocab,
            max_rows,
            slab_rows,
        })
    }

    /// The reserved scratch row for the qwen4_exp MTP draft head. Its logits
    /// live in the draft's own private `BufferArena`, so it cannot key the
    /// scratch by a logits-arena row the way every other caller does.
    pub(crate) fn draft_scratch_row(&self) -> usize {
        self.max_rows
    }

    /// One-row draft projection into an ARBITRARY destination (the qwen4_exp
    /// MTP head's private arena), using the reserved scratch row. Everything
    /// else — the shared `Exl3LaunchState` section, the fp32-C small-row tier
    /// at K > 4, the pitched narrow copy — is the same path the target's own
    /// head takes, which is the point: a draft scored against a different head
    /// measures the head, not the draft.
    pub(crate) fn project_draft(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.project(gpu, src, 1, self.draft_scratch_row(), dst, stream)
    }

    /// Project `src` BF16 `[rows, H]` into `dst` BF16 `[rows, V]` (contiguous
    /// at the LOGICAL vocab stride). `scratch_row` keys the fp16 slabs and
    /// must be the destination's row index in the logits arena (disjoint
    /// concurrent callers therefore get disjoint scratch).
    pub(crate) fn project(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        rows: usize,
        scratch_row: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (k, n_pad) = (self.w.in_dim, self.w.out_dim);
        ensure!(
            rows >= 1 && scratch_row + rows <= self.slab_rows,
            "EXL3 lm_head: rows={rows} at scratch row {scratch_row} exceeds the \
             {}-row scratch (logits arena capacity + 1 reserved draft row)",
            self.slab_rows,
        );
        let _section = self.launch.section(gpu, stream)?;
        let a = self.a_f16.offset(scratch_row * k * 2);
        let c = self.c_f16.offset(scratch_row * n_pad * 2);
        ops::exl3_bf16_to_f16(gpu, src, a, rows * k, stream)?;
        // Small-m fused GEMV first (m<=8, K in 2..=4 — the GEMV tier has no
        // instances at other K, so those go straight to the GEMM); Ok(false)
        // = heuristic/envelope refusal, fall through to the GEMM (any m,
        // slab-chunked in-kernel). fp16 C lands in the padded slab.
        let mut launched = false;
        if rows <= ops::EXL3_GEMV_MAX_M && ops::exl3_gemv_serves_k(self.w.k_bits) {
            launched = ops::exl3_gemv(
                gpu,
                a,
                self.w.trellis,
                c,
                rows,
                k,
                n_pad,
                self.w.k_bits,
                self.cb,
                false,
                self.launch.locks,
                self.w.suh,
                a, // A_had aliases A
                self.w.svh,
                None,
                self.launch.sm_count,
                stream,
            )?;
        }
        if !launched && rows <= ops::EXL3_GEMV_MAX_M {
            // Small-row GEMM (K > 4, or the GEMV heuristic declined): keep the
            // accumulation and the split-K hand-off in fp32, like the dense
            // arm's small-row tier and the fp32-logits path, then narrow +
            // convert in one strided pass. The f16-C GEMM below is for rows
            // > 8 only, where the slab does not exist.
            ops::exl3_gemm(
                gpu,
                a,
                self.w.trellis,
                self.c_f32_single,
                rows,
                k,
                n_pad,
                self.w.k_bits,
                self.cb,
                true,
                self.launch.locks,
                self.w.suh,
                a, // A_had aliases A
                self.w.svh,
                None,
                self.launch.sm_count,
                stream,
            )?;
            return ops::exl3_f32_to_bf16_2d(
                gpu,
                self.c_f32_single,
                dst,
                rows,
                self.vocab,
                n_pad,
                self.vocab,
                stream,
            );
        }
        if !launched {
            ops::exl3_gemm(
                gpu,
                a,
                self.w.trellis,
                c,
                rows,
                k,
                n_pad,
                self.w.k_bits,
                self.cb,
                false,
                self.launch.locks,
                self.w.suh,
                a, // A_had aliases A
                self.w.svh,
                None,
                self.launch.sm_count,
                stream,
            )?;
        }
        // Narrow each padded row to V columns with ONE pitched copy, then
        // convert in place (each element read-then-written once at the same
        // index, so src==dst is safe).
        gpu.copy_d2d_2d_async(
            c,
            n_pad * 2,
            dst,
            self.vocab * 2,
            self.vocab * 2,
            rows,
            stream,
        )?;
        ops::exl3_f16_to_bf16(gpu, dst, dst, rows * self.vocab, stream)
    }

    /// Single-token projection with FP32 logits output (`use_fp32_logits`
    /// decode path): the f32-C kernel variant writes its own one-row padded
    /// f32 slab, then a narrow copy delivers the first V floats — no
    /// conversion rounding anywhere past the accumulator.
    pub(crate) fn project_single_fp32(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        dst_f32: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (k, n_pad) = (self.w.in_dim, self.w.out_dim);
        let _section = self.launch.section(gpu, stream)?;
        let a = self.a_f16; // row 0: the single-token head is primary-stream only
        ops::exl3_bf16_to_f16(gpu, src, a, k, stream)?;
        // GEMV only where instantiated (K in 2..=4); otherwise the f32-C GEMM.
        let launched = ops::exl3_gemv_serves_k(self.w.k_bits)
            && ops::exl3_gemv(
                gpu,
                a,
                self.w.trellis,
                self.c_f32_single,
                1,
                k,
                n_pad,
                self.w.k_bits,
                self.cb,
                true,
                self.launch.locks,
                self.w.suh,
                a,
                self.w.svh,
                None,
                self.launch.sm_count,
                stream,
            )?;
        if !launched {
            ops::exl3_gemm(
                gpu,
                a,
                self.w.trellis,
                self.c_f32_single,
                1,
                k,
                n_pad,
                self.w.k_bits,
                self.cb,
                true,
                self.launch.locks,
                self.w.suh,
                a,
                self.w.svh,
                None,
                self.launch.sm_count,
                stream,
            )?;
        }
        gpu.copy_d2d_async(self.c_f32_single, dst_f32, self.vocab * 4, stream)
    }
}

impl TransformerModel {
    /// Install the native EXL3 LM head (factory, post-construction — the
    /// `set_dflash_proposer` precedent; keeps the `new` signature stable).
    pub(crate) fn set_lm_head_exl3(&mut self, head: Exl3LmHead) {
        self.lm_head_exl3 = Some(std::sync::Arc::new(head));
    }

    /// Share the native EXL3 head with the qwen4_exp MTP draft head. The
    /// checkpoint has exactly ONE `lm_head` trellis — no `mtp.lm_head` — so
    /// the draft BORROWS this one rather than materializing a copy, and by
    /// construction goes through the same single `Exl3LaunchState`.
    pub(crate) fn lm_head_exl3_shared(&self) -> Option<std::sync::Arc<Exl3LmHead>> {
        self.lm_head_exl3.clone()
    }

    /// BF16 vocab projection through the native EXL3 head, with the scratch
    /// row derived from where `dst` sits in the logits arena (0 for the
    /// arena base; `finalize_last` / the mixed head pass interior rows).
    /// ONE entry point for every BF16 dispatch site so the scratch-keying
    /// invariant lives in a single place.
    pub(super) fn lm_head_exl3_project(
        &self,
        head: &Exl3LmHead,
        src: DevicePtr,
        rows: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let base = self.buffers.logits();
        let row_bytes = self.config.vocab_size * 2;
        ensure!(
            dst.0 >= base.0 && ((dst.0 - base.0) as usize).is_multiple_of(row_bytes),
            "EXL3 lm_head: destination {:#x} is not a row of the logits arena \
             (base {:#x}, row {row_bytes} B) — cannot key the rotation scratch",
            dst.0,
            base.0,
        );
        let scratch_row = (dst.0 - base.0) as usize / row_bytes;
        head.project(self.gpu.as_ref(), src, rows, scratch_row, dst, stream)
    }
}
