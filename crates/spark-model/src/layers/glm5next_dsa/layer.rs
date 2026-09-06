// SPDX-License-Identifier: AGPL-3.0-only

//! `Glm5NextDsaLayer` — the DSA block: NoPE MLA attention over indexer-selected tokens.
//!
//! Decode, end to end:
//!
//! ```text
//! hidden ─┬─ q_a_proj ─ RMSNorm ─┬─ q_absorb ────────────── Q (latent space)
//!         │                      └─ indexer.wq_b ────────── q_idx  ─┐
//!         ├─ indexer.wk ─ LayerNorm(w,b) ─ state.k_normed ──────────┤
//!         ├─ compress_gate ─────────────── state.gate ──────────────┼─ select_tokens
//!         ├─ weights_proj ──────────────── head weights ────────────┘        │
//!         └─ kv_a_proj ─ RMSNorm ─ FP8 ─── paged latent cache                │
//!                                                                            ▼
//!                                            glm5next_dsa_mla_decode_fp8 (gather)
//! ```
//!
//! # 🪤 Four silent-wrong-answer traps this file exists to hold
//!
//! * **Two RMSNorm kernels differ only by a `+1`.** `rms_norm` computes
//!   `x * rms * (1 + w)`; `rms_norm_vanilla` computes `x * rms * w`. Same signature, same
//!   shapes. GLM is plain, so every norm here takes the *vanilla* entry point.
//! * **`indexer.k_norm` is an `nn.LayerNorm` with a bias**, not an RMSNorm at all — mean
//!   subtraction plus a bias term. It takes `nllb_layernorm_bf16(x, w, b, …)`.
//! * **`weights_proj` output must already carry `index_heads^-0.5`.** `dsa_index_scores`
//!   does not apply it. Folded into the weight at load — see [`Glm5NextDsaWeights`].
//! * **Q must be absorbed into latent space before it reaches the decode kernel.** The
//!   kernel dots Q against the 512-dim latent directly, so `q_absorb` is `q_b_proj`
//!   pre-multiplied by `kv_b_proj`'s K half. A raw `q_b_proj` is the right shape per head
//!   (256 vs 512 is not) but the wrong space.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use super::attend::{DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaDecodeKernel, decode_attention};
use super::select::{DsaSelectInputs, DsaSelectScratch, select_tokens};
use super::state::Glm5NextDsaState;
use super::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use crate::layer::{ForwardContext, LayerState, TransformerLayer};

/// GEMM launch: `C[M, N] = A[M, K] @ B[N, K]^T`. Grid `(ceil(N/16), ceil(M/16))`,
/// block `(16, 16)` — one thread per output element.
fn gemm(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    gemv: KernelHandle,
    batchm: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    // M=1 decode -> GEMV; M=2..8 (a K-token verify sweep) -> ONE weight read for all rows;
    // wider -> the tile GEMM. `ops::dense_mm_bf16` owns the policy and the grid coupling.
    crate::layers::ops::dense_mm_bf16(
        gpu,
        &crate::layers::ops::DenseMmKernels {
            gemm: k,
            gemv,
            batchm,
        },
        a,
        b,
        c,
        m,
        n,
        kk,
        stream,
    )
}

/// Every kernel a DSA block launches, beyond the selection set.
#[derive(Clone, Copy)]
pub struct Glm5NextDsaLayerKernels {
    /// `C = A @ B^T`, BF16 out.
    pub gemm: KernelHandle,
    /// Same, FP32 out — the selector wants `q_idx` and the head weights in FP32.
    pub gemm_f32: KernelHandle,
    /// M=1 twins of the two above. `gemv_f32` may be a 0 handle on a target that predates
    /// `dense_gemv_bf16_fp32out`; `gemm` refuses nothing and falls back to the tile arm.
    pub gemv: KernelHandle,
    pub gemv_f32: KernelHandle,
    /// 🔴 `dense_gemv_bf16_batchm` — `2 ..= 8` rows in ONE weight sweep, the arm that makes a
    /// K-token verify pay for q_a/q_b/kv_a/kv_b/o once instead of K times. `0` = unavailable.
    pub gemv_batchm: KernelHandle,
    /// 🪤 **vanilla** = `x * rms * w`. The other `rms_norm` adds 1 to the weight.
    pub rms_norm: KernelHandle,
    /// RMSNorm + FP8 + paged slot write, GLM-target.
    pub latent_write: KernelHandle,
}

impl Glm5NextDsaLayerKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            // 🪤 Module is "gemm", NOT the file stem. `common/KERNEL.toml` [modules] maps
            // `dense_gemm_bf16 = "gemm"`, and an unlisted .cu takes its stem — so the two
            // conventions coexist and only the TOML says which applies. Guessing the stem
            // here resolved to nothing and would have failed at first construction.
            gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
            gemm_f32: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            // 🪤 try_kernel, not kernel: this entry point had ZERO Rust callers before
            // 2026-08-28, so a target that never compiled it must fall back, not refuse.
            gemv_f32: crate::layers::try_kernel(gpu, "gemv", "dense_gemv_bf16_fp32out"),
            gemv_batchm: crate::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            latent_write: gpu
                .kernel("glm5next_mla_latent_write", "glm5next_mla_latent_write_fp8")?,
        })
    }
}

/// One DSA block's weights, already sharded for this rank.
pub struct Glm5NextDsaWeights {
    // ── MLA ──
    pub q_a_proj: DevicePtr,
    pub q_a_layernorm: DevicePtr,
    /// `[local_heads * kv_lora_rank, q_lora_rank]` BF16 — `q_b_proj` **absorbed** through
    /// `kv_b_proj`'s K half, so Q arrives in latent space. See the module header.
    pub q_absorb: DevicePtr,
    pub kv_a_proj: DevicePtr,
    pub kv_a_layernorm: DevicePtr,
    /// `[hidden, local_heads * kv_lora_rank]` BF16, row-parallel — all-reduced by the caller.
    ///
    /// 🪤 **Absorbed**, not the raw checkpoint `o_proj`: the decode kernel leaves its output
    /// in the 512-dim LATENT space, so the projection carries `kv_b_proj`'s V half folded in.
    /// The raw weight is `local_heads * v_head_dim` wide — half of this — and feeding the
    /// latent to it reads 2x past every row rather than merely computing the wrong thing.
    pub o_absorb: DevicePtr,
    // ── indexer (REPLICATED across ranks; see `tp`) ──
    pub wk: DevicePtr,
    pub k_norm_weight: DevicePtr,
    /// 🪤 REQUIRED. `k_norm` is a LayerNorm; a `.weight`-only bind silently drops the
    /// mean subtraction and the bias.
    pub k_norm_bias: DevicePtr,
    pub compress_gate: DevicePtr,
    pub wq_b: DevicePtr,
    /// 🪤 Pre-multiplied by `index_heads^-0.5` at load — `dsa_index_scores` does not scale.
    pub weights_proj: DevicePtr,
    /// `[index_kpool, index_head_dim]` **FP32**. 🪤 BF16 on disk; upconverted at load.
    pub ape: DevicePtr,
}

/// The PREFILL selector runs once for the whole row group instead of once per token.
/// ON by default; `ATLAS_DSA_SELECT_ROWS=0` is the kill-switch back to per-row selection.
///
/// Bit-identical by construction, and the kernels are untouched. `dsa_index_scores`
/// already carries the row axis on `gridDim.y` with a per-row `q_pos[Q]`, and its
/// candidacy test (`dsa_indexer.cu`) is *"a pool is a candidate only when it is complete
/// AND its LAST token is visible to this query"* — `pool_valid[p] && end_c <= q_pos[r]`.
/// Both terms are per-row or per-pool; neither reads the scalar `P`. So widening `P` from
/// row `r`'s own pool count to the group's is a no-op for row `r`: every extra pool is
/// either incomplete or ends past `q_pos[r]`, scores `-FLT_MAX`, and `dsa_topk_pools`
/// excludes it under a TOTAL order (score DESC, then pool index ASC — unique, so the
/// top-`select_k` prefix is unique). There is no partial-pool scoring to disagree about.
///
/// Measured on the 9K prefill probe, n3+n4, fresh container, control -> arm -> control ->
/// arm on ONE image: 146.073 s -> 133.550 s, **-8.57 %**, ~36x the control spread, with all
/// four arms byte-identical on the sealed hash. Promoted to the default after the six
/// canonical probes reproduced the sealed reference 6/6 with the batching on
/// (2026-09-06, `scripts/glm53-dsa-promote/gate-6probe.sh`).
///
/// The four `[max_rows]` twins are allocated only when this is on, so the kill-switch arm
/// keeps the pre-batching heap layout byte for byte. That matters here: this workspace is
/// the one where merely making an allocation unconditionally was itself enough to move the
/// model's sampled output (see `bt`/`sl` below, and A55).
///
/// 🔴 PREFILL ONLY. The K-token verify keeps its per-row selection: it runs the
/// device-geometry (`replay_safe`) path, and `select_tokens` refuses a ceiling launch at
/// `q_rows > 1` because a replayed graph would index the wrong rows.
/// Whether ONE selection pass covers the whole row group, or each row selects on its own.
///
/// Extracted from `decode_k` so the choice is a table a test can drive, not a boolean buried
/// in a 300-line function. Every term is load-bearing:
///
/// * `workspace_ready` — the four `[max_rows]` twins exist. False under
///   `ATLAS_DSA_SELECT_ROWS=0`, which is what keeps that arm's heap layout byte for byte.
/// * `is_prefill` — stated by the caller. NOTHING in `ForwardContext` implies it:
///   `decode_step` is false for prefill AND verify, and `graph_capture` is false for prefill
///   AND for an eager verify (`verify_a` hard-codes it; `verify_b/c/c2/d/fused` take it from
///   `use_graphs`, false under `ATLAS_GLM_VERIFY_GRAPHS=0`, high-speed swap, or
///   `ATLAS_LORA_EAGER`).
/// * `!graph_capture` — independent of the above, and kept for its own reason:
///   `select_rows_batched` issues a host `copy_h2d`, which is
///   `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` inside a recording stream.
/// * `k > 1` — a single row has nothing to batch, and there is no behaviour to change.
pub(crate) fn batch_select_enabled(
    workspace_ready: bool,
    is_prefill: bool,
    graph_capture: bool,
    k: usize,
) -> bool {
    workspace_ready && is_prefill && !graph_capture && k > 1
}

pub(crate) fn dsa_select_rows_enabled() -> bool {
    std::env::var("ATLAS_DSA_SELECT_ROWS").as_deref() != Ok("0")
}

/// Scratch reused across decode steps. Allocated once per layer.
pub struct Glm5NextDsaWorkspace {
    q_a: DevicePtr,
    q_resid: DevicePtr,
    q_abs: DevicePtr,
    kv_a: DevicePtr,
    q_idx: DevicePtr,
    head_weights: DevicePtr,
    q_pos: DevicePtr,
    q_mask: DevicePtr,
    /// `[max_rows, ...]` twins of `q_idx` / `head_weights` / `q_pos` / `q_mask`, used only
    /// by the batched prefill selector. NULL when `ATLAS_DSA_SELECT_ROWS=0` — see
    /// [`dsa_select_rows_enabled`] for why they are not allocated unconditionally.
    q_idx_rows: DevicePtr,
    head_weights_rows: DevicePtr,
    q_pos_rows: DevicePtr,
    q_mask_rows: DevicePtr,
    slot: DevicePtr,
    attn_out: DevicePtr,
    /// Block table for the paged gather, `[max_dsa_context]` i32. PERSISTENT.
    /// 🔴 This used to be a `gpu.alloc` + `gpu.free` on EVERY DSA layer of EVERY
    /// decode token — 11 allocs + 11 frees per token. `cuMemAlloc` serialises against
    /// the driver, and nsys (2026-08-28) charged the alloc/copy/free cluster ~98 us of
    /// GPU-idle per DSA layer, 1.08 ms of an 79 ms step.
    bt: DevicePtr,
    /// Sequence length for the paged gather, one i32. PERSISTENT, same reason.
    sl: DevicePtr,
    /// Capacity of `bt` in ENTRIES, so the forward can refuse rather than overrun it.
    bt_cap: usize,
    /// Widest verify this scratch can serve. `1` on the serial decode path.
    max_rows: usize,
    /// `[index_head_dim]` BF16 staging for the indexer row, at a FIXED address.
    ///
    /// 🔴 The projections used to write straight into `k_normed`/`gate` at
    /// `offset(pos * D * 2)` — a host-computed address, which a captured graph freezes at
    /// the capture-time row. Under capture they land here and `dsa_indexer_store` places
    /// them from a device-side `pos`. Same arithmetic, one extra 256-byte copy.
    stage_k: DevicePtr,
    stage_gate: DevicePtr,
    /// `[5]` i32 selector geometry, written on device once per step by `dsa_write_geom`.
    geom_dev: DevicePtr,
    select: DsaSelectScratch,
}

impl Glm5NextDsaWorkspace {
    /// `max_rows` is the widest speculative verify this workspace serves. The projection
    /// buffers, `attn_out` and the selector's OUTPUT row scale with it; the indexer staging
    /// rows, the block table and the selector's within-pass temporaries stay per-row,
    /// because [`Glm5NextDsaLayer::decode_k`] still SELECTS one token at a time.
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextDsaConfig, max_rows: usize) -> Result<Self> {
        let rows = max_rows.max(1);
        // Sized at the DSA context cap so a growing sequence never reallocates.
        // `q_rows = rows`: selection still runs one row at a time, but its OUTPUT is
        // `[max_rows, out_width]` so a K-row verify can attend every row in one launch.
        let geom =
            super::select::DsaSelectGeometry::plan(cfg, super::state::max_dsa_context(cfg), rows)?;
        let bt_cap = super::state::max_dsa_context(cfg).max(1);
        let persist = std::env::var("ATLAS_GLM_DSA_ALLOC_PER_STEP").as_deref() != Ok("1");
        let batch_select = dsa_select_rows_enabled();
        Ok(Self {
            q_a: gpu.alloc(rows * (cfg.q_lora_rank * 2))?,
            q_resid: gpu.alloc(rows * (cfg.q_lora_rank * 2))?,
            q_abs: gpu.alloc(rows * (cfg.local_heads * cfg.kv_lora_rank * 2))?,
            kv_a: gpu.alloc(rows * (cfg.kv_lora_rank * 2))?,
            q_idx: gpu.alloc(cfg.index_heads * cfg.index_head_dim * 4)?,
            head_weights: gpu.alloc(cfg.index_heads * 4)?,
            q_pos: gpu.alloc(4)?,
            q_idx_rows: if batch_select {
                gpu.alloc(rows * cfg.index_heads * cfg.index_head_dim * 4)?
            } else {
                DevicePtr(0)
            },
            head_weights_rows: if batch_select {
                gpu.alloc(rows * cfg.index_heads * 4)?
            } else {
                DevicePtr(0)
            },
            q_pos_rows: if batch_select {
                gpu.alloc(rows * 4)?
            } else {
                DevicePtr(0)
            },
            q_mask_rows: if batch_select {
                // Same "one real query position per row" the scalar `q_mask` encodes,
                // written once so the batched pass never needs a per-step H2D for it.
                let p = gpu.alloc(rows)?;
                gpu.memset_async(p, 1, rows, 0)?;
                gpu.synchronize(0)?;
                p
            } else {
                DevicePtr(0)
            },
            q_mask: {
                // Decode always presents one real query position. Set ONCE — writing it per
                // token cost a blocking H2D per DSA layer and made the step uncapturable.
                let p = gpu.alloc(1)?;
                gpu.memset_async(p, 1, 1, 0)?;
                gpu.synchronize(0)?;
                p
            },
            slot: gpu.alloc(8)?,
            attn_out: gpu.alloc(rows * (cfg.local_heads * cfg.kv_lora_rank * 2))?,
            // One entry per cached token is the worst case (block_size == 1), so the
            // DSA context cap bounds it for every block size.
            //
            // 🔴 Allocated ONLY when `ATLAS_GLM_DSA_PERSIST_BT=1`. Not a micro-optimisation:
            // making these two allocations UNCONDITIONALLY — even leaving them unused —
            // is by itself enough to change the model's sampled output (measured t27,
            // 2026-08-28). See A55 and the note at the use site.
            bt: if persist {
                gpu.alloc(bt_cap * 4)?
            } else {
                DevicePtr(0)
            },
            // 🔴 ANOMALIES A65: `[rows]`, NOT one. `attend_rows` runs ONE launch for all
            // k rows and `glm5next_dsa_mla_decode` reads `seq_lens[blockIdx.y]`, so a
            // single i32 here left every row past the first reading past the allocation.
            sl: if persist { gpu.alloc(rows * 4)? } else { DevicePtr(0) },
            bt_cap,
            max_rows: rows,
            stage_k: gpu.alloc(cfg.index_head_dim * 2)?,
            stage_gate: gpu.alloc(cfg.index_head_dim * 2)?,
            geom_dev: gpu.alloc(5 * 4)?,
            select: DsaSelectScratch::alloc(gpu, cfg, &geom)?,
        })
    }
}

pub struct Glm5NextDsaLayer {
    pub cfg: Glm5NextDsaConfig,
    pub weights: Glm5NextDsaWeights,
    pub kernels: Glm5NextDsaLayerKernels,
    pub select_kernels: Glm5NextDsaKernels,
    pub decode_kernel: Glm5NextDsaDecodeKernel,
    pub workspace: Glm5NextDsaWorkspace,
    /// Index in the MODEL stack (0..num_hidden_layers). Diagnostics only.
    pub layer_idx: usize,
    /// Index in the KV POOL — the running ordinal over KV-cache-consuming layers,
    /// which for GLM-5.3 is 0..11 over the sparse layers, not 0..45.
    ///
    /// 🪤 These two are NOT interchangeable. The pool is sized to
    /// `ModelConfig::num_attention_layers()`; indexing it with `layer_idx` reads
    /// past the end of the allocation on every layer after the first.
    pub attn_layer_idx: usize,
    pub rms_eps: f32,
    /// FP8 latent-cache scale. Reads and writes must agree; the write takes `1/scale`.
    pub kv_scale: f32,
    /// Persistent block-table buffers instead of a `gpu.alloc`/`gpu.free` per DSA layer per
    /// token. ON by default; `ATLAS_GLM_DSA_ALLOC_PER_STEP=1` restores the old path.
    pub persist_bt: bool,
}

impl Glm5NextDsaLayer {
    /// Project `hidden` into the indexer cache at position `pos`, then advance.
    ///
    /// Writes `k_normed` and `gate` **directly into the state rows** rather than through a
    /// staging buffer: the selector reads `k[raw * D + d]` over the whole context, so the
    /// cache is the natural destination and a copy would buy nothing.
    pub fn indexer_forward(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        state: &mut Glm5NextDsaState,
        // Some(pos) => write through the FIXED staging row and let `dsa_indexer_store`
        // place it from this device-side position. None => the host-offset path.
        pos_dev: Option<DevicePtr>,
        stream: u64,
    ) -> Result<()> {
        // 🔴 BEFORE any write. Everything below writes into row `state.len()`; past the cap
        // that row is off the end of the buffer, and the resulting sticky CUDA 700 kills the
        // whole context, not just this request. A62.
        state.ensure_room(1)?;
        let d = self.cfg.index_head_dim;
        let pos = state.len();
        let off = state.row_offset(pos);
        let w = &self.workspace;
        let (k_dst, gate_dst) = match pos_dev {
            Some(_) => (w.stage_k, w.stage_gate),
            None => (state.k_normed.offset(off), state.gate.offset(off)),
        };

        // k_raw -> the state row, then LayerNorm in place.
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.wk,
            k_dst,
            1,
            d,
            self.cfg.hidden,
            stream,
        )?;
        // 🪤 LayerNorm WITH BIAS, in place, one row.
        KernelLaunch::new(gpu, self.select_kernels.k_norm)
            .grid([1, 1, 1])
            .block([d.min(1024) as u32, 1, 1])
            .shared_mem((d.min(1024) * 4) as u32)
            .arg_ptr(k_dst)
            .arg_ptr(self.weights.k_norm_weight)
            .arg_ptr(self.weights.k_norm_bias)
            .arg_u32(1)
            .arg_u32(d as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;

        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.compress_gate,
            gate_dst,
            1,
            d,
            self.cfg.hidden,
            stream,
        )?;

        // Per-head selector weights, FP32 straight out of the GEMM, from the LAYER INPUT.
        // `weights_proj` is `[index_heads, hidden]` and the reference is
        // `weights_proj(hidden) * index_heads**-0.5`, with the scale already folded into the
        // weight at load (`build.rs` transform 2). Computed here rather than in
        // `select_and_attend` for the plain reason that this is the function that HAS
        // `hidden`; `select_and_attend` does not, which is how it came to read `q_resid`
        // instead and overrun it by 5120 bytes. See A55.
        gemm(
            gpu,
            self.kernels.gemm_f32,
            self.kernels.gemv_f32,
            // No FP32-out batchm twin exists; the selector's two sites stay on gemv/tile.
            KernelHandle(0),
            hidden,
            self.weights.weights_proj,
            self.workspace.head_weights,
            1,
            self.cfg.index_heads,
            self.cfg.hidden,
            stream,
        )?;

        match pos_dev {
            // 🔴 Placement and the validity mark both from a DEVICE position — a memset at
            // `valid.offset(pos)` is one more host-baked address a graph would freeze.
            Some(pd) => {
                KernelLaunch::new(gpu, self.select_kernels.indexer_store)
                    .grid([1, 1, 1])
                    .block([d.min(1024) as u32, 1, 1])
                    .arg_ptr(w.stage_k)
                    .arg_ptr(w.stage_gate)
                    .arg_ptr(pd)
                    .arg_ptr(state.k_normed)
                    .arg_ptr(state.gate)
                    .arg_ptr(state.valid)
                    .arg_u32(d as u32)
                    .launch(stream)?;
            }
            // Validity is per position and this one is real.
            None => gpu.memset_async(state.valid.offset(pos), 1, 1, stream)?,
        }
        state.advance(1)
    }

    /// Everything after the indexer write: selector inputs and the selection for ONE query
    /// row, into row `row` of the workspace's selection scratch.
    ///
    /// 🔴 SELECTION stays per-row inside a K-token verify: the selector's geometry, its
    /// top-k over `[0, len)` and its visibility test are all functions of THIS token's
    /// position, and the indexer cache grows by one row between them. The gather-ATTEND
    /// does not — every row reads the same cache with its own index row — so it is hoisted
    /// out to one K-row launch in `attend_rows`. Three serial launches of 32 head-blocks
    /// each left most of the GPU idle: 7.13 ms/step of the K=3 budget (nsys 2026-08-29).
    #[allow(clippy::too_many_arguments)]
    fn select_row(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        state: &Glm5NextDsaState,
        q_pos_dev: DevicePtr,
        replay_safe: bool,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        let geom = state.geometry(&self.cfg, 1)?;

        // Selector Q and head weights, FP32 straight out of the GEMM.
        gemm(
            gpu,
            self.kernels.gemm_f32,
            self.kernels.gemv_f32,
            // No FP32-out batchm twin exists; the selector's two sites stay on gemv/tile.
            KernelHandle(0),
            w.q_resid.offset(row * self.cfg.q_lora_rank * 2),
            self.weights.wq_b,
            w.q_idx,
            1,
            self.cfg.index_heads * self.cfg.index_head_dim,
            self.cfg.q_lora_rank,
            stream,
        )?;
        // 🔴 `head_weights` is NOT computed here any more — see `indexer_forward`. It used to
        // be, from `w.q_resid` with `K = cfg.hidden`, which was wrong twice over: the
        // reference projects the LAYER INPUT (`gen_dsa_indexer_golden.py`:
        // `weights_proj(hidden) * NH**-0.5`), and `q_resid` is only `[q_lora_rank] = 1536`
        // BF16, so reading 4096 of them ran **5120 bytes past the end of the allocation**.
        // That out-of-bounds read was ANOMALIES A55: the head weights were a function of
        // whatever the allocator had placed after `q_resid`, which is why the model's output
        // moved when the heap moved, when allocations were zeroed, and when an unrelated
        // buffer was added. Found by red-zoning the allocator and bisecting the guard bands.

        let inputs = DsaSelectInputs {
            k_normed: state.k_normed,
            gate: state.gate,
            valid: state.valid,
            ape: self.weights.ape,
            q: w.q_idx,
            weights: w.head_weights,
            q_pos: q_pos_dev,
            // Always 1 for a decode step; written once at workspace alloc, never per token.
            q_mask: w.q_mask,
            first_key: 0,
            geom_dev: if replay_safe {
                w.geom_dev
            } else {
                DevicePtr::NULL
            },
        };
        // Under capture the grid and the shared-memory request go to the context CEILING and
        // the live extents come off `geom_dev`, so ONE graph serves every context length.
        let launch = if replay_safe {
            super::select::DsaSelectLaunch::Ceiling {
                max_pools: super::select::contiguous_pool_count(
                    self.cfg.index_kpool,
                    super::state::max_dsa_context(&self.cfg),
                ),
            }
        } else {
            super::select::DsaSelectLaunch::Exact
        };
        let t = crate::layers::glm5next_layer::profile::start();
        // Row `row` of the `[max_rows, out_width]` selection scratch. The kernels still run
        // one query row (`q_rows == 1`), they just land in this row's slot, so `attend_rows`
        // can read all K index rows in one launch.
        select_tokens(
            gpu,
            &self.select_kernels,
            &self.cfg,
            &geom,
            &inputs,
            &w.select.row(row, &self.cfg),
            launch,
            stream,
        )?;
        use crate::layers::glm5next_layer::profile;
        profile::end(profile::DSA_SELECT, t, gpu, stream);
        Ok(())
    }

    /// The selection for ALL `k` query rows in ONE pass — the prefill twin of
    /// [`Self::attend_rows`]. On by default; disabled by `ATLAS_DSA_SELECT_ROWS=0`.
    ///
    /// Per 9,000-token prefill this replaces 4 x 9,000 x 11 single-row launches. Measured
    /// at `6228baa2` (nsys s3-candcap, n3+n4): `dsa_topk_pools` and `dsa_expand_selection`
    /// each run **grid(1,1,1)** — one block, 48 SMs idle — 99,000 times for 5.773 s and
    /// 5.071 s, with `dsa_index_scores` a further 4.211 s. 15.06 s of a 147.8 s prefill
    /// spent at ~2 % occupancy. Each row's block does exactly the work its own launch did;
    /// only the launch count changes.
    ///
    /// Exactness, in three parts, all of them properties the kernels already have:
    ///   1. `q_pos` is `[Q]` and the candidacy test reads `q_pos[r]`, so causality is
    ///      per-row. A pool that ends past row `r` is not a candidate for row `r`.
    ///   2. `pool_valid[p]` gates completeness, so the in-progress pool is excluded for
    ///      every row exactly as it is today.
    ///   3. Non-candidates score `-FLT_MAX` and `dsa_topk_pools` selects under a total
    ///      order over (score, pool index), so a longer `P` walk reaches the identical
    ///      top-`select_k` set AND order.
    /// Widening `P` to the group's pool count therefore cannot change any row's selection.
    ///
    /// The `q_idx` projections stay per-row: there is no FP32-out `batchm` twin, and they
    /// are 2.765 s of `dense_gemv_bf16_fp32out` that this lane does not claim.
    #[allow(clippy::too_many_arguments)]
    fn select_rows_batched(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &Glm5NextDsaState,
        q_pos_host: &[i32],
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        // `q_rows = k`, and `len` is the cache length AFTER all k indexer writes — which is
        // the point: `P` is the group's final pool count and each row masks itself back down
        // to its own horizon via `q_pos[r]`.
        let geom = state.geometry(&self.cfg, k)?;
        let idx_row = self.cfg.index_heads * self.cfg.index_head_dim;
        for row in 0..k {
            gemm(
                gpu,
                self.kernels.gemm_f32,
                self.kernels.gemv_f32,
                // Same as `select_row`: no FP32-out batchm twin exists.
                KernelHandle(0),
                w.q_resid.offset(row * self.cfg.q_lora_rank * 2),
                self.weights.wq_b,
                w.q_idx_rows.offset(row * idx_row * 4),
                1,
                idx_row,
                self.cfg.q_lora_rank,
                stream,
            )?;
        }
        let q_pos_bytes: Vec<u8> = q_pos_host.iter().flat_map(|p| p.to_le_bytes()).collect();
        gpu.copy_h2d(&q_pos_bytes, w.q_pos_rows)?;
        let inputs = DsaSelectInputs {
            k_normed: state.k_normed,
            gate: state.gate,
            valid: state.valid,
            ape: self.weights.ape,
            q: w.q_idx_rows,
            weights: w.head_weights_rows,
            q_pos: w.q_pos_rows,
            q_mask: w.q_mask_rows,
            first_key: 0,
            // Host geometry only. The device-geometry path is decode-only and
            // `select_tokens` refuses it at `q_rows > 1`.
            geom_dev: DevicePtr::NULL,
        };
        let t = crate::layers::glm5next_layer::profile::start();
        // The BASE of the `[max_rows, out_width]` scratch, not a row slice: the kernels
        // carry the row axis themselves, so rows 0..k land in their own slots and
        // `attend_rows` reads all k exactly as before.
        select_tokens(
            gpu,
            &self.select_kernels,
            &self.cfg,
            &geom,
            &inputs,
            &w.select,
            super::select::DsaSelectLaunch::Exact,
            stream,
        )?;
        crate::layers::glm5next_layer::profile::end(
            crate::layers::glm5next_layer::profile::DSA_SELECT,
            t,
            gpu,
            stream,
        );
        Ok(())
    }

    /// The gather-attend for ALL `rows` query rows in ONE launch.
    ///
    /// Every row reads the same paged latent cache with its own selection row, its own
    /// `seq_len` and its own block-table row, so `gridDim.y` carries the row axis and the
    /// per-row arithmetic is untouched — bit-identical to the serial launches it replaces.
    /// `q_abs`, `attn_out`, the metadata's `seq_len`/`block_table` and the selection scratch
    /// are all `[rows, ...]` at the SAME strides the kernel indexes.
    #[allow(clippy::too_many_arguments)]
    fn attend_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        state: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        block_table_dev: DevicePtr,
        seq_lens_dev: DevicePtr,
        paging: &DsaDecodePaging,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::glm5next_layer::profile;
        let w = &self.workspace;
        // Only `out_width` and the `q_rows == num_seqs` agreement are read here.
        let geom = state.geometry(&self.cfg, rows)?;
        let paging = DsaDecodePaging {
            num_seqs: rows,
            ..*paging
        };
        let t = profile::start();
        let pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        decode_attention(
            gpu,
            self.decode_kernel,
            &self.cfg,
            &geom,
            &paging,
            &DsaDecodeInputs {
                q: w.q_abs,
                k_cache: pool,
                v_cache: pool, // absorbed NoPE MLA: K and V are the same latent
                out: w.attn_out,
                block_tables: block_table_dev,
                seq_lens: seq_lens_dev,
                sel_indices: w.select.tokens(),
                k_scale: self.kv_scale,
                v_scale: self.kv_scale,
            },
            stream,
        )?;
        profile::end(profile::DSA_ATTEND, t, gpu, stream);
        Ok(())
    }
    /// ONE drafter CONTEXT row: the KV latent and the indexer entry, with no query, no
    /// selection and no attend.
    ///
    /// The MTP drafter's context rows only have to EXIST in these two caches — their block
    /// output is discarded. Both caches are pure functions of the row's own input, exactly as
    /// the Qwen drafter prefill exploits, so a context row costs `kv_a` + `latent_write` + the
    /// indexer's `wk`, not a decode step. No MoE, no `o_proj`, no `lm_head`.
    ///
    /// 🪤 `seq_len` is BOTH the row's KV slot and its RoPE position (the indexer takes its
    /// position from `state.len()`), so the drafter's row space must stay DENSE — every pair
    /// key from 0 up must have been written. That is what `prefill_drafter` + the catch-up
    /// feed are for.
    #[allow(clippy::too_many_arguments)]
    pub fn write_kv_row(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?;
        match st.len().cmp(&seq_len) {
            std::cmp::Ordering::Greater => st.rewind_to(seq_len)?,
            std::cmp::Ordering::Less => bail!(
                "DSA layer {}: indexer cache holds {} rows but the drafter is at {seq_len} — \
                 rows are MISSING, not merely stale.",
                self.layer_idx,
                st.len()
            ),
            std::cmp::Ordering::Equal => {}
        }
        let w = &self.workspace;
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.kv_a_proj,
            w.kv_a,
            1,
            self.cfg.kv_lora_rank,
            self.cfg.hidden,
            stream,
        )?;
        let block_size = kv_cache.config().block_size;
        let logical = seq_len / block_size;
        let physical = *block_table.get(logical).ok_or_else(|| {
            anyhow::anyhow!(
                "DSA layer {}: block table has {} entries, needs logical block {logical} for \
                 drafter row {seq_len}",
                self.layer_idx,
                block_table.len()
            )
        })? as usize;
        let slot = (physical * block_size + seq_len % block_size) as i64;
        gpu.copy_h2d(&slot.to_le_bytes(), w.slot)?;
        KernelLaunch::new(gpu, self.kernels.latent_write)
            .grid([1, 1, 1])
            .block([self.cfg.kv_lora_rank as u32, 1, 1])
            .arg_ptr(w.kv_a)
            .arg_ptr(self.weights.kv_a_layernorm)
            .arg_ptr(kv_cache.k_pool_ptr(self.attn_layer_idx))
            .arg_ptr(w.slot)
            .arg_u32(self.cfg.kv_lora_rank as u32)
            .arg_f32(self.rms_eps)
            .arg_f32(1.0 / self.kv_scale)
            .launch(stream)?;
        self.indexer_forward(gpu, hidden, st, None, stream)
    }

    /// K tokens of one sequence: the projections batched, selection and attention NOT.
    ///
    /// The weight-heavy halves — `q_a`, the absorbed `q_b`, `kv_a` and the `o_absorb` output
    /// projection — sweep their weights ONCE for all K rows (1,290 MB/rank/token between them).
    /// Everything between them is a function of the individual token's position: the paged KV
    /// slot, the indexer row, the selector geometry over `[0, len)` and the gather-attend.
    ///
    /// 🔴 Bit-identical to K serial [`TransformerLayer::decode`] calls, which is the
    /// requirement: an accepted draft must be the token the unspeculated engine would have
    /// emitted. `ops::dense_mm_bf16` reproduces each row's K-iteration order and reduction tree,
    /// and `rms_norm_vanilla`'s grid is the token axis.
    ///
    /// 🪤 REFUSES a SCALAR (`num_seqs == 1`) `attn_metadata` at k > 1. Those scalars — position,
    /// KV slot, seq len — describe ONE token, so K rows sharing them would write K queries into
    /// the same paged slot and select over the same position: a wrong answer with no shape error.
    ///
    /// 🔴 It ACCEPTS a K-ROW `attn_metadata` (`num_seqs == k`), which is what the graphed verify
    /// paths (`verify_b`/`verify_c`) already upload: positions `[k]` u32, slot `[k]` i64, seq_len
    /// `[k]` i32, block_table `[k][max_blocks_per_seq]` i32, all at stable device addresses
    /// written BEFORE capture or replay. Row `r` reads element `r` of each. Without this the
    /// layer fell through to its own per-row `copy_h2d`, and an H2D on a capturing stream fails
    /// with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED — which is why the K-token verify was eager.
    /// At `k == 1` every row offset is 0, so that path is unchanged byte for byte.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_k(
        &self,
        hidden: DevicePtr,
        k: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        // TRUE only on a prefill sub-chunk. Passed explicitly by
        // [`crate::layers::glm5next_layer::Glm5NextLayer::forward_k`]'s two call sites; it
        // is NOT inferable from the context. `decode_step` is false for prefill AND for a
        // speculative verify, and `graph_capture` is false for prefill AND for an EAGER
        // verify — `verify_a` hard-codes `graph_capture: false`, and `verify_b/c/c2/d` set
        // it from `use_graphs`, which is false under `ATLAS_GLM_VERIFY_GRAPHS=0`, under
        // high-speed swap, and under `ATLAS_LORA_EAGER`. See `select_rows_batched`.
        is_prefill: bool,
    ) -> Result<()> {
        use crate::layers::glm5next_layer::profile;
        // Captured before the KV borrows below, for the block-table trim at the
        // upload site. See the comment there (ANOMALIES A58).
        let bt_block_size = kv_cache.block_size().max(1);
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?;
        // 🔴 The indexer stream must advance in lockstep with the KV cache — a drift selects
        // over the wrong context. Two drifts are possible and they are NOT symmetric:
        //
        // * AHEAD (`len > seq_len`) is the speculative-verify reject. The K rows of a verify
        //   were written, the sequence rolled back to the accepted prefix, and the rows past
        //   it are now unreachable: the selector reads `[0, len)` and the next write starts
        //   at `seq_len`, so they are overwritten before anything can select over them.
        //   Rewind and continue — this is the KV cache's own semantics for rejected slots,
        //   and making it self-healing here is why no rollback callback has to reach into
        //   eleven DSA layers.
        // * BEHIND (`len < seq_len`) means rows were never written. Nothing can repair that,
        //   so it stays a hard error.
        match st.len().cmp(&seq_len) {
            std::cmp::Ordering::Greater => st.rewind_to(seq_len)?,
            std::cmp::Ordering::Less => bail!(
                "DSA layer {}: indexer cache holds {} tokens but the sequence is at {} — \
                 rows are MISSING, not merely stale. The indexer stream must advance in \
                 lockstep with the KV cache.",
                self.layer_idx,
                st.len(),
                seq_len
            ),
            std::cmp::Ordering::Equal => {}
        }
        if k == 0 || k > self.workspace.max_rows {
            bail!(
                "DSA layer {}: a {k}-token verify does not fit a workspace built for {}",
                self.layer_idx,
                self.workspace.max_rows
            );
        }
        // A K-row pass may use `attn_metadata` ONLY when it carries one entry per row.
        if k > 1
            && let Some(m) = ctx.attn_metadata.as_ref()
            && m.num_seqs as usize != k
            && ctx.decode_step
        {
            bail!(
                "DSA layer {}: a {k}-row pass cannot share attn_metadata describing {} \
                 token(s) — its position and KV slot describe a single token",
                self.layer_idx,
                m.num_seqs
            );
        }
        // 🪤 Guarded on `num_seqs == k`, NOT on `decode_step`. A chunked prefill also passes
        // metadata through this call at k == 1, and its `num_seqs` is the chunk width — so a
        // one-token chunk is the only case where the two could be confused, and `decode_step`
        // still separates them there.
        let rowwise_meta = (k > 1)
            .then(|| ctx.attn_metadata.as_ref())
            .flatten()
            .filter(|m| m.num_seqs as usize == k);
        let gpu = ctx.gpu;
        let w = &self.workspace;
        let t_proj = crate::layers::glm5next_layer::profile::start();

        // ── q path ──
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.q_a_proj,
            w.q_a,
            k,
            self.cfg.q_lora_rank,
            self.cfg.hidden,
            stream,
        )?;
        // 🪤 vanilla: x * rms * w, no `1 +`.
        KernelLaunch::new(gpu, self.kernels.rms_norm)
            // 🪤 `rms_norm_vanilla`'s grid IS the token axis, so k rows is one launch doing
            // block-for-block what k launches did — bit-identical.
            .grid([k as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(w.q_a)
            .arg_ptr(self.weights.q_a_layernorm)
            .arg_ptr(w.q_resid)
            .arg_u32(self.cfg.q_lora_rank as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        // Q absorbed into latent space in one GEMM.
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            w.q_resid,
            self.weights.q_absorb,
            w.q_abs,
            k,
            self.cfg.local_heads * self.cfg.kv_lora_rank,
            self.cfg.q_lora_rank,
            stream,
        )?;

        // ── kv path: latent -> FP8 -> paged slot ──
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.kv_a_proj,
            w.kv_a,
            k,
            self.cfg.kv_lora_rank,
            self.cfg.hidden,
            stream,
        )?;
        let mut attend_bt = DevicePtr::NULL;
        let mut attend_sl = DevicePtr::NULL;
        // Row 0 allocates the shared bt/sl scratch on the per-step-alloc path; it is freed
        // after the attend, which reads it. ANOMALIES A65.
        let mut owns_scratch = false;
        let mut attend_paging: Option<DsaDecodePaging> = None;
        // 🔴 PREFILL ONLY, and said so EXPLICITLY. `!ctx.graph_capture` does NOT mean
        // "prefill": `verify_a` builds its context with `graph_capture: false` outright, and
        // `verify_b/c/c2/d/fused` set it from `use_graphs`, which is false whenever
        // `ATLAS_GLM_VERIFY_GRAPHS=0`, high-speed swap is engaged, or `ATLAS_LORA_EAGER` is
        // set. `!ctx.decode_step` does not separate them either — a verify sets it false too.
        // So the caller states it. An eager K-row verify would otherwise silently take a path
        // that was measured and qualified on prefill alone.
        //
        // `!ctx.graph_capture` is KEPT as a second, independent condition rather than
        // replaced: `select_rows_batched` does a host `copy_h2d` of `q_pos`, which is
        // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED inside a recording stream. Both must hold.
        //
        // The buffers are NULL under `ATLAS_DSA_SELECT_ROWS=0`, so this is false on that arm
        // and the heap layout is unchanged there.
        let batch_select =
            batch_select_enabled(w.q_idx_rows.0 != 0, is_prefill, ctx.graph_capture, k);
        let mut batch_q_pos: Vec<i32> = Vec::with_capacity(if batch_select { k } else { 0 });
        for row in 0..k {
            let pos = seq_len + row;
            let block_size = kv_cache.config().block_size;
            // 🔴 Every per-step scalar this layer needs — position, KV slot, seq_len, block
            // table — is ALREADY uploaded once per decode step by `decode_a` into
            // `attn_metadata`, at stable addresses, BEFORE any graph capture or replay. Reading
            // those pointers instead of doing our own `copy_h2d` removes FIVE blocking H2Ds
            // (each one a `cuStreamSynchronize`) per DSA layer per token — 55 stream drains on
            // this model — and is what makes the decode step capturable at all: an H2D inside a
            // capturing stream fails with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED.
            //
            // 🪤 The two encodings must agree byte for byte, and they do: `positions` is the
            // u32 `seq_len` (same bits as our i32), `slot` the same i64 `block*block_size +
            // seq_len % block_size`, `seq_len` the same i32 `seq_len + 1`, and `block_table`
            // the same ids as i32 rather than u32.
            // 🪤 ONLY on a real decode step — `prefill_default` calls this same `decode` per
            // token with the prefill context, where these are arrays or NULL. See
            // `ForwardContext::decode_step`.
            let meta = if ctx.decode_step {
                ctx.attn_metadata.as_ref()
            } else {
                rowwise_meta
            };
            // Row strides into the K-row arrays. At k == 1 every one of these is 0.
            let bt_stride = meta.map_or(0, |m| m.max_blocks_per_seq as usize) * 4;
            let slot_dev = match meta {
                Some(m) => m.slot.offset(row * 8),
                None => {
                    let logical = pos / block_size;
                    let physical = *block_table.get(logical).ok_or_else(|| {
                        anyhow::anyhow!(
                            "DSA layer {}: block table has {} entries, needs logical block \
                             {logical} for position {pos}",
                            self.layer_idx,
                            block_table.len()
                        )
                    })? as usize;
                    let slot = (physical * block_size + pos % block_size) as i64;
                    gpu.copy_h2d(&slot.to_le_bytes(), w.slot)?;
                    w.slot
                }
            };
            KernelLaunch::new(gpu, self.kernels.latent_write)
                .grid([1, 1, 1])
                .block([self.cfg.kv_lora_rank as u32, 1, 1])
                .arg_ptr(w.kv_a.offset(row * self.cfg.kv_lora_rank * 2))
                .arg_ptr(self.weights.kv_a_layernorm)
                .arg_ptr(kv_cache.k_pool_ptr(self.attn_layer_idx))
                .arg_ptr(slot_dev)
                .arg_u32(self.cfg.kv_lora_rank as u32)
                .arg_f32(self.rms_eps)
                .arg_f32(1.0 / self.kv_scale)
                .launch(stream)?;

            // ── indexer stream, then select + gather-attend ──
            use crate::layers::glm5next_layer::profile;
            profile::end(profile::DSA_PROJ, t_proj, gpu, stream);
            let t = profile::start();
            // Replay-safe placement only while a graph is RECORDING. An eager step keeps the
            // host-offset path, so the shipping numbers and byte-identity are untouched.
            let replay_safe = ctx.graph_capture
                && meta.is_some()
                && self.select_kernels.indexer_store.0 != 0
                && self.select_kernels.write_geom.0 != 0;
            let pos_dev = if replay_safe {
                meta.map(|m| m.positions.offset(row * 4))
            } else {
                None
            };
            self.indexer_forward(gpu, hidden.offset(row * self.cfg.hidden * 2), st, pos_dev, stream)?;
            profile::end(profile::DSA_INDEXER, t, gpu, stream);

            let (q_pos_dev, bt_dev_meta, sl_dev_meta) = match meta {
                Some(m) => (
                    m.positions.offset(row * 4),
                    Some(m.block_table.offset(row * bt_stride)),
                    Some(m.seq_len.offset(row * 4)),
                ),
                None => {
                    let qp = pos as i32;
                    gpu.copy_h2d(&qp.to_le_bytes(), w.q_pos)?;
                    (w.q_pos, None, None)
                }
            };
            let (d_bt, d_sl) = match (bt_dev_meta, sl_dev_meta) {
                // The step-scoped upload already holds both; nothing to copy.
                (Some(b), Some(l)) => (b, l),
                _ => {
                    // Upload only the prefix the gather can index. The paged gather reads
                    // `block_table[pos / block_size]` for `pos` in `[0, seq_len + k)` (the
                    // `block_table.get(logical)` site above), so every entry past
                    // `(seq_len + k) / block_size + 1` is dead weight on the wire.
                    //
                    // 🔴 ANOMALIES A58: it was also a silent kill switch for the GLM drafter.
                    // `Glm5NextMtpHead::alloc_state` pre-claims its whole private pool up front
                    // (`max_seq_len / 16 + 2` entries — a mid-decode allocation inside a captured
                    // region is not an option), so at `--max-seq-len 262144` it presented 16,386
                    // entries for a 20-token sequence against a `bt_cap` of 16,384. `bt_cap` is
                    // `max_dsa_context`, a count of TOKENS used as a count of BLOCKS — the two
                    // collide at 16,384. Every propose then bailed, the drafter produced nothing,
                    // and acceptance read exactly `p1 = 0.000` while the target kept verifying
                    // correctly and emitting byte-identical output. Measured 2026-08-30: healthy
                    // at 196,608, dead at 262,144, output identical on both.
                    let bt_used = {
                        let needed = bt_entries_needed(seq_len, k, bt_block_size);
                        &block_table[..needed.min(block_table.len())]
                    };
                    let bt: Vec<u8> = bt_used.iter().flat_map(|b| b.to_le_bytes()).collect();
                    if bt_used.len() > w.bt_cap {
                        anyhow::bail!(
                            "DSA layer {}: block table needs {} entries for seq_len {} + {} rows \
                     but the persistent buffer holds {}. This is a BLOCK count against a buffer \
                     sized by max_dsa_context (a TOKEN count); do not write past the allocation.",
                            self.layer_idx,
                            bt_used.len(),
                            seq_len,
                            k,
                            w.bt_cap
                        );
                    }
                    // Persistent `w.bt`/`w.sl` instead of a `gpu.alloc` + `gpu.free` per DSA layer per
                    // token: worth a measured 1.1 ms/token (nsys 2026-08-28 — 11 x ~98 us of GPU idle
                    // for the alloc/copy/free cluster). Kill switch `ATLAS_GLM_DSA_ALLOC_PER_STEP=1`.
                    //
                    // 🪤 This was gated OFF for most of a day because turning it on changed the model's
                    // output — which turned out to be ANOMALIES A55 and not this code at all: the DSA
                    // indexer was reading 5120 bytes past `q_resid`, so the answer depended on what the
                    // allocator had put next. With that fixed the two settings are byte-identical, and
                    // the whole engine is layout-independent (verified by 4 KB poisoned guard bands on
                    // 3431 allocations producing the same completions as no guard bands at all).
                    // 🔴 ANOMALIES A65. The deferred `attend_rows` reads `seq_lens[row]` and
                    // `block_tables + row * max_blocks_per_seq`, so these two buffers outlive
                    // the row that wrote them. `sl` is `[k]` and each row writes its OWN slot;
                    // the block table is uploaded once and shared with a row stride of ZERO
                    // (see `max_blocks_per_seq` below) — the k rows ARE one sequence, so they
                    // genuinely share one table. Writing a single-row `sl`/`bt` per row left
                    // every row past the first attending over another row's (or no) memory.
                    // `bt_entries_needed(seq_len, k, ..)` does not depend on `row`, so the
                    // table's length is the same on every pass and re-uploading it is a no-op.
                    let (d_bt, d_sl) = if self.persist_bt {
                        (w.bt, w.sl)
                    } else if row == 0 {
                        (gpu.alloc(bt.len().max(4))?, gpu.alloc(k * 4)?)
                    } else {
                        // Row 0 owns the scratch; later rows write their own `sl` slot into it.
                        (attend_bt, attend_sl)
                    };
                    gpu.copy_h2d(&bt, d_bt)?;
                    gpu.copy_h2d(&((pos + 1) as i32).to_le_bytes(), d_sl.offset(row * 4))?;
                    (d_bt, d_sl)
                }
            };
            let owns_bt = bt_dev_meta.is_none();

            let paging = DsaDecodePaging {
                num_seqs: 1,
                num_q_heads: self.cfg.local_heads,
                num_kv_heads: 1,
                // 🔴 THIS IS A KERNEL ARGUMENT, so a CUDA graph BAKES IT IN at capture time.
                // `block_table.len()` grows every time the sequence crosses a block boundary,
                // so a captured graph replayed at a longer context keeps walking the capture's
                // block count — right answer for the first few tokens, wrong one after. Take
                // the metadata's ceiling, which verify_b/verify_c hold CONSTANT at
                // `self.max_blocks_per_seq` and zero-pad every uploaded row out to.
                // Without this the graphed K=3 verify diverged from eager on exactly the long
                // probes (pyadd/open128/open512) and matched on the 32-token ones.
                // 🔴 ANOMALIES A65: ZERO on the no-metadata path, which is the ROW STRIDE
                // the kernel applies to `block_tables`. All k rows of this call are the same
                // sequence and share the one table uploaded above, so a stride of 0 is the
                // correct sharing — `block_table.len()` walked row 1 off the end of a
                // single-row buffer. (The metadata path really does carry k padded rows.)
                max_blocks_per_seq: match meta {
                    Some(m) => m.max_blocks_per_seq as usize,
                    None => 0,
                },
                block_size,
                cache_stride_bytes: (block_size * self.cfg.kv_lora_rank) as u64,
            };
            if replay_safe {
                // S is exactly the `seq_len + 1` the attention metadata already holds, which is
                // `st.len()` after the indexer advance. Nothing about the pass is host-decided.
                KernelLaunch::new(gpu, self.select_kernels.write_geom)
                    .grid([1, 1, 1])
                    .block([1, 1, 1])
                    .arg_ptr(d_sl)
                    .arg_ptr(w.geom_dev)
                    .arg_u32(self.cfg.index_kpool as u32)
                    .arg_u32(self.cfg.index_topk as u32)
                    .arg_u32(super::select::topk_tile() as u32)
                    .launch(stream)?;
            }
            if batch_select {
                // `indexer_forward` left THIS row's head weights in the scalar slot; stash
                // them at row stride so the one batched pass below can read `weights[r*H]`.
                gpu.copy_d2d_async(
                    w.head_weights,
                    w.head_weights_rows.offset(row * self.cfg.index_heads * 4),
                    self.cfg.index_heads * 4,
                    stream,
                )?;
                batch_q_pos.push(pos as i32);
            } else {
                self.select_row(gpu, row, st, q_pos_dev, replay_safe, stream)?;
            }
            // The attend needs the BASE of the K-row metadata, not this row's slice: it
            // carries the row axis on `gridDim.y`. On the no-metadata path k is 1 and these
            // are the single-row `w.bt`/`w.sl`, so the base IS the row.
            if row == 0 {
                attend_bt = d_bt;
                attend_sl = d_sl;
                attend_paging = Some(paging);
                owns_scratch = owns_bt && !self.persist_bt;
            }
        }

        // ── ONE selection pass for all K rows (default; off under ...SELECT_ROWS=0) ──
        // AFTER the row loop, so every row's indexer write is already in the cache. That
        // ordering is what makes the hoist exact rather than merely cheaper: row `r` gates
        // on `end_c <= q_pos[r]`, so rows appended after it stay invisible to it, and the
        // pools this pass compresses over are the group's final set.
        if batch_select && !batch_q_pos.is_empty() {
            self.select_rows_batched(gpu, k, st, &batch_q_pos, stream)?;
        }

        // ── ONE gather-attend for all K rows ──
        if let Some(paging) = attend_paging {
            self.attend_rows(gpu, k, st, kv_cache, attend_bt, attend_sl, &paging, stream)?;
        }
        // 🔴 ANOMALIES A65: freed HERE, not in the row loop. The attend above reads both
        // buffers, so freeing them per row handed it memory that had already been released.
        if owns_scratch {
            gpu.free(attend_bt)?;
            gpu.free(attend_sl)?;
        }

        // ── output projection, row-parallel: the caller all-reduces ──
        let t_proj = profile::start();
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            w.attn_out,
            self.weights.o_absorb,
            hidden,
            k,
            self.cfg.hidden,
            self.cfg.local_heads * self.cfg.kv_lora_rank,
            stream,
        )?;
        profile::end(profile::DSA_PROJ, t_proj, gpu, stream);
        Ok(())
    }
}

impl TransformerLayer for Glm5NextDsaLayer {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(Glm5NextDsaState::alloc(gpu, &self.cfg)?))
    }

    /// The replay's writes end at `seq_len + k`; the buffer ends at `capacity`. A62.
    fn check_replay_room(&self, state: &dyn LayerState, seq_len: usize, k: usize) -> Result<()> {
        state
            .as_any()
            .downcast_ref::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?
            .ensure_room_through(seq_len + k)
            .with_context(|| {
                format!("DSA replay pre-check (before launch_graph, seq_len {seq_len} + k {k})")
            })
    }

    /// The indexer cache length is the one thing this layer keeps on the host. A replayed
    /// graph writes the next row (the store kernel reads its position from device memory)
    /// but never calls `decode`, so the counter has to be advanced here or the NEXT eager
    /// step plans its selection over a stale length — and `decode`'s own lockstep check
    /// would fire.
    fn sync_replayed_step(
        &self,
        state: &mut dyn LayerState,
        seq_len: usize,
        k: usize,
    ) -> Result<()> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?
            .sync_to(seq_len, k)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_k(
            hidden,
            1,
            state,
            kv_cache,
            seq_len,
            block_table,
            ctx,
            stream,
            // A single-token decode, never a prefill sub-chunk. Moot at k == 1, stated anyway.
            false,
        )
    }
}

#[cfg(test)]
mod tests;

/// Block-table entries the paged gather can index for `k` query rows starting at
/// `seq_len`, plus one entry of slack.
///
/// The gather reads `block_table[pos / block_size]` for `pos` in `[0, seq_len + k)`,
/// so the highest index touched is `(seq_len + k - 1) / block_size`. Everything above
/// that is never read — see the A58 note at the upload site for why uploading it
/// anyway was a silent kill switch for the GLM drafter.
pub(super) fn bt_entries_needed(seq_len: usize, k: usize, block_size: usize) -> usize {
    (seq_len + k) / block_size.max(1) + 2
}

#[cfg(test)]
mod bt_trim_tests {
    use super::bt_entries_needed;

    /// The A58 reproducer, in arithmetic: the GLM drafter pre-claims
    /// `max_seq_len / 16 + 2` entries, so at `--max-seq-len 262144` it hands 16,386
    /// against a `bt_cap` of 16,384 — for a 20-token sequence. Trimmed, it needs 3.
    #[test]
    fn a58_short_sequence_at_262k_declared_context() {
        let pool = 262_144 / 16 + 2;
        assert_eq!(pool, 16_386, "the pre-claimed pool that overran bt_cap");
        assert!(pool > 16_384, "and it is over the persistent buffer");
        assert_eq!(bt_entries_needed(20, 3, 16), 3);
        assert!(bt_entries_needed(20, 3, 16) <= 16_384);
    }

    /// Every position the gather can touch must be inside the trim.
    #[test]
    fn trim_covers_every_indexable_position() {
        for &(seq_len, k, bs) in &[
            (0usize, 1usize, 16usize),
            (1, 1, 16),
            (15, 1, 16),
            (16, 1, 16),
            (17, 4, 16),
            (4095, 4, 16),
            (131_072, 3, 16),
            (262_143, 4, 16),
            (1000, 1, 64),
        ] {
            let n = bt_entries_needed(seq_len, k, bs);
            let highest = (seq_len + k).saturating_sub(1) / bs;
            assert!(
                highest < n,
                "seq_len={seq_len} k={k} bs={bs}: highest index {highest} not < {n}"
            );
        }
    }

    /// The trim must stay far under the persistent buffer for any sequence DSA can
    /// actually select over (`max_dsa_context` = 16,384 tokens).
    #[test]
    fn trim_fits_the_persistent_buffer_across_the_dsa_window() {
        assert!(bt_entries_needed(16_384, 4, 16) <= 16_384);
    }

    /// block_size 0 must not divide by zero.
    #[test]
    fn zero_block_size_does_not_panic() {
        assert_eq!(bt_entries_needed(8, 1, 0), 11);
    }
}
