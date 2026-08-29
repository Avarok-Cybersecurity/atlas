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

use anyhow::{Result, bail};
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
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextDsaConfig) -> Result<Self> {
        // Sized at the DSA context cap so a growing sequence never reallocates.
        let geom =
            super::select::DsaSelectGeometry::plan(cfg, super::state::max_dsa_context(cfg), 1)?;
        let bt_cap = super::state::max_dsa_context(cfg).max(1);
        let persist = std::env::var("ATLAS_GLM_DSA_ALLOC_PER_STEP").as_deref() != Ok("1");
        Ok(Self {
            q_a: gpu.alloc(cfg.q_lora_rank * 2)?,
            q_resid: gpu.alloc(cfg.q_lora_rank * 2)?,
            q_abs: gpu.alloc(cfg.local_heads * cfg.kv_lora_rank * 2)?,
            kv_a: gpu.alloc(cfg.kv_lora_rank * 2)?,
            q_idx: gpu.alloc(cfg.index_heads * cfg.index_head_dim * 4)?,
            head_weights: gpu.alloc(cfg.index_heads * 4)?,
            q_pos: gpu.alloc(4)?,
            q_mask: {
                // Decode always presents one real query position. Set ONCE — writing it per
                // token cost a blocking H2D per DSA layer and made the step uncapturable.
                let p = gpu.alloc(1)?;
                gpu.memset_async(p, 1, 1, 0)?;
                gpu.synchronize(0)?;
                p
            },
            slot: gpu.alloc(8)?,
            attn_out: gpu.alloc(cfg.local_heads * cfg.kv_lora_rank * 2)?,
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
            sl: if persist { gpu.alloc(4)? } else { DevicePtr(0) },
            bt_cap,
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

    /// Everything after the indexer write: selector inputs, selection, gather-attend.
    /// Leaves `[local_heads, kv_lora_rank]` BF16 in the workspace's `attn_out`.
    fn select_and_attend(
        &self,
        gpu: &dyn GpuBackend,
        state: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        block_table_dev: DevicePtr,
        seq_lens_dev: DevicePtr,
        q_pos_dev: DevicePtr,
        paging: &DsaDecodePaging,
        replay_safe: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let w = &self.workspace;
        let geom = state.geometry(&self.cfg, 1)?;

        // Selector Q and head weights, FP32 straight out of the GEMM.
        gemm(
            gpu,
            self.kernels.gemm_f32,
            self.kernels.gemv_f32,
            // No FP32-out batchm twin exists; the selector's two sites stay on gemv/tile.
            KernelHandle(0),
            w.q_resid,
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
        select_tokens(
            gpu,
            &self.select_kernels,
            &self.cfg,
            &geom,
            &inputs,
            &w.select,
            launch,
            stream,
        )?;

        use crate::layers::glm5next_layer::profile;
        profile::end(profile::DSA_SELECT, t, gpu, stream);
        let t = profile::start();
        let pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        decode_attention(
            gpu,
            self.decode_kernel,
            &self.cfg,
            &geom,
            paging,
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
        Ok(w.attn_out)
    }
}

impl TransformerLayer for Glm5NextDsaLayer {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(Glm5NextDsaState::alloc(gpu, &self.cfg)?))
    }

    /// The indexer cache length is the one thing this layer keeps on the host. A replayed
    /// graph writes the next row (the store kernel reads its position from device memory)
    /// but never calls `decode`, so the counter has to be advanced here or the NEXT eager
    /// step plans its selection over a stale length — and `decode`'s own lockstep check
    /// would fire.
    fn advance_replayed_step(&self, state: &mut dyn LayerState) -> Result<()> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?
            .advance(1)
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
        let st = state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?;
        if st.len() != seq_len {
            bail!(
                "DSA layer {}: indexer cache holds {} tokens but the sequence is at {}. \
                 The indexer stream must advance in lockstep with the KV cache — a drift \
                 selects over the wrong context.",
                self.layer_idx,
                st.len(),
                seq_len
            );
        }
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
            1,
            self.cfg.q_lora_rank,
            self.cfg.hidden,
            stream,
        )?;
        // 🪤 vanilla: x * rms * w, no `1 +`.
        KernelLaunch::new(gpu, self.kernels.rms_norm)
            .grid([1, 1, 1])
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
            1,
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
            1,
            self.cfg.kv_lora_rank,
            self.cfg.hidden,
            stream,
        )?;
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
            None
        };
        let slot_dev = match meta {
            Some(m) => m.slot,
            None => {
                let logical = seq_len / block_size;
                let physical = *block_table.get(logical).ok_or_else(|| {
                    anyhow::anyhow!(
                        "DSA layer {}: block table has {} entries, needs logical block \
                         {logical} for position {seq_len}",
                        self.layer_idx,
                        block_table.len()
                    )
                })? as usize;
                let slot = (physical * block_size + seq_len % block_size) as i64;
                gpu.copy_h2d(&slot.to_le_bytes(), w.slot)?;
                w.slot
            }
        };
        KernelLaunch::new(gpu, self.kernels.latent_write)
            .grid([1, 1, 1])
            .block([self.cfg.kv_lora_rank as u32, 1, 1])
            .arg_ptr(w.kv_a)
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
            meta.map(|m| m.positions)
        } else {
            None
        };
        self.indexer_forward(gpu, hidden, st, pos_dev, stream)?;
        profile::end(profile::DSA_INDEXER, t, gpu, stream);

        let (q_pos_dev, bt_dev_meta, sl_dev_meta) = match meta {
            Some(m) => (m.positions, Some(m.block_table), Some(m.seq_len)),
            None => {
                let pos = seq_len as i32;
                gpu.copy_h2d(&pos.to_le_bytes(), w.q_pos)?;
                (w.q_pos, None, None)
            }
        };
        let (d_bt, d_sl) = match (bt_dev_meta, sl_dev_meta) {
            // The step-scoped upload already holds both; nothing to copy.
            (Some(b), Some(l)) => (b, l),
            _ => {
                let bt: Vec<u8> = block_table.iter().flat_map(|b| b.to_le_bytes()).collect();
                if block_table.len() > w.bt_cap {
                    anyhow::bail!(
                        "DSA layer {}: block table {} entries exceeds the {}-entry persistent \
                 buffer; raise max_dsa_context, do not write past the allocation.",
                        self.layer_idx,
                        block_table.len(),
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
                let (d_bt, d_sl) = if self.persist_bt {
                    (w.bt, w.sl)
                } else {
                    (gpu.alloc(bt.len().max(4))?, gpu.alloc(4)?)
                };
                gpu.copy_h2d(&bt, d_bt)?;
                gpu.copy_h2d(&((seq_len + 1) as i32).to_le_bytes(), d_sl)?;
                (d_bt, d_sl)
            }
        };
        let owns_bt = bt_dev_meta.is_none();

        let paging = DsaDecodePaging {
            num_seqs: 1,
            num_q_heads: self.cfg.local_heads,
            num_kv_heads: 1,
            max_blocks_per_seq: block_table.len(),
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
                .launch(stream)?;
        }
        let attn = self.select_and_attend(
            gpu,
            st,
            kv_cache,
            d_bt,
            d_sl,
            q_pos_dev,
            &paging,
            replay_safe,
            stream,
        )?;
        if owns_bt && !self.persist_bt {
            gpu.free(d_bt)?;
            gpu.free(d_sl)?;
        }

        // ── output projection, row-parallel: the caller all-reduces ──
        let t_proj = profile::start();
        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            attn,
            self.weights.o_absorb,
            hidden,
            1,
            self.cfg.hidden,
            self.cfg.local_heads * self.cfg.kv_lora_rank,
            stream,
        )?;
        profile::end(profile::DSA_PROJ, t_proj, gpu, stream);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
