// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash **DSA (DeepSeek Sparse Attention) production surface**.
//!
//! Scoped to `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`.
//!
//! The CUDA kernels already exist and are numerically proven against HF 5.16.1 on
//! real weights — `kernels/gb10/common/dsa_indexer.cu`, gated by
//! `examples/dsa_indexer_microtest.rs` (GATE 4 kpool indexer, GATE 5 NoPE MLA over
//! the selected tokens). What was missing, and is what this module adds, is the
//! **production** surface: kernel resolution and geometry that a real layer can
//! bind, rather than an example wiring pointers by hand.
//!
//! The CPU reference in [`crate::layers::glm5next_dsa_ref`] stays the source of
//! truth for the equations. Nothing here re-derives them.
//!
//! # Shape of the pipeline
//!
//! ```text
//! k,gate,valid,ape -> kpool_compress -> pool keys/indices/valid
//!                  -> index_scores  -> [Q, P] scores + candidate validity
//!                  -> topk_pools    -> [Q, select_k] pool ids
//!                  -> expand_selection -> [Q, out_width] token ids (-1 = invalid)
//!                  -> NoPE MLA restricted to those tokens
//! ```
//!
//! # 🪤 Traps carried from Slice 8 (do not re-derive)
//!
//! * `indexer.k_norm` is a **`nn.LayerNorm`** — mean-subtracting, **with a bias** —
//!   not an RMSNorm. `indexer.k_norm.bias` existing in the checkpoint is the only
//!   tell; every other norm in GLM-5.3 is a bias-free RMSNorm.
//! * The pool softmax runs over the **pool-slot axis, per channel**, not over
//!   `head_dim` and not over pools.
//! * Pooling starts at the **first valid token**, so left padding is skipped rather
//!   than pooled.
//! * A pool counts only if **every** `kpool` slot is valid — a trailing partial pool
//!   is not a pool.
//! * **NoPE**: `qk_rope_head_dim == 0`, so the `k_rot` slice is zero-width. See the
//!   `rope > 0` guards in `qwen3_attention` — a NoPE checkpoint carries no
//!   `wkv_a_rope` and leaves `rope_theta` unset.
//! * The `-1` sentinel destination must be **fully written**. vLLM's day-0 GLM bug
//!   was a `torch.empty` top-k buffer whose tail was never written, so uninitialised
//!   memory became "token indices".

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

pub mod attend;
pub mod binding;
pub mod build;
pub mod layer;
pub mod select;
pub mod state;
pub mod tp;

/// Module name the DSA kernels resolve from. Unlisted `.cu` files take their file
/// stem as the module name, so `kernels/gb10/common/dsa_indexer.cu` is `dsa_indexer`.
pub const DSA_MODULE: &str = "dsa_indexer";

/// Module carrying the bias-bearing BF16 LayerNorm the indexer's `k_norm` needs. Lives in
/// `common/`, so every target merges it; the name is the `.cu` file stem.
pub const LAYERNORM_MODULE: &str = "nllb_encoder";

/// `#define KV_LORA_DIM` in `kernels/gb10/common/mla_paged_decode{,_fp8}.cu`.
///
/// Mirrored here so the config can refuse a checkpoint the kernel cannot read.
/// Changing the kernel without changing this constant is the bug this guards.
pub const KERNEL_KV_LORA_DIM: usize = 512;

/// `float lg[8]` in `dsa_kpool_compress` — the most pool slots the compression
/// kernel can hold. The kernel loops `s < KP && s < 8`, so a larger `index_kpool`
/// is silently truncated rather than rejected. Mirrored here so config validation
/// refuses it instead.
pub const KERNEL_MAX_KPOOL: usize = 8;

/// Every kernel the DSA path launches.
///
/// Resolved with `kernel()` (not `try_kernel`): a missing DSA entry point is a hard
/// error, never a silent fallback onto a dense-attention path. A sparse layer that
/// quietly runs dense is a correctness bug that looks like a performance bug.
#[derive(Clone, Copy)]
pub struct Glm5NextDsaKernels {
    pub kpool_compress: KernelHandle,
    pub compact_pools: KernelHandle,
    pub index_scores: KernelHandle,
    pub topk_pools: KernelHandle,
    pub expand_selection: KernelHandle,
    /// `indexer.k_norm`, which is an **`nn.LayerNorm` with a bias** — not an RMSNorm.
    ///
    /// 🪤 Do NOT reach for an RMSNorm kernel here. A `.weight`-only norm silently drops
    /// both the mean subtraction and the bias, and nothing about the shapes says so:
    /// `k_norm.weight` and `k_norm.bias` are both `[index_head_dim]`. The binder already
    /// lists the bias as REQUIRED for exactly this reason.
    ///
    /// ✅ No new kernel needed — `common/nllb_encoder.cu` already carries an in-place
    /// BF16 LayerNorm taking `(x, weight, bias, rows, dim, eps)`, and `common/` is merged
    /// into every target. Found by grepping `kernels/` before scoping a build, per the
    /// campaign's standing rule; this is the fifth thing that turned out to already exist.
    pub k_norm: KernelHandle,
    /// Derives this step's selector geometry ON DEVICE from `seq_len`, so a captured
    /// graph replays over the live context instead of the capture-time one.
    /// `try_kernel` — without it the layer keeps the host-scalar path and graphs stay off.
    pub write_geom: KernelHandle,
    /// Places the staged indexer row at a DEVICE-side position and marks it valid.
    /// The host `k_normed.offset(pos * D * 2)` it replaces was the other frozen scalar.
    pub indexer_store: KernelHandle,
    /// 🔬 ORACLE ONLY — see [`MASKED_ATTN_MAX_KEYS`].
    pub topk_to_mask: KernelHandle,
    /// 🔬 ORACLE ONLY — see [`MASKED_ATTN_MAX_KEYS`].
    pub mla_masked_attn: KernelHandle,
}

/// 🔴 `dsa_mla_masked_attn` is an **oracle**, not a serve path. Resolved from source
/// 2026-08-27; do not re-derive.
///
/// It stages the whole `[S]` score row in shared memory, so `4·S ≤ 49,152` caps it at
/// **12,288 keys** — a limit that does not shrink with sparsity, because the dense mask
/// and not the selected set sets the footprint. GLM-5.3 advertises 262,144.
///
/// The production path gathers the selected tokens through the page table instead:
///
/// * HF `transformers` 5.16.1 builds the dense `[B, Q, kv_len]` mask and sets
///   `_supports_flash_attn = False`, saying so in its own docstring — *"cannot be mapped
///   to FA without a custom kernel that can select on a per indices bases per row"*. The
///   mask is pure set membership (`scatter_add(...).ne(0)`, duplicates collapse, no
///   additive weighting), so a per-row gather is **exactly equivalent**, not an
///   approximation.
/// * vLLM ships that kernel (FlashMLA sparse / FlashInfer paged MLA), and the SM121
///   backend serving our own frozen oracle is the SM90 NoPE sparse-MLA path over a plain
///   bf16 paged cache.
///
/// ⇒ Atlas's serve path is `mla_paged_decode{,_fp8}` — block-table paged, online
/// softmax, **no `S` term in shared memory** — taught to walk a selected-index row
/// instead of `0..seq_len`. That kernel variant is NOT yet written; until it is, DSA
/// decode has no production consumer and these two handles must stay test-only.
pub const MASKED_ATTN_MAX_KEYS: usize = 12_288;

impl Glm5NextDsaKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            kpool_compress: gpu.kernel(DSA_MODULE, "dsa_kpool_compress")?,
            compact_pools: gpu.kernel(DSA_MODULE, "dsa_compact_pools")?,
            index_scores: gpu.kernel(DSA_MODULE, "dsa_index_scores")?,
            topk_pools: gpu.kernel(DSA_MODULE, "dsa_topk_pools")?,
            expand_selection: gpu.kernel(DSA_MODULE, "dsa_expand_selection")?,
            k_norm: gpu.kernel(LAYERNORM_MODULE, "nllb_layernorm_bf16")?,
            write_geom: crate::layers::try_kernel(gpu, DSA_MODULE, "dsa_write_geom"),
            indexer_store: crate::layers::try_kernel(gpu, DSA_MODULE, "dsa_indexer_store"),
            topk_to_mask: gpu.kernel(DSA_MODULE, "dsa_topk_to_mask")?,
            mla_masked_attn: gpu.kernel(DSA_MODULE, "dsa_mla_masked_attn")?,
        })
    }
}

/// DSA geometry for one layer, read from the checkpoint config — never defaulted.
///
/// Head counts are **per-rank local** for the MLA side and **full** for the indexer,
/// which is replicated. See [`tp`] for why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm5NextDsaConfig {
    pub hidden: usize,
    // ── indexer (replicated) ──
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_kpool: usize,
    pub index_topk: usize,
    pub always_select_tail: bool,
    // ── MLA ──
    /// Attention heads **this rank owns**.
    pub local_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    /// **Zero** on GLM-5.3. Kept explicit so a nonzero value is a loud change.
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    /// Longest context a sequence's indexer cache is reserved for, in tokens — the serve's
    /// `--max-seq-len`. Not a kernel limit (the top-k select is tiled); an ALLOCATION, and
    /// the biggest per-sequence one GLM-5.3 makes. See [`state::max_dsa_context`].
    pub max_context: usize,
}

impl Glm5NextDsaConfig {
    /// `config` carries per-rank-local attention head counts: `serve_phases::topology`
    /// divides `num_attention_heads` by `tp_size` before any loader runs.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        let c = Self {
            hidden: config.hidden_size,
            index_heads: config.index_n_heads,
            index_head_dim: config.index_head_dim,
            index_kpool: config.index_kpool,
            index_topk: config.index_topk,
            always_select_tail: config.index_kpool_always_select_tail,
            local_heads: config.num_attention_heads,
            q_lora_rank: config.q_lora_rank,
            kv_lora_rank: config.kv_lora_rank,
            qk_nope_head_dim: config.qk_nope_head_dim,
            qk_rope_head_dim: config.qk_rope_head_dim,
            v_head_dim: config.v_head_dim,
            // 🔴 `serve_max_seq_len` is set from `--max-seq-len` in serve_phases::topology.
            // Zero means nobody set it (a unit test, a tool) — fall back to the old fixed
            // 16,384-token reservation rather than allocating for 1 M positions.
            max_context: if config.serve_max_seq_len > 0 {
                config.serve_max_seq_len
            } else {
                16_384
            },
        };
        c.validate()?;
        Ok(c)
    }

    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    /// True when there is no RoPE section at all — GLM-5.3.
    pub fn is_nope(&self) -> bool {
        self.qk_rope_head_dim == 0
    }
    /// Pools selected per query, capped by how many pools exist.
    pub fn select_k(&self, n_pools: usize) -> usize {
        (self.index_topk / self.index_kpool).min(n_pools)
    }
    /// Width of the emitted index row; the tail adds `kpool - 1` slots.
    pub fn out_width(&self) -> usize {
        self.index_topk
            + if self.always_select_tail {
                self.index_kpool - 1
            } else {
                0
            }
    }
    /// KV latent cache width. **No rope section under NoPE**, so this is exactly
    /// `kv_lora_rank` — 512 for GLM-5.3, where DeepSeek-V4-Flash uses 576.
    pub fn kv_cache_dim(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    pub fn validate(&self) -> Result<()> {
        if self.index_kpool == 0 || self.index_topk == 0 {
            bail!(
                "DSA needs index_kpool>0 and index_topk>0; got {}/{}",
                self.index_kpool,
                self.index_topk
            );
        }
        if !self.index_topk.is_multiple_of(self.index_kpool) {
            bail!(
                "DSA: index_topk ({}) must be a multiple of index_kpool ({}) — \
                 the pool budget is index_topk/index_kpool",
                self.index_topk,
                self.index_kpool
            );
        }
        // 🔴 CORRECTED 2026-08-27 (was 64). `dsa_kpool_compress` holds the pool
        // logits in `float lg[8]` and loops `s < KP && s < 8`. A kpool in 9..=64
        // therefore pools only the FIRST 8 slots while `pool_indices`/`pool_valid`
        // are still written for all KP — a silently wrong pooled key, no crash and
        // no shape error. The prior bound of 64 admitted exactly that window. GLM's
        // kpool is 4, so nothing shipped through the gap; the guard was simply
        // describing a register budget the kernel does not have.
        if self.index_kpool > KERNEL_MAX_KPOOL {
            bail!(
                "DSA: index_kpool {} exceeds the {}-slot bound dsa_kpool_compress keeps \
                 in registers (`float lg[8]`); slots past it are silently dropped from \
                 the pooled key while still counting as valid",
                self.index_kpool,
                KERNEL_MAX_KPOOL,
            );
        }
        if self.kv_lora_rank == 0 {
            bail!("DSA is MLA: kv_lora_rank must be > 0");
        }
        // 🔴 HARD ASSERTION, tied to a kernel constant.
        //
        // `glm-5.3-flash/nvfp4/glm5next_dsa_mla_decode.cu` hardcodes
        // `#define GLM_KV_LORA_DIM 512` for the latent width, while taking the cache
        // stride (`kv_cache_dim`) as a runtime argument. GLM-5.3 is correct only
        // because its `kv_lora_rank` is ALSO 512 — a coincidence, not a design.
        //
        // A GLM revision with a different latent would read the cache at the wrong
        // width and produce plausible garbage with no crash, which is the exact
        // failure class this campaign has already paid for twice (#341, #347). Fail
        // at config time instead. If this ever fires, the fix is to parameterise
        // `GLM_KV_LORA_DIM` in GLM's own decode kernel — NOT to relax this check.
        if self.kv_lora_rank != KERNEL_KV_LORA_DIM {
            bail!(
                "GLM-5.3 DSA: kv_lora_rank is {}, but glm5next_dsa_mla_decode hardcodes                  GLM_KV_LORA_DIM={}. The decode kernel would read the latent at the                  wrong width. Parameterise GLM_KV_LORA_DIM in kernels/gb10/\
                 glm-5.3-flash/nvfp4/glm5next_dsa_mla_decode.cu before serving this checkpoint.",
                self.kv_lora_rank,
                KERNEL_KV_LORA_DIM,
            );
        }
        if self.local_heads == 0 {
            bail!("DSA: this rank owns zero attention heads");
        }
        Ok(())
    }
}
