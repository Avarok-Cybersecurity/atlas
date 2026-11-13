// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the port sprint's (`port/consumer-opt`) SERVING rows of
//! `kernels/<hw>/HARDWARE.toml` `[defaults]`, as the runtime type baked into
//! [`crate::TargetDefaults::port`].
//!
//! A sub-table of [`crate::TargetDefaults`] rather than more fields on it, so
//! the sprint's rows land in files of their own: `build_defaults_port.rs`
//! parses them, `build.rs` bakes them into the same generated
//! `TARGET_DEFAULTS` constant (one generated file, one `include!`, one content
//! hash), and `spark_model::layers::ops::target_defaults_port` resolves them
//! against the environment.
//!
//! Every row is OFF (or its baseline string or number) on every target when it
//! lands. A row-off path is byte-identical to the behavior before the row
//! existed, so GB10, Hopper, B200, B300, Metal and Strix serve exactly as they
//! did.
//!
//! Rows arrive through the coordinator's fragment apply pass: each `@port:`
//! marker below is an insertion point, and text is inserted immediately BEFORE
//! the marker line.

/// The port sprint's rows of one compiled target's `[defaults]` table.
///
/// `Copy` and entirely `'static`, like [`crate::TargetDefaults`], of which it
/// is the `port` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortDefaults {
    /// When a 4-bit BF16 load route is on (dense_affine4_from_bf16,
    /// gdn_affine4_from_bf16, attn_affine4_from_bf16), every projection the
    /// Marlin kernels serve (N a multiple of 64, K of 128, more than one group)
    /// is quantized once at load by round-to-nearest int4 at group 128 (GPTQ's
    /// symmetric uint4b8 grid, --w4-clip-search shrink candidates per group)
    /// instead of the MLX affine codes, and served through W4Linear on vLLM's
    /// Marlin W4A16 GEMM (kernels/ada/common/marlin_ada.cu, host
    /// crates/spark-model/src/port/marlin/): decode and verify rows 1..64 in one
    /// launch on one thread configuration fixed per weight (row invariant),
    /// prompts on vLLM's M split.
    pub marlin_w4a16: bool,
    /// A dense Qwen3.5-family GGUF (arch qwen35) keeps its IQ4_XS, Q4_K and Q6_K
    /// projections as the file's raw super-blocks in VRAM (the dense FFN, the
    /// full-attention q/k/v/o, and the two GDN input projections with their
    /// value-head row reorder applied to the packed bytes), tagged
    /// WeightDtype::Iq4Xs / Q4K / Q6K for the sm_89 keep-packed decode GEMVs (M =
    /// 1..8) and MMQ prefill GEMMs (kernels/ada/common/kquant_mmvq_q4_ada.cu,
    /// iq4_xs_mmq_ada.cu, kquant_q8_1_ada.cu over the vendored llama.cpp MIT
    /// engine; host port/gguf_q4/), instead of expanding them to BF16 at 2 bytes
    /// a weight.
    pub gguf_native_iq4: bool,
    /// EXL3 trellis matrices (ExLlamaV3's QTIP-style codes, 1 to 8 bits a weight
    /// with mul1 half-bit steps, read by spark-runtime's weights/exl3 reader) are
    /// served from their packed codes on sm_89: the input rotated into the
    /// Hadamard domain (exl3_hadamard_ada.cu), the decode-row linear decoding
    /// each 16 x 16 tile straight into mma fragments for 1 to 128 rows with
    /// split-K partials (exl3_linear_ada.cu), and longer prefills declined to the
    /// BF16 dequant and the caller's GEMM (host port/exl3/, after ExLlamaV3 MIT
    /// via TensorFold Apache-2.0).
    pub weights_exl3: bool,
    /// On a plain-BF16 dense Qwen3.5 checkpoint (Holo 3.1 0.8B and 4B), the dense
    /// FFN's gate, up and down are quantized once at load by the GPU quantizer
    /// into MLX affine codes (a bf16 scale and bias per group of inputs;
    /// --w4-bits and --w4-group, default 4 bits at group 64, 4.5 bits a weight),
    /// each BF16 source freed right after its own quantize, and served through
    /// W4Linear: decode and verify rows 1..16 on TensorFold's lane MMA GEMV
    /// (kernels/ada/common/affine4_lane_ada.cu), prompts on the affine prompt
    /// GEMM or a BF16 dequant plus cuBLASLt
    /// (weight_loader/qwen35_dense/affine4_from_bf16.rs).
    pub dense_affine4_from_bf16: bool,
    /// On a plain-BF16 dense Qwen3.5 checkpoint, each GDN layer's `[Q|K|V|Z]`
    /// in-projection (the loader's BF16 concat) and out_proj are quantized once
    /// at load to MLX affine codes (the common --w4-bits and --w4-group, or
    /// AVAROK_W4_GDN_GROUP for these projections alone, for example 32), each
    /// BF16 source freed after its own quantize, and served through W4Linear on
    /// the same kernels (weight_loader/qwen35_dense/gdn_affine4_from_bf16.rs).
    pub gdn_affine4_from_bf16: bool,
    /// On a plain-BF16 dense Qwen3.5 checkpoint, each full-attention layer's q
    /// (doubled when gated), k, v and o are quantized once at load to MLX affine
    /// codes (--w4-bits, --w4-group), each checkpoint tensor freed after its own
    /// quantize, and served through W4Linear on the same kernels
    /// (weight_loader/qwen35_dense/attn_affine4_from_bf16.rs).
    pub attn_affine4_from_bf16: bool,
    /// The affine routes' prompt GEMM (more than 16 rows) quantizes the
    /// activation rows to int8 per group and runs the W4A8 int8 tensor-core GEMM
    /// over the same 4-bit codes (kernels/ada/common/affine_prefill_ada.cu,
    /// port::affine::prefill) instead of the BF16-activation prompt GEMM; a chunk
    /// with a non-finite or overflowing activation group falls back to BF16 and
    /// is counted.
    pub affine4_prefill_int8: bool,
    /// On a plain-BF16 dense Qwen3.5 checkpoint (Holo 3.1 0.8B and 4B), the dense
    /// FFN's gate, up and down are quantized once at load with the existing NVFP4
    /// quantizer (E2M1 codes, E4M3 per-16 scales, one F32 tensor scale; 4.5 bits
    /// a weight), each BF16 source freed right after its own quantize, and served
    /// through W4Linear over the w4a16_gemv family every CUDA target already
    /// compiles (weight_loader/qwen35_dense/nvfp4_from_bf16.rs,
    /// port/w4/nvfp4.rs): decode and verify rows 1..16 on w4a16_gemv / _batch2 /
    /// _batch3 / _batchN (row invariant by construction), prompts on w4a16_gemm
    /// or a BF16 dequant plus cuBLASLt.
    pub dense_nvfp4_from_bf16: bool,
    /// The input embedding table (one row read per token) is copied once at load
    /// into page-locked host memory and the model reads it through the region's
    /// device-visible alias, so every existing gather, the MTP head's device-side
    /// embed_from_argmax included, keeps working over PCIe, and the device copy
    /// is freed after the loader's prune step
    /// (crates/spark-model/src/port/embed_host/).
    pub embed_host_offload: bool,
    /// Batched sequence swap
    /// (`spark-model/src/model/trait_impl/sequence/state_io.rs`,
    /// `spark-runtime/src/kv_cache/block_io.rs`): a swapped-out sequence's image
    /// (each block's K and V for every attention layer, then each GDN layer's h
    /// and conv state) is gathered with batched async copies
    /// (`cuMemcpyBatchAsync` where it resolved, else one async copy per region
    /// and one event wait per chunk) into one reused page-locked staging buffer
    /// of at most one `--pinned-pool-chunk-mb` chunk, and the file write of one
    /// chunk overlaps the copy of the next; restore mirrors it.
    pub kv_batched_swap: bool,
    /// SSM tier blob store eviction order
    /// (`spark-model/src/model/ssm_tier/store.rs`): a capped host-RAM
    /// `MemBlobStore` gives up the least recently used blob (a fault-in hit and a
    /// put both make a key the most recent) instead of the oldest insert, and the
    /// model removes the tier blobs the snapshot index orphans when a fresh save
    /// re-homes a spilled prefix or the tail sweep drops one
    /// (`spark-runtime/src/radix_tree/snapshot_insert.rs`).
    pub ssm_store_lru: bool,
    /// Host-RAM KV tier (`spark-runtime/src/kv_host_tier/`, glue
    /// `spark-model/src/model/kv_tier_glue.rs`): every full prompt block the
    /// prefix cache newly holds is copied to a pinned host pool on a dedicated
    /// copy stream (one extra block reference keeps the block alive until its D2H
    /// completed), and a later prompt whose prefix the GPU cache already evicted
    /// restores those blocks by one batched H2D into fresh blocks that re-enter
    /// the radix tree, so its prefill sees an ordinary GPU hit.
    pub kv_host_tier: bool,
    /// Radix prefix-cache eviction takes its victim from an ordered (last_access,
    /// node id) index of evictable leaves, maintained at every transition of the
    /// victim predicate, instead of scanning every node once per evicted block
    /// (`spark-runtime/src/radix_tree/inner/lru_index.rs`).
    pub radix_lru_index: bool,
    /// A prefix-cache hit may end inside a KV block: the chunked prefill copies
    /// that block into a private block (every attention layer's K and V, D2D on
    /// the prefill stream before the block table is used) instead of sharing the
    /// donor's writable tail, which is the issue #1193 aliasing bug
    /// (`radix_tree/inner/cow_walk.rs`, `prefill_b/prefix_cow.rs`).
    pub prefix_cow_tail: bool,
    // @port:fields
}

/// What a target that declares no port row gets: every row's baseline.
///
/// Kept in agreement with `build_defaults_port.rs`'s `Default for PortFields`
/// by the row templates (one baseline per row, written into both files in the
/// same apply pass).
pub const PORT_BASELINE: PortDefaults = PortDefaults {
    marlin_w4a16: false,
    gguf_native_iq4: false,
    weights_exl3: false,
    dense_affine4_from_bf16: false,
    gdn_affine4_from_bf16: false,
    attn_affine4_from_bf16: false,
    affine4_prefill_int8: false,
    dense_nvfp4_from_bf16: false,
    embed_host_offload: false,
    kv_batched_swap: false,
    ssm_store_lru: false,
    kv_host_tier: false,
    radix_lru_index: false,
    prefix_cow_tail: false,
    // @port:baseline
};
