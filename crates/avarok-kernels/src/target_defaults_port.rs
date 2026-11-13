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
    /// Collect the per-request logprob confidence summary (mean top-1
    /// probability, minimum top-1 margin, fraction of tokens with top-1 below
    /// 0.5, over the first AVAROK_CONF_SUMMARY_K generated tokens, default 32)
    /// for every chat request and feed the atlas_conf_* series on /metrics.
    pub conf_summary: bool,
    /// D-Cut v2 for the batched MTP verify (`scheduler/mtp_dcut_cost.rs`):
    /// instead of keeping a fixed 0.75 of the optional draft rows, keep the
    /// number that maximizes expected tokens per second, B* = argmax_B (n + sum
    /// of survivals) / cost(2n + B), with cost(R) an online EWMA of the measured
    /// round wall per verify row count and survival taken from a logistic
    /// acceptance model of the drafter's top-1 confidence fitted while serving by
    /// damped Newton-IRLS (`scheduler/mtp_accept_estimator.rs`, after vLLM
    /// adaptive verification).
    pub mtp_dcut_cost_model: bool,
    /// When a batched decode or batched verify CUDA-graph cache is full and
    /// misses, update its least-recently-used executable in place with
    /// cuGraphExecUpdate when the new capture has the same topology, instead of
    /// destroying it and instantiating the capture
    /// (`model/trait_impl/graph_family.rs`).
    pub graph_exec_update: bool,
    /// GDN write-on-accept for the batched speculative verify.
    pub gdn_woa_mtp: bool,
    /// Which LM-head rows the MTP draft-only NVFP4 head holds.
    pub mtp_vocab_rank: &'static str,
    /// Run the MTP propose chain device-resident: each draft position after the
    /// first gathers its input embeddings on the device from the previous
    /// position's ids, and the round reads every draft back once, instead of one
    /// blocking readback per draft (per-sequence propose) or per position
    /// (batched propose).
    pub mtp_propose_graph: bool,
    /// Pick MTP verify positions on the device
    /// (kernels/ada/common/spec_accept_sample_ada.cu, lane L12) instead of
    /// copying the `[K, vocab]` BF16 verify logits to the host and running the
    /// pre-sample pipeline there.
    pub mtp_device_accept: bool,
    /// Copy drafter for MTP speculative rounds
    /// (`spark-server/src/scheduler/copy_draft/`): a per-sequence index over
    /// committed tokens (16/8/4/2-token windows, latest earlier occurrence,
    /// backward match extension, period continuation) proposes the tokens that
    /// followed an earlier copy of the context.
    pub copy_draft: &'static str,
    /// The dense FFN's FP8 speculative-verify rows (M = 2..8) on the sm_89
    /// batch-M GEMVs (kernels/ada/common/fp8_gemv_ada_batchm.cu,
    /// layers/dense_ffn_ada_kernels/verify.rs): one pass over each FP8 weight for
    /// all M rows, through the same down arm the M=1 decode forward takes,
    /// instead of the prefill ladder (w8a16_gemv_batch4 at M <= 4, the padded
    /// tile GEMMs or the lossy W8A8 arm at M 5..8).
    pub ffn_fp8_verify_gemv: bool,
    /// The sm_89 FP8 decode and verify GEMVs load their weights with an explicit
    /// evict-first L2 policy (createpolicy.fractional.L2::evict_first on
    /// ld.global.nc.L1::no_allocate.L2::cache_hint; the _l2ef twins in
    /// kernels/ada/common/fp8_gemv_ada.cu and fp8_gemv_ada_batchm.cu, after
    /// Cinference, Apache-2.0), so the once-read weight stream stops displacing
    /// the activations, scales and the next layer's GDN state from the 32 MB L2.
    pub l2_evict_first_weights: bool,
    /// The prefill FFN-norm of a Qwen3 / Qwen3.5 layer (residual add + offset
    /// RMSNorm) also writes the dense FFN's W8A8 gate/up activation, E4M3 codes
    /// and 1x128 FP32 scales, in the same launch
    /// (kernels/ada/common/rmsnorm_quant_fp8_ada.cu,
    /// layers/dense_ffn_w8a8_prefill/norm_fused.rs), so per_token_group_quant_fp8
    /// and its re-read of the normed rows go.
    pub rmsnorm_fp8_quant_fused: bool,
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
    /// Asynchronous host-tier KV restore (spark-server
    /// scheduler/restore_queue.rs, lane L27): a new request whose prefix the host
    /// KV tier (kv_host_tier) restores is parked outside the batch while its
    /// host-to-device copy runs on the tier's copy stream, every other sequence
    /// keeps its decode ticks, and the request re-enters the front of the pending
    /// queue at the top of the tick after the copy finished, then admits as an
    /// HBM prefix hit.
    pub kv_restore_async: bool,
    /// Disk tier and warm restart for the host KV tier (spark-runtime
    /// kv_host_tier/disk.rs, lane L27): blocks the host tier evicts are demoted
    /// to one record file each under --kv-tier-dir (temp file plus atomic rename,
    /// O_DIRECT when aligned with a logged buffered fallback, budget
    /// --kv-tier-disk-gb, least recently used first), a later lookup promotes
    /// them back after checking the record's checksums, and a restarted server
    /// adopts the records a previous run wrote for the same model fingerprint.
    pub kv_tier_disk: bool,
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
    /// Quest page-sparse decode attention (`spark-model/src/port/sparse_quest/`,
    /// `kernels/ada/common/attn_page_sparse_ada.cu`), plugged into the
    /// sparse-attention seam: per attention layer, 64-token page and KV head the
    /// channel min and max of the keys (f16, kept in the slot of the page's first
    /// physical block, charged to the KV budget at model build), and per decode
    /// or verify step the Quest bound `sum_c max(q_c*kmin_c, q_c*kmax_c)` of
    /// every page; the step attends only to the first `--attn-sparse-sink-pages`
    /// pages, the `--attn-sparse-topk-pages` best pages and the last
    /// `--attn-sparse-recent-tokens` tokens through a compacted block table, once
    /// the context passes `--attn-sparse-min-ctx` (below it the view is the dense
    /// table, bit-identical).
    pub attn_page_sparse: bool,
    /// Bounded KV view (APPROXIMATE; `spark-model/src/port/kv_view/`, core
    /// `spark-runtime/src/kv_select/` and `kv_cache/sketch.rs`, kernels
    /// `kv_view_table_ada.cu`, `kv_page_sketch_ada.cu`, `kv_page_score_ada.cu`;
    /// KVMem's idea, arXiv:2609.04852, implemented independently): once per user
    /// turn, at the end of the final user message's prefill, the context's
    /// 128-token blocks are scored against that message's last query rows
    /// (page-softmax over a per-page pre-RoPE mean-key sketch) and the
    /// full-attention layers' decode and verify attention then reads only the
    /// view (sink, the recent window, the question and the best blocks within
    /// `--kv-view-budget-tokens`, plus every token appended after), and so do the
    /// paged prefill chunks of the conversation's next turn; GDN layers and the
    /// MTP head stay dense, the first output token comes from the full prefill.
    pub kv_view_selection: bool,
    /// RetroInfer retrieval attention (spark-model port/retro, lanes L33 and
    /// L34): at or above --retro-min-context tokens, each full-attention layer's
    /// decode attends exactly to the first --retro-sink tokens, the recent tail
    /// and the tokens of the few highest-scoring clusters of a segmented
    /// spherical k-means index over its keys (retrieval zone,
    /// --retro-retrieval-budget of the clusters), estimates the next clusters
    /// through their centroids and value sums (--retro-estimation-budget), and
    /// drops the rest; the index is built once per sequence, grown every
    /// --retro-update-interval generated tokens and reused by sequences that
    /// share a prefix.
    pub retro_attn: bool,
    /// APPROXIMATE block-sparse prefill attention, XAttention-style
    /// (`kernels/ada/common/prefill_block_sparse_ada.cu`,
    /// `spark-model/src/port/sparse_prefill/`).
    pub prefill_sparse_attn: bool,
    /// Host-resident window desk (`spark-model/src/port/kv_window/`, page API
    /// `spark-runtime/src/kv_host_tier/pages.rs`, view kernel
    /// `kv_page_fetch_ada.cu`): APPROXIMATE attention for contexts past what VRAM
    /// holds.
    pub kv_window_desk: bool,
    /// Convert an LFM3 checkpoint's projections to 128x128 block-scaled FP8
    /// (E4M3) at load: the dense FFN (w1, w3, w2), the attention q/k/v/o (their
    /// BF16 is freed) and the gated short convolution's in_proj and out_proj
    /// (E4M3 decode copies; the BF16 stays for prefill, +0.22 GB).
    pub lfm3_fp8_all: bool,
    /// Time every hot-swap by phase (drain, teardown, load, total) and keep the
    /// last 8 swap durations per model, feed the atlas_model_swap_seconds{phase},
    /// atlas_model_swap_total{outcome}, atlas_model_swap_estimate_seconds{model}
    /// and atlas_model_resident_bytes series on /metrics, and add x_atlas
    /// (resident, resident_bytes, swap_seconds_estimate) to the /v1/models list
    /// and to a retrieved model.
    pub model_swap_timing: bool,
    /// Multi-sequence DECODE (2 to 8 streams) serves a native-FP8 dense FFN on
    /// the sm_89 batch-M GEMVs (kernels/ada/common/fp8_gemv_ada_batchm.cu) in
    /// both layer kinds, instead of the prefill ladder (w8a16_gemv_batch4 at
    /// 2..4 streams, the W8A8 arm at 5..8), on decode steps only
    /// (layers/dense_ffn/fp8_decode_gemv.rs).
    pub ffn_fp8_decode_gemv: bool,
    /// MTP by live batch width (spark-server scheduler/mtp_width_policy.rs):
    /// the throughput gate keeps one set of baselines and one standing decision
    /// per batch width, counts its re-probe and refresh intervals in tokens per
    /// stream, and backs off re-probes that confirm the standing decision.
    pub mtp_width_policy: bool,
    /// The MTP-verify GDN projections of a BF16-kept GDN layer on
    /// dense_gemv_bf16_batchm, 16 rows a weight pass
    /// (layers/ops/gdn_verify_gemv.rs): in_proj_qkvz at 3 or more rows and
    /// out_proj_dense at 2 or more, instead of one dense_gemv per row or the
    /// scalar dense_gemm. Every row is bit-identical to the M=1 decode GEMV.
    pub gdn_verify_gemv: bool,
    /// A BF16 LM head's speculative-verify rows above 8 on
    /// dense_gemv_bf16_batchm, 16 rows a weight pass
    /// (layers/ops/lm_head_verify_gemv.rs), instead of the scalar dense_gemm.
    /// Every row is bit-identical to the M=1 decode head.
    pub lm_head_verify_gemv: bool,
    /// A greedy decode step whose sequences still have a min_tokens floor to
    /// reach stays on the device argmax (spark-server
    /// scheduler/min_floor_fast.rs): the floor's EOS mask is checked against
    /// the argmax afterwards, and only a step whose argmax is an EOS id is
    /// redone on the host. No per-step logits copy; the pick's tie-break
    /// becomes the device argmax's, as for a request without a floor, so an
    /// exact top tie can resolve differently from the row off.
    pub min_tokens_gpu_argmax: bool,
    /// A native-FP8 dense FFN over 5 to 16 rows (a decode step of 5 to 8
    /// streams, a speculative verify of up to 16 rows) on the sm_89 16-row
    /// tensor-core GEMM (kernels/ada/common/fp8_gemm_ada_m16.cu): BF16
    /// activations, E4M3 weights widened in registers, one weight pass per
    /// projection (layers/dense_ffn/fp8_m16_tc.rs). It reassociates the K
    /// reduction, so a row is within 2 BF16 ULP of the decode GEMV, not
    /// bit-identical to it.
    pub ffn_fp8_m16_tc: bool,
    // @port:fields
}

/// What a target that declares no port row gets: every row's baseline.
///
/// Kept in agreement with `build_defaults_port.rs`'s `Default for PortFields`
/// by the row templates (one baseline per row, written into both files in the
/// same apply pass).
pub const PORT_BASELINE: PortDefaults = PortDefaults {
    conf_summary: false,
    mtp_dcut_cost_model: false,
    graph_exec_update: false,
    gdn_woa_mtp: false,
    mtp_vocab_rank: "id",
    mtp_propose_graph: false,
    mtp_device_accept: false,
    copy_draft: "off",
    ffn_fp8_verify_gemv: false,
    l2_evict_first_weights: false,
    rmsnorm_fp8_quant_fused: false,
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
    kv_restore_async: false,
    kv_tier_disk: false,
    radix_lru_index: false,
    prefix_cow_tail: false,
    attn_page_sparse: false,
    kv_view_selection: false,
    retro_attn: false,
    prefill_sparse_attn: false,
    kv_window_desk: false,
    lfm3_fp8_all: false,
    model_swap_timing: false,
    ffn_fp8_decode_gemv: false,
    mtp_width_policy: false,
    gdn_verify_gemv: false,
    lm_head_verify_gemv: false,
    min_tokens_gpu_argmax: false,
    ffn_fp8_m16_tc: false,
    // @port:baseline
};
