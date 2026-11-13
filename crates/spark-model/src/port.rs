// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the root of the port sprint's (`port/consumer-opt`) new
//! top-level spark-model modules.
//!
//! Declared from `lib.rs` rather than `layers/mod.rs`, which is at the
//! 500-line cap. Each lane's tree lives under `port/<name>/` and is declared
//! here; lines are inserted immediately BEFORE the marker below by the
//! coordinator's fragment apply pass.

/// The affine 4-bit weight route: MLX-affine weights quantized from BF16 at load, served on the Ada
/// lane GEMV (lanes L17a, L17b, L17c).
pub mod affine;
/// The load-time BF16 -> affine routes' serve options (bits, group, clip
/// search, int8 prompt GEMM). Lane L23a.
pub mod affine_opts;
/// The input embedding table in mapped host memory (row
/// `embed_host_offload`). Lane L24.
pub mod embed_host;
/// The EXL3 trellis weight route: ExLlamaV3-format matrices served from their packed codes on the Ada
/// decode-row linear through `W4Linear` (lane L20b, row `weights_exl3`).
pub mod exl3;
/// GDN write-on-accept for the batched verify (L10): the MTP request rule,
/// the K-generic stash geometry and the fold counter.
pub mod gdn_woa;
/// The keep-packed GGUF IQ4_XS / Q4_K / Q6_K route: the file's raw super-blocks served on the Ada mmvq
/// and MMQ kernels through `W4Linear` (lane L19, row `gguf_native_iq4`).
pub mod gguf_q4;
/// CUDA-graph cache counters and topology-signature tables (`graph_exec_update`, lane L09).
pub mod graph_stats;
/// The bounded KV view (lanes L31 and L32; row `kv_view_selection`, approximate, off).
pub mod kv_view;
pub mod logit_dump;
/// The Marlin W4A16 route: GPTQ/AWQ int4 checkpoints and opt-in round-to-nearest int4 from BF16,
/// repacked on the host and served on vLLM's Marlin GEMM (lane L18b; kernels by lane L18a).
pub mod marlin;
/// The GGUF Q6_K output head served through `W4Linear` where `kquant_moe` is
/// absent (`--lm-head-packed`). Lane L24.
pub mod packed_head;
/// Quest page-sparse decode attention (lane L30, row `attn_page_sparse`, approximate, off).
pub mod sparse_quest;
/// Lane L12: device-side speculative acceptance (row `mtp_device_accept`).
pub mod spec_accept;
/// Per-model state the port lanes add (`ModelPortState`), one field per lane.
pub mod state;
/// The packed-weight linear contract (`W4Linear`, master plan 3.7.2), its
/// shared scratch and dispatch rules. Lane L21; `w4/nvfp4.rs` is L23b's.
pub mod w4;
/// What the 4-bit (`W4Linear`) routes installed at load (`W4Role`, the
/// per-role records) and its `/metrics` series. Lane L23b.
pub mod w4_stats;
// @port:mods
