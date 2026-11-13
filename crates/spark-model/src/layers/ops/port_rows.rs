// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the module list of the port sprint's (`port/consumer-opt`)
//! row resolvers.
//!
//! One `pub mod <row>;` per row; each row's file is
//! `layers/ops/port_rows/<row>.rs`, owned by the lane that adds the row. It
//! holds the row's `ENV` name, its `resolve` (called from
//! `target_defaults_port::resolve`) and its `row()` accessor. Lines are
//! inserted immediately BEFORE the marker below by the coordinator's fragment
//! apply pass.

pub mod affine4_prefill_int8;
pub mod attn_affine4_from_bf16;
pub mod attn_page_sparse;
pub mod conf_summary;
pub mod copy_draft;
pub mod dense_affine4_from_bf16;
pub mod dense_nvfp4_from_bf16;
pub mod embed_host_offload;
pub mod ffn_fp8_verify_gemv;
pub mod gdn_affine4_from_bf16;
pub mod gdn_woa_mtp;
pub mod gguf_native_iq4;
pub mod graph_exec_update;
pub mod kv_batched_swap;
pub mod kv_host_tier;
pub mod kv_restore_async;
pub mod kv_tier_disk;
pub mod kv_view_selection;
pub mod kv_window_desk;
pub mod l2_evict_first_weights;
pub mod lfm3_fp8_all;
pub mod marlin_w4a16;
pub mod model_swap_timing;
pub mod mtp_dcut_cost_model;
pub mod mtp_device_accept;
pub mod mtp_propose_graph;
pub mod mtp_vocab_rank;
pub mod prefill_sparse_attn;
pub mod prefix_cow_tail;
pub mod radix_lru_index;
pub mod retro_attn;
pub mod rmsnorm_fp8_quant_fused;
pub mod ssm_store_lru;
pub mod weights_exl3;
// @port:mods
