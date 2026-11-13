// SPDX-License-Identifier: AGPL-3.0-only
// The row lines below are generated from one template (plan 3.4) that has the same
// `assert_eq!(<row>, <value>)` shape for bool and string rows, so every row reads alike.
#![allow(clippy::bool_assert_comparison)]

//! Responsibility: the port sprint's (`port/consumer-opt`) rows of the
//! CHECKED-IN `kernels/<hw>/HARDWARE.toml` `[defaults]` tables, parsed with the
//! real build-script parser (`build_defaults_port.rs`, declared inside
//! `build_defaults.rs`).
//!
//! The same four questions `tests/target_defaults.rs` asks of the pre-sprint
//! rows, asked of the port rows: every declaring target states every row; the
//! baked constant matches its own tree; ada holds its write-time values; every
//! other target holds the baseline. The last one is the proof that GB10,
//! Hopper, B200 and B300 serve unchanged.
//!
//! Rows arrive through the coordinator's fragment apply pass, which inserts
//! text immediately BEFORE each `@port:` marker line.

#[allow(dead_code)]
#[path = "../build_defaults.rs"]
mod build_defaults;

use build_defaults::build_defaults_port::PortFields;
use build_defaults::{Defaults, read_defaults};

use std::path::PathBuf;

/// Every NVIDIA target that carries a `[defaults]` table (the same list as
/// `tests/target_defaults.rs`).
const DECLARING: &[&str] = &["gb10", "hopper", "b200", "b300", "ada"];

/// Every port row, by name. Each must be declared explicitly by every
/// DECLARING target, in the port block at the end of its `[defaults]`.
const PORT_LEVERS: &[&str] = &[
    "conf_summary",
    "mtp_dcut_cost_model",
    "graph_exec_update",
    "gdn_woa_mtp",
    "mtp_vocab_rank",
    "mtp_propose_graph",
    "mtp_device_accept",
    "copy_draft",
    "ffn_fp8_verify_gemv",
    "l2_evict_first_weights",
    "rmsnorm_fp8_quant_fused",
    "marlin_w4a16",
    "gguf_native_iq4",
    "weights_exl3",
    "dense_affine4_from_bf16",
    "gdn_affine4_from_bf16",
    "attn_affine4_from_bf16",
    "affine4_prefill_int8",
    "dense_nvfp4_from_bf16",
    "embed_host_offload",
    "kv_batched_swap",
    "ssm_store_lru",
    "kv_host_tier",
    "kv_restore_async",
    "kv_tier_disk",
    "radix_lru_index",
    "prefix_cow_tail",
    "attn_page_sparse",
    "kv_view_selection",
    "retro_attn",
    // @port:levers
];

fn kernels_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/avarok-kernels is two levels below the workspace root")
        .join("kernels")
}

fn declared(hw: &str) -> Defaults {
    read_defaults(&kernels_root(), hw)
}

/// Every declaring target states every port row, inside its port block (after
/// the block header, before the `# @port:defaults` marker), so an absent row
/// and a deliberate `false` never look identical.
#[test]
fn every_declaring_target_states_every_port_lever() {
    for hw in DECLARING {
        let raw = std::fs::read_to_string(kernels_root().join(hw).join("HARDWARE.toml"))
            .unwrap_or_else(|e| panic!("kernels/{hw}/HARDWARE.toml: {e}"));
        let head = raw
            .find("# ── Port sprint rows (port/consumer-opt) ──")
            .unwrap_or_else(|| panic!("kernels/{hw}/HARDWARE.toml has no port block header"));
        let marker = raw
            .find("\n# @port:defaults")
            .unwrap_or_else(|| panic!("kernels/{hw}/HARDWARE.toml has no `# @port:defaults`"));
        assert!(
            head < marker,
            "kernels/{hw}: the port block header must precede its marker"
        );
        let block = &raw[head..marker + 1];
        for lever in PORT_LEVERS {
            assert!(
                block.contains(&format!("\n{lever} = ")),
                "kernels/{hw}/HARDWARE.toml must declare the port row `{lever}` \
                 explicitly in its port block, not inherit it from the baseline"
            );
        }
    }
}

/// The port rows this BINARY was built with are the ones its own hardware tree
/// declares.
#[test]
fn the_baked_port_rows_match_their_own_hardware_tree() {
    let baked = avarok_kernels::TARGET_DEFAULTS;
    let declared = read_defaults(&kernels_root(), baked.hw);
    assert_eq!(baked.hw, declared.hw);
    assert_eq!(baked.port.conf_summary, declared.port.conf_summary);
    assert_eq!(
        baked.port.mtp_dcut_cost_model,
        declared.port.mtp_dcut_cost_model
    );
    assert_eq!(
        baked.port.graph_exec_update,
        declared.port.graph_exec_update
    );
    assert_eq!(baked.port.gdn_woa_mtp, declared.port.gdn_woa_mtp);
    assert_eq!(baked.port.mtp_vocab_rank, declared.port.mtp_vocab_rank);
    assert_eq!(
        baked.port.mtp_propose_graph,
        declared.port.mtp_propose_graph
    );
    assert_eq!(
        baked.port.mtp_device_accept,
        declared.port.mtp_device_accept
    );
    assert_eq!(baked.port.copy_draft, declared.port.copy_draft);
    assert_eq!(
        baked.port.ffn_fp8_verify_gemv,
        declared.port.ffn_fp8_verify_gemv
    );
    assert_eq!(
        baked.port.l2_evict_first_weights,
        declared.port.l2_evict_first_weights
    );
    assert_eq!(
        baked.port.rmsnorm_fp8_quant_fused,
        declared.port.rmsnorm_fp8_quant_fused
    );
    assert_eq!(baked.port.marlin_w4a16, declared.port.marlin_w4a16);
    assert_eq!(baked.port.gguf_native_iq4, declared.port.gguf_native_iq4);
    assert_eq!(baked.port.weights_exl3, declared.port.weights_exl3);
    assert_eq!(
        baked.port.dense_affine4_from_bf16,
        declared.port.dense_affine4_from_bf16
    );
    assert_eq!(
        baked.port.gdn_affine4_from_bf16,
        declared.port.gdn_affine4_from_bf16
    );
    assert_eq!(
        baked.port.attn_affine4_from_bf16,
        declared.port.attn_affine4_from_bf16
    );
    assert_eq!(
        baked.port.affine4_prefill_int8,
        declared.port.affine4_prefill_int8
    );
    assert_eq!(
        baked.port.dense_nvfp4_from_bf16,
        declared.port.dense_nvfp4_from_bf16
    );
    assert_eq!(
        baked.port.embed_host_offload,
        declared.port.embed_host_offload
    );
    assert_eq!(baked.port.kv_batched_swap, declared.port.kv_batched_swap);
    assert_eq!(baked.port.ssm_store_lru, declared.port.ssm_store_lru);
    assert_eq!(baked.port.kv_host_tier, declared.port.kv_host_tier);
    assert_eq!(baked.port.kv_restore_async, declared.port.kv_restore_async);
    assert_eq!(baked.port.kv_tier_disk, declared.port.kv_tier_disk);
    assert_eq!(baked.port.radix_lru_index, declared.port.radix_lru_index);
    assert_eq!(baked.port.prefix_cow_tail, declared.port.prefix_cow_tail);
    assert_eq!(baked.port.attn_page_sparse, declared.port.attn_page_sparse);
    assert_eq!(
        baked.port.kv_view_selection,
        declared.port.kv_view_selection
    );
    assert_eq!(baked.port.retro_attn, declared.port.retro_attn);
    // @port:baked-eq
}

/// Ada's port rows hold their write-time values (the baseline, in this sprint,
/// until a GPU receipt flips one).
#[test]
fn ada_port_rows_hold_their_write_time_values() {
    let d = declared("ada");
    assert_eq!(d.hw, "ada");
    assert_eq!(declared("ada").port.conf_summary, false);
    assert_eq!(declared("ada").port.mtp_dcut_cost_model, false);
    assert_eq!(declared("ada").port.graph_exec_update, false);
    assert_eq!(declared("ada").port.gdn_woa_mtp, false);
    assert_eq!(declared("ada").port.mtp_vocab_rank, "id");
    assert_eq!(declared("ada").port.mtp_propose_graph, false);
    assert_eq!(declared("ada").port.mtp_device_accept, false);
    assert_eq!(declared("ada").port.copy_draft, "off");
    assert_eq!(declared("ada").port.ffn_fp8_verify_gemv, false);
    assert_eq!(declared("ada").port.l2_evict_first_weights, false);
    assert_eq!(declared("ada").port.rmsnorm_fp8_quant_fused, false);
    assert_eq!(declared("ada").port.marlin_w4a16, false);
    assert_eq!(declared("ada").port.gguf_native_iq4, false);
    assert_eq!(declared("ada").port.weights_exl3, false);
    assert_eq!(declared("ada").port.dense_affine4_from_bf16, false);
    assert_eq!(declared("ada").port.gdn_affine4_from_bf16, false);
    assert_eq!(declared("ada").port.attn_affine4_from_bf16, false);
    assert_eq!(declared("ada").port.affine4_prefill_int8, false);
    assert_eq!(declared("ada").port.dense_nvfp4_from_bf16, false);
    assert_eq!(declared("ada").port.embed_host_offload, false);
    assert_eq!(declared("ada").port.kv_batched_swap, false);
    assert_eq!(declared("ada").port.ssm_store_lru, false);
    assert_eq!(declared("ada").port.kv_host_tier, false);
    assert_eq!(declared("ada").port.kv_restore_async, false);
    assert_eq!(declared("ada").port.kv_tier_disk, false);
    assert_eq!(declared("ada").port.radix_lru_index, false);
    assert_eq!(declared("ada").port.prefix_cow_tail, false);
    assert_eq!(declared("ada").port.attn_page_sparse, false);
    assert_eq!(declared("ada").port.kv_view_selection, false);
    assert_eq!(declared("ada").port.retro_attn, false);
    // @port:ada
}

/// Every non-ada declaring target keeps every port row at its baseline.
#[test]
fn non_ada_targets_keep_every_port_row_at_its_baseline() {
    for hw in ["gb10", "hopper", "b200", "b300"] {
        assert_eq!(declared(hw).hw, hw);
        assert!(!declared(hw).port.conf_summary, "{hw}");
        assert!(!declared(hw).port.mtp_dcut_cost_model, "{hw}");
        assert!(!declared(hw).port.graph_exec_update, "{hw}");
        assert!(!declared(hw).port.gdn_woa_mtp, "{hw}");
        assert_eq!(declared(hw).port.mtp_vocab_rank, "id", "{hw}");
        assert!(!declared(hw).port.mtp_propose_graph, "{hw}");
        assert!(!declared(hw).port.mtp_device_accept, "{hw}");
        assert_eq!(declared(hw).port.copy_draft, "off", "{hw}");
        assert!(!declared(hw).port.ffn_fp8_verify_gemv, "{hw}");
        assert!(!declared(hw).port.l2_evict_first_weights, "{hw}");
        assert!(!declared(hw).port.rmsnorm_fp8_quant_fused, "{hw}");
        assert!(!declared(hw).port.marlin_w4a16, "{hw}");
        assert!(!declared(hw).port.gguf_native_iq4, "{hw}");
        assert!(!declared(hw).port.weights_exl3, "{hw}");
        assert!(!declared(hw).port.dense_affine4_from_bf16, "{hw}");
        assert!(!declared(hw).port.gdn_affine4_from_bf16, "{hw}");
        assert!(!declared(hw).port.attn_affine4_from_bf16, "{hw}");
        assert!(!declared(hw).port.affine4_prefill_int8, "{hw}");
        assert!(!declared(hw).port.dense_nvfp4_from_bf16, "{hw}");
        assert!(!declared(hw).port.embed_host_offload, "{hw}");
        assert!(!declared(hw).port.kv_batched_swap, "{hw}");
        assert!(!declared(hw).port.ssm_store_lru, "{hw}");
        assert!(!declared(hw).port.kv_host_tier, "{hw}");
        assert!(!declared(hw).port.kv_restore_async, "{hw}");
        assert!(!declared(hw).port.kv_tier_disk, "{hw}");
        assert!(!declared(hw).port.radix_lru_index, "{hw}");
        assert!(!declared(hw).port.prefix_cow_tail, "{hw}");
        assert!(!declared(hw).port.attn_page_sparse, "{hw}");
        assert!(!declared(hw).port.kv_view_selection, "{hw}");
        assert!(!declared(hw).port.retro_attn, "{hw}");
        // @port:non-ada
    }
}

/// A target with no `[defaults]` table (metal, strix, strix-hip) gets every
/// port row's baseline, exactly as it gets every other row's.
#[test]
fn a_target_that_declares_nothing_gets_the_port_baseline() {
    for hw in ["metal", "strix", "strix-hip"] {
        assert_eq!(declared(hw).port, PortFields::default(), "{hw}");
    }
}
