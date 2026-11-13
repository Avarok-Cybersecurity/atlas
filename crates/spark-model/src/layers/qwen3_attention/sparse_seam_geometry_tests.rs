// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU oracles of the sparse-attention seam's bookkeeping
//! (port lane L29): how batched rows group into sequences, how published ids
//! map onto them, the prefill stream geometry and write floor, the startup
//! refusals, the KV-budget charge, the metrics text, the identity toggle and
//! the served KV dtypes.

use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::KvCacheDtype;

use super::identity::enabled_by;
use super::metrics::{Counters, PlanKind, render_with};
use super::route::{assign_ids, group_rows};
use super::sites::prefill_rows;
use super::tests::Recorder;
use super::{
    KvDtypeTag, PAD_SEQ_ID, PoolGeometry, ServeFacts, charge_pool_with, charge_with, validate_with,
};
use crate::layer::BatchedAttnMetadata;

#[test]
fn batched_decode_rows_are_one_sequence_each() {
    let g = group_rows(&[10, 11, 40], &[vec![1, 2], vec![3, 4], vec![5, 6, 7]]);
    assert_eq!(g.rows_per_seq, 1);
    assert_eq!(g.host_seq_lens, vec![11, 12, 41]);
    assert_eq!(g.first_pos, vec![10, 11, 40]);
    assert_eq!(g.new_tokens, vec![1, 1, 1]);
    assert_eq!(g.padding, vec![false; 3]);
}

#[test]
fn verify_rows_of_one_sequence_group_together() {
    let t = vec![4u32, 5, 6];
    let g = group_rows(&[30, 31, 32], &[t.clone(), t.clone(), t]);
    assert_eq!(g.rows_per_seq, 3);
    assert_eq!(g.first_row, vec![0]);
    assert_eq!(g.first_pos, vec![30]);
    assert_eq!(g.new_tokens, vec![3]);
    assert_eq!(g.host_seq_lens, vec![31, 32, 33]);
}

#[test]
fn batched_verify_with_padding_groups_by_k() {
    let (a, b, pad) = (vec![1u32], vec![2u32], vec![99u32]);
    let tables = [a.clone(), a, b.clone(), b, pad.clone(), pad];
    let g = group_rows(&[7, 8, 20, 21, 0, 0], &tables);
    assert_eq!(g.rows_per_seq, 2);
    assert_eq!(g.first_row, vec![0, 2, 4]);
    assert_eq!(g.padding, vec![false, false, true]);
    assert_eq!(g.first_pos, vec![7, 20, 0]);
    assert_eq!(g.new_tokens, vec![2, 2, 0]);
    assert_eq!(g.host_seq_lens, vec![8, 9, 21, 22, 0, 0]);
}

#[test]
fn unequal_verify_widths_fall_back_to_one_row_per_sequence() {
    let (a, b) = (vec![1u32], vec![2u32]);
    let tables = [a.clone(), a.clone(), a, b.clone(), b];
    let g = group_rows(&[7, 8, 9, 20, 21], &tables);
    assert_eq!(g.rows_per_seq, 1);
    assert_eq!(g.first_row.len(), 5);
    // A padding run that is not a multiple of K also falls back.
    let g = group_rows(&[7, 8, 0], &[vec![1], vec![1], vec![9]]);
    assert_eq!(g.rows_per_seq, 1);
}

#[test]
fn published_ids_map_onto_real_groups_and_padding_gets_the_pad_id() {
    assert_eq!(
        assign_ids(&[11, 22], &[false, true, false]),
        vec![11, PAD_SEQ_ID, 22]
    );
    assert!(
        assign_ids(&[11], &[false, false]).is_empty(),
        "a mismatch publishes nothing"
    );
    assert!(assign_ids(&[], &[false]).is_empty());
}

#[test]
fn single_stream_prefill_rows_honor_the_write_floor() {
    let r = prefill_rows(512, 1024, 0, None);
    assert_eq!(r.extent, vec![1536]);
    assert_eq!(r.q_first, vec![1024]);
    assert_eq!(r.q_len, vec![512]);
    assert_eq!(r.written_first, vec![1024]);
    assert_eq!(r.written_len, vec![512]);
    let r = prefill_rows(512, 0, 200, None);
    assert_eq!((r.written_first, r.written_len), (vec![200], vec![312]));
    let r = prefill_rows(512, 0, 900, None);
    assert_eq!(
        (r.written_first, r.written_len),
        (vec![512], vec![0]),
        "a floor past the chunk writes nothing"
    );
}

fn batched(chunk_len: u32, cu: Vec<i32>, kv_lens: Vec<i32>) -> BatchedAttnMetadata {
    BatchedAttnMetadata {
        positions_stacked: DevicePtr::NULL,
        positions_h_stacked: DevicePtr::NULL,
        positions_w_stacked: DevicePtr::NULL,
        slot_stacked: DevicePtr::NULL,
        block_table_ptrs: DevicePtr(0xBB),
        seq_len_ptrs: DevicePtr::NULL,
        batch_size: (cu.len().max(1) - 1) as u32,
        chunk_len,
        total_tokens: 0,
        cu_seqlens: DevicePtr::NULL,
        cu_seqlens_host: cu,
        kv_lens: DevicePtr(0xCC),
        kv_lens_host: kv_lens,
        max_blocks_per_seq: 16,
        staged_bytes: 0,
    }
}

#[test]
fn batched_prefill_rows_follow_the_varlen_geometry() {
    let b = batched(300, vec![0, 100, 400], vec![1100, 2300]);
    let r = prefill_rows(400, 0, 0, Some(&b));
    assert_eq!(r.q_len, vec![100, 300]);
    assert_eq!(r.extent, vec![1100, 2300]);
    assert_eq!(r.q_first, vec![1000, 2000]);
    assert_eq!(r.written_len, vec![100, 300]);
    // The stacked write floor trims the first stream only.
    let r = prefill_rows(400, 0, 150, Some(&b));
    assert_eq!(r.written_first, vec![1100, 2050]);
    assert_eq!(r.written_len, vec![0, 250]);
    // Uniform legacy chunks: no prefix sums, no per-stream extents.
    let b = batched(128, Vec::new(), Vec::new());
    let r = prefill_rows(384, 0, 0, Some(&b));
    assert_eq!(r.q_len, vec![128; 3]);
    assert_eq!(r.extent, vec![128; 3]);
}

#[test]
fn startup_refusals_need_a_selector() {
    let anything = ServeFacts {
        kv_dtypes: &[KvCacheDtype::Turbo4],
        high_speed_swap: true,
        vision_input: true,
    };
    validate_with(None, &anything).unwrap();
    let sel = Recorder::dense("quest");
    let ok = ServeFacts {
        kv_dtypes: &[KvCacheDtype::Fp8, KvCacheDtype::Bf16],
        high_speed_swap: false,
        vision_input: false,
    };
    validate_with(Some(&sel), &ok).unwrap();
    let cases = [
        (
            ServeFacts {
                kv_dtypes: &[KvCacheDtype::Fp8, KvCacheDtype::Nvfp4],
                high_speed_swap: false,
                vision_input: false,
            },
            "BF16 or FP8",
        ),
        (
            ServeFacts {
                kv_dtypes: &[KvCacheDtype::Bf16],
                high_speed_swap: true,
                vision_input: false,
            },
            "--high-speed-swap",
        ),
        (
            ServeFacts {
                kv_dtypes: &[KvCacheDtype::Bf16],
                high_speed_swap: false,
                vision_input: true,
            },
            "--no-vision",
        ),
    ];
    for (facts, needle) in cases {
        let err = validate_with(Some(&sel), &facts).unwrap_err().to_string();
        assert!(err.contains(needle) && err.contains("`quest`"), "{err}");
    }
}

#[test]
fn the_kv_budget_pays_for_the_selector_reservation() {
    assert_eq!(charge_with(None, 1 << 30).unwrap(), 1 << 30);
    let free = Recorder::dense("free");
    assert_eq!(charge_with(Some(&free), 1 << 30).unwrap(), 1 << 30);
    let mut sel = Recorder::dense("big");
    sel.reserved = 64 << 20;
    assert_eq!(
        charge_with(Some(&sel), 1 << 30).unwrap(),
        (1 << 30) - (64 << 20)
    );
    let err = charge_with(Some(&sel), 32 << 20).unwrap_err().to_string();
    assert!(err.contains("`big`"), "{err}");
}

fn pool(kv_block_bytes: usize) -> PoolGeometry {
    PoolGeometry {
        kv_block_bytes,
        block_size: 16,
        layer_dims: vec![(4, 256); 8],
    }
}

#[test]
fn the_per_page_sidecar_is_charged_in_proportion_to_the_pool() {
    // No selector, and a selector with no per-page bytes: nothing extra, and the
    // geometry still reaches the selector once.
    assert_eq!(
        charge_pool_with(None, 1 << 30, &pool(1000)).unwrap(),
        1 << 30
    );
    let mut sel = Recorder::dense("view");
    sel.reserved = 64 << 20;
    let fixed = (1 << 30) - (64 << 20);
    assert_eq!(
        charge_pool_with(Some(&sel), 1 << 30, &pool(1000)).unwrap(),
        fixed
    );
    assert_eq!(sel.pool.lock().unwrap().as_ref(), Some(&pool(1000)));
    // 1000 B KV block + 250 B sidecar: pages = fixed / 1250, charge = pages * 250,
    // and the pool the build sizes from what is left holds at least those pages.
    sel.per_block = 250;
    let left = charge_pool_with(Some(&sel), 1 << 30, &pool(1000)).unwrap();
    let pages = fixed / 1250;
    assert_eq!(left, fixed - pages * 250);
    assert!(left / 1000 >= pages && pages * 1250 <= fixed);
    // A sidecar bigger than the whole budget is a startup error naming the selector.
    sel.per_block = u64::MAX;
    let err = charge_pool_with(Some(&sel), 1 << 30, &pool(1000))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("`view`") && err.contains("per KV page"),
        "{err}"
    );
}

#[test]
fn the_pool_geometry_reports_uniform_dims() {
    let p = pool(1000);
    assert_eq!((p.attn_layers(), p.uniform_dims()), (8, Some((4, 256))));
    let mixed = PoolGeometry {
        layer_dims: vec![(4, 256), (8, 128)],
        ..p
    };
    assert_eq!(mixed.uniform_dims(), None);
}

#[test]
fn metrics_render_zeros_without_a_selector() {
    let mut out = String::new();
    render_with(&mut out, None, &Counters::default());
    for line in [
        "atlas_sparse_active{selector=\"none\"} 0",
        "atlas_sparse_plan_total{kind=\"dense\"} 0",
        "atlas_sparse_plan_total{kind=\"prefill_view\"} 0",
        "atlas_sparse_selected_pages_total 0",
        "atlas_sparse_graph_bypass_total 0",
        "atlas_sparse_view_ptr_changes_total 0",
        "atlas_sparse_reserved_bytes 0",
        "# TYPE atlas_sparse_plan_total counter",
        "# TYPE atlas_sparse_active gauge",
    ] {
        assert!(out.lines().any(|l| l == line), "missing `{line}` in\n{out}");
    }
    assert_eq!(out.matches("# TYPE atlas_sparse_plan_total").count(), 1);
}

#[test]
fn metrics_name_the_active_selector_and_count_its_plans() {
    let c = Counters::default();
    c.count_plan(PlanKind::View);
    c.count_plan(PlanKind::View);
    c.add_selected_pages(40);
    let mut out = String::new();
    render_with(&mut out, Some(("quest", 4096)), &c);
    for line in [
        "atlas_sparse_active{selector=\"quest\"} 1",
        "atlas_sparse_plan_total{kind=\"view\"} 2",
        "atlas_sparse_selected_pages_total 40",
        "atlas_sparse_reserved_bytes 4096",
    ] {
        assert!(out.lines().any(|l| l == line), "missing `{line}` in\n{out}");
    }
}

#[test]
fn the_identity_toggle_uses_the_row_grammar() {
    for off in [
        None,
        Some(""),
        Some("0"),
        Some("false"),
        Some("OFF"),
        Some(" no "),
    ] {
        assert!(!enabled_by(off), "{off:?}");
    }
    for on in [Some("1"), Some("true"), Some("on"), Some("yes")] {
        assert!(enabled_by(on), "{on:?}");
    }
}

#[test]
fn only_bf16_and_fp8_kv_are_served() {
    assert_eq!(
        KvDtypeTag::from_cache(KvCacheDtype::Bf16),
        Some(KvDtypeTag::Bf16)
    );
    assert_eq!(
        KvDtypeTag::from_cache(KvCacheDtype::Fp8),
        Some(KvDtypeTag::Fp8)
    );
    for d in [
        KvCacheDtype::Nvfp4,
        KvCacheDtype::Turbo4,
        KvCacheDtype::Turbo8,
        KvCacheDtype::Bf16KTurbo4V,
    ] {
        assert_eq!(KvDtypeTag::from_cache(d), None, "{d:?}");
    }
    assert_eq!(KvDtypeTag::Bf16.bytes_per_elem(), 2);
    assert_eq!(KvDtypeTag::Fp8.bytes_per_elem(), 1);
}
