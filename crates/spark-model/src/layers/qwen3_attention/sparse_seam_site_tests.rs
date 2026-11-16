// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend tests of the sparse-attention seam's call
//! sites (port lane L29): the hook order and the `SeqBatch` each hook sees at
//! the single-sequence decode, the batched verify and the prefill sites, the
//! published ids and last-chunk flag, the write floor, which prefill chunks
//! may be planned, and that a layer the seam cannot serve never reaches the
//! selector.

use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::KvCacheDtype;

use super::mock_tests::{DIMS, HD, Rig};
use super::registry::{StepInfo, push_step};
use super::route::{DecodeRoute, Tables};
use super::tests::Recorder;
use super::{PrefillPlan, PrefillSite, SeamDims, SeamRows};
use crate::layer::AttnMetadataDev;

const Q: DevicePtr = DevicePtr(1);
const OUT: DevicePtr = DevicePtr(2);

#[test]
fn single_decode_runs_written_then_query_then_plan_with_the_published_id() {
    let rig = Rig::new();
    let layer = rig.layer(KvCacheDtype::Bf16);
    let kv = rig.kv(KvCacheDtype::Bf16, false);
    let (bt, sl) = (rig.upload(&[3, 5, 7]), rig.upload(&[38]));
    let meta = rig.meta(bt, sl, 3, 1);
    let sel = Recorder::dense("rec");
    let _step = push_step(StepInfo {
        seq_ids: vec![42],
        last_chunk: false,
    });
    let ctx = rig.ctx(false);
    let route = layer
        .seam_decode_single(&sel, &ctx, &kv, &meta, Q, OUT, 37, &[3, 5, 7], DIMS, 0)
        .unwrap();
    let dense = Tables {
        block_table: bt,
        seq_lens: sl,
        max_blocks_per_seq: 3,
    };
    assert_eq!(route, Some(DecodeRoute::Run(dense)));
    let calls = sel.calls();
    assert_eq!(calls.len(), 3, "{calls:#?}");
    let common = "phase=Decode graph=false n=1 rows_per_seq=1 lens=[38] ids=[42] first=[37] \
                  new=[1] tables=1";
    assert!(
        calls[0].starts_with(&format!("kv {common} written=[37]/[1]")),
        "{calls:#?}"
    );
    assert!(
        calls[1].starts_with(&format!("query {common} pre_rope=false")),
        "{calls:#?}"
    );
    assert!(
        calls[2].starts_with(&format!("plan {common}")),
        "{calls:#?}"
    );
}

#[test]
fn batched_verify_rows_reach_the_selector_as_one_sequence() {
    let rig = Rig::new();
    let layer = rig.layer(KvCacheDtype::Bf16);
    let kv = rig.kv(KvCacheDtype::Bf16, false);
    let (bt, sl) = (rig.upload(&[2, 6, 2, 6, 2, 6]), rig.upload(&[31, 32, 33]));
    let meta = rig.meta(bt, sl, 2, 3);
    let tables = vec![vec![2u32, 6]; 3];
    let rows = SeamRows {
        seq_lens: &[30, 31, 32],
        block_tables: &tables,
    };
    let sel = Recorder::dense("rec");
    let _step = push_step(StepInfo {
        seq_ids: vec![7],
        last_chunk: false,
    });
    let ctx = rig.ctx(true);
    let route = layer
        .seam_decode_multi(&sel, &ctx, &kv, &meta, rows, 3, Q, 256, OUT, DIMS, 0)
        .unwrap();
    assert!(matches!(route, Some(DecodeRoute::Run(_))));
    let calls = sel.calls();
    let common = "phase=Verify graph=true n=3 rows_per_seq=3 lens=[31, 32, 33] ids=[7] \
                  first=[30] new=[3] tables=1";
    assert!(
        calls[0].starts_with(&format!("kv {common} written=[30]/[3]")),
        "{calls:#?}"
    );
    assert!(
        calls[2].starts_with(&format!("plan {common}")),
        "{calls:#?}"
    );
    // Rows the site cannot describe keep the seam off for the call.
    let short = SeamRows {
        seq_lens: &[30],
        block_tables: &tables[..1],
    };
    let none = layer
        .seam_decode_multi(&sel, &ctx, &kv, &meta, short, 3, Q, 256, OUT, DIMS, 0)
        .unwrap();
    assert_eq!(none, None);
    assert_eq!(sel.calls().len(), 3, "no hook ran for the undescribed call");
}

#[test]
fn a_layer_the_seam_cannot_serve_never_reaches_the_selector() {
    let rig = Rig::new();
    let mut layer = rig.layer(KvCacheDtype::Bf16);
    layer.sliding_window = Some(64);
    let kv = rig.kv(KvCacheDtype::Bf16, false);
    let (bt, sl) = (rig.upload(&[1]), rig.upload(&[5]));
    let meta = rig.meta(bt, sl, 1, 1);
    let sel = Recorder::dense("rec");
    let ctx = rig.ctx(false);
    let route = layer
        .seam_decode_single(&sel, &ctx, &kv, &meta, Q, OUT, 4, &[1], DIMS, 0)
        .unwrap();
    assert_eq!(route, None);
    assert!(sel.calls().is_empty());
}

fn paged_site<'a>(
    meta: &'a AttnMetadataDev,
    table: &'a [u32],
    chunk0: bool,
    hd: u32,
) -> PrefillSite<'a> {
    PrefillSite {
        num_tokens: 64,
        seq_len_start: if chunk0 { 0 } else { 128 },
        kv_write_start: 16,
        host_block_table: table,
        meta: (!chunk0).then_some(meta),
        batched: None,
        chunk0,
        q: DevicePtr(0x900),
        dims: SeamDims { hd, ..DIMS },
        stream: 0,
    }
}

#[test]
fn prefill_hooks_see_the_chunk_the_write_floor_and_the_last_chunk_flag() {
    let rig = Rig::new();
    let layer = rig.layer(KvCacheDtype::Bf16);
    let kv = rig.kv(KvCacheDtype::Bf16, false);
    let meta = rig.meta(DevicePtr(0xB0), DevicePtr(0x50), 12, 1);
    let table: Vec<u32> = (0..12).collect();
    let site = paged_site(&meta, &table, false, HD);
    let sel = Recorder::dense("rec");
    let _step = push_step(StepInfo {
        seq_ids: vec![5],
        last_chunk: true,
    });
    let ctx = rig.ctx(false);
    layer
        .seam_prefill_query(&sel, &ctx, &kv, &site, true)
        .unwrap();
    layer
        .seam_prefill_kv_written(&sel, &ctx, &kv, &site)
        .unwrap();
    layer.seam_prefill_done(&sel, &ctx, &kv, &site).unwrap();
    let calls = sel.calls();
    let common = "phase=Prefill graph=false n=1 rows_per_seq=1 lens=[192] ids=[5] first=[128] \
                  new=[64] tables=1";
    assert!(
        calls[0].starts_with(&format!("query {common} pre_rope=true")),
        "{calls:#?}"
    );
    assert!(
        calls[1].starts_with(&format!("kv {common} written=[144]/[48]")),
        "{calls:#?}"
    );
    assert!(
        calls[2].starts_with(&format!("done {common} last=true")),
        "{calls:#?}"
    );
}

#[test]
fn chunk_zero_hooks_have_no_device_table_and_may_lack_pre_rope_q() {
    let rig = Rig::new();
    let layer = rig.layer(KvCacheDtype::Bf16);
    let kv = rig.kv(KvCacheDtype::Bf16, false);
    let meta = rig.meta(DevicePtr(0xB0), DevicePtr(0x50), 12, 1);
    let table: Vec<u32> = (0..4).collect();
    let site = paged_site(&meta, &table, true, HD);
    let sel = Recorder::dense("rec");
    let ctx = rig.ctx(false);
    layer
        .seam_prefill_query(&sel, &ctx, &kv, &site, false)
        .unwrap();
    let calls = sel.calls();
    let want = "query phase=Prefill graph=false n=1 rows_per_seq=1 lens=[64] ids=[] first=[0] \
                new=[64] tables=1 pre_rope=false";
    assert_eq!(calls, vec![want.to_string()]);
}

#[test]
fn the_prefill_plan_serves_paged_single_stream_chunks_only() {
    let rig = Rig::new();
    let layer = rig.layer(KvCacheDtype::Bf16);
    let kv = rig.kv(KvCacheDtype::Bf16, false);
    let meta = rig.meta(DevicePtr(0xB0), DevicePtr(0x50), 12, 1);
    let table: Vec<u32> = (0..12).collect();
    let mut sel = Recorder::dense("rec");
    sel.prefill = Box::new(|| PrefillPlan::View {
        block_table: DevicePtr(0xC0),
        kv_len: 100,
    });
    let ctx = rig.ctx(false);
    let paged = paged_site(&meta, &table, false, HD);
    let v = layer
        .seam_prefill_route(&sel, &ctx, &kv, &paged, 192)
        .unwrap()
        .unwrap();
    assert_eq!(
        (v.block_table, v.kv_len, v.seq_len_start),
        (DevicePtr(0xC0), 100, 36)
    );
    let chunk0 = paged_site(&meta, &table, true, HD);
    let none = layer.seam_prefill_route(&sel, &ctx, &kv, &chunk0, 64);
    assert_eq!(none.unwrap(), None);
    let wide = paged_site(&meta, &table, false, 512);
    let none = layer.seam_prefill_route(&sel, &ctx, &kv, &wide, 192);
    assert_eq!(none.unwrap(), None);
    assert_eq!(
        sel.calls().len(),
        1,
        "only the paged chunk asked for a plan"
    );
}
