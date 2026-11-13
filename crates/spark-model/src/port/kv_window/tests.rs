// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the window desk's pure parts (lane L36): the policy (identity
//! below the window, batch and rate limits, recall deltas), the view oracle against a
//! brute-force reference, the configuration grammar and the MTP window.

use super::policy::{self, Geometry, STEP_TOKENS, Slot};
use super::selector::{compact_row, view_bytes};
use super::{WindowConfig, mtp_window, resolve_config};

const BS: usize = 16;

fn geom() -> Geometry {
    Geometry::new(BS, 2048, 32768, 2048)
}

#[test]
fn a_window_covering_the_context_is_the_identity() {
    let g = geom();
    for kv in [0usize, 1, 15, 16, 2048, 30000, 32768, 32768 + 2048, 34000] {
        assert!(g.is_identity(kv), "kv={kv}");
        assert!(policy::demote_batch(&g, kv, &vec![Slot::Resident; kv.div_ceil(BS)]).is_empty());
    }
    // Any window at least as long as the context.
    for window in [4096usize, 65536, 1 << 20] {
        let g = Geometry::new(BS, 0, window, 0);
        for kv in (0..=window).step_by(97) {
            assert!(g.is_identity(kv), "window={window} kv={kv}");
        }
    }
}

#[test]
fn the_demotable_range_spares_the_head_the_window_and_the_open_block() {
    let g = geom();
    // 40,000 positions: 2,500 full blocks; the window starts in block (40000 - 32768) / 16 = 452.
    assert_eq!(g.demotable(40_000), 128..452);
    // The block holding the window's first position stays.
    let r = g.demotable(40_008);
    assert_eq!(r.end, (40_008 - 32_768) / BS);
    // A tiny window never reaches the open (partial) block.
    let g2 = Geometry::new(BS, 0, 1, 0);
    assert_eq!(
        g2.demotable(100),
        0..6,
        "100 positions: 6 full blocks, block 6 is open"
    );
    // Head protection rounds up to whole blocks.
    assert_eq!(Geometry::new(BS, 17, 1024, 0).head_blocks(), 2);
}

#[test]
fn demotion_waits_for_a_full_batch_and_takes_only_resident_blocks() {
    let g = geom();
    let batch = STEP_TOKENS / BS;
    assert_eq!(g.batch_blocks(), batch);
    // Demotable range of batch - 1 blocks: nothing yet.
    let kv = (128 + batch - 1) * BS + 32768;
    let slots = vec![Slot::Resident; kv / BS + 1];
    assert!(policy::demote_batch(&g, kv, &slots).is_empty());
    // One more block: the whole batch goes.
    let kv = kv + BS;
    let slots = vec![Slot::Resident; kv / BS + 1];
    let b = policy::demote_batch(&g, kv, &slots);
    assert_eq!(b.len(), batch);
    assert_eq!(b[0], 128);
    // Blocks already off the GPU or recalled are not demoted again.
    let mut slots = slots;
    slots[128] = Slot::Hidden;
    slots[129] = Slot::Recalled;
    slots[130] = Slot::Fetching;
    assert!(
        policy::demote_batch(&g, kv, &slots).is_empty(),
        "below a batch again"
    );
    // One call never demotes more than the cap.
    let kv = (128 + policy::MAX_DEMOTE_BLOCKS + 500) * BS + 32768;
    let slots = vec![Slot::Resident; kv / BS + 1];
    assert_eq!(
        policy::demote_batch(&g, kv, &slots).len(),
        policy::MAX_DEMOTE_BLOCKS
    );
}

#[test]
fn recall_rounds_are_rate_limited_and_need_an_eviction() {
    let g = geom();
    let resident = vec![Slot::Resident; 10];
    assert!(
        !policy::recall_due(&g, 50_000, None, &resident),
        "no eviction yet"
    );
    let mut slots = resident;
    slots[3] = Slot::Hidden;
    assert!(policy::recall_due(&g, 50_000, None, &slots));
    assert!(!policy::recall_due(
        &g,
        50_000 + STEP_TOKENS - 1,
        Some(50_000),
        &slots
    ));
    assert!(policy::recall_due(
        &g,
        50_000 + STEP_TOKENS,
        Some(50_000),
        &slots
    ));
    let no_budget = Geometry::new(BS, 2048, 32768, 0);
    assert!(!policy::recall_due(&no_budget, 50_000, None, &slots));
}

#[test]
fn a_recall_plan_fetches_only_the_delta_within_its_budget() {
    // Budget of 4 pages.
    let g = Geometry::new(BS, 0, 32768, 4 * BS);
    let mut slots = vec![Slot::Resident; 20];
    for s in &mut slots[2..12] {
        *s = Slot::Hidden;
    }
    slots[5] = Slot::Recalled;
    slots[6] = Slot::Recalled;
    // Ranked: 6 stays, 9 and 3 come, 7 comes; 5 retires. Resident 15 and stray 99 are ignored.
    let plan = policy::plan_recall(&g, &slots, &[6, 15, 9, 9, 3, 99, 7, 8, 10]);
    assert_eq!(plan.retire, vec![5]);
    assert_eq!(plan.fetch, vec![3, 7, 9]);
    // An empty ranking retires every recalled page and fetches nothing.
    let plan = policy::plan_recall(&g, &slots, &[]);
    assert_eq!(plan.retire, vec![5, 6]);
    assert!(plan.fetch.is_empty());
    // A copy still running outside the target uses budget until it lands.
    slots[5] = Slot::Hidden;
    slots[6] = Slot::Hidden;
    slots[2] = Slot::Fetching;
    slots[4] = Slot::Fetching;
    let plan = policy::plan_recall(&g, &slots, &[9, 10, 11, 3]);
    assert!(plan.retire.is_empty());
    assert_eq!(plan.fetch, vec![9, 10], "4 - 2 in flight = 2 places");
}

#[test]
fn resident_tokens_count_every_hidden_block_once() {
    let g = Geometry::new(BS, 0, 32768, 0);
    let mut slots = vec![Slot::Resident; 10];
    assert_eq!(policy::resident_tokens(&g, 155, &slots), 155);
    slots[2] = Slot::Hidden;
    slots[3] = Slot::Fetching;
    slots[4] = Slot::Recalled;
    assert_eq!(policy::resident_tokens(&g, 155, &slots), 155 - 2 * BS);
}

/// Brute force: walk positions, keep those whose block is not the hole.
fn brute(table: &[u32], len: u32, bs: u32, hole: Option<u32>) -> (Vec<u32>, u32) {
    let mut kept = Vec::new();
    let mut n = 0u32;
    for pos in 0..len {
        let b = table[(pos / bs) as usize];
        if Some(b) == hole {
            continue;
        }
        n += 1;
        if pos % bs == 0 {
            kept.push(b);
        }
    }
    (kept, n)
}

#[test]
fn the_view_oracle_matches_a_position_walk() {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for _ in 0..300 {
        let blocks = 1 + (next() % 80) as usize;
        let hole = 1000u32;
        let mut table: Vec<u32> = (0..blocks as u32).map(|i| i * 3 + 1).collect();
        // Hide some full blocks, never the last.
        for e in table.iter_mut().take(blocks.saturating_sub(1)) {
            if next().is_multiple_of(3) {
                *e = hole;
            }
        }
        let len = ((blocks - 1) * BS) as u32 + 1 + (next() % BS as u64) as u32;
        let got = compact_row(&table, len, BS as u32, Some(hole));
        assert_eq!(
            got,
            brute(&table, len, BS as u32, Some(hole)),
            "table={table:?} len={len}"
        );
        // With nothing hidden the view is the table and the length.
        let (v, l) = compact_row(&table, len, BS as u32, None);
        assert_eq!(l, len);
        assert_eq!(v.len(), blocks);
    }
}

#[test]
fn verify_rows_keep_their_offsets_in_the_view() {
    let hole = 77;
    let table = vec![0, 1, hole, hole, 4, 5, 6];
    let base = 6 * BS as u32 + 3;
    let lens: Vec<u32> = (0..4)
        .map(|r| compact_row(&table, base + r, BS as u32, Some(hole)).1)
        .collect();
    assert_eq!(lens, vec![base - 32, base - 31, base - 30, base - 29]);
    let (v, _) = compact_row(&table, base, BS as u32, Some(hole));
    assert_eq!(v, vec![0, 1, 4, 5, 6]);
}

#[test]
fn view_buffers_are_aligned() {
    assert_eq!(view_bytes(1, 1), 512);
    assert!(view_bytes(40, 16384).is_multiple_of(256));
    assert!(view_bytes(40, 16384) < super::selector::VIEW_BYTES);
}

#[test]
fn sizes_resolve_cli_then_environment_then_default() {
    let none = |_: &str| None;
    assert_eq!(
        resolve_config(None, None, none).unwrap(),
        WindowConfig::default()
    );
    let env = |k: &str| match k {
        "AVAROK_KV_WINDOW_TOKENS" => Some("65536".to_string()),
        "AVAROK_KV_WINDOW_HEAD_TOKENS" => Some(" 4096 ".to_string()),
        "AVAROK_KV_WINDOW_RECALL_TOKENS" => Some("1024".to_string()),
        "AVAROK_KV_WINDOW_HOST_GB" => Some("".to_string()),
        _ => None,
    };
    let c = resolve_config(None, None, env).unwrap();
    assert_eq!(
        c,
        WindowConfig {
            window_tokens: 65536,
            head_tokens: 4096,
            recall_tokens: 1024,
            host_gib: super::DEFAULT_HOST_GIB,
        }
    );
    let c = resolve_config(Some(8192), Some(0), env).unwrap();
    assert_eq!((c.window_tokens, c.head_tokens), (8192, 0), "the CLI wins");
    let bad = |k: &str| (k == "AVAROK_KV_WINDOW_TOKENS").then(|| "lots".to_string());
    assert!(resolve_config(None, None, bad).is_err());
    assert!(
        resolve_config(Some(512), None, none).is_err(),
        "below one batch"
    );
    assert!(resolve_config(Some(1 << 25), None, none).is_err());
    let big_recall = |k: &str| (k == "AVAROK_KV_WINDOW_RECALL_TOKENS").then(|| "40000".to_string());
    assert!(resolve_config(None, None, big_recall).is_err());
    let no_host = |k: &str| (k == "AVAROK_KV_WINDOW_HOST_GB").then(|| "0".to_string());
    assert!(resolve_config(None, None, no_host).is_err());
}

#[test]
fn the_mtp_window_is_the_identity_unless_the_desk_is_on() {
    assert_eq!(mtp_window::window_for(false, 32768), None);
    assert_eq!(mtp_window::window_for(true, 0), None);
    assert_eq!(mtp_window::window_for(true, 32768), Some(32768));
    // Applying `None` publishes nothing (no process state is touched).
    mtp_window::apply(None);
}

#[test]
fn the_selector_plans_a_fixed_view_and_compacts_prefill_on_the_host() {
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};

    use super::desk::DeviceHooks;
    use super::selector::{SeamDevice, VIEW_KERNEL, WindowDeskSelector};
    use crate::layers::qwen3_attention::sparse_seam::{
        KvDtypeTag, KvLayerView, KvSelector, PrefillPlan, SeamCtx, SeamPhase, SeqBatch, SparsePlan,
    };

    fn ctx(gpu: &dyn GpuBackend, phase: SeamPhase) -> SeamCtx<'_> {
        SeamCtx {
            gpu,
            attn_layer: 0,
            kv: KvLayerView {
                k_pool: DevicePtr(0),
                v_pool: DevicePtr(0),
                dtype: KvDtypeTag::Bf16,
                block_size: BS as u32,
                kv_heads: 1,
                head_dim: 64,
                kv_scale: 1.0,
                v_scale: 1.0,
                block_stride_bytes: 0,
                cache_stride: 0,
            },
            stream: 0,
            in_graph: false,
            phase,
            num_q_heads: 2,
            inv_sqrt_d: 0.125,
        }
    }
    fn batch<'a>(
        n: usize,
        table: DevicePtr,
        lens: DevicePtr,
        host_lens: &'a [u32],
        host_tables: &'a [&'a [u32]],
    ) -> SeqBatch<'a> {
        SeqBatch {
            n,
            block_table: table,
            seq_lens: lens,
            max_blocks_per_seq: 8,
            host_seq_lens: host_lens,
            seq_ids: &[],
            rows_per_seq: n,
            q_row_stride: 128,
            q_rows: n,
            first_pos: &[],
            new_tokens: &[],
            host_block_tables: host_tables,
            block_table_ptrs: DevicePtr(0),
        }
    }

    let gpu = MockGpuBackend::new();
    let table = gpu.alloc(2 * 8 * 4).unwrap();
    let lens = gpu.alloc(2 * 4).unwrap();
    let host_table: Vec<u32> = (0..8).collect();
    let tables: [&[u32]; 1] = [&host_table];
    let sel = WindowDeskSelector::new();
    let c = ctx(&gpu, SeamPhase::Verify);
    let s = batch(2, table, lens, &[70, 71], &tables);
    let first = match sel.plan(&c, &s, DevicePtr(0)).unwrap() {
        SparsePlan::View {
            block_table,
            seq_lens,
            max_blocks_per_seq,
        } => {
            assert_eq!(max_blocks_per_seq, 8);
            (block_table, seq_lens)
        }
        _ => panic!("the desk plans a view on every step"),
    };
    let again = match sel.plan(&c, &s, DevicePtr(0)).unwrap() {
        SparsePlan::View {
            block_table,
            seq_lens,
            ..
        } => (block_table, seq_lens),
        _ => panic!("view"),
    };
    assert_eq!(
        first, again,
        "fixed buffers: a captured graph never changes topology"
    );
    let launches = gpu.launches_snapshot();
    assert!(
        launches
            .iter()
            .filter(|l| l.grid == [2, 1, 1] && l.block == [32, 1, 1])
            .count()
            >= 2,
        "one compaction per plan"
    );
    assert!(
        gpu.kernel_lookups_snapshot()
            .iter()
            .any(|(m, f)| m == super::selector::MODULE && f == VIEW_KERNEL)
    );
    // Prefill: nothing hidden plans Dense; a table holding the hole gets a host-compacted view.
    let pre = batch(1, table, lens, &[70], &tables);
    let pc = ctx(&gpu, SeamPhase::Prefill);
    assert!(matches!(
        sel.plan_prefill(&pc, &pre, DevicePtr(0)).unwrap(),
        PrefillPlan::Dense
    ));
    SeamDevice.publish_hole(&gpu, Some(3), 0).unwrap();
    match sel.plan_prefill(&pc, &pre, DevicePtr(0)).unwrap() {
        PrefillPlan::View { kv_len, .. } => assert_eq!(kv_len, 70 - BS as u32),
        PrefillPlan::Dense => panic!("a hidden page needs the compacted prefill view"),
    }
    SeamDevice.publish_hole(&gpu, None, 0).unwrap();
}

#[test]
fn the_series_render_and_read_zero_with_the_desk_off() {
    let mut out = String::new();
    super::render_prometheus(&mut out);
    for name in [
        "atlas_kv_window_resident_tokens",
        "atlas_kv_window_fetch_bytes_total",
        "atlas_kv_window_fetch_seconds_total",
        "atlas_kv_window_demoted_pages_total",
    ] {
        assert!(out.contains(&format!("# TYPE {name} ")), "{name}");
        assert!(out.lines().any(|l| l.starts_with(name)), "{name}");
    }
    assert!(!super::enabled());
    assert!(!super::holds_hidden_pages(&[1, 2, 3]));
    assert!(super::swap_out_guard(&[1, 2, 3]).is_ok());
}
