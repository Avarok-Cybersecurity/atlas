// SPDX-License-Identifier: AGPL-3.0-only

//! Mock-backend tests of the MTP head v2 hooks: with every feature off the
//! hooks issue nothing; under a chain round the ids move into the round's
//! slot and are read back once; the ranked head launches the remap.

use super::*;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

fn off_head(rows: usize) -> HeadV2 {
    HeadV2 {
        ranked: None,
        rows: rows as u32,
        chain: ChainState::off(),
        window: AtomicU32::new(u32::MAX),
    }
}

fn chain_head(gpu: &MockGpuBackend, rows: usize) -> HeadV2 {
    let cells = MAX_CHAIN_DRAFTS * MAX_CHAIN_WIDTH;
    HeadV2 {
        ranked: None,
        rows: rows as u32,
        chain: ChainState {
            enabled: true,
            gather_k: KernelHandle(7),
            ids: gpu.alloc(cells * 4).unwrap(),
            lp: gpu.alloc(cells * 4).unwrap(),
            active: AtomicBool::new(false),
            pos: AtomicU32::new(0),
            width: AtomicU32::new(0),
        },
        window: AtomicU32::new(u32::MAX),
    }
}

fn put_ids(gpu: &MockGpuBackend, at: DevicePtr, ids: &[u32]) {
    let b: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    gpu.copy_h2d(&b, at).unwrap();
}

#[test]
fn every_hook_is_a_no_op_with_both_rows_off() {
    let gpu = MockGpuBackend::new();
    let h = off_head(100);
    let ids = gpu.alloc(64).unwrap();
    let table = gpu.alloc(1024).unwrap();
    assert!(!h.chain_embed(&gpu, table, table, 1, 8, 0).unwrap());
    assert!(!h.after_argmax(&gpu, ids, 4, 0).unwrap());
    h.collect_lp(&gpu, ids, 4, 0).unwrap();
    let mask = [i32::MAX; 4];
    assert_eq!(
        h.grammar_argmax(&gpu, ids, 4, Some(&mask), ids, None, 0, 0)
            .unwrap(),
        None
    );
    assert_eq!(gpu.launch_count(), 0, "no kernel may run");
    assert_eq!(gpu.d2d_count(), 0, "no copy may run");
    assert_eq!(gpu.d2h_blocking_count(), 0, "no readback may run");
    let t = [3u32, 4, 5];
    let (v, n) = h.window_view(&t, 40, 16);
    assert!(matches!(v, Cow::Borrowed(_)) && n == 40);
}

#[test]
fn the_id_head_holds_exactly_its_first_rows() {
    let h = off_head(100);
    assert!(h.holds(0) && h.holds(99) && !h.holds(100) && !h.holds(248_046));
    assert_eq!(h.token_of_row(42), 42);
}

#[test]
fn a_chain_round_collects_every_position_and_reads_back_once() {
    let gpu = MockGpuBackend::new();
    let h = chain_head(&gpu, 100);
    assert!(h.chain.fits(3, 2) && !h.chain.fits(3, 1) && !h.chain.fits(MAX_CHAIN_WIDTH + 1, 2));
    let scratch = gpu.alloc(64).unwrap();
    let table = gpu.alloc(4096).unwrap();
    let embeds = gpu.alloc(4096).unwrap();
    {
        let _round = h.chain.begin(3).unwrap();
        for (pos, ids) in [[10u32, 20, 30], [11, 21, 31]].iter().enumerate() {
            h.chain.set_pos(pos);
            // Position 0 embeds from host ids; later positions gather on the device.
            let gathered = h.chain_embed(&gpu, table, embeds, 3, 8, 0).unwrap();
            assert_eq!(gathered, pos > 0, "pos {pos}");
            put_ids(&gpu, scratch, ids);
            assert!(h.after_argmax(&gpu, scratch, 3, 0).unwrap(), "collected");
        }
        let before = gpu.d2h_blocking_count();
        let (flat, lp) = h.chain.read(&gpu, 2, false).unwrap();
        assert_eq!(
            gpu.d2h_blocking_count() - before,
            1,
            "one readback per round"
        );
        assert_eq!(flat, vec![10, 20, 30, 11, 21, 31]);
        assert!(lp.is_empty());
    }
    // The gather of position 1 read position 0's slot.
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].func, 7);
    assert_eq!(launches[0].grid, [3, 1, 1]);
    assert_eq!(launches[0].args[0], MockArg::Buffer(h.chain.ids));
    // The round ended: hooks are inert again.
    assert!(!h.after_argmax(&gpu, scratch, 3, 0).unwrap());
}

#[test]
fn a_round_ends_on_an_error_path_too() {
    let gpu = MockGpuBackend::new();
    let h = chain_head(&gpu, 100);
    let run = || -> anyhow::Result<()> {
        let _round = h.chain.begin(1)?;
        anyhow::bail!("mid-chain failure")
    };
    assert!(run().is_err());
    assert!(!h.chain.active(), "the guard closed the round");
    assert!(h.chain.begin(MAX_CHAIN_WIDTH + 1).is_err());
}

#[test]
fn log_probs_are_kept_per_position() {
    let gpu = MockGpuBackend::new();
    let h = chain_head(&gpu, 100);
    let lp_src = gpu.alloc(64).unwrap();
    let _round = h.chain.begin(2).unwrap();
    for (pos, v) in [[-0.5f32, -1.0], [-0.25, -2.0]].iter().enumerate() {
        h.chain.set_pos(pos);
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        gpu.copy_h2d(&b, lp_src).unwrap();
        h.collect_lp(&gpu, lp_src, 2, 0).unwrap();
    }
    let (_, lp) = h.chain.read(&gpu, 2, true).unwrap();
    assert_eq!(lp, vec![-0.5, -1.0, -0.25, -2.0]);
}

#[test]
fn a_ranked_head_remaps_on_the_device_and_translates_rows() {
    let gpu = MockGpuBackend::new();
    let row_to_id = vec![13u32, 11, 248_046];
    let dev = gpu.alloc(12).unwrap();
    let mut h = off_head(3);
    h.ranked = Some(Arc::new(RankedHead {
        in_head: vocab_rank::membership(&row_to_id, 248_070),
        row_to_id,
        row_to_id_dev: dev,
        remap_k: KernelHandle(9),
    }));
    let ids = gpu.alloc(16).unwrap();
    assert!(
        !h.after_argmax(&gpu, ids, 1, 0).unwrap(),
        "no chain: not collected"
    );
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    assert_eq!(
        (l[0].func, l[0].grid, l[0].block),
        (9, [1, 1, 1], [128, 1, 1])
    );
    assert_eq!(l[0].args[0], MockArg::Buffer(ids));
    assert_eq!(l[0].args[1], MockArg::Buffer(dev));
    assert_eq!(
        l[0].args[4],
        MockArg::Buffer(DevicePtr::NULL),
        "no chain copy"
    );
    assert!(h.holds(248_046) && h.holds(13) && !h.holds(2));
    assert_eq!(h.token_of_row(2), 248_046);
    assert_eq!(h.token_of_row(7), 7, "out of range: unchanged");
}

#[test]
fn the_head_window_overrides_the_process_default() {
    // The process default is shared with window.rs's own test, so it is never
    // read here; a head-level setting must win regardless of it.
    let h = off_head(10);
    h.set_kv_window(Some(64));
    assert_eq!(h.kv_window(), Some(64));
    h.set_kv_window(None);
    assert_eq!(h.kv_window(), None);
    h.set_kv_window(Some(u32::MAX));
    assert_eq!(
        h.kv_window(),
        Some(u32::MAX - 1),
        "the sentinel is never stored"
    );
}
