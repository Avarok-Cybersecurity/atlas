// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: unit tests of `w4_proj_dispatch.rs` on the mock backend: decode windows and
//! the reserved scratch, the shared-input k/v pair, the prefill route and its decline fallback,
//! and a CPU oracle of the batched q/k/v scatter (bytes compared, row invariance across batch
//! widths and staging blocks).

use super::fixture::{
    Fakes, H, MAX_TOKENS, Rig, SEEDS, kv_dim, oracle_row, per_seq_qkv, q_dim, q_proj_dim,
};
use super::*;
use crate::port::w4::fake::{Call, FakeW4};

#[test]
fn decode_rows_walk_the_window_and_hand_out_the_reserved_scratch() {
    let rig = Rig::new(false);
    let mut layer = rig.bf16_layer(false);
    let fakes = Fakes::new(false, 4, false);
    fakes.install(&mut layer);
    let (x, out) = (DevicePtr(0x10_0000), DevicePtr(0x20_0000));
    let o = layer.w4_proj.get(W4Slot::O).unwrap().clone();
    layer
        .w4_decode_rows(&o, "o", &rig.gpu, x, 10, out, H, q_dim(), 0)
        .unwrap();
    let row = |r: usize| (x.offset(r * q_dim() * 2), out.offset(r * H * 2));
    let want: Vec<_> = [(0, 4), (4, 4), (8, 2)]
        .iter()
        .map(|&(f, rows)| {
            let (x, out) = row(f);
            ("o", Call::Decode { x, rows, out })
        })
        .collect();
    assert_eq!(fakes.calls(), want);
    let s = layer.w4_ready(&rig.gpu, 0).unwrap().s;
    assert!(fakes.o.scratches.lock().iter().all(|&seen| seen == s));
    // A shape mismatch is an error naming the projection, before any call.
    let e = layer
        .w4_decode_rows(
            &o,
            "attention o_proj",
            &rig.gpu,
            x,
            1,
            out,
            H + 2,
            q_dim(),
            0,
        )
        .unwrap_err()
        .to_string();
    assert!(e.contains("attention o_proj"), "{e}");
    assert_eq!(fakes.calls().len(), 3);
}

/// k and v share the input row; weights that fuse take ONE `decode_pair`, weights that do not
/// take the trait default (two decodes), and both write the same regions.
#[test]
fn kv_decode_goes_through_decode_pair() {
    let rig = Rig::new(false);
    let log = crate::port::w4::fake::log();
    let mk = |name, n, k, fuse: bool| {
        let f = FakeW4::new(name, n, k, 16, &log);
        std::sync::Arc::new(if fuse { f.fusing_pair() } else { f })
    };
    let mut layer = rig.bf16_layer(false);
    let (q, o) = (
        mk("q", q_proj_dim(false), H, false),
        mk("o", H, q_dim(), false),
    );
    let (k, v) = (mk("k", kv_dim(), H, true), mk("v", kv_dim(), H, true));
    layer.try_set_w4_proj(q, k, v, o).unwrap();
    let (x, ko, vo) = (
        DevicePtr(0x10_0000),
        DevicePtr(0x20_0000),
        DevicePtr(0x30_0000),
    );
    layer
        .w4_decode_kv(&rig.gpu, x, ko, vo, kv_dim(), H, 0)
        .unwrap();
    let pair = Call::Pair {
        x,
        rows: 1,
        out_a: ko,
        out_b: vo,
    };
    assert_eq!(*log.lock(), vec![("k", pair)]);
    // A k/v width the layer does not have is refused before any call.
    assert!(
        layer
            .w4_decode_kv(&rig.gpu, x, ko, vo, kv_dim() + 2, H, 0)
            .is_err()
    );
    assert_eq!(log.lock().len(), 1);
}

#[test]
fn prefill_inside_the_window_decodes_and_wider_prefills_chunk() {
    let rig = Rig::new(false);
    let mut layer = rig.bf16_layer(false);
    let fakes = Fakes::new(false, 16, false);
    fakes.install(&mut layer);
    let ctx = rig.ctx();
    let k = layer.w4_proj.get(W4Slot::K).unwrap().clone();
    let (x, out) = (DevicePtr(0x10_0000), DevicePtr(0x20_0000));
    layer
        .w4_prefill_rows(&k, "k", &ctx, x, 16, out, kv_dim(), H, 0)
        .unwrap();
    layer
        .w4_prefill_rows(&k, "k", &ctx, x, 5000, out, kv_dim(), H, 0)
        .unwrap();
    let mut want = vec![("k", Call::Decode { x, rows: 16, out })];
    for (first, m) in [(0usize, 1667usize), (1667, 1667), (3334, 1666)] {
        let (x, out) = (x.offset(first * H * 2), out.offset(first * kv_dim() * 2));
        want.push(("k", Call::Prefill { x, m, out }));
    }
    assert_eq!(fakes.calls(), want);
}

#[test]
fn a_declined_prefill_chunk_falls_back_to_one_dequant_and_the_bf16_gemm() {
    let rig = Rig::new(false);
    let mut layer = rig.bf16_layer(false);
    let log = crate::port::w4::fake::log();
    let mk = |name, n, k| std::sync::Arc::new(FakeW4::new(name, n, k, 16, &log));
    let q = std::sync::Arc::new(
        FakeW4::new("q", q_proj_dim(false), H, 16, &log).answering(&[true, false, false]),
    );
    let (k, v, o) = (
        mk("k", kv_dim(), H),
        mk("v", kv_dim(), H),
        mk("o", H, q_dim()),
    );
    layer.try_set_w4_proj(q.clone(), k, v, o).unwrap();
    let ctx = rig.ctx();
    let w = layer.w4_proj.get(W4Slot::Q).unwrap().clone();
    let declines = crate::port::w4::prefill_declines();
    let launches = rig.gpu.launch_count();
    let (x, out) = (DevicePtr(0x10_0000), DevicePtr(0x20_0000));
    layer
        .w4_prefill_rows(&w, "q", &ctx, x, 5000, out, q_proj_dim(false), H, 0)
        .unwrap();
    let calls = log.lock().clone();
    let dequants: Vec<_> = calls
        .iter()
        .filter(|(_, c)| matches!(c, Call::Dequant { .. }))
        .collect();
    assert_eq!(
        dequants.len(),
        1,
        "one dequant serves every declined chunk of the call"
    );
    let prefills = calls
        .iter()
        .filter(|(_, c)| matches!(c, Call::Prefill { .. }))
        .count();
    assert_eq!(prefills, 3);
    assert!(crate::port::w4::prefill_declines() >= declines + 2);
    let Call::Dequant { out: bf16 } = dequants[0].1 else {
        unreachable!()
    };
    let r = layer.w4_ready(&rig.gpu, 0).unwrap();
    assert_eq!(bf16, scratch::bf16_dequant(&rig.gpu, &r.spec).unwrap());
    // The two declined chunks each ran one BF16 GEMM over the dequantized weight, at the
    // declined chunks' own output rows (the accepted first chunk ran none).
    let gemms: Vec<_> = rig.gpu.launches_snapshot()[launches..]
        .iter()
        .filter(|l| {
            l.args
                .contains(&spark_runtime::gpu::mock::MockArg::Buffer(bf16))
        })
        .cloned()
        .collect();
    assert_eq!(gemms.len(), 2);
    for (g, first) in gemms.iter().zip([1667usize, 3334]) {
        let oi = out.offset(first * q_proj_dim(false) * 2);
        assert!(
            g.args
                .contains(&spark_runtime::gpu::mock::MockArg::Buffer(oi))
        );
    }
}

/// The batched q/k/v tier against the CPU oracle: every sequence's `[Q | K | V]` slice equals the
/// M=1 decode of its own input row, at a width that needs chunking (window 3) and one that crosses
/// a staging block (`STAGE_ROWS + 6`), and in place for one row.
#[test]
fn strided_qkv_bytes_equal_the_per_row_oracle() {
    for n in [1usize, 2, 7, STAGE_ROWS + 6] {
        assert!(n <= MAX_TOKENS);
        let rig = Rig::new(false);
        let mut layer = rig.bf16_layer(false);
        let fakes = Fakes::new(false, 3, true);
        fakes.install(&mut layer);
        let normed = rig.buffers.norm_output();
        let x = rig.fill_x(normed, n, H, n);
        let qkv = rig.buffers.qkv_output();
        let (qd, kvd, stride) = (q_proj_dim(false), kv_dim(), per_seq_qkv(false));
        let (nq, hd) = (super::fixture::NQ as u32, super::fixture::HD as u32);
        layer.prepare_w4_proj(&rig.gpu, 0).unwrap();
        let launches = rig.gpu.launch_count();
        layer
            .w4_qkv_strided(&rig.gpu, normed, n, H, qd, kvd, nq, hd, qkv, stride, 0)
            .unwrap();
        let got = rig.read(qkv, n * stride);
        for r in 0..n {
            let row = &x[r * H..(r + 1) * H];
            let mut want = oracle_row(SEEDS[0], qd, row);
            want.extend(oracle_row(SEEDS[1], kvd, row));
            want.extend(oracle_row(SEEDS[2], kvd, row));
            assert_eq!(
                &got[r * stride..(r + 1) * stride],
                &want[..],
                "n={n} row {r}"
            );
        }
        let decodes = fakes
            .calls()
            .iter()
            .filter(|(name, _)| *name == "q")
            .count();
        let blocks: usize = (0..n)
            .step_by(STAGE_ROWS)
            .map(|b| (n - b).min(STAGE_ROWS).div_ceil(3))
            .sum();
        assert_eq!(
            decodes, blocks,
            "n={n}: q decodes in windows of 3 per staging block"
        );
        assert_eq!(rig.gpu.launch_count(), launches, "ungated: no split launch");
    }
}

/// Planted control for the oracle above: a q weight with a different seed must fail it, so the
/// byte comparison can tell a wrong projection from a right one.
#[test]
fn the_strided_oracle_rejects_a_wrong_weight() {
    let rig = Rig::new(false);
    let mut layer = rig.bf16_layer(false);
    let fakes = Fakes::new(false, 16, true);
    fakes.install(&mut layer);
    let n = 4;
    let normed = rig.buffers.norm_output();
    let x = rig.fill_x(normed, n, H, 1);
    let (qd, kvd, stride) = (q_proj_dim(false), kv_dim(), per_seq_qkv(false));
    let (nq, hd) = (super::fixture::NQ as u32, super::fixture::HD as u32);
    let qkv = rig.buffers.qkv_output();
    layer
        .w4_qkv_strided(&rig.gpu, normed, n, H, qd, kvd, nq, hd, qkv, stride, 0)
        .unwrap();
    let got = rig.read(qkv, stride);
    let wrong = oracle_row(SEEDS[0] + 1, qd, &x[..H]);
    assert_ne!(
        &got[..qd * 2],
        &wrong[..],
        "a wrong q weight must not match"
    );
    assert_eq!(&got[..qd * 2], &oracle_row(SEEDS[0], qd, &x[..H])[..]);
}

#[test]
fn a_gated_batch_splits_every_row_in_place() {
    let rig = Rig::new(true);
    let mut layer = rig.bf16_layer(true);
    let fakes = Fakes::new(true, 16, false);
    fakes.install(&mut layer);
    let (n, stride) = (5, per_seq_qkv(true));
    let qkv = rig.buffers.qkv_output();
    let (nq, hd) = (super::fixture::NQ as u32, super::fixture::HD as u32);
    let (qd, kvd) = (q_proj_dim(true), kv_dim());
    layer.prepare_w4_proj(&rig.gpu, 0).unwrap();
    let first = rig.gpu.launch_count();
    let normed = rig.buffers.norm_output();
    layer
        .w4_qkv_strided(&rig.gpu, normed, n, H, qd, kvd, nq, hd, qkv, stride, 0)
        .unwrap();
    let launches = rig.gpu.launches_snapshot()[first..].to_vec();
    assert_eq!(launches.len(), n);
    for (i, l) in launches.iter().enumerate() {
        assert_eq!(
            l.args[0],
            spark_runtime::gpu::mock::MockArg::Buffer(qkv.offset(i * stride))
        );
    }
    // A stride that does not match the projections is refused before any call.
    let calls = fakes.calls().len();
    let bad = layer.w4_qkv_strided(&rig.gpu, normed, n, H, qd, kvd, nq, hd, qkv, stride + 2, 0);
    assert!(bad.is_err());
    assert_eq!(fakes.calls().len(), calls);
}
