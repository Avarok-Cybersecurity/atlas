// SPDX-License-Identifier: AGPL-3.0-only
// After TensorFold zig/src/core/exl3_layout.zig:196-250 (the geometry, TP part and expert layout tests;
// Apache-2.0, ashhart/TensorFold, commit 3885dce). Changes: translated to Rust; added strips, tensor
// part and byte-view tests. See EXL3-ATTRIBUTION.md.

//! Responsibility: tests of the EXL3 layout: group geometry and K2, tensor-parallel parts, stacked and
//! ragged expert layouts, the strips order, and [`Exl3Tensor`] parts and byte views.

use super::fixture::{Rng, synth_group};
use super::format::{Codebook, Width};
use super::layout::{self, ExpertLayout, ExpertLayoutKind, Group, LayoutError, Span, Split};
use super::tensor::{Exl3Tensor, ScaleForm};

#[test]
fn group_geometry_and_k2() {
    let g = Group::from_shape(&[320, 144, 32]).unwrap();
    assert_eq!(g.k(), 5120);
    assert_eq!(g.n(), 2304);
    assert_eq!(g.k2(), 4);
    assert_eq!(g.bits(), 2.0);
    assert_eq!(g.tile_bytes(), 64);
    assert_eq!(g.trellis_bytes(), 320 * 144 * 64);
    assert_eq!(
        Group::from_shape(&[320, 144, 30]),
        Err(LayoutError::BadTrellis)
    );
    assert_eq!(Group::from_shape(&[320, 144]), Err(LayoutError::BadTrellis));
    assert_eq!(
        Group::from_shape(&[320, 144, 136]),
        Err(LayoutError::BadTrellis)
    );
    assert_eq!(
        Group::from_shape(&[0, 144, 32]),
        Err(LayoutError::BadTrellis)
    );
    assert_eq!(
        Group::from_shape(&[(u32::MAX / 16) as usize + 1, 1, 32]),
        Err(LayoutError::BadTrellis)
    );
}

#[test]
fn tp_parts_rows_contiguous_columns_a_run_a_k_tile_row() {
    // A 3-bit routed down projection, 2304 -> 5120.
    let w2 = Group::from_shape(&[144, 320, 48]).unwrap();
    let r1 = w2.part(Split::Row, 1, 2).unwrap();
    assert_eq!(r1.group.k_tiles, 72);
    assert_eq!(r1.trellis.len, 72 * 320 * 96);
    assert_eq!(r1.trellis.offset, r1.trellis.len);
    assert_eq!(r1.suh, Span::run(2 * 1152, 2 * 1152));
    assert_eq!(r1.svh.len, 2 * 5120);
    let w1 = Group::from_shape(&[320, 144, 32]).unwrap();
    let c1 = w1.part(Split::Col, 1, 2).unwrap();
    assert_eq!(c1.group.n_tiles, 72);
    assert_eq!(c1.trellis.count, 320);
    assert_eq!(c1.trellis.stride, 144 * 64);
    assert_eq!(c1.trellis.offset, 72 * 64);
    assert_eq!(c1.trellis.at(2), 72 * 64 + 2 * 144 * 64);
    assert_eq!(c1.group.trellis_bytes(), c1.trellis.bytes());
    assert_eq!(c1.bytes(), c1.trellis.bytes() + 2 * 5120 + 2 * 1152);
    // The 8-bit indexer key projection (512 -> 128) cannot split its 128 outputs over two ranks.
    let wk = Group::from_shape(&[32, 8, 128]).unwrap();
    assert_eq!(
        wk.part(Split::Col, 0, 2),
        Err(LayoutError::SplitNotOnBlocks)
    );
    assert_eq!(
        wk.trellis_bytes(),
        wk.part(Split::Whole, 1, 2).unwrap().trellis.bytes()
    );
    assert_eq!(
        wk.part(Split::Whole, 2, 2),
        Err(LayoutError::SplitNotOnBlocks)
    );
    assert_eq!(
        wk.part(Split::Whole, 0, 0),
        Err(LayoutError::SplitNotOnBlocks)
    );
}

#[test]
fn expert_layouts_stack_one_width_and_go_ragged_on_mixed_widths() {
    let two = Group::from_shape(&[144, 320, 32])
        .unwrap()
        .part(Split::Row, 0, 2)
        .unwrap();
    let three = Group::from_shape(&[144, 320, 48])
        .unwrap()
        .part(Split::Row, 0, 2)
        .unwrap();
    let stack = ExpertLayout::of(&[two, two, two]).unwrap();
    assert_eq!(stack.kind, ExpertLayoutKind::Stack);
    assert_eq!(stack.widths(), 1);
    assert_eq!(stack.count(), 3);
    // A mixed-bit pack's layer 2 down projection on rank 0: 383 experts at 2 bits and one at 3 bits.
    let mut parts = vec![two; 384];
    parts[200] = three;
    let rag = ExpertLayout::of(&parts).unwrap();
    assert_eq!(rag.kind, ExpertLayoutKind::Ragged);
    assert_eq!(rag.trellis_bytes(), 566_968_320);
    assert_eq!(rag.histogram()[4], 383);
    assert_eq!(rag.histogram()[6], 1);
    assert_eq!(rag.widths(), 2);
    assert_eq!(rag.byte_offset(201) % 16, 0);
    assert_eq!(
        rag.byte_offset(200) + three.trellis.bytes(),
        rag.byte_offset(201)
    );
    let odd = Group::from_shape(&[144, 160, 32])
        .unwrap()
        .part(Split::Row, 0, 2)
        .unwrap();
    assert_eq!(
        ExpertLayout::of(&[two, odd]),
        Err(LayoutError::ExpertShapesDiffer)
    );
    assert_eq!(ExpertLayout::of(&[]), Err(LayoutError::NoExperts));
}

#[test]
fn strips_put_each_column_blocks_tiles_in_k_order() {
    let mut rng = Rng::new(3);
    let (kt, nt, words) = (5usize, 16usize, 24usize);
    let stored = rng.i16s(kt * nt * words);
    let strips = layout::strips(&stored, kt, nt, words).unwrap();
    assert_eq!(strips.len(), stored.len());
    for nb in 0..nt / 8 {
        for k in 0..kt {
            for t in 0..8 {
                let dst = ((nb * kt + k) * 8 + t) * words;
                let src = (k * nt + nb * 8 + t) * words;
                assert_eq!(&strips[dst..dst + words], &stored[src..src + words]);
            }
        }
    }
    assert_eq!(
        layout::strips(&stored, kt, 12, words),
        Err(LayoutError::BadTrellis)
    );
    assert_eq!(
        layout::strips(&stored[1..], kt, nt, words),
        Err(LayoutError::BadTrellis)
    );
}

fn tensor(cb: Codebook, halves: u16, k: usize, n: usize, seed: u64, bias: bool) -> Exl3Tensor {
    let w = Width::from_halves(halves).unwrap();
    let g = synth_group(w, k, n, seed);
    Exl3Tensor::new(
        "t".into(),
        cb,
        w,
        k,
        n,
        g.trellis,
        g.suh,
        g.svh,
        bias.then_some(g.bias),
        ScaleForm::Fp16,
    )
    .unwrap()
}

#[test]
fn tensor_invariants_are_checked() {
    let t = tensor(Codebook::Mul1, 7, 256, 128, 1, true);
    assert_eq!(t.tiles_k(), 16);
    assert_eq!(t.tiles_n(), 8);
    assert_eq!(t.k2(), 7);
    assert_eq!(t.bpw_x100(), 350);
    assert_eq!(t.trellis_bytes(), 16 * 8 * 56 * 2);
    assert_eq!(
        t.device_bytes(),
        t.trellis_bytes() + 2 * (256 + 128) + 2 * 128
    );
    assert_eq!(t.group(), Group::from_shape(&[16, 8, 56]).unwrap());
    assert_eq!(t.tile(1, 2), &t.trellis[(8 + 2) * 56..(8 + 3) * 56]);
    let bad = |cb, k: usize, n: usize, trellis_len: usize, suh_len: usize| {
        Exl3Tensor::new(
            "b".into(),
            cb,
            Width::from_halves(7).unwrap(),
            k,
            n,
            vec![0; trellis_len],
            vec![0; suh_len],
            vec![0; n],
            None,
            ScaleForm::Fp16,
        )
    };
    assert!(bad(Codebook::Mul1, 128, 128, 8 * 8 * 56, 128).is_ok());
    assert!(
        bad(Codebook::Mcg, 128, 128, 8 * 8 * 56, 128).is_err(),
        "half width without mul1"
    );
    assert!(
        bad(Codebook::Mul1, 64, 128, 4 * 8 * 56, 64).is_err(),
        "K not a multiple of 128"
    );
    assert!(
        bad(Codebook::Mul1, 128, 128, 8 * 8 * 56 - 1, 128).is_err(),
        "short trellis"
    );
    assert!(
        bad(Codebook::Mul1, 128, 128, 8 * 8 * 56, 127).is_err(),
        "short suh"
    );
}

#[test]
fn byte_views_are_little_endian_and_strips_reorders() {
    let t = tensor(Codebook::Mcg, 4, 128, 256, 5, true);
    let bytes = t.trellis_le_bytes();
    assert_eq!(bytes.len(), t.trellis_bytes());
    assert_eq!(i16::from_le_bytes([bytes[2], bytes[3]]), t.trellis[1]);
    let suh = t.suh_le_bytes();
    assert_eq!(u16::from_le_bytes([suh[0], suh[1]]), t.suh[0]);
    assert_eq!(t.svh_le_bytes().len(), 2 * 256);
    assert_eq!(t.bias_le_bytes().map(|b| b.len()), Some(2 * 256));
    let strips = t.strips_le_bytes().unwrap();
    let words = layout::strips(&t.trellis, 8, 16, 32).unwrap();
    assert_eq!(i16::from_le_bytes([strips[0], strips[1]]), words[0]);
    assert_eq!(strips.len(), bytes.len());
}

#[test]
fn parts_dequantize_to_their_slice_of_the_parent() {
    let t = tensor(Codebook::Mul1, 5, 256, 256, 9, true);
    let col = t.part(Split::Col, 1, 2).unwrap();
    assert_eq!((col.k, col.n), (256, 128));
    assert_eq!(col.bias.as_deref(), t.bias.as_ref().map(|b| &b[128..]));
    let row = t.part(Split::Row, 1, 2).unwrap();
    assert_eq!((row.k, row.n), (128, 256));
    assert_eq!(row.bias, None, "only rank 0 of a row split keeps the bias");
    assert_eq!(t.part(Split::Row, 0, 2).unwrap().bias, t.bias);
    assert_eq!(t.part(Split::Whole, 1, 2).unwrap(), t);
    for kb in 0..2 {
        assert_eq!(
            col.dequantize_block(kb, 0).unwrap(),
            t.dequantize_block(kb, 1).unwrap()
        );
    }
    for nb in 0..2 {
        assert_eq!(
            row.dequantize_block(0, nb).unwrap(),
            t.dequantize_block(1, nb).unwrap()
        );
    }
    // The ranks' row-split partial outputs add up to the whole layer's.
    let mut rng = Rng::new(17);
    let x: Vec<f64> = (0..256).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let whole = t.matvec_f64(&x).unwrap();
    let r0 = t
        .part(Split::Row, 0, 2)
        .unwrap()
        .matvec_f64(&x[..128])
        .unwrap();
    let r1 = row.matvec_f64(&x[128..]).unwrap();
    for c in 0..256 {
        let sum = r0[c] + r1[c];
        assert!(
            (sum - whole[c]).abs() <= 1e-9 * (1.0 + whole[c].abs()),
            "column {c}"
        );
    }
    assert!(
        t.part(Split::Col, 0, 4).is_err(),
        "N = 256 cannot split over 4 ranks of 128"
    );
}
