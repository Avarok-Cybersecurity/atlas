// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend tests of the `marlin_w4a16` loader route (`route.rs`): the row-off
//! and module-absent paths touch nothing, the shape rule agrees with what `build` accepts, an
//! eligible projection becomes a g128 `kU4B8` weight equal to the CPU quantizer, and the ledger
//! helpers (buffers, bytes, disown) are consistent with the weight.
#![allow(clippy::useless_vec)]

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;
use crate::port::marlin::quantize::{MAX_CLIP_SEARCH, RtnOpts, rtn_int4_sym};
use crate::port::marlin::reference::f32_to_bf16;
use crate::port::marlin::repack_tests::{Rng, random_logical};
use crate::port::marlin::{build, release};
use crate::port::w4::W4Format;

fn bf16_weight(gpu: &MockGpuBackend, n: usize, k: usize, seed: u64) -> (Vec<u16>, DevicePtr) {
    let mut r = Rng::new(seed);
    let w: Vec<u16> = (0..n * k).map(|_| f32_to_bf16(0.1 * r.unit())).collect();
    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let src = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(&bytes, src).unwrap();
    (w, src)
}

#[test]
fn the_row_off_path_touches_nothing() {
    let gpu = MockGpuBackend::new();
    let (_, src) = bf16_weight(&gpu, 256, 512, 1);
    let allocs = gpu.live_alloc_count();
    let got = quantize_projection_with(&gpu, false, src, 256, 512, 0, 0, "t").unwrap();
    assert!(got.is_none());
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "no kernel lookup with the row off"
    );
    assert!(gpu.launches_snapshot().is_empty());
    assert_eq!(gpu.live_alloc_count(), allocs, "nothing allocated");
    assert_eq!(eligibility(&gpu, false, 256, 512), Err(Decline::RowOff));
}

#[test]
fn a_target_without_the_module_keeps_the_caller_quantizer() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(kernels::MODULE);
    let (_, src) = bf16_weight(&gpu, 256, 512, 2);
    let got = quantize_projection_with(&gpu, true, src, 256, 512, 0, 0, "t").unwrap();
    assert!(got.is_none());
    assert_eq!(
        eligibility(&gpu, true, 256, 512),
        Err(Decline::ModuleAbsent)
    );
    assert!(gpu.kernel_lookups_snapshot().is_empty());
}

#[test]
fn the_shape_rule_matches_what_build_accepts() {
    // The 4B's projection shapes (FFN up and down, GDN in-projection, attention q): eligible. Too
    // large to build in a debug test, so only the pure rule is asked.
    for (n, k) in [(9216, 2560), (2560, 9216), (12288, 2560), (4096, 2560)] {
        assert!(shape_ok(n, k, MarlinType::U4B8).is_ok(), "[{n}x{k}]");
        assert!(
            shape_ok(n, k, MarlinType::U4).is_ok(),
            "[{n}x{k}] with zero points"
        );
    }
    let shapes = [
        (256, 512),
        (192, 512),
        (320, 384),
        (100, 2560),
        (256, 128),
        (256, 200),
        (128, 192),
        (64, 256),
    ];
    for (n, k) in shapes {
        let gpu = MockGpuBackend::new();
        let pure = shape_ok(n, k, MarlinType::U4B8);
        let built = if k.is_multiple_of(DEFAULT_GROUP) && n.is_multiple_of(8) {
            build(
                &gpu,
                &random_logical(n, k, DEFAULT_GROUP, false, (n * k) as u64),
                0,
            )
            .map(|_| ())
        } else {
            Err(anyhow::anyhow!("not a g128 shape"))
        };
        assert_eq!(
            pure.is_ok(),
            built.is_ok(),
            "[{n}x{k}] shape_ok {pure:?} vs build {built:?}"
        );
        let e = eligibility(&gpu, true, n, k);
        assert_eq!(e.is_ok(), pure.is_ok(), "[{n}x{k}]");
        if let Err(d) = e {
            assert_eq!(d.label(), "shape");
        }
    }
    // The planted controls of the rule: one group, a ragged N.
    assert!(
        shape_ok(256, 128, MarlinType::U4B8)
            .unwrap_err()
            .contains("single group")
    );
    assert!(shape_ok(100, 2560, MarlinType::U4B8).is_err());
}

#[test]
fn an_eligible_projection_is_the_cpu_rtn_weight() {
    let gpu = MockGpuBackend::new();
    let (n, k) = (256usize, 512usize);
    let (w, src) = bf16_weight(&gpu, n, k, 3);
    for clip in [0u8, 3] {
        let b = quantize_projection_with(&gpu, true, src, n, k, clip, 0, "t")
            .unwrap()
            .expect("an eligible shape");
        assert_eq!(
            b.format(),
            W4Format::MarlinInt4 {
                group: 128,
                zero_point: false
            }
        );
        assert_eq!(b.max_decode_rows(), 64);
        let out = gpu.alloc(n * k * 2).unwrap();
        b.dequant_bf16(&gpu, out, 0).unwrap();
        let got: Vec<u16> = gpu
            .read_alloc(out)
            .unwrap()
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let want = rtn_int4_sym(
            &w,
            n,
            k,
            &RtnOpts {
                group: 128,
                clip_search: clip,
            },
        )
        .unwrap()
        .dequant_nk();
        assert_eq!(got, want, "clip {clip}");
        assert!(
            gpu.read_alloc(src).is_some(),
            "the BF16 source is the caller's to free"
        );
        assert!(release(&gpu, &*b).unwrap());
    }
}

#[test]
fn ledger_helpers_agree_with_the_weight() {
    let gpu = MockGpuBackend::new();
    for zp in [false, true] {
        let l = random_logical(256, 512, 128, zp, 17);
        let b = super::super::from_logical(&gpu, &l, 0).unwrap();
        let bufs = buffers_of(&*b).expect("a Marlin weight");
        assert_eq!(bufs.len(), if zp { 4 } else { 3 });
        assert!(bufs.iter().all(|(p, bytes)| !p.is_null() && *bytes > 0));
        for &(p, bytes) in &bufs {
            assert_eq!(gpu.read_alloc(p).unwrap().len(), bytes, "buffer size");
        }
        let total: usize = bufs.iter().map(|&(_, bytes)| bytes).sum();
        let sms = gpu.sm_count().unwrap() as usize;
        assert_eq!(total, marlin_bytes(256, 512, zp, sms));
        assert_eq!(total, b.device_bytes());
        // Disown: the ledger frees, release does not.
        assert!(disown(&*b));
        assert!(!disown(&*b), "a second disown finds nothing");
        assert!(
            !release(&gpu, &*b).unwrap(),
            "release of a disowned weight frees nothing"
        );
        assert!(gpu.read_alloc(bufs[0].0).is_some());
        for (p, _) in bufs {
            gpu.free(p).unwrap();
        }
    }
}

#[test]
fn other_w4_weights_are_not_marlin() {
    let log = crate::port::w4::fake::log();
    let f = crate::port::w4::fake::FakeW4::new("fake", 64, 128, 16, &log);
    assert!(marlin_of(&f).is_none());
    assert!(buffers_of(&f).is_none());
    assert!(!disown(&f));
}

#[test]
fn rtn_options_are_g128_int4_and_clamp_the_clip_search() {
    let o = rtn_opts(2);
    assert_eq!(
        (o.bits, o.group, o.clip_search, o.int8_prefill),
        (4, 128, 2, false)
    );
    assert_eq!(rtn_opts(200).clip_search, MAX_CLIP_SEARCH);
    assert!(super::super::check_rtn_opts(9216, 2560, &rtn_opts(200)).is_ok());
    assert!(in_table(
        MarlinType::U4B8,
        3,
        false,
        &config::UNDERFILL_CONFIG
    ));
    assert!(!in_table(
        MarlinType::U4B8,
        3,
        true,
        &config::UNDERFILL_CONFIG
    ));
}
