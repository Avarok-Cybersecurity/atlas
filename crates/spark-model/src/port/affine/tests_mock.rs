// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the affine route on the mock backend: the constructor's launches and buffers, the
//! launch geometry and by-value descriptors of every decode path, the `W4Linear` contract (rows,
//! scratch, pair fusion and its fallbacks), partner recognition and release, and that a target without the Ada
//! modules issues no kernel lookup.

use anyhow::Result;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::kernels::{LANE_MODULE, PACK_MODULE, many_bytes, segs_bytes};
use super::quantize::check_flag;
use super::weight::{AffineWeight, lane_smem};
use super::{AffineKernels, affine_of, quantize_from_bf16, quantize_weight, release};
use crate::port::w4::{W4Box, W4Format, W4Linear, W4Opts, W4Scratch};

fn opts(group: u16) -> W4Opts {
    W4Opts {
        bits: 4,
        group,
        clip_search: 0,
        int8_prefill: false,
    }
}

fn source(gpu: &MockGpuBackend, n: usize, k: usize) -> DevicePtr {
    gpu.alloc(n * k * 2).unwrap()
}

fn scratch(gpu: &MockGpuBackend, bytes: usize) -> W4Scratch {
    W4Scratch {
        ws: gpu.alloc(bytes.max(1)).unwrap(),
        ws_bytes: bytes,
        tickets: DevicePtr::NULL,
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
    }
}

fn int_arg(l: &MockLaunch, i: usize) -> i32 {
    match &l.args[i] {
        MockArg::Bytes(b) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        a => panic!("arg {i} is {a:?}, not an int"),
    }
}

fn bytes_arg(l: &MockLaunch, i: usize) -> Vec<u8> {
    match &l.args[i] {
        MockArg::Bytes(b) => b.clone(),
        a => panic!("arg {i} is {a:?}, not bytes"),
    }
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn i32_at(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

#[test]
fn the_constructor_quantizes_packs_and_frees_its_staging() -> Result<()> {
    let gpu = MockGpuBackend::new();
    let (n, k) = (9216usize, 2560usize);
    let src = source(&gpu, n, k);
    let before = gpu.live_alloc_count();
    let w = quantize_weight(&gpu, src, n, k, &opts(64), 0)?;
    // Four buffers stay (codes, scales, biases, tickets); the four staging buffers are gone.
    assert_eq!(gpu.live_alloc_count(), before + 4);
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 4, "quantize, pack, two transposes");
    assert_eq!(l[0].grid, [(n * 40).div_ceil(4) as u32, 1, 1]);
    assert_eq!(l[0].block, [128, 1, 1]);
    assert_eq!(
        (int_arg(&l[0], 1), int_arg(&l[0], 2), int_arg(&l[0], 3)),
        (n as i32, k as i32, 0)
    );
    assert_eq!(
        l[1].grid,
        [(9216 * 2560 / 8usize).div_ceil(256) as u32, 1, 1]
    );
    assert_eq!(int_arg(&l[1], 4), 64, "pack group");
    assert_eq!((w.n, w.k, w.npad, w.group, w.bits), (n, k, 9216, 64, 4));
    let tickets = gpu.read_alloc(w.tickets).unwrap();
    assert!(tickets.iter().all(|b| *b == 0), "tickets start at zero");
    assert_eq!(tickets.len(), 4 * 144);
    w.release(&gpu)?;
    assert_eq!(gpu.live_alloc_count(), before);
    Ok(())
}

#[test]
fn the_constructor_refuses_bad_shapes_before_allocating() {
    let gpu = MockGpuBackend::new();
    let src = source(&gpu, 64, 96);
    let live = gpu.live_alloc_count();
    for (n, k, o) in [
        (64usize, 96usize, opts(64)), // K not a multiple of the group
        (64, 128, opts(16)),          // no such group
        (
            64,
            128,
            W4Opts {
                clip_search: 11,
                ..opts(64)
            },
        ),
        (0, 128, opts(64)),
    ] {
        assert!(
            quantize_from_bf16(&gpu, src, n, k, &o, 0).is_err(),
            "[{n}x{k}]"
        );
    }
    assert_eq!(gpu.live_alloc_count(), live);
    assert!(gpu.launches_snapshot().is_empty());
}

#[test]
fn the_flag_names_the_bad_group() {
    assert!(check_flag(u32::MAX, 8, 128, 64).is_ok());
    let e = check_flag(5, 8, 128, 64).unwrap_err().to_string();
    assert!(
        e.contains("group 5") && e.contains("row 2") && e.contains("inputs 64..128"),
        "{e}"
    );
}

#[test]
fn a_target_without_the_ada_modules_issues_no_lookup() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(LANE_MODULE);
    assert!(!AffineKernels::available(&gpu));
    assert!(AffineKernels::resolve(&gpu).is_err());
    let src = source(&gpu, 64, 128);
    assert!(quantize_from_bf16(&gpu, src, 64, 128, &opts(64), 0).is_err());
    assert!(gpu.kernel_lookups_snapshot().is_empty());
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(PACK_MODULE);
    assert!(!AffineKernels::available(&gpu));
}

fn boxed(gpu: &MockGpuBackend, n: usize, k: usize, group: u16) -> W4Box {
    let src = source(gpu, n, k);
    quantize_from_bf16(gpu, src, n, k, &opts(group), 0).unwrap()
}

#[test]
fn the_w4linear_face() {
    let gpu = MockGpuBackend::new();
    let b = boxed(&gpu, 2560, 9216, 64);
    assert_eq!(b.format(), W4Format::Affine { bits: 4, group: 64 });
    assert_eq!((b.n(), b.k(), b.max_decode_rows()), (2560, 9216, 16));
    let w = affine_of(&*b).unwrap();
    assert_eq!(b.device_bytes(), w.layout().total() + 4 * 40);
    // 16 rows of 144 group sums, then twice the partials of 40 tiles x 8 slices.
    assert_eq!(b.scratch_bytes(16), 16 * 144 * 4 + 2 * 40 * 8 * 16 * 64 * 4);
    assert_eq!(
        b.scratch_bytes(64),
        b.scratch_bytes(16),
        "decode never takes more than 16 rows"
    );
}

#[test]
fn decode_launches_the_sums_then_the_split_gemv() -> Result<()> {
    let gpu = MockGpuBackend::new();
    let b = boxed(&gpu, 2560, 9216, 64);
    let w: AffineWeight = affine_of(&*b).unwrap();
    let s = scratch(&gpu, b.scratch_bytes(16));
    let (x, out) = (gpu.alloc(4 * 9216 * 2)?, gpu.alloc(4 * 2560 * 2)?);
    let first = gpu.launches_snapshot().len();
    b.decode(&gpu, x, 4, out, &s, 7)?;
    let l = &gpu.launches_snapshot()[first..];
    assert_eq!(l.len(), 2);
    assert_eq!(
        (l[0].grid, l[0].block),
        ([4, 3, 1], [64, 1, 1]),
        "sums: rows x ceil(144/64)"
    );
    assert_eq!(
        l[0].args[1],
        MockArg::Buffer(s.ws),
        "sums at the workspace start"
    );
    assert_eq!(l[1].grid, [40 * 8, 1, 1], "a CTA per (tile, slice)");
    assert_eq!(l[1].shared_mem, lane_smem(64, 4));
    assert!(l.iter().all(|x| x.stream == 7));
    let part = bytes_arg(&l[1], 2);
    assert_eq!(part.len(), 48);
    assert_eq!(u64_at(&part, 0), w.words.0);
    assert_eq!(u64_at(&part, 24), out.0);
    assert_eq!(
        (
            i32_at(&part, 32),
            i32_at(&part, 36),
            i32_at(&part, 40),
            i32_at(&part, 44)
        ),
        (2560, 2560, 8, 40)
    );
    assert_eq!(int_arg(&l[1], 3), 4, "M");
    assert_eq!(
        l[1].args[6],
        MockArg::Buffer(s.ws.offset(w.sums_bytes(4))),
        "partials after the sums"
    );
    assert_eq!(l[1].args[7], MockArg::Buffer(w.tickets));
    Ok(())
}

#[test]
fn an_unsplit_shape_takes_the_persistent_lane_body() -> Result<()> {
    let gpu = MockGpuBackend::new();
    let b = boxed(&gpu, 12288, 2560, 64); // 192 tiles: sk = 1
    let s = scratch(&gpu, b.scratch_bytes(1));
    let (x, out) = (gpu.alloc(2560 * 2)?, gpu.alloc(12288 * 2)?);
    let first = gpu.launches_snapshot().len();
    b.decode(&gpu, x, 1, out, &s, 0)?;
    let l = &gpu.launches_snapshot()[first..];
    assert_eq!(l.len(), 2);
    assert_eq!(l[1].shared_mem, lane_smem(64, 8));
    // min(192 tiles, 2 CTAs an SM x 48 mock SMs).
    assert_eq!(l[1].grid, [96, 1, 1]);
    assert_eq!(l[1].args.len(), 6, "no partials, no tickets");
    Ok(())
}

#[test]
fn decode_refuses_bad_rows_scratch_and_alignment() -> Result<()> {
    let gpu = MockGpuBackend::new();
    let b = boxed(&gpu, 256, 512, 32);
    let s = scratch(&gpu, b.scratch_bytes(16));
    let (x, out) = (gpu.alloc(17 * 512 * 2)?, gpu.alloc(17 * 256 * 2)?);
    assert!(b.decode(&gpu, x, 0, out, &s, 0).is_err());
    assert!(b.decode(&gpu, x, 17, out, &s, 0).is_err());
    assert!(
        b.decode(&gpu, x.offset(8), 1, out, &s, 0).is_err(),
        "x must be 16-byte aligned"
    );
    let small = W4Scratch {
        ws_bytes: 64,
        ..scratch(&gpu, 64)
    };
    assert!(b.decode(&gpu, x, 16, out, &small, 0).is_err());
    assert!(b.decode(&gpu, x, 16, out, &s, 0).is_ok());
    Ok(())
}

#[test]
fn a_pair_of_affine_weights_is_one_sums_and_one_seg_launch() -> Result<()> {
    let gpu = MockGpuBackend::new();
    let (gate, up) = (boxed(&gpu, 9216, 2560, 64), boxed(&gpu, 9216, 2560, 64));
    let (ga, ua) = (affine_of(&*gate).unwrap(), affine_of(&*up).unwrap());
    let s = scratch(&gpu, gate.scratch_bytes(2));
    let x = gpu.alloc(2 * 2560 * 2)?;
    let (oa, ob) = (gpu.alloc(2 * 9216 * 2)?, gpu.alloc(2 * 9216 * 2)?);
    let first = gpu.launches_snapshot().len();
    gate.decode_pair(&*up, &gpu, x, 2, oa, ob, &s, 0)?;
    let l = &gpu.launches_snapshot()[first..];
    assert_eq!(l.len(), 2);
    assert_eq!(
        l[1].grid,
        [2 * 144 * 2, 1, 1],
        "both members' (tile, slice) CTAs"
    );
    let segs = bytes_arg(&l[1], 2);
    assert_eq!(segs.len(), 264);
    assert_eq!(i32_at(&segs, 256), 2, "count");
    // Member 0 and 1: their own tickets, member 1's blocks and partials after member 0's.
    assert_eq!(u64_at(&segs, 48), ga.tickets.0);
    assert_eq!((i32_at(&segs, 56), i32_at(&segs, 60)), (0, 0));
    assert_eq!(u64_at(&segs, 64 + 48), ua.tickets.0);
    assert_eq!((i32_at(&segs, 64 + 56), i32_at(&segs, 64 + 60)), (288, 288));
    assert_eq!(u64_at(&segs, 64 + 24), ob.0);
    Ok(())
}

#[test]
fn a_pair_without_room_for_both_partials_runs_two_gemvs_over_one_sums() -> Result<()> {
    let gpu = MockGpuBackend::new();
    let (gate, up) = (boxed(&gpu, 9216, 2560, 64), boxed(&gpu, 9216, 2560, 64));
    let ga = affine_of(&*gate).unwrap();
    let s = scratch(&gpu, ga.sums_bytes(1) + ga.work_bytes());
    let x = gpu.alloc(2560 * 2)?;
    let (oa, ob) = (gpu.alloc(9216 * 2)?, gpu.alloc(9216 * 2)?);
    let first = gpu.launches_snapshot().len();
    gate.decode_pair(&*up, &gpu, x, 1, oa, ob, &s, 0)?;
    let l = &gpu.launches_snapshot()[first..];
    assert_eq!(l.len(), 3, "one sums, two split GEMVs");
    assert_eq!(l[1].args[1], l[2].args[1], "the same sums");
    Ok(())
}

/// A partner of another format: decode_pair must fall back to two plain decodes.
struct Other {
    calls: std::sync::atomic::AtomicUsize,
}

impl W4Linear for Other {
    fn format(&self) -> W4Format {
        W4Format::Nvfp4
    }
    fn n(&self) -> usize {
        9216
    }
    fn k(&self) -> usize {
        2560
    }
    fn device_bytes(&self) -> usize {
        0
    }
    fn max_decode_rows(&self) -> usize {
        16
    }
    fn scratch_bytes(&self, _max_rows: usize) -> usize {
        0
    }
    fn decode(
        &self,
        _gpu: &dyn GpuBackend,
        _x: DevicePtr,
        _rows: usize,
        _out: DevicePtr,
        _s: &W4Scratch,
        _stream: u64,
    ) -> Result<()> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
    fn prefill(
        &self,
        _gpu: &dyn GpuBackend,
        _x: DevicePtr,
        _m: usize,
        _out: DevicePtr,
        _s: &W4Scratch,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }
    fn dequant_bf16(&self, _gpu: &dyn GpuBackend, _out: DevicePtr, _stream: u64) -> Result<()> {
        Ok(())
    }
}

#[test]
fn a_foreign_partner_or_the_same_weight_falls_back_to_two_decodes() -> Result<()> {
    let gpu = MockGpuBackend::new();
    let gate = boxed(&gpu, 9216, 2560, 64);
    let other = Other {
        calls: Default::default(),
    };
    let s = scratch(&gpu, 2 * gate.scratch_bytes(1));
    let x = gpu.alloc(2560 * 2)?;
    let (oa, ob) = (gpu.alloc(9216 * 2)?, gpu.alloc(9216 * 2)?);
    let first = gpu.launches_snapshot().len();
    gate.decode_pair(&other, &gpu, x, 1, oa, ob, &s, 0)?;
    assert_eq!(
        gpu.launches_snapshot().len() - first,
        2,
        "the affine decode alone"
    );
    assert_eq!(other.calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    // The same weight twice cannot share its tickets in one launch: two full decodes.
    let first = gpu.launches_snapshot().len();
    gate.decode_pair(&*gate, &gpu, x, 1, oa, ob, &s, 0)?;
    assert_eq!(gpu.launches_snapshot().len() - first, 4);
    Ok(())
}

#[test]
fn release_is_idempotent_and_a_released_weight_refuses_calls() -> Result<()> {
    let gpu = MockGpuBackend::new();
    let live = gpu.live_alloc_count();
    let b = boxed(&gpu, 256, 512, 64);
    assert!(affine_of(&*b).is_some());
    assert!(
        b.as_any().is_some(),
        "an affine box can be recognised by a partner"
    );
    assert!(release(&gpu, &*b)?);
    assert!(!release(&gpu, &*b)?, "a second release frees nothing");
    // The source stays (the caller frees it): one allocation more than before.
    assert_eq!(gpu.live_alloc_count(), live + 1);
    assert!(affine_of(&*b).is_none(), "a released weight is no partner");
    let s = scratch(&gpu, 1 << 20);
    let (x, out) = (gpu.alloc(512 * 2)?, gpu.alloc(256 * 2)?);
    assert!(b.decode(&gpu, x, 1, out, &s, 0).is_err());
    assert!(b.dequant_bf16(&gpu, out, 0).is_err());
    // A foreign weight is not affine and release leaves it alone.
    let other = Other {
        calls: Default::default(),
    };
    assert!(affine_of(&other).is_none());
    assert!(!release(&gpu, &other)?);
    Ok(())
}

#[test]
fn descriptor_images_have_the_cuda_layout() {
    let gpu = MockGpuBackend::new();
    let src = source(&gpu, 256, 512);
    let w = quantize_weight(&gpu, src, 256, 512, &opts(64), 0).unwrap();
    let (segs, blocks, work) = segs_bytes(&[(&w, DevicePtr(0x1000)), (&w, DevicePtr(0x2000))]);
    assert_eq!(segs.len(), 264);
    let sk = w.sk();
    assert_eq!(blocks, 2 * w.tiles() * sk);
    assert_eq!(work, if sk > 1 { 2 * w.tiles() * sk } else { 0 });
    let (many, mblocks) = many_bytes(&[(&w, DevicePtr(0x1000))], 3);
    assert_eq!(many.len(), 232);
    assert_eq!(
        (i32_at(&many, 48), i32_at(&many, 52)),
        (0, 3.min(w.tiles() as i32))
    );
    assert_eq!(i32_at(&many, 224), 1);
    assert_eq!(mblocks, w.tiles().min(3));
}
