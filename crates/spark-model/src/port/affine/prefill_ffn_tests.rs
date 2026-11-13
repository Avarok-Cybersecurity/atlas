// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the int8 dense FFN prompt path on the mock backend (lane L17b): the six launches a chunk in
//! order, the chunks the per-projection path would take, h written over the gate output and read by the down
//! projection's fallback, and every decline launching nothing.

use spark_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::port::affine::prefill::encode_parts;
use crate::port::affine::quantize_from_bf16;
use crate::port::w4::{W4Box, W4Opts, W4Scratch, select};

const HIDDEN: usize = 128;
const INTER: usize = 192;

fn weight(gpu: &MockGpuBackend, n: usize, k: usize, int8: bool) -> W4Box {
    let src = gpu.alloc(n * k * 2).unwrap();
    let opts = W4Opts {
        bits: 4,
        group: 64,
        clip_search: 0,
        int8_prefill: int8,
    };
    quantize_from_bf16(gpu, src, n, k, &opts, 0).unwrap()
}

struct Layer {
    gpu: MockGpuBackend,
    gate: W4Box,
    up: W4Box,
    down: W4Box,
    s: W4Scratch,
}

fn layer(int8: bool, rows: usize) -> Layer {
    let gpu = MockGpuBackend::new();
    let gate = weight(&gpu, INTER, HIDDEN, int8);
    let up = weight(&gpu, INTER, HIDDEN, int8);
    let down = weight(&gpu, HIDDEN, INTER, int8);
    let s = W4Scratch {
        ws: gpu.alloc(256).unwrap(),
        ws_bytes: 256,
        tickets: DevicePtr::NULL,
        xq: gpu.alloc(rows * INTER).unwrap(),
        xs: gpu.alloc(rows * INTER / 8).unwrap(),
    };
    Layer {
        gpu,
        gate,
        up,
        down,
        s,
    }
}

fn io(gpu: &MockGpuBackend, rows: usize) -> FfnIo {
    FfnIo {
        x: gpu.alloc(rows * HIDDEN * 2).unwrap(),
        rows,
        gate_out: gpu.alloc(rows * INTER * 2).unwrap(),
        up_out: gpu.alloc(rows * INTER * 2).unwrap(),
        out: gpu.alloc(rows * HIDDEN * 2).unwrap(),
    }
}

fn ptr_at(l: &MockLaunch, i: usize) -> DevicePtr {
    match &l.args[i] {
        MockArg::Buffer(p) => *p,
        a => panic!("arg {i} is {a:?}, not a pointer"),
    }
}

fn int_at(l: &MockLaunch, i: usize) -> i32 {
    match &l.args[i] {
        MockArg::Bytes(b) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        a => panic!("arg {i} is {a:?}, not an int"),
    }
}

fn run(l: &Layer, io: FfnIo) -> bool {
    ffn_int8(&*l.gate, &*l.up, &*l.down, &l.gpu, io, &l.s, 3).unwrap()
}

#[test]
fn six_launches_a_chunk_in_the_per_projection_chunks() {
    let rows = 4100;
    let l = layer(true, rows);
    let io = io(&l.gpu, rows);
    let before = l.gpu.launches_snapshot().len();
    assert!(run(&l, io));
    let launches = l.gpu.launches_snapshot()[before..].to_vec();
    let chunks = select::prefill_chunks(rows, select::prefill_chunk_cap(16), 16);
    assert_eq!(chunks.len(), 3, "4100 rows at a 2048-row cap");
    assert_eq!(launches.len(), 6 * chunks.len());
    let f = flags(&l.gpu, 3).unwrap();
    for (c, (first, len)) in chunks.iter().copied().enumerate() {
        let k = &launches[6 * c..6 * c + 6];
        let arity: Vec<usize> = k.iter().map(|x| x.args.len()).collect();
        assert_eq!(
            arity,
            vec![7, 8, 7, 10, 13, 12],
            "quantize, int8 many, bf16 many, swiglu, int8, bf16"
        );
        let gc = io.gate_out.offset(first * INTER * 2);
        let uc = io.up_out.offset(first * INTER * 2);
        // One quantization of this chunk's x serves both projections.
        assert_eq!(ptr_at(&k[0], 0), io.x.offset(first * HIDDEN * 2));
        assert_eq!(k[0].grid, [len as u32, (HIDDEN / 64).div_ceil(4) as u32, 1]);
        // The fused SwiGLU reads gate and up and writes h over gate, with the layer's strides.
        assert_eq!(
            (ptr_at(&k[3], 0), ptr_at(&k[3], 1), ptr_at(&k[3], 2)),
            (gc, uc, gc)
        );
        assert_eq!(
            (int_at(&k[3], 3), int_at(&k[3], 4), int_at(&k[3], 8)),
            (INTER as i32, INTER as i32, INTER as i32)
        );
        // The down projection's int8 GEMM writes this chunk's rows; its BF16 fallback reads h.
        assert_eq!(ptr_at(&k[4], 6), io.out.offset(first * HIDDEN * 2));
        assert_eq!(int_at(&k[4], 7), len as i32);
        assert_eq!((ptr_at(&k[5], 0), int_at(&k[5], 9)), (gc, INTER as i32));
        for (i, x) in k.iter().enumerate() {
            assert_eq!(x.stream, 3);
            let last = x.args.len() - 1;
            assert_eq!(
                ptr_at(x, last),
                f,
                "launch {i} of chunk {c} raises or reads the flags"
            );
        }
    }
}

#[test]
fn declines_launch_nothing() {
    let rows = 300;
    // Not wider than the decode window.
    let l = layer(true, rows);
    let n0 = l.gpu.launches_snapshot().len();
    assert!(!run(&l, io(&l.gpu, 16)));
    assert_eq!(l.gpu.launches_snapshot().len(), n0);
    // The int8 row off on the weights.
    let off = layer(false, rows);
    let n1 = off.gpu.launches_snapshot().len();
    assert!(!run(&off, io(&off.gpu, rows)));
    assert_eq!(off.gpu.launches_snapshot().len(), n1);
    // Shapes that do not chain (down takes the gate's output width); its quantize launches come first.
    let bad = weight(&l.gpu, HIDDEN, HIDDEN, true);
    let n2 = l.gpu.launches_snapshot().len();
    let i = io(&l.gpu, rows);
    assert!(!ffn_int8(&*l.gate, &*l.up, &*bad, &l.gpu, i, &l.s, 0).unwrap());
    // No int8 scratch.
    let bare = W4Scratch {
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
        ..l.s
    };
    assert!(!ffn_int8(&*l.gate, &*l.up, &*l.down, &l.gpu, i, &bare, 0).unwrap());
    // A misaligned input.
    let skew = FfnIo {
        x: i.x.offset(2),
        ..i
    };
    assert!(!run(&l, skew));
    assert_eq!(
        l.gpu.launches_snapshot().len(),
        n2,
        "no decline launched anything"
    );
    // A format this path does not serve.
    let log = crate::port::w4::fake::log();
    let fake = crate::port::w4::fake::FakeW4::new("gate", INTER, HIDDEN, 16, &log);
    assert!(!ffn_int8(&fake, &*l.up, &*l.down, &l.gpu, i, &l.s, 0).unwrap());
    assert_eq!(l.gpu.launches_snapshot().len(), n2);
}

#[test]
fn absent_modules_decline_without_a_prompt_lookup() {
    let l = layer(true, 64);
    l.gpu.mark_module_absent(launch::GEMM_MODULE);
    l.gpu.mark_module_absent(launch::ACT_MODULE);
    let lookups = l.gpu.kernel_lookups_snapshot().len();
    let launches = l.gpu.launches_snapshot().len();
    assert!(!run(&l, io(&l.gpu, 64)));
    let after = l.gpu.kernel_lookups_snapshot();
    assert!(
        after[lookups..]
            .iter()
            .all(|(m, _)| m != launch::GEMM_MODULE && m != launch::ACT_MODULE),
        "{after:?}"
    );
    assert_eq!(l.gpu.launches_snapshot().len(), launches);
}

#[test]
fn the_fused_path_is_byte_shaped_like_the_unfused_one() {
    // The same chunk through the per-projection path takes, per projection, the quantizer and the int8 GEMM over
    // the same rows; here the descriptors of the `_many` grid name both outputs with the chunk's offsets.
    let rows = 40;
    let l = layer(true, rows);
    let io = io(&l.gpu, rows);
    let before = l.gpu.launches_snapshot().len();
    assert!(run(&l, io));
    let k = l.gpu.launches_snapshot()[before..].to_vec();
    let g = Packed::of(&crate::port::affine::affine_of(&*l.gate).unwrap()).unwrap();
    let u = Packed::of(&crate::port::affine::affine_of(&*l.up).unwrap()).unwrap();
    let (parts, blocks) = encode_parts(&[(g, io.gate_out), (u, io.up_out)], rows).unwrap();
    assert_eq!(k[1].grid, [blocks as u32, 1, 1]);
    assert!(k[1].args.contains(&MockArg::Bytes(parts.to_vec())));
    assert!(k[2].args.contains(&MockArg::Bytes(parts.to_vec())));
}
