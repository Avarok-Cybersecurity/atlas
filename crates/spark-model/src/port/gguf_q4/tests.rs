// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the keep-packed GGUF route (port lane L19): the byte arithmetic, every
//! `W4Linear` entry point's launches on the mock backend (geometry, scratch rules, the pair fusion, the
//! decline), the constructors' validation and ownership, and the CPU oracle with its planted controls.

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::dequant_cpu::GgmlType;
use spark_runtime::weights::{WeightDtype, WeightTensor};

use super::launch::{self, GGUF_Q4_MMQ_SMEM, MmvqShape, Q4Type};
use super::reference::{
    Q8Block, bf16_to_f32, dequant_weight, gemv_f64, grade, q8_1_values, quantize_q8_1_row,
    within_bound,
};
use super::*;
use crate::port::w4::{W4Format, W4Scratch};

/// xorshift64*: deterministic bytes.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 56) as u8
    }
}

fn bf16(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

/// Random packed bytes of `[n, k]` whose f16 scales are small and finite.
fn random_weight(rng: &mut Rng, ty: Q4Type, n: usize, k: usize) -> Vec<u8> {
    let bb = ty.block_bytes();
    let mut v = Vec::with_capacity(n * (k / 256) * bb);
    for _ in 0..n * (k / 256) {
        let mut b: Vec<u8> = (0..bb).map(|_| rng.byte()).collect();
        let scale = half::f16::from_f32(0.001 + 0.01 * rng.unit().abs())
            .to_bits()
            .to_le_bytes();
        match ty {
            // d at the front; Q4_K also dmin.
            Q4Type::Iq4Xs => b[0..2].copy_from_slice(&scale),
            Q4Type::Q4K => {
                b[0..2].copy_from_slice(&scale);
                b[2..4].copy_from_slice(&scale);
            }
            // Q6_K keeps d at the end.
            Q4Type::Q6K => b[208..210].copy_from_slice(&scale),
        }
        v.extend_from_slice(&b);
    }
    v
}

fn scratch(ws: DevicePtr, ws_bytes: usize) -> W4Scratch {
    W4Scratch {
        ws,
        ws_bytes,
        tickets: DevicePtr::NULL,
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
    }
}

const TYPES: [(Q4Type, GgmlType); 3] = [
    (Q4Type::Iq4Xs, GgmlType::Iq4Xs),
    (Q4Type::Q4K, GgmlType::Q4K),
    (Q4Type::Q6K, GgmlType::Q6K),
];

#[test]
fn byte_arithmetic_matches_the_block_geometry() {
    // The 4B FFN gate: [9216, 2560].
    assert_eq!(
        launch::weight_bytes(Q4Type::Iq4Xs, 9216, 2560),
        9216 * 10 * 136
    );
    assert_eq!(
        launch::weight_bytes(Q4Type::Q4K, 9216, 2560),
        9216 * 10 * 144
    );
    assert_eq!(
        launch::weight_bytes(Q4Type::Q6K, 9216, 2560),
        9216 * 10 * 210
    );
    assert_eq!(launch::q8_rows_bytes(8, 2560), 8 * 80 * 36);
    assert_eq!(launch::mmq_act_bytes(1, 2560), 128 * 20 * 144);
    assert_eq!(launch::mmq_act_bytes(129, 2560), 256 * 20 * 144);
    // Decode-only scratch is the rows buffer; with prefill rows it is the MMQ buffer.
    assert_eq!(
        launch::scratch_bytes(2560, 4),
        (4 * 80 * 36usize).div_ceil(256) * 256
    );
    assert_eq!(launch::scratch_bytes(2560, 2048), 2048 * 20 * 144);
    assert_eq!(GGUF_Q4_MMQ_SMEM, 512 + 128 * 76 * 4 + 128 * 144);
}

#[test]
fn entry_names_follow_the_kernel_files() {
    assert_eq!(
        Q4Type::Iq4Xs.mmvq_entry(MmvqShape::Block),
        "atlas_mmvq_iq4_xs_ada"
    );
    assert_eq!(
        Q4Type::Q4K.mmvq_entry(MmvqShape::Warp),
        "atlas_mmvq_q4_k_w_ada"
    );
    assert_eq!(
        Q4Type::Q6K.mmq_entries(),
        (
            "atlas_q6_k_mmq128_nc".to_string(),
            "atlas_q6_k_mmq128_wc".to_string()
        )
    );
    assert_eq!(Q4Type::Q4K.mmq_act_entry(), "atlas_q8_1_mmq_ds4_bf16_ada");
    assert_eq!(Q4Type::Iq4Xs.mmq_act_entry(), "atlas_q8_1_mmq_d4_bf16_ada");
    assert_eq!(Q4Type::Q6K.mmq_act_entry(), "atlas_q8_1_mmq_d4_bf16_ada");
    assert_eq!(Q4Type::Iq4Xs.dequant_entry().0, launch::DEQUANT_ADA_MODULE);
    assert_eq!(Q4Type::Q6K.dequant_entry().0, launch::DEQUANT_GB10_MODULE);
    assert_eq!(
        TYPES.map(|(q, _)| q.ggml_id()),
        [23, 12, 14],
        "ggml ids of IQ4_XS, Q4_K, Q6_K"
    );
}

#[test]
fn constructors_validate_type_shape_and_size() {
    let gpu = MockGpuBackend::new();
    let raw = vec![0u8; 2 * 136];
    assert!(from_gguf_tensor(&gpu, GgmlType::Q8_0, &raw, 1, 512, 0).is_err());
    assert!(from_gguf_tensor(&gpu, GgmlType::Iq4Xs, &raw, 1, 500, 0).is_err());
    assert!(from_gguf_tensor(&gpu, GgmlType::Iq4Xs, &raw, 0, 512, 0).is_err());
    assert!(from_gguf_tensor(&gpu, GgmlType::Iq4Xs, &raw[1..], 1, 512, 0).is_err());
    assert!(from_device_blocks(&gpu, GgmlType::Q4K, DevicePtr::NULL, 1, 256).is_err());
}

#[test]
fn a_target_without_the_ada_modules_errors_before_any_lookup() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(launch::MMVQ_MODULE);
    let raw = vec![0u8; 136];
    let err = from_gguf_tensor(&gpu, GgmlType::Iq4Xs, &raw, 1, 256, 0).unwrap_err();
    assert!(format!("{err:#}").contains("not in this build"), "{err:#}");
    assert!(gpu.kernel_lookups_snapshot().is_empty());
    assert!(!launch::available(&gpu, Q4Type::Iq4Xs));
}

#[test]
fn an_uploaded_weight_reports_its_contract_and_holds_the_file_bytes() {
    let gpu = MockGpuBackend::new();
    let mut rng = Rng(19);
    for (q, ty) in TYPES {
        let raw = random_weight(&mut rng, q, 3, 512);
        let w = from_gguf_tensor(&gpu, ty, &raw, 3, 512, 0).unwrap();
        let want_format = match q {
            Q4Type::Iq4Xs => W4Format::GgufIq4Xs,
            Q4Type::Q4K => W4Format::GgufQ4K,
            Q4Type::Q6K => W4Format::GgufQ6K,
        };
        assert_eq!(w.format(), want_format);
        assert_eq!((w.n(), w.k(), w.max_decode_rows()), (3, 512, 8));
        assert_eq!(w.device_bytes(), raw.len());
        let d = gguf_of(&*w).expect("registered");
        assert!(d.owned);
        assert_eq!(gpu.read_alloc(d.blocks).unwrap(), raw);
    }
}

#[test]
fn decode_quantizes_then_runs_one_gemv_and_checks_its_bounds() {
    let gpu = MockGpuBackend::new();
    let (n, k) = (300, 1024);
    let raw = vec![0u8; launch::weight_bytes(Q4Type::Q4K, n, k)];
    let w = from_gguf_tensor(&gpu, GgmlType::Q4K, &raw, n, k, 0).unwrap();
    let ws = gpu.alloc(w.scratch_bytes(8)).unwrap();
    let s = scratch(ws, w.scratch_bytes(8));
    let (x, out) = (gpu.alloc(8 * k * 2).unwrap(), gpu.alloc(8 * n * 2).unwrap());
    let before = gpu.launches_snapshot().len();
    w.decode(&gpu, x, 5, out, &s, 0).unwrap();
    let l = &gpu.launches_snapshot()[before..];
    assert_eq!(l.len(), 2);
    // q8_1 rows: 5 rows x 32 blocks, one warp a block, 128 threads a CUDA block.
    assert_eq!((l[0].grid, l[0].block), ([40, 1, 1], [128, 1, 1]));
    // mmvq: one block of four warps per output row.
    assert_eq!((l[1].grid, l[1].block), ([300, 1, 1], [32, 4, 1]));

    assert!(w.decode(&gpu, x, 0, out, &s, 0).is_err());
    assert!(w.decode(&gpu, x, 9, out, &s, 0).is_err());
    let small = scratch(ws, launch::q8_rows_bytes(5, k) - 1);
    assert!(w.decode(&gpu, x, 5, out, &small, 0).is_err());
}

#[test]
fn prefill_runs_the_mmq_engine_or_declines_when_the_workspace_is_short() {
    let gpu = MockGpuBackend::new();
    let (n, k, m) = (200, 512, 300);
    let raw = vec![0u8; launch::weight_bytes(Q4Type::Iq4Xs, n, k)];
    let w = from_gguf_tensor(&gpu, GgmlType::Iq4Xs, &raw, n, k, 0).unwrap();
    let bytes = w.scratch_bytes(m);
    let ws = gpu.alloc(bytes).unwrap();
    let (x, out) = (gpu.alloc(m * k * 2).unwrap(), gpu.alloc(m * n * 2).unwrap());
    let before = gpu.launches_snapshot().len();
    assert!(w.prefill(&gpu, x, m, out, &scratch(ws, bytes), 0).unwrap());
    let l = &gpu.launches_snapshot()[before..];
    assert_eq!(l.len(), 2);
    assert_eq!((l[0].grid, l[0].block), ([300, 1, 1], [128, 1, 1]));
    assert_eq!((l[1].grid, l[1].block), ([2, 3, 1], [32, 8, 1]));
    assert_eq!(l[1].shared_mem, GGUF_Q4_MMQ_SMEM);

    let before = gpu.launches_snapshot().len();
    let short = scratch(ws, launch::mmq_act_bytes(m, k) - 1);
    assert!(!w.prefill(&gpu, x, m, out, &short, 0).unwrap());
    assert!(
        !w.prefill(&gpu, x, m, out, &scratch(DevicePtr::NULL, usize::MAX), 0)
            .unwrap()
    );
    assert_eq!(
        gpu.launches_snapshot().len(),
        before,
        "a decline launches nothing"
    );
}

#[test]
fn decode_pair_shares_one_quantization_with_a_gguf_partner() {
    let gpu = MockGpuBackend::new();
    let k = 512;
    let (gate_raw, up_raw, wide_raw) = (
        vec![0u8; 64 * 2 * 144],
        vec![0u8; 64 * 2 * 136],
        vec![0u8; 64 * 4 * 144],
    );
    let gate = from_gguf_tensor(&gpu, GgmlType::Q4K, &gate_raw, 64, k, 0).unwrap();
    let up = from_gguf_tensor(&gpu, GgmlType::Iq4Xs, &up_raw, 64, k, 0).unwrap();
    let other_k = from_gguf_tensor(&gpu, GgmlType::Q4K, &wide_raw, 64, 1024, 0).unwrap();
    let bytes = launch::scratch_bytes(1024, 8);
    let s = scratch(gpu.alloc(bytes).unwrap(), bytes);
    let x = gpu.alloc(8 * 1024 * 2).unwrap();
    let (a, b) = (
        gpu.alloc(8 * 64 * 2).unwrap(),
        gpu.alloc(8 * 64 * 2).unwrap(),
    );
    let before = gpu.launches_snapshot().len();
    gate.decode_pair(&*up, &gpu, x, 2, a, b, &s, 0).unwrap();
    assert_eq!(
        gpu.launches_snapshot().len() - before,
        3,
        "one quantization, two GEMVs"
    );
    // A partner over another width cannot share the activation: two plain decodes.
    let before = gpu.launches_snapshot().len();
    other_k
        .decode_pair(&*gate, &gpu, x, 2, a, b, &s, 0)
        .unwrap();
    assert_eq!(
        gpu.launches_snapshot().len() - before,
        4,
        "two quantizations, two GEMVs"
    );
}

#[test]
fn dequant_bf16_launches_one_block_per_super_block() {
    let gpu = MockGpuBackend::new();
    for (q, ty) in TYPES {
        let raw = vec![0u8; launch::weight_bytes(q, 7, 768)];
        let w = from_gguf_tensor(&gpu, ty, &raw, 7, 768, 0).unwrap();
        let out = gpu.alloc(7 * 768 * 2).unwrap();
        let before = gpu.launches_snapshot().len();
        w.dequant_bf16(&gpu, out, 0).unwrap();
        let l = &gpu.launches_snapshot()[before..];
        assert_eq!(l.len(), 1);
        assert_eq!((l[0].grid, l[0].block), ([21, 1, 1], [256, 1, 1]), "{q:?}");
        let (module, func) = q.dequant_entry();
        assert!(
            gpu.kernel_lookups_snapshot()
                .contains(&(module.to_string(), func.to_string()))
        );
    }
}

#[test]
fn release_frees_only_what_the_weight_owns() {
    let gpu = MockGpuBackend::new();
    let raw = vec![0u8; 136];
    let owned = from_gguf_tensor(&gpu, GgmlType::Iq4Xs, &raw, 1, 256, 0).unwrap();
    let ptr = gguf_of(&*owned).unwrap().blocks;
    assert!(release(&gpu, &*owned).unwrap());
    assert!(gpu.read_alloc(ptr).is_none(), "the owned buffer is freed");
    assert!(
        !release(&gpu, &*owned).unwrap(),
        "a second release is harmless"
    );

    let store = gpu.alloc(136).unwrap();
    let t = WeightTensor {
        ptr: store,
        shape: vec![1, 256],
        dtype: WeightDtype::Iq4Xs,
    };
    let borrowed = from_weight_tensor(&gpu, &t)
        .unwrap()
        .expect("a packed dtype");
    assert!(!gguf_of(&*borrowed).unwrap().owned);
    assert!(release(&gpu, &*borrowed).unwrap());
    assert!(
        gpu.read_alloc(store).is_some(),
        "the store keeps its buffer"
    );
}

#[test]
fn from_weight_tensor_takes_only_packed_2d_tensors() {
    let gpu = MockGpuBackend::new();
    let p = gpu.alloc(512).unwrap();
    let bf16 = WeightTensor {
        ptr: p,
        shape: vec![1, 256],
        dtype: WeightDtype::BF16,
    };
    assert!(from_weight_tensor(&gpu, &bf16).unwrap().is_none());
    let flat = WeightTensor {
        ptr: p,
        shape: vec![256],
        dtype: WeightDtype::Q4K,
    };
    assert!(from_weight_tensor(&gpu, &flat).is_err());
    let head = WeightTensor {
        ptr: p,
        shape: vec![2, 256],
        dtype: WeightDtype::Q6K,
    };
    let w = from_weight_tensor(&gpu, &head).unwrap().unwrap();
    assert_eq!((w.format(), w.n(), w.k()), (W4Format::GgufQ6K, 2, 256));
}

#[test]
fn fuse_rows_concatenates_whole_rows() {
    let gpu = MockGpuBackend::new();
    let mut rng = Rng(7);
    let a = random_weight(&mut rng, Q4Type::Q4K, 3, 256);
    let b = random_weight(&mut rng, Q4Type::Q4K, 2, 256);
    let (pa, pb) = (gpu.alloc(a.len()).unwrap(), gpu.alloc(b.len()).unwrap());
    gpu.copy_h2d(&a, pa).unwrap();
    gpu.copy_h2d(&b, pb).unwrap();
    let w = fuse_rows(&gpu, GgmlType::Q4K, &[(pa, 3), (pb, 2)], 256, 0).unwrap();
    assert_eq!((w.n(), w.k()), (5, 256));
    let got = gpu.read_alloc(gguf_of(&*w).unwrap().blocks).unwrap();
    assert_eq!(got, [a, b].concat());
    assert!(fuse_rows(&gpu, GgmlType::Q4K, &[], 256, 0).is_err());
}

#[test]
fn the_q8_1_mirror_matches_the_kernel_rules() {
    // amax 1.0: d = 1/127, codes round half away from zero, the scale is stored as f16.
    let mut row = vec![0u16; 64];
    row[0] = bf16(1.0);
    row[1] = bf16(-0.5);
    row[2] = bf16(0.25);
    let q = quantize_q8_1_row(&row);
    assert_eq!(q.len(), 2);
    assert_eq!(q[0].qs[0], 127);
    assert_eq!(q[0].qs[1], -64); // -63.5 rounds away from zero
    assert_eq!(q[0].qs[2], 32); // 31.75
    assert_eq!(q[0].d, half::f16::from_f32(1.0 / 127.0).to_f32());
    // An all-zero block has zero codes and a zero scale.
    assert_eq!(
        q[1],
        Q8Block {
            qs: [0; 32],
            d: 0.0,
            s: 0.0
        }
    );
    // The partial sum is of the original values, stored as f16.
    assert_eq!(q[0].s, half::f16::from_f32(0.75).to_f32());
    let v = q8_1_values(&q);
    assert_eq!(v.len(), 64);
    assert_eq!(v[0], 127.0 * q[0].d);
}

/// The value a correct kernel produces: f32 sums of per-32-block partials in block order, rounded to bf16.
fn emulated_gpu(
    w: &[f32],
    n: usize,
    k: usize,
    x: &[f32],
    m: usize,
    drop_last_block: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            let blocks = k / 32 - usize::from(drop_last_block) * 8;
            for b in 0..blocks {
                let mut part = 0f32;
                for i in 32 * b..32 * b + 32 {
                    part += w[j * k + i] * x[r * k + i];
                }
                acc += part;
            }
            out[r * n + j] = bf16_to_f32(bf16(acc));
        }
    }
    out
}

#[test]
fn the_cpu_oracle_accepts_a_correct_gemv_and_rejects_planted_faults() {
    let mut rng = Rng(0x19_1919);
    let (n, k, m) = (24, 1024, 3);
    for (q, _) in TYPES {
        let raw = random_weight(&mut rng, q, n, k);
        let w = dequant_weight(q, &raw, n, k).unwrap();
        let xb: Vec<u16> = (0..m * k).map(|_| bf16(rng.unit())).collect();
        let x: Vec<f32> = xb
            .chunks_exact(k)
            .flat_map(|r| q8_1_values(&quantize_q8_1_row(r)))
            .collect();
        let (want, mag) = gemv_f64(&w, n, k, &x, m);

        let good = emulated_gpu(&w, n, k, &x, m, false);
        assert_eq!(
            grade(&good, &want, &mag, k).0,
            0,
            "{q:?}: a correct GEMV passes"
        );
        for ((g, w), a) in good.iter().zip(&want).zip(&mag) {
            assert!(within_bound(*g, *w, *a, k));
        }

        // Planted control 1: a GEMV that drops the last super-block.
        let dropped = emulated_gpu(&w, n, k, &x, m, true);
        let (over, _) = grade(&dropped, &want, &mag, k);
        assert!(
            over > good.len() / 2,
            "{q:?}: dropping a super-block is caught ({over})"
        );

        // Planted control 2: row 1 computed against row 0's activations (a row-mixing bug).
        let mut mixed = good.clone();
        mixed[n..2 * n].copy_from_slice(&good[..n]);
        assert!(
            grade(&mixed, &want, &mag, k).0 > n / 2,
            "{q:?}: row mixing is caught"
        );

        // Planted control 3: a non-finite output.
        let mut nan = good.clone();
        nan[0] = f32::NAN;
        assert_eq!(grade(&nan, &want, &mag, k).0, 1);
    }
}

#[test]
fn dequant_weight_is_the_runtime_decoder() {
    let mut rng = Rng(3);
    let raw = random_weight(&mut rng, Q4Type::Iq4Xs, 2, 256);
    let w = dequant_weight(Q4Type::Iq4Xs, &raw, 2, 256).unwrap();
    let mut want = vec![0f32; 512];
    spark_runtime::weights::dequant_cpu::dequant_to_f32(GgmlType::Iq4Xs, &raw, 512, &mut want)
        .unwrap();
    assert_eq!(w, want);
    assert!(dequant_weight(Q4Type::Iq4Xs, &raw, 2, 200).is_err());
}

#[test]
fn the_mock_backend_is_a_gpu_backend() {
    // Guards the imports above against a trait move.
    let gpu = MockGpuBackend::new();
    let g: &dyn GpuBackend = &gpu;
    assert!(g.has_module(launch::MMQ_MODULE));
}
