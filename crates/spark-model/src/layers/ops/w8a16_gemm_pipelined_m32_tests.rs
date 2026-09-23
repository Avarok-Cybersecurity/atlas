// SPDX-License-Identifier: AGPL-3.0-only

//! Launch-geometry and guard tests for the 32-row M-tile W8A16 GEMM wrappers
//! on the mock backend. Numerics are the GPU oracle
//! (`examples/native_fp8_gdn_proj_m32_microtest`), not this file: what a CPU
//! test CAN prove is that the launcher hands the kernel exactly the nine
//! parameters it declares, in order, with the grid the tile geometry implies,
//! that the contiguous entry derives its pitches from K and N, that the
//! by-M selector flips at exactly 32 rows and on a missing handle, and that
//! every guard refuses BEFORE a launch is recorded.

use super::*;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

const FULL_K: u64 = 0xB128;
const M32_K: u64 = 0xB032;

struct Fixture {
    gpu: MockGpuBackend,
    input: DevicePtr,
    weight: DevicePtr,
    scale: DevicePtr,
    output: DevicePtr,
}

impl Fixture {
    fn new() -> Self {
        let gpu = MockGpuBackend::new();
        let input = gpu.alloc(64 * 4096 * 2).unwrap();
        let weight = gpu.alloc(256 * 4096).unwrap();
        let scale = gpu.alloc(2 * 32 * 4).unwrap();
        let output = gpu.alloc(64 * 4096 * 2).unwrap();
        Self {
            gpu,
            input,
            weight,
            scale,
            output,
        }
    }

    fn launches(&self) -> Vec<MockLaunch> {
        self.gpu.launches_snapshot()
    }
}

fn u32_arg(v: u32) -> MockArg {
    MockArg::Bytes(v.to_ne_bytes().to_vec())
}

/// The nine kernel parameters, in the order the `.cu` entry point declares
/// them — the arity class `kernel_arity.rs` exists for.
fn assert_m32_launch(l: &MockLaunch, f: &Fixture, m: u32, n: u32, k: u32, lda: u32, ldc: u32) {
    assert_eq!(l.func, M32_K);
    assert_eq!(l.grid, [n.div_ceil(32), m.div_ceil(32), 1]);
    assert_eq!(l.block, [256, 1, 1]);
    assert_eq!(
        l.args,
        vec![
            MockArg::Buffer(f.input),
            MockArg::Buffer(f.weight),
            MockArg::Buffer(f.scale),
            MockArg::Buffer(f.output),
            u32_arg(m),
            u32_arg(n),
            u32_arg(k),
            u32_arg(lda),
            u32_arg(ldc),
        ]
    );
}

#[test]
fn strided_launch_carries_both_pitches_and_tiles_m_by_32() {
    let f = Fixture::new();
    // R=32 attention q_proj shape on the 35B: N=8192, K=2048, A rows `h`
    // apart, C rows `per_seq_qkv` (= 8192 + 2*512 + 8192 gate) apart.
    w8a16_gemm_pipelined_m32_strided(
        &f.gpu,
        KernelHandle(M32_K),
        f.input,
        f.weight,
        f.scale,
        f.output,
        32,
        8192,
        2048,
        2048,
        17408,
        0,
    )
    .unwrap();
    let l = f.launches();
    assert_eq!(l.len(), 1);
    assert_m32_launch(&l[0], &f, 32, 8192, 2048, 2048, 17408);
}

/// 33..64 rows are two M tiles (`grid.y = 2`); the kernel is not clamped to
/// 32 and the wrapper does not refuse — the attention tier at R=64 relies on
/// exactly this.
#[test]
fn strided_launch_above_32_rows_adds_m_tiles() {
    let f = Fixture::new();
    for (m, tiles) in [(33u32, 2u32), (64, 2), (65, 3), (160, 5)] {
        w8a16_gemm_pipelined_m32_strided(
            &f.gpu,
            KernelHandle(M32_K),
            f.input,
            f.weight,
            f.scale,
            f.output,
            m,
            2048,
            4096,
            4096,
            2048,
            0,
        )
        .unwrap();
        let l = f.launches();
        assert_eq!(l.last().unwrap().grid, [64, tiles, 1], "m={m}");
    }
}

#[test]
fn contiguous_launch_derives_pitches_from_k_and_n() {
    let f = Fixture::new();
    w8a16_gemm_pipelined_m32(
        &f.gpu,
        KernelHandle(M32_K),
        f.input,
        f.weight,
        f.scale,
        f.output,
        7,
        12288,
        2048,
        0,
    )
    .unwrap();
    let l = f.launches();
    assert_eq!(l.len(), 1);
    assert_m32_launch(&l[0], &f, 7, 12288, 2048, 2048, 12288);
}

/// THE selector: 1..=32 rows with the twin linked -> the 32-tile (nine
/// args); 33 rows -> the 128-tile `w8a16_gemm_pipelined` (seven args, its
/// own grid); a zero twin handle -> the 128-tile at every width.
#[test]
fn by_m_flips_at_exactly_32_rows_and_on_a_missing_handle() {
    let f = Fixture::new();
    let launch = |m: u32, m32: u64| {
        let before = f.gpu.launch_count();
        w8a16_gemm_pipelined_by_m(
            &f.gpu,
            KernelHandle(FULL_K),
            KernelHandle(m32),
            f.input,
            f.weight,
            f.scale,
            f.output,
            m,
            2048,
            4096,
            0,
        )
        .unwrap();
        let all = f.launches();
        assert_eq!(all.len(), before + 1, "exactly one launch");
        all.last().unwrap().clone()
    };

    for m in [1u32, 5, 16, 17, 32] {
        let l = launch(m, M32_K);
        assert_m32_launch(&l, &f, m, 2048, 4096, 4096, 2048);
        assert!(w8a16_pipelined_prefers_m32(m, 4096, KernelHandle(M32_K)));
    }
    for m in [33u32, 64, 160] {
        let l = launch(m, M32_K);
        assert_eq!(l.func, FULL_K, "m={m} belongs to the 128-tile");
        assert_eq!(l.grid, [64, m.div_ceil(128), 1]);
        assert_eq!(l.args.len(), 7, "the 128-tile kernel takes no pitches");
        assert!(!w8a16_pipelined_prefers_m32(m, 4096, KernelHandle(M32_K)));
    }
    let l = launch(32, 0);
    assert_eq!(l.func, FULL_K, "no twin linked -> the 128-tile, unchanged");
    assert_eq!(l.args.len(), 7);
    assert!(!w8a16_pipelined_prefers_m32(32, 4096, KernelHandle(0)));
    assert!(!w8a16_pipelined_prefers_m32(0, 4096, KernelHandle(M32_K)));
    // A K that is not whole scale blocks keeps the 128 tile (which tolerates
    // a partial block) instead of tripping the twin's guard mid-verify.
    assert!(!w8a16_pipelined_prefers_m32(32, 4000, KernelHandle(M32_K)));
    let before = f.gpu.launch_count();
    w8a16_gemm_pipelined_by_m(
        &f.gpu,
        KernelHandle(FULL_K),
        KernelHandle(M32_K),
        f.input,
        f.weight,
        f.scale,
        f.output,
        32,
        2048,
        4000,
        0,
    )
    .unwrap();
    let all = f.launches();
    assert_eq!(all.len(), before + 1);
    assert_eq!(
        all.last().unwrap().func,
        FULL_K,
        "K=4000 belongs to the 128-tile"
    );
}

/// Every guard refuses before anything is recorded: a launch with a bad
/// pitch would either fault (`cp.async` misalignment) or read a stale scale
/// block (K not a multiple of 128), and neither is a recoverable error.
#[test]
fn guards_refuse_without_launching() {
    let f = Fixture::new();
    // (m, n, k, lda, ldc, what)
    let bad = [
        (0u32, 2048u32, 4096u32, 4096u32, 2048u32, "m=0"),
        (8, 2048, 4000, 4096, 2048, "K not a multiple of 128"),
        (8, 2048, 0, 4096, 2048, "K=0"),
        (8, 2048, 4096, 4092, 2048, "lda shorter than K"),
        (8, 2048, 4096, 4096, 2047, "ldc shorter than N"),
        (8, 2048, 4096, 4100, 2048, "lda not a multiple of 8"),
    ];
    for (m, n, k, lda, ldc, what) in bad {
        let err = w8a16_gemm_pipelined_m32_strided(
            &f.gpu,
            KernelHandle(M32_K),
            f.input,
            f.weight,
            f.scale,
            f.output,
            m,
            n,
            k,
            lda,
            ldc,
            0,
        )
        .expect_err(what);
        assert!(
            err.to_string().contains("w8a16_gemm_pipelined_m32"),
            "{what}: {err}"
        );
        assert_eq!(
            f.gpu.launch_count(),
            0,
            "{what}: refused launch was recorded"
        );
    }
    // The contiguous entry inherits the K guard (its pitches are K and N).
    let err = w8a16_gemm_pipelined_m32(
        &f.gpu,
        KernelHandle(M32_K),
        f.input,
        f.weight,
        f.scale,
        f.output,
        8,
        2048,
        4000,
        0,
    )
    .expect_err("contiguous K guard");
    assert!(err.to_string().contains("multiple of 128"));
    assert_eq!(f.gpu.launch_count(), 0);
}
