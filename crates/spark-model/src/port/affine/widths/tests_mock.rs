// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the widths host path (lane L17c) on the mock GPU backend: the `W4Linear` contract
//! (format, sizes, row window, declined prefill, release), the launch geometry and arguments of the
//! sums and GEMV kernels, the shape-fixed split choice and its scratch fallback, and the refusal of
//! unsupported widths and of targets without the module, before any kernel lookup.

use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{AffineNWeight, kernels, layout};
use crate::port::w4::{W4Format, W4Opts, W4Scratch};

fn i32_arg(a: &MockArg) -> i32 {
    match a {
        MockArg::Bytes(b) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        MockArg::Buffer(p) => panic!("expected a scalar, got buffer {p:?}"),
    }
}

fn scratch(gpu: &MockGpuBackend, ws_bytes: usize, tickets: bool) -> W4Scratch {
    W4Scratch {
        ws: if ws_bytes > 0 {
            gpu.alloc(ws_bytes).unwrap()
        } else {
            DevicePtr::NULL
        },
        ws_bytes,
        tickets: if tickets {
            gpu.alloc(4 * layout::SPLIT_TILES_MAX).unwrap()
        } else {
            DevicePtr::NULL
        },
        xq: DevicePtr::NULL,
        xs: gpu.alloc(layout::sums_bytes(16, 9216)).unwrap(),
    }
}

#[test]
fn mock_launch_path_follows_the_contract() {
    let gpu = MockGpuBackend::new();
    let (n, k, bits) = (2560usize, 9216usize, 3u8);
    let src = gpu.alloc(n * k * 2).unwrap();
    let opts = W4Opts {
        bits,
        group: 64,
        clip_search: 2,
        int8_prefill: true,
    };
    let w = super::quantize_from_bf16(&gpu, src, n, k, &opts, 0).expect("mock quantizes");
    assert_eq!(w.format(), W4Format::Affine { bits, group: 64 });
    assert_eq!((w.n(), w.k(), w.max_decode_rows()), (n, k, 16));
    assert_eq!(w.device_bytes(), layout::device_bytes(n, k, bits));
    let need = w.scratch_bytes(16);
    assert_eq!(need, layout::split_work_bytes(n, k));
    assert!(need > 0, "the 4B down shape splits");
    let (x, out) = (
        gpu.alloc(16 * k * 2).unwrap(),
        gpu.alloc(16 * n * 2).unwrap(),
    );

    let before = gpu.launches_snapshot().len();
    w.decode(&gpu, x, 5, out, &scratch(&gpu, need, true), 0)
        .unwrap();
    let l = gpu.launches_snapshot();
    let (sums, gemv) = (&l[before], &l[before + 1]);
    assert_eq!(
        (sums.grid, sums.block),
        ([5, 3, 1], [64, 1, 1]),
        "144 groups over 64 threads"
    );
    assert_eq!(gemv.shared_mem, layout::split_smem(bits));
    assert_eq!(gemv.args.len(), 15, "split form carries work and tickets");
    assert_eq!(
        gemv.grid[0] as usize,
        layout::tiles(n) * layout::split_k(n, k)
    );
    assert_eq!(i32_arg(&gemv.args[10]), 5, "rows");
    assert_eq!(i32_arg(&gemv.args[8]), 8, "sk");

    for s in [
        scratch(&gpu, 0, true),
        scratch(&gpu, need - 4, true),
        scratch(&gpu, need, false),
    ] {
        let before = gpu.launches_snapshot().len();
        w.decode(&gpu, x, 16, out, &s, 0).unwrap();
        let g = &gpu.launches_snapshot()[before + 1];
        assert_eq!(
            (g.shared_mem, g.args.len()),
            (layout::gemv_smem(bits), 13),
            "unsplit without split scratch"
        );
        assert_eq!(i32_arg(&g.args[8]), 8, "same sk, so the same bits");
        assert_eq!(g.grid[0], layout::gemv_grid(n, bits, 48));
    }

    let s = scratch(&gpu, need, true);
    assert!(
        w.decode(&gpu, x, 0, out, &s, 0).is_err() && w.decode(&gpu, x, 17, out, &s, 0).is_err()
    );
    assert!(
        w.decode(
            &gpu,
            x,
            1,
            out,
            &W4Scratch {
                xs: DevicePtr::NULL,
                ..s
            },
            0
        )
        .is_err()
    );
    assert!(
        !w.prefill(&gpu, x, 64, out, &s, 0).unwrap(),
        "prefill declines"
    );
    let before = gpu.launches_snapshot().len();
    w.dequant_bf16(&gpu, out, 0).unwrap();
    assert_eq!(gpu.launches_snapshot()[before].block, [256, 1, 1]);

    assert!(
        super::release(&gpu, &*w).unwrap(),
        "a widths weight is released"
    );
    assert!(
        w.decode(&gpu, x, 1, out, &s, 0).is_err(),
        "no decode after release"
    );
    assert!(
        super::release(&gpu, &*w).unwrap(),
        "a second release frees nothing and still reports the type"
    );
}

#[test]
fn unsupported_requests_and_other_targets_fail_without_a_lookup() {
    let gpu = MockGpuBackend::new();
    let src = gpu.alloc(64 * 128 * 2).unwrap();
    for (bits, group, clip) in [(4u8, 64u16, 0u8), (3, 32, 0), (7, 64, 0), (3, 64, 11)] {
        let opts = W4Opts {
            bits,
            group,
            clip_search: clip,
            int8_prefill: false,
        };
        assert!(
            super::quantize_from_bf16(&gpu, src, 64, 128, &opts, 0).is_err(),
            "{bits}b g{group} clip {clip}"
        );
    }
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "refused before any lookup"
    );
    gpu.mark_module_absent(kernels::MODULE);
    let opts = W4Opts {
        bits: 5,
        group: 64,
        clip_search: 0,
        int8_prefill: false,
    };
    let err =
        super::quantize_from_bf16(&gpu, src, 64, 128, &opts, 0).expect_err("no module, no weight");
    assert!(err.to_string().contains("not built"), "{err}");
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "no lookup on a target without the module"
    );
    assert!(!super::available(&gpu));
    assert!(AffineNWeight::from_bf16(&gpu, src, 64, 128, 6, 0, 0).is_err());
}
