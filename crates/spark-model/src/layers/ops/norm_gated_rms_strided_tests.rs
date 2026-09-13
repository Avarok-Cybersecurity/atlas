// SPDX-License-Identifier: AGPL-3.0-only

//! The gated-RMS-norm launch-count pin (#927).
//!
//! H100, 2026-09-11 round 7, `Qwen/Qwen3.8-27B-FP8`, batch 16, steady-state
//! n=16 decode step 43.595 ms: `gated_rms_norm_f32_input` fired **768** times —
//! 48 SSM layers × 16 sequences — for 1.612 ms = **3.70% of the step**, 2.1 µs
//! per launch. Every other per-layer kernel in that step is at 48 or 64; this
//! one was the outlier, and it was visible only because the trace was taken at
//! `--cuda-graph-trace=node` granularity.
//!
//! What this file pins is the thing that made 768 possible: the per-sequence
//! LOOP. The strided entry point must take one launch for any row count, and
//! its stride arguments must be exactly the pointer deltas the loop computed —
//! otherwise it is one launch normalizing the wrong rows.

use super::*;
use crate::weight_map::DenseWeight;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

const NORM_K: u64 = 0xF17E;
const STRIDED_K: u64 = 0xF17F;
/// Qwen3.8-27B GDN: 32 value heads × 128, fused QKVZ 16384 wide.
const NV: u32 = 32;
const VD: u32 = 128;
const QKVZ: u32 = 16384;
const VALUE_DIM: u32 = NV * VD;
/// SSM layers on the 27B — the multiplier between "per layer" and "per step".
const SSM_LAYERS: usize = 48;

struct Bufs {
    gdn_out: spark_runtime::gpu::DevicePtr,
    z_base: spark_runtime::gpu::DevicePtr,
    normed_out: spark_runtime::gpu::DevicePtr,
    weight: DenseWeight,
}

fn bufs(gpu: &MockGpuBackend) -> Bufs {
    Bufs {
        gdn_out: gpu.alloc(16 * VALUE_DIM as usize * 4).unwrap(),
        z_base: gpu.alloc(16 * QKVZ as usize * 2).unwrap(),
        normed_out: gpu.alloc(16 * VALUE_DIM as usize * 2).unwrap(),
        weight: DenseWeight {
            weight: gpu.alloc(VD as usize * 2).unwrap(),
        },
    }
}

/// The loop this replaces, verbatim in shape: one launch per sequence, each at
/// that sequence's own base pointers.
fn per_seq_loop(gpu: &MockGpuBackend, b: &Bufs, n: usize) {
    for i in 0..n {
        gated_rms_norm(
            gpu,
            KernelHandle(NORM_K),
            b.gdn_out.offset(i * VALUE_DIM as usize * 4),
            b.z_base.offset(i * QKVZ as usize * 2),
            &b.weight,
            b.normed_out.offset(i * VALUE_DIM as usize * 2),
            NV,
            VD,
            VD,
            1e-6,
            VD,
            0,
        )
        .unwrap();
    }
}

fn strided_once(gpu: &MockGpuBackend, b: &Bufs, n: usize) {
    gated_rms_norm_strided(
        gpu,
        KernelHandle(STRIDED_K),
        b.gdn_out,
        b.z_base,
        &b.weight,
        b.normed_out,
        NV,
        n as u32,
        VD,
        VD,
        1e-6,
        VD,
        VALUE_DIM,
        QKVZ,
        VALUE_DIM,
        0,
    )
    .unwrap();
}

/// 48 launches per step, not 768.
#[test]
fn the_strided_gated_norm_costs_one_launch_per_layer_at_any_row_count() {
    for n in [2usize, 5, 8, 12, 16] {
        let gpu = MockGpuBackend::new();
        let b = bufs(&gpu);
        let before = gpu.launch_count();
        strided_once(&gpu, &b, n);
        let per_layer = gpu.launch_count() - before;
        assert_eq!(per_layer, 1, "n={n}: one launch per layer");
        assert_eq!(
            per_layer * SSM_LAYERS,
            48,
            "n={n}: {SSM_LAYERS} SSM layers => 48 launches per step, not 768"
        );
    }
}

/// The control: the loop it replaces really does scale with the row count, so
/// the test above is measuring a change and not a tautology. At n=16 across 48
/// SSM layers that is the 768 the H100 trace recorded.
#[test]
fn the_per_sequence_loop_is_the_768_launch_shape() {
    let gpu = MockGpuBackend::new();
    let b = bufs(&gpu);
    per_seq_loop(&gpu, &b, 16);
    assert_eq!(gpu.launch_count(), 16);
    assert_eq!(gpu.launch_count() * SSM_LAYERS, 768);
}

/// One launch is only correct if it covers the same rows. The grid must be
/// `(heads, sequences)` — a grid of `(heads, 1)` would silently normalize
/// sequence 0 and leave the rest holding whatever the previous step left.
#[test]
fn the_strided_grid_covers_every_sequence_and_head() {
    let gpu = MockGpuBackend::new();
    let b = bufs(&gpu);
    strided_once(&gpu, &b, 16);
    let l = &gpu.launches_snapshot()[0];
    assert_eq!(l.grid, [NV, 16, 1], "grid is (heads_per_seq, num_seqs, 1)");
    assert_eq!(l.block, [VD, 1, 1], "one block per row, VD threads wide");
}

/// THE address pin. The strided launch's three sequence strides must equal the
/// pointer deltas the loop computed for consecutive sequences — including the
/// asymmetric one: `gdn_out` and `normed_out` step by `value_dim` (in FLOATS
/// and BF16 respectively), but the gate steps by the whole `qkvz_size` block,
/// because z lives inside each sequence's deinterleaved QKVZ row. Getting that
/// third stride wrong is the bug this test exists for, and it would not show up
/// as a crash — only as gates read from the wrong sequence.
#[test]
fn the_strided_arguments_reproduce_the_loops_pointer_deltas() {
    let gpu = MockGpuBackend::new();
    let b = bufs(&gpu);
    per_seq_loop(&gpu, &b, 2);
    let loops = gpu.launches_snapshot();
    let base = |l: &spark_runtime::gpu::mock::MockLaunch, i: usize| match &l.args[i] {
        MockArg::Buffer(p) => p.0,
        other => panic!("arg {i} is not a buffer: {other:?}"),
    };
    // Deltas the loop actually used, measured rather than assumed.
    let d_input = base(&loops[1], 0) - base(&loops[0], 0);
    let d_gate = base(&loops[1], 1) - base(&loops[0], 1);
    let d_output = base(&loops[1], 3) - base(&loops[0], 3);
    assert_eq!(d_input, VALUE_DIM as u64 * 4, "gdn_out rows are f32");
    assert_eq!(d_gate, QKVZ as u64 * 2, "z rows are a whole QKVZ block");
    assert_eq!(d_output, VALUE_DIM as u64 * 2, "normed_out rows are bf16");

    let gpu2 = MockGpuBackend::new();
    let b2 = bufs(&gpu2);
    strided_once(&gpu2, &b2, 2);
    let l = &gpu2.launches_snapshot()[0];
    let u32_arg = |v: u32| MockArg::Bytes(v.to_ne_bytes().to_vec());
    // Bases are sequence 0's, and the strides are in each buffer's OWN element
    // type — which is why they are not all equal despite two of the three
    // deltas above being the same count of elements.
    assert_eq!(l.args[0], MockArg::Buffer(b2.gdn_out));
    assert_eq!(l.args[1], MockArg::Buffer(b2.z_base));
    assert_eq!(l.args[3], MockArg::Buffer(b2.normed_out));
    assert_eq!(
        l.args[8],
        u32_arg(d_input as u32 / 4),
        "input_seq_stride, f32"
    );
    assert_eq!(
        l.args[9],
        u32_arg(d_gate as u32 / 2),
        "gate_seq_stride, bf16"
    );
    assert_eq!(
        l.args[10],
        u32_arg(d_output as u32 / 2),
        "output_seq_stride, bf16"
    );
    // The per-head arguments are the packed kernel's, unchanged — that is what
    // makes the strided kernel bit-identical per (sequence, head) row.
    assert_eq!(l.args[4], u32_arg(VD), "hidden_size");
    assert_eq!(l.args[6], u32_arg(VD), "gate_stride between HEADS");
}
