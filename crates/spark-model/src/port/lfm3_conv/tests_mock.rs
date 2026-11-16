// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: unit tests of the LFM3 conv layer's host plumbing: the
//! snapshot blob, the geometry checks, the state selection rules, the launch
//! ABI of the three kernel entries and the state lifecycle (alloc, snapshot,
//! restore, release) on the mock backend.

use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::launch::{self, SeqLayout, Strides};
use super::{CONV_WIDTH, Lfm3ConvCfg, Lfm3ConvLayer, Lfm3ConvState, STATE_LEN, init, state};
use crate::layer::{EmptyLayerState, SsmLayerState, TransformerLayer};
use crate::layers::FfnComponent;
use crate::weight_map::DenseWeight;

// ── blob, configuration, state selection ────────────────────────────────────

#[test]
fn blob_round_trips_and_refuses_foreign_geometry() {
    let payload: Vec<u8> = (0..(8 * STATE_LEN * 2)).map(|i| i as u8).collect();
    let blob = state::encode_blob(8, &payload).unwrap();
    assert_eq!(blob.len(), state::BLOB_HEADER + payload.len());
    assert_eq!(state::decode_blob(8, &blob).unwrap(), &payload[..]);
    assert!(state::decode_blob(16, &blob).is_err(), "other hidden");
    let mut bad = blob.clone();
    bad[0] = b'X';
    assert!(state::decode_blob(8, &bad).is_err(), "bad magic");
    assert!(
        state::decode_blob(8, &blob[..blob.len() - 1]).is_err(),
        "truncated"
    );
    assert!(
        state::encode_blob(8, &payload[1..]).is_err(),
        "short payload"
    );
}

fn cfg(hidden: usize) -> Lfm3ConvCfg {
    Lfm3ConvCfg {
        hidden,
        conv_width: CONV_WIDTH,
        eps: 1e-5,
        layer_idx: 3,
    }
}

#[test]
fn cfg_refuses_other_conv_widths() {
    assert!(cfg(2048).validate().is_ok());
    assert_eq!(cfg(2048).state_bytes(), 2048 * 6 * 2);
    assert_eq!(cfg(2048).bcx_width(), 6144);
    let mut c = cfg(2048);
    c.conv_width = 4;
    assert!(c.validate().is_err());
    assert!(cfg(0).validate().is_err());
}

fn dummy_ssm_state() -> SsmLayerState {
    SsmLayerState {
        h_state: DevicePtr(0x10),
        conv_state: DevicePtr(0x20),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    }
}

#[test]
fn padding_rows_use_the_pad_window_and_strangers_are_refused() {
    let pad = DevicePtr(0xAB00);
    let own = Lfm3ConvState {
        conv: DevicePtr(0xCD00),
        bytes: 24,
    };
    assert_eq!(init::row_state_of(&own, pad, 0).unwrap(), DevicePtr(0xCD00));
    assert_eq!(init::row_state_of(&dummy_ssm_state(), pad, 0).unwrap(), pad);
    assert!(init::row_state_of(&EmptyLayerState, pad, 0).is_err());
    let released = Lfm3ConvState {
        conv: DevicePtr::NULL,
        bytes: 24,
    };
    assert!(init::row_state_of(&released, pad, 0).is_err());
}

// ── launch ABI and state lifecycle on the mock backend ──────────────────────

fn mock_kernels(gpu: &MockGpuBackend) -> launch::ConvKernels {
    launch::ConvKernels::resolve(gpu).unwrap()
}

#[test]
fn resolution_fails_without_the_ada_module() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(launch::MODULE);
    let err = launch::ConvKernels::resolve(&gpu).unwrap_err().to_string();
    assert!(err.contains("lfm3_shortconv_ada"), "{err}");
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "no lookup is issued against an absent module"
    );
}

#[test]
fn decode_launches_sixteen_sequences_per_call_with_offset_rows() {
    let gpu = MockGpuBackend::new();
    let k = mock_kernels(&gpu);
    let states: Vec<DevicePtr> = (0..20).map(|i| DevicePtr(0x1_0000 + i * 0x100)).collect();
    let (bcx, w, y) = (
        DevicePtr(0x100_0000),
        DevicePtr(0x200_0000),
        DevicePtr(0x300_0000),
    );
    launch::decode(&gpu, &k, bcx, w, y, Strides::dense(2048), &states, 7).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 2);
    assert_eq!((l[0].grid, l[0].block), ([16, 16, 1], [128, 1, 1]));
    assert_eq!(l[1].grid, [16, 4, 1]);
    assert_eq!(l[0].args.len(), 7 + launch::DECODE_SEQS);
    assert_eq!(l[1].args[0], MockArg::Buffer(bcx.offset(16 * 6144 * 2)));
    assert_eq!(l[1].args[2], MockArg::Buffer(y.offset(16 * 2048 * 2)));
    assert_eq!(l[1].args[6], MockArg::Bytes(4u32.to_le_bytes().to_vec()));
    assert_eq!(l[1].args[7], MockArg::Buffer(states[16]));
    assert_eq!(
        l[1].args[11],
        MockArg::Buffer(DevicePtr::NULL),
        "unused slots are NULL"
    );
    assert!(l.iter().all(|x| x.stream == 7));
}

#[test]
fn prefill_chunk_launches_y_then_the_state_kernel() {
    let gpu = MockGpuBackend::new();
    let k = mock_kernels(&gpu);
    let st = DevicePtr(0x5000);
    let (bcx, w, y) = (
        DevicePtr(0x100_0000),
        DevicePtr(0x200_0000),
        DevicePtr(0x300_0000),
    );
    let single = SeqLayout::Single(st);
    launch::prefill_chunk(&gpu, &k, bcx, w, y, single, 130, Strides::dense(2048), 3).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 2);
    assert_eq!(l[0].grid, [16, 3, 1], "ceil(130 / 64) token tiles");
    assert_eq!(l[0].args.len(), 11);
    assert_eq!(
        l[0].args[3],
        MockArg::Buffer(DevicePtr::NULL),
        "single form: no cu"
    );
    assert_eq!(l[0].args[5], MockArg::Buffer(st));
    assert_eq!(l[1].grid, [16, 1, 1]);
    assert_eq!(l[1].args.len(), 8);
    assert_eq!(l[1].args[3], MockArg::Buffer(st));
    let packed = SeqLayout::Packed {
        cu_seqlens: DevicePtr(0x7000),
        state_table: DevicePtr(0x8000),
        num_seqs: 5,
    };
    launch::prefill_chunk(&gpu, &k, bcx, w, y, packed, 300, Strides::dense(2048), 3).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l[2].args[3], MockArg::Buffer(DevicePtr(0x7000)));
    assert_eq!(
        l[2].args[5],
        MockArg::Buffer(DevicePtr::NULL),
        "packed form: no state0"
    );
    assert_eq!(l[3].grid, [16, 5, 1]);
}

/// A layer over the mock with no FFN, for the state-lifecycle entry points
/// (they never touch the FFN). `Lfm3ConvLayer::new` refuses a non-dense FFN,
/// so the struct is assembled directly.
fn mock_layer(gpu: &MockGpuBackend, hidden: usize) -> Lfm3ConvLayer {
    let c = cfg(hidden);
    let pad_state = gpu.alloc(c.state_bytes()).unwrap();
    let w = DenseWeight {
        weight: DevicePtr(0x9000),
    };
    Lfm3ConvLayer {
        cfg: c,
        operator_norm: w,
        in_proj: w,
        conv: w,
        out_proj: w,
        post_operator_norm: w,
        ffn_norm: w,
        ffn: FfnComponent::None,
        post_ffn_norm: w,
        fp8: None,
        k: init::LayerKernels::resolve(gpu).unwrap(),
        pad_state,
    }
}

#[test]
fn layer_reports_its_capabilities() {
    let gpu = MockGpuBackend::new();
    let l = mock_layer(&gpu, 64);
    assert!(!l.uses_ssm_pool());
    assert!(!l.is_ssm_layer());
    assert!(l.has_aux_state());
    assert!(l.graph_stale_on_new_sequence());
    assert!(l.decode_rollback_unsupported());
    assert!(l.decode_verify_multi_unsupported());
    assert!(!l.decode_graph_unsupported());
    assert!(!l.decode_multi_seq_unsupported());
}

#[test]
fn state_alloc_snapshot_restore_release_round_trip() {
    let gpu = MockGpuBackend::new();
    let l = mock_layer(&gpu, 64);
    let mut a = l.alloc_state(&gpu).unwrap();
    let mut b = l.alloc_state(&gpu).unwrap();
    let a_ptr = a.as_any().downcast_ref::<Lfm3ConvState>().unwrap().conv;
    let b_ptr = b.as_any().downcast_ref::<Lfm3ConvState>().unwrap().conv;
    assert_ne!(a_ptr, b_ptr, "each sequence owns its window");
    assert!(
        gpu.read_alloc(a_ptr).unwrap().iter().all(|&v| v == 0),
        "fresh window is zero"
    );
    let pattern: Vec<u8> = (0..l.cfg.state_bytes())
        .map(|i| (i * 7 % 251) as u8)
        .collect();
    gpu.copy_h2d(&pattern, a_ptr).unwrap();
    let blob = l
        .snapshot_aux(a.as_ref(), &gpu, 0)
        .unwrap()
        .expect("aux blob");
    l.restore_aux(b.as_mut(), &blob, &gpu, 0).unwrap();
    assert_eq!(gpu.read_alloc(b_ptr).unwrap(), pattern);
    // A blob from another width is refused and leaves the window alone.
    let other = mock_layer(&gpu, 32);
    let foreign = other
        .snapshot_aux(other.alloc_state(&gpu).unwrap().as_ref(), &gpu, 0)
        .unwrap()
        .expect("aux blob");
    assert!(l.restore_aux(b.as_mut(), &foreign, &gpu, 0).is_err());
    assert_eq!(gpu.read_alloc(b_ptr).unwrap(), pattern);
    // Release is idempotent and leaves other state types alone.
    l.release_state(a.as_mut(), &gpu).unwrap();
    assert!(
        a.as_any()
            .downcast_ref::<Lfm3ConvState>()
            .unwrap()
            .conv
            .is_null()
    );
    l.release_state(a.as_mut(), &gpu).unwrap();
    let mut pooled = dummy_ssm_state();
    l.release_state(&mut pooled, &gpu).unwrap();
    assert_eq!(
        pooled.conv_state,
        DevicePtr(0x20),
        "pool memory is never freed here"
    );
    l.release_state(b.as_mut(), &gpu).unwrap();
}
