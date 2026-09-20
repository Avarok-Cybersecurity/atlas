// SPDX-License-Identifier: AGPL-3.0-only

//! Host-only admission checks for resident KDA state and prefix snapshots.

use avarok_core::kimi_k3::{KdaConfig, KdaState, LayerCache};
use spark_model::kimi_k3::{
    K3CpuFallbackState, K3KdaDecodeKernels, KdaDeviceState, launch_k3_kda_decode_token_on_device,
};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn valid_resident_geometry_decodes_and_restores_authoritative_snapshot() {
    let gpu = MockGpuBackend::new();
    let cfg = KdaConfig::twin_0_40b();
    let kernels = K3KdaDecodeKernels::resolve(&gpu).unwrap();
    let mut state = K3CpuFallbackState::new(LayerCache::Kda(KdaState::new(&cfg)));
    state.ensure_device_kda(&gpu).unwrap();
    let device = state.device_kda.as_ref().unwrap();
    gpu.copy_h2d(&7.25f32.to_le_bytes(), device.recurrent)
        .unwrap();
    launch_k3_kda_decode_token_on_device(
        &gpu,
        &kernels,
        &vec![0.1; cfg.conv_dim()],
        &vec![0.2; cfg.conv_elems()],
        &vec![-0.5; cfg.qkv_dim()],
        &vec![0.25; cfg.heads],
        &cfg,
        device,
        0,
    )
    .unwrap();
    assert_eq!(gpu.launch_count(), 2);
    let bytes = state.snapshot(&gpu, 0).unwrap();
    let LayerCache::Kda(saved) = LayerCache::from_bytes(&bytes).unwrap() else {
        panic!("expected KDA");
    };
    assert_eq!(saved.recurrent[0], 7.25);
    state.restore(&gpu, &bytes, 0).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
    state.ensure_device_kda(&gpu).unwrap();
    assert_eq!(state.snapshot(&gpu, 0).unwrap(), bytes);
    state.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn resident_download_rejects_wrong_geometry_before_copying() {
    let gpu = MockGpuBackend::new();
    let cfg = KdaConfig::twin_0_40b();
    let mut host = KdaState::new(&cfg);
    let device = KdaDeviceState::alloc_and_upload(&gpu, &host).unwrap();
    host.recurrent.push(0.0);
    assert!(device.download(&gpu, &mut host).is_err());
    assert_eq!(gpu.d2h_blocking_count(), 0);
    device.free(&gpu).unwrap();
}

#[test]
fn resident_state_rejects_wrong_geometry_before_launch() {
    let gpu = MockGpuBackend::new();
    let kernels = K3KdaDecodeKernels::resolve(&gpu).unwrap();
    let cfg = KdaConfig::twin_0_40b();
    for truncate_conv in [true, false] {
        let mut host = KdaState::new(&cfg);
        if truncate_conv {
            host.conv.pop();
        } else {
            host.recurrent.pop();
        }
        let device = KdaDeviceState::alloc_and_upload(&gpu, &host).unwrap();
        let result = launch_k3_kda_decode_token_on_device(
            &gpu,
            &kernels,
            &vec![0.1; cfg.conv_dim()],
            &vec![0.2; cfg.conv_elems()],
            &vec![-0.5; cfg.qkv_dim()],
            &vec![0.25; cfg.heads],
            &cfg,
            &device,
            0,
        );
        device.free(&gpu).unwrap();
        assert!(
            result.is_err(),
            "wrong-sized resident state must be rejected"
        );
        assert_eq!(gpu.launch_count(), 0);
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn restore_rejects_wrong_kda_geometry_without_releasing_live_state() {
    let gpu = MockGpuBackend::new();
    let cfg = KdaConfig::twin_0_40b();
    let mut state = K3CpuFallbackState::new(LayerCache::Kda(KdaState::new(&cfg)));
    state.ensure_device_kda(&gpu).unwrap();
    let before = state.snapshot(&gpu, 0).unwrap();
    for truncate_conv in [true, false] {
        let mut wrong = KdaState::new(&cfg);
        if truncate_conv {
            wrong.conv.pop();
        } else {
            wrong.recurrent.pop();
        }
        let blob = LayerCache::Kda(wrong).to_bytes();
        assert!(state.restore(&gpu, &blob, 0).is_err());
        assert!(state.device_kda.is_some());
        assert_eq!(gpu.alloc_count(), 2);
        assert_eq!(state.snapshot(&gpu, 0).unwrap(), before);
    }
    state.release(&gpu).unwrap();
}

#[test]
fn restore_rejects_another_mixer_variant() {
    let gpu = MockGpuBackend::new();
    let kda = LayerCache::Kda(KdaState::new(&KdaConfig::twin_0_40b()));
    let mla = LayerCache::Mla(Default::default());
    for (current, other) in [(kda.clone(), mla.clone()), (mla, kda)] {
        let before = current.to_bytes();
        let mut state = K3CpuFallbackState::new(current);
        assert!(state.restore(&gpu, &other.to_bytes(), 0).is_err());
        assert_eq!(state.snapshot(&gpu, 0).unwrap(), before);
    }
}
