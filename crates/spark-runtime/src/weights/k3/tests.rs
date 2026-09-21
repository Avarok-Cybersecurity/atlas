// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::gpu::mock::MockGpuBackend;
use avarok_core::config::parse_config;
use avarok_core::scope::ModelResource;
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};

fn config(rank: usize, world: usize) -> ModelConfig {
    let mut c = parse_config(include_str!(
        "../../../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json"
    ))
    .unwrap();
    c.weight_prefix.clear();
    c.hidden_size = 8;
    c.vocab_size = 8;
    c.intermediate_size = 8;
    c.shared_expert_intermediate_size = 8;
    c.num_hidden_layers = 2;
    c.layer_types.truncate(2);
    c.num_experts = 1;
    c.tp_rank = rank;
    c.tp_world_size = world;
    c.num_attention_heads /= world;
    c.num_key_value_heads /= world;
    c.linear_num_key_heads /= world;
    c.linear_num_value_heads /= world;
    c.linear_num_key_heads = 1;
    c.linear_num_value_heads = 1;
    c.linear_key_head_dim = 8;
    c.linear_value_head_dim = 8;
    c.moe_intermediate_size = 64;
    c.moe_latent_size = 64;
    c
}

fn fixture(dir: &Path, bad_scale: bool) -> Vec<u8> {
    use avarok_core::kimi_k3::{
        MixerKind,
        tp::{TpAxis, tensor_plan},
    };
    let c = config(0, 2);
    let packed: Vec<u8> = (0..64 * 32).map(|i| (i % 251) as u8).collect();
    let mut owned: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    for name in avarok_core::kimi_k3::weights::required_names(&c) {
        if name.contains(".block_sparse_moe.experts.") {
            owned.push((
                format!("{name}_packed"),
                Dtype::U8,
                vec![64, 32],
                packed.clone(),
            ));
            let bad = bad_scale && name.ends_with(".w2.weight");
            owned.push((
                format!("{name}_scale"),
                Dtype::U8,
                vec![64, if bad { 1 } else { 2 }],
                vec![127; if bad { 64 } else { 128 }],
            ));
        } else {
            let (axis, n, k) = tensor_plan(&name, MixerKind::Kda, &c);
            let shape = if axis == TpAxis::Replicated {
                avarok_core::kimi_k3::weights_geometry::expected_replicated_shape(&name, &c)
                    .unwrap()
            } else {
                vec![n, k]
            };
            let bytes = vec![0; shape.iter().product::<usize>() * 2];
            owned.push((name, Dtype::BF16, shape, bytes));
        }
    }
    let views = owned
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), TensorView::new(*d, s.clone(), b).unwrap()))
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    packed
}

#[test]
fn packed_rank_slices_before_gpu_allocation_and_reports_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let full = fixture(dir.path(), false);
    let gpu = MockGpuBackend::new();
    gpu.set_max_allocation_bytes(1024); // full packed tensor is 2048: old loader fails.
    let mut store = K3SafetensorsLoader::new(config(1, 2))
        .unwrap()
        .load(dir.path(), &gpu, 0)
        .unwrap();
    assert_eq!(store.prepartitioned_tp(), Some((1, 2)));
    let t = store
        .get("model.layers.1.block_sparse_moe.experts.0.w2.weight_packed")
        .unwrap();
    assert_eq!(t.shape, vec![64, 16]);
    let expected: Vec<u8> = full
        .chunks_exact(32)
        .flat_map(|r| r[16..].iter().copied())
        .collect();
    assert_eq!(gpu.read_alloc(t.ptr).unwrap(), expected);
    assert!(store.resident_bytes() >= 3 * 1088);
    store.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn invalid_pair_fails_before_any_allocation() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path(), true);
    let gpu = MockGpuBackend::new();
    assert!(
        K3SafetensorsLoader::new(config(0, 2))
            .unwrap()
            .load(dir.path(), &gpu, 0)
            .is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn allocation_failure_releases_prior_weights() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path(), false);
    let gpu = MockGpuBackend::new();
    gpu.set_max_allocation_bytes(256); // sorted-first lm_head is 128 bytes; packed allocation fails later.
    assert!(
        K3SafetensorsLoader::new(config(0, 2))
            .unwrap()
            .load(dir.path(), &gpu, 0)
            .is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn refuses_invalid_topology_and_oom_budget() {
    assert!(K3SafetensorsLoader::new(config(2, 2)).is_err());
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path(), false);
    let gpu = MockGpuBackend::new();
    assert!(
        K3SafetensorsLoader::new(config(0, 2))
            .unwrap()
            .load(dir.path(), &gpu, usize::MAX)
            .is_err()
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn malformed_replicated_tensor_is_refused_before_gpu_allocation() {
    for (dtype, shape, data, expected) in [
        (Dtype::U8, vec![8, 8], vec![0u8; 64], "tensor role"),
        (Dtype::BF16, vec![1], vec![0u8; 2], "shape"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        serialize_to_file(
            [(
                "model.embed_tokens.weight",
                TensorView::new(dtype, shape, &data).unwrap(),
            )],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let gpu = MockGpuBackend::new();
        let err = K3SafetensorsLoader::new(config(0, 2))
            .unwrap()
            .load(dir.path(), &gpu, 0)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains(expected), "{err}");
        assert_eq!(gpu.alloc_count(), 0);
    }
}

#[test]
fn incomplete_checkpoint_is_refused_before_upload() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path(), false);
    let path = dir.path().join("model.safetensors");
    let bytes = std::fs::read(&path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let remaining = tensors
        .names()
        .into_iter()
        .filter(|n| *n != "model.norm.weight")
        .map(|n| (n, tensors.tensor(n).unwrap()))
        .collect::<Vec<_>>();
    serialize_to_file(remaining, None, &path).unwrap();
    let gpu = MockGpuBackend::new();
    let err = K3SafetensorsLoader::new(config(0, 2))
        .unwrap()
        .load(dir.path(), &gpu, 0)
        .err()
        .unwrap()
        .to_string();
    assert!(
        err.contains("missing required tensor model.norm.weight"),
        "{err}"
    );
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn binding_admission_matches_exact_copies_and_reserve_boundary() {
    for fp32 in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path(), false);
        if fp32 {
            // Convert every dense tensor, including layer weights which must
            // alias their source allocation rather than add a binding allowance.
            let path = dir.path().join("model.safetensors");
            let bytes = std::fs::read(&path).unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let owned = tensors
                .names()
                .into_iter()
                .map(|name| {
                    let tensor = tensors.tensor(name).unwrap();
                    let (dtype, data) = if tensor.dtype() == Dtype::BF16 {
                        (Dtype::F32, vec![0; tensor.data().len() * 2])
                    } else {
                        (tensor.dtype(), tensor.data().to_vec())
                    };
                    (name.to_string(), dtype, tensor.shape().to_vec(), data)
                })
                .collect::<Vec<_>>();
            let views = owned
                .iter()
                .map(|(name, dtype, shape, data)| {
                    (
                        name.as_str(),
                        TensorView::new(*dtype, shape.clone(), data).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            serialize_to_file(views, None, &path).unwrap();
        }
        let gpu = MockGpuBackend::new();
        let loader = K3SafetensorsLoader::new(config(1, 2)).unwrap();
        let mut measured = loader.load(dir.path(), &gpu, 0).unwrap();
        // Observe actual uploaded allocations; do not reuse the preflight planner
        // to calculate its own expected result. The loader does not bind copies.
        let local = gpu.live_bytes().unwrap();
        assert_eq!(local, measured.resident_bytes());
        let staging = measured
            .weights
            .values()
            .map(|tensor| gpu.read_alloc(tensor.ptr).unwrap().len())
            .max()
            .unwrap();
        measured.release(&gpu).unwrap();
        // Engine embed [8,8], head [8,8], norm [8] get BF16 copies only
        // when their source is FP32. No other FP32 tensor adds a device copy.
        let extra = if fp32 { (8 * 8 + 8 * 8 + 8) * 2 } else { 0 };
        let exact_reserve = gpu.free_memory().unwrap() - local - staging - extra;
        let mut admitted = loader.load(dir.path(), &gpu, exact_reserve).unwrap();
        assert_eq!(gpu.live_bytes().unwrap(), local);
        admitted.release(&gpu).unwrap();
        let error = loader
            .load(dir.path(), &gpu, exact_reserve + 1)
            .err()
            .expect("one byte over the full bound must be refused")
            .to_string();
        assert!(error.contains("OOM preflight"), "{error}");
        assert!(
            error.contains(&format!("binding copies {extra}")),
            "{error}"
        );
        assert_eq!(gpu.alloc_count(), 0);
    }
}
