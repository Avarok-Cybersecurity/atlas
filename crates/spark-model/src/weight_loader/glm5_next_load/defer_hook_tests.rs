// SPDX-License-Identifier: AGPL-3.0-only

//! The defer hook: which tensors `Glm5NextWeightLoader` keeps off the device,
//! and that reading them from the shard gives the SAME bytes the resident path
//! gave.
//!
//! 🔴 The negative half is the point again. The hook withholds tensors, so the
//! claim "`LibertAIDAI/GLM-5.3-Flash-NVFP4` is untouched" is only worth a test
//! that the predicate answers `false` for every shape that checkpoint has.

use avarok_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
use spark_runtime::weights::{DeferredTensor, WeightDtype, WeightStore, WeightTensor};

use super::{bind_expert, is_full_width_mtp_expert};
use crate::weight_loader::ModelWeightLoader;

const LAYERS: usize = 45;

fn qualified(layer: usize, leaf: &str) -> String {
    format!("model.language_model.layers.{layer}.{leaf}")
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect()
}

/// Sign-mixed, non-uniform: a swapped nibble or a dropped scale changes it.
fn ramp(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 - n as f32 / 2.0) * 0.125)
        .collect()
}

// ------------------------------------------------------------- the predicate

#[test]
fn only_the_mtp_layers_full_width_routed_experts_are_deferred() {
    let yes = |n: &str| is_full_width_mtp_expert(n, WeightDtype::BF16, LAYERS);
    for p in ["gate_proj", "up_proj", "down_proj"] {
        assert!(yes(&qualified(
            LAYERS,
            &format!("mlp.experts.0.{p}.weight")
        )));
        assert!(yes(&qualified(
            LAYERS,
            &format!("mlp.experts.287.{p}.weight")
        )));
    }

    // 🪤 The SHARED expert is a plain float tensor `LayerSource` reads; the
    // router and its bias likewise. Withholding any of them breaks the bind.
    for leaf in [
        "mlp.shared_experts.gate_proj.weight",
        "mlp.gate.weight",
        "mlp.gate.e_score_correction_bias",
        "self_attn.q_proj.weight",
        "eh_proj.weight",
    ] {
        assert!(!yes(&qualified(LAYERS, leaf)), "{leaf}");
    }

    // The TEXT stack's experts are the kernel's own operand in both exports.
    // Deferring one would withhold the weights the forward reads.
    assert!(!yes(&qualified(44, "mlp.experts.0.gate_proj.weight")));
    assert!(!yes(&qualified(0, "mlp.experts.0.gate_proj.weight")));

    // Not a layer tensor at all.
    assert!(!yes("lm_head.weight"));
    assert!(!yes(
        "model.language_model.layers.x.mlp.experts.0.up_proj.weight"
    ));
}

/// The community checkpoint packs this block like every other, so on it the
/// predicate is false for every tensor and nothing is deferred at all.
#[test]
fn a_packed_expert_is_never_deferred_whatever_the_layer() {
    for dtype in [WeightDtype::UInt8, WeightDtype::FP8E4M3, WeightDtype::FP32] {
        assert!(!is_full_width_mtp_expert(
            &qualified(LAYERS, "mlp.experts.0.gate_proj.weight"),
            dtype,
            LAYERS
        ));
    }
    // The scale siblings ride along with a packed expert and are not `.weight`.
    assert!(!is_full_width_mtp_expert(
        &qualified(LAYERS, "mlp.experts.0.gate_proj.weight_scale"),
        WeightDtype::BF16,
        LAYERS
    ));
}

/// 🪤 `layers.45` is `num_hidden_layers`, not a property of GLM. A config with
/// a different depth must move the whole rule with it.
#[test]
fn the_deferred_layer_comes_from_the_config_not_a_literal() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.num_hidden_layers = 7;
    let hook = super::Glm5NextWeightLoader
        .defer_predicate(&config)
        .expect("glm5_next declares a defer predicate");

    assert!(hook(
        &qualified(7, "mlp.experts.3.down_proj.weight"),
        WeightDtype::BF16
    ));
    assert!(!hook(
        &qualified(45, "mlp.experts.3.down_proj.weight"),
        WeightDtype::BF16
    ));
}

/// The trait default is "upload everything" — a loader that never overrides
/// this can never lose a tensor it needs.
#[test]
fn a_loader_that_does_not_override_the_hook_defers_nothing() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(
        crate::weight_loader::qwen35::Qwen35WeightLoader
            .defer_predicate(&config)
            .is_none()
    );
}

// --------------------------------------------------------- the deferred bind

/// Write `bytes` into a scratch shard at `offset`, preceded by filler standing
/// in for the safetensors header.
fn stage_shard(tag: &str, offset: u64, bytes: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("avarok-glm5next-defer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{tag}.safetensors"));
    let mut blob = vec![0xAAu8; offset as usize];
    blob.extend_from_slice(bytes);
    std::fs::write(&path, &blob).unwrap();
    path
}

/// The load-bearing claim: an expert read from its SHARD and an expert read
/// back off the DEVICE produce byte-identical NVFP4. The two arms exist only
/// because of where the bytes come from; if they ever disagree, the official
/// export is quantised differently from every other path and nothing else
/// would say so.
#[test]
fn a_deferred_expert_quantises_to_the_same_bytes_as_a_resident_one() {
    let values = ramp(32);
    let (rows, cols) = (2usize, 16usize);
    let bytes = bf16_bytes(&values);

    // Arm A: resident BF16, the way a store with no defer hook arrives.
    let gpu_a = MockGpuBackend::new();
    let mut map = std::collections::HashMap::new();
    for p in ["gate_proj", "up_proj", "down_proj"] {
        let ptr = gpu_a.alloc(bytes.len()).unwrap();
        gpu_a.copy_h2d(&bytes, ptr).unwrap();
        map.insert(
            qualified(LAYERS, &format!("mlp.experts.0.{p}.weight")),
            WeightTensor {
                ptr,
                shape: vec![rows, cols],
                dtype: WeightDtype::BF16,
            },
        );
    }
    let resident = bind_expert(&gpu_a, &WeightStore::from_map(map), LAYERS, 0).unwrap();

    // Arm B: deferred — nothing full-width was ever uploaded.
    let path = stage_shard("expert", 137, &bytes);
    let gpu_b = MockGpuBackend::new();
    let mut store = WeightStore::from_map(std::collections::HashMap::new());
    for p in ["gate_proj", "up_proj", "down_proj"] {
        store.defer(
            qualified(LAYERS, &format!("mlp.experts.0.{p}.weight")),
            DeferredTensor {
                path: path.clone(),
                offset: 137,
                shape: vec![rows, cols],
                dtype: WeightDtype::BF16,
            },
        );
    }
    let deferred = bind_expert(&gpu_b, &store, LAYERS, 0).unwrap();

    for (a, b, what) in [
        (&resident.gate_proj, &deferred.gate_proj, "gate"),
        (&resident.up_proj, &deferred.up_proj, "up"),
        (&resident.down_proj, &deferred.down_proj, "down"),
    ] {
        assert_eq!(
            gpu_a.read_alloc(a.packed).unwrap(),
            gpu_b.read_alloc(b.packed).unwrap(),
            "{what}: packed codes differ between the shard and device paths"
        );
        assert_eq!(
            gpu_a.read_alloc(a.scale).unwrap(),
            gpu_b.read_alloc(b.scale).unwrap(),
            "{what}: block scales differ"
        );
        assert_eq!(a.scale_2, b.scale_2, "{what}: global scale differs");
    }

    // Only the NVFP4 reached the device: 3 x (packed [2, 8] + scales [2, 1]).
    assert_eq!(store.derived().len(), 6);
    assert_eq!(store.derived().bytes(), 3 * (rows * cols / 2 + rows));
    assert!(
        store.derived().bytes() < 3 * bytes.len(),
        "the point of deferring is that less reaches the device than is on disk"
    );

    let _ = std::fs::remove_file(&path);
}

/// A deferred tensor is served from its ON-DISK bytes, so a width this arm was
/// not taught is a refusal rather than a reinterpretation.
#[test]
fn a_deferred_expert_at_an_unsupported_width_is_refused() {
    let path = stage_shard("odd-width", 0, &[0x21u8; 16]);
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::from_map(std::collections::HashMap::new());
    store.defer(
        qualified(LAYERS, "mlp.experts.0.gate_proj.weight"),
        DeferredTensor {
            path: path.clone(),
            offset: 0,
            shape: vec![2, 8],
            dtype: WeightDtype::UInt8,
        },
    );
    let err = bind_expert(&gpu, &store, LAYERS, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("deferred"), "{err}");

    let _ = std::fs::remove_file(&path);
}
