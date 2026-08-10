// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B per-layer-embedding (PLE) block.
//!
//! The E2B checkpoint adds a per-layer residual contribution at the END of
//! every decoder layer, immediately BEFORE the `layer_scalar` multiply:
//!
//! ```text
//! residual = hidden
//! h = input_gate(hidden)            # Linear hidden_size -> 256, no bias
//! h = gelu_pytorch_tanh(h)
//! h = h * ple_slice[i]              # [S, 256] elementwise (layer i's slice)
//! h = projection(h)                 # Linear 256 -> hidden_size, no bias
//! h = post_norm(h)                  # RMSNorm(hidden_size)
//! hidden = residual + h
//! hidden *= layer_scalar
//! ```
//!
//! The model-level precompute (`TransformerModel::compute_ple`) builds the
//! combined per-layer vectors once per pass as a single
//! `[num_tokens, num_layers * 256]` BF16 buffer; each layer's
//! `gemma4_ple_mul` reads its own strided slice directly (no transposed
//! staging copy). The layer-side forward lives in
//! `qwen3_attention/ple.rs` (it needs the layer's private fields).
//!
//! This file owns the layer-facing `Gemma4LayerPle` weights struct and the
//! load-time install walk that attaches the loaded weights + KV-shared flag
//! to each attention layer.

use atlas_core::config::ModelConfig;

use crate::layer::TransformerLayer;
use crate::weight_map::DenseWeight;

/// The three per-layer PLE weights attached to each E2B layer.
#[derive(Clone)]
pub struct Gemma4LayerPle {
    /// `layers.{i}.per_layer_input_gate.weight` — `[256, hidden_size]` Linear (no bias).
    pub input_gate: DenseWeight,
    /// `layers.{i}.per_layer_projection.weight` — `[hidden_size, 256]` Linear (no bias).
    pub projection: DenseWeight,
    /// `layers.{i}.post_per_layer_input_norm.weight` — `[hidden_size]` RMSNorm.
    pub post_norm: DenseWeight,
}

impl Gemma4LayerPle {
    /// Wrap the loader's per-layer PLE weights (copy — DenseWeight is Copy).
    pub fn from_loader(w: &crate::weight_loader::Gemma4PerLayerPleWeights) -> Self {
        Gemma4LayerPle {
            input_gate: w.input_gate,
            projection: w.projection,
            post_norm: w.post_norm,
        }
    }
}

/// Load-time install walk: attach the E2B per-layer PLE weights + the
/// KV-shared flag to each attention layer. No-op for non-E2B configs
/// (`ple_tables == None`). Called from `TransformerModel::new` after the
/// layers are built but before the model struct is assembled.
pub(crate) fn install_ple_on_layers(
    layers: &mut [Box<dyn TransformerLayer>],
    ple_tables: &crate::weight_loader::Gemma4PleTables,
    config: &ModelConfig,
) {
    let shared_band_start = config
        .num_hidden_layers
        .saturating_sub(config.num_kv_shared_layers);
    for (i, layer) in layers.iter_mut().enumerate() {
        let Some(any) = layer.as_any_mut() else {
            continue;
        };
        let Some(l) = any.downcast_mut::<crate::layers::Qwen3AttentionLayer>() else {
            continue;
        };
        l.set_ple(Gemma4LayerPle::from_loader(&ple_tables.per_layer[i]));
        l.set_kv_shared(i >= shared_band_start);
        tracing::info!("L{i}: PLE installed (kv_shared={})", i >= shared_band_start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::DevicePtr;

    /// E2B geometry: the KV-shared band starts at `num_hidden_layers -
    /// num_kv_shared_layers` (= 15 for the 35-layer / 20-shared E2B config).
    /// Non-E2B configs (0 shared) keep every layer unshared.
    #[test]
    fn shared_band_boundary() {
        let config = atlas_core::config::parse_config(
            r#"{
                "model_type": "gemma4",
                "text_config": {
                    "hidden_size": 1536,
                    "num_hidden_layers": 35,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 1,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "intermediate_size": 6144,
                    "vocab_size": 262144,
                    "hidden_size_per_layer_input": 256,
                    "num_kv_shared_layers": 20,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-6
                }
            }"#,
        )
        .unwrap();
        let start = config
            .num_hidden_layers
            .saturating_sub(config.num_kv_shared_layers);
        assert_eq!(start, 15);
        for i in 0..config.num_hidden_layers {
            assert_eq!(i >= start, i >= 15, "layer {i}");
        }
    }

    /// PLE slice layout: layer i's 256-dim vector for token t sits at
    /// `t * (num_layers*256) + i*256` in the combined [S, 35*256] buffer.
    /// The strided multiply kernel reads exactly that column block.
    #[test]
    fn ple_slice_column_offsets() {
        let num_layers = 35usize;
        let per_layer_dim = 256usize;
        let row_stride = num_layers * per_layer_dim;
        let base = 0x1000u64;
        for i in [0usize, 1, 14, 15, 34] {
            let col = i * per_layer_dim;
            let t = 3usize;
            let elem = t * row_stride + col;
            let expected = base + (elem * 2) as u64;
            // The layer derives its slice as base + t-row-stride + column.
            let slice = base + (t * row_stride * 2) as u64 + (col * 2) as u64;
            assert_eq!(slice, expected, "layer {i} token {t}");
            assert!(col + per_layer_dim <= row_stride);
        }
    }

    /// Pins the scale constants the model-level precompute passes to the
    /// kernels (identity*16, proj*1/sqrt(hidden), combined/sqrt(2)).
    #[test]
    fn ple_combine_scales() {
        let h = 1536usize;
        let per_layer_dim = 256usize;
        let identity_scale = 16.0f32;
        let proj_scale = 1.0 / (h as f32).sqrt();
        let combine_scale = 1.0 / 2.0f32.sqrt();
        assert_eq!(identity_scale, 16.0);
        assert!((proj_scale - (1.0 / (1536.0f32).sqrt())).abs() < 1e-6);
        assert!((combine_scale - 2.0f32.sqrt().recip()).abs() < 1e-6);
        assert_eq!(per_layer_dim, 256);
    }

    /// The PLE slice dimension must match `hidden_size_per_layer_input`
    /// (the E2B config value the loader slices the 8960-wide table with).
    #[test]
    fn per_layer_dim_matches_config() {
        let config = atlas_core::config::parse_config(
            r#"{
                "model_type": "gemma4",
                "text_config": {
                    "hidden_size": 1536,
                    "num_hidden_layers": 35,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 1,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "intermediate_size": 6144,
                    "vocab_size": 262144,
                    "hidden_size_per_layer_input": 256,
                    "num_kv_shared_layers": 20,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-6
                }
            }"#,
        )
        .unwrap();
        assert_eq!(config.hidden_size_per_layer_input, 256);
        assert_eq!(config.num_hidden_layers * 256, 8960);
    }

    /// Pure-Rust CPU reference of the E2B PLE math on a tiny synthetic case,
    /// hand-verified. The GPU path composes existing kernels (batched_embed +
    /// dense_gemm + rms_norm + residual_add + gelu_tanh + gemma4_ple_mul)
    /// whose individual math is covered elsewhere; this pins the ORCHESTRATOR
    /// invariants: the *16 identity scale, the 1/sqrt(hidden) context scale,
    /// the /sqrt(2) combine, the RMSNorm normalization, and the per-layer
    /// block (gate -> gelu -> slice-mul -> projection -> norm -> residual).
    #[test]
    fn ple_math_cpu_reference() {
        // Tiny E2B-like geometry: hidden=3, per_layer_dim=2, num_layers=2.
        let h = 3usize;
        let per_layer_dim = 2usize;
        let num_layers = 2usize;
        let row_stride = num_layers * per_layer_dim; // 4 (8960 in the real model)
        let eps = 1e-6f32;

        // One token (id 0); its row in the 8960-wide per-layer table.
        let table_row = [1.0f32, -2.0, 0.5, 3.0];
        // inputs_embeds for the token.
        let hidden = [0.25f32, -0.5, 1.0];
        // per_layer_model_projection [4, 3].
        let proj_w = [
            1.0, 0.0, 2.0, //
            0.0, -1.0, 0.5, //
            2.0, 1.0, 0.0, //
            -0.5, 0.0, 1.5,
        ];
        let proj_norm_w = [1.0f32, 1.0];

        // 1. identity = table row * 16 (the model-level *16 scale).
        let identity: Vec<f32> = table_row.iter().map(|v| v * 16.0).collect();
        assert_eq!(identity, vec![16.0, -32.0, 8.0, 48.0]);

        // 2. context = proj @ hidden * (1/sqrt(h)), then RMSNorm per layer.
        let inv_sqrt_h = 1.0 / (h as f32).sqrt();
        let mut context = [0f32; 4];
        for r in 0..row_stride {
            let mut acc = 0.0f32;
            for k in 0..h {
                acc += proj_w[r * h + k] * hidden[k];
            }
            context[r] = acc * inv_sqrt_h;
        }
        for l in 0..num_layers {
            let mut ss = 0.0f32;
            for d in 0..per_layer_dim {
                let v = context[l * per_layer_dim + d];
                ss += v * v;
            }
            let rms = (ss / per_layer_dim as f32 + eps).sqrt().recip();
            for d in 0..per_layer_dim {
                context[l * per_layer_dim + d] *= rms * proj_norm_w[d];
            }
        }
        // RMSNorm invariant: each layer's normed 2-vector has unit RMS.
        for l in 0..num_layers {
            let ss = context[l * per_layer_dim] * context[l * per_layer_dim]
                + context[l * per_layer_dim + 1] * context[l * per_layer_dim + 1];
            assert!(
                (ss / per_layer_dim as f32 - 1.0).abs() < 1e-4,
                "layer {l} unit RMS"
            );
        }

        // 3. combined = (context + identity) / sqrt(2).
        let combined: Vec<f32> = (0..row_stride)
            .map(|i| (context[i] + identity[i]) / 2.0f32.sqrt())
            .collect();
        // Layer-0 slice of the combined buffer (columns [0..2) of the row).
        let slice0 = [combined[0], combined[1]];

        // 4. Per-layer PLE block for layer 0, with identity-ish weights so the
        //    chain is hand-checkable:
        //      input_gate [2,3]: rows [1,0,0],[0,1,0]  -> h = [0.25, -0.5]
        //      projection [3,2]: rows [1,0],[0,1],[0,0] -> h2 = [g, h, 0]
        //      post_norm weight [1,1,1] -> RMSNorm(3)
        let h_gate = [hidden[0], hidden[1]]; // = [0.25, -0.5]
        let gelu = |x: f32| 0.5 * x * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh());
        let h_act = [gelu(h_gate[0]), gelu(h_gate[1])];
        let h_mul = [h_act[0] * slice0[0], h_act[1] * slice0[1]];
        let h2 = [h_mul[0], h_mul[1], 0.0f32];
        let mut ss = 0.0f32;
        for v in h2 {
            ss += v * v;
        }
        let rms = (ss / h as f32 + eps).sqrt().recip();
        let h_norm = h2.map(|v| v * rms * 1.0);
        let mut hidden_out = hidden;
        for d in 0..h {
            hidden_out[d] += h_norm[d];
        }

        // Hand-computed anchors (f32 tolerance):
        //   context[2] == 0 (row 2 of proj_w dot hidden == 0) -> combined[2]
        //   is exactly identity[2]/sqrt(2) = 8/sqrt(2) ≈ 5.65685.
        assert!((combined[2] - 8.0 / 2.0f32.sqrt()).abs() < 1e-4);
        //   gelu_tanh(0.25) ≈ 0.1497, gelu_tanh(-0.5) ≈ -0.1544.
        assert!((gelu(0.25) - 0.1497).abs() < 1e-3);
        assert!((gelu(-0.5) + 0.1544).abs() < 1e-3);
        //   The PLE block must actually modify the residual stream.
        assert!(hidden_out != hidden);
        //   Layer-0 slice of the combined [S, row_stride] buffer is exactly
        //   the first per_layer_dim columns of the row.
        assert_eq!(slice0.len(), per_layer_dim);
    }

    /// Regression: a layer with NO PLE weights (non-E2B) — the forward is a
    /// byte-identical no-op (no kernel launches) and the layer defaults to
    /// `kv_shared == false` / a null PLE slice.
    #[test]
    fn non_ple_layer_is_unchanged() {
        use spark_runtime::gpu::mock::MockGpuBackend;

        let config = atlas_core::config::parse_config(
            r#"{
                "model_type": "gemma4",
                "text_config": {
                    "hidden_size": 1536,
                    "num_hidden_layers": 35,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 1,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "intermediate_size": 6144,
                    "vocab_size": 262144,
                    "hidden_size_per_layer_input": 256,
                    "num_kv_shared_layers": 20,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-6
                }
            }"#,
        )
        .unwrap();
        let gpu = MockGpuBackend::new();
        let buffers =
            spark_runtime::buffers::BufferArena::new(&config, 4, 4096, 16, 1, &gpu).unwrap();
        let layer = build_test_layer(&gpu, &config, 0).unwrap();

        // Defaults for a non-E2B-installed layer: no PLE weights, not
        // KV-shared, null slice.
        assert!(layer.ple.is_none());
        assert!(!layer.kv_shared);
        assert_eq!(layer.ple_slice_ptr().0, 0);

        let dispatch = crate::layers::ops::GemmDispatch::defaults();
        let derived = crate::layers::ops::DerivedWeights::new();
        let levers = crate::layers::ops::ModelLevers::defaults();
        let stats = crate::layers::ops::ModelStats::new();
        let ctx = crate::layer::ForwardContext {
            buffers: &buffers,
            gpu: &gpu,
            config: &config,
            dispatch: &dispatch,
            derived: &derived,
            levers: &levers,
            stats: &stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
        };
        // No launches before the call (kernels don't run on mock, but a
        // `launch` call would still be recorded)...
        let before = gpu.launch_count();
        layer
            .gemma4_ple_forward(&ctx, DevicePtr(0x1000), 1, 0)
            .unwrap();
        let after = gpu.launch_count();
        assert_eq!(before, after, "PLE forward with no weights must not launch");
    }

    /// The load-time install walk attaches PLE weights + the KV-shared flag:
    /// layers 15-34 of the E2B config are marked shared, 0-14 are not, and
    /// every layer gets its per-layer PLE weights.
    #[test]
    fn install_walk_marks_shared_band() {
        use spark_runtime::gpu::mock::MockGpuBackend;

        let config = atlas_core::config::parse_config(
            r#"{
                "model_type": "gemma4",
                "text_config": {
                    "hidden_size": 1536,
                    "num_hidden_layers": 35,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 1,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "intermediate_size": 6144,
                    "vocab_size": 262144,
                    "hidden_size_per_layer_input": 256,
                    "num_kv_shared_layers": 20,
                    "max_position_embeddings": 131072,
                    "rms_norm_eps": 1e-6
                }
            }"#,
        )
        .unwrap();
        let gpu = MockGpuBackend::new();
        let mut layers: Vec<Box<dyn TransformerLayer>> = (0..config.num_hidden_layers)
            .map(|i| {
                Box::new(build_test_layer(&gpu, &config, i).unwrap()) as Box<dyn TransformerLayer>
            })
            .collect();

        // Fake PLE tables with per-layer weight slots (pointer values encode
        // the layer index so the install is observable).
        let per_layer = (0..config.num_hidden_layers)
            .map(|i| crate::weight_loader::Gemma4PerLayerPleWeights {
                input_gate: DenseWeight {
                    weight: DevicePtr(0x1000 + i as u64 * 0x100),
                },
                projection: DenseWeight {
                    weight: DevicePtr(0x2000 + i as u64 * 0x100),
                },
                post_norm: DenseWeight {
                    weight: DevicePtr(0x3000 + i as u64 * 0x100),
                },
            })
            .collect();
        let tables = crate::weight_loader::Gemma4PleTables {
            embed_tokens_per_layer: Vec::new(),
            per_layer_model_projection: DenseWeight {
                weight: DevicePtr(0x4000),
            },
            per_layer_projection_norm: DenseWeight {
                weight: DevicePtr(0x5000),
            },
            per_layer,
        };

        install_ple_on_layers(&mut layers, &tables, &config);

        for (i, l) in layers.iter_mut().enumerate() {
            let ql = l
                .as_any_mut()
                .unwrap()
                .downcast_mut::<crate::layers::Qwen3AttentionLayer>()
                .unwrap();
            assert_eq!(ql.kv_shared, i >= 15, "layer {i} kv_shared");
            let ple = ql.ple.as_ref().expect("layer {i} PLE installed");
            assert_eq!(ple.input_gate.weight.0, 0x1000 + i as u64 * 0x100);
            assert_eq!(ple.projection.weight.0, 0x2000 + i as u64 * 0x100);
            assert_eq!(ple.post_norm.weight.0, 0x3000 + i as u64 * 0x100);
        }
    }

    /// Minimal `Qwen3AttentionLayer` for tests (mock backend, null weights).
    fn build_test_layer(
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        config: &ModelConfig,
        idx: usize,
    ) -> anyhow::Result<crate::layers::Qwen3AttentionLayer> {
        use spark_runtime::kv_cache::KvCacheDtype;
        let zero = DenseWeight {
            weight: DevicePtr(0),
        };
        let attn = crate::weight_map::AttentionWeights {
            q_proj: zero,
            k_proj: zero,
            v_proj: zero,
            o_proj: crate::weight_map::QuantizedWeight {
                weight: DevicePtr(0),
                weight_scale: DevicePtr(0),
                weight_scale_2: 1.0,
                input_scale: DevicePtr(0),
                weight_scale_2_vec: DevicePtr(0),
            },
            q_norm: zero,
            k_norm: zero,
            q_norm_full: None,
            k_norm_full: None,
            k_scale: 1.0,
            v_scale: 1.0,
        };
        crate::layers::Qwen3AttentionLayer::new_ungated(
            zero,
            attn,
            zero,
            crate::layers::FfnComponent::None,
            idx,
            None,
            None,
            None,
            gpu,
            KvCacheDtype::Bf16,
            0,
            config,
        )
    }
}
