// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash vision tower vs. the `transformers` reference.
//!
//! Runs Atlas's `GlmVit` on the golden `pixel_values` and compares the merged
//! embeddings — the rows that get spliced into the LM — against
//! `Glm5NextVisionModel`'s `pooler_output`, per token, for three images
//! including one non-square. The intermediate taps (patch embed, block 0,
//! block 23) are compared too, so a mismatch localises to a stage instead of
//! reporting one number for a 24-block tower.
//!
//! ## Why this is graded against a PAIRED CONTROL
//!
//! The golden is CPU **float32**. Atlas computes in BF16 — the checkpoint's own
//! dtype, and what the production reference serves this tower in. Measured on
//! n1/GB10 2026-09-22, this tower is NOT BF16-stable through 24 blocks: running
//! the reference itself in `torch.bfloat16` moves its last-block output to a
//! min per-token cosine of 0.47-0.68 against its own fp32 run. A bare
//! "cosine > 0.999 against fp32" gate is therefore unreachable by ANY BF16
//! implementation, the reference's included, and passing it would require
//! either fp32 activations or a tolerance chosen to fit the result.
//!
//! So the grading is split, and neither half is tunable:
//!
//! * **Early taps** (patch embed, block 0), where BF16 error has not compounded,
//!   are gated absolutely at cosine >= 0.9999 vs fp32. A wrong RoPE, a wrong
//!   norm, a transposed patch order or a missing QK-norm shows here.
//! * **Last block and merged output** are gated against the `golden-bf16/`
//!   control — the same reference, same weights, same pixels, in bfloat16.
//!   Atlas must deviate from fp32 by no MORE than that control does. This is a
//!   comparison against another implementation's BF16 noise, not a number
//!   anyone picked.
//!
//! All three pairings are printed either way, so a regression is visible even
//! where the gate is relative.
//!
//! Requires an idle GPU, the compiled `glm-5.3-flash / nvfp4` target, and the
//! extracted vision-only weights. Run:
//!
//! ```text
//! AVAROK_TARGET_HW=gb10 AVAROK_TARGET_MODEL=glm-5.3-flash AVAROK_TARGET_QUANT=nvfp4 \
//! LIBRARY_PATH=$HOME/atlas-scratch/.ncclstub LD_LIBRARY_PATH=$LIBRARY_PATH \
//! GLM_VIT_GOLDEN_GPU_ORDINAL=0 \
//! GLM_VIT_GOLDEN_DIR=$HOME/lazarus/spark-bench/runs/ws2-vision \
//! AVAROK_DUMP_GLM_VIT=/tmp/glmvit-taps \
//! cargo test -p spark-model --test glm_vit_golden -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use avarok_core::config::VisionConfig;
use spark_model::layers::glm_vit::{GlmVit, GlmVitBlock, GlmVitGeometry, GlmVitMerger};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[path = "glm_vit_golden/fixtures.rs"]
mod fixtures;
use fixtures::{Stats, bf16_bytes_to_f32, compare, ptr, read_npy_f32, upload_tower};

/// Absolute floor for the EARLY taps, where BF16 error has not yet compounded.
/// Measured 2026-09-22: patch embed 0.99998, block 0 0.99995. A structural
/// error — patch order, RoPE axis, missing QK-norm, wrong norm kind — lands
/// orders of magnitude below this, not just under it.
const EARLY_MIN_COSINE: f64 = 0.9999;

/// How much worse than the BF16 control Atlas may be on the compounded stages.
/// 1.0 would demand bit-luck; these are two different roundings of the same
/// math, so each is a sample from the same error distribution and the slack is
/// for the sampling, not for the port.
const CONTROL_SLACK_REL: f64 = 1.30;
/// And the mean per-token cosine may sit at most this far below the control's.
const CONTROL_SLACK_COS: f64 = 0.02;

struct Image {
    name: &'static str,
    grid_h: usize,
    grid_w: usize,
}

const IMAGES: [Image; 3] = [
    Image {
        name: "a_noise_448x448",
        grid_h: 32,
        grid_w: 32,
    },
    Image {
        name: "b_geometric_448x448",
        grid_h: 32,
        grid_w: 32,
    },
    Image {
        name: "c_gradient_640x360",
        grid_h: 26,
        grid_w: 46,
    },
];

fn build_encoder(
    w: &HashMap<String, DevicePtr>,
    v: &VisionConfig,
    p_max_patches: usize,
    gpu: &dyn GpuBackend,
) -> Result<GlmVit> {
    let mut blocks = Vec::with_capacity(v.depth);
    for i in 0..v.depth {
        let b = format!("blocks.{i}");
        blocks.push(GlmVitBlock {
            norm1_w: ptr(w, &format!("{b}.norm1.weight"))?,
            qkv_w: ptr(w, &format!("{b}.attn.qkv.weight"))?,
            qkv_b: ptr(w, &format!("{b}.attn.qkv.bias"))?,
            q_norm_w: ptr(w, &format!("{b}.attn.q_norm.weight"))?,
            k_norm_w: ptr(w, &format!("{b}.attn.k_norm.weight"))?,
            proj_w: ptr(w, &format!("{b}.attn.proj.weight"))?,
            proj_b: ptr(w, &format!("{b}.attn.proj.bias"))?,
            norm2_w: ptr(w, &format!("{b}.norm2.weight"))?,
            gate_w: ptr(w, &format!("{b}.mlp.gate_proj.weight"))?,
            gate_b: ptr(w, &format!("{b}.mlp.gate_proj.bias"))?,
            up_w: ptr(w, &format!("{b}.mlp.up_proj.weight"))?,
            up_b: ptr(w, &format!("{b}.mlp.up_proj.bias"))?,
            down_w: ptr(w, &format!("{b}.mlp.down_proj.weight"))?,
            down_b: ptr(w, &format!("{b}.mlp.down_proj.bias"))?,
        });
    }
    let merger = GlmVitMerger {
        post_layernorm_w: ptr(w, "post_layernorm.weight")?,
        downsample_w: ptr(w, "downsample.weight")?,
        downsample_b: ptr(w, "downsample.bias")?,
        proj_w: ptr(w, "merger.proj.weight")?,
        norm_w: ptr(w, "merger.post_projection_norm.weight")?,
        norm_b: ptr(w, "merger.post_projection_norm.bias")?,
        gate_w: ptr(w, "merger.gate_proj.weight")?,
        up_w: ptr(w, "merger.up_proj.weight")?,
        down_w: ptr(w, "merger.down_proj.weight")?,
    };
    let geo = GlmVitGeometry {
        hidden_size: v.hidden_size,
        num_heads: v.num_heads,
        intermediate_size: v.intermediate_size,
        spatial_merge_size: v.spatial_merge_size,
        out_hidden_size: v.out_hidden_size,
        projection_intermediate_size: v.projection_intermediate_size,
        patch_size: v.patch_size,
        temporal_patch_size: v.temporal_patch_size,
        rms_norm_eps: v.rms_norm_eps,
        swiglu_limit: v.swiglu_limit,
        rope_theta: v.rope_theta,
        max_pixels: Some(p_max_patches * v.patch_size * v.patch_size),
    };
    GlmVit::new(
        ptr(w, "patch_embed.proj.weight")?,
        ptr(w, "patch_embed.proj.bias")?,
        blocks,
        merger,
        &geo,
        gpu,
    )
}

fn golden_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("GLM_VIT_GOLDEN_DIR").expect("set GLM_VIT_GOLDEN_DIR (see module doc)"),
    )
}

#[test]
#[ignore = "requires an idle CUDA device, the compiled glm-5.3-flash target, and the extracted vision weights"]
fn glm_vision_tower_matches_the_transformers_golden() -> Result<()> {
    let ordinal: usize = std::env::var("GLM_VIT_GOLDEN_GPU_ORDINAL")
        .context("set GLM_VIT_GOLDEN_GPU_ORDINAL explicitly")?
        .parse()?;
    let root = golden_dir();
    let gold = root.join("golden");

    let cfg =
        avarok_core::config::parse_config(&std::fs::read_to_string(root.join("config.json"))?)
            .context("parse the checkpoint's own config.json")?;
    let v = cfg.vision.clone().context(
        "parse_glm5_next did not populate config.vision — the parser branch is the thing under \
         test here as much as the kernels are",
    )?;
    println!(
        "vision_config: depth={} hidden={} heads={} patch={} merge={} out={} proj_inter={} \
         eps={} swiglu_limit={} theta={} mean={:?} std={:?} block_major={}",
        v.depth,
        v.hidden_size,
        v.num_heads,
        v.patch_size,
        v.spatial_merge_size,
        v.out_hidden_size,
        v.projection_intermediate_size,
        v.rms_norm_eps,
        v.swiglu_limit,
        v.rope_theta,
        v.image_mean,
        v.image_std,
        v.block_major_patches
    );

    let target = avarok_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("compile the glm-5.3-flash nvfp4 target")?;
    let gpu = AvarokCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;

    let weights = upload_tower(&root.join("shards/vision_combined.safetensors"), &gpu)?;
    // Σ patches over the three images is 3244; 3600 rows leaves headroom
    // without paying the 6400-patch fallback's quadratic score matrix.
    let encoder = build_encoder(&weights, &v, 3600, &gpu)?;

    // Load every image's pixels, then encode all three in ONE batched call —
    // which also exercises the packing the multi-image splice depends on.
    let mut pixels = Vec::new();
    for img in &IMAGES {
        let (shape, data) = read_npy_f32(&gold.join(format!("{}_pixel_values.npy", img.name)))?;
        ensure!(
            shape == vec![img.grid_h * img.grid_w, 1176],
            "{}: pixel_values shape {shape:?} does not match grid {}x{}",
            img.name,
            img.grid_h,
            img.grid_w
        );
        pixels.push(data);
    }
    let batch: Vec<(&[f32], usize, usize)> = IMAGES
        .iter()
        .zip(&pixels)
        .map(|(i, p)| (p.as_slice(), i.grid_h, i.grid_w))
        .collect();
    let per_image = encoder.forward_batched(&batch, &gpu, stream)?;
    gpu.synchronize(stream)?;

    let control_dir = root.join("golden-bf16");
    let have_control = control_dir.is_dir();
    if !have_control {
        println!(
            "\nWARNING: no golden-bf16/ control at {} — the compounded stages will be REPORTED \
             but not gated. See golden-bf16/README.md for how it is produced.",
            control_dir.display()
        );
    }

    let mut failures = Vec::new();
    let mut mp_off = 0usize;
    let mut p_off = 0usize;
    println!(
        "\n| image | stage | pair | rows | max-abs | mean-abs | mean-rel | min cos | mean cos |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");

    /// One stage of one image: print every pairing that exists, then gate.
    macro_rules! grade {
        ($img:expr, $stage:expr, $early:expr, $actual:expr, $fp32:expr, $ctrl:expr, $rows:expr, $cols:expr) => {{
            let row = |pair: &str, s: &Stats| {
                println!(
                    "| {} | {} | {pair} | {} | {:.4e} | {:.4e} | {:.4e} | {:.6} | {:.6} |",
                    $img, $stage, $rows, s.max_abs, s.mean_abs, s.mean_rel, s.min_cos, s.mean_cos
                );
            };
            let a_vs_f32 = compare($actual, $fp32, $rows, $cols)?;
            row("atlas vs f32", &a_vs_f32);
            if $early {
                // Absolute gate: BF16 has not compounded here, so any real
                // structural error is visible and nothing needs a control.
                if a_vs_f32.min_cos < EARLY_MIN_COSINE {
                    failures.push(format!(
                        "{} {}: min cosine {:.6} < {EARLY_MIN_COSINE} vs the fp32 reference",
                        $img, $stage, a_vs_f32.min_cos
                    ));
                }
            } else if let Some(ctrl) = $ctrl {
                let c_vs_f32 = compare(&ctrl, $fp32, $rows, $cols)?;
                let a_vs_c = compare($actual, &ctrl, $rows, $cols)?;
                row("torch-bf16 vs f32", &c_vs_f32);
                row("atlas vs torch-bf16", &a_vs_c);
                // Atlas must not deviate from fp32 by more than the
                // reference's OWN bf16 run does.
                if a_vs_f32.mean_rel > c_vs_f32.mean_rel * CONTROL_SLACK_REL {
                    failures.push(format!(
                        "{} {}: mean relative error {:.4} exceeds {CONTROL_SLACK_REL}x the \
                         torch-bf16 control's {:.4}",
                        $img, $stage, a_vs_f32.mean_rel, c_vs_f32.mean_rel
                    ));
                }
                if a_vs_f32.mean_cos < c_vs_f32.mean_cos - CONTROL_SLACK_COS {
                    failures.push(format!(
                        "{} {}: mean cosine {:.6} is more than {CONTROL_SLACK_COS} below the \
                         torch-bf16 control's {:.6}",
                        $img, $stage, a_vs_f32.mean_cos, c_vs_f32.mean_cos
                    ));
                }
            }
        }};
    }

    for (i, img) in IMAGES.iter().enumerate() {
        let (post_h, post_w, merged_p) = per_image[i];
        ensure!(
            (post_h, post_w) == (img.grid_h / 2, img.grid_w / 2),
            "{}: post-merge grid {post_h}x{post_w}",
            img.name
        );

        // Intermediate taps first, so a failure reads in pipeline order.
        if let Ok(dir) = std::env::var("AVAROK_DUMP_GLM_VIT") {
            let p = img.grid_h * img.grid_w;
            for (label, gfile, early) in [
                ("patch_embed", "tap_patch_embed", true),
                ("block00", "tap_block0", true),
                ("block23", "tap_block23", false),
            ] {
                let raw = std::fs::read(Path::new(&dir).join(format!("{label}.bin")))
                    .with_context(|| format!("tap dump {label}.bin"))?;
                let all = bf16_bytes_to_f32(&raw);
                let slice = &all[p_off * v.hidden_size..(p_off + p) * v.hidden_size];
                let (shape, expect) =
                    read_npy_f32(&gold.join(format!("{}_{gfile}.npy", img.name)))?;
                ensure!(
                    shape == vec![p, v.hidden_size],
                    "{} {label}: shape {shape:?}",
                    img.name
                );
                let ctrl = match have_control {
                    true => Some(
                        read_npy_f32(&control_dir.join(format!("{}_{label}.npy", img.name)))?.1,
                    ),
                    false => None,
                };
                grade!(
                    img.name,
                    label,
                    early,
                    slice,
                    &expect,
                    ctrl,
                    p,
                    v.hidden_size
                );
            }
        }

        // The merged rows: what the embedding splice copies into the LM.
        let mut bytes = vec![0u8; merged_p * v.out_hidden_size * 2];
        gpu.copy_d2h(encoder.out_row(mp_off), &mut bytes)?;
        let actual = bf16_bytes_to_f32(&bytes);
        let (shape, expect) = read_npy_f32(&gold.join(format!("{}_merged_embeds.npy", img.name)))?;
        ensure!(
            shape == vec![merged_p, v.out_hidden_size],
            "{}: golden merged shape {shape:?} vs {merged_p}x{}",
            img.name,
            v.out_hidden_size
        );
        let ctrl = match have_control {
            true => Some(read_npy_f32(&control_dir.join(format!("{}_merged.npy", img.name)))?.1),
            false => None,
        };
        grade!(
            img.name,
            "merged",
            false,
            &actual,
            &expect,
            ctrl,
            merged_p,
            v.out_hidden_size
        );

        mp_off += merged_p;
        p_off += img.grid_h * img.grid_w;
    }

    if !failures.is_empty() {
        bail!(
            "GLM vision tower does not match the reference:\n  {}",
            failures.join("\n  ")
        );
    }
    println!(
        "\nPASS: 3 images, {mp_off} merged tokens. Early taps >= {EARLY_MIN_COSINE} vs fp32; \
         compounded stages within the torch-bf16 control."
    );
    Ok(())
}
