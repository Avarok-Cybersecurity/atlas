// SPDX-License-Identifier: AGPL-3.0-only

//! Atlas's GLM image preprocessor vs. `Glm5NextImageProcessor`'s own
//! `pixel_values`.
//!
//! The encoder golden (`glm_vit_golden`) feeds Atlas the REFERENCE's
//! `pixel_values`, so it proves the tower and says nothing about how those
//! pixels are produced. Everything the host side decides — the `smart_resize`
//! canvas, whether the content is scaled or padded, what the pad is filled
//! with, the normalization constants, and above all the 2x2-block-major patch
//! ORDER — is invisible to it. Each of those is silent when wrong: a raster
//! patch order or a SigLIP-normalised pixel still produces a fluent answer
//! about the wrong picture.
//!
//! So this pins the other half, against the same three fixtures. No GPU.
//!
//! Two questions this test ANSWERED rather than assumed, both recorded in
//! `vision_preprocess_glm::resize_and_pad`:
//!
//! * the pad is black pixels applied BEFORE normalization (padded cells hold
//!   `(0 - mean)/std`, not `0.0`), and
//! * the fit never upscales — image c's 360x640 content sits unresampled in a
//!   364x644 canvas, rather than being stretched 1.00625x to fill it.
//!
//! Run (no GPU, no kernels needed):
//!
//! ```text
//! AVAROK_GLM_VISION=1 \
//! GLM_VIT_GOLDEN_DIR=$HOME/lazarus/spark-bench/runs/ws2-vision \
//! cargo test -p spark-model --test glm_vision_preprocess_pin -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};

#[path = "glm_vit_golden/fixtures.rs"]
mod fixtures;
use fixtures::read_npy_f32;

/// Ceiling on max-abs error against the reference's `pixel_values`.
///
/// Both sides compute `(u8/255 - mean)/std` in f32 from the same PNG, so the
/// only legitimate difference is f32 rounding — `1e-5` is orders of magnitude
/// above that and orders of magnitude below one 1/255 pixel step (0.0146 in
/// normalized units), which is the smallest difference that could come from a
/// real disagreement about a pixel's VALUE rather than its bits.
const MAX_ABS: f32 = 1e-5;

struct Fixture {
    name: &'static str,
    png: &'static str,
    grid_h: usize,
    grid_w: usize,
}

const FIXTURES: [Fixture; 3] = [
    Fixture {
        name: "a_noise_448x448",
        png: "a_noise_448x448.png",
        grid_h: 32,
        grid_w: 32,
    },
    Fixture {
        name: "b_geometric_448x448",
        png: "b_geometric_448x448.png",
        grid_h: 32,
        grid_w: 32,
    },
    // The one that matters: 640x360 is not a multiple of 28, so this is the
    // only fixture that exercises smart_resize, the no-upscale rule and the
    // pad. It is also the only non-square grid, which is where an h/w
    // transposition in the block-major index would finally show.
    Fixture {
        name: "c_gradient_640x360",
        png: "c_gradient_640x360.png",
        grid_h: 26,
        grid_w: 46,
    },
];

fn golden_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("GLM_VIT_GOLDEN_DIR").expect("set GLM_VIT_GOLDEN_DIR (see module doc)"),
    )
}

/// The preprocessor's entry point takes a `data:` URI, which is what the API
/// edge hands it, so the test goes in the same way a request would.
fn png_data_uri(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes);
    Ok(format!("data:image/png;base64,{b64}"))
}

#[test]
#[ignore = "requires the reference fixtures in GLM_VIT_GOLDEN_DIR and AVAROK_GLM_VISION=1"]
fn glm_preprocess_matches_the_reference_pixel_values() -> Result<()> {
    ensure!(
        avarok_core::config::glm_vision_enabled(),
        "set AVAROK_GLM_VISION=1: with the gate off `parse_glm5_next` leaves \
         config.vision at None and there is no GLM vision config to test"
    );
    let root = golden_dir();
    let gold = root.join("golden");
    let cfg =
        avarok_core::config::parse_config(&std::fs::read_to_string(root.join("config.json"))?)
            .context("parse the checkpoint's own config.json")?;
    let v = cfg
        .vision
        .context("config.vision is None even with the gate on")?;
    ensure!(
        v.block_major_patches,
        "the GLM parser must declare block-major patch order"
    );
    println!(
        "mean={:?} std={:?} patch={} merge={} min/max_image_tokens={}/{}",
        v.image_mean,
        v.image_std,
        v.patch_size,
        v.spatial_merge_size,
        v.min_image_tokens,
        v.max_image_tokens
    );

    let mut failures = Vec::new();
    println!("\n| image | grid (h x w) | patches | max-abs | mean-abs | pad cells |");
    println!("|---|---|---|---|---|---|");
    for f in &FIXTURES {
        let uri = png_data_uri(&gold.join(f.png))?;
        let (pixels, grid_h, grid_w) =
            spark_model::vision_preprocess::preprocess_image_with_max_pixels(&uri, &v, None)
                .with_context(|| format!("preprocess {}", f.name))?;
        // The grid is the first thing to disagree if smart_resize or the
        // no-upscale rule is wrong, and it is checked before the pixels so a
        // geometry error reports as one.
        ensure!(
            (grid_h, grid_w) == (f.grid_h, f.grid_w),
            "{}: Atlas produced a {grid_h}x{grid_w} patch grid, the reference a {}x{}",
            f.name,
            f.grid_h,
            f.grid_w
        );

        let (shape, expect) = read_npy_f32(&gold.join(format!("{}_pixel_values.npy", f.name)))?;
        let patch_dim = 3 * v.temporal_patch_size * v.patch_size * v.patch_size;
        ensure!(
            shape == vec![grid_h * grid_w, patch_dim] && pixels.len() == expect.len(),
            "{}: golden pixel_values {shape:?} vs Atlas {} floats",
            f.name,
            pixels.len()
        );

        // A padded cell normalizes to -mean/std. Counting them is what makes
        // "the pad is black pixels, applied before normalization" a claim the
        // test can fail on rather than a comment.
        let pad_ch: Vec<f32> = (0..3).map(|c| -v.image_mean[c] / v.image_std[c]).collect();
        let mut pad_cells = 0usize;
        let (mut max_abs, mut sum_abs) = (0.0f32, 0.0f64);
        for (i, (&a, &b)) in pixels.iter().zip(&expect).enumerate() {
            ensure!(a.is_finite(), "{}: non-finite value at {i}", f.name);
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
            let channel = (i % patch_dim) / (v.temporal_patch_size * v.patch_size * v.patch_size);
            if (b - pad_ch[channel]).abs() < 1e-5 {
                pad_cells += 1;
            }
        }
        println!(
            "| {} | {grid_h} x {grid_w} | {} | {max_abs:.3e} | {:.3e} | {pad_cells} |",
            f.name,
            grid_h * grid_w,
            sum_abs / pixels.len() as f64
        );
        if max_abs > MAX_ABS {
            failures.push(format!("{}: max-abs {max_abs:.3e} > {MAX_ABS:.0e}", f.name));
        }
    }

    if !failures.is_empty() {
        bail!(
            "Atlas's GLM preprocessing does not match the reference processor:\n  {}",
            failures.join("\n  ")
        );
    }
    println!("\nPASS: 3 fixtures match the reference pixel_values within {MAX_ABS:.0e}");
    Ok(())
}
