// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `mhc_microtest.rs` to keep it under the 500-LoC cap.
//! Test-only harness code; no serving path runs any of it.

#![allow(unused_imports)]

use crate::*;
use anyhow::{Context, Result, bail};
use half::bf16;
use serde_json::Value;
use spark_model::layers::ops::{Glm5NextMhcKernels, hc_head_mean, hc_post, hc_pre};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::collections::BTreeMap;

pub(crate) fn run() -> Result<()> {
    let dir = std::env::var("MHC_PACKET_DIR")
        .unwrap_or_else(|_| "/home/msi1/atlas-scratch/mhc-family".to_string());
    let g = Golden::load()?;
    let hid = g.fixture("hidden")? as usize;
    let hc = g.fixture("hc_mult")? as usize;
    let mix = g.fixture("mix")? as usize;
    let iters = g.fixture("hc_sinkhorn_iters")? as u32;
    let hc_eps = g.fixture("hc_eps")? as f32;
    let norm_eps = g.fixture("rms_norm_eps")? as f32;
    if mix != (2 + hc) * hc {
        bail!("fixture mix {mix} != (2+hc)*hc");
    }
    println!(
        "mHC gate — hidden={hid} hc_mult={hc} mix={mix} sinkhorn_iters={iters} \
         hc_eps={hc_eps:e} rms_norm_eps={norm_eps:e}"
    );

    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    // Two mHC entry points, deliberately separate:
    //   `hyper_connection::hc_pre`   — DeepSeek-V4's, frozen, ends on an EXACT column projection.
    //   `glm5next_mhc::glm5next_hc_pre` — GLM's, same signature, that block removed.
    // `hc_post` is deviation-free and is shared verbatim.
    let k_pre_v4: KernelHandle = gpu.kernel("hyper_connection", "hc_pre")?;
    let k_post_v4: KernelHandle = gpu.kernel("hyper_connection", "hc_post")?;
    // GATE 0 — TARGET INDEPENDENCE. Every kernel GLM's mHC needs must resolve from the single
    // module `glm5next_mhc`. If this succeeds, a GLM kernel target never has to carry the
    // DeepSeek-V4 `hyper_connection` module to complete its own hyper-connection.
    let glm = Glm5NextMhcKernels::resolve(&gpu)?;
    println!(
        "gate 0: GLM mHC resolves hc_pre + hc_post + hc_head from module '{}' alone",
        spark_model::layers::ops::GLM5NEXT_MHC_MODULE
    );
    // (arm, hc_pre, hc_post). The GLM arm uses GLM's hc_post; the V4 arm keeps V4's. Both must
    // hit the same golden — hc_post is deviation-free, and that is asserted, not assumed.
    let arms: [(&str, KernelHandle, KernelHandle); 2] = [
        ("glm", glm.hc_pre, glm.hc_post),
        ("v4", k_pre_v4, k_post_v4),
    ];
    // Max |column sum - 1| of the produced `comb`, per arm. The V4 arm pins columns to exactly
    // 1; GLM's eps-ending Sinkhorn must NOT. If these two ever agree, the entry points have
    // collapsed into one and the whole separation is fiction.
    let mut colsum_dev: BTreeMap<&str, f32> = BTreeMap::new();

    let mut rows: Vec<Row> = Vec::new();

    for &layer in LAYERS.iter() {
        let p = Packet::open(&format!("{dir}/mhc_layer{layer}.safetensors"))?;
        // Per-tensor dtype + shape assertions. Zero unknown, zero silent skips.
        let mut params: BTreeMap<&str, (Vec<f32>, DevicePtr)> = BTreeMap::new();
        for site in SITES.iter() {
            for (part, want_shape, want_dt) in [
                ("fn", vec![mix, hc * hid], "BF16"),
                ("base", vec![mix], "F32"),
                ("scale", vec![3usize], "F32"),
            ] {
                let name = format!("hc_{site}_{part}");
                let (v, shape, dt) = p.f32s(&name)?;
                if shape != want_shape {
                    bail!("{name}: shape {shape:?} != {want_shape:?}");
                }
                if dt != want_dt {
                    bail!("{name}: dtype {dt} != {want_dt} — the mHC dtype ladder moved");
                }
                let ptr = up_f32(&gpu, &v)?;
                params.insert(Box::leak(name.into_boxed_str()), (v, ptr));
            }
        }

        for &(regime, t) in REGIMES.iter() {
            for &(arm, k_pre, k_post) in arms.iter() {
                // The f32 arm isolates ARITHMETIC: Atlas's highway is f32, so feeding the raw f32
                // stream and comparing against the f32 golden asks only "is the math the same".
                let mut rng = Lcg::new(0x0E1C_0DE5);
                let streams: Vec<f32> = rng.t(t * hc * hid).iter().map(|x| x * 0.5).collect();

                let d_streams = up_f32(&gpu, &streams)?;
                let d_y = gpu.alloc(t * hid * 2)?;
                let d_post = gpu.alloc(t * hc * 4)?;
                let d_comb = gpu.alloc(t * hc * hc * 4)?;
                let d_out = gpu.alloc(t * hc * hid * 4)?;

                let mut cur = d_streams;
                for &site in SITES.iter() {
                    let (_, fn_p) = &params[format!("hc_{site}_fn").as_str()];
                    let (_, base_p) = &params[format!("hc_{site}_base").as_str()];
                    let (_, scale_p) = &params[format!("hc_{site}_scale").as_str()];
                    let block_out: Vec<f32> = rng.t(t * hid).iter().map(|x| x * 0.5).collect();

                    hc_pre(
                        &gpu, k_pre, cur, *fn_p, *scale_p, *base_p, d_y, d_post, d_comb, t as u32,
                        hid as u32, hc as u32, iters, norm_eps, hc_eps, 0,
                    )?;
                    gpu.synchronize(0)?;

                    let got_post = down_f32(&gpu, d_post, t * hc)?;
                    let got_comb = down_f32(&gpu, d_comb, t * hc * hc)?;
                    let got_y = down_bf16(&gpu, d_y, t * hid)?;

                    // Defining property of each entry point: V4 pins comb columns to EXACTLY 1;
                    // GLM's eps-ending Sinkhorn leaves them at 1 - O(hc_eps).
                    {
                        let mut worst = 0.0f32;
                        for tok in 0..t {
                            let o = tok * hc * hc;
                            for j in 0..hc {
                                let mut c = 0.0f32;
                                for i in 0..hc {
                                    c += got_comb[o + i * hc + j];
                                }
                                worst = worst.max((c - 1.0).abs());
                            }
                        }
                        let e = colsum_dev.entry(arm).or_insert(0.0);
                        *e = e.max(worst);
                    }

                    for (stage, got) in [
                        ("post", &got_post),
                        ("comb", &got_comb),
                        ("collapsed", &got_y),
                    ] {
                        let key = format!("{site}__{stage}");
                        let want = g.get(layer, "f32", regime, &key)?;
                        let e = residual(&key, got, &want)?;
                        let (b, mag) = floor_b(&g, layer, regime, &key)?;
                        rows.push(Row {
                            arm,
                            layer,
                            regime,
                            site,
                            stage,
                            e,
                            b,
                            mag,
                        });
                    }

                    // ── Attribution, not assertion ──
                    // `hc_pre` ends its Sinkhorn with an EXACT column projection that GLM's reference
                    // does NOT have: HF divides by `(colsum + hc_eps)` on every pass, so its columns
                    // settle at `1 - O(hc_eps)`, while Atlas pins them to exactly 1. If that single
                    // deviation is the WHOLE story, then re-normalising the reference's own `comb`
                    // columns to exactly 1 must collapse the residual onto the activation floor.
                    // A residual that stays put here would mean a second, unexplained difference.
                    {
                        let key = format!("{site}__comb");
                        let (want, stride, n) = g.get(layer, "f32", regime, &key)?;
                        // `long2176` stores `comb` strided (34,816 > the 4,096 unstrided cap), and a
                        // column sum needs all `hc` entries of a column. Attribution therefore runs on
                        // the unstrided regimes; the deviation is per-token and token-independent, so
                        // decode1/short7/medium64 across 4 layers x 2 sites is the same claim.
                        if stride == 1 {
                            let mut fixed = want.clone();
                            for tok in 0..n / (hc * hc) {
                                let o = tok * hc * hc;
                                for j in 0..hc {
                                    let mut c = 0.0f32;
                                    for i in 0..hc {
                                        c += fixed[o + i * hc + j];
                                    }
                                    if c > 0.0 {
                                        for i in 0..hc {
                                            fixed[o + i * hc + j] /= c;
                                        }
                                    }
                                }
                            }
                            let e = residual(&key, &got_comb, &(fixed, 1, n))?;
                            let (b, mag) = floor_b(&g, layer, regime, &key)?;
                            rows.push(Row {
                                arm,
                                layer,
                                regime,
                                site,
                                stage: "comb@colproj",
                                e,
                                b,
                                mag,
                            });
                        }
                    }

                    let d_block = up_bf16(&gpu, &block_out)?;
                    hc_post(
                        &gpu, k_post, d_block, cur, d_post, d_comb, d_out, t as u32, hid as u32,
                        hc as u32, 0,
                    )?;
                    gpu.synchronize(0)?;
                    let got_out = down_f32(&gpu, d_out, t * hc * hid)?;
                    let key = format!("{site}__site_out");
                    let want = g.get(layer, "f32", regime, &key)?;
                    let e = residual(&key, &got_out, &want)?;
                    let (b, mag) = floor_b(&g, layer, regime, &key)?;
                    rows.push(Row {
                        arm,
                        layer,
                        regime,
                        site,
                        stage: "site_out",
                        e,
                        b,
                        mag,
                    });

                    gpu.free(d_block)?;
                    // Chain: the ffn site consumes the attn site's residual write.
                    if cur != d_streams {
                        gpu.free(cur)?;
                    }
                    cur = gpu.alloc(t * hc * hid * 4)?;
                    let bytes: Vec<u8> = got_out.iter().flat_map(|x| x.to_le_bytes()).collect();
                    gpu.copy_h2d(&bytes, cur)?;
                }
                // ── hc_head: the FINAL collapse, after both sites ──
                // 🔴 GLM's is a parameterless MEAN. The V4 arm has nothing comparable — its
                // kernel needs `hc_head.{fn,base,scale}` and this checkpoint has none — so only
                // the GLM path can be gated here, and that asymmetry IS the finding.
                if arm == "glm" {
                    hc_head_mean(
                        &gpu,
                        glm.hc_head,
                        cur,
                        d_y,
                        t as u32,
                        hid as u32,
                        hc as u32,
                        0,
                    )?;
                    gpu.synchronize(0)?;
                    let got_head = down_bf16(&gpu, d_y, t * hid)?;
                    let want = g.get(layer, "f32", regime, "hc_head_out")?;
                    let e = residual("hc_head_out", &got_head, &want)?;
                    let (b, mag) = floor_b(&g, layer, regime, "hc_head_out")?;
                    rows.push(Row {
                        arm,
                        layer,
                        regime,
                        site: "final",
                        stage: "hc_head",
                        e,
                        b,
                        mag,
                    });
                }
                if cur != d_streams {
                    gpu.free(cur)?;
                }
                for q in [d_streams, d_y, d_post, d_comb, d_out] {
                    gpu.free(q)?;
                }
            }
        }
        for (_, (_, ptr)) in params.iter() {
            gpu.free(*ptr)?;
        }
    }

    println!(
        "\n{:4} {:>3} {:9} {:5} {:12} {:>11} {:>11} {:>11} {:>7}  verdict",
        "arm", "L", "regime", "site", "stage", "E (gpu-f32)", "B (bf16fl)", "|ref|max", "E/B"
    );
    // The GLM arm gets NO waiver: its whole reason to exist is that the projection is gone, so
    // any above-floor row is a real failure. The V4 arm is EXPECTED to sit above the floor on
    // `comb` — that is the projection it deliberately keeps — and is checked for exactly that.
    let colproj_ok: std::collections::BTreeSet<(&str, usize, &str, &str)> = rows
        .iter()
        .filter(|r| r.stage == "comb@colproj" && r.b > 0.0 && r.e <= r.b)
        .map(|r| (r.arm, r.layer, r.regime, r.site))
        .collect();
    let mut fails = 0usize;
    let mut v4_known = 0usize;
    let mut glm_worst = 0.0f32;
    for r in &rows {
        let ratio = if r.b > 0.0 { r.e / r.b } else { f32::INFINITY };
        let mut ok = ratio <= 1.0;
        let mut label = if ok { "at floor" } else { "ABOVE FLOOR" };
        // `comb@colproj` is a DIAGNOSTIC, not a criterion: it re-normalises the reference's own
        // comb columns to exactly 1 and asks which arm that matches. It is expected to be at
        // floor for V4 and ABOVE floor for GLM — that inversion is the separation proof, and is
        // asserted below. Scoring it as a criterion would demand both at once.
        let is_diagnostic = r.stage == "comb@colproj";
        if !is_diagnostic && r.arm == "glm" && ratio.is_finite() && ratio > glm_worst {
            glm_worst = ratio;
        }
        if !ok && is_diagnostic {
            ok = true;
            label = if r.arm == "glm" {
                "GLM does NOT project (expected)"
            } else {
                "diagnostic"
            };
        }
        if !ok
            && r.arm == "v4"
            && r.stage == "comb"
            && colproj_ok.contains(&(r.arm, r.layer, r.regime, r.site))
        {
            ok = true;
            v4_known += 1;
            label = "V4 colproj (expected)";
        }
        if !ok {
            fails += 1;
        }
        if !ok || r.stage == "comb" || r.stage == "comb@colproj" || r.stage == "hc_head" {
            println!(
                "{:4} {:>3} {:9} {:5} {:12} {:>11.4e} {:>11.4e} {:>11.4e} {:>7.3}  {}",
                r.arm, r.layer, r.regime, r.site, r.stage, r.e, r.b, r.mag, ratio, label
            );
        }
    }

    let glm_dev = colsum_dev.get("glm").copied().unwrap_or(f32::NAN);
    let v4_dev = colsum_dev.get("v4").copied().unwrap_or(f32::NAN);
    println!("\nmax |comb column sum - 1|:  glm {glm_dev:.4e}   v4 {v4_dev:.4e}");
    println!(
        "{} rows ({} glm / {} v4), {} unexplained above floor, {} V4 colproj rows as expected",
        rows.len(),
        rows.iter().filter(|r| r.arm == "glm").count(),
        rows.iter().filter(|r| r.arm == "v4").count(),
        fails,
        v4_known
    );
    println!("worst GLM E/B = {glm_worst:.3}");

    if fails > 0 {
        bail!("mHC gate FAILED: {fails} stage(s) above the reference's own bf16 floor");
    }
    if glm_worst > 1.0 {
        bail!("mHC gate FAILED: the GLM arm must be at floor everywhere, worst E/B {glm_worst:.3}");
    }
    // The MIRROR. At L0/decode1/attn the reference's own bf16 floor collapses to 3.0e-7, which
    // is the one place the ~hc_eps projection is resolvable. There, each arm must match exactly
    // one version of the reference and not the other:
    //   GLM matches the raw reference, and NOT the column-projected one.
    //   V4  matches the column-projected reference, and NOT the raw one.
    // Two independent facts; either one alone would also be satisfied by a no-op.
    let at = |arm: &str, stage: &str| -> Option<f32> {
        rows.iter()
            .find(|r| {
                r.arm == arm
                    && r.layer == 0
                    && r.regime == "decode1"
                    && r.site == "attn"
                    && r.stage == stage
            })
            .map(|r| r.e / r.b)
    };
    for (arm, raw_should_match) in [("glm", true), ("v4", false)] {
        let raw = at(arm, "comb").unwrap_or(f32::NAN);
        let proj = at(arm, "comb@colproj").unwrap_or(f32::NAN);
        let ok = if raw_should_match {
            raw <= 1.0 && proj > 1.0
        } else {
            raw > 1.0 && proj <= 1.0
        };
        println!(
            "mirror @L0/decode1/attn  {arm:4} raw E/B {raw:.3}  colproj E/B {proj:.3}  -> {}",
            if ok { "as expected" } else { "WRONG" }
        );
        if !ok {
            bail!(
                "{arm} arm failed the mirror: raw E/B {raw:.3}, colproj E/B {proj:.3}. The two \
                 hc_pre entry points are not behaving as separate implementations."
            );
        }
    }

    // Separation proof. If these collapse together, the GLM entry point is not doing its job
    // (or the V4 one silently lost its projection) and every "at floor" above is meaningless.
    // Thresholds: V4's exact projection leaves only f32 rounding on a 4-term column sum
    // (measured 2.4e-7); GLM's eps-ending Sinkhorn leaves ~hc_eps (measured 1.2e-6). 5e-7 sits
    // between them by an order of magnitude on each side.
    let v4_pinned = v4_dev.is_finite() && v4_dev < 5e-7;
    if !v4_pinned {
        bail!(
            "V4 hc_pre no longer pins comb columns to exactly 1 (max dev {v4_dev:e}) — its \
               proven behaviour changed"
        );
    }
    let glm_unpinned = glm_dev.is_finite() && glm_dev > 5e-7;
    if !glm_unpinned {
        bail!(
            "glm5next_hc_pre pins comb columns to exactly 1 (max dev {glm_dev:e}) — it is \
               still doing the DeepSeek-V4 projection"
        );
    }
    let head_rows = rows.iter().filter(|r| r.stage == "hc_head").count();
    if head_rows != LAYERS.len() * REGIMES.len() {
        bail!("expected one hc_head row per layer x regime, got {head_rows}");
    }
    if v4_known == 0 {
        bail!("the V4 arm showed no colproj deviation — the two arms are not distinguishable");
    }

    println!(
        "\nmHC gate PASS\n  GLM  glm5next_mhc::glm5next_hc_pre — at the bf16 floor on every \
         stage, no waiver. Columns at 1-O(hc_eps), matching HF's eps-ending Sinkhorn.\n  V4   \
         hyper_connection::hc_pre — UNCHANGED: columns still pinned to exactly 1, and it still \
         shows the ~hc_eps comb deviation vs GLM's reference. Its source file is untouched by \
         this change."
    );
    println!(
        "  HEAD glm5next_mhc::glm5next_hc_head — parameterless MEAN, {head_rows} rows at the \
         bf16 floor. DeepSeek-V4's learned sigmoid-weighted hc_head is NOT used and could not \
         be: this checkpoint carries zero hc_head tensors."
    );
    println!(
        "⛔ historical note: hc_head. Atlas's is DeepSeek-V4's LEARNED sigmoid-weighted sum; \
         GLM's Glm5NextTextHyperHead is a parameterless MEAN and the checkpoint carries ZERO \
         hc_head tensors. That one is ADAPT, not REUSE."
    );
    Ok(())
}
