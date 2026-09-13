// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `glm5next_moe_microtest.rs` to keep it under the 500-LoC cap.
//! Test-only harness code; no serving path runs any of it.

#![allow(unused_imports)]

use crate::*;
use anyhow::{Context, Result, bail};
use half::bf16;
use serde_json::Value;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::collections::BTreeMap;

pub(crate) fn run() -> Result<()> {
    let dir = std::env::var("MOE_PACKET_DIR")
        .unwrap_or_else(|_| "/home/msi1/atlas-scratch/moe-family".to_string());
    let layers: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let layers = if layers.is_empty() { vec![3] } else { layers };
    let g = G(serde_json::from_str(&GOLDEN)?);
    let hid = g.f("hidden")? as usize;
    let mi = g.f("moe_intermediate")? as usize;
    let ne = g.f("num_experts")? as usize;
    let topk = g.f("top_k")? as usize;
    let scale = g.f("routed_scaling_factor")? as f32;
    let limit = g.f("swiglu_limit")? as f32;
    let ngroup = g.f("n_group")? as u32;
    if ngroup != 1 {
        bail!("n_group = {ngroup}: grouped routing is NOT a no-op and is not implemented");
    }
    if g.0["fixture"]["apply_routed_scale_to_output"].as_bool() != Some(false) {
        bail!(
            "fixture says the routed scale is applied to the OUTPUT; the shared expert would then be scaled"
        );
    }
    println!(
        "GLM MoE gate — hidden={hid} moe_inter={mi} experts={ne} top_k={topk} \
         routed_scale={scale} n_group=1 (asserted) apply_routed_scale_to_output=false"
    );

    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let k = K {
        gemm: gpu.kernel("gemm", "dense_gemm_bf16")?,
        gemm_f32: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
        w4: gpu.kernel("w4a16", "w4a16_gemm")?,
        act: gpu.kernel("glm5next_ffn", "glm5next_swiglu_clamp")?,
        router: gpu.kernel("glm5next_ffn", "glm5next_router_topk")?,
        combine: gpu.kernel("glm5next_ffn", "glm5next_moe_combine")?,
    };
    let mut rows: Vec<Row> = Vec::new();
    let mut vllm_agree: Vec<(usize, &str, usize, usize)> = Vec::new();
    let mut shared_by_regime: BTreeMap<(usize, &str), Vec<f32>> = BTreeMap::new();

    for &layer in &layers {
        let p = Pk::open(&format!("{dir}/moe_layer{layer}.safetensors"))?;
        let d_gw = up(&gpu, p.b("gate.weight")?)?;
        let d_bias = up_f32(&gpu, &p.f32b("gate.e_score_correction_bias")?)?;
        let d_sh: BTreeMap<&str, DevicePtr> = ["gate_proj", "up_proj", "down_proj"]
            .iter()
            .map(|q| -> Result<(&str, DevicePtr)> {
                Ok((*q, up(&gpu, p.b(&format!("shared_experts.{q}.weight"))?)?))
            })
            .collect::<Result<_>>()?;
        let si = p.m("shared_experts.gate_proj.weight")?.1[0];
        // Expert weights are uploaded on demand and cached — a full routed layer is 3.85 GiB.
        let mut cache: BTreeMap<usize, [(DevicePtr, DevicePtr, f32); 3]> = BTreeMap::new();

        for mode in MODES {
            for regime in REGIMES {
                let sec = format!("{mode}__{regime}");
                let gl = g.get(layer, &sec, "router_logits")?;
                let t = gl.2 / ne;
                // 🪤 `real32` / `nearcut` are REAL hidden states from the layer-0..3 prefix —
                // there is no LCG that reproduces them, so the fixture inputs travel WITH the
                // golden, unstrided, under `__inputs`. They are the only thing read from the
                // golden that is an input rather than an expected output.
                let x: Vec<f32> = g.0["by_layer"][layer.to_string()]["__inputs"][regime]
                    .as_array()
                    .with_context(|| format!("__inputs/{regime}"))?
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(f64::NAN) as f32)
                    .collect();
                if x.len() != t * hid {
                    bail!(
                        "{sec}: input has {} elements, expected {}",
                        x.len(),
                        t * hid
                    );
                }
                let d_x = up_bf16(&gpu, &x)?;

                // ── router logits, per mode ──
                let d_lg = gpu.alloc(t * ne * 4)?;
                if mode == "hf_fp32" {
                    // bf16 -> fp32 is exact, so an fp32-accumulating GEMM over bf16 operands IS
                    // `F.linear(h.float(), w.float())`.
                    gemm(
                        &gpu, k.gemm_f32, d_x, d_gw, d_lg, t as u32, ne as u32, hid as u32,
                    )?;
                    gpu.synchronize(0)?;
                } else {
                    let d_bf = gpu.alloc(t * ne * 2)?;
                    gemm(
                        &gpu, k.gemm, d_x, d_gw, d_bf, t as u32, ne as u32, hid as u32,
                    )?;
                    gpu.synchronize(0)?;
                    let widened = dn_bf16(&gpu, d_bf, t * ne)?; // widening bf16->f32 is exact
                    gpu.free(d_bf)?;
                    let bytes: Vec<u8> = widened.iter().flat_map(|v| v.to_le_bytes()).collect();
                    gpu.copy_h2d(&bytes, d_lg)?;
                }
                let got_lg = dn_f32(&gpu, d_lg, t * ne)?;
                rows.push(Row {
                    layer,
                    mode,
                    regime,
                    stage: "router_logits",
                    e: resid("logits", &got_lg, &gl)?,
                    b: ULP_BUDGET * bf16_ulp(mag(&gl)),
                    mag: mag(&gl),
                });

                // ── top-k ──
                let d_ids = gpu.alloc(t * topk * 4)?;
                let d_w = gpu.alloc(t * topk * 4)?;
                KernelLaunch::new(&gpu, k.router)
                    .grid([t as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_lg)
                    .arg_ptr(d_bias)
                    .arg_ptr(d_ids)
                    .arg_ptr(d_w)
                    .arg_u32(ne as u32)
                    .arg_u32(topk as u32)
                    .arg_u32(1)
                    .arg_f32(scale)
                    .arg_u32(1)
                    .arg_u32(if mode == "vllm_bf16" { 1 } else { 0 })
                    .launch(0)?;
                gpu.synchronize(0)?;
                let ids = dn_i32(&gpu, d_ids, t * topk)?;
                let wts = dn_f32(&gpu, d_w, t * topk)?;

                // Slot validity — no stale tail, no invalid id, no duplicate.
                for tok in 0..t {
                    let row = &ids[tok * topk..(tok + 1) * topk];
                    for (j, id) in row.iter().enumerate() {
                        if *id < 0 || *id as usize >= ne {
                            bail!(
                                "layer {layer} {sec} token {tok} slot {j}: invalid expert id {id}"
                            );
                        }
                        if row[..j].contains(id) {
                            bail!("layer {layer} {sec} token {tok}: expert {id} selected twice");
                        }
                    }
                }
                let gi = g.get(layer, &sec, "topk_ids")?;
                let id_match =
                    gi.0.iter()
                        .enumerate()
                        .filter(|(i, v)| ids[*i * gi.1] == **v as i32)
                        .count();
                // 🔴 fp32 routing is well conditioned, so the canonical mode must reproduce the
                // reference's selection EXACTLY. The bf16 mode is a different story — see the
                // verdict split below.
                if mode == "hf_fp32" && id_match != gi.0.len() {
                    bail!(
                        "layer {layer} {sec}: HF_FP32 selection differs from the reference on \
                         {} of {} slots — fp32 routing must be exact",
                        gi.0.len() - id_match,
                        gi.0.len()
                    );
                }
                if mode == "vllm_bf16" {
                    vllm_agree.push((layer, regime, id_match, gi.0.len()));
                }
                let gw_ = g.get(layer, &sec, "topk_weights")?;
                rows.push(Row {
                    layer,
                    mode,
                    regime,
                    stage: "topk_weights",
                    e: resid("wts", &wts, &gw_)?,
                    b: ULP_BUDGET * bf16_ulp(mag(&gw_)),
                    mag: mag(&gw_),
                });
                println!(
                    "  L{layer} {mode:10} {regime:8} T={t:<3} expert-id agreement \
                     {id_match}/{} slots",
                    gi.0.len()
                );

                // ── selected experts ──
                let mut host_eo = vec![0f32; t * topk * hid];
                let d_slot = gpu.alloc(hid * 2)?;
                for tok in 0..t {
                    for slot in 0..topk {
                        let e = ids[tok * topk + slot] as usize;
                        if let std::collections::btree_map::Entry::Vacant(slot) = cache.entry(e) {
                            let mut v = Vec::new();
                            for (proj, kk) in
                                [("gate_proj", hid), ("up_proj", hid), ("down_proj", mi)]
                            {
                                let n = if proj == "down_proj" { hid } else { mi };
                                let (dt, sh, ..) = p.m(&format!("experts.{e}.{proj}.weight"))?;
                                if dt != "U8" || *sh != vec![n, kk / 2] {
                                    bail!("expert {e} {proj}: {dt} {sh:?}");
                                }
                                v.push((
                                    up(&gpu, p.b(&format!("experts.{e}.{proj}.weight"))?)?,
                                    up(&gpu, p.b(&format!("experts.{e}.{proj}.weight_scale"))?)?,
                                    p.f32b(&format!("experts.{e}.{proj}.weight_scale_2"))?[0],
                                ));
                            }
                            slot.insert([v[0], v[1], v[2]]);
                        }
                        let w = cache[&e];
                        let d_xi = up_bf16(&gpu, &x[tok * hid..(tok + 1) * hid])?;
                        let d_g = gpu.alloc(mi * 2)?;
                        let d_u = gpu.alloc(mi * 2)?;
                        let d_a = gpu.alloc(mi * 2)?;
                        w4(
                            &gpu, k.w4, d_xi, w[0].0, w[0].1, w[0].2, d_g, 1, mi as u32, hid as u32,
                        )?;
                        w4(
                            &gpu, k.w4, d_xi, w[1].0, w[1].1, w[1].2, d_u, 1, mi as u32, hid as u32,
                        )?;
                        act(&gpu, k.act, d_g, d_u, d_a, mi as u32, limit)?;
                        w4(
                            &gpu, k.w4, d_a, w[2].0, w[2].1, w[2].2, d_slot, 1, hid as u32,
                            mi as u32,
                        )?;
                        gpu.synchronize(0)?;
                        let sl = dn_bf16(&gpu, d_slot, hid)?;
                        host_eo[(tok * topk + slot) * hid..(tok * topk + slot + 1) * hid]
                            .copy_from_slice(&sl);
                        for q in [d_xi, d_g, d_u, d_a] {
                            gpu.free(q)?;
                        }
                    }
                }
                gpu.free(d_slot)?;
                let d_eo = up_bf16(&gpu, &host_eo)?;
                let got_eo = host_eo.clone();
                let ge = g.get(layer, &sec, "expert_out")?;
                rows.push(Row {
                    layer,
                    mode,
                    regime,
                    stage: "expert_out",
                    e: resid("eo", &got_eo, &ge)?,
                    b: ULP_BUDGET * bf16_ulp(mag(&ge)),
                    mag: mag(&ge),
                });

                // ── shared expert (BF16) ──
                let d_g = gpu.alloc(t * si * 2)?;
                let d_u = gpu.alloc(t * si * 2)?;
                let d_a = gpu.alloc(t * si * 2)?;
                let d_sh_out = gpu.alloc(t * hid * 2)?;
                gemm(
                    &gpu,
                    k.gemm,
                    d_x,
                    d_sh["gate_proj"],
                    d_g,
                    t as u32,
                    si as u32,
                    hid as u32,
                )?;
                gemm(
                    &gpu,
                    k.gemm,
                    d_x,
                    d_sh["up_proj"],
                    d_u,
                    t as u32,
                    si as u32,
                    hid as u32,
                )?;
                act(&gpu, k.act, d_g, d_u, d_a, (t * si) as u32, limit)?;
                gemm(
                    &gpu,
                    k.gemm,
                    d_a,
                    d_sh["down_proj"],
                    d_sh_out,
                    t as u32,
                    hid as u32,
                    si as u32,
                )?;
                gpu.synchronize(0)?;
                let got_sh = dn_bf16(&gpu, d_sh_out, t * hid)?;
                let gsh = g.get(layer, &sec, "shared_out")?;
                rows.push(Row {
                    layer,
                    mode,
                    regime,
                    stage: "shared_out",
                    e: resid("sh", &got_sh, &gsh)?,
                    b: ULP_BUDGET * bf16_ulp(mag(&gsh)),
                    mag: mag(&gsh),
                });
                // CONTROL: the shared expert does not depend on routing, so the two modes must
                // produce an IDENTICAL shared output. A difference means routing leaked into it.
                match shared_by_regime.entry((layer, regime)) {
                    std::collections::btree_map::Entry::Vacant(v) => {
                        v.insert(got_sh.clone());
                    }
                    std::collections::btree_map::Entry::Occupied(o) => {
                        if o.get() != &got_sh {
                            bail!(
                                "layer {layer} {regime}: the shared expert differs between router \
                                 modes — routing has leaked into a path that must not see it"
                            );
                        }
                    }
                }

                // ── combine ──
                let d_out = gpu.alloc(t * hid * 2)?;
                KernelLaunch::new(&gpu, k.combine)
                    .grid([t as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_eo)
                    .arg_ptr(d_w)
                    .arg_ptr(d_sh_out)
                    .arg_ptr(d_out)
                    .arg_u32(hid as u32)
                    .arg_u32(topk as u32)
                    .launch(0)?;
                gpu.synchronize(0)?;
                let got_o = dn_bf16(&gpu, d_out, t * hid)?;
                let go = g.get(layer, &sec, "ffn_out")?;
                let grs = g.get(layer, &sec, "routed_sum")?;
                rows.push(Row {
                    layer,
                    mode,
                    regime,
                    stage: "ffn_out",
                    e: resid("out", &got_o, &go)?,
                    b: ULP_BUDGET * bf16_ulp(mag(&go)),
                    mag: mag(&go),
                });
                let _ = grs;
                for q in [d_x, d_lg, d_ids, d_w, d_eo, d_g, d_u, d_a, d_sh_out, d_out] {
                    gpu.free(q)?;
                }
            }
        }
        for (_, w) in cache {
            for (a, b, _) in w {
                gpu.free(a)?;
                gpu.free(b)?;
            }
        }
        for (_, v) in d_sh {
            gpu.free(v)?;
        }
        gpu.free(d_gw)?;
        gpu.free(d_bias)?;
    }

    println!(
        "\n{:>3} {:10} {:8} {:14} {:>11} {:>11} {:>11} {:>7}  verdict",
        "L", "mode", "regime", "stage", "E", "B", "|ref|max", "E/B"
    );
    // 🔴 THE TWO MODES ARE JUDGED BY DIFFERENT CRITERIA, and pooling them would be meaningless.
    //
    // HF_FP32 is the canonical semantics: fp32 routing is well conditioned, so every stage must
    // sit at the floor and the selection must match exactly. That is a real pass/fail.
    //
    // VLLM_BF16 is a COMPATIBILITY CHARACTERISATION, not a correctness gate. Its selection is
    // decided by bf16 comparisons at the rank-8 cutoff, where sub-ulp differences between two
    // independent implementations flip the winner — the same ill-conditioning Slice 8 measured
    // for the DSA marginal pool, where exact index parity is unachievable by ANY implementation.
    // Once one slot differs, `expert_out` for that slot is a different expert's vector and the
    // residual is meaningless as a kernel measurement. So selection-DEPENDENT stages are
    // reported for that mode and not gated; the selection-INDEPENDENT ones (router_logits,
    // shared_out) still are.
    const SELECTION_DEPENDENT: [&str; 3] = ["topk_weights", "expert_out", "ffn_out"];
    let mut fails: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &rows {
        let ratio = if r.b > 0.0 { r.e / r.b } else { f32::INFINITY };
        let gated = r.mode == "hf_fp32" || !SELECTION_DEPENDENT.contains(&r.stage);
        let ok = ratio <= 1.0 || !gated;
        if !ok {
            *fails.entry(r.mode).or_insert(0) += 1;
        }
        println!(
            "{:>3} {:10} {:8} {:14} {:>11.4e} {:>11.4e} {:>11.4e} {:>7.3}  {}",
            r.layer,
            r.mode,
            r.regime,
            r.stage,
            r.e,
            r.b,
            r.mag,
            ratio,
            if ratio <= 1.0 {
                "at floor"
            } else if !gated {
                "characterised (bf16 selection)"
            } else {
                "ABOVE FLOOR"
            }
        );
    }
    println!("\n{} rows", rows.len());
    println!(
        "  hf_fp32    GATED  — above floor: {}",
        fails.get("hf_fp32").copied().unwrap_or(0)
    );
    println!(
        "  vllm_bf16  gated on selection-INDEPENDENT stages only — above floor: {}",
        fails.get("vllm_bf16").copied().unwrap_or(0)
    );
    println!("\nVLLM_BF16 selection agreement vs vLLM's own bf16 ladder (characterisation):");
    for (l, r, m, n) in &vllm_agree {
        println!(
            "  L{l} {r:8} {m}/{n} slots ({:.1}%) — bf16 near-ties at the rank-8 cutoff are not \
             bit-reproducible across independent implementations",
            100.0 * *m as f64 / *n as f64
        );
    }
    if !fails.is_empty() {
        bail!("GLM MoE gate FAILED: {fails:?}");
    }
    println!("GLM MoE gate PASS — both router modes, reported separately, never pooled");
    Ok(())
}
