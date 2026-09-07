// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `kda_layer_microtest.rs` to keep it under the 500-LoC cap.
//! Test-only harness code; no serving path runs any of it.

#![allow(unused_imports)]

use crate::*;
use anyhow::{Context, Result, bail};
use half::bf16;
use kda_layer_cpu::*;
use kda_layer_harness::*;
use serde_json::Value;
use spark_model::layers::glm5next_kda::binding::{
    self, AttnBlockKind, KdaDtype, KdaTensorSource, RawTensor,
};
use spark_model::layers::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use spark_model::layers::glm5next_kda_ref as kref;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::collections::BTreeMap;

pub(crate) struct Row {
    pub(crate) name: &'static str,
    /// Largest |value| in the fp32 golden sample — without it a residual is unreadable.
    pub(crate) mag: f64,
    /// **A** — pure-fp32 reference vs the fp32 golden: HF/reference math only.
    pub(crate) floor_a: f64,
    /// **B** — bf16 golden vs fp32 golden: the activation-dtype budget.
    pub(crate) floor_b: f64,
    /// **D** — GPU vs a CPU reference on Atlas's exact bf16 ladder: kernel residual only.
    pub(crate) floor_d: f64,
    pub(crate) gpu_vs_bf16: f64,
    pub(crate) gpu_vs_f32: f64,
    pub(crate) ck_rel: f64,
}

pub(crate) fn ck_rel(a: f64, b: f64) -> f64 {
    let d = a.abs().max(b.abs()).max(1e-30);
    (a - b).abs() / d
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn row(
    name: &'static str,
    gpu: &[f32],
    cpu: &[f32],
    pure: Option<&[f32]>,
    gold: &Value,
    key: &str,
) -> Result<Row> {
    let get = |dt: &str| -> Result<(Vec<f32>, f64, usize, usize)> {
        let e = &gold[dt][key];
        if e.is_null() {
            bail!("golden {dt} is missing stage {key}");
        }
        let data: Vec<f32> = e["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        Ok((
            data,
            e["ck"].as_f64().unwrap(),
            e["n"].as_u64().unwrap() as usize,
            e["stride"].as_u64().unwrap() as usize,
        ))
    };
    let (gbf, ck_bf, n, stride) = get("bf16")?;
    let (gf32, _, n2, s2) = get("f32")?;
    if n != gpu.len() || n2 != gpu.len() {
        bail!("stage {key}: golden n={n} but GPU produced {}", gpu.len());
    }
    if stride != s2 {
        bail!("stage {key}: bf16/f32 goldens disagree on stride ({stride} vs {s2})");
    }
    let sg = sample(gpu, stride);
    Ok(Row {
        name,
        mag: gf32.iter().fold(0.0f64, |m, x| m.max((*x as f64).abs())),
        floor_a: match pure {
            Some(p) => maxabs(&sample(p, stride), &gf32),
            None => f64::NAN,
        },
        floor_b: maxabs(&gbf, &gf32),
        floor_d: maxabs(gpu, cpu),
        gpu_vs_bf16: maxabs(&sg, &gbf),
        gpu_vs_f32: maxabs(&sg, &gf32),
        ck_rel: ck_rel(checksum(gpu), ck_bf),
    })
}

pub(crate) fn print_table(title: &str, rows: &[Row], rows_ref: Option<f64>) {
    println!("\n  {title}");
    println!(
        "    {:<18} {:>10} {:>10} {:>10} {:>10} {:>11} {:>10} {:>9}",
        "stage", "|max|", "A:ref", "B:bf16", "D:kernel", "GPUvsHFbf16", "GPUvsHFf32", "ck_rel"
    );
    for r in rows {
        println!(
            "    {:<18} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} {:>11.3e} {:>10.3e} {:>9.2e}",
            r.name, r.mag, r.floor_a, r.floor_b, r.floor_d, r.gpu_vs_bf16, r.gpu_vs_f32, r.ck_rel
        );
    }
    println!("    (C: dequantised-real-weight — N/A, nothing in a KDA block is quantised)");
    if let Some(d) = rows_ref {
        println!(
            "    kda_reference_layer cross-check (recurrent formulation, fp32): max_abs {d:.3e}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────── main

/// The three KDA layers executed against a real-checkpoint oracle: early (dense FFN neighbour),
/// middle, and the LAST KDA layer before MTP. Chosen from the audited `kda_layers` list, not by
/// convenience — the point is to prove the reusable component, not layer 0.
pub(crate) const EXEC_LAYERS: &[usize] = &[0, 22, 44];

pub(crate) fn report(
    title: &str,
    t: usize,
    dm: Dims,
    gs: &Stages,
    cs: &Stages,
    ca: Option<&Stages>,
    golden: Option<&Value>,
    regime: &str,
    tail3: &dyn Fn(&[f32]) -> Vec<f32>,
) -> Result<bool> {
    let prefix = if regime == "prefill7_decode1" {
        "prefill_"
    } else {
        ""
    };
    stage_report(title, t, dm, gs, cs, ca, golden, regime, prefix, tail3)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn report_decode_leg(
    title: &str,
    dm: Dims,
    gs: &Stages,
    cs: &Stages,
    ca: Option<&Stages>,
    golden: Option<&Value>,
    tail3: &dyn Fn(&[f32]) -> Vec<f32>,
) -> Result<bool> {
    stage_report(
        title,
        1,
        dm,
        gs,
        cs,
        ca,
        golden,
        "prefill7_decode1",
        "",
        tail3,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_report(
    title: &str,
    t: usize,
    dm: Dims,
    gs: &Stages,
    cs: &Stages,
    ca: Option<&Stages>,
    golden: Option<&Value>,
    regime: &str,
    prefix: &str,
    tail3: &dyn Fn(&[f32]) -> Vec<f32>,
) -> Result<bool> {
    let g_tail = tail3(&gs.conv_state);
    let c_tail = tail3(&cs.conv_state);
    let a_tail = ca.map(|a| tail3(&a.conv_state));

    let Some(v) = golden else {
        // Synthetic run: only GPU-vs-CPU (floor D) exists — there is no HF golden for LCG weights.
        let d = [
            ("qkv_proj", maxabs(&gs.qkv_proj, &cs.qkv_proj)),
            ("conv+L2 q", maxabs(&gs.q, &cs.q)),
            ("conv+L2 k", maxabs(&gs.k, &cs.k)),
            ("conv v", maxabs(&gs.v, &cs.v)),
            ("gate", maxabs(&gs.gate, &cs.gate)),
            ("beta", maxabs(&gs.beta, &cs.beta)),
            ("kda core", maxabs(&gs.core, &cs.core)),
            ("recurrent state", maxabs(&gs.state, &cs.state)),
            ("out_gate", maxabs(&gs.out_gate, &cs.out_gate)),
            ("o_norm", maxabs(&gs.o_norm, &cs.o_norm)),
            ("o_proj / final", maxabs(&gs.final_out, &cs.final_out)),
            ("conv state", maxabs(&g_tail, &c_tail)),
        ];
        println!("\n  {title}  (T={t}) — floor D only, no HF golden for synthetic weights");
        let mut worst = 0.0f64;
        for (n, e) in d {
            println!("    {n:<22} D:GPUvsCPU {e:>11.3e}");
            worst = worst.max(e);
        }
        let pass = worst <= 5.0e-2;
        println!(
            "    worst floor-D residual {worst:.3e}  [{}]",
            if pass { "ok" } else { "FAIL" }
        );
        return Ok(pass);
    };

    let gold = &v["regimes"];
    let sel = |dt: &str| -> Value { gold[format!("{dt}__{regime}")].clone() };
    let bundle = serde_json::json!({ "bf16": sel("bf16"), "f32": sel("f32") });
    let key = |n: &str| format!("{prefix}{n}");

    let rows = vec![
        row(
            "qkv_proj",
            &gs.qkv_proj,
            &cs.qkv_proj,
            ca.map(|a| a.qkv_proj.as_slice()),
            &bundle,
            &key("qkv_proj"),
        )?,
        row(
            "conv+L2 q",
            &gs.q,
            &cs.q,
            ca.map(|a| a.q.as_slice()),
            &bundle,
            &key("q_l2"),
        )?,
        row(
            "conv+L2 k",
            &gs.k,
            &cs.k,
            ca.map(|a| a.k.as_slice()),
            &bundle,
            &key("k_l2"),
        )?,
        row(
            "conv v",
            &gs.v,
            &cs.v,
            ca.map(|a| a.v.as_slice()),
            &bundle,
            &key("v_raw"),
        )?,
        row(
            "gate",
            &gs.gate,
            &cs.gate,
            ca.map(|a| a.gate.as_slice()),
            &bundle,
            &key("gate"),
        )?,
        row(
            "beta",
            &gs.beta,
            &cs.beta,
            ca.map(|a| a.beta.as_slice()),
            &bundle,
            &key("beta"),
        )?,
        row(
            "kda core",
            &gs.core,
            &cs.core,
            ca.map(|a| a.core.as_slice()),
            &bundle,
            &key("core"),
        )?,
        row(
            "recurrent state",
            &gs.state,
            &cs.state,
            ca.map(|a| a.state.as_slice()),
            &bundle,
            &key("state"),
        )?,
        row(
            "out_gate",
            &gs.out_gate,
            &cs.out_gate,
            ca.map(|a| a.out_gate.as_slice()),
            &bundle,
            &key("out_gate"),
        )?,
        row(
            "o_norm",
            &gs.o_norm,
            &cs.o_norm,
            ca.map(|a| a.o_norm.as_slice()),
            &bundle,
            &key("o_norm_out"),
        )?,
        row(
            "o_proj / final",
            &gs.final_out,
            &cs.final_out,
            ca.map(|a| a.final_out.as_slice()),
            &bundle,
            &key("final_out"),
        )?,
        row(
            "conv state",
            &g_tail,
            &c_tail,
            a_tail.as_deref(),
            &bundle,
            &key("conv_state"),
        )?,
    ];
    print_table(
        &format!("{title}  (T={t})"),
        &rows,
        ca.and_then(|a| a.ref_layer_delta),
    );

    // E = the integrated-layer residual at the layer output, judged against floor B there.
    let fin = rows.iter().find(|r| r.name == "o_proj / final").unwrap();
    let stt = rows.iter().find(|r| r.name == "recurrent state").unwrap();
    let ratio = fin.gpu_vs_bf16 / fin.floor_b.max(1e-30);
    println!(
        "    E: final_out GPU vs HF-bf16 {:.3e}   floor B {:.3e}   E/B {:.2}",
        fin.gpu_vs_bf16, fin.floor_b, ratio
    );
    let _ = dm;
    // 🪤 An ABSOLUTE tolerance does not transfer between layers: on real weights the layer output
    // grows ~100x from layer 0 to layer 44 (|final_out|max 3.6e-2 -> 4.1e0 on the same fixture).
    // So the gate is the RATIO to the measured bf16 floor, with a fallback expressed as a
    // fraction of the stage's own magnitude rather than a fixed constant. The carried state gets
    // its own gate because no later stage can correct it.
    let gate = |r: &Row| r.gpu_vs_bf16 <= (r.floor_b * 8.0).max(r.mag * 0.01);
    let pass = gate(fin) && gate(stt);
    if !pass {
        println!("    [FAIL] residual exceeds 8x the bf16 floor");
    }
    Ok(pass)
}
