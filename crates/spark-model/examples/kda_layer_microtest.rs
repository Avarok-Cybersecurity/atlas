// SPDX-License-Identifier: AGPL-3.0-only
//! Slice 6 REMAINDER — the integrated GLM-5.3-Flash KDA layer, end to end.
//!
//! Three parts:
//!
//! 1. **Synthetic full-layer oracle** — LCG weights at production geometry, four regimes
//!    (decode T=1 · short prefill · ragged prefill · ragged prefill -> decode), GPU vs a CPU
//!    reference built from `glm5next_kda_ref` primitives, cross-checked against
//!    `glm5next_kda_ref::kda_reference_layer` on the sub-path that function covers.
//! 2. **Real layer-0 checkpoint oracle** — the 16-tensor 262.7 MiB packet extracted from
//!    `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `9e0d74e3`, shard 1/120, versus a golden produced by
//!    the genuine `transformers` 5.16.1 `Glm5NextTextLinearAttention` on the same bytes. Nothing
//!    in a KDA block is quantised, so this golden IS the production numerics.
//! 3. **Pad-corruption regression (Slice 5)** — a ragged prefill whose padded tail is poisoned,
//!    followed by a decode. Correct prefill outputs are NOT sufficient evidence: the original
//!    bug left every output right and the carried state off by 1.623e13.
//!
//! Floors are reported separately so a residual is never confused with a rounding budget:
//!   * **A** HF/reference math — CPU reference in fp32 vs the fp32 HF golden.
//!   * **B** bf16 activation/input — the bf16 HF golden vs the fp32 HF golden.
//!   * **C** dequantised-real-weight — **N/A for KDA**: nothing in the block is quantised.
//!   * **D** GPU kernel residual — GPU vs a CPU reference fed the SAME bf16-rounded values.
//!   * **E** complete integrated-layer residual — GPU final output vs the bf16 HF golden.
//!
//! 🪤 Atlas's L2 writes **bf16** (fused on decode, `l2_norm_bf16` on prefill); HF normalises in
//! **fp32 inside** the KDA kernel. Atlas therefore carries one extra bf16 rounding on q|k that
//! HF does not, and every downstream stage inherits it. That is a contract difference, not an
//! error — it is why floor D is measured against a CPU reference that reproduces Atlas's exact
//! dtype ladder rather than against HF.
//!
//!   KDA_LAYER0_PACKET=/path/to/layer0.safetensors \
//!   cargo run -p spark-model --release --example kda_layer_microtest \
//!       --features cuda,gpu-examples

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use half::bf16;
use serde_json::Value;
use spark_model::layers::glm5next_kda::binding::{self, AttnBlockKind, KdaTensorSource};
use spark_model::layers::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[path = "common/kda_layer_cpu.rs"]
pub(crate) mod kda_layer_cpu;
use kda_layer_cpu::*;

#[path = "common/kda_layer_harness.rs"]
pub(crate) mod kda_layer_harness;
use kda_layer_harness::*;

#[path = "common/kda_layer_report.rs"]
pub(crate) mod kda_layer_report;
use kda_layer_report::*;

#[path = "common/golden.rs"]
pub(crate) mod golden;

pub(crate) static GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/spark-model/src/layers/glm5next_kda_ref/kda_layer_golden.json",
        "gen_kda_layer_golden.py",
    )
});

/// Chunk width for the prefill scan. `C = 64` needs 81 920 B of shared memory and the backend
/// has no `cuFuncSetAttribute` opt-in, so 32 is the ceiling at D = 128 (blocker 12). Slice 5
/// verified identical results at C = 2..32, and HF runs C = 64 — agreement across both is part
/// of what this test shows.
pub(crate) const CHUNK: usize = 32;

// ───────────────────────────────────────────────────────────────────── plumbing

pub(crate) fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
pub(crate) fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
pub(crate) fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
pub(crate) fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

pub(crate) struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.u()).collect()
    }
    fn scaled(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.u() * s).collect()
    }
}

pub(crate) fn r(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
/// Rounding that the `pure` (floor-A) pass switches off. Weights are BF16 *on disk* — that is
/// the checkpoint, not a rounding — so only INTERMEDIATE values are affected.
pub(crate) fn rq(x: f32, pure: bool) -> f32 {
    if pure { x } else { r(x) }
}
pub(crate) fn round_bf16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| r(*x)).collect()
}
pub(crate) fn sample(v: &[f32], stride: usize) -> Vec<f32> {
    v.iter().step_by(stride).copied().collect()
}
pub(crate) fn maxabs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
}
pub(crate) fn checksum(s: &[f32]) -> f64 {
    s.iter()
        .enumerate()
        .map(|(i, v)| *v as f64 * (i as f64 + 1.0))
        .sum()
}

// ────────────────────────────────────────────────── CPU reference, Atlas's ladder

// ─────────────────────────────────────────────────────────────── GPU harness
//
// The GPU path lives ENTIRELY in `spark_model::layers::glm5next_kda`. Nothing here re-implements
// any of its math; this is a driver plus an independent CPU oracle.

// ────────────────────────────────────────────────────────────── comparison table

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let v: Value = serde_json::from_str(&GOLDEN)?;
    let f = &v["fixture"];
    let dm = Dims {
        hid: f["hidden"].as_u64().unwrap() as usize,
        h: f["heads"].as_u64().unwrap() as usize,
        d: f["head_dim"].as_u64().unwrap() as usize,
        ks: f["kernel"].as_u64().unwrap() as usize,
    };
    let cfg = Glm5NextKdaConfig {
        hidden: dm.hid,
        heads: dm.h,
        head_dim: dm.d,
        conv_kernel: dm.ks,
        gate_lower_bound: f["lower_bound"].as_f64().unwrap() as f32,
        rms_norm_eps: f["rms_eps"].as_f64().unwrap() as f32,
        l2_eps: f["l2_eps"].as_f64().unwrap() as f32,
        chunk: CHUNK,
    };

    println!("GLM-5.3-Flash KDA layer family — Atlas vs HF transformers 5.16.1");
    println!("  checkpoint {}", f["checkpoint"]);
    println!(
        "  hidden={} heads={} head_dim={} conv_dim={} kernel={} act={} o_norm_act={}",
        dm.hid,
        dm.h,
        dm.d,
        dm.conv_dim(),
        dm.ks,
        f["hidden_act"],
        f["o_norm_act"]
    );
    println!(
        "  READ from config: gate_lower_bound={} rms_norm_eps={:e} (never defaulted)",
        cfg.gate_lower_bound, cfg.rms_norm_eps
    );
    println!(
        "  Atlas chunk C={CHUNK} (smem ceiling), HF chunk C={}",
        f["hf_chunk"]
    );

    let probe: Vec<f32> = v["lcg_probe"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    if Lcg(0x5EED_1A70)
        .vec(probe.len())
        .iter()
        .zip(&probe)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("LCG mismatch with the generator");
    }
    println!("  LCG parity with the generator: ok");

    let kernels = Glm5NextKdaKernels::resolve(gpu)?;
    println!(
        "  {} kernel entry points resolved (no fallback path)",
        Glm5NextKdaKernels::ENTRY_POINTS
    );
    let ws = Glm5NextKdaWorkspace::new(gpu, &cfg, 8)?;

    let mut ok = true;

    // ── PART 1 — synthetic full-layer oracle ─────────────────────────────────
    println!("\n=== PART 1 — synthetic full-layer oracle (LCG weights, production geometry) ===");
    let mut wr = Lcg(0xA11A_5000);
    let wts = Wts {
        q: round_bf16(&wr.scaled(dm.qkv() * dm.hid, 0.02)),
        k: round_bf16(&wr.scaled(dm.qkv() * dm.hid, 0.02)),
        v: round_bf16(&wr.scaled(dm.qkv() * dm.hid, 0.02)),
        conv: round_bf16(&wr.scaled(dm.conv_dim() * dm.ks, 0.5)),
        f_a: round_bf16(&wr.scaled(dm.d * dm.hid, 0.02)),
        f_b: round_bf16(&wr.scaled(dm.qkv() * dm.d, 0.05)),
        dt_bias: wr.scaled(dm.qkv(), 0.5),
        a_log: wr.scaled(dm.h, 0.5),
        b: round_bf16(&wr.scaled(dm.h * dm.hid, 0.02)),
        g_a: round_bf16(&wr.scaled(dm.d * dm.hid, 0.02)),
        g_b: round_bf16(&wr.scaled(dm.qkv() * dm.d, 0.05)),
        o_norm: round_bf16(&wr.scaled(dm.d, 1.0)),
        o: round_bf16(&wr.scaled(dm.hid * dm.qkv(), 0.02)),
    };
    let syn = Glm5NextKdaWeights {
        q_proj: dwt(gpu, &wts.q)?,
        k_proj: dwt(gpu, &wts.k)?,
        v_proj: dwt(gpu, &wts.v)?,
        conv: dwt(gpu, &wts.conv)?,
        f_a: dwt(gpu, &wts.f_a)?,
        f_b: dwt(gpu, &wts.f_b)?,
        dt_bias: up_f32(gpu, &wts.dt_bias)?,
        a_log: up_f32(gpu, &wts.a_log)?,
        b_proj: dwt(gpu, &wts.b)?,
        g_a: dwt(gpu, &wts.g_a)?,
        g_b: dwt(gpu, &wts.g_b)?,
        o_norm: dwt(gpu, &wts.o_norm)?,
        o_proj: dwt(gpu, &wts.o)?,
    };
    ok &= run_suite(
        gpu,
        Glm5NextKdaLayer::new(usize::MAX, cfg, syn, kernels)?,
        &ws,
        dm,
        cfg,
        &wts,
        None,
        "synthetic",
    )?;

    // ── PART 2 — bind EVERY KDA block in the checkpoint ──────────────────────
    let dir = std::env::var("KDA_PACKET_DIR")
        .unwrap_or_else(|_| "/home/msi1/atlas-scratch/kda-family".to_string());
    let audit: Value = serde_json::from_str(&std::fs::read_to_string(format!(
        "{dir}/kda_family_audit.json"
    ))?)?;
    let kda_layers: Vec<usize> = audit["kda_layers"]
        .as_array()
        .context("audit has no kda_layers")?
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();

    println!(
        "\n=== PART 2 — typed binding of ALL {} KDA blocks ===",
        kda_layers.len()
    );
    println!("  packets {dir}");
    let mut family: Vec<(usize, Glm5NextKdaLayer)> = Vec::new();
    let (mut tot_bound, mut tot_unknown, mut tot_bytes, mut tot_nonattn) =
        (0usize, 0usize, 0usize, 0usize);
    let mut sig: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for &l in &kda_layers {
        let pkt = Packet::open(&format!("{dir}/layer{l}.safetensors"))?;
        let kind = binding::classify_attn_block(&pkt.names());
        if kind != AttnBlockKind::Kda {
            bail!("layer {l} classifies as {kind:?}, not Kda — refusing to bind it as KDA");
        }
        // Name/dtype/shape signature, so uniformity is measured here and not assumed.
        let mut names: Vec<String> = pkt
            .names()
            .into_iter()
            .filter(|n| n.starts_with("self_attn."))
            .map(|n| {
                let t = pkt.get(&n).unwrap();
                format!("{n}:{}:{:?}", t.dtype.name(), t.shape)
            })
            .collect();
        names.sort();
        sig.entry(names.join("|")).or_default().push(l);

        let (w, rep) = binding::bind_kda_weights(gpu, &cfg, l, &pkt)?;
        tot_bound += rep.bound;
        tot_unknown += rep.unknown_self_attn.len();
        tot_bytes += rep.bytes;
        tot_nonattn += rep.non_attn_seen;
        family.push((l, Glm5NextKdaLayer::new(l, cfg, w, kernels)?));
    }
    println!(
        "  bound {tot_bound}/{} self_attn tensors across {} layers · UNKNOWN {tot_unknown} · \
         silent skips 0 · {:.2} GiB · {tot_nonattn} non-attn tensors seen and deliberately NOT bound",
        binding::KDA_TENSORS.len() * kda_layers.len(),
        kda_layers.len(),
        tot_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "  distinct (name, dtype, shape) signatures across the family: {}",
        sig.len()
    );
    for ls in sig.values() {
        println!("    one signature covers {} layers: {:?}", ls.len(), ls);
    }
    if sig.len() != 1 {
        bail!(
            "the KDA family is NOT uniform — {} distinct signatures; STOP",
            sig.len()
        );
    }
    if tot_bound != binding::KDA_TENSORS.len() * kda_layers.len() || tot_unknown != 0 {
        bail!("binding accounting failed");
    }

    // ── PART 3 — execute early / middle / late blocks against the real oracle ─
    println!("\n=== PART 3 — real-weight execution: KDA layers {EXEC_LAYERS:?} ===");
    for &l in EXEC_LAYERS {
        if !kda_layers.contains(&l) {
            bail!("layer {l} is not a KDA layer");
        }
        let pkt = Packet::open(&format!("{dir}/layer{l}.safetensors"))?;
        let (w, _) = binding::bind_kda_weights(gpu, &cfg, l, &pkt)?;
        let host = host_weights(&pkt);
        // The golden holds one regime block per layer; hand `run_suite` just this layer's.
        let lv = serde_json::json!({ "regimes": v["by_layer"][l.to_string()] });
        if lv["regimes"].is_null() {
            bail!("golden has no block for layer {l}");
        }
        ok &= run_suite(
            gpu,
            Glm5NextKdaLayer::new(l, cfg, w, kernels)?,
            &ws,
            dm,
            cfg,
            &host,
            Some(&lv),
            &format!("layer{l}"),
        )?;
    }

    println!(
        "\n  family instantiated: {} bound KDA blocks share one workspace and one kernel set",
        family.len()
    );
    println!(
        "\n{}",
        if ok {
            "RESULT: PASS — the reusable KDA layer executes every tested block at the bf16 floor"
        } else {
            "RESULT: FAIL"
        }
    );
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
