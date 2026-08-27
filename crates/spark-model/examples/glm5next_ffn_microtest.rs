// SPDX-License-Identifier: AGPL-3.0-only
//! Slice 10 gates 3 + 4 — GLM-5.3-Flash FFN against HF `transformers` 5.16.1, real weights.
//!
//! **Gate 3** — the ModelOpt NVFP4 path, at PRODUCTION dims, on real layer-3 expert tensors.
//! "The kernel already works for another model" is explicitly not accepted as proof, so every
//! floor is measured here against a reference written from the OCP MX / ModelOpt spec rather
//! than from Atlas's own LUT.
//!
//! **Gate 4** — the dense FFN of layer 0 and the BF16 shared expert of layer 3.
//!
//! Floors, separated:
//!   **A** reference math — fp32 activations, dequantised fp32 weights. The trusted answer.
//!   **B** BF16 activation floor — |A(bf16 acts) − A(fp32 acts)|, from the golden itself.
//!   **C** quantisation/dequant floor — Atlas's CUDA `dequant_nvfp4_to_bf16` vs the independent
//!         Python ModelOpt dequant. 🪤 This is NOT "quantisation error": the pre-quantisation
//!         weights do not exist in this checkpoint, so the error of NVFP4 *as a representation*
//!         is unmeasurable here. What IS measurable — and what actually protects us — is that
//!         two independent implementations decode the same bits to the same numbers.
//!   **D** GPU kernel residual — Atlas `w4a16_gemm` vs A.
//!   **E** integrated FFN residual — the whole gate/up → clamp → silu·mul → down chain vs A.
//!
//!   MOE_PACKET_DIR=/home/msi1/atlas-scratch/moe-family \
//!   cargo run -p spark-model --release --example glm5next_ffn_microtest \
//!       --features cuda,gpu-examples

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use half::bf16;
use serde_json::Value;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

#[path = "common/golden.rs"]
mod golden;

static GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| golden::load("crates/spark-model/src/layers/glm5next_moe_ref/ffn_golden.json", "gen_ffn_golden.py"));
/// `(name, T, input_scale)`. 🪤 `clamp64` exists ONLY to make the swiglu clamp fire — at scale
/// 0.5 nothing reaches ±10 and a kernel with no clamp at all passes every other regime.
const REGIMES: [(&str, usize, f32); 4] = [
    ("t1", 1, 0.5),
    ("t7", 7, 0.5),
    ("t64", 64, 0.5),
    ("clamp64", 64, 8.0),
];

// ───────────────────────────────────────────────────────────────────── plumbing
fn up_bytes(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    up_bytes(g, &b)
}
fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

struct Lcg(u64);
impl Lcg {
    fn t(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((((self.0 >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0) as f32) * scale
            })
            .collect()
    }
}
fn input(t: usize, hid: usize, scale: f32) -> Vec<f32> {
    Lcg(0x0FFF_5EED).t(t * hid, scale)
}

// ───────────────────────────────────────────── golden
struct Golden(Value);
impl Golden {
    fn f(&self, k: &str) -> Result<f64> {
        self.0["fixture"][k]
            .as_f64()
            .with_context(|| format!("fixture.{k}"))
    }
    fn get(&self, sec: &str, name: &str) -> Result<(Vec<f32>, usize, usize, f64)> {
        let s = &self.0[sec][name];
        if s.is_null() {
            bail!("golden missing {sec}/{name}");
        }
        Ok((
            s["data"]
                .as_array()
                .context("data")?
                .iter()
                .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
                .collect(),
            s["stride"].as_u64().context("stride")? as usize,
            s["n"].as_u64().context("n")? as usize,
            s["ck"].as_f64().context("ck")?,
        ))
    }
}

/// Max abs difference against a strided golden row, with the element count checked first.
fn resid(what: &str, got: &[f32], g: &(Vec<f32>, usize, usize, f64)) -> Result<f32> {
    let (want, stride, n, _) = g;
    if got.len() != *n {
        bail!(
            "{what}: golden describes {n} elements, produced {}",
            got.len()
        );
    }
    let mut worst = 0.0f32;
    for (i, w) in want.iter().enumerate() {
        worst = worst.max((got[i * stride] - w).abs());
    }
    Ok(worst)
}
/// One BF16 ulp at magnitude `x`. BF16 keeps 8 significand bits, so ulp = 2^(exp-7).
fn bf16_ulp(x: f32) -> f32 {
    if x == 0.0 {
        return f32::MIN_POSITIVE;
    }
    let e = x.abs().log2().floor() as i32;
    (2.0f32).powi(e - 7)
}

/// The floor for a BF16-OUTPUT kernel compared against an FP32 reference.
///
/// Two irreducible terms, neither of which is a tolerance knob:
///   * the output is rounded to bf16, worth one ulp of `mag` on its own;
///   * the K-dimension sum (K = 4096 here) runs in a different order than torch's, and the
///     difference lands in the last bits before that rounding.
/// `ULP_BUDGET` bounds the second at a few ulp. A residual that needs MORE than this is not
/// representation noise and must not be waved through.
const ULP_BUDGET: f32 = 4.0;
fn bf16_output_floor(activation_floor: f32, mag: f32) -> f32 {
    activation_floor.max(ULP_BUDGET * bf16_ulp(mag))
}

fn magnitude(g: &(Vec<f32>, usize, usize, f64)) -> f32 {
    g.0.iter().fold(0.0f32, |a, b| a.max(b.abs()))
}

// ───────────────────────────────────────────── packet
struct Packet {
    raw: Vec<u8>,
    base: usize,
    hdr: BTreeMap<String, (String, Vec<usize>, usize, usize)>,
}
impl Packet {
    fn open(path: &str) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        let hn = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let j: Value = serde_json::from_slice(&raw[8..8 + hn])?;
        let mut hdr = BTreeMap::new();
        for (k, m) in j.as_object().context("hdr")? {
            if k == "__metadata__" {
                continue;
            }
            hdr.insert(
                k.clone(),
                (
                    m["dtype"].as_str().context("dtype")?.to_string(),
                    m["shape"]
                        .as_array()
                        .context("shape")?
                        .iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .collect(),
                    m["data_offsets"][0].as_u64().context("o0")? as usize,
                    m["data_offsets"][1].as_u64().context("o1")? as usize,
                ),
            );
        }
        Ok(Self {
            raw,
            base: 8 + hn,
            hdr,
        })
    }
    fn meta(&self, n: &str) -> Result<&(String, Vec<usize>, usize, usize)> {
        self.hdr.get(n).with_context(|| format!("missing {n}"))
    }
    fn bytes(&self, n: &str) -> Result<&[u8]> {
        let (_, _, a, b) = self.meta(n)?;
        Ok(&self.raw[self.base + a..self.base + b])
    }
    fn f32_scalar(&self, n: &str) -> Result<f32> {
        let (dt, ..) = self.meta(n)?;
        if dt != "F32" {
            bail!("{n}: expected F32, got {dt}");
        }
        let b = self.bytes(n)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

struct Row {
    gate: &'static str,
    what: String,
    regime: &'static str,
    floor: &'static str,
    e: f32,
    b: f32,
    mag: f32,
}

#[allow(clippy::too_many_arguments)]
fn w4a16(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    bp: DevicePtr,
    bs: DevicePtr,
    s2: f32,
    c: DevicePtr,
    m: u32,
    n: u32,
    kk: u32,
) -> Result<()> {
    // Geometry copied from `ops::w4a16_gemm`, not guessed: M_TILE = N_TILE = 64 and the CTA is
    // 128 threads (4 warps). Launching it with 256 threads reads past the shared-memory tiles
    // and faults — which is how this was found.
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(64), m.div_ceil(64), 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(bp)
        .arg_ptr(bs)
        .arg_f32(s2)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(kk)
        .launch(0)
}

fn gemm_bf16(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    kk: u32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(16), m.div_ceil(16), 1])
        .block([16, 16, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(kk)
        .launch(0)
}

fn swiglu(
    g: &dyn GpuBackend,
    k: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    n: u32,
    limit: f32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out)
        .arg_u32(n)
        .arg_f32(limit)
        .launch(0)
}

fn main() -> Result<()> {
    let dir = std::env::var("MOE_PACKET_DIR")
        .unwrap_or_else(|_| "/home/msi1/atlas-scratch/moe-family".to_string());
    let g = Golden(serde_json::from_str(GOLDEN)?);
    let hid = g.f("hidden")? as usize;
    let inter = g.f("intermediate")? as usize;
    let mi = g.f("moe_intermediate")? as usize;
    let si = g.f("shared_intermediate")? as usize;
    let limit = g.f("swiglu_limit")? as f32;
    let gs = g.f("group_size")? as usize;
    if g.0["fixture"]["has_input_scale"].as_bool() != Some(false) {
        bail!("fixture claims an input_scale exists; this gate is the WEIGHT-ONLY W4A16 path");
    }
    println!(
        "GLM FFN gate — hidden={hid} dense_inter={inter} moe_inter={mi} shared_inter={si} \
         swiglu_limit={limit} group_size={gs} input_scale=NONE (W4A16)"
    );

    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let k_w4a16 = gpu.kernel("w4a16", "w4a16_gemm")?;
    let k_deq = gpu.kernel("dequant_nvfp4_bf16", "dequant_nvfp4_to_bf16")?;
    let k_gemm = gpu.kernel("gemm", "dense_gemm_bf16")?;
    let k_act = gpu.kernel("glm5next_ffn", "glm5next_swiglu_clamp")?;
    let mut rows: Vec<Row> = Vec::new();

    // ───────────────────────── gate 4: dense FFN, layer 0 (BF16) ─────────────────────────
    {
        let p = Packet::open(&format!("{dir}/dense_layer0.safetensors"))?;
        for (n, want) in [
            ("gate_proj.weight", vec![inter, hid]),
            ("up_proj.weight", vec![inter, hid]),
            ("down_proj.weight", vec![hid, inter]),
        ] {
            let (dt, sh, ..) = p.meta(n)?;
            if dt != "BF16" || *sh != want {
                bail!("dense {n}: {dt} {sh:?}, expected BF16 {want:?}");
            }
        }
        let d_gate = up_bytes(&gpu, p.bytes("gate_proj.weight")?)?;
        let d_up = up_bytes(&gpu, p.bytes("up_proj.weight")?)?;
        let d_down = up_bytes(&gpu, p.bytes("down_proj.weight")?)?;
        for &(rn, t, sc) in REGIMES.iter() {
            let x = input(t, hid, sc);
            let d_x = up_bf16(&gpu, &x)?;
            let d_g = gpu.alloc(t * inter * 2)?;
            let d_u = gpu.alloc(t * inter * 2)?;
            let d_a = gpu.alloc(t * inter * 2)?;
            let d_o = gpu.alloc(t * hid * 2)?;
            gemm_bf16(
                &gpu,
                k_gemm,
                d_x,
                d_gate,
                d_g,
                t as u32,
                inter as u32,
                hid as u32,
            )?;
            gemm_bf16(
                &gpu,
                k_gemm,
                d_x,
                d_up,
                d_u,
                t as u32,
                inter as u32,
                hid as u32,
            )?;
            swiglu(&gpu, k_act, d_g, d_u, d_a, (t * inter) as u32, limit)?;
            gemm_bf16(
                &gpu,
                k_gemm,
                d_a,
                d_down,
                d_o,
                t as u32,
                hid as u32,
                inter as u32,
            )?;
            gpu.synchronize(0)?;
            let sec = format!("dense0__bf16__{rn}");
            let f32sec = format!("dense0__f32__{rn}");
            for (stage, ptr, n) in [
                ("gate_out", d_g, t * inter),
                ("act", d_a, t * inter),
                ("ffn_out", d_o, t * hid),
            ] {
                let got = down_bf16(&gpu, ptr, n)?;
                let want = g.get(&sec, stage)?;
                let wf = g.get(&f32sec, stage)?;
                let e = resid(stage, &got, &want)?;
                let b =
                    wf.0.iter()
                        .zip(&want.0)
                        .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
                rows.push(Row {
                    gate: "4",
                    what: "dense0".into(),
                    regime: rn,
                    floor: stage,
                    e,
                    b: bf16_output_floor(b, magnitude(&wf)),
                    mag: magnitude(&wf),
                });
            }
            // 🔴 NEGATIVE CONTROL for the clamp asymmetry. Recompute the activation on the
            // host from the SAME gate/up the GPU produced, but with the WRONG (symmetric) gate
            // clamp, and require Atlas to be far from it. Without this, a kernel that clamps
            // `gate` on both sides passes every positive check — the two agree exactly wherever
            // `gate > -limit`, which is everywhere at the default input scale.
            if rn == "clamp64" {
                let gate_v = down_bf16(&gpu, d_g, t * inter)?;
                let up_v = down_bf16(&gpu, d_u, t * inter)?;
                let act_v = down_bf16(&gpu, d_a, t * inter)?;
                let mut sym_max = 0.0f32; // WRONG: symmetric gate clamp
                let mut none_max = 0.0f32; // WRONG: no clamping at all
                let mut below = 0usize;
                let mut above = 0usize;
                let mut up_out = 0usize;
                for i in 0..t * inter {
                    if gate_v[i] < -limit {
                        below += 1;
                    }
                    if gate_v[i] > limit {
                        above += 1;
                    }
                    if up_v[i].abs() > limit {
                        up_out += 1;
                    }
                    let silu = |x: f32| x / (1.0 + (-x).exp());
                    let uu = up_v[i].clamp(-limit, limit);
                    sym_max =
                        sym_max.max((act_v[i] - silu(gate_v[i].clamp(-limit, limit)) * uu).abs());
                    none_max = none_max.max((act_v[i] - silu(gate_v[i]) * up_v[i]).abs());
                }
                if below == 0 || above == 0 || up_out == 0 {
                    bail!(
                        "clamp64 did not exercise the clamp: gate>+{limit} {above}, \
                         gate<-{limit} {below}, |up|>{limit} {up_out}"
                    );
                }
                println!(
                    "clamp control: gate>+{limit} {above} · gate<-{limit} {below} · \
                     |up|>{limit} {up_out}"
                );
                println!(
                    "  vs NO clamp at all       : {none_max:.4e}   <- must be large; proves the \
                     clamp fires"
                );
                println!(
                    "  vs SYMMETRIC gate clamp  : {sym_max:.4e}   <- 🪤 SMALL, and that is a \
                     FINDING: silu(g) -> 0 for g < -{limit}, so clamping `gate` from below is \
                     semantically wrong but numerically almost inert. Do not conclude from a \
                     passing FFN test that the asymmetry is implemented correctly."
                );
                // Only the first is a real assertion. The clamp must demonstrably do something.
                if none_max < bf16_ulp(magnitude(&g.get("dense0__f32__clamp64", "act")?)) * 100.0 {
                    bail!(
                        "removing the clamp entirely changed nothing ({none_max:e}) — the \
                           clamp is not being exercised"
                    );
                }
            }
            for q in [d_x, d_g, d_u, d_a, d_o] {
                gpu.free(q)?;
            }
        }
        for q in [d_gate, d_up, d_down] {
            gpu.free(q)?;
        }
    }

    // ───────────────────────── gate 3: NVFP4 experts, layer 3 ─────────────────────────
    let p = Packet::open(&format!("{dir}/moe_layer3_e0_3.safetensors"))?;
    for e in 0..4usize {
        let sec = format!("expert3_{e}");
        let mut wp: BTreeMap<&str, (DevicePtr, DevicePtr, f32, usize, usize)> = BTreeMap::new();
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let (n, k) = if proj == "down_proj" {
                (hid, mi)
            } else {
                (mi, hid)
            };
            let pn = format!("experts.{e}.{proj}.weight");
            let sn = format!("experts.{e}.{proj}.weight_scale");
            let gn = format!("experts.{e}.{proj}.weight_scale_2");
            let (dtp, shp, ..) = p.meta(&pn)?;
            let (dts, shs, ..) = p.meta(&sn)?;
            if dtp != "U8" || *shp != vec![n, k / 2] {
                bail!("{pn}: {dtp} {shp:?}, expected U8 [{n}, {}]", k / 2);
            }
            if dts != "F8_E4M3" || *shs != vec![n, k / gs] {
                bail!("{sn}: {dts} {shs:?}, expected F8_E4M3 [{n}, {}]", k / gs);
            }
            if p.hdr.keys().any(|x| x.contains("input_scale")) {
                bail!("an input_scale tensor appeared — this is not the W4A16 path");
            }
            wp.insert(
                proj,
                (
                    up_bytes(&gpu, p.bytes(&pn)?)?,
                    up_bytes(&gpu, p.bytes(&sn)?)?,
                    p.f32_scalar(&gn)?,
                    n,
                    k,
                ),
            );
        }

        // ── floor C: Atlas's CUDA dequant vs the independent ModelOpt reference ──
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let &(pp, sp, s2, n, k) = &wp[proj];
            let d_out = gpu.alloc(n * k * 2)?;
            KernelLaunch::new(&gpu, k_deq)
                .grid([n as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(pp)
                .arg_ptr(sp)
                .arg_ptr(d_out)
                .arg_f32(s2)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(0)?;
            gpu.synchronize(0)?;
            let got = down_bf16(&gpu, d_out, n * k)?;
            let want = g.get(&sec, &format!("deq_{proj}"))?;
            // 🔴 Floor C is a BIT-EXACTNESS test, not a tolerance. Atlas writes bf16, so round
            // the fp32 reference to bf16 FIRST and then demand equality: two independent decoders
            // of the same packed bits must produce the same numbers. Comparing an fp32 reference
            // against a bf16 result and calling the gap "the dequant floor" would hide a real
            // LUT / scale-order defect inside bf16 rounding — exactly the class of bug that
            // PR #341 and #347 were.
            let want_bf16: Vec<f32> = want.0.iter().map(|x| bf16::from_f32(*x).to_f32()).collect();
            let want_b = (want_bf16, want.1, want.2, want.3);
            let resid_c = resid(proj, &got, &want_b)?;
            let mismatches = want_b
                .0
                .iter()
                .enumerate()
                .filter(|(i, w)| got[*i * want_b.1] != **w)
                .count();
            rows.push(Row {
                gate: "3",
                what: format!("e{e}.{proj}"),
                regime: "dequant",
                floor: "C bit-exact",
                e: resid_c,
                // Zero. Any nonzero residual here is a decode disagreement.
                b: 0.0,
                mag: magnitude(&want),
            });
            if mismatches != 0 {
                bail!(
                    "expert {e} {proj}: Atlas's CUDA dequant disagrees with the ModelOpt \
                     reference on {mismatches} of {} sampled elements",
                    want_b.0.len()
                );
            }
            gpu.free(d_out)?;
        }

        for &(rn, t, sc) in REGIMES.iter() {
            let x = input(t, hid, sc);
            let d_x = up_bf16(&gpu, &x)?;
            let (gp, gsx, gs2, gn_, gk) = wp["gate_proj"];
            let (upp, ups, us2, un, uk) = wp["up_proj"];
            let (dp, dsx, ds2, dn, dk) = wp["down_proj"];
            let d_g = gpu.alloc(t * gn_ * 2)?;
            let d_u = gpu.alloc(t * un * 2)?;
            let d_a = gpu.alloc(t * gn_ * 2)?;
            let d_o = gpu.alloc(t * dn * 2)?;
            w4a16(
                &gpu, k_w4a16, d_x, gp, gsx, gs2, d_g, t as u32, gn_ as u32, gk as u32,
            )?;
            w4a16(
                &gpu, k_w4a16, d_x, upp, ups, us2, d_u, t as u32, un as u32, uk as u32,
            )?;
            swiglu(&gpu, k_act, d_g, d_u, d_a, (t * gn_) as u32, limit)?;
            w4a16(
                &gpu, k_w4a16, d_a, dp, dsx, ds2, d_o, t as u32, dn as u32, dk as u32,
            )?;
            gpu.synchronize(0)?;

            // D: the projection kernel alone, against reference math.
            let got_g = down_bf16(&gpu, d_g, t * gn_)?;
            let wa = g.get(&sec, &format!("{rn}__gate_f32"))?;
            let wb = g.get(&sec, &format!("{rn}__gate_bf16act"))?;
            let bfloor =
                wa.0.iter()
                    .zip(&wb.0)
                    .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
            rows.push(Row {
                gate: "3",
                what: format!("e{e}"),
                regime: rn,
                floor: "D gate_proj",
                e: resid("gate", &got_g, &wa)?,
                b: bf16_output_floor(bfloor, magnitude(&wa)),
                mag: magnitude(&wa),
            });
            // E: the whole expert FFN.
            let got_o = down_bf16(&gpu, d_o, t * dn)?;
            let oa = g.get(&sec, &format!("{rn}__out_f32"))?;
            let ob = g.get(&sec, &format!("{rn}__out_bf16act"))?;
            let ofloor =
                oa.0.iter()
                    .zip(&ob.0)
                    .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
            rows.push(Row {
                gate: "3",
                what: format!("e{e}"),
                regime: rn,
                floor: "E expert_ffn",
                e: resid("out", &got_o, &oa)?,
                b: bf16_output_floor(ofloor, magnitude(&oa)),
                mag: magnitude(&oa),
            });
            for q in [d_x, d_g, d_u, d_a, d_o] {
                gpu.free(q)?;
            }
        }
        for (_, (a, b, ..)) in wp {
            gpu.free(a)?;
            gpu.free(b)?;
        }
    }

    // ───────────────────────── gate 4: shared expert, layer 3 (BF16) ─────────────────────
    {
        for (n, want) in [
            ("shared_experts.gate_proj.weight", vec![si, hid]),
            ("shared_experts.up_proj.weight", vec![si, hid]),
            ("shared_experts.down_proj.weight", vec![hid, si]),
        ] {
            let (dt, sh, ..) = p.meta(n)?;
            if dt != "BF16" || *sh != want {
                bail!(
                    "shared {n}: {dt} {sh:?}, expected BF16 {want:?} — the shared expert is NOT quantised"
                );
            }
        }
        let d_gate = up_bytes(&gpu, p.bytes("shared_experts.gate_proj.weight")?)?;
        let d_up = up_bytes(&gpu, p.bytes("shared_experts.up_proj.weight")?)?;
        let d_down = up_bytes(&gpu, p.bytes("shared_experts.down_proj.weight")?)?;
        for &(rn, t, sc) in REGIMES.iter() {
            let x = input(t, hid, sc);
            let d_x = up_bf16(&gpu, &x)?;
            let d_g = gpu.alloc(t * si * 2)?;
            let d_u = gpu.alloc(t * si * 2)?;
            let d_a = gpu.alloc(t * si * 2)?;
            let d_o = gpu.alloc(t * hid * 2)?;
            gemm_bf16(
                &gpu, k_gemm, d_x, d_gate, d_g, t as u32, si as u32, hid as u32,
            )?;
            gemm_bf16(
                &gpu, k_gemm, d_x, d_up, d_u, t as u32, si as u32, hid as u32,
            )?;
            swiglu(&gpu, k_act, d_g, d_u, d_a, (t * si) as u32, limit)?;
            gemm_bf16(
                &gpu, k_gemm, d_a, d_down, d_o, t as u32, hid as u32, si as u32,
            )?;
            gpu.synchronize(0)?;
            let got = down_bf16(&gpu, d_o, t * hid)?;
            let wa = g.get("shared3", &format!("f32__{rn}__out"))?;
            let wb = g.get("shared3", &format!("bf16__{rn}__out"))?;
            let b =
                wa.0.iter()
                    .zip(&wb.0)
                    .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
            rows.push(Row {
                gate: "4",
                what: "shared3".into(),
                regime: rn,
                floor: "E shared_ffn",
                e: resid("shared", &got, &wb)?,
                b: bf16_output_floor(b, magnitude(&wb)),
                mag: magnitude(&wb),
            });
            for q in [d_x, d_g, d_u, d_a, d_o] {
                gpu.free(q)?;
            }
        }
        for q in [d_gate, d_up, d_down] {
            gpu.free(q)?;
        }
    }

    println!(
        "\n{:2} {:14} {:8} {:14} {:>11} {:>11} {:>11} {:>7}  verdict",
        "G", "what", "regime", "floor", "E", "B", "|ref|max", "E/B"
    );
    let mut fails = 0usize;
    for r in &rows {
        // A zero floor means the row is a BIT-EXACTNESS test (floor C): the criterion is
        // equality, not a ratio. Dividing by it would report a perfect result as infinite.
        let exact = r.b == 0.0;
        let ratio = if exact { 0.0 } else { r.e / r.b };
        let ok = if exact { r.e == 0.0 } else { ratio <= 1.0 };
        if !ok {
            fails += 1;
        }
        println!(
            "{:2} {:14} {:8} {:14} {:>11.4e} {:>11.4e} {:>11.4e} {:>7.3}  {}",
            r.gate,
            r.what,
            r.regime,
            r.floor,
            r.e,
            r.b,
            r.mag,
            ratio,
            if !ok {
                "ABOVE FLOOR"
            } else if exact {
                "BIT-EXACT"
            } else {
                "at floor"
            }
        );
    }
    println!("\n{} rows, {} above floor", rows.len(), fails);
    if fails > 0 {
        bail!("GLM FFN gate FAILED: {fails} stage(s) above floor");
    }
    println!("GLM FFN gate PASS (gates 3 + 4)");
    Ok(())
}
