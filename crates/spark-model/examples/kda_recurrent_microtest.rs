// SPDX-License-Identifier: AGPL-3.0-only
//! Numeric gate for the GLM-5.3-Flash KDA recurrent DECODE kernel
//! (`kda_recurrent::kda_recurrent_decode_f32` / `_bf16`) against HuggingFace 5.16.1.
//!
//! ## Why
//! Slice 4: the first *stateful* GLM5Next primitive. One token at a time, carrying
//! `state[H,D,D]`. It consumes the Slice-3 `kda_gate` output rather than recomputing the
//! gate, so a failure here is a recurrence failure and nothing else.
//!
//! ## The exp() reconciliation (do not remove this note)
//! `gate` is the **log-decay** — `lower_bound*sigmoid(...)` in `[lower_bound, 0]`. The
//! recurrence exponentiates it. All three sources agree:
//!   HF   `g_i = g[:, i][..., None].exp()`
//!   vLLM `b_state *= exp(b_gate[None, :])`
//!   CPU  `let decay = gate[base + kd].exp();`
//! Atlas's own GDN is the OPPOSITE on both axes — `compute_gdn_gates` stores `__expf(g)`
//! and `gated_delta_rule_decode` takes `exp(g_t)`, one scalar per head. Mixing the two
//! conventions is a silent double-exp or missing-exp.
//!
//! ## Oracles
//! 1. `kda_golden.json` — HF at H=2, D=4: `core_recurrent`, `state_recurrent`,
//!    `state_after_prefill4`, `core_split_prefill_then_decode`. Zero initial state.
//! 2. `kda_recurrent_prod_golden.json` — HF at **production** H=64, D=128, 3 decode steps
//!    from a **non-zero** carried state. Full `o` per step, a prime-strided state sample
//!    per step, and an fp64 index-weighted checksum over the *entire* state per step.
//! 3. `layers::glm5next_kda_ref::kda_recurrent` — the HF-bound CPU reference.
//!
//! ## Numeric floors are attributed, not chosen
//! Every production comparison reports the CPU-reference-vs-HF floor alongside the
//! GPU-vs-HF figure and gates on their ratio, so the kernel cannot hide inside libm spread.
//!
//!   cargo run -p spark-model --release --example kda_recurrent_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use half::bf16;
use serde_json::Value;
use spark_model::layers::glm5next_kda_ref::{KdaDims, kda_recurrent_prenorm};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const FIXTURE: &str = include_str!("../src/layers/glm5next_kda_ref/kda_golden.json");
#[path = "common/golden.rs"]
mod golden;

static PROD: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| golden::load("crates/spark-model/src/layers/glm5next_kda_ref/kda_recurrent_prod_golden.json", "gen_kda_recurrent_prod_golden.py"));

const PROD_H: usize = 64;
const PROD_D: usize = 128;
const BLOCK: u32 = 128;

/// Absolute bound. Production outputs are O(1e-2), so this is generous in relative terms;
/// the binding constraint is the floor ratio below, not this.
const MAX_ABS: f64 = 2.0e-6;
/// Relative bound over elements above the magnitude guard in `compare`.
const MAX_REL: f64 = 1.0e-4;
/// The kernel must not be materially worse than the CPU reference is against the same
/// golden. This is the claim that matters: residual is libm spread, not kernel error.
const MAX_FLOOR_RATIO: f64 = 2.0;

// ───────────────────────────────────────────────────────────────── plumbing

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn arr(v: &Value, section: &str, name: &str) -> Vec<f32> {
    v[section][name]["data"]
        .as_array()
        .unwrap_or_else(|| panic!("missing {section}.{name}"))
        .iter()
        .map(|x| x.as_f64().expect("numeric") as f32)
        .collect()
}

#[derive(Default, Clone)]
struct Err2 {
    max_abs: f64,
    max_rel: f64,
    exact: usize,
    total: usize,
}

/// Relative error is scored only over elements within three decades of the tensor's own
/// peak magnitude. A fixed absolute guard does not work here: the recurrent output is
/// O(1e-2) and its smallest elements are near zero, where any absolute error at all is a
/// huge relative one. Scoring those measures the float grid, not the kernel.
const REL_GUARD_FRACTION: f64 = 1e-3;

fn compare(got: &[f32], want: &[f32]) -> Err2 {
    assert_eq!(
        got.len(),
        want.len(),
        "length {} vs {}",
        got.len(),
        want.len()
    );
    let mut e = Err2 {
        total: got.len(),
        ..Default::default()
    };
    let peak = want.iter().fold(0.0f64, |m, w| m.max((*w as f64).abs()));
    let guard = (peak * REL_GUARD_FRACTION).max(1e-30);
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64 - *w as f64).abs();
        e.max_abs = e.max_abs.max(d);
        if (*w as f64).abs() > guard {
            e.max_rel = e.max_rel.max(d / (*w as f64).abs());
        }
        if g.to_bits() == w.to_bits() {
            e.exact += 1;
        }
    }
    e
}

fn report(label: &str, e: &Err2, dtype: &str) {
    println!(
        "  {label:<48} {dtype:<5} max_abs={:.3e} max_rel={:.3e} exact={}/{}",
        e.max_abs, e.max_rel, e.exact, e.total
    );
}

fn within(e: &Err2) -> bool {
    e.max_abs <= MAX_ABS && e.max_rel <= MAX_REL
}

fn ratio(e: &Err2, floor: &Err2) -> f64 {
    if floor.max_abs > 0.0 {
        e.max_abs / floor.max_abs
    } else if e.max_abs == 0.0 {
        1.0
    } else {
        f64::INFINITY
    }
}

/// fp64 index-weighted checksum over the full state — no indexing or ordering error
/// survives it, and it costs 8 bytes in the golden instead of 4 MiB.
fn checksum(s: &[f32]) -> f64 {
    s.iter()
        .enumerate()
        .map(|(i, v)| *v as f64 * (i as f64 + 1.0))
        .sum()
}

// ───────────────────────────────────────────────────────────────── launch

struct Rec<'a> {
    g: &'a dyn GpuBackend,
    k_f32: KernelHandle,
    k_bf16: KernelHandle,
}

impl Rec<'_> {
    /// One decode token. `state` is updated in place on the host side too.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &mut Vec<f32>,
        h: usize,
        d: usize,
        bf16_inputs: bool,
    ) -> Result<Vec<f32>> {
        let g = self.g;
        let (dq, dk, dv) = if bf16_inputs {
            (up_bf16(g, q)?, up_bf16(g, k)?, up_bf16(g, v)?)
        } else {
            (up_f32(g, q)?, up_f32(g, k)?, up_f32(g, v)?)
        };
        let dgate = up_f32(g, gate)?;
        let dbeta = up_f32(g, beta)?;
        let dstate = up_f32(g, state)?;
        let dout = g.alloc(h * d * 4)?;
        let scale = 1.0f32 / (d as f32).sqrt();

        KernelLaunch::new(g, if bf16_inputs { self.k_bf16 } else { self.k_f32 })
            .grid([h as u32, 1, 1])
            .block([BLOCK.min(d as u32), 1, 1])
            .shared_mem((3 * d * 4) as u32)
            .arg_ptr(dq)
            .arg_ptr(dk)
            .arg_ptr(dv)
            .arg_ptr(dgate)
            .arg_ptr(dbeta)
            .arg_ptr(dstate)
            .arg_ptr(dout)
            .arg_u32(h as u32)
            .arg_u32(d as u32)
            .arg_f32(scale)
            .launch(0)?;
        g.synchronize(0)?;

        *state = down_f32(g, dstate, h * d * d)?;
        down_f32(g, dout, h * d)
    }
}

// ───────────────────────────────────────────────────────────────── A + B (fixture)

/// A — single-token recurrence from ZERO state, and B — multi-step decode comparing the
/// output AND the full carried state after EVERY token, against the CPU reference (which
/// is HF-bound), with the final state and all outputs checked against HF directly.
fn check_fixture(rec: &Rec) -> Result<bool> {
    let v: Value = serde_json::from_str(FIXTURE)?;
    let f = &v["fixture"];
    let (h, d, t) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["tokens"].as_u64().unwrap() as usize,
    );
    let dims1 = KdaDims {
        hidden: 0,
        heads: h,
        head_dim: d,
        tokens: 1,
    };
    let (q, k, vv) = (
        arr(&v, "outputs", "q_l2"),
        arr(&v, "outputs", "k_l2"),
        arr(&v, "inputs", "v_in"),
    );
    let (gate, beta) = (arr(&v, "outputs", "gate"), arr(&v, "outputs", "beta"));
    let want_o = arr(&v, "outputs", "core_recurrent");
    let per = h * d;

    let mut state = vec![0.0f32; h * d * d];
    let mut cpu_state = vec![0.0f32; h * d * d];
    let mut all_o = Vec::new();
    let mut ok = true;

    for tok in 0..t {
        let (a, b) = (tok * per, (tok + 1) * per);
        let o = rec.step(
            &q[a..b],
            &k[a..b],
            &vv[a..b],
            &gate[a..b],
            &beta[tok * h..(tok + 1) * h],
            &mut state,
            h,
            d,
            false,
        )?;
        let cpu_o = kda_recurrent_prenorm(
            &q[a..b],
            &k[a..b],
            &vv[a..b],
            &gate[a..b],
            &beta[tok * h..(tok + 1) * h],
            dims1,
            &mut cpu_state,
        );
        let eo = compare(&o, &want_o[a..b]);
        let es = compare(&state, &cpu_state);
        let ec = compare(&o, &cpu_o);
        println!(
            "  token {tok}: o vs HF max_abs={:.3e}  o vs CPU-ref max_abs={:.3e}  state vs CPU-ref max_abs={:.3e}",
            eo.max_abs, ec.max_abs, es.max_abs
        );
        ok &= within(&eo) && within(&es);
        all_o.extend_from_slice(&o);
    }

    report(
        "fixture: all tokens o vs HF",
        &compare(&all_o, &want_o),
        "f32",
    );
    let ef = compare(&state, &arr(&v, "outputs", "state_recurrent"));
    report("fixture: final state vs HF", &ef, "f32");
    Ok(ok && within(&ef))
}

// ───────────────────────────────────────────────────────────────── C (continuation)

/// C — prefill(4) → decode(2), starting the GPU decode kernel from HF's OWN carried state
/// (`state_after_prefill4`), not from a reproduction of it.
fn check_continuation(rec: &Rec) -> Result<bool> {
    let v: Value = serde_json::from_str(FIXTURE)?;
    let f = &v["fixture"];
    let (h, d, t) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["tokens"].as_u64().unwrap() as usize,
    );
    let (q, k, vv) = (
        arr(&v, "outputs", "q_l2"),
        arr(&v, "outputs", "k_l2"),
        arr(&v, "inputs", "v_in"),
    );
    let (gate, beta) = (arr(&v, "outputs", "gate"), arr(&v, "outputs", "beta"));
    let split = 4usize;
    let per = h * d;

    let mut state = arr(&v, "outputs", "state_after_prefill4"); // HF-produced
    let want = arr(&v, "outputs", "core_split_prefill_then_decode");
    let mut ok = true;
    for tok in split..t {
        let (a, b) = (tok * per, (tok + 1) * per);
        let o = rec.step(
            &q[a..b],
            &k[a..b],
            &vv[a..b],
            &gate[a..b],
            &beta[tok * h..(tok + 1) * h],
            &mut state,
            h,
            d,
            false,
        )?;
        let e = compare(&o, &want[a..b]);
        report(&format!("continuation token {tok} vs HF"), &e, "f32");
        ok &= within(&e);
    }
    let ef = compare(&state, &arr(&v, "outputs", "state_recurrent"));
    report("continuation final state vs HF", &ef, "f32");
    Ok(ok && within(&ef))
}

// ───────────────────────────────────────────────────────── production geometry

/// Reproduces the Python generator's LCG bit for bit: 64-bit integer ops plus an exact
/// division by 2^24. No transcendental, no platform libm.
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.unit()).collect()
    }
}

/// A (non-zero carried state) + B (per-step output AND state) at H=64, D=128, against HF.
fn check_production(rec: &Rec, bf16_inputs: bool) -> Result<bool> {
    let v: Value = serde_json::from_str(PROD)?;
    let f = &v["fixture"];
    let (h, d, steps) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["steps"].as_u64().unwrap() as usize,
    );
    assert_eq!(
        (h, d),
        (PROD_H, PROD_D),
        "golden is not production geometry"
    );
    let stride = f["sample_stride"].as_u64().unwrap() as usize;
    let lb = f["lower_bound"].as_f64().unwrap() as f32;

    // The generator's LCG must be reproduced exactly or the whole fixture is different.
    let probe: Vec<f32> = v["lcg_probe"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let mut pr = Lcg(0x5EED_2EC0);
    let mine = pr.vec(probe.len());
    if mine
        .iter()
        .zip(&probe)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        println!("    ! LCG mismatch: rust {mine:?} vs python {probe:?}");
        return Ok(false);
    }
    println!(
        "  LCG reproduction: {} draws bit-identical to the generator",
        probe.len()
    );

    // Same draw order as the generator.
    let mut rng = Lcg(0x5EED_2EC0);
    let state0: Vec<f32> = rng.vec(h * d * d).iter().map(|x| x * 0.05).collect();
    let q_s: Vec<Vec<f32>> = (0..steps).map(|_| rng.vec(h * d)).collect();
    let k_s: Vec<Vec<f32>> = (0..steps).map(|_| rng.vec(h * d)).collect();
    let v_s: Vec<Vec<f32>> = (0..steps).map(|_| rng.vec(h * d)).collect();
    let g_s: Vec<Vec<f32>> = (0..steps)
        .map(|_| {
            rng.vec(h * d)
                .iter()
                .map(|x| lb * (1.0 / (1.0 + (-(x * 3.0)).exp())))
                .collect()
        })
        .collect();
    let b_s: Vec<Vec<f32>> = (0..steps)
        .map(|_| {
            rng.vec(h)
                .iter()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect()
        })
        .collect();

    let l2 = |x: &[f32]| -> Vec<f32> {
        x.chunks_exact(d)
            .flat_map(|r| {
                let inv = 1.0 / (r.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
                r.iter().map(move |a| a * inv)
            })
            .collect()
    };

    let dims1 = KdaDims {
        hidden: 0,
        heads: h,
        head_dim: d,
        tokens: 1,
    };
    let mut state = state0.clone();
    let mut cpu_state = state0.clone();
    let mut ok = true;
    let dtype = if bf16_inputs { "bf16" } else { "f32" };

    // The CPU reference must see EXACTLY the inputs the GPU sees, or the comparison mixes
    // input rounding into what is supposed to be kernel error.
    let as_seen = |x: &[f32]| -> Vec<f32> {
        if bf16_inputs {
            x.iter().map(|a| bf16::from_f32(*a).to_f32()).collect()
        } else {
            x.to_vec()
        }
    };

    for s in 0..steps {
        let (qn, kn) = (l2(&q_s[s]), l2(&k_s[s]));
        let o = rec.step(
            &qn,
            &kn,
            &v_s[s],
            &g_s[s],
            &b_s[s],
            &mut state,
            h,
            d,
            bf16_inputs,
        )?;
        let cpu_o = kda_recurrent_prenorm(
            &as_seen(&qn),
            &as_seen(&kn),
            &as_seen(&v_s[s]),
            &g_s[s],
            &b_s[s],
            dims1,
            &mut cpu_state,
        );

        let want_o = arr(&v, "outputs", &format!("o_step{s}"));
        let floor = compare(&cpu_o, &want_o);
        let e = compare(&o, &want_o);
        let kern = compare(&o, &cpu_o);
        report(
            &format!("prod step{s}: CPU-ref vs HF (floor, no GPU)"),
            &floor,
            dtype,
        );
        report(
            &format!("prod step{s}: GPU o vs HF (floor + kernel)"),
            &e,
            dtype,
        );
        report(
            &format!("prod step{s}: GPU o vs CPU-ref, SAME inputs (kernel only)"),
            &kern,
            dtype,
        );
        let state_kern = compare(&state, &cpu_state);
        report(
            &format!("prod step{s}: GPU state vs CPU-ref state (kernel only)"),
            &state_kern,
            dtype,
        );

        let want_samp = arr(&v, "outputs", &format!("state_sample{s}"));
        let samp: Vec<f32> = state.iter().step_by(stride).copied().collect();
        let es = compare(&samp, &want_samp);
        report(&format!("prod step{s}: state sample vs HF"), &es, dtype);

        let want_ck = v["state_checksums"][s].as_f64().unwrap();
        let got_ck = checksum(&state);
        let ck_rel = (got_ck - want_ck).abs() / want_ck.abs().max(1.0);
        println!(
            "  prod step{s}: full-state fp64 checksum rel_err={ck_rel:.3e}  (got {got_ck:.6e})"
        );

        let r = ratio(&e, &floor);
        println!("  prod step{s}: GPU/floor max_abs ratio = {r:.3}");
        if bf16_inputs {
            // Against the fp32 golden the bf16 path is dominated by INPUT ROUNDING, so that
            // figure is reported, not gated. What IS gated is kernel-only error: given the
            // same bf16-rounded inputs, the GPU must track the CPU reference to fp32 accuracy.
            ok &= within(&kern) && within(&state_kern);
        } else {
            ok &= within(&e) && within(&es) && r <= MAX_FLOOR_RATIO && ck_rel < 1e-6;
        }
    }
    Ok(ok)
}

// ───────────────────────────────────────────────────────────────── D (adversarial)

/// D — prove each semantic independently by breaking it and showing the result moves.
fn check_adversarial(rec: &Rec) -> Result<bool> {
    let (h, d) = (8usize, 32usize);
    let dims1 = KdaDims {
        hidden: 0,
        heads: h,
        head_dim: d,
        tokens: 1,
    };
    let mut rng = Lcg(0xD00D);
    let n = h * d;
    let l2 = |x: &[f32]| -> Vec<f32> {
        x.chunks_exact(d)
            .flat_map(|r| {
                let inv = 1.0 / (r.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
                r.iter().map(move |a| a * inv)
            })
            .collect()
    };
    let q = l2(&rng.vec(n));
    let k = l2(&rng.vec(n));
    let vv = rng.vec(n);
    let gate: Vec<f32> = rng
        .vec(n)
        .iter()
        .map(|x| -5.0 * (1.0 / (1.0 + (-(x * 3.0)).exp())))
        .collect();
    let beta: Vec<f32> = rng
        .vec(h)
        .iter()
        .map(|x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    let state0: Vec<f32> = rng.vec(h * d * d).iter().map(|x| x * 0.1).collect();

    let run = |gate: &[f32], beta: &[f32], st: &Vec<f32>| -> Result<(Vec<f32>, Vec<f32>)> {
        let mut s = st.clone();
        let o = rec.step(&q, &k, &vv, gate, beta, &mut s, h, d, false)?;
        Ok((o, s))
    };
    let (base_o, base_s) = run(&gate, &beta, &state0)?;
    let mut ok = true;
    let mut probe = |label: &str, o: &[f32], s: &[f32], need: bool| {
        let dv = compare(o, &base_o).max_abs.max(compare(s, &base_s).max_abs);
        let verdict = if (dv > 1e-4) == need { "ok" } else { "FAIL" };
        println!("  {label:<52} divergence={dv:.3e}  [{verdict}]");
        if (dv > 1e-4) != need {
            ok = false;
        }
    };

    // D1 — decay varies across the key channel D. Flatten it per head and the result moves.
    let flat_gate: Vec<f32> = (0..h)
        .flat_map(|hh| std::iter::repeat_n(gate[hh * d], d))
        .collect();
    let (o, s) = run(&flat_gate, &beta, &state0)?;
    probe(
        "D1 decay is per key-channel (vs per-head flat)",
        &o,
        &s,
        true,
    );

    // D2 — beta varies by head.
    let flat_beta = vec![beta[0]; h];
    let (o, s) = run(&gate, &flat_beta, &state0)?;
    probe("D2 beta is per head (vs uniform)", &o, &s, true);

    // D3 — the state update is k (x) delta, not delta (x) k. Transposing every head's state
    // block would be a no-op if the orientation were symmetric. It is not.
    let mut t_state = state0.clone();
    for hh in 0..h {
        for a in 0..d {
            for b in (a + 1)..d {
                t_state.swap(hh * d * d + a * d + b, hh * d * d + b * d + a);
            }
        }
    }
    let (o, s) = run(&gate, &beta, &t_state)?;
    probe(
        "D3 k (x) delta orientation (vs transposed state)",
        &o,
        &s,
        true,
    );

    // D4 — q is applied AFTER the state update. Compare against a reference that reads the
    // pre-update state; they must differ, and the kernel must match the post-update one.
    let mut ref_state = state0.clone();
    let ref_o = kda_recurrent_prenorm(&q, &k, &vv, &gate, &beta, dims1, &mut ref_state);
    let post = compare(&base_o, &ref_o);
    let mut pre_state = state0.clone();
    let pre_o = {
        // Same recurrence but reading the DECAYED-only state for the output.
        let mut o = vec![0.0f32; h * d];
        for hh in 0..h {
            let s = &mut pre_state[hh * d * d..(hh + 1) * d * d];
            for kk in 0..d {
                let dec = gate[hh * d + kk].exp();
                for vi in 0..d {
                    s[kk * d + vi] *= dec;
                }
            }
            let scale = 1.0f32 / (d as f32).sqrt();
            for vi in 0..d {
                let mut acc = 0.0f32;
                for kk in 0..d {
                    acc += s[kk * d + vi] * (q[hh * d + kk] * scale);
                }
                o[hh * d + vi] = acc;
            }
        }
        o
    };
    let pre = compare(&base_o, &pre_o);
    println!(
        "  D4 q applied AFTER update: vs post-update ref max_abs={:.3e}, vs pre-update ref max_abs={:.3e}",
        post.max_abs, pre.max_abs
    );
    if !(post.max_abs < 1e-6 && pre.max_abs > 1e-4) {
        println!(
            "    ! q ordering is not distinguishable, or does not match the post-update reference"
        );
        ok = false;
    }

    // D5 — 1/sqrt(D) placement. The kernel pre-scales q (HF's `query = query * scale`); the
    // CPU reference scales each term at the output. Algebraically identical, so the residual
    // here is pure rounding and must be tiny.
    println!(
        "  D5 scale placement (pre-scale q vs per-term at output) max_abs={:.3e}",
        post.max_abs
    );
    if post.max_abs > 1e-6 {
        println!("    ! scale placement is not numerically equivalent");
        ok = false;
    }
    Ok(ok)
}

// ───────────────────────────────────────────────────────────────── main

fn main() -> Result<()> {
    let g = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &g;
    let rec = Rec {
        g: gpu,
        k_f32: gpu.kernel("kda_recurrent", "kda_recurrent_decode_f32")?,
        k_bf16: gpu.kernel("kda_recurrent", "kda_recurrent_decode_bf16")?,
    };
    println!("kda_recurrent: both entry points resolved from PTX (no fallback path)\n");

    println!("A/B — fixture H=2 D=4, zero initial state, per-token output AND state");
    let a = check_fixture(&rec)?;

    println!("\nC — prefill(4) -> decode(2) from HF's own carried state");
    let c = check_continuation(&rec)?;

    println!("\nA/B — production H=64 D=128, NON-ZERO carried state, fp32 oracle path");
    let p32 = check_production(&rec, false)?;

    println!("\nproduction, bf16 production-input path");
    let pbf = check_production(&rec, true)?;

    println!("\nD — adversarial semantics");
    let d = check_adversarial(&rec)?;

    println!(
        "\ngrid = (H, 1, 1)  block = ({}, 1, 1)  shared = 3*D*4 bytes  one thread per V",
        BLOCK.min(PROD_D as u32)
    );
    let st = PROD_H * PROD_D * PROD_D * 4;
    println!(
        "state per layer = {} B ({:.2} MiB) fp32; traffic per token = 2 reads + 2 writes = {:.2} MiB",
        st,
        st as f64 / (1024.0 * 1024.0),
        4.0 * st as f64 / (1024.0 * 1024.0)
    );

    if a && c && p32 && pbf && d {
        println!("\nPASS");
        Ok(())
    } else {
        bail!("FAIL — see the lines marked ! above");
    }
}
