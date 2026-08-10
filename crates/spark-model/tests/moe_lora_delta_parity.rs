// SPDX-License-Identifier: AGPL-3.0-only
//
//! MoE LoRA delta-parity oracle (PR #335 merge gate — the Rust half of
//! `scripts/moe_lora_oracle.py`; contract: docs/design/lora-moe-embed.md §E
//! "Only after an oracle passes do we build the fused/S-LoRA kernels").
//!
//! Tiny dims (hidden=32, moe_inter=16, experts=4, r=4 padded to max_rank=8,
//! top_k=2, 6 tokens), random BF16 A/B, base output pre-filled with random
//! BF16. Each check drives the SAME entry point the production fold uses and
//! asserts `adapted - base == scale·(B@(A@x))` against a host reference that
//! reproduces the kernels' exact BF16 boundaries (bf16 xa → bf16 delta →
//! fp32 `base + scale·delta` → bf16), within 1 bf16 ULP (`cmp_tol`):
//!
//!   CHECK 1 — router fold: `lora::apply_router_lora` (prefill logits delta,
//!             `apply_lora_delta` GEMM path) — scale = alpha/r plumbing.
//!   CHECK 2 — grouped prefill down fold: `ops::moe_lora_grouped_down`
//!             (x_gather=0, sorted rows, device `expert_offsets`), including
//!             an UNADAPTED expert (NULL table cell folds nothing).
//!   CHECK 3 — chunked windows bit-identical to the single-window fold.
//!   CHECK 4 — `moe_row_adapter` base-row skip: skipped token's rows
//!             bit-identical to base, other rows bit-identical to CHECK 2.
//!   CHECK 5 — grouped gate/up fold (x_gather=1, token-major x gathered via
//!             `sorted_token_ids`; n_experts table shorter than the layer).
//!   CHECK 6 — decode gather fold: `ops::moe_lora_gather_bgmv` keyed on
//!             `indices_dev`, with a per-row base skip.
//!   CHECK 7 — phase-1 host-loop entry `lora::apply_expert_lora_sorted`
//!             matches CHECK 2's device fold bit-for-bit (same contract).
//!
//! `#[ignore]`d: requires a GB10 GPU + the compiled kernel set. CI builds it
//! against the libcuda stubs (signature drift guard) but never runs it:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*' \
//!     cargo test -p spark-model --test moe_lora_delta_parity -- --ignored --nocapture

use anyhow::Result;
use spark_model::layers::ops;
use spark_model::layers::ops::lora_delta::{LoraKernels, LoraPair};
use spark_model::layers::ops::moe_lora_grouped::{MoeExpertRoute, pack_expert_tables};
use spark_model::lora::{ExpertLoraLayer, ExpertProj, apply_expert_lora_sorted, apply_router_lora};
use spark_model::weight_map::DenseWeight;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[path = "arm2_common/support.rs"]
mod support;
use support::{Rng, bf16_bits_to_f32, cmp_tol, f32_to_bf16_bits, rd_u16, setup, up_i32, up_u16};

const H: usize = 32; // hidden
const INTER: usize = 16; // moe_intermediate_size
const E: usize = 4; // num_experts
const R: usize = 4; // true adapter rank
const MAX_RANK: usize = 8; // padded pool rank (pad rows/cols MUST be inert)
const TOP_K: usize = 2;
const T: usize = 6; // tokens
const TE: usize = T * TOP_K; // total expanded rows
const ALPHA: f32 = 3.0; // scale = alpha/r = 0.75 (non-trivial, catches r-vs-rank slips)
const SCALE: f32 = ALPHA / R as f32;
const SEED: u64 = 0x_10AA_D317_0335_0001;

/// One padded pair (loader layout, loading.rs:86-102): A `[MAX_RANK, k_in]`
/// real rows at the head + zero pad rows; B `[n_out, MAX_RANK]` row stride
/// MAX_RANK, zero pad cols.
struct HostPair {
    a: Vec<u16>,
    b: Vec<u16>,
    k_in: usize,
    n_out: usize,
}

fn gen_pair(rng: &mut Rng, k_in: usize, n_out: usize) -> HostPair {
    let mut a = vec![0u16; MAX_RANK * k_in];
    for j in 0..R {
        for k in 0..k_in {
            a[j * k_in + k] = f32_to_bf16_bits(rng.unit() * 2.0 - 1.0);
        }
    }
    let mut b = vec![0u16; n_out * MAX_RANK];
    for n in 0..n_out {
        for j in 0..R {
            b[n * MAX_RANK + j] = f32_to_bf16_bits(rng.unit() * 2.0 - 1.0);
        }
    }
    HostPair { a, b, k_in, n_out }
}

fn gen_bf16(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0))
        .collect()
}

/// The oracle: scale·(B@(A@x)) with the kernels' exact rounding boundaries —
/// bf16 xa (shrink store), bf16 delta (expand), scale applied in fp32 AFTER
/// the delta rounding (moe_lora_grouped_down.cu / lora_delta.rs:259-268).
fn host_fold_row(p: &HostPair, x: &[u16], base: &mut [u16]) {
    let mut xa = [0u16; MAX_RANK];
    for (j, slot) in xa.iter_mut().enumerate() {
        let mut acc = 0f32;
        for k in 0..p.k_in {
            acc += bf16_bits_to_f32(x[k]) * bf16_bits_to_f32(p.a[j * p.k_in + k]);
        }
        *slot = f32_to_bf16_bits(acc);
    }
    for (n, out) in base.iter_mut().enumerate() {
        let mut acc = 0f32;
        for j in 0..MAX_RANK {
            acc += bf16_bits_to_f32(xa[j]) * bf16_bits_to_f32(p.b[n * MAX_RANK + j]);
        }
        let delta = bf16_bits_to_f32(f32_to_bf16_bits(acc));
        *out = f32_to_bf16_bits(bf16_bits_to_f32(*out) + SCALE * delta);
    }
}

fn assert_tol(label: &str, got: &[u16], want: &[u16]) {
    let (pass, exact, max_ulp, worst) = cmp_tol(got, want);
    println!(
        "  {label}: exact {exact}/{} max_ulp {max_ulp} (worst idx {worst})",
        got.len()
    );
    assert!(
        pass,
        "{label}: kernel diverged from the delta oracle by {max_ulp} bf16 ULP"
    );
}

fn to_pair(hp: &HostPair, gpu: &dyn GpuBackend) -> Result<(LoraPair, DevicePtr, DevicePtr)> {
    let a = up_u16(gpu, &hp.a)?;
    let b = up_u16(gpu, &hp.b)?;
    Ok((
        LoraPair {
            a: DenseWeight { weight: a },
            b: DenseWeight { weight: b },
            rank: R as u32,
            k_in: hp.k_in as u32,
            n_out: hp.n_out as u32,
            scale: SCALE,
            max_rank: MAX_RANK as u32,
        },
        a,
        b,
    ))
}

fn route_of(pairs: &[(u16, LoraPair)], gpu: &dyn GpuBackend) -> Result<MoeExpertRoute> {
    let entries: Vec<(u16, u64, u64, f32)> = pairs
        .iter()
        .map(|(e, p)| (*e, p.a.weight.0, p.b.weight.0, p.scale))
        .collect();
    let t = pack_expert_tables(&entries).expect("non-empty");
    let up64 = |v: &[u64]| -> Result<DevicePtr> {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let d = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, d)?;
        Ok(d)
    };
    let sbytes: Vec<u8> = t.scale.iter().flat_map(|s| s.to_le_bytes()).collect();
    let sdev = gpu.alloc(sbytes.len())?;
    gpu.copy_h2d(&sbytes, sdev)?;
    let sample = &pairs[0].1;
    Ok(MoeExpertRoute {
        a_table: up64(&t.a)?,
        b_table: up64(&t.b)?,
        scale_table: sdev,
        n_experts: t.n_experts,
        k_in: sample.k_in,
        n_out: sample.n_out,
        max_rank: sample.max_rank,
    })
}

// ONE #[test]: the CUDA context is current only on the initializing thread
// (same constraint as tests/arm2_leg2_*.rs).
#[test]
#[ignore = "requires a GB10 GPU + compiled kernel set (CI links libcuda stubs only)"]
fn moe_lora_delta_parity() -> Result<()> {
    let (backend, st) = setup()?;
    let gpu: &dyn GpuBackend = &backend;
    let kernels = LoraKernels::new(gpu)?;
    let mut rng = Rng(SEED);

    // Routing fixture: token t -> experts (t%E, (t+1)%E), sorted-by-expert rows.
    let tok_experts: Vec<[usize; 2]> = (0..T).map(|t| [t % E, (t + 1) % E]).collect();
    let mut sorted_token_ids: Vec<i32> = Vec::new();
    let mut expert_offsets: Vec<i32> = vec![0];
    for e in 0..E {
        for (t, xs) in tok_experts.iter().enumerate() {
            if xs.contains(&e) {
                sorted_token_ids.push(t as i32);
            }
        }
        expert_offsets.push(sorted_token_ids.len() as i32);
    }
    assert_eq!(sorted_token_ids.len(), TE);
    let offs_dev = up_i32(gpu, &expert_offsets)?;
    let stid_dev = up_i32(gpu, &sorted_token_ids)?;
    let expert_of_row = |r: usize| {
        (0..E)
            .find(|&e| r < expert_offsets[e + 1] as usize)
            .unwrap()
    };

    // Scratch (mirrors MoeLoraWeights: xa [cap, max_rank], delta [cap, cols]).
    let xa_dev = gpu.alloc(TE * MAX_RANK * 2)?;
    let delta_dev = gpu.alloc(TE * H.max(E) * 2)?;

    // ── CHECK 1: router fold (apply_router_lora → apply_lora_delta) ────────
    let router = gen_pair(&mut rng, H, E);
    let (router_pair, ..) = to_pair(&router, gpu)?;
    let x_tok = gen_bf16(&mut rng, T * H); // token-major hidden states
    let x_tok_dev = up_u16(gpu, &x_tok)?;
    let logits = gen_bf16(&mut rng, T * E);
    let logits_dev = up_u16(gpu, &logits)?;
    apply_router_lora(
        gpu,
        &kernels,
        &router_pair,
        x_tok_dev,
        logits_dev,
        T as u32,
        TE as u32,
        xa_dev,
        delta_dev,
        st,
    )?;
    gpu.synchronize(st)?;
    let mut want = logits.clone();
    for t in 0..T {
        host_fold_row(
            &router,
            &x_tok[t * H..(t + 1) * H],
            &mut want[t * E..(t + 1) * E],
        );
    }
    assert_tol("CHECK 1 router", &rd_u16(gpu, logits_dev, T * E)?, &want);

    // ── CHECK 2: grouped prefill down fold (experts {0,2,3}; 1 unadapted) ──
    let down: Vec<(u16, HostPair)> = [0u16, 2, 3]
        .iter()
        .map(|&e| (e, gen_pair(&mut rng, INTER, H)))
        .collect();
    let down_pairs: Vec<(u16, LoraPair)> = down
        .iter()
        .map(|(e, hp)| Ok((*e, to_pair(hp, gpu)?.0)))
        .collect::<Result<_>>()?;
    let down_route = route_of(&down_pairs, gpu)?;
    assert_eq!(down_route.n_experts, E as u32); // max id 3 → table len 4, cell 1 NULL
    let x_sorted = gen_bf16(&mut rng, TE * INTER); // post-SiLU sorted activations
    let x_sorted_dev = up_u16(gpu, &x_sorted)?;
    let base_down = gen_bf16(&mut rng, TE * H);
    let out_dev = up_u16(gpu, &base_down)?;
    ops::moe_lora_grouped_down(
        gpu,
        &kernels,
        &down_route,
        x_sorted_dev,
        out_dev,
        offs_dev,
        stid_dev,
        DevicePtr::NULL,
        xa_dev,
        0,
        TE as u32,
        0,
        st,
    )?;
    gpu.synchronize(st)?;
    let mut want_down = base_down.clone();
    for r in 0..TE {
        if let Some((_, hp)) = down.iter().find(|(e, _)| *e as usize == expert_of_row(r)) {
            host_fold_row(
                hp,
                &x_sorted[r * INTER..(r + 1) * INTER],
                &mut want_down[r * H..(r + 1) * H],
            );
        }
    }
    let got_down = rd_u16(gpu, out_dev, TE * H)?;
    assert_tol("CHECK 2 grouped down", &got_down, &want_down);

    // ── CHECK 3: chunked windows ≡ single window (bit-identical) ───────────
    let out2_dev = up_u16(gpu, &base_down)?;
    for (lo, hi) in [(0u32, 7u32), (7, TE as u32)] {
        ops::moe_lora_grouped_down(
            gpu,
            &kernels,
            &down_route,
            x_sorted_dev,
            out2_dev,
            offs_dev,
            stid_dev,
            DevicePtr::NULL,
            xa_dev,
            lo,
            hi,
            0,
            st,
        )?;
    }
    gpu.synchronize(st)?;
    let got_chunked = rd_u16(gpu, out2_dev, TE * H)?;
    assert_eq!(
        got_chunked, got_down,
        "CHECK 3: chunked fold must be bit-identical"
    );
    println!("  CHECK 3 chunked windows: bit-identical");

    // ── CHECK 4: moe_row_adapter base-row skip (token 0 = base) ────────────
    let mut row_map = vec![0i32; T];
    row_map[0] = -1;
    let map_dev = up_i32(gpu, &row_map)?;
    let out3_dev = up_u16(gpu, &base_down)?;
    ops::moe_lora_grouped_down(
        gpu,
        &kernels,
        &down_route,
        x_sorted_dev,
        out3_dev,
        offs_dev,
        stid_dev,
        map_dev,
        xa_dev,
        0,
        TE as u32,
        0,
        st,
    )?;
    gpu.synchronize(st)?;
    let got_skip = rd_u16(gpu, out3_dev, TE * H)?;
    for r in 0..TE {
        let want_row = if sorted_token_ids[r] == 0 {
            &base_down
        } else {
            &got_down
        };
        assert_eq!(
            &got_skip[r * H..(r + 1) * H],
            &want_row[r * H..(r + 1) * H],
            "CHECK 4: row {r} (token {})",
            sorted_token_ids[r]
        );
    }
    println!("  CHECK 4 row-adapter skip: base rows untouched, adapted rows identical");

    // ── CHECK 5: gate/up fold (x_gather=1, table SHORTER than num_experts) ──
    let gate: Vec<(u16, HostPair)> = [0u16, 1]
        .iter()
        .map(|&e| (e, gen_pair(&mut rng, H, INTER)))
        .collect();
    let gate_pairs: Vec<(u16, LoraPair)> = gate
        .iter()
        .map(|(e, hp)| Ok((*e, to_pair(hp, gpu)?.0)))
        .collect::<Result<_>>()?;
    let gate_route = route_of(&gate_pairs, gpu)?;
    assert_eq!(gate_route.n_experts, 2); // experts 2/3 beyond the table → unadapted
    let base_gate = gen_bf16(&mut rng, TE * INTER);
    let gout_dev = up_u16(gpu, &base_gate)?;
    ops::moe_lora_grouped_down(
        gpu,
        &kernels,
        &gate_route,
        x_tok_dev,
        gout_dev,
        offs_dev,
        stid_dev,
        DevicePtr::NULL,
        xa_dev,
        0,
        TE as u32,
        1,
        st,
    )?;
    gpu.synchronize(st)?;
    let mut want_gate = base_gate.clone();
    for r in 0..TE {
        if let Some((_, hp)) = gate.iter().find(|(e, _)| *e as usize == expert_of_row(r)) {
            let t = sorted_token_ids[r] as usize; // token-major x gather
            host_fold_row(
                hp,
                &x_tok[t * H..(t + 1) * H],
                &mut want_gate[r * INTER..(r + 1) * INTER],
            );
        }
    }
    assert_tol(
        "CHECK 5 gate/up",
        &rd_u16(gpu, gout_dev, TE * INTER)?,
        &want_gate,
    );

    // ── CHECK 6: decode gather fold (indices_dev; token 5 = base skip) ──────
    let indices: Vec<u32> = tok_experts
        .iter()
        .flat_map(|xs| xs.iter().map(|&e| e as u32))
        .collect();
    let idx_dev = support::up_u32(gpu, &indices)?;
    let mut dec_map = vec![0i32; T];
    dec_map[5] = -1;
    let dec_map_dev = up_i32(gpu, &dec_map)?;
    let x_dec = gen_bf16(&mut rng, TE * INTER); // post-swiglu per flat slot
    let x_dec_dev = up_u16(gpu, &x_dec)?;
    let base_dec = gen_bf16(&mut rng, TE * H);
    let dout_dev = up_u16(gpu, &base_dec)?;
    ops::moe_lora_gather_bgmv(
        gpu,
        &kernels,
        &down_route,
        x_dec_dev,
        dout_dev,
        idx_dev,
        dec_map_dev,
        xa_dev,
        TE as u32,
        TOP_K as u32,
        0,
        st,
    )?;
    gpu.synchronize(st)?;
    let mut want_dec = base_dec.clone();
    for row in 0..TE {
        let e = indices[row] as usize;
        if dec_map[row / TOP_K] < 0 {
            continue; // base token: bit-identical
        }
        if let Some((_, hp)) = down.iter().find(|(de, _)| *de as usize == e) {
            host_fold_row(
                hp,
                &x_dec[row * INTER..(row + 1) * INTER],
                &mut want_dec[row * H..(row + 1) * H],
            );
        }
    }
    assert_tol(
        "CHECK 6 decode gather",
        &rd_u16(gpu, dout_dev, TE * H)?,
        &want_dec,
    );

    // ── CHECK 7: phase-1 host-loop entry ≡ device grouped fold ─────────────
    let mut layer = ExpertLoraLayer::default();
    for (e, p) in &down_pairs {
        layer.pairs.insert((*e, ExpertProj::Down), *p);
    }
    let out4_dev = up_u16(gpu, &base_down)?;
    let offs_host: Vec<u32> = expert_offsets.iter().map(|&v| v as u32).collect();
    apply_expert_lora_sorted(
        gpu,
        &kernels,
        &layer,
        ExpertProj::Down,
        &offs_host,
        x_sorted_dev,
        out4_dev,
        TE as u32,
        xa_dev,
        delta_dev,
        st,
    )?;
    gpu.synchronize(st)?;
    assert_tol(
        "CHECK 7 host-loop entry",
        &rd_u16(gpu, out4_dev, TE * H)?,
        &want_down,
    );

    println!("moe_lora_delta_parity: ALL CHECKS PASSED (scale={SCALE})");
    Ok(())
}
