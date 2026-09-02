// SPDX-License-Identifier: AGPL-3.0-only
//! Prefill-MoE DETERMINISM leg: the same prefill batch run N times must land
//! on a BIT-IDENTICAL fp32 routed accumulator.
//!
//! This is the regression gate for the defect
//! `ops::exl3_matmul::moe_prefill_det` fixes. Upstream's fused `exl3_moe`
//! epilogue atomicAdds each of a token's top_k expert rows into ONE shared
//! fp32 accumulator row, with the expert-to-group assignment drawn from an
//! in-kernel DYNAMIC ticket scheduler — so the commit order, and with it the
//! non-associative fp32 sum, differs run to run. On the serving model that
//! made 7 of 8 identical temp-0 prompts produce different prompt logprobs.
//!
//! Both arms are exercised from ONE set of slabs, differing only in
//! `Exl3MoePrefillScratch::slot_f32`:
//!
//!  * DETERMINISTIC (`Some`, the serving default): GATED — 8 identical runs
//!    must give one distinct accumulator. This is what fails if the fix is
//!    reverted or mis-ordered.
//!  * ATOMIC (`None`, the `--deterministic-moe-prefill false` kill switch):
//!    the NEGATIVE CONTROL. The same 8 runs are expected to produce more than
//!    one distinct accumulator; if they do not, the shape stopped racing and
//!    the gate above has become vacuous, so it is reported as a failure
//!    rather than a pass (the EP control in `legs_moe_prefill` precedent).
//!
//! The accumulator, not the BF16 output, is compared on purpose: the egress
//! cast swallows most 1-ulp reorder deltas, which is exactly why the
//! end-to-end symptom looked stochastic and length-dependent.
//!
//! Shape: the skewed T=192 batch — ~60% of slots on expert 0 (> 128 rows), so
//! the OVERFLOW tier's epilogue is under the same gate as the fused kernel's,
//! and enough experts stay under 128 rows to keep several expert groups
//! running concurrently. Both arms are also gated against the f64 reference,
//! so "deterministic" cannot be bought with a wrong answer.

use anyhow::Result;
use spark_model::layers::ops::{Exl3MoeOverflowCtx, Exl3MoePrefillScratch};

use crate::legs_moe::ProjSet;
use crate::legs_moe_prefill::{H, I, Slabs, TOP_K, alloc_slabs, gen_inputs, ref_all, run_native};
use crate::util::{Ctx, Lcg, gate_leg};

/// Experts of the synthetic layer (matches `legs_moe_prefill`).
const E: usize = 16;
/// Tokens in the racing batch — `legs_moe_prefill::T_MAX`.
const T: usize = 192;
/// Repeats per arm. 8 is the count the serving-side echo probe needed to see
/// the race every time it was run.
const REPS: usize = 8;

/// One arm's accumulator bytes for each repeat.
fn run_reps(
    ctx: &Ctx,
    sl: &Slabs,
    tables: &[spark_model::layers::ops::Exl3MoeProj; 3],
    ov: &Exl3MoeOverflowCtx,
    input_bf16: &[u16],
    ids: &[u32],
    probs: &[f32],
) -> Result<(Vec<Vec<u8>>, Vec<f64>, usize, f64)> {
    let mut accs = Vec::with_capacity(REPS);
    let mut last = Vec::new();
    let mut n_ov = 0usize;
    // One warmup outside the clock (first-touch of the slabs, module load).
    run_native(ctx, sl, tables, ov, input_bf16, ids, probs, T, 0, E)?;
    let t0 = std::time::Instant::now();
    for _ in 0..REPS {
        let (y, _, _, ov_n) = run_native(ctx, sl, tables, ov, input_bf16, ids, probs, T, 0, E)?;
        let mut bytes = vec![0u8; T * H * 4];
        ctx.g.copy_d2h(sl.scratch.out_f32, &mut bytes)?;
        accs.push(bytes);
        last = y;
        n_ov = ov_n;
    }
    // Wall time per batch INCLUDING this harness's H2D staging, sort and D2H
    // readback — an arm-vs-arm ratio, not a serving TTFT number.
    let ms = t0.elapsed().as_secs_f64() * 1e3 / REPS as f64;
    Ok((accs, last, n_ov, ms))
}

fn distinct(accs: &[Vec<u8>]) -> usize {
    let mut uniq: Vec<&Vec<u8>> = Vec::new();
    for a in accs {
        if !uniq.iter().any(|u| **u == *a) {
            uniq.push(a);
        }
    }
    uniq.len()
}

pub fn leg_moe_prefill_determinism(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let g = ctx.g;
    let mut ok = true;

    let gate = ProjSet::generate(ctx, rng, E, H, I)?;
    let upp = ProjSet::generate(ctx, rng, E, H, I)?;
    let down = ProjSet::generate(ctx, rng, E, I, H)?;
    let (gate_t, gate_own) = gate.table(ctx, 0)?;
    let (up_t, up_own) = upp.table(ctx, 0)?;
    let (down_t, down_own) = down.table(ctx, 0)?;
    let tables = [gate_t, up_t, down_t];
    let host = |p: &ProjSet| -> Vec<[u64; 3]> {
        p.dev
            .iter()
            .map(|d| [d.trellis.0, d.suh.0, d.svh.0])
            .collect()
    };
    let (gh, uh, dh) = (host(&gate), host(&upp), host(&down));
    let ov = Exl3MoeOverflowCtx {
        gate_host: &gh,
        up_host: &uh,
        down_host: &dh,
    };

    let det = alloc_slabs(ctx)?;
    // The kill-switch arm over the SAME device buffers: only the epilogue
    // selection differs, so a difference between the arms cannot come from
    // different memory, inputs or routing.
    let atomic = Slabs {
        scratch: Exl3MoePrefillScratch {
            slot_f32: None,
            slot_cap: 0,
            ..det.scratch
        },
        owned: Vec::new(),
        ..det
    };

    // Skewed routing: expert 0 takes ~60% of the slots (> 128 rows -> the
    // overflow tier), the rest spread over experts 1..E so several expert
    // groups race in the fused kernel.
    let (input_bf16, input_f16, probs) = gen_inputs(rng, T);
    let ids: Vec<u32> = (0..T * TOP_K)
        .map(|_| {
            if rng.f() < 0.6 {
                0u32
            } else {
                1 + (rng.next() % (E as u64 - 1)) as u32
            }
        })
        .collect();
    let y64 = ref_all(&input_f16, &ids, &probs, &gate, &upp, &down, T, 0, E);

    let (det_accs, det_y, det_ov, det_ms) =
        run_reps(ctx, &det, &tables, &ov, &input_bf16, &ids, &probs)?;
    let (atom_accs, atom_y, atom_ov, atom_ms) =
        run_reps(ctx, &atomic, &tables, &ov, &input_bf16, &ids, &probs)?;
    let (d_det, d_atom) = (distinct(&det_accs), distinct(&atom_accs));

    // Neither arm may buy stability with a wrong answer.
    ok &= gate_leg(
        &format!("moe-prefill DET epilogue T={T} skew (overflow experts={det_ov})"),
        &det_y,
        &y64,
        crate::legs_moe_prefill::MOE_REL_RMS,
        crate::legs_moe_prefill::MOE_MAX_Z,
    );
    ok &= gate_leg(
        &format!("moe-prefill ATOMIC epilogue T={T} skew (overflow experts={atom_ov})"),
        &atom_y,
        &y64,
        crate::legs_moe_prefill::MOE_REL_RMS,
        crate::legs_moe_prefill::MOE_MAX_Z,
    );

    // Cost of the ordering, reported not gated: this harness's per-batch wall
    // time carries its own staging/sort/readback, so read the RATIO only.
    println!(
        "moe-prefill epilogue cost T={T} skew: det {det_ms:.3} ms/batch vs \
         atomic {atom_ms:.3} ms/batch ({:+.1}%)",
        (det_ms / atom_ms - 1.0) * 100.0
    );

    // THE GATE.
    let bit_identical = d_det == 1;
    println!(
        "moe-prefill DETERMINISM: {REPS} identical batches -> {d_det} distinct \
         fp32 accumulators (want 1) = {bit_identical}"
    );
    ok &= bit_identical;

    // THE CONTROL: the same batches on upstream's atomic epilogue must race.
    println!(
        "moe-prefill CONTROL (--deterministic-moe-prefill false): {REPS} \
         identical batches -> {d_atom} distinct fp32 accumulators (want > 1)"
    );
    if d_atom <= 1 {
        println!(
            "FAIL — the atomic control did not race at this shape; the \
             determinism gate above is VACUOUS."
        );
        ok = false;
    }

    for p in det.owned.iter() {
        g.free(*p).ok();
    }
    for p in gate_own.into_iter().chain(up_own).chain(down_own) {
        g.free(p).ok();
    }
    gate.free(g);
    upp.free(g);
    down.free(g);
    Ok(ok)
}
