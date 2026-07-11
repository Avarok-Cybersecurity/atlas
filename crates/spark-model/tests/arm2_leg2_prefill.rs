// SPDX-License-Identifier: AGPL-3.0-only
//
//! ARM-2 Phase-K Leg-2 — native-MXFP4 (E8M0) numeric gate, **Family B (prefill)**.
//!
//! Regression net for the E8M0 prefill W4A16 grouped/fused kernels (`moe_w4a16`):
//!   - CHECK 4 — 5 W4A16 entries, bit-exact `_e8m0` wrapper vs NVFP4-equivalent
//!     (same nibbles + power-of-2-equivalent scales -> bit-identical output).
//!   - CHECK 5 — real-shape MULTI-EXPERT integration (V4 routed dims). NO numeric
//!     assert: `compute-sanitizer memcheck` is the oracle. A clean run (host-side
//!     too) = the routed-GEMM path is memory-safe at real shapes. This is the
//!     gate the EP=2 CELL-BLOCKED illegal-address scar (2026-07-09) demands.
//!
//! `#[ignore]`d: requires a GB10 GPU. CI builds+links this against the libcuda
//! stubs (catches kernel-signature drift) but never runs it. On a GB10 host:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=deepseek-v4-flash ATLAS_TARGET_QUANT=nvfp4 \
//!     cargo test -p spark-model --test arm2_leg2_prefill -- --ignored --nocapture
//! For CHECK 5, wrap under: compute-sanitizer --tool memcheck --report-api-errors no <bin>.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

#[path = "arm2_common/support.rs"]
mod support;
use support::*;

const BMOD: &str = "moe_w4a16";
const SEED: u64 = 0x_ADA2_1E62_5EED_0002;

// ONE `#[test]` per binary: the CUDA context lives on the AtlasRegistry
// singleton and is current only on the thread that first initialized it. cargo
// runs each `#[test]` on its own thread, so a per-check backend init would break
// (only the first thread has a current context). Mirror the original single-
// `main` harness: init the backend once, run all checks on this thread.
// Run CHECK 5 under `compute-sanitizer --tool memcheck --report-api-errors no`.
#[test]
#[ignore] // Requires GB10 GPU (native-MXFP4 E8M0 prefill kernels); CHECK 5 wants memcheck
fn leg2_family_b_prefill() -> Result<()> {
    let (backend, st) = setup()?;
    let gpu: &dyn GpuBackend = &backend;
    check4_family_b_prefill_bit_exact(gpu, st)?;
    check5_multi_expert_memcheck(gpu, st)?;
    Ok(())
}

// ══════════ CHECK 4 — Family B prefill, 5 entries, bit-exact e8m0 vs NVFP4-equivalent ══════════
fn check4_family_b_prefill_bit_exact(gpu: &dyn GpuBackend, st: u64) -> Result<()> {
    let mut rng = Rng(SEED);
    let mut all_ok = true;

    println!("CHECK 4  Family B prefill — 5 W4A16 entries, e8m0 vs NVFP4-equivalent (bit-exact):");
    // entry: (label, base, e8m0, transposed, op, K set). K hits unroll boundaries + realistic.
    let grouped_entries: &[(&str, &str, &str, bool, GOp, &[usize])] = &[
        (
            "ptrtable(k16,non-t)",
            "moe_w4a16_grouped_gemm_ptrtable",
            "moe_w4a16_grouped_gemm_ptrtable_e8m0",
            false,
            GOp::Ptr64,
            &[64, 128, 448],
        ),
        (
            "ptrtable_t(k32)",
            "moe_w4a16_grouped_gemm_ptrtable_t",
            "moe_w4a16_grouped_gemm_ptrtable_t_e8m0",
            true,
            GOp::PtrN128,
            &[64, 128, 448],
        ),
        (
            "ptrtable_t_k64(down*)",
            "moe_w4a16_grouped_gemm_ptrtable_t_k64",
            "moe_w4a16_grouped_gemm_ptrtable_t_k64_e8m0",
            true,
            GOp::PtrK64N128,
            &[64, 128, 448],
        ),
    ];
    let (n_b, m_b) = (256usize, 128usize);
    for (label, base, e8m0, t, op, ks) in grouped_entries.iter().copied() {
        let kbase = gpu.kernel(BMOD, base)?;
        let ke8 = gpu.kernel(BMOD, e8m0)?;
        let mut ok = true;
        let mut detail = String::new();
        for &k in ks {
            let w = gen_wt_bitexact(&mut rng, k, n_b, t);
            let a: Vec<u16> = (0..m_b * k)
                .map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0))
                .collect();
            let a_p = up_u16(gpu, &a)?;
            let wp = up_u8(gpu, &w.packed)?;
            let ws_e = up_u8(gpu, &w.s_e8m0)?;
            let ws_n = up_u8(gpu, &w.s_nvfp4)?;
            let bpt = up_u64(gpu, &[wp.0])?;
            let bst_e = up_u64(gpu, &[ws_e.0])?;
            let bst_n = up_u64(gpu, &[ws_n.0])?;
            let s2 = up_f32(gpu, &[1.0])?;
            let off = up_i32(gpu, &[0, m_b as i32])?;
            let sti: Vec<i32> = (0..m_b as i32).collect();
            let sti_p = up_i32(gpu, &sti)?;
            let mt = (m_b as u32).div_ceil(64);
            let run = |kern: KernelHandle, bst: DevicePtr| -> Result<Vec<u16>> {
                let c = gpu.alloc(m_b * n_b * 2)?;
                launch_grouped(
                    gpu, op, kern, a_p, bpt, bst, s2, c, off, sti_p, 1, n_b as u32, k as u32, mt,
                    st,
                )?;
                gpu.synchronize(st)?;
                let v = rd_u16(gpu, c, m_b * n_b)?;
                gpu.free(c).ok();
                Ok(v)
            };
            let ce = run(ke8, bst_e)?;
            let cn = run(kbase, bst_n)?;
            let (p, d, _) = cmp_bits(&cn, &ce);
            ok &= p;
            detail.push_str(&format!(
                " K{k}:{}",
                if p { "ok".into() } else { format!("DIFF{d}") }
            ));
            for pp in [a_p, wp, ws_e, ws_n, bpt, bst_e, bst_n, s2, off, sti_p] {
                gpu.free(pp).ok();
            }
        }
        all_ok &= ok;
        println!(
            "   [{}] {} =>{}",
            if ok { "PASS" } else { "FAIL" },
            label,
            detail
        );
    }
    // fused entries (gate+up, 2 outputs).
    let fused_entries: &[(&str, &str, &str, FOp, &[usize])] = &[
        (
            "fused_gate_up_t(k32)",
            "moe_w4a16_fused_gate_up_t",
            "moe_w4a16_fused_gate_up_t_e8m0",
            FOp::FusedN128,
            &[64, 128, 448],
        ),
        (
            "fused_gate_up_t_k64(gate/up*)",
            "moe_w4a16_fused_gate_up_t_k64",
            "moe_w4a16_fused_gate_up_t_k64_e8m0",
            FOp::FusedK64N128,
            &[64, 128, 448],
        ),
    ];
    for (label, base, e8m0, op, ks) in fused_entries.iter().copied() {
        let kbase = gpu.kernel(BMOD, base)?;
        let ke8 = gpu.kernel(BMOD, e8m0)?;
        let mut ok = true;
        let mut detail = String::new();
        for &k in ks {
            let gw = gen_wt_bitexact(&mut rng, k, n_b, true);
            let uw = gen_wt_bitexact(&mut rng, k, n_b, true);
            let a: Vec<u16> = (0..m_b * k)
                .map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0))
                .collect();
            let a_p = up_u16(gpu, &a)?;
            let gwp = up_u8(gpu, &gw.packed)?;
            let gse = up_u8(gpu, &gw.s_e8m0)?;
            let gsn = up_u8(gpu, &gw.s_nvfp4)?;
            let uwp = up_u8(gpu, &uw.packed)?;
            let use_ = up_u8(gpu, &uw.s_e8m0)?;
            let usn = up_u8(gpu, &uw.s_nvfp4)?;
            let gpt = up_u64(gpu, &[gwp.0])?;
            let gse_t = up_u64(gpu, &[gse.0])?;
            let gsn_t = up_u64(gpu, &[gsn.0])?;
            let upt = up_u64(gpu, &[uwp.0])?;
            let use_t = up_u64(gpu, &[use_.0])?;
            let usn_t = up_u64(gpu, &[usn.0])?;
            let s2 = up_f32(gpu, &[1.0])?;
            let off = up_i32(gpu, &[0, m_b as i32])?;
            let sti: Vec<i32> = (0..m_b as i32).collect();
            let sti_p = up_i32(gpu, &sti)?;
            let mt = (m_b as u32).div_ceil(64);
            let run = |kern: KernelHandle,
                       gst: DevicePtr,
                       ust: DevicePtr|
             -> Result<(Vec<u16>, Vec<u16>)> {
                let cg = gpu.alloc(m_b * n_b * 2)?;
                let cu = gpu.alloc(m_b * n_b * 2)?;
                launch_fused(
                    gpu, op, kern, a_p, gpt, gst, s2, upt, ust, s2, cg, cu, off, sti_p, 1,
                    n_b as u32, k as u32, mt, st,
                )?;
                gpu.synchronize(st)?;
                let g = rd_u16(gpu, cg, m_b * n_b)?;
                let u = rd_u16(gpu, cu, m_b * n_b)?;
                gpu.free(cg).ok();
                gpu.free(cu).ok();
                Ok((g, u))
            };
            let (ge, ue) = run(ke8, gse_t, use_t)?;
            let (gn, un) = run(kbase, gsn_t, usn_t)?;
            let (pg, dg, _) = cmp_bits(&gn, &ge);
            let (pu, du, _) = cmp_bits(&un, &ue);
            ok &= pg && pu;
            detail.push_str(&format!(
                " K{k}:{}",
                if pg && pu {
                    "ok".into()
                } else {
                    format!("gDIFF{dg}/uDIFF{du}")
                }
            ));
            for pp in [
                a_p, gwp, gse, gsn, uwp, use_, usn, gpt, gse_t, gsn_t, upt, use_t, usn_t, s2, off,
                sti_p,
            ] {
                gpu.free(pp).ok();
            }
        }
        all_ok &= ok;
        println!(
            "   [{}] {} =>{}",
            if ok { "PASS" } else { "FAIL" },
            label,
            detail
        );
    }
    println!(
        "(*) = V4-serve-path entry (fused_gate_up_t_k64 gate/up + ptrtable_t_k64 down). Others off-path, tested for RIDER-2 completeness."
    );
    assert!(
        all_ok,
        "CHECK 4 FAIL: a Family-B entry diverged (see per-entry DIFF above)"
    );
    Ok(())
}

// ══════════ CHECK 5 — real-shape MULTI-EXPERT integration (memory-safety gate) ══════════
// CHECK 4 tiles are single-expert (private buffers, identity sorted_token_ids) and
// never exercise real expert_offsets / sorted_token_ids / per-expert ptr-table
// indexing at real shapes — exactly where a per-16-vs-per-32 stride slip or a
// gather OOB hides. This drives the EXACT V4-serve prefill routed path
// (fused_gate_up_t_k64_e8m0 → ptrtable_t_k64_e8m0 down) at real DeepSeek-V4-Flash
// routed dims with realistic prefill routing. NO numeric assert — a clean
// `compute-sanitizer memcheck` (host-side too) = the routed-GEMM path is
// memory-safe at real shapes (any remaining fault is upstream: routing/HC/attn/EP).
// Ceiling: covers the routed GEMM path only (silu skipped — addressing-neutral).
fn check5_multi_expert_memcheck(gpu: &dyn GpuBackend, st: u64) -> Result<()> {
    let mut rng = Rng(SEED);

    println!(
        "CHECK 5  real-shape multi-expert integration (V4 routed dims; compute-sanitizer memcheck is the oracle):"
    );
    let h = 4096usize; // dim (hidden)
    let inter = 2048usize; // moe_inter_dim
    let ne = 256usize; // n_routed_experts
    let top_k = 6usize; // n_activated_experts
    let t_tokens = 256usize;
    let total_expanded = t_tokens * top_k;

    // EXACT production e8m0 symbols (fused gate/up + ptrtable_t_k64 down).
    let k_gu = gpu.kernel(BMOD, "moe_w4a16_fused_gate_up_t_k64_e8m0")?;
    let k_dn = gpu.kernel(BMOD, "moe_w4a16_grouped_gemm_ptrtable_t_k64_e8m0")?;

    // Realistic routing: assign each (token,slot) an expert, sort by expert →
    // real expert_offsets (variable counts, empty experts) + sorted_token_ids.
    let mut rows: Vec<(u32, i32)> = Vec::with_capacity(total_expanded);
    for tok in 0..t_tokens {
        for _ in 0..top_k {
            rows.push(((rng.next_u64() % ne as u64) as u32, tok as i32));
        }
    }
    // Layer 1 is a hash-MoE layer (layer_idx < num_hash_layers=3): static tid2eid
    // routing is NOT load-balanced → piles tokens on few experts, so one expert
    // exceeds M_TILE=64 → mt>=2, exercising the multi-m-tile k_base loop that
    // balanced routing (mt=1) never hits. Force that imbalance.
    for r in rows.iter_mut().take(200) {
        r.0 = 7;
    }
    rows.sort_by_key(|r| r.0);
    let sti: Vec<i32> = rows.iter().map(|r| r.1).collect();
    let mut offs: Vec<i32> = vec![0i32; ne + 1];
    for &(e, _) in &rows {
        offs[e as usize + 1] += 1;
    }
    for e in 0..ne {
        offs[e + 1] += offs[e];
    }
    let max_per_expert = (0..ne).map(|e| offs[e + 1] - offs[e]).max().unwrap_or(0);
    let mt = (max_per_expert as u32).max(1).div_ceil(64);

    // Shared E8M0 weights (memcheck cares about addressing, not values):
    // ptr-tables point all `ne` experts at one gate/up [h,inter] + one down [inter,h].
    let gw = gen_wt_bitexact(&mut rng, h, inter, true);
    let uw = gen_wt_bitexact(&mut rng, h, inter, true);
    let dw = gen_wt_bitexact(&mut rng, inter, h, true);
    let gwp = up_u8(gpu, &gw.packed)?;
    let gws = up_u8(gpu, &gw.s_e8m0)?;
    let uwp = up_u8(gpu, &uw.packed)?;
    let uws = up_u8(gpu, &uw.s_e8m0)?;
    let dwp = up_u8(gpu, &dw.packed)?;
    let dws = up_u8(gpu, &dw.s_e8m0)?;
    let gpt = up_u64(gpu, &vec![gwp.0; ne])?;
    let gst = up_u64(gpu, &vec![gws.0; ne])?;
    let upt = up_u64(gpu, &vec![uwp.0; ne])?;
    let ust = up_u64(gpu, &vec![uws.0; ne])?;
    let dpt = up_u64(gpu, &vec![dwp.0; ne])?;
    let dst = up_u64(gpu, &vec![dws.0; ne])?;
    let s2 = up_f32(gpu, &vec![1.0f32; ne])?;

    let a: Vec<u16> = (0..t_tokens * h)
        .map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0))
        .collect();
    let a_p = up_u16(gpu, &a)?;
    let off_p = up_i32(gpu, &offs)?;
    let sti_p = up_i32(gpu, &sti)?;
    let gate_out = gpu.alloc(total_expanded * inter * 2)?;
    let up_out = gpu.alloc(total_expanded * inter * 2)?;
    let down_out = gpu.alloc(total_expanded * h * 2)?;

    // gate_up (real sorted_token_ids gather), then down (null sti = identity),
    // exactly as forward_prefill_routed.rs sequences them.
    launch_fused(
        gpu,
        FOp::FusedK64N128,
        k_gu,
        a_p,
        gpt,
        gst,
        s2,
        upt,
        ust,
        s2,
        gate_out,
        up_out,
        off_p,
        sti_p,
        ne as u32,
        inter as u32,
        h as u32,
        mt,
        st,
    )?;
    launch_grouped(
        gpu,
        GOp::PtrN128,
        k_dn,
        gate_out,
        dpt,
        dst,
        s2,
        down_out,
        off_p,
        DevicePtr(0),
        ne as u32,
        h as u32,
        inter as u32,
        mt,
        st,
    )?;
    gpu.synchronize(st)?;

    println!(
        "   [PASS] launched+synced gate_up_e8m0 + down_e8m0 over {ne} experts, {total_expanded} rows (max/expert={max_per_expert}, mt={mt}). Clean iff memcheck reports 0 errors."
    );
    for pp in [
        gwp, gws, uwp, uws, dwp, dws, gpt, gst, upt, ust, dpt, dst, s2, a_p, off_p, sti_p,
        gate_out, up_out, down_out,
    ] {
        gpu.free(pp).ok();
    }
    Ok(())
}
