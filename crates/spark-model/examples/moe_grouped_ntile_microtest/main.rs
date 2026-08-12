// SPDX-License-Identifier: AGPL-3.0-only
//! Correctness gate for the MoE grouped-GEMM N-tile width in
//! `kernels/gb10/common/moe_w4a16_grouped_gemm.cu`.
//!
//! ## The defect this proves and fixes
//! `ops::moe_w4a16_grouped_gemm_ptrtable{,_n128}` launch
//! `moe_w4a16_grouped_gemm_ptrtable` / `..._ptrtable_t` with
//! `grid.x = ceil(N_out / 128)`, but those two entries used to own only
//! `N_TILE = 64` columns per CTA. The top of the N range was therefore never
//! written and stayed at the caller's memset zero. On Nemotron-H Lightning-30B
//! (the sorted grouped prefill path, live today) grouped UP `N_out=1856` wrote
//! 15x64 = 960 of 1856 columns and grouped DOWN `N_out=2688` wrote 21x64 =
//! 1344 of 2688 — the routed-expert contribution was roughly halved and
//! structurally masked. The fix widens ONLY those two entries to
//! `N_TILE_PT = 128`, matching every peer shadow of the file.
//!
//! ## Why bit-parity is the right gate
//! The N tile selects WHICH COLUMNS a CTA owns. It never changes the order in
//! which any single output element accumulates over `k_base` (same k loop,
//! same per-element `mma.m16n8k16` sequence), so the retuned kernel at
//! `grid.x = ceil(N/128)` must be BYTE-IDENTICAL to a 64-wide tile launched on
//! its own correct grid, `grid.x = ceil(N/64)`.
//!
//! The legacy `moe_w4a16_grouped_gemm` entry in the same file is deliberately
//! left at `N_TILE 64`, which makes it the perfect reference: it is the
//! unmodified 64-wide code path, and its math is identical to `_ptrtable`
//! when fed a stacked weight buffer, a uniform per-expert `scale2`, and a NULL
//! (identity) gather.
//!
//! ## Legs (per shape, per expert-load scenario)
//!   1. GOLDEN   `moe_w4a16_grouped_gemm`            @ grid.x = ceil(N/64)
//!   2. FIXED    `moe_w4a16_grouped_gemm_ptrtable`   @ grid.x = ceil(N/128)
//!   2b. FIXED+gather — same, with identity `sorted_token_ids`
//!   3. FIXED_T  `moe_w4a16_grouped_gemm_ptrtable_t` @ grid.x = ceil(N/128)
//!   4. WITNESS  `moe_w4a16_grouped_gemm`            @ grid.x = ceil(N/128)
//!      — a 64-wide tile on the production grid, i.e. exactly what the two
//!      ptrtable entries did before this fix. It MUST differ from GOLDEN, and
//!      every output column >= 64*ceil(N/128) MUST be all-zero. Without this
//!      negative control a passing parity leg would prove nothing.
//!
//! Exit: 0 = pass, 1 = a gate failed, 2 = harness/setup error.
//!
//! The PTX set is selected BY MODEL TARGET (default
//! `nemotron-3-nano-30b-a3b`, the Lightning target) rather than via
//! `ptx_modules()`, which returns target 0. Only the two non-shadowed Nemotron
//! MoE targets resolve `moe_w4a16_grouped_gemm.cu` from `common/`; every other
//! target ships a shadow that is already 128-wide, so running against target 0
//! would pass without exercising this fix at all.
//!
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example moe_grouped_ntile_microtest -- [seed] [model-target]

mod fixture;

use anyhow::{Result, bail};
use fixture::{
    LIGHTNING_TARGET, Leg, M_TILE, MODULE, Rng, Shape, build_case, run_leg, scenario_counts,
    time_leg, zero_col_mask,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

fn run() -> Result<bool> {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x5EED_1234);
    let want = args.get(2).map_or(LIGHTNING_TARGET, |s| s.as_str());

    // Pick the PTX set for the target that actually resolves this kernel from
    // `common/`. A single-target build has exactly one set; use it as-is.
    let sets = atlas_kernels::all_ptx_sets();
    let chosen = if sets.len() == 1 {
        &sets[0]
    } else {
        match sets.iter().find(|s| s.target.model == want) {
            Some(s) => s,
            None => bail!(
                "target '{want}' is not in this binary ({} sets); rebuild with \
                 ATLAS_TARGET_MODEL='*' or ATLAS_TARGET_MODEL={want}",
                sets.len()
            ),
        }
    };
    println!(
        "kernel target: {}/{} ({} modules)",
        chosen.target.model,
        chosen.target.quant,
        chosen.modules.len()
    );

    let backend = AtlasCudaBackend::new(0, &chosen.modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let k_legacy = gpu.kernel(MODULE, "moe_w4a16_grouped_gemm")?;
    let k_pt = gpu.kernel(MODULE, "moe_w4a16_grouped_gemm_ptrtable")?;
    let k_pt_t = gpu.kernel(MODULE, "moe_w4a16_grouped_gemm_ptrtable_t")?;
    if k_legacy.0 == 0 || k_pt.0 == 0 || k_pt_t.0 == 0 {
        bail!("grouped-GEMM kernels did not resolve in module '{MODULE}'");
    }

    // Nemotron-H Lightning-30B A3B: hidden 2688, expert inter 1856.
    let shapes = [
        Shape {
            name: "UP  ",
            n: 1856,
            k: 2688,
        },
        Shape {
            name: "DOWN",
            n: 2688,
            k: 1856,
        },
    ];

    let mut all_ok = true;
    for shape in &shapes {
        for kind in ["decode", "mixed"] {
            let mut rng = Rng(seed ^ ((shape.n as u64) << 20) ^ (kind.len() as u64));
            let counts = scenario_counts(kind, &mut rng);
            let total: usize = counts.iter().sum();
            let max_m_tiles = counts
                .iter()
                .map(|c| c.div_ceil(M_TILE).max(1))
                .max()
                .unwrap_or(1) as u32;
            let g64 = shape.n.div_ceil(64) as u32;
            let g128 = shape.n.div_ceil(128) as u32;
            let covered_by_64 = (g128 as usize) * 64; // what the defect actually wrote

            let case = build_case(gpu, shape, &counts, total, &mut rng)?;
            let leg_gold = Leg {
                kernel: k_legacy,
                ptrtable: None,
                grid_x: g64,
            };
            let leg_fix = Leg {
                kernel: k_pt,
                ptrtable: Some((case.b_tbl, case.s_tbl, case.scale2_tbl, DevicePtr::NULL)),
                grid_x: g128,
            };
            let leg_fix_g = Leg {
                ptrtable: Some((case.b_tbl, case.s_tbl, case.scale2_tbl, case.sorted_ids)),
                ..leg_fix
            };
            let leg_fix_t = Leg {
                kernel: k_pt_t,
                ptrtable: Some((case.b_tbl_t, case.s_tbl_t, case.scale2_tbl, DevicePtr::NULL)),
                grid_x: g128,
            };
            let leg_witness = Leg {
                grid_x: g128,
                ..leg_gold
            };

            let golden = run_leg(gpu, stream, leg_gold, &case, shape, max_m_tiles, total)?;
            let fixed = run_leg(gpu, stream, leg_fix, &case, shape, max_m_tiles, total)?;
            let fixed_g = run_leg(gpu, stream, leg_fix_g, &case, shape, max_m_tiles, total)?;
            let fixed_t = run_leg(gpu, stream, leg_fix_t, &case, shape, max_m_tiles, total)?;
            let witness = run_leg(gpu, stream, leg_witness, &case, shape, max_m_tiles, total)?;

            // Cost of correctness: `witness` is what production launched before
            // this fix (half the columns for half the work), `fixed` is the same
            // grid now doing all of it, `golden` is the same full work spread
            // over twice as many narrower CTAs.
            let ms_gold = time_leg(gpu, stream, leg_gold, &case, shape, max_m_tiles)?;
            let ms_fix = time_leg(gpu, stream, leg_fix, &case, shape, max_m_tiles)?;
            let ms_fix_t = time_leg(gpu, stream, leg_fix_t, &case, shape, max_m_tiles)?;
            let ms_wit = time_leg(gpu, stream, leg_witness, &case, shape, max_m_tiles)?;
            case.free(gpu);

            let zc_gold = zero_col_mask(&golden, total, shape.n);
            let zc_wit = zero_col_mask(&witness, total, shape.n);
            let wit_zero_cols = zc_wit.iter().filter(|z| **z).count();
            let wit_tail_all_zero = zc_wit[covered_by_64..].iter().all(|z| *z);
            let gold_tail_written = zc_gold[covered_by_64..].iter().any(|z| !*z);

            let ok_fix = fixed == golden;
            let ok_gather = fixed_g == golden;
            let ok_t = fixed_t == golden;
            let ok_ctrl = witness != golden && wit_tail_all_zero && gold_tail_written;

            println!(
                "{} N={:<5} K={:<5} {:<6} rows={:<4} m_tiles={} grid.x 64w={:<3} 128w={:<3} | \
                 ptrtable=={} ptrtable+gather=={} ptrtable_t=={} | \
                 witness differs={} witness zero-cols={}/{} (cols >= {} all zero={})",
                shape.name,
                shape.n,
                shape.k,
                kind,
                total,
                max_m_tiles,
                g64,
                g128,
                ok_fix,
                ok_gather,
                ok_t,
                witness != golden,
                wit_zero_cols,
                shape.n,
                covered_by_64,
                wit_tail_all_zero,
            );
            println!(
                "     ms/launch: golden(64w, grid.x={g64}) {ms_gold:.3}  \
                 ptrtable(128w, grid.x={g128}) {ms_fix:.3}  \
                 ptrtable_t(128w) {ms_fix_t:.3}  \
                 [pre-fix half-work launch {ms_wit:.3}]"
            );
            if !ok_fix || !ok_gather || !ok_t {
                println!(
                    "  FAIL: retuned N_TILE_PT output is not byte-identical to the 64-wide golden"
                );
                all_ok = false;
            }
            if !ok_ctrl {
                println!(
                    "  FAIL: negative control did not reproduce the defect \
                     (differs={}, tail all-zero={}, golden tail written={}) — \
                     the parity legs then prove nothing",
                    witness != golden,
                    wit_tail_all_zero,
                    gold_tail_written
                );
                all_ok = false;
            }
        }
    }

    println!(
        "\n{}",
        if all_ok {
            "PASS — the 128-wide ptrtable entries are byte-identical to the 64-wide \
             kernel on its own correct grid, and the pre-fix launch is provably \
             half-zero."
        } else {
            "FAIL"
        }
    );
    Ok(all_ok)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("harness error: {e:#}");
            std::process::exit(2);
        }
    }
}
