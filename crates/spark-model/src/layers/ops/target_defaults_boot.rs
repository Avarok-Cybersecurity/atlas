// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the `target defaults (<hw>): …` boot line over the resolved
//! serving defaults.
//!
//! Moved out of `target_defaults.rs` unchanged (that file had crossed the
//! 500-line cap) and re-exported there, so `target_defaults::summary_line` and
//! `target_defaults::format_levers` are the paths every caller already uses.

use super::{Resolved, TargetLevers, resolved};

/// `target defaults (<hw>): …` — one line naming every resolved value and
/// which came from the environment.
///
/// Built here rather than in `spark-server` so the line and the resolution are
/// the same code: a log that formats its own idea of the table is how a dead
/// lever stays invisible for a campaign (`serve_flags.rs`'s own lesson).
pub fn summary_line() -> String {
    format_levers(resolved())
}

/// [`summary_line`] over a table the caller already has — pure, so the line can
/// be graded for ANY target from a CPU test without touching the process
/// environment or sealing the `OnceLock`.
pub fn format_levers(l: &TargetLevers) -> String {
    let onoff =
        |r: Resolved<bool>| format!("{}{}", if r.value { "on" } else { "off" }, r.source.tag());
    // `u32::MAX` is the no-cap baseline, not a chosen bound. Printing
    // 4294967295 in the serve log would read as a decision someone made.
    let cap = |v: u32| {
        if v == u32::MAX {
            "max".to_string()
        } else {
            v.to_string()
        }
    };
    format!(
        "target defaults ({hw}): sm_count={sms} \
         lm_head_batchm_max={batchm}{batchm_src} \
         ssm_batched_recurrent={recurrent} gdn_prefill_tc={gdn_tc} \
         ssm_ba_gates_hopper={ba_gates} decode_split_silu={silu} \
         attn_decode_splitk={splitk}{splitk_src} ffn_m16_tc={ffn_m16_tc} \
         attn_m16_tc={attn_m16_tc} lm_head_m16_tc={lm_head_m16_tc} \
         attn_ncol_gemv={attn_ncol_gemv} ffn_gateup_fused={gateup} \
         fp8_act_quant_hopper={act_quant} dense_fp8_from_bf16={fp8_from_bf16} \
         gdn_fp8_from_bf16={gdn_fp8} attn_fp8_from_bf16={attn_fp8} \
         ffn_bf16_cublas_prefill={bf16_cublas} memory_fit_consumer={memory_fit} \
         gdn_batched_gemm={gdn_gemm} attn_gated_bf16_batchm={gated_batchm} \
         w8a8_prefill_max_m={w8a8_wide}/{w8a8_narrow}{w8a8_src}",
        hw = if l.hw.is_empty() { "unknown" } else { l.hw },
        // Not a resolvable lever — it is a FACT about the part, cross-checked
        // at boot against the driver. Printed on this line because the levers
        // that will read it (grid sizing) are on it, and a reader comparing
        // two campaign logs needs both in one grep.
        sms = avarok_kernels::TARGET_SM_COUNT,
        batchm = l.lm_head_batchm_max.value,
        batchm_src = l.lm_head_batchm_max.source.tag(),
        recurrent = onoff(l.ssm_batched_recurrent),
        gdn_tc = onoff(l.gdn_prefill_tc),
        ba_gates = onoff(l.ssm_ba_gates_hopper),
        act_quant = onoff(l.fp8_act_quant_hopper),
        silu = onoff(l.decode_split_silu),
        splitk = l.attn_decode_splitk.value.label(),
        splitk_src = l.attn_decode_splitk.source.tag(),
        ffn_m16_tc = onoff(l.ffn_m16_tc),
        attn_m16_tc = onoff(l.attn_m16_tc),
        lm_head_m16_tc = onoff(l.lm_head_m16_tc),
        attn_ncol_gemv = onoff(l.attn_ncol_gemv),
        gateup = onoff(l.ffn_gateup_fused),
        fp8_from_bf16 = onoff(l.dense_fp8_from_bf16),
        gdn_fp8 = onoff(l.gdn_fp8_from_bf16),
        attn_fp8 = onoff(l.attn_fp8_from_bf16),
        bf16_cublas = onoff(l.ffn_bf16_cublas_prefill),
        gdn_gemm = onoff(l.gdn_batched_gemm),
        gated_batchm = onoff(l.attn_gated_bf16_batchm),
        memory_fit = onoff(l.memory_fit_consumer),
        // Printed as widening/narrowing. `max` reads as "no cap" rather than
        // 4294967295, which would look like a number someone chose.
        w8a8_wide = cap(l.w8a8_prefill_max_m_widening.value),
        w8a8_narrow = cap(l.w8a8_prefill_max_m_narrowing.value),
        w8a8_src = l.w8a8_prefill_max_m_widening.source.tag(),
    )
}
