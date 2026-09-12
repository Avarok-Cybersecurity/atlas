// SPDX-License-Identifier: AGPL-3.0-only

//! The serve log's one line, split from the table it reports.
//!
//! Split out at the repo's 500-line ceiling when the #927 fusion rows and
//! the #927 strided GDN row landed together: each branch left
//! `target_defaults.rs` just under the cap and the merge crossed it. The
//! seam is the `#[path]` child `target_defaults_splitk_tests.rs` already
//! uses, so the formatter and the resolution stay one compilation unit —
//! `format_levers` is re-exported from the parent and every existing
//! `target_defaults::{summary_line, format_levers}` path is unchanged.
//!
//! Still ONE line built from the resolved table, for the reason the parent
//! module documents: a log that formats its own idea of the table is how a
//! dead lever stays invisible for a campaign.

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
    let c = l.cublas.value;
    let scope = if c.any() {
        let mut names = Vec::new();
        for (on, name) in [
            (c.ffn, "ffn"),
            (c.attn, "attn"),
            (c.ssm, "ssm"),
            (c.head, "head"),
        ] {
            if on {
                names.push(name);
            }
        }
        names.join(",")
    } else {
        "off".to_string()
    };
    format!(
        "target defaults ({hw}): cublas_gemm_scope={scope}{cublas_src} \
         ffn_batch16_tier={batch16} ffn_m16_tc={ffn_m16} attn_m16_tc={attn_m16} \
         attn_ncol_gemv={ncol} lm_head_m16_tc={head_m16} \
         lm_head_batchm_max={batchm}{batchm_src} ssm_batched_recurrent={recurrent} \
         gdn_decode_hopper={gdn_decode} \
         gdn_decode_strided_hopper={gdn_decode_strided} gdn_prefill_tc={gdn_tc} \
         gdn_spine_vsplit={gdn_vsplit}{gdn_vsplit_src} \
         ssm_ba_gates_hopper={ba_gates} fp8_act_quant_hopper={act_quant} \
         ffn_gateup_fused={gateup} \
         attn_qkv_fused={qkv_fused} \
         decode_split_silu={silu} \
         ssm_decode_ring_slots={ring}{ring_src} \
         w8a8_prefill_max_m={w8a8_wide}/{w8a8_narrow}{w8a8_src} \
         attn_decode_splitk={splitk}{splitk_src}",
        hw = if l.hw.is_empty() { "unknown" } else { l.hw },
        cublas_src = l.cublas.source.tag(),
        batch16 = onoff(l.ffn_batch16_tier),
        ffn_m16 = onoff(l.ffn_m16_tc),
        attn_m16 = onoff(l.attn_m16_tc),
        ncol = onoff(l.attn_ncol_gemv),
        head_m16 = onoff(l.lm_head_m16_tc),
        batchm = l.lm_head_batchm_max.value,
        batchm_src = l.lm_head_batchm_max.source.tag(),
        recurrent = onoff(l.ssm_batched_recurrent),
        gdn_decode = onoff(l.gdn_decode_hopper),
        gdn_decode_strided = onoff(l.gdn_decode_strided_hopper),
        gdn_tc = onoff(l.gdn_prefill_tc),
        gdn_vsplit = l.gdn_spine_vsplit.value,
        gdn_vsplit_src = l.gdn_spine_vsplit.source.tag(),
        ba_gates = onoff(l.ssm_ba_gates_hopper),
        act_quant = onoff(l.fp8_act_quant_hopper),
        gateup = onoff(l.ffn_gateup_fused),
        qkv_fused = onoff(l.attn_qkv_fused),
        silu = onoff(l.decode_split_silu),
        ring = match l.ssm_decode_ring_slots.value {
            Some(n) => n.to_string(),
            None => "auto".to_string(),
        },
        ring_src = l.ssm_decode_ring_slots.source.tag(),
        // Printed as widening/narrowing. `max` reads as "no cap" rather than
        // 4294967295, which would look like a number someone chose.
        w8a8_wide = cap(l.w8a8_prefill_max_m_widening.value),
        w8a8_narrow = cap(l.w8a8_prefill_max_m_narrowing.value),
        w8a8_src = l.w8a8_prefill_max_m_widening.source.tag(),
        splitk = l.attn_decode_splitk.value.label(),
        splitk_src = l.attn_decode_splitk.source.tag(),
    )
}
