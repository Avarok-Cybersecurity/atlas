// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the crate-public half of GDN write-on-accept for the
//! batched speculative verify (port lane L10, row `gdn_woa_mtp`): the
//! scheduler's request rule, the K-generic stash geometry the model and the
//! layers share, and the `atlas_gdn_woa_folds_total{k}` counter.
//!
//! The layer half (the engage decision, the launches, the fold) lives in
//! `layers/qwen3_ssm/{woa.rs, woa_wyn.rs}`; the model half (stash binding,
//! fold dispatch) in `model/trait_impl/gdn_woa.rs`. This file holds only what
//! a caller outside spark-model (the scheduler, `/metrics`) needs, as pure
//! functions where it can, so the rules are pinned by CPU tests.
//!
//! Design after the ReplaySSM record-and-fold idea (Cinference / NInfer,
//! `docs/maintainer/replayssm-gdn.md`, Apache-2.0). Idea only: no upstream
//! code is copied.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

/// Widest verify width any write-on-accept twin serves (the wyN family's
/// top instantiation, and the pointer-table slab count per layer).
pub const MAX_WOA_ROWS: usize = 16;

/// Narrowest width with a twin (K = 2, the wy2 parent).
pub const MIN_WOA_ROWS: usize = 2;

/// Floats one verify row adds to a sequence's stash:
/// `vn[nv][vd] | g[nv] | sk[nk][kd]`.
pub const fn stash_row_floats(nk: usize, nv: usize, kd: usize, vd: usize) -> usize {
    nv * vd + nv + nk * kd
}

/// Per-sequence stash floats for verify width `k`:
/// `vn[k][nv][vd] | g[k][nv] | sk[k][nk][kd]`, the layout of both
/// `gated_delta_rule_wy4_woa.cu` (k = 4) and `gated_delta_rule_wyn_woa.cu`.
pub const fn stash_seq_floats_k(k: usize, nk: usize, nv: usize, kd: usize, vd: usize) -> usize {
    k * stash_row_floats(nk, nv, kd, vd)
}

/// Rows per sequence the model's stash must hold.
///
/// Row off: 4, the K=4 pair's stash exactly as before the row. Row on: the
/// widest width this model's batched verify can be asked for, so every twin
/// width fits: the DFlash band `dflash_k` (gamma + 1, capped at
/// [`MAX_WOA_ROWS`]) when a DFlash drafter is installed, else 4 (the MTP
/// batched verify never exceeds K = 4: `can_batch_verify` bounds it to
/// 2..=4).
pub const fn stash_rows_cap(row_on: bool, dflash_k: Option<usize>) -> usize {
    if !row_on {
        return 4;
    }
    match dflash_k {
        Some(k) if k > 4 => {
            if k > MAX_WOA_ROWS {
                MAX_WOA_ROWS
            } else {
                k
            }
        }
        _ => 4,
    }
}

/// Scale the layer-reported K=4 stash width (`gdn_woa_stash_seq_floats`,
/// four rows) to `rows` rows. Exact: the K=4 width is `4 * row_floats`.
pub const fn scale_k4_stash_floats(k4_floats: usize, rows: usize) -> usize {
    if rows == 4 {
        k4_floats
    } else {
        k4_floats / 4 * rows
    }
}

/// Everything the MTP batched verify's write-on-accept request depends on.
#[derive(Clone, Copy, Debug)]
pub struct MtpRequestInputs {
    /// The `gdn_woa_mtp` row.
    pub row_on: bool,
    /// `AVAROK_GDN_WOA=1` (`gdn_flags::gdn_woa_enabled`): still required.
    pub woa_on: bool,
    /// `--ssm-h-dtype f16`: the twins read and fold FP32 state only.
    pub h_f16: bool,
    /// `--exact-verify`: the per-token exact chain owns the verify.
    pub exact_verify: bool,
}

/// The ONE place the MTP batched verify decides whether to ASK for
/// write-on-accept. The layers still make the final engage decision per
/// launch (`qwen3_ssm::woa::woa_decision`); asking is only sound when the
/// caller folds every verdict before its commits, which
/// `verify_k4_batch_step` does whenever this returns true.
///
/// UNIFORM batches only (`ks` all equal, one depth run). Under ragged D-Cut
/// depths the batched GDN body runs one launch per depth run and each run
/// clears the layer's single engaged word at entry
/// (`trait_decode_batched_conv_gdn_multi.rs`), so a later run would erase the
/// record of an earlier run's twin and the fold would restore from Hi tables
/// the twin never wrote. A ragged batch keeps the parent kernels and the host
/// restore, byte for byte as before.
pub fn mtp_request_shape(i: MtpRequestInputs, ks: &[usize]) -> bool {
    let Some(&k) = ks.first() else {
        return false;
    };
    i.row_on
        && i.woa_on
        && !i.h_f16
        && !i.exact_verify
        && (MIN_WOA_ROWS..=4).contains(&k)
        && ks.iter().all(|&x| x == k)
}

/// [`mtp_request_shape`] over the process's resolved row and flags.
pub fn mtp_batched_request(ks: &[usize]) -> bool {
    use crate::layers::qwen3_ssm::gdn_flags;
    mtp_request_shape(
        MtpRequestInputs {
            row_on: crate::layers::ops::port_rows::gdn_woa_mtp::row(),
            woa_on: gdn_flags::gdn_woa_enabled(),
            h_f16: gdn_flags::ssm_h_fp16_enabled(),
            exact_verify: gdn_flags::verify_exact_enabled(),
        },
        ks,
    )
}

/// Post-verdict fold launches that committed state, by verify width
/// (index = K, 0..=16). Counted once per fold dispatch over the batch
/// (`model/trait_impl/gdn_woa.rs`), whichever kernel the verify replayed:
/// the fold either applies the twin's stash or performs the parent's
/// partial-accept restore, and either way the host skipped its own h copy.
static FOLDS: [AtomicU64; MAX_WOA_ROWS + 1] = [const { AtomicU64::new(0) }; MAX_WOA_ROWS + 1];

/// Count one fold dispatch at verify width `k` (widths above the table land
/// in its last bucket rather than vanishing).
pub fn record_fold(k: usize) {
    FOLDS[k.min(MAX_WOA_ROWS)].fetch_add(1, Ordering::Relaxed);
}

/// Snapshot of the fold counters, index = K.
pub fn folds_by_k() -> [u64; MAX_WOA_ROWS + 1] {
    std::array::from_fn(|k| FOLDS[k].load(Ordering::Relaxed))
}

/// Append `atlas_gdn_woa_folds_total{k}` to the `/metrics` text, in the
/// `# HELP` / `# TYPE` form of spark-server's `port_metrics` helpers. The
/// `k="4"` series is always present (zero on a row-off serve, on every
/// target); other widths appear once they count.
pub fn render_prometheus(out: &mut String) {
    render_from(out, &folds_by_k());
}

fn render_from(out: &mut String, folds: &[u64; MAX_WOA_ROWS + 1]) {
    let name = "atlas_gdn_woa_folds_total";
    let _ = writeln!(
        out,
        "# HELP {name} GDN write-on-accept post-verdict fold dispatches by verify width k.\n\
         # TYPE {name} counter"
    );
    for (k, &v) in folds.iter().enumerate() {
        if k == 4 || v > 0 {
            let _ = writeln!(out, "{name}{{k=\"{k}\"}} {v}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ON: MtpRequestInputs = MtpRequestInputs {
        row_on: true,
        woa_on: true,
        h_f16: false,
        exact_verify: false,
    };

    #[test]
    fn uniform_ladder_widths_ask() {
        for k in 2..=4 {
            assert!(mtp_request_shape(ON, &[k; 8]), "k={k}");
            assert!(mtp_request_shape(ON, &[k, k]), "k={k}");
        }
    }

    #[test]
    fn every_gate_declines_alone() {
        let ks = [4usize; 4];
        assert!(!mtp_request_shape(
            MtpRequestInputs {
                row_on: false,
                ..ON
            },
            &ks
        ));
        assert!(!mtp_request_shape(
            MtpRequestInputs {
                woa_on: false,
                ..ON
            },
            &ks
        ));
        assert!(!mtp_request_shape(
            MtpRequestInputs { h_f16: true, ..ON },
            &ks
        ));
        assert!(!mtp_request_shape(
            MtpRequestInputs {
                exact_verify: true,
                ..ON
            },
            &ks
        ));
    }

    /// Ragged D-Cut depths keep the parent: one engaged word per layer cannot
    /// describe two depth runs.
    #[test]
    fn ragged_batches_never_ask() {
        assert!(!mtp_request_shape(ON, &[4, 4, 3]));
        assert!(!mtp_request_shape(ON, &[4, 2]));
        assert!(!mtp_request_shape(ON, &[3, 3, 2, 2]));
    }

    /// Widths outside the MTP batched range (and the empty batch) never ask.
    #[test]
    fn out_of_range_widths_never_ask() {
        assert!(!mtp_request_shape(ON, &[]));
        assert!(!mtp_request_shape(ON, &[1, 1]));
        assert!(!mtp_request_shape(ON, &[5, 5]));
    }

    #[test]
    fn row_off_stash_is_the_k4_stash() {
        for dflash in [None, Some(2), Some(4), Some(8), Some(16), Some(17)] {
            assert_eq!(stash_rows_cap(false, dflash), 4, "{dflash:?}");
        }
        let k4 = stash_seq_floats_k(4, 16, 48, 128, 128);
        assert_eq!(scale_k4_stash_floats(k4, 4), k4);
    }

    #[test]
    fn row_on_stash_covers_the_widest_batched_width() {
        assert_eq!(stash_rows_cap(true, None), 4);
        assert_eq!(stash_rows_cap(true, Some(3)), 4);
        assert_eq!(stash_rows_cap(true, Some(8)), 8);
        assert_eq!(stash_rows_cap(true, Some(16)), 16);
        assert_eq!(stash_rows_cap(true, Some(17)), 16);
        let k4 = stash_seq_floats_k(4, 16, 32, 128, 128);
        for rows in [4, 8, 16] {
            assert_eq!(
                scale_k4_stash_floats(k4, rows),
                stash_seq_floats_k(rows, 16, 32, 128, 128)
            );
        }
    }

    /// The K=4 width is the shipped wy4 pair's (131,840 bytes per sequence at
    /// 48 v-heads x 16 k-heads x 128), and the row width is K-independent.
    #[test]
    fn stash_geometry_matches_the_kernels() {
        assert_eq!(stash_seq_floats_k(4, 16, 48, 128, 128) * 4, 131_840);
        assert_eq!(stash_row_floats(16, 32, 128, 128), 32 * 128 + 32 + 16 * 128);
        for k in MIN_WOA_ROWS..=MAX_WOA_ROWS {
            assert_eq!(
                stash_seq_floats_k(k, 16, 32, 128, 128),
                k * stash_row_floats(16, 32, 128, 128)
            );
        }
    }

    #[test]
    fn metrics_always_carry_the_k4_series() {
        let mut s = String::new();
        render_from(&mut s, &[0; MAX_WOA_ROWS + 1]);
        assert_eq!(
            s,
            "# HELP atlas_gdn_woa_folds_total GDN write-on-accept post-verdict fold \
             dispatches by verify width k.\n# TYPE atlas_gdn_woa_folds_total counter\n\
             atlas_gdn_woa_folds_total{k=\"4\"} 0\n"
        );
        let mut f = [0u64; MAX_WOA_ROWS + 1];
        f[2] = 3;
        f[16] = 1;
        let mut s = String::new();
        render_from(&mut s, &f);
        assert!(s.contains("atlas_gdn_woa_folds_total{k=\"2\"} 3\n"));
        assert!(s.contains("atlas_gdn_woa_folds_total{k=\"4\"} 0\n"));
        assert!(s.contains("atlas_gdn_woa_folds_total{k=\"16\"} 1\n"));
        assert!(!s.contains("k=\"3\""));
    }

    #[test]
    fn record_fold_clamps_wide_widths_into_the_last_bucket() {
        let before = folds_by_k();
        record_fold(40);
        record_fold(4);
        let after = folds_by_k();
        assert!(after[MAX_WOA_ROWS] > before[MAX_WOA_ROWS]);
        assert!(after[4] > before[4]);
    }
}
