// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the pure half of the GDN W4 projection arm (`w4_proj.rs`):
//! which projection a call is for and its shape in the model config, the row
//! plan that splits a call into `decode` and `prefill` chunks, and the shape
//! rules an `(in_proj_qkvz, out_proj)` pair must meet before any launch. No
//! GPU and no layer: every rule here is decidable from integers, so each one
//! is pinned by `w4_proj_plan_tests.rs`.
//!
//! # The row plan
//!
//! * A call of `rows <= max_decode_rows` is ONE `decode` (any site).
//! * A decode-side call (M = 1 decode, batched decode, MTP verify,
//!   multi-sequence decode) wider than the window runs as `decode` chunks of
//!   `max_decode_rows` rows. The contract makes `decode` row invariant, so
//!   every decode-side row gets exactly the bits of an M = 1 decode of the
//!   same token, at every batch width, inside or outside a CUDA graph. That is
//!   what keeps MTP verify at temperature 0 equal to plain decoding on this
//!   route, and it never calls `prefill`, which is not required to be
//!   capturable and may decline into a full-weight BF16 dequant.
//! * A prompt wider than the window runs as `prefill` chunks cut by the
//!   shared rule (`port::w4::select::prefill_chunks`): even chunks of at most
//!   `prefill_chunk_cap(max_decode_rows)` rows, each wider than the window,
//!   which is the `m > max_decode_rows()` precondition of `prefill` and the
//!   row bound the shared scratch (`port::w4::scratch`) is sized for.

use anyhow::{Result, ensure};
use avarok_core::config::ModelConfig;

use crate::port::w4::{W4Linear, select};

#[cfg(test)]
#[path = "w4_proj_plan_tests.rs"]
mod tests;

/// Which of the two projections a call is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum W4Role {
    Qkvz,
    OutProj,
}

impl W4Role {
    /// The projection's name in error messages (no allocation per call).
    pub(super) fn what(self) -> &'static str {
        match self {
            Self::Qkvz => "GDN in_proj_qkvz",
            Self::OutProj => "GDN out_proj",
        }
    }

    /// `(n, k)` the model config gives this projection: `[Q|K|V|Z]` is
    /// `[ssm_qkvz_size, hidden]`, `out_proj` is `[hidden, value_dim]`.
    pub(super) fn dims(self, c: &ModelConfig) -> (usize, usize) {
        let value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
        match self {
            Self::Qkvz => (c.ssm_qkvz_size(), c.hidden_size),
            Self::OutProj => (c.hidden_size, value_dim),
        }
    }

    /// The once-per-model latch of the declined-prefill log line.
    pub(super) fn decline_key(self) -> &'static str {
        match self {
            Self::Qkvz => "log:gdn_w4_qkvz_prefill_declined",
            Self::OutProj => "log:gdn_w4_out_proj_prefill_declined",
        }
    }
}

/// Where a call comes from: decode-side rows (M = 1, batched decode, verify,
/// multi-sequence) or a prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum W4At {
    Rows,
    Prefill,
}

/// How one chunk of rows is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum W4Step {
    Decode,
    Prefill,
}

/// One contiguous run of rows `[start, start + len)` and its step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct W4Chunk {
    pub start: usize,
    pub len: usize,
    pub step: W4Step,
}

/// The row plan of the module docs. `max_decode_rows` must be at least 1
/// (`check_pair_against` refuses a weight reporting 0 before any plan is
/// made); a 0 here is treated as 1 so the plan still covers every row once.
pub(super) fn plan(at: W4At, rows: usize, max_decode_rows: usize) -> Vec<W4Chunk> {
    if rows == 0 {
        return Vec::new();
    }
    let max = max_decode_rows.max(1);
    if rows <= max {
        return vec![W4Chunk {
            start: 0,
            len: rows,
            step: W4Step::Decode,
        }];
    }
    match at {
        W4At::Rows => (0..rows)
            .step_by(max)
            .map(|start| W4Chunk {
                start,
                len: max.min(rows - start),
                step: W4Step::Decode,
            })
            .collect(),
        W4At::Prefill => select::prefill_chunks(rows, select::prefill_chunk_cap(max), max)
            .into_iter()
            .map(|(start, len)| W4Chunk {
                start,
                len,
                step: W4Step::Prefill,
            })
            .collect(),
    }
}

/// Shape rules a pair must meet with no model config at hand: non-empty,
/// `in_proj_qkvz`'s K equal to `out_proj`'s N (both the hidden size, which
/// also catches a swapped pair), and both K a multiple of 8, so the row
/// offsets every chunk takes (`start * k * 2` bytes) stay 16-byte aligned.
pub(super) fn check_pair(qn: usize, qk: usize, on: usize, ok: usize) -> Result<()> {
    ensure!(
        qn > 0 && qk > 0 && on > 0 && ok > 0,
        "GDN W4 pair: empty projection (in_proj_qkvz [{qn}x{qk}], out_proj [{on}x{ok}])"
    );
    ensure!(
        qk == on,
        "GDN W4 pair: in_proj_qkvz K ({qk}) and out_proj N ({on}) must both be the hidden size; \
         were the two projections passed in the wrong order?"
    );
    ensure!(
        qk.is_multiple_of(8) && ok.is_multiple_of(8),
        "GDN W4 pair: K must be a multiple of 8 so every row starts 16-byte aligned \
         (in_proj_qkvz K {qk}, out_proj K {ok})"
    );
    Ok(())
}

/// The full check a call makes before its first launch: [`check_pair`], then
/// each weight against the shape this model's config gives its projection
/// (and a nonzero decode window), through the shared `select::check_shape`.
pub(super) fn check_pair_against(
    qkvz: &dyn W4Linear,
    out: &dyn W4Linear,
    c: &ModelConfig,
) -> Result<()> {
    check_pair(qkvz.n(), qkvz.k(), out.n(), out.k())?;
    for (w, role) in [(qkvz, W4Role::Qkvz), (out, W4Role::OutProj)] {
        let (n, k) = role.dims(c);
        select::check_shape(w, n, k, role.what())?;
    }
    Ok(())
}
