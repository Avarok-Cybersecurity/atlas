// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the opt-in DFlash fit (`AVAROK_DFLASH_FIT=1`): price a
//! DFlash serve by what it will really allocate, and when `--dflash-gamma` is
//! not given pick the widest verify width the card can hold. Off by default;
//! with the env unset or off, nothing in this file runs and every number the
//! preflight prints is the one it always printed.
//!
//! # Why a Qwen3.5-4B DFlash serve at 8 streams was refused
//!
//! The boot log of an earlier GPU run on this card: "only 8.08 GB remains but 13.32 GB is
//! needed". The 13.32 GiB is `inference=11314 MB + buffer_arena=2327 MB`, and
//! the 11,314 MiB is exactly
//!
//! ```text
//!   SSM pools   6,960   8 base blobs (408) + 8 verify slots x 819
//!   headroom    4,096   clamped from 512 + 921 + 2,560 + 1,024 = 5,017
//!   GDN prefill   258   8,192 tokens
//! ```
//!
//! * **Per verify slot 819 MiB, at every slot.** DFlash's verify is gamma + 1
//!   = 16 rows (the drafter's `block_size` is 16; `default_dflash_gamma`
//!   clamps block + 2 to 15), and snapshot rollback keeps one h state per
//!   verify row past the first: 15 x 48 MiB + 16 x 3 MiB + a 51 MiB checkpoint.
//!   The pools are uniform (`SsmStatePool::new`, `uniform_h`), so all eight
//!   slots pay the full block and the pools alone are 6.8 GiB of an 8.08 GiB
//!   budget.
//! * **The preflight ignored `--dflash-gamma` for the pools.** The head and
//!   `SsmStatePool::new` size from the flag (`DflashBuildArgs::gamma`), but the
//!   preflight sized its pools from the drafter's block size even when the flag
//!   was given (`pool_num_drafts`), so `--dflash-gamma 7` still reserved 15-wide
//!   pools. With this row on the reserve follows the flag.
//! * **The ctx accumulators are priced at the worst case.** 2,560 MiB of the
//!   headroom is 8 sequences x 8,192 tokens x 8 taps x 2,560 x 2 B, priced at
//!   the compiled-in 16,384 cap and the whole `--max-seq-len`. The head
//!   allocates per request (`alloc_state_windowed`: the smaller of the
//!   `AVAROK_DFLASH_CTX_CAP` window and the request's own budget), and the
//!   preflight never read that env. With this row on it does.
//! * **The buffer arena is the 8,192-token prefill arena** (2,327 MiB at the 4B),
//!   which `--max-prefill-tokens` already lowers.
//!
//! # What the row does
//!
//! 1. The pool width follows `--dflash-gamma` (the head's real gamma).
//! 2. The accumulators are priced at `AVAROK_DFLASH_CTX_CAP` (`0` = uncapped).
//! 3. With no `--dflash-gamma`, the widest gamma (down to [`MIN_GAMMA`]) for
//!    which pools, headroom, arena, the drafter's weights and its post-sizing
//!    allocations, and the KV floor all fit beside the resident weights is
//!    chosen and pinned (`ReservePreflight::dflash_gamma`, which `serve_load`
//!    writes into `args.dflash_gamma` before the drafter is resolved). The
//!    boot log prints every term. A card that holds no width logs what even
//!    [`MIN_GAMMA`] needs and leaves the default alone.
//!
//! A narrower gamma lowers the tokens a verify step can take; it never changes
//! what is accepted (the verify is exact). The decision is made from the same
//! SSOT functions the allocators use (`ssm_pool_reserve_bytes`,
//! `spec_headroom`), so reserve and allocation cannot disagree.

use avarok_core::config::ModelConfig;
use spark_model::ssm_reserve::SsmRollbackMode;

use super::headroom::checkpoint_bytes;
use super::spec_headroom;
use crate::cli;

const MIB: usize = 1024 * 1024;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The environment switch (toggle grammar: unset, `0`, `false`, `off`, `no` are
/// off; anything else is on).
pub(super) const ENV: &str = "AVAROK_DFLASH_FIT";

/// Narrowest gamma the fit will choose: verify width 4, the narrowest the
/// batched GDN verify kernels cover without falling to the sequential arm.
pub(super) const MIN_GAMMA: usize = 3;

/// What the fit keeps back for CUDA context growth beside the sum it checks.
/// The audit measures free memory after the context exists; the preflight's
/// `beside_weights` is the pre-load figure.
pub(super) const FIT_MARGIN: usize = 256 * MIB;

/// Scratch `factory::build`'s `dflash_reserve` adds (`300 << 20`).
const DRAFTER_SCRATCH: usize = 300 * MIB;

/// Is the fit on? Read every call: the preflight runs once, and a CPU test
/// drives [`enabled_from`] instead.
pub(super) fn enabled() -> bool {
    enabled_from(std::env::var(ENV).ok().as_deref())
}

/// [`enabled`] for a given raw value.
pub(super) fn enabled_from(raw: Option<&str>) -> bool {
    match raw.map(|v| v.trim().to_ascii_lowercase()) {
        None => false,
        Some(v) => !matches!(v.as_str(), "" | "0" | "false" | "off" | "no"),
    }
}

/// The head's ctx window cap in tokens (`AVAROK_DFLASH_CTX_CAP`, default
/// 16,384; `0` = uncapped, which prices the whole `--max-seq-len`).
pub(super) fn head_ctx_cap() -> usize {
    match spark_model::layers::dflash_head::dflash_ctx_cap() {
        0 => usize::MAX,
        cap => cap,
    }
}

/// Every term of the fit that does not depend on gamma, plus the memory it
/// must fit in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Terms {
    /// Free memory beside the resident weights (`memory_beside_weights`).
    pub(super) beside_weights: usize,
    /// The inference reserve minus its SSM pools and its CUDA headroom: GDN
    /// prefill, Marconi, the ring, per-sequence state.
    pub(super) fixed: usize,
    /// The buffer arena.
    pub(super) arena: usize,
    /// The drafter's weights and everything it allocates after the KV sizing.
    pub(super) drafter: usize,
    /// KV for `--max-batch-size` sequences at the floor context.
    pub(super) kv_floor: usize,
}

impl Terms {
    /// Bytes a serve needs beside the weights at a gamma whose pools and CUDA
    /// headroom are given.
    pub(super) fn need(&self, pools: usize, headroom: usize) -> usize {
        self.fixed
            .saturating_add(pools)
            .saturating_add(headroom)
            .saturating_add(self.arena)
            .saturating_add(self.drafter)
            .saturating_add(self.kv_floor)
            .saturating_add(FIT_MARGIN)
    }
}

/// What the fit concluded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Decision {
    /// The widest gamma that fits, `None` when even [`MIN_GAMMA`] does not.
    pub(super) gamma: Option<usize>,
    /// Bytes needed at the default gamma.
    pub(super) need_default: usize,
    /// Bytes needed at the chosen gamma (at [`MIN_GAMMA`] when none fits).
    pub(super) need_chosen: usize,
}

/// The widest gamma in `MIN_GAMMA..=default_gamma` whose need fits
/// `terms.beside_weights`. `pools_at(g)` and `headroom_at(g)` are the SSM pool
/// reserve and the CUDA headroom at gamma `g`.
pub(super) fn pick(
    default_gamma: usize,
    terms: &Terms,
    pools_at: impl Fn(usize) -> usize,
    headroom_at: impl Fn(usize) -> usize,
) -> Decision {
    let need_at = |g: usize| terms.need(pools_at(g), headroom_at(g));
    let need_default = need_at(default_gamma);
    let floor = MIN_GAMMA.min(default_gamma);
    let chosen = (floor..=default_gamma)
        .rev()
        .find(|&g| need_at(g) <= terms.beside_weights);
    Decision {
        gamma: chosen,
        need_default,
        need_chosen: need_at(chosen.unwrap_or(floor)),
    }
}

/// The drafter's footprint: its weights plus what `factory::build`'s
/// `dflash_reserve` charges for the allocations made after the KV sizing
/// (drafter KV, fused KV, prompt-hidden capture, FP8 mirrors, scratch).
#[allow(clippy::too_many_arguments)]
pub(super) fn drafter_bytes_from(
    weights: usize,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    drafter_hidden: usize,
    target_hidden: usize,
    max_seq_len: usize,
    fp8_mirrors: bool,
) -> usize {
    let kv_dim = kv_heads * head_dim;
    let drafter_kv = max_seq_len * layers * 2 * kv_dim * 2;
    let fused_kv = layers * 2 * kv_dim * drafter_hidden * 2;
    let capture = max_seq_len * target_hidden * 2;
    let mirrors = if fp8_mirrors { weights / 2 } else { 0 };
    weights + drafter_kv + fused_kv + capture + mirrors + DRAFTER_SCRATCH
}

/// The drafter's footprint from its checkpoint directory (`--draft-model`),
/// or `None` for a remote id or an unreadable directory.
fn drafter_bytes(args: &cli::ServeArgs, config: &ModelConfig) -> Option<usize> {
    let dir = std::path::Path::new(args.draft_model.as_deref()?);
    let weights = checkpoint_bytes(dir)? as usize;
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let get = |k: &str| -> Option<usize> { Some(v.get(k)?.as_u64()? as usize) };
    Some(drafter_bytes_from(
        weights,
        get("num_hidden_layers")?,
        get("num_key_value_heads")?,
        get("head_dim")?,
        get("hidden_size")?,
        config.hidden_size,
        args.max_seq_len,
        // The same predicate as the allocating gate (`!= Some("0")`).
        std::env::var("AVAROK_DFLASH_DRAFTER_FP8").ok().as_deref() != Some("0"),
    ))
}

/// What `preflight_reserve` hands the fit.
pub(super) struct PlanInputs {
    pub(super) beside_weights: usize,
    pub(super) kv_floor: usize,
    pub(super) arena: usize,
    /// The inference reserve minus its SSM pools and its CUDA headroom.
    pub(super) fixed: usize,
    /// Per-seq h and conv blob bytes across all SSM layers, FP32 width.
    pub(super) h_blob: usize,
    pub(super) conv_blob: usize,
    pub(super) mtp_state_slots: usize,
    pub(super) h_f16_pool: bool,
    pub(super) rollback: SsmRollbackMode,
    /// The gamma the serve resolves without the fit.
    pub(super) default_gamma: usize,
}

/// What the preflight swaps into its reserve when the fit chose a gamma.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Applied {
    pub(super) gamma: usize,
    pub(super) ssm_pool_bytes: usize,
    pub(super) cuda_headroom: usize,
}

/// The SSM pool reserve at gamma `g`, as `preflight_reserve` prices it for DFlash.
fn pools_at(args: &cli::ServeArgs, i: &PlanInputs, g: usize) -> usize {
    spark_model::ssm_reserve::ssm_pool_reserve_bytes(
        args.max_batch_size,
        i.h_blob,
        i.conv_blob,
        true,
        g,
        i.mtp_state_slots,
        true,
        i.h_f16_pool,
        i.rollback,
    )
}

/// The CUDA headroom at gamma `g`: the DFlash term reads the gamma from args.
fn headroom_at(args: &cli::ServeArgs, config: &ModelConfig, g: usize) -> usize {
    let mut a = args.clone();
    a.dflash_gamma = Some(g);
    spec_headroom::cuda_headroom(&a, config).0
}

/// Run the fit for this serve. `None` when the row is off, the serve is not
/// DFlash, gamma is pinned by the flag, the drafter's size cannot be read, no
/// gamma fits, or the default already fits.
pub(super) fn plan(args: &cli::ServeArgs, config: &ModelConfig, i: &PlanInputs) -> Option<Applied> {
    if !enabled() || !args.dflash || args.dflash_gamma.is_some() {
        return None;
    }
    let Some(drafter) = drafter_bytes(args, config) else {
        tracing::info!(
            "DFlash fit: skipped, the drafter's size could not be read from --draft-model \
             (a local checkpoint directory is needed)"
        );
        return None;
    };
    let terms = Terms {
        beside_weights: i.beside_weights,
        fixed: i.fixed,
        arena: i.arena,
        drafter,
        kv_floor: i.kv_floor,
    };
    let d = pick(
        i.default_gamma,
        &terms,
        |g| pools_at(args, i, g),
        |g| headroom_at(args, config, g),
    );
    let gb = |b: usize| b as f64 / GIB;
    match d.gamma {
        Some(g) if g == i.default_gamma => {
            tracing::info!(
                "DFlash fit: the default gamma {g} fits: needs {:.2} GiB beside the weights, \
                 {:.2} GiB available (reserve terms {:.2} fixed + arena {:.2} + drafter {:.2} + \
                 KV floor {:.2} + {:.2} margin, plus pools and headroom)",
                gb(d.need_chosen),
                gb(terms.beside_weights),
                gb(terms.fixed),
                gb(terms.arena),
                gb(terms.drafter),
                gb(terms.kv_floor),
                gb(FIT_MARGIN),
            );
            None
        }
        Some(g) => {
            tracing::warn!(
                "DFlash fit: gamma {g} instead of {}: the default needs {:.2} GiB beside the \
                 weights and {:.2} GiB is available; gamma {g} needs {:.2} GiB (pools {:.2}, \
                 headroom {:.2}, fixed {:.2}, arena {:.2}, drafter {:.2}, KV floor {:.2}, margin \
                 {:.2}). Pinned like --dflash-gamma {g}.",
                i.default_gamma,
                gb(d.need_default),
                gb(terms.beside_weights),
                gb(d.need_chosen),
                gb(pools_at(args, i, g)),
                gb(headroom_at(args, config, g)),
                gb(terms.fixed),
                gb(terms.arena),
                gb(terms.drafter),
                gb(terms.kv_floor),
                gb(FIT_MARGIN),
            );
            Some(Applied {
                gamma: g,
                ssm_pool_bytes: pools_at(args, i, g),
                cuda_headroom: headroom_at(args, config, g),
            })
        }
        None => {
            tracing::warn!(
                "DFlash fit: no gamma fits: even {} needs {:.2} GiB beside the weights and \
                 {:.2} GiB is available. Lower --max-batch-size, --max-seq-len or \
                 --max-prefill-tokens, or set AVAROK_DFLASH_CTX_CAP, and the fit is run again.",
                MIN_GAMMA.min(i.default_gamma),
                gb(d.need_chosen),
                gb(terms.beside_weights),
            );
            None
        }
    }
}

#[cfg(test)]
#[path = "dflash_fit_tests.rs"]
mod tests;
