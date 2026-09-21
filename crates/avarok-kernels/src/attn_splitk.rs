// SPDX-License-Identifier: AGPL-3.0-only

//! The paged-decode attention SPLIT-K policy: how many KV splits a launch
//! uses, as a pure function of CONFIGURATION.
//!
//! # The defect (#928)
//!
//! `crates/spark-model/.../decode/run_paged_decode.rs` imported
//! `avarok_core::device::sm121::NUM_SMS` — the GB10 constant, 48 — and picked
//!
//! ```text
//! current_ctas = num_q_heads * split_ref_seqs(num_seqs, max_decode_seqs);
//! num_splits   = if current_ctas >= NUM_SMS { 1 } else { NUM_SMS / current_ctas };
//! ```
//!
//! On Qwen3.8-27B at `--max-batch-size 16` that is `24 * 16 = 384 >= 48`, so
//! `num_splits` was 1 at EVERY batch size including C=1. nsys on 1xH100 80GB
//! HBM3 (round 13 cell T1N) has the receipt: `paged_decode_attn_fp8` runs
//! `grid=(24,1,1)` — 24 CTAs on 132 SMs — 231.51 us/launch for 9.93 MB of KV,
//! i.e. 42.9 GB/s = **1.28% of HBM**, and with the BF16-KV sibling the pair is
//! 3.79 ms of a 16.69 ms C=1 decode step (22.7%). The source comment
//! dismissing occupancy here ("attention occupancy is NOT the long-ctx
//! bottleneck") records a GB10 A/B on a 48-SM part.
//!
//! Two things were wrong and both are fixed here: the SM count is a property
//! of the compiled TARGET (`kernels/<hw>/HARDWARE.toml` `[hardware] sm_count`,
//! baked as [`crate::TARGET_SM_COUNT`]), and the occupancy the rule should
//! size for is the SINGLE-STREAM shape, which is the one that starves.
//!
//! # The determinism invariant — why this module is pure
//!
//! The online-softmax split-merge is NON-ASSOCIATIVE. If `num_splits` moved
//! with the runtime co-batched count, one sequence would traverse a different
//! reduction tree alone than beside fifteen others and flip its temp-0 argmax
//! — the nondeterminism `split_ref_seqs` was introduced to stop
//! (`tasks/determinism_investigation.md`). So [`SplitkPolicy::Auto`] and
//! [`SplitkPolicy::Pinned`] read `(sm_count, num_q_heads, max_decode_seqs)`
//! and NOTHING else: the split count is fixed for the life of a serve.
//! [`SplitkPolicy::Legacy`] is the pre-#928 rule preserved verbatim, including
//! its dependence on `split_ref_seqs`, because every target but Hopper still
//! runs it and "unchanged" has to mean unchanged.
//!
//! Short contexts are handled INSIDE the kernel
//! (`kernels/hopper/common/paged_decode_splitk_hopper.cuh`,
//! `PD_MIN_KV_PER_SPLIT`), from each sequence's own `seq_len`: the host may not
//! branch on `seq_lens`, which is device memory and behind a captured CUDA
//! graph, and a per-sequence rule stays co-batch invariant where a per-batch
//! one would not.

/// Waves of CTAs the `auto` policy aims to put on the device at the
/// single-stream shape.
///
/// TWO, not one: the attention CTAs are memory-latency bound (a paged K/V
/// gather, a shuffle reduction and an `__expf` per position), so one CTA per
/// SM leaves the SM stalled on loads. Two resident waves is the smallest
/// number that lets one cover the other's misses, and it is also where the
/// split stops being free — every extra split is another partial for the
/// reduce to merge and another eighth-of-a-CTA of Q load to repeat. Round 14
/// measures the curve: the microtest prints GB/s for `num_splits` 1/2/4/6 at
/// three context lengths.
pub const SPLITK_TARGET_WAVES: u32 = 2;

/// Hard ceiling on the split count, and the bound the split-K workspace is
/// sized against.
///
/// A cap rather than a pure occupancy answer because the workspace is
/// `rows * num_q_heads * num_splits * (head_dim + 2)` F32 — linear in this
/// number — and because a model with very few q heads (an MQA decode head, a
/// draft model) would otherwise ask for a split per KV block. 16 covers
/// `2 * 148 / 24 = 13` on the widest declared target
/// (`kernels/b200`, 148 SMs) with room, and an explicit
/// `attn_decode_splitk = N` is clamped to it.
pub const MAX_DECODE_SPLITS: u32 = 16;

/// How a target picks its paged-decode split count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitkPolicy {
    /// The pre-#928 rule: `sm_count / (num_q_heads * split_ref_seqs)`, or 1
    /// when that product already covers the device. Reads the runtime
    /// reference batch, which is why it is not the default anywhere new.
    Legacy,
    /// Fill [`SPLITK_TARGET_WAVES`] waves at the SINGLE-STREAM shape:
    /// `clamp(ceil(WAVES * sm_count / num_q_heads), 1, MAX_DECODE_SPLITS)`.
    Auto,
    /// An operator-pinned count, clamped to `1..=MAX_DECODE_SPLITS`. `0` and
    /// `off` arrive here as `Pinned(1)` — one split IS no split-K.
    Pinned(u32),
}

impl SplitkPolicy {
    /// The spelling this policy round-trips through [`parse`], for the serve
    /// log's `target defaults (<hw>): …` line.
    pub fn label(self) -> String {
        match self {
            SplitkPolicy::Legacy => "legacy".to_string(),
            SplitkPolicy::Auto => "auto".to_string(),
            SplitkPolicy::Pinned(n) => n.to_string(),
        }
    }
}

/// `legacy` | `auto` | `0`/`off`/`false`/`no` | a decimal count.
///
/// `None` for anything else — the caller keeps the target's declaration rather
/// than guessing, because a typo that silently resolved to `auto` would arm a
/// geometry change on a card with no receipt for it.
pub fn parse(spelling: &str) -> Option<SplitkPolicy> {
    match spelling.trim().to_ascii_lowercase().as_str() {
        "legacy" => Some(SplitkPolicy::Legacy),
        "auto" => Some(SplitkPolicy::Auto),
        "0" | "off" | "false" | "no" => Some(SplitkPolicy::Pinned(1)),
        other => other
            .parse::<u32>()
            .ok()
            .map(|n| SplitkPolicy::Pinned(n.clamp(1, MAX_DECODE_SPLITS))),
    }
}

/// The target's declaration, overridden by `AVAROK_ATTN_DECODE_SPLITK`.
///
/// Returns `(policy, came_from_env)`. THE one resolution rule: both the
/// dispatch (via `spark_model`'s `target_defaults::resolve`) and the buffer
/// arena (via [`policy_from_env`]) call this, so the split count a launch uses
/// and the split count the workspace was sized for cannot disagree.
pub fn resolve_policy(declared: &str, env_raw: Option<&str>) -> (SplitkPolicy, bool) {
    let declared = parse(declared).unwrap_or(SplitkPolicy::Legacy);
    match env_raw.and_then(parse) {
        Some(p) => (p, true),
        None => (declared, false),
    }
}

/// [`resolve_policy`] against the process environment and this binary's baked
/// declaration.
///
/// For the buffer-sizing call site, which lives below `spark-model` in the
/// dependency graph and so cannot reach its resolver. It is the SAME pure
/// rule, not a second one; the serve log still reports the value once, from
/// `spark-model`.
pub fn policy_from_env() -> SplitkPolicy {
    resolve_policy(
        crate::TARGET_DEFAULTS.attn_decode_splitk,
        std::env::var("AVAROK_ATTN_DECODE_SPLITK").ok().as_deref(),
    )
    .0
}

/// `clamp(ceil(WAVES * sm_count / num_q_heads), 1, MAX_DECODE_SPLITS)`.
///
/// The single-stream occupancy answer: at C=1 the grid is
/// `(num_q_heads, num_splits, 1)`, so this is the split count that puts
/// `WAVES * sm_count` CTAs on the device with one sequence in flight. At a
/// wider batch the same count over-subscribes — total WORK is unchanged, only
/// its partition is — which is the trade this policy takes deliberately: a
/// C=1 step is 24 CTAs without it and a C=16 step is already 3 waves with it.
pub fn auto_splits(sm_count: u32, num_q_heads: u32) -> u32 {
    let heads = num_q_heads.max(1);
    let target = SPLITK_TARGET_WAVES.saturating_mul(sm_count.max(1));
    target.div_ceil(heads).clamp(1, MAX_DECODE_SPLITS)
}

/// The pre-#928 rule, preserved verbatim for every target that still declares
/// it. `ref_seqs` is `split_ref_seqs(num_seqs, max_decode_seqs)`.
pub fn legacy_splits(sm_count: u32, num_q_heads: u32, ref_seqs: u32) -> u32 {
    let current_ctas = num_q_heads.max(1).saturating_mul(ref_seqs.max(1));
    if current_ctas >= sm_count {
        1
    } else {
        sm_count / current_ctas
    }
}

/// The split count for a launch.
///
/// ⚠️ `legacy_ref_seqs` is read by [`SplitkPolicy::Legacy`] ONLY. Every other
/// arm is a pure function of configuration, which is the determinism invariant
/// this module exists to hold — see the module header, and
/// `the_auto_split_count_does_not_move_with_the_co_batched_count`.
pub fn num_splits(
    policy: SplitkPolicy,
    sm_count: u32,
    num_q_heads: u32,
    legacy_ref_seqs: u32,
) -> u32 {
    match policy {
        SplitkPolicy::Legacy => legacy_splits(sm_count, num_q_heads, legacy_ref_seqs),
        SplitkPolicy::Auto => auto_splits(sm_count, num_q_heads),
        SplitkPolicy::Pinned(n) => n.clamp(1, MAX_DECODE_SPLITS),
    }
}

/// `[o[head_dim], m, l]` slots the split-K workspace must hold.
///
/// The split-K kernel addresses
/// `((seq * num_q_heads) + head) * num_splits + split`, so the arena has to
/// cover the WIDEST batch the decode-metadata layout will accept
/// (`DecodeMetaLayout::rows()`), not the pinned max batch — a short
/// allocation here is an out-of-bounds device write, silently.
///
/// `Legacy` keeps its old constant bound and its old arena: that rule picks
/// `sm_count / (num_q_heads * max(max_batch, num_seqs))`, so
/// `num_seqs * num_q_heads * num_splits <= sm_count` for every batch, which is
/// exactly what `sizes.rs` allocated before #928.
pub fn workspace_slots(
    policy: SplitkPolicy,
    sm_count: u32,
    num_q_heads: u32,
    max_rows: u32,
    legacy_ref_seqs: u32,
) -> u32 {
    match policy {
        SplitkPolicy::Legacy => sm_count.max(1),
        other => max_rows
            .max(1)
            .saturating_mul(num_q_heads.max(1))
            .saturating_mul(num_splits(other, sm_count, num_q_heads, legacy_ref_seqs)),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// GQA-PACKED PAGED DECODE — the other half of the same launch geometry.
// ════════════════════════════════════════════════════════════════════════════
//
// Hosted in this module rather than a new one because it answers the same
// question the split policy answers — HOW MANY CTAs a paged-decode launch
// gets, and over what — and because the two are coupled: the packed kernel
// divides the CTA count by [`DECODE_GQA_PACK_WIDTH`], which is only safe where
// the split policy has already resolved one split. One module, one answer.
//
// # What the packed kernel is
//
// `paged_decode_attn{,_fp8}` launch `grid (num_q_heads, num_seqs)`. With
// `gqa_ratio = num_q_heads / num_kv_heads` heads sharing a KV head, the
// `gqa_ratio` CTAs of a group each walk that head's WHOLE K and V
// independently — `kv_head = q_head / gqa_ratio` is all that separates their
// address streams, and it is equal across the group. The packed twins
// (`kernels/gb10/common/paged_decode_attn_{fp8,bf16}_gqa.cu`) put the whole
// group in ONE CTA, hold its query vectors in registers, and read each K and
// V row once: `grid (num_kv_heads, num_seqs)`.
//
// # Why it is restricted to the non-split arm
//
// Packing divides the CTA count by the pack width, so a shape that needed
// split-K to fill the device needs MORE splits after packing, derived from
// `num_kv_heads` instead of `num_q_heads`. A different split count is a
// different partition of the KV range, hence a different online-softmax merge
// tree, hence different output bytes — the exact non-associativity the split
// policy above is pinned against. The non-split arm has no such coupling, so
// the packed kernel there is BIT-IDENTICAL to the unpacked one and needs no
// gate record re-opened. Extending packing to the split arm is a separate,
// numerics-visible change and is deliberately not made here.

// # The fused last-CTA reduce, CONSIDERED AND DECLINED (2026-09-20)
//
// The obvious companion change is to fold `paged_decode_attn_reduce_fp8` into
// the split kernel — an arrival counter per `(seq, head)`, last CTA merges —
// removing one launch per layer per token. It is not done, and the reason is
// this module's own arithmetic rather than a difficulty:
//
// * On every target that declares [`SplitkPolicy::Legacy`] — which is all of
//   them but Hopper — `legacy_splits(48, 24, ref_seqs)` is `1` for every
//   `ref_seqs >= 2`, and `ref_seqs` is `max(max_batch_size, num_seqs)`. So on
//   GB10 the reduce kernel is LAUNCHED ONLY AT `--max-batch 1`. Every
//   co-batched shape a campaign measures takes the non-split arm, where there
//   is no second launch to remove.
// * The saving it targets is a LAUNCH, and the launch it removes is inside a
//   captured CUDA graph, where launch cost is already the cheap case. What it
//   adds is a `__threadfence` and an atomic on every split CTA, plus a merge
//   serialised into the last arriver while the rest of the device drains.
// * The counter has nowhere to live inside this change's blast radius. The
//   split-K workspace is sized by [`workspace_slots`] and allocated in
//   `spark-runtime`'s buffer arena, so a counter region means changing the
//   arena's layout; a module-scope `__device__` array instead would need a
//   compile-time capacity bound with no configuration to derive it from, and
//   would be shared — unsynchronised — by any two launches of the same kernel
//   in flight on different streams.
//
// Worth recording that it would be BIT-IDENTICAL if done: the separate reduce
// merges splits in INDEX order `0..num_splits`, and a last-CTA reduction that
// keeps that order does the same additions in the same sequence. The reason
// not to do it is that it is small, and small on an arm GB10 does not take.

/// Query heads one packed CTA carries.
///
/// A COMPILE-TIME shape on both sides: the kernels size `q_reg`/`o_reg`
/// register arrays from `#define PD_GQA`, because a runtime bound would put
/// them in local memory and defeat the change. This constant is the Rust
/// spelling of that `#define`, and
/// `cuda_sources_declare_the_pack_width_rust_dispatches_on` fails if the two
/// drift.
///
/// 6 is the Qwen3.8-27B decode ratio (`nq = 24`, `nkv = 4`). A model with any
/// other ratio is REFUSED by [`gqa_pack_shape_ok`] and keeps the unpacked
/// kernel; it is not approximated.
pub const DECODE_GQA_PACK_WIDTH: u32 = 6;

/// The head dim the packed kernels are compiled for.
///
/// Same reason as the split-K siblings' [`crate::attn_splitk`] head-dim pin:
/// the sources fix `PD_HDIM` at 256 and derive every lane's element count from
/// it, so a 128- or 512-wide head would have every lane stride past its own
/// row.
pub const DECODE_GQA_PACK_HEAD_DIM: u32 = 256;

/// The packed kernels are OFF unless explicitly armed.
///
/// Declared here rather than in `kernels/<hw>/HARDWARE.toml` because there is
/// no receipt for this arm on any target yet: it is a launch-geometry change
/// whose payoff (`DECODE_GQA_PACK_WIDTH`x fewer KV load instructions) trades
/// against a measured occupancy cost — ptxas reports 227 registers for the FP8
/// twin and 243 for the BF16 one, zero spills, i.e. ONE resident CTA per SM,
/// against 48/56 registers and up to five for the unpacked kernels. That trade
/// can only be settled by an A/B on the box. `AVAROK_ATTN_DECODE_GQA_PACK=1`
/// is how that A/B is run — the same mechanism `AVAROK_ATTN_DECODE_SPLITK=auto`
/// uses for the policy above. A target that wins the A/B moves this to a
/// `[defaults]` row with the receipt beside it.
pub const DECODE_GQA_PACK_DECLARED: bool = false;

/// `1`/`on`/`true`/`yes` | `0`/`off`/`false`/`no`.
///
/// `None` for anything else, so a typo keeps the declaration rather than
/// silently arming (or disarming) a geometry change — same rule as [`parse`].
pub fn parse_gqa_pack(spelling: &str) -> Option<bool> {
    match spelling.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Some(true),
        "0" | "off" | "false" | "no" => Some(false),
        _ => None,
    }
}

/// [`DECODE_GQA_PACK_DECLARED`], overridden by `AVAROK_ATTN_DECODE_GQA_PACK`.
///
/// Pure in its arguments so the resolution is testable without touching the
/// process environment; [`gqa_pack_enabled`] is the one site that reads it.
pub fn resolve_gqa_pack(declared: bool, env_raw: Option<&str>) -> bool {
    env_raw.and_then(parse_gqa_pack).unwrap_or(declared)
}

/// The armed/disarmed answer for this process, resolved ONCE.
///
/// `OnceLock`, not a per-call `env::var`: this is read on the decode path, per
/// layer per token, and `spark_model`'s `hot_path_env_guards` exists because
/// that read takes the process-wide environment lock and costs 5.76 us at 16
/// threads. Resolving once is also what CUDA graph capture requires — the
/// kernel a captured graph holds cannot depend on an environment read that
/// might answer differently on replay.
pub fn gqa_pack_enabled() -> bool {
    static ARMED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ARMED.get_or_init(|| {
        resolve_gqa_pack(
            DECODE_GQA_PACK_DECLARED,
            std::env::var("AVAROK_ATTN_DECODE_GQA_PACK").ok().as_deref(),
        )
    })
}

/// Whether this launch's SHAPE is one the packed kernels can serve.
///
/// Every condition is a hard precondition of the kernel, not a preference:
///
/// * `num_kv_heads > 0` — it is the grid's x extent and the divisor below.
/// * `num_q_heads == num_kv_heads * DECODE_GQA_PACK_WIDTH` — the kernel
///   recovers its heads as `kv_head * PD_GQA + h`, the exact inverse of the
///   unpacked kernel's `q_head / gqa_ratio`. Any other ratio reads and writes
///   the wrong heads, silently, so it is refused rather than clamped. Stated
///   as a MULTIPLICATION on purpose: `num_q_heads / num_kv_heads` truncates,
///   so 25 q heads over 4 kv heads would pass a division test and then have
///   the last head read past the group.
/// * `head_dim == DECODE_GQA_PACK_HEAD_DIM` — see that constant.
///
/// The caller keeps the unpacked kernel when this is false. That is a
/// documented route, not a fallback that hides an error: the unpacked kernel
/// is the one every target has been serving.
pub fn gqa_pack_shape_ok(num_q_heads: u32, num_kv_heads: u32, head_dim: u32) -> bool {
    num_kv_heads > 0
        && num_q_heads == num_kv_heads.saturating_mul(DECODE_GQA_PACK_WIDTH)
        && head_dim == DECODE_GQA_PACK_HEAD_DIM
}

/// CTAs a packed launch puts on the device, against the unpacked count.
///
/// Exposed because it is the whole trade and it belongs next to the constant
/// that sets it, not only in a comment: `(nkv, num_seqs)` against
/// `(nq, num_seqs)` is a `DECODE_GQA_PACK_WIDTH`-fold cut in CTAs for an
/// unchanged amount of work. At the GB10 shape (`nkv = 4`, 48 SMs) the packed
/// grid stops under-filling the device at `num_seqs >= 12`; below that the
/// launch leaves SMs idle, which is the cost side of the A/B this is gated
/// behind.
pub fn gqa_pack_ctas(num_kv_heads: u32, num_seqs: u32) -> u32 {
    num_kv_heads.saturating_mul(num_seqs)
}

#[cfg(test)]
#[path = "attn_splitk_tests.rs"]
mod tests;
