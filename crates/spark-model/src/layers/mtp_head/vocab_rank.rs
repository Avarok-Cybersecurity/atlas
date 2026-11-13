// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0. Source: tools/convert/proposal.py:21-42 (select_shortlist, the forced-id rule) and the data file tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64 (token counts by NInfer authors; source corpus not published), turned into data/mtp_vocab_rank_qwen35.u32 by data/gen_mtp_vocab_rank.py. Changes: translated from Python to Rust, the selection extended to any head width with the table completed in id order, a tokenizer identity check, by Atlas contributors.

//! Responsibility: the frequency-ranked MTP draft vocabulary
//! (`mtp_vocab_rank = "freq"`).
//!
//! The draft-only NVFP4 head (`factory/lm_head_setup.rs`) holds
//! `--mtp-vocab` rows. With `id` (the baseline) those are the first N token
//! ids; with `freq` they are the N most frequent tokens of the shipped table,
//! with the tokenizer's added tokens forced in. A draft can never propose a
//! token outside its head, and one miss rejects that position and every later
//! draft of the chain, so the row choice is an acceptance lever. Held-out
//! coverage at 100,000 rows (the Atlas default): 98.1% ranked vs 86.2% for
//! the first-N ids (`data/mtp_vocab_rank_qwen35.json`, `coverage`).
//!
//! What lives here:
//! * the shipped table and its provenance, parsed once;
//! * the tokenizer identity check: the table's counts were made with the
//!   Qwen3.6 tokenizer, and the generator proved the Qwen3.5 tokenizer decodes
//!   every manifest (id, token) pair identically; the table is used only when
//!   the served model's `tokenizer.json` hashes (FNV-1a-64) to a verified file
//!   and its vocabulary size is one the provenance accepts;
//! * [`select_rows`], the head's row -> token-id map for any width;
//! * the registry that hands a ranked head's map (built by
//!   [`build_ranked_draft_head`], `ranked_head.rs`) to every `MtpHead` built on
//!   that head;
//! * the grammar-mask translation and the `atlas_mtp_draft_*` series.
//!
//! Only drafts change: every draft is verified by the main LM head, so the
//! emitted tokens are those of the id-ranked head (greedy SHA equal is the
//! row's receipt).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, KernelHandle};

pub(crate) use super::ranked_head::build_ranked_draft_head;

/// The shipped ranking: u32 little-endian token ids, most frequent first.
const TABLE_BYTES: &[u8] = include_bytes!("../../../data/mtp_vocab_rank_qwen35.u32");
/// Its provenance: source, forced ids, verified tokenizer hashes, coverage.
const PROVENANCE_JSON: &str = include_str!("../../../data/mtp_vocab_rank_qwen35.json");

/// Device kernel module and entry point that maps head rows to token ids
/// (`kernels/ada/common/draft_vocab_remap.cu`; Ada only in this sprint).
pub const REMAP_MODULE: &str = "draft_vocab_remap";
pub const REMAP_KERNEL: &str = "draft_vocab_remap";

/// Rows gathered and quantized per chunk when the ranked head is built: a
/// BF16 staging buffer of `CHUNK_ROWS x hidden` (20 MB at hidden 2560, 40 MB
/// at 5120) instead of a full `rows x hidden` copy at load time.
pub const CHUNK_ROWS: usize = 4096;

/// Which rows the draft-only head holds (the `mtp_vocab_rank` row's values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankMode {
    /// The first N token ids (the pre-sprint head).
    Id,
    /// The N most frequent tokens of the shipped table.
    Freq,
}

impl RankMode {
    /// The row value as resolved for this process.
    pub fn resolved() -> Self {
        Self::from_value(crate::layers::ops::port_rows::mtp_vocab_rank::row())
    }

    /// Map a row value (`id`, `freq`) to a mode; anything else is `Id`.
    pub fn from_value(v: &str) -> Self {
        if v == "freq" { Self::Freq } else { Self::Id }
    }
}

/// The shipped table plus the parts of its provenance the runtime checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankTable {
    /// Token ids, most frequent first.
    pub ids: Vec<u32>,
    /// Ids forced into every head (the tokenizer's added tokens).
    pub forced: Vec<u32>,
    /// FNV-1a-64 of every `tokenizer.json` the identity check passed for.
    pub tokenizer_fnv1a64: Vec<u64>,
    /// Vocabulary sizes the table may serve (tokenizer-capped and padded).
    pub vocab_sizes: Vec<usize>,
}

/// FNV-1a 64-bit over `data`; the generator computes the same function.
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Parse a rank file and its provenance JSON.
pub fn parse_table(bytes: &[u8], provenance: &str) -> Result<RankTable> {
    anyhow::ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(4),
        "rank table: {} bytes is not a whole number of u32 ids",
        bytes.len()
    );
    let ids: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let v: serde_json::Value =
        serde_json::from_str(provenance).context("rank table provenance is not JSON")?;
    let rows = v["rows"].as_u64().context("provenance: no `rows`")? as usize;
    anyhow::ensure!(
        rows == ids.len(),
        "rank table holds {} ids, provenance says {rows}",
        ids.len()
    );
    let u32s = |key: &serde_json::Value| -> Vec<u32> {
        key.as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as u32)
                    .collect()
            })
            .unwrap_or_default()
    };
    let forced = u32s(&v["forced_ids"]);
    let check = &v["tokenizer_check"];
    let tokenizer_fnv1a64 = check["tokenizers"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|t| t["fnv1a64"].as_str())
                .filter_map(|s| u64::from_str_radix(s, 16).ok())
                .collect()
        })
        .unwrap_or_default();
    let vocab_sizes = u32s(&check["accepted_vocab_sizes"])
        .into_iter()
        .map(|x| x as usize)
        .collect();
    Ok(RankTable {
        ids,
        forced,
        tokenizer_fnv1a64,
        vocab_sizes,
    })
}

/// The shipped table, parsed once. `None` (with one warning) if the embedded
/// files are inconsistent, which disables the `freq` mode, never the serve.
pub fn shipped() -> Option<&'static RankTable> {
    static TABLE: OnceLock<Option<RankTable>> = OnceLock::new();
    TABLE
        .get_or_init(|| match parse_table(TABLE_BYTES, PROVENANCE_JSON) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::warn!("MTP draft vocab: shipped rank table unusable ({e:#})");
                None
            }
        })
        .as_ref()
}

/// Why the table may not be used with a model, or `Ok` when it may.
pub fn table_applies(
    t: &RankTable,
    vocab: usize,
    tokenizer_fnv: Option<u64>,
) -> Result<(), String> {
    if !t.vocab_sizes.contains(&vocab) {
        return Err(format!(
            "vocabulary {vocab} is not one the table serves ({:?})",
            t.vocab_sizes
        ));
    }
    match tokenizer_fnv {
        None => Err("the served model's tokenizer.json was not found".to_string()),
        Some(h) if !t.tokenizer_fnv1a64.contains(&h) => Err(format!(
            "tokenizer.json (fnv1a64 {h:016x}) is not a tokenizer the table was verified for"
        )),
        Some(_) => Ok(()),
    }
}

/// Lock a process-wide table, ignoring poisoning: every holder below only
/// replaces or reads whole values, so a panic elsewhere cannot leave one torn.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

static MODEL_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Record the directory of the model about to be built (called by
/// `spark-server` right after it resolves the model directory), so the
/// identity check can hash that model's `tokenizer.json`.
pub fn note_model_dir(dir: &Path) {
    *lock(&MODEL_DIR) = Some(dir.to_path_buf());
}

/// FNV-1a-64 of the noted model's `tokenizer.json`, if there is one.
pub(super) fn tokenizer_fingerprint() -> Option<u64> {
    let dir = lock(&MODEL_DIR).clone()?;
    std::fs::read(dir.join("tokenizer.json"))
        .ok()
        .map(|b| fnv1a64(&b))
}

/// The head's row -> token-id map for a head of `n` rows over `vocab` ids.
///
/// Every id in `forced` below `vocab` is included (Cinference's
/// `select_shortlist` forces the special tokens in); the remaining rows take
/// the table's ids in order, skipping ids at or past `vocab` and repeats, and
/// when the table runs out the unused ids follow in ascending order, so any
/// `n <= vocab` is served. The result is unique, `min(n, vocab)` long, and its
/// first rows are the table's prefix. Forced ids come last.
pub fn select_rows(table: &[u32], vocab: usize, n: usize, forced: &[u32]) -> Vec<u32> {
    let n = n.min(vocab);
    let mut forced_set: Vec<u32> = forced
        .iter()
        .copied()
        .filter(|&f| (f as usize) < vocab)
        .collect();
    forced_set.sort_unstable();
    forced_set.dedup();
    if forced_set.len() >= n {
        forced_set.truncate(n);
        return forced_set;
    }
    let mut seen = vec![false; vocab];
    for &f in &forced_set {
        seen[f as usize] = true;
    }
    let want = n - forced_set.len();
    let mut rows = Vec::with_capacity(n);
    for &id in table {
        if rows.len() == want {
            break;
        }
        let i = id as usize;
        if i < vocab && !seen[i] {
            seen[i] = true;
            rows.push(id);
        }
    }
    let mut next = 0usize;
    while rows.len() < want && next < vocab {
        if !seen[next] {
            seen[next] = true;
            rows.push(next as u32);
        }
        next += 1;
    }
    rows.extend_from_slice(&forced_set);
    rows
}

/// Bitmap over `vocab` ids of the tokens a head holds.
pub fn membership(row_to_id: &[u32], vocab: usize) -> Vec<u64> {
    let mut bits = vec![0u64; vocab.div_ceil(64)];
    for &id in row_to_id {
        let i = id as usize;
        if i < vocab {
            bits[i / 64] |= 1u64 << (i % 64);
        }
    }
    bits
}

/// Whether `id` is set in a [`membership`] bitmap.
pub fn is_member(bits: &[u64], id: u32) -> bool {
    let i = id as usize;
    bits.get(i / 64)
        .is_some_and(|w| w & (1u64 << (i % 64)) != 0)
}

/// Fraction of `counts` (indexed by token id) that a head holding `row_to_id`
/// covers: the coverage numbers of the provenance JSON, host only.
pub fn coverage(row_to_id: &[u32], counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let held: u64 = row_to_id
        .iter()
        .filter_map(|&id| counts.get(id as usize))
        .sum();
    held as f64 / total as f64
}

/// A ranked draft head: the row -> token-id map on the host and the device.
#[derive(Debug)]
pub struct RankedHead {
    /// Head row r holds token `row_to_id[r]`.
    pub row_to_id: Vec<u32>,
    /// The same map on the device (`rows` u32), read by the remap kernel.
    pub row_to_id_dev: DevicePtr,
    /// Membership bitmap over the vocabulary (out-of-head accounting and the
    /// grammar mask translation).
    pub in_head: Vec<u64>,
    /// `draft_vocab_remap` handle (non-zero: the head is never built without it).
    pub remap_k: KernelHandle,
}

/// Ranked heads built by the current model build, keyed by the packed weight
/// pointer of the draft head. Cleared at the start of every LM-head setup, so
/// an entry never outlives the build that made it (a recycled device address
/// after a model swap cannot hit a stale map).
static REGISTRY: Mutex<Vec<(u64, Arc<RankedHead>)>> = Mutex::new(Vec::new());

/// Forget every ranked head (start of `setup_lm_heads`).
pub(crate) fn reset_registry() {
    lock(&REGISTRY).clear();
}

/// Record the map of the ranked head whose packed weights start at `weight`.
pub(super) fn register(weight: DevicePtr, head: Arc<RankedHead>) {
    lock(&REGISTRY).push((weight.0, head));
}

/// The ranked map of the draft head whose packed weights start at `weight`,
/// if that head was built ranked by the current model build. Not consuming:
/// a multi-module drafter builds several `MtpHead`s on one head.
pub(crate) fn ranked_for(weight: DevicePtr) -> Option<Arc<RankedHead>> {
    lock(&REGISTRY)
        .iter()
        .find(|(p, _)| *p == weight.0)
        .map(|(_, h)| h.clone())
}

/// Translate an XGrammar token bitmask into the ranked head's row space and
/// take the masked argmax over BF16 logits: row r is allowed iff its token
/// `row_to_id[r]` is. Returns the winning TOKEN id, or `None` when the mask
/// allows no row of this head (the caller proposes the pad draft 0, as the
/// id-ranked path does for an empty mask). First-largest wins on ties, as the
/// id-ranked CPU path does.
pub fn masked_argmax_rows(logits_bf16: &[u8], row_to_id: &[u32], bitmask: &[i32]) -> Option<u32> {
    let rows = (logits_bf16.len() / 2).min(row_to_id.len());
    let mut best: Option<(f32, u32)> = None;
    for (r, &tok) in row_to_id.iter().enumerate().take(rows) {
        let word = tok as usize / 32;
        let allowed = word < bitmask.len() && (bitmask[word] & (1i32 << (tok % 32))) != 0;
        if !allowed {
            continue;
        }
        let hi = u16::from_le_bytes([logits_bf16[2 * r], logits_bf16[2 * r + 1]]);
        let x = f32::from_bits((hi as u32) << 16);
        if best.is_none_or(|(b, _)| x > b) {
            best = Some((x, tok));
        }
    }
    best.map(|(_, t)| t)
}

// ── /metrics: `atlas_mtp_draft_*` (L11) ────────────────────────────────────

static DRAFT_ROWS: AtomicU64 = AtomicU64::new(0);
static OUT_OF_HEAD: AtomicU64 = AtomicU64::new(0);

/// Publish the draft head's row count (one MTP head per serve).
pub(crate) fn set_draft_rows(rows: usize) {
    DRAFT_ROWS.store(rows as u64, Ordering::Relaxed);
}

/// Count one committed token the draft head could not have proposed.
pub(crate) fn note_out_of_head() {
    OUT_OF_HEAD.fetch_add(1, Ordering::Relaxed);
}

/// The `atlas_mtp_draft_*` series, in the `port_metrics` text format.
pub fn render_prometheus(out: &mut String) {
    use std::fmt::Write;
    let _ = writeln!(
        out,
        "# HELP atlas_mtp_draft_vocab_rows Rows of the MTP draft head's vocabulary \
         projection (0 = no MTP head).\n\
         # TYPE atlas_mtp_draft_vocab_rows gauge\n\
         atlas_mtp_draft_vocab_rows {}",
        DRAFT_ROWS.load(Ordering::Relaxed)
    );
    let _ = writeln!(
        out,
        "# HELP atlas_mtp_draft_out_of_head_total Committed tokens seen at an MTP propose \
         that the draft head does not hold.\n\
         # TYPE atlas_mtp_draft_out_of_head_total counter\n\
         atlas_mtp_draft_out_of_head_total {}",
        OUT_OF_HEAD.load(Ordering::Relaxed)
    );
}

#[cfg(test)]
#[path = "vocab_rank_tests.rs"]
mod vocab_rank_tests;
