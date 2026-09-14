// SPDX-License-Identifier: AGPL-3.0-only
//
// Cross-process persistence for the Tier-2 [`RuleLevelCache`] — issue #918.
//
// WHY THE *RULE*-LEVEL CACHE AND NOT THE COMPILED GRAMMAR
// -------------------------------------------------------
// #918 reports "~3.4 s cold grammar compile per new tool schema". The
// CPU measurement that motivated this module (M5 Max, release, Qwen3
// ByteLevel-BPE tokenizer, 151,669-token vocabulary, the coherency
// gate's `get_weather` schema through `compile_qwen3_coder_tool_grammar`)
// shows the cost is per *process*, not per schema:
//
//   grammar construction (EBNF + parse + normalize + optimize)   5.5 ms
//   mask prewarm, first grammar in the process (123 masks)     621.2 ms
//   mask prewarm, same schema again                              0.0 ms
//   mask prewarm, a DIFFERENT schema (new tool + field names)    16.0 ms
//
// The second distinct schema costs ~2.5% of the first because
// [`RuleLevelCache`] keys masks *structurally* — every JSON tool schema
// reuses the same string/number/whitespace/punctuation sub-rules. So a
// disk cache keyed by (tokenizer, schema) would only ever help a repeat
// of the *same* schema, and would miss the reuse that already carries
// 97% of the load. Persisting the rule-level entries instead makes a
// fresh process warm for schemas it has never seen.
//
// Scaling that ~600 ms to the H100 host in the #918 report (248,320
// tokens, ~1.64x the vocabulary, a slower host core than the M5) lands
// on the ~3.2 s of cold mask generation the issue still has open.
//
// FORMAT
// ------
// A single little-endian binary file. The header pins everything a
// cached mask depends on — format version, `usize` width (the accepted
// bitset is written as raw `BitVec` words), the tokenizer fingerprint
// and the vocabulary size — and a trailing FNV-1a digest covers the
// whole body, so a truncated or corrupt file is a cache MISS rather
// than a wrong mask. Any mismatch returns `Ok(0)`: the caller recomputes.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use bitvec::order::Lsb0;
use bitvec::vec::BitVec;

use super::mask::{AdaptiveTokenMask, StoreType};
use super::rule_cache::{RuleLevelCache, RuleMaskKey};

/// File magic — `ATLAS` + "grammar masks", version 1.
const MAGIC: &[u8; 8] = b"ATLASGM1";
/// Bumped whenever the encoding or the mask semantics change, so a
/// snapshot written by an older build is a miss, never a mis-decode.
const FORMAT_VERSION: u32 = 1;

/// Errors from reading or writing a mask snapshot. A *stale* or
/// corrupt snapshot is NOT an error — it is a miss (`Ok(0)`); only
/// genuine I/O failures surface here.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("mask snapshot I/O failed: {0}")]
    Io(String),
}

impl From<std::io::Error> for SnapshotError {
    fn from(e: std::io::Error) -> Self {
        SnapshotError::Io(e.to_string())
    }
}

/// The identity a snapshot is only valid for: the tokenizer it was
/// computed against, plus that tokenizer's vocabulary size.
///
/// A mask is a partition of the sorted decoded vocabulary, so a
/// different tokenizer (or a different `vocab_size` cap over the same
/// tokenizer) invalidates every entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotIdentity {
    /// [`crate::tokenizer::TokenizerInfo::fingerprint`].
    pub tokenizer_fingerprint: u64,
    /// [`crate::tokenizer::TokenizerInfo::vocab_size`].
    pub vocab_size: usize,
}

/// Bits per `usize` on this target — part of the header because the
/// accepted bitset is written as raw `BitVec<usize, Lsb0>` words.
const fn usize_bits() -> u32 {
    usize::BITS
}

// ── FNV-1a (64-bit) ────────────────────────────────────────────────
//
// A fixed-seed, dependency-free digest. `ahash`'s default state is
// randomized per process, which is exactly wrong for a value that has
// to be stable across restarts.

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold `bytes` into an FNV-1a accumulator.
pub fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// FNV-1a over `bytes`, starting from the standard offset basis.
pub fn fnv1a_of(bytes: &[u8]) -> u64 {
    fnv1a(FNV_OFFSET, bytes)
}

// ── Encoding ───────────────────────────────────────────────────────

/// A byte sink that digests everything it is handed, so the trailing
/// checksum never needs a second pass over the buffer.
struct Writer {
    buf: Vec<u8>,
    digest: u64,
}

impl Writer {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            digest: FNV_OFFSET,
        }
    }
    fn bytes(&mut self, b: &[u8]) {
        self.digest = fnv1a(self.digest, b);
        self.buf.extend_from_slice(b);
    }
    fn u8(&mut self, v: u8) {
        self.bytes(&[v]);
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.bytes(&v.to_le_bytes());
    }
    fn i32_slice(&mut self, v: &[i32]) {
        self.u32(v.len() as u32);
        for &x in v {
            self.i32(x);
        }
    }
}

/// A checked byte source. Every read is bounds-guarded; a short read
/// yields `None`, which the caller turns into a cache miss.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let out = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(out)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn i32(&mut self) -> Option<i32> {
        Some(i32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn i32_vec(&mut self) -> Option<Vec<i32>> {
        let n = self.u32()? as usize;
        let mut out = Vec::with_capacity(n.min(1 << 20));
        for _ in 0..n {
            out.push(self.i32()?);
        }
        Some(out)
    }
}

fn store_tag(t: StoreType) -> u8 {
    match t {
        StoreType::Accepted => 0,
        StoreType::Rejected => 1,
        StoreType::AcceptedBitset => 2,
    }
}

fn store_from_tag(tag: u8) -> Option<StoreType> {
    match tag {
        0 => Some(StoreType::Accepted),
        1 => Some(StoreType::Rejected),
        2 => Some(StoreType::AcceptedBitset),
        _ => None,
    }
}

/// Serialize `entries` into the snapshot byte format.
pub fn encode(
    identity: SnapshotIdentity,
    entries: &[(RuleMaskKey, Arc<AdaptiveTokenMask>)],
) -> Vec<u8> {
    let mut w = Writer::new();
    w.bytes(MAGIC);
    w.u32(FORMAT_VERSION);
    w.u32(usize_bits());
    w.u64(identity.tokenizer_fingerprint);
    w.u64(identity.vocab_size as u64);
    w.u64(entries.len() as u64);
    for (key, mask) in entries {
        w.u64(key.fsm_hash);
        w.i32(key.fsm_new_node_id);
        w.i32(key.state_cnt);
        w.i32(key.edge_cnt);
        w.u8(store_tag(mask.store_type));
        w.i32_slice(&mask.accepted_indices);
        w.i32_slice(&mask.rejected_indices);
        w.i32_slice(&mask.uncertain_indices);
        let words = mask.accepted_bitset.as_raw_slice();
        w.u64(mask.accepted_bitset.len() as u64);
        w.u64(words.len() as u64);
        for &word in words {
            w.u64(word as u64);
        }
    }
    let digest = w.digest;
    w.buf.extend_from_slice(&digest.to_le_bytes());
    w.buf
}

/// Parse a snapshot. Returns `None` for ANY reason the bytes are not
/// usable under `identity` — wrong magic, older format, different
/// `usize` width, different tokenizer, truncation, checksum mismatch.
/// Those are cache misses, not failures.
pub fn decode(
    bytes: &[u8],
    identity: SnapshotIdentity,
) -> Option<Vec<(RuleMaskKey, Arc<AdaptiveTokenMask>)>> {
    let body_len = bytes.len().checked_sub(8)?;
    let (body, tail) = bytes.split_at(body_len);
    if fnv1a_of(body) != u64::from_le_bytes(tail.try_into().ok()?) {
        return None;
    }
    let mut r = Reader::new(body);
    if r.take(8)? != MAGIC
        || r.u32()? != FORMAT_VERSION
        || r.u32()? != usize_bits()
        || r.u64()? != identity.tokenizer_fingerprint
        || r.u64()? != identity.vocab_size as u64
    {
        return None;
    }
    let count = r.u64()? as usize;
    let mut out = Vec::with_capacity(count.min(1 << 16));
    for _ in 0..count {
        let key = RuleMaskKey {
            fsm_hash: r.u64()?,
            fsm_new_node_id: r.i32()?,
            state_cnt: r.i32()?,
            edge_cnt: r.i32()?,
        };
        let store_type = store_from_tag(r.u8()?)?;
        let accepted_indices = r.i32_vec()?;
        let rejected_indices = r.i32_vec()?;
        let uncertain_indices = r.i32_vec()?;
        let bit_len = r.u64()? as usize;
        let word_count = r.u64()? as usize;
        let mut words: Vec<usize> = Vec::with_capacity(word_count.min(1 << 20));
        for _ in 0..word_count {
            words.push(r.u64()? as usize);
        }
        let mut accepted_bitset: BitVec<usize, Lsb0> = BitVec::from_vec(words);
        if accepted_bitset.len() < bit_len {
            return None;
        }
        accepted_bitset.truncate(bit_len);
        out.push((
            key,
            Arc::new(AdaptiveTokenMask {
                store_type,
                accepted_indices,
                rejected_indices,
                accepted_bitset,
                uncertain_indices,
            }),
        ));
    }
    // Trailing bytes mean the writer and reader disagree about the
    // format; refuse rather than half-trust it.
    if r.pos != body.len() {
        return None;
    }
    Some(out)
}

// ── File-level helpers ─────────────────────────────────────────────

/// Write `cache`'s entries to `path`, atomically (temp file + rename)
/// so a concurrent reader never observes a half-written snapshot.
///
/// At most `max_entries` are written, taken from the MOST-recently-used
/// end of the cache's LRU order. The rule cache is bounded by a memory
/// budget (~1/3 of a GiB by default), and a snapshot that large has no
/// business sitting next to a checkpoint: measured mask footprint is
/// ~19 KB/mask at a 151,669-token vocabulary, so the cap is what keeps
/// the file in the tens of megabytes.
///
/// Returns the number of entries written.
pub fn save_to_file(
    cache: &RuleLevelCache,
    identity: SnapshotIdentity,
    path: &Path,
    max_entries: usize,
) -> Result<usize, SnapshotError> {
    let all = cache.entries();
    let entries = &all[all.len().saturating_sub(max_entries)..];
    let bytes = encode(identity, entries);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    // The pid keeps two servers starting at once from colliding on the
    // temp name; the rename is what makes the visible file atomic.
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(entries.len())
}

/// Load `path` into `cache`. Returns the number of entries imported —
/// `0` when the file is absent, stale, or corrupt (a miss).
///
/// Entries already present in `cache` win: [`RuleLevelCache::add`]
/// rejects a duplicate key, so loading never overwrites a mask this
/// process computed itself.
pub fn load_from_file(
    cache: &RuleLevelCache,
    identity: SnapshotIdentity,
    path: &Path,
) -> Result<usize, SnapshotError> {
    let mut bytes = Vec::new();
    match std::fs::File::open(path) {
        Ok(mut f) => f.read_to_end(&mut bytes)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e.into()),
    };
    let Some(entries) = decode(&bytes, identity) else {
        return Ok(0);
    };
    let mut imported = 0usize;
    for (key, mask) in entries {
        if cache.add(key, mask) {
            imported += 1;
        }
    }
    Ok(imported)
}

#[cfg(test)]
#[path = "mask_snapshot_tests.rs"]
mod tests;
