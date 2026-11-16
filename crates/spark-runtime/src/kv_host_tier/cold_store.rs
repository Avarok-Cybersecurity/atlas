// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/distributed/storage_controllers/eviction_controller.py (commit 3aefae1, Apache-2.0; L1 eviction demotes to a slower L2 backend); no code copied.

//! Responsibility: the seam between the host tier and a slower tier below it (lane L27's disk
//! tier, `disk.rs`).
//!
//! When the host tier evicts a READY block and a [`ColdStore`] is attached, it hands the block's
//! bytes (exactly `block_bytes`, all layers, K then V, the host-RAM image) and its tokens to
//! [`ColdStore::demote_with_tokens`]. A later lookup that misses in host RAM but finds the key
//! among the cold keys asks [`ColdStore::promote_with_tokens`] for the same bytes; the block
//! becomes visible only after the whole image was read into a fresh host run.
//!
//! Identity: the key chains the namespace (model fingerprint), the adapter, the parent key and
//! the tokens through SHA-256, so a record of another model or another prefix can never answer
//! for this key. The token-aware methods let a store that keeps tokens compare them as the host
//! index does; their defaults fall back to the key-only contract methods (3.7.5).

use anyhow::Result;

use super::fingerprint::BlockKey;

/// A tier below host RAM (contract 3.7.5, plus two defaulted token-aware methods). Called with
/// the host tier's lock held, from the scheduler thread: implementations should be quick and
/// must not call back into the tier.
pub trait ColdStore: Send + Sync {
    /// Keep block `k`'s bytes `b` (overwrite an older copy). Errors are logged by the tier and
    /// the block is dropped as if no cold store existed.
    fn demote(&self, k: &BlockKey, b: &[u8]) -> Result<()>;
    /// Fill `out` (exactly one block) with the bytes of `k` as demoted. `Ok(false)` when the
    /// store does not hold `k` for the current namespace or holds a block of another size.
    fn promote(&self, k: &BlockKey, out: &mut [u8]) -> Result<bool>;
    /// Keys of every block of namespace `ns_fp` already in the store (a warm restart). Also names
    /// the namespace later demotes are written under.
    fn adopt_on_restart(&self, ns_fp: u64) -> Result<Vec<BlockKey>>;
    /// [`Self::demote`] with the block's tokens, for a store that keeps and checks them.
    fn demote_with_tokens(&self, k: &BlockKey, tokens: &[u32], b: &[u8]) -> Result<()> {
        let _ = tokens;
        self.demote(k, b)
    }
    /// [`Self::promote`] that refuses (`Ok(false)`) a block whose stored tokens differ from
    /// `tokens`, for a store that keeps them.
    fn promote_with_tokens(&self, k: &BlockKey, tokens: &[u32], out: &mut [u8]) -> Result<bool> {
        let _ = tokens;
        self.promote(k, out)
    }
}
