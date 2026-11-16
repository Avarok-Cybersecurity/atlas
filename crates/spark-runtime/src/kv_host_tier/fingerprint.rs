// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/token_hasher.py:233-262 (commit 3aefae1, Apache-2.0); no code copied.

//! Responsibility: the host tier's block identity. A [`BlockKey`] chains the namespace (the
//! model fingerprint from `ssm_tier/fingerprint.rs`, passed in as an opaque 64-bit value), the
//! adapter id, the previous block's key and this block's tokens through SHA-256, truncated to
//! 128 bits. The radix tree's 64-bit FNV `context_hash` is enough inside a tree that also
//! compares tokens; a hash-only host index should not trust 64 bits, and the tier still stores
//! each block's tokens and compares them on every hit.

use sha2::{Digest, Sha256};

/// 128-bit chained block identity (contract 3.7.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockKey(pub [u8; 16]);

/// Domain separation: a key never equals a hash computed for anything else.
const DOMAIN: &[u8] = b"atlas.kv_host_tier.block.v1";

impl BlockKey {
    /// The key of the block that follows `prev` (None for the first block of a prefix).
    pub fn chain(
        namespace: u64,
        adapter: u32,
        prev: Option<&BlockKey>,
        tokens: &[u32],
    ) -> BlockKey {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        h.update(namespace.to_le_bytes());
        h.update(adapter.to_le_bytes());
        match prev {
            Some(p) => {
                h.update([1u8]);
                h.update(p.0);
            }
            None => h.update([0u8]),
        }
        h.update((tokens.len() as u32).to_le_bytes());
        for t in tokens {
            h.update(t.to_le_bytes());
        }
        let digest = h.finalize();
        let mut out = [0u8; 16];
        out.copy_from_slice(&digest[..16]);
        BlockKey(out)
    }

    /// Keys of every FULL block of `tokens` (a trailing partial block gets none).
    pub fn chain_blocks(
        namespace: u64,
        adapter: u32,
        tokens: &[u32],
        block_size: usize,
    ) -> Vec<BlockKey> {
        if block_size == 0 {
            return Vec::new();
        }
        let mut keys = Vec::with_capacity(tokens.len() / block_size);
        let mut prev: Option<BlockKey> = None;
        for chunk in tokens.chunks_exact(block_size) {
            let k = BlockKey::chain(namespace, adapter, prev.as_ref(), chunk);
            keys.push(k);
            prev = Some(k);
        }
        keys
    }

    /// The first 8 bytes, little endian: a short form for logs.
    pub fn short(&self) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.0[..8]);
        u64::from_le_bytes(b)
    }

    /// Lower-case hex of all 16 bytes (file names, logs).
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(32);
        for b in self.0 {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// Inverse of [`Self::to_hex`].
    pub fn from_hex(s: &str) -> Option<BlockKey> {
        if s.len() != 32 || !s.is_ascii() {
            return None;
        }
        let mut out = [0u8; 16];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
        }
        Some(BlockKey(out))
    }
}
