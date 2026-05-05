// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::error::Result;

/// Token embedding lookup — fetch embedding vectors by token ID.
///
/// This is a simple gather operation. On SM121 it's memory-bound
/// (embedding table is in LPDDR5X), so we maximize coalesced access.
pub fn token_embedding_lookup(
    _token_ids_ptr: u64,  // [batch, seq_len] i32
    _embeddings_ptr: u64, // [vocab_size, hidden_size] BF16
    _output_ptr: u64,     // [batch, seq_len, hidden_size] BF16
    _batch: u32,
    _seq_len: u32,
    _hidden_size: u32,
    _stream_ptr: u64,
) -> Result<()> {
    // TODO: This is simple enough to be a cudarc launch of a small kernel
    todo!("Token embedding lookup")
}
