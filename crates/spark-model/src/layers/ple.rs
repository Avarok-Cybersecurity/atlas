// SPDX-License-Identifier: AGPL-3.0-only

//! PLE — hashed n-gram injection into the hyper-connection highway.
//!
//! Qwen3.8-Flash-Next runs this on ONE layer (`ple_layer_ids` is 1-indexed,
//! so `[2]` means model layer 1). From the reference's own docstring:
//!
//! > PLE projects each token's concatenated n-gram embedding to a shared
//! > value and one key per residual stream. The normalized stream activations
//! > gate those values, then a dilated depthwise convolution adds local
//! > lexical context.
//!
//! That is **cross-attention into an n-gram table**, not an additive
//! embedding — the reading that cost real time on LongCat. The forward, from
//! `Qwen4ExpTextPLELayer.forward`:
//!
//! ```text
//! embeddings   = ple_embedding(input_ids)                  # [T, 2560]
//! key_normed   = norm_key(key_proj(emb)) -> [T, hc, H]
//! value        = value_proj(emb)                           # [T, 2560]
//! query_normed = norm_query(hidden)      -> [T, hc, H]     # hidden is [T, 10240]
//! gate  = (key_normed * query_normed).sum(-1) / sqrt(H)    # [T, hc]
//! gate  = sign(gate) * sqrt(max(|gate|, 1e-6))             # SIGNED SQRT
//! gated = sigmoid(gate) * value                            # [T, hc, H]
//! out   = gated.flatten() + silu(conv1d(norm_conv(gated.flatten())))
//! ```
//!
//! and the decoder layer adds it to the highway BEFORE that layer's
//! attention hyper-connection: `hidden_states = hidden_states + ple(...)`.
//!
//! Three things here bite, all quietly:
//!
//! 1. **The signed square root** on the gate. Nobody would guess it; omit it
//!    and the gate distribution is wrong but perfectly finite.
//! 2. **`conv1d` is depthwise AND dilated** — `groups = 10240`,
//!    `kernel_size = 4`, `dilation = ngram_size = 3`, so the state is
//!    `(4-1)*3 = 9` steps, not 3.
//! 3. **All three norms are the offset-from-1 form** (`normed * (1 + w)`,
//!    `Qwen4ExpTextRMSNorm`) and **grouped** with `group_size = hidden_size`
//!    — four independent 2560-wide norms inside the 10240 vector, same as
//!    `hc_norm`. See `bench/qwen4_exp/ARCHITECTURE.md` §6.
//!
//! The n-gram table is ~320M rows x 160 dims. It is NOT resident: the row
//! cache, pinned arena and deferred-load path from #746 serve it off NVMe.
//! What does NOT transfer from #746 is the id computation — see `ids.rs`.

#[path = "ple/ids.rs"]
pub mod ids;

#[cfg(test)]
#[path = "ple/tests.rs"]
mod tests;

#[path = "ple/dump.rs"]
pub mod dump;

#[path = "ple/layer.rs"]
mod layer;

pub use ids::{PleIdDims, ple_ngram_ids};
pub use layer::{NgramRowFormat, PleLayer, PleSeqState, PleWeights};
