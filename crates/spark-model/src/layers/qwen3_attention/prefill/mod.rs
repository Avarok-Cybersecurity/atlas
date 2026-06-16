// SPDX-License-Identifier: AGPL-3.0-only

//! Q/K/V projection + Flash Attention prefill paths.
//!
//! Wave-3 refactor split this 2619-line file into methods-per-file
//! sub-modules. `paged.rs` / `cache_skip.rs` each hold a single
//! monolithic orchestrator method (`prefill_attention_paged` /
//! `prefill_attention_with_cache_skip`) whose body interleaves 10+
//! sections with deep cross-section state coupling. The function-bounded
//! phases are peeled off into siblings to keep every file under the
//! 500-LoC cap: `paged_rope.rs` (RoPE dispatch) and `cache_skip_attn.rs`
//! (contiguous Q/K/V Flash Attention). The remaining bodies stay as one
//! ordered sequence to preserve kernel-launch / state ordering.

mod cache_skip;
mod cache_skip_attn;
mod cache_skip_mla;
mod cache_skip_qkv;
mod paged;
mod paged_attn;
mod paged_attn_batched;
mod paged_mla;
mod paged_oproj;
mod paged_qkv;
mod paged_rope;
