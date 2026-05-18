// SPDX-License-Identifier: AGPL-3.0-only

//! Q/K/V projection + Flash Attention prefill paths.
//!
//! Wave-3 refactor split this 2619-line file into two methods-per-file
//! sub-modules. Both `paged.rs` and `cache_skip.rs` exceed the 500-LoC
//! cap because each contains a single monolithic 1000-1400 LoC method
//! (`prefill_attention_paged` / `prefill_attention_with_cache_skip`)
//! whose body interleaves 10+ sections with deep cross-section state
//! coupling. Splitting further requires extracting each section as a
//! helper method with 10-20 args — multi-day kernel-level surgery
//! beyond this wave's scope.

mod cache_skip;
mod cache_skip_mla;
mod cache_skip_qkv;
mod paged;
mod paged_attn;
mod paged_attn_batched;
mod paged_mla;
mod paged_oproj;
mod paged_qkv;
