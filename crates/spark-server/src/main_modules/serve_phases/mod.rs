// SPDX-License-Identifier: AGPL-3.0-only

//! Helper functions extracted from `serve.rs` to keep that file under the
//! 500-LoC cap. Split across themed sub-files because the combined helper
//! body is larger than 500 LoC itself.

mod build;
mod config;
mod kv_cache;
mod preflight;
mod runtime;
mod tokenizer_runtime;
mod topology;
mod weights;

pub(super) use build::{
    build_high_speed_swap_config, build_model, build_prefix_cache, maybe_run_ep_worker,
    validate_head_high_speed_swap,
};
pub(super) use config::{
    apply_model_default_num_drafts, cap_vocab_size_to_tokenizer, load_model_config,
    merge_sidecar_quant_config, resolve_model_dir,
};
pub(super) use kv_cache::{
    KvCacheConfig, PrefillBudget, resolve_kv_cache_config, resolve_prefill_budget,
};
pub(super) use preflight::{
    ReservePreflight, init_gpu_backend, post_load_memory_audit, preflight_reserve,
};
pub(super) use runtime::{
    SamplingDefaults, load_eos_tokens, load_sampling_defaults, log_behavior_audit,
    log_response_store_audit, open_dump_writer, resolve_model_name, resolve_tool_call_parser,
};
pub(super) use tokenizer_runtime::{TokenizerRuntime, resolve_tokenizer_runtime};
pub(super) use topology::{Topology, init_nccl_comm, resolve_topology};
pub(super) use weights::{auto_detect_weight_prefix, load_dflash_drafter, load_weight_store};
