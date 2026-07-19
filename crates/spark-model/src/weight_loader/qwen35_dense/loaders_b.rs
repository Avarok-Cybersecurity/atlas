// SPDX-License-Identifier: AGPL-3.0-only

//! Small `ModelWeightLoader` method bodies split out of `qwen35_dense.rs`
//! for the ≤500 LoC file-size cap. Called from the trait impl in the parent.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{DenseWeight, dense};

pub(super) fn load_embedding(store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.embed_tokens.weight"))
}

pub(super) fn load_final_norm(store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.norm.weight"))
}

