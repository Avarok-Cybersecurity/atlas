// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill prefix-cache lookup seam: hybrid models use the paired hit.

use spark_runtime::prefix_cache::PrefixMatch;

use super::super::types::TransformerModel;

impl TransformerModel {
    /// Prefix-cache lookup for a prefill. Hybrid/GDN models (any SSM layer)
    /// require a restorable SSM snapshot at the matched KV length; a KV-only
    /// walk is a miss. Pure-attention models keep the KV-only hit.
    pub(in crate::model) fn lookup_prefill_prefix(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> PrefixMatch {
        if self.config.num_ssm_layers() > 0 {
            self.prefix_cache
                .lookup_paired(tokens, block_size, session_hash, adapter_id)
        } else {
            self.prefix_cache
                .lookup(tokens, block_size, session_hash, adapter_id)
        }
    }
}

#[cfg(test)]
mod tests {
    use spark_runtime::prefix_cache::PrefixMatch;

    /// The old serve path logged this and then recomputed all KV. A hybrid
    /// hit is KV+SSM or it is a miss — this string must not come back.
    const LYING_HIT: &str = "but no SSM snapshot";

    #[test]
    fn unpaired_kv_is_not_a_serve_hit() {
        let m = PrefixMatch {
            matched_blocks: vec![1, 2, 3, 4],
            matched_disk_block_ids: Vec::new(),
            matched_tokens: 64,
            ssm_snapshot: None,
            ssm_snapshot_tokens: 0,
            ssm_snapshot_tier_key: None,
            ssm_snapshot_tier_tokens: 0,
            ssm_snapshot_is_tail: false,
        };
        assert_eq!(
            m.paired_ssm_tokens(16),
            None,
            "serve path must report a miss when KV hits without SSM"
        );
    }

    #[test]
    fn paired_snapshot_is_a_serve_hit() {
        let m = PrefixMatch {
            matched_blocks: vec![1, 2, 3, 4],
            matched_disk_block_ids: Vec::new(),
            matched_tokens: 64,
            ssm_snapshot: Some(3),
            ssm_snapshot_tokens: 64,
            ssm_snapshot_tier_key: None,
            ssm_snapshot_tier_tokens: 0,
            ssm_snapshot_is_tail: false,
        };
        assert_eq!(m.paired_ssm_tokens(16), Some(64));
    }

    #[test]
    fn lying_hit_log_is_gone_from_serve_paths() {
        for (name, src) in [
            (
                "prefix_lookup.rs",
                include_str!("prefill_b/prefix_lookup.rs"),
            ),
            ("prefill_a.rs", include_str!("prefill_a.rs")),
            ("prefill_c.rs", include_str!("prefill_c.rs")),
        ] {
            assert!(
                !src.contains(LYING_HIT),
                "{name} still logs a prefix-cache hit when SSM cannot restore"
            );
        }
    }
}
