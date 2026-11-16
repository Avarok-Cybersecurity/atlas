// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: tests of the RetroInfer sizes (against upstream's numbers),
//! the parameter validation, and the CLI / environment / default resolution.

use std::collections::HashMap;

use super::api::{CliValues, env, resolve_params};
use super::config::*;
use crate::layers::qwen3_attention::sparse_seam::KvDtypeTag;

#[test]
fn python_rounding_is_half_to_even() {
    assert_eq!(round_half_even(0.5), 0);
    assert_eq!(round_half_even(1.5), 2);
    assert_eq!(round_half_even(2.5), 2);
    assert_eq!(round_half_even(2.4999), 2);
    assert_eq!(round_half_even(2.5001), 3);
    assert_eq!(round_half_even(-2.5), -2);
    assert_eq!(round_half_even(7.0), 7);
}

#[test]
fn sizes_match_upstream_at_128k() {
    // generate_config(context_len=131072): 16 segments, 8192 centroids;
    // retroinfer_cache: nprobe round(8192 * 0.018) = 147, es round(1900.544) = 1901.
    let p = RetroParams::default();
    assert_eq!(n_segments(131072, p.segment_tokens), 16);
    assert_eq!(n_centroids(131072, p.avg_cluster, 16), 8192);
    assert_eq!(nprobe(8192, p.retrieval_budget), 147);
    assert_eq!(estimation_clusters(8192, p.estimation_budget, 147), 1901);
    // UPDATE_CENTROIDS = max(round(1024/16)//8*8, 8) = 64; UPDATE_NPROBE = 1; UPDATE_ES = 15.
    assert_eq!(update_centroids(1024, 16), 64);
    assert_eq!(update_nprobe(64, p.retrieval_budget), 1);
    assert_eq!(update_es(64, p.estimation_budget), 15);
}

#[test]
fn centroid_rounding_follows_the_lcm_rule() {
    // 12288 / 8192 = 1.5 rounds to 2 segments; lcm(8, 2) = 8; round(12288/16) = 768.
    assert_eq!(n_segments(12288, 8192), 2);
    assert_eq!(n_centroids(12288, 16, 2), 768);
    // 3 segments: lcm 24; round(24576/16) = 1536 = 64 * 24.
    assert_eq!(n_segments(24576, 8192), 3);
    assert_eq!(n_centroids(24576, 16, 3) % 24, 0);
    // Below the lcm the count is lifted to it.
    assert_eq!(n_centroids(40, 16, 1), 8);
    // A tie between the two multiples takes the lower one: n = 12 with factor 8.
    assert_eq!(n_centroids(192, 16, 1), 8);
    // nprobe never exceeds the centroids and is at least one.
    assert_eq!(nprobe(8, 0.018), 1);
    assert_eq!(nprobe(4, 1.0), 4);
    assert_eq!(estimation_clusters(8, 0.9, 1), 7);
}

#[test]
fn capacities_bound_the_index_and_its_zones() {
    let p = RetroParams::default();
    for ctx in [16384u32, 65536, 131072, 262144] {
        let cap = centroid_capacity(&p, ctx);
        let segs = n_segments(ctx, p.segment_tokens);
        assert!(cap >= n_centroids(ctx, p.avg_cluster, segs), "{ctx}");
        // An index grown entirely by updates fits too.
        assert!(
            cap >= ctx / p.update_interval * p.update_centroids(),
            "{ctx}"
        );
        let (rcap, ecap) = zone_caps(&p, cap);
        assert!(rcap >= 2 * nprobe(cap, p.retrieval_budget), "{ctx}");
        assert!(
            ecap >= estimation_clusters(cap, p.estimation_budget, nprobe(cap, p.retrieval_budget))
        );
        assert_eq!(
            exact_capacity(&p, rcap),
            p.sink + p.window + p.update_interval + MAX_VERIFY_ROWS + rcap * p.avg_cluster * 2
        );
    }
    let cfg = RetroCfg::new(p, 4, 16, 256, 0.0625, KvDtypeTag::Bf16, 131072, 16).unwrap();
    assert_eq!(cfg.group, 4);
    assert!(cfg.build_scratch_bytes(131072, 8192) > 4 * 131072 * 4);
    assert!(
        RetroCfg::new(
            RetroParams::default(),
            3,
            16,
            256,
            0.0625,
            KvDtypeTag::Bf16,
            1024,
            16
        )
        .is_err()
    );
}

#[test]
fn validation_names_the_flag() {
    let bad = |f: fn(&mut RetroParams)| {
        let mut p = RetroParams::default();
        f(&mut p);
        p.validate().unwrap_err().to_string()
    };
    assert!(bad(|p| p.retrieval_budget = 0.0).contains("--retro-retrieval-budget"));
    assert!(bad(|p| p.estimation_budget = 1.0).contains("--retro-estimation-budget"));
    assert!(
        bad(|p| {
            p.retrieval_budget = 0.6;
            p.estimation_budget = 0.6;
        })
        .contains("must not exceed 1")
    );
    assert!(bad(|p| p.window = 4).contains("--retro-window"));
    assert!(bad(|p| p.update_interval = 32).contains("--retro-update-interval"));
    assert!(bad(|p| p.min_context = 100).contains("--retro-min-context"));
    assert!(bad(|p| p.index_cache_mb = 1).contains("--retro-index-cache-mb"));
    assert!(RetroParams::default().validate().is_ok());
}

#[test]
fn build_hash_tracks_build_parameters_only() {
    let a = RetroParams::default();
    let mut b = a.clone();
    b.retrieval_budget = 0.05;
    b.estimation_budget = 0.1;
    assert_eq!(
        a.build_hash(),
        b.build_hash(),
        "zone budgets do not change the index"
    );
    b.segment_tokens = 4096;
    assert_ne!(a.build_hash(), b.build_hash());
}

fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let m: HashMap<String, String> = pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    move |k| m.get(k).cloned()
}

#[test]
fn cli_beats_env_beats_default() {
    let get = env_of(&[
        (env::WINDOW, "128"),
        (env::SINK, "8"),
        (env::RETRIEVAL_BUDGET, "0.05"),
    ]);
    let cli = CliValues {
        window: Some(256),
        ..Default::default()
    };
    let p = resolve_params(&cli, &get).unwrap();
    assert_eq!(p.window, 256, "CLI wins");
    assert_eq!(p.sink, 8, "env next");
    assert_eq!(p.retrieval_budget, 0.05);
    assert_eq!(p.segment_tokens, DEFAULT_SEGMENT_TOKENS, "default last");
    assert!(!p.shadow);
}

#[test]
fn env_errors_are_startup_errors() {
    let get = env_of(&[(env::SEGMENT_TOKENS, "lots")]);
    let e = resolve_params(&CliValues::default(), &get)
        .unwrap_err()
        .to_string();
    assert!(e.contains("AVAROK_RETRO_SEGMENT_TOKENS"), "{e}");
    let get = env_of(&[(env::SHADOW_LAYERS, "0,x")]);
    assert!(resolve_params(&CliValues::default(), &get).is_err());
    let get = env_of(&[(env::MIN_CONTEXT, "512")]);
    assert!(
        resolve_params(&CliValues::default(), &get).is_err(),
        "range checked after merging"
    );
}

#[test]
fn shadow_toggle_and_layers() {
    let get = env_of(&[
        (env::SHADOW, "on"),
        (env::SHADOW_LAYERS, "1, 3"),
        (env::SHADOW_EVERY, "8"),
    ]);
    let p = resolve_params(&CliValues::default(), &get).unwrap();
    assert!(p.shadow);
    assert_eq!(p.shadow_every, 8);
    assert!(p.shadow_layer(3) && !p.shadow_layer(2));
    let get = env_of(&[(env::SHADOW, "off")]);
    assert!(!resolve_params(&CliValues::default(), &get).unwrap().shadow);
    let cli = CliValues {
        shadow: true,
        ..Default::default()
    };
    let p = resolve_params(&cli, &env_of(&[])).unwrap();
    assert!(p.shadow && p.shadow_layer(7));
}
