// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: tests of the index growth policy: when the initial build
//! and the updates run and over which range, the counts each step leaves, the
//! kernel input it hands over, and the oracle checks of L33 (an
//! update equals a fresh one-segment build of its range; an index grown by
//! updates recalls like a fresh build).

use spark_runtime::gpu::DevicePtr;

use super::config::RetroParams;
use super::oracle::{HeadKv, KmeansSpec, OracleLayer, build_head};
use super::oracle_eval::{measure_head, score_clusters, select};
use super::oracle_tests::clustered;
use super::update::*;

fn params() -> RetroParams {
    RetroParams {
        segment_tokens: 1024,
        min_context: 1024,
        update_interval: 256,
        ..RetroParams::default()
    }
}

#[test]
fn initial_build_waits_for_min_context_and_aligns() {
    let p = RetroParams::default();
    assert_eq!(initial_end(16383, &p, 16), None);
    let e = initial_end(16384 + 7, &p, 16).unwrap();
    assert_eq!(e % 16, 0);
    assert!(16384 + 7 - e >= p.window, "the window stays exact");
    assert!(16384 + 7 - e < p.window + 16);
    let s = initial_step(131072, &p, 16).unwrap();
    assert_eq!(s.first_pos, p.sink);
    assert_eq!(s.tokens, s.after.indexed_end - p.sink);
    assert_eq!(s.spec.segments, 16);
    assert_eq!(s.after.listed, s.tokens);
    assert_eq!(
        s.after.nprobe,
        super::config::nprobe(s.after.n_centroids, p.retrieval_budget)
    );
}

#[test]
fn updates_fire_once_the_tail_holds_window_plus_interval() {
    let p = RetroParams::default();
    let c = initial_step(20000, &p, 16).unwrap().after;
    let tail0 = 20000 - c.indexed_end;
    assert!(tail0 < p.window + 16);
    let fire = c.indexed_end + p.window + p.update_interval;
    assert_eq!(due(&c, fire - 1, &p, 16), None);
    assert_eq!(due(&c, fire, &p, 16), Some(1024));
    let u = update_step(&c, fire, &p, 16).unwrap();
    assert_eq!(u.first_pos, c.indexed_end);
    assert_eq!(
        u.spec,
        KmeansSpec {
            centroids: 64,
            segments: 1,
            iters: 10
        }
    );
    assert_eq!(u.after.n_centroids, c.n_centroids + 64);
    assert_eq!(u.after.nprobe, c.nprobe + 1, "UPDATE_NPROBE");
    assert_eq!(u.after.es, c.es + 15, "UPDATE_ES");
    assert_eq!(u.after.indexed_end, c.indexed_end + 1024);
    assert_eq!(
        due(&u.after, fire, &p, 16),
        None,
        "the tail is back to the window"
    );
    // An empty index never updates (dense below min context).
    assert_eq!(due(&Default::default(), 1 << 20, &p, 16), None);
}

#[test]
fn pending_builds_append_after_the_current_entries() {
    let p = RetroParams::default();
    let c = initial_step(20000, &p, 16).unwrap().after;
    let u = update_step(&c, c.indexed_end + 2000, &p, 16).unwrap();
    let pb = u.pending(&c, DevicePtr(0x1000), 4096);
    assert_eq!(pb.centroid_offset, c.n_centroids);
    assert_eq!(pb.token_offset, c.listed);
    assert_eq!(pb.tokens, 1024);
    assert_eq!(pb.centroids, 64);
    assert_eq!(pb.segments, 1);
    assert_eq!(pb.scratch_bytes, 4096);
}

#[test]
fn a_cached_prefix_is_extended_by_a_segmented_append() {
    let p = RetroParams::default();
    let prefix = initial_step(40000, &p, 16).unwrap().after;
    // A short remainder stays in the tail for the regular updates.
    assert!(append_step(&prefix, 40000 + 500, &p, 16).is_none());
    let a = append_step(&prefix, 100_000, &p, 16).unwrap();
    assert_eq!(a.first_pos, prefix.indexed_end);
    assert_eq!(a.after.indexed_end, initial_end(100_000, &p, 16).unwrap());
    let total = a.after.n_centroids;
    assert_eq!(
        a.after.nprobe,
        super::config::nprobe(total, p.retrieval_budget),
        "zones recomputed"
    );
    assert!(fits(&a.after, total, a.after.listed));
    assert!(!fits(&a.after, total - 1, a.after.listed));
}

#[test]
fn an_update_equals_a_fresh_one_segment_build_of_its_range() {
    let d = 16;
    let n = 2400usize;
    let (k, v, _) = clustered(n, d, 64, 12, 3.0, 0.3, 21);
    let kv = HeadKv {
        k: &k,
        v: &v,
        head_dim: d,
    };
    let p = params();
    let mut l = OracleLayer::build(&[kv], &p, 1536);
    let before = l.heads[0].clone();
    let c0 = l.counts;
    l.catch_up(&[kv], &p, 1536 + p.window + 256, 16);
    assert_eq!(l.counts.indexed_end, 1536 + 256);
    let fresh = build_head(kv, 1536, 1536 + 256, KmeansSpec::update(&p));
    let nc0 = c0.n_centroids as usize;
    assert_eq!(
        &l.heads[0].sizes[..nc0],
        &before.sizes[..],
        "old clusters untouched"
    );
    assert_eq!(&l.heads[0].sizes[nc0..], &fresh.sizes[..]);
    assert_eq!(&l.heads[0].centroids[nc0 * d..], &fresh.centroids[..]);
    assert_eq!(&l.heads[0].lists[nc0..], &fresh.lists[..]);
    assert_eq!(l.counts.n_centroids as usize, nc0 + fresh.n_centroids());
}

#[test]
fn an_index_grown_by_updates_recalls_like_a_fresh_build() {
    let d = 32;
    let n = 4096u32;
    let (k, v, dirs) = clustered(n as usize, d, 256, 16, 4.0, 0.1, 23);
    let kv = HeadKv {
        k: &k,
        v: &v,
        head_dim: d,
    };
    let p = params();
    // Grown: built at 2048 tokens, then eight 256-token updates up to 4096.
    let mut grown = OracleLayer::build(&[kv], &p, initial_end(2048, &p, 16).unwrap());
    grown.catch_up(&[kv], &p, n, 16);
    assert!(grown.counts.indexed_end >= n - p.window - 256);
    let fresh = OracleLayer::build(&[kv], &p, grown.counts.indexed_end);
    let scale = 1.0 / (d as f32).sqrt();
    let mut worst = 0f64;
    // Queries aimed at clusters in the generated part (tokens 2048..4032).
    for cl in [130usize, 170, 200, 240] {
        let q: Vec<f32> = dirs[cl].iter().map(|x| 16.0 * x).collect();
        let m = |l: &OracleLayer| {
            let s = score_clusters(&l.heads[0], l.counts.n_centroids as usize, &q, 1, scale);
            let z = select(&s, l.counts.nprobe as usize, l.counts.es as usize);
            measure_head(kv, &l.heads[0], &l.counts, &z, &q, n, scale, None)
        };
        let (g, f) = (m(&grown), m(&fresh));
        assert!(f.mass_total > 0.95, "fresh {f:?}");
        worst = worst.max((g.mass_total - f.mass_total).abs());
    }
    assert!(
        worst <= 0.01,
        "grown vs fresh mass recall differ by {worst}"
    );
}
