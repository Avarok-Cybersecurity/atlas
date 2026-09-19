// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//! Real-file oracle for expert streaming on the DeepSeek-V4.1 Flash Q2_K
//! shards: every byte the cache serves is the byte the loader's mmap would
//! have served, through eviction and re-fetch, for experts and engram rows;
//! then one token's gather measured (bytes, ms, GB/s, and the hit rate of the
//! same token replayed).
//!
//! Gated `#[ignore]`. Run (cold numbers need the page cache dropped first):
//!   ATLAS_SKIP_BUILD=1 cargo test -p spark-runtime --release -- --ignored deepseek_v41_stream_oracle --nocapture

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use super::expert_lru::ExpertLru;
use super::expert_stream::{EngramRowReader, ExpertSliceMap, ExpertSource, ShardFiles};
use super::sidecar;
use crate::gpu::DevicePtr;

const MODEL_DIR: &str = "/home/rstesiak/models/dsv41-q2k";

fn files() -> Arc<ShardFiles> {
    Arc::new(ShardFiles::open_dir(Path::new(MODEL_DIR)).expect("open the shard set"))
}

/// The mmap bytes of `[off, off + len)` in shard `shard`, the loader's own path.
fn mmap_bytes(files: &ShardFiles, shard: usize, off: u64, len: usize) -> Vec<u8> {
    let (_f, mmap, _g) = sidecar::open_gguf(files.path(shard)).expect("mmap shard");
    mmap[off as usize..off as usize + len].to_vec()
}

struct HeapArena(Vec<u8>);

impl HeapArena {
    fn new(bytes: usize) -> Self {
        HeapArena(vec![0u8; bytes])
    }
    fn lru(&mut self, layout: super::expert_stream::SlotLayout) -> ExpertLru {
        let p = self.0.as_mut_ptr();
        ExpertLru::new(p, DevicePtr(p as u64), self.0.len(), layout).unwrap()
    }
}

fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

#[test]
#[ignore = "requires the on-disk DeepSeek-V4.1-Flash Q2_K shards"]
fn deepseek_v41_stream_oracle_experts_match_the_mmap() {
    let files = files();
    let map = ExpertSliceMap::new(files.clone()).expect("expert slice map");
    let lay = map.slot_layout();
    println!(
        "moe layers {} ({}..{}), experts {}, slot {} bytes ({:.2} MiB) = gate {} + up {} + down {}",
        map.layers().len(),
        map.layers()[0].layer,
        map.layers().last().unwrap().layer,
        map.num_experts(),
        lay.bytes,
        lay.bytes as f64 / 1048576.0,
        lay.gate_bytes,
        lay.up_bytes,
        lay.down_bytes
    );
    assert_eq!(map.num_experts(), 384);
    assert_eq!(
        lay.bytes,
        46_080 * (84 + 84 + 110),
        "one expert = 46,080 blocks of Q2_K, Q2_K, Q3_K"
    );

    let first = map.layers()[0].layer as u32;
    let last = map.layers().last().unwrap().layer as u32;
    let probes = [(first, 0u32), (3, 200), (last, 383), (first + 1, 17)];

    // two slots: the third probe evicts the first, the re-fetch reads it again
    let mut arena = HeapArena::new(2 * lay.bytes);
    let mut lru = arena.lru(lay);
    let check = |lru: &mut ExpertLru, layer: u32, e: u32| {
        lru.begin_token();
        let (slot, _) = lru.fetch(&map, layer, e).unwrap();
        // SAFETY: the heap arena outlives the slot.
        let got = unsafe { std::slice::from_raw_parts(slot.host, lay.bytes) };
        let l = map.layer(layer as usize).unwrap();
        for (loc, off, len, name) in [
            (&l.gate, lay.gate_off, lay.gate_bytes, "gate"),
            (&l.up, lay.up_off, lay.up_bytes, "up"),
            (&l.down, lay.down_off, lay.down_bytes, "down"),
        ] {
            let (shard, at) = map.slice_at(loc, e as usize);
            let want = mmap_bytes(&files, shard, at, len);
            assert!(
                got[off..off + len] == want[..],
                "layer {layer} expert {e} {name}: cache bytes differ from the mmap"
            );
        }
        println!(
            "  layer {layer:>2} expert {e:>3}: {} bytes identical to the mmap (shard {})",
            lay.bytes, l.gate.shard
        );
    };
    for &(l, e) in &probes {
        check(&mut lru, l, e);
    }
    assert_eq!(lru.stats().evictions, 2);
    assert!(!lru.contains(probes[0].0, probes[0].1));
    // re-fetch the evicted first probe: read again, identical again
    check(&mut lru, probes[0].0, probes[0].1);
    let st = lru.stats();
    assert_eq!((st.misses, st.evictions), (5, 3));
    println!("  evict + re-fetch: identical; stats {st:?}");
}

#[test]
#[ignore = "requires the on-disk DeepSeek-V4.1-Flash Q2_K shards"]
fn deepseek_v41_stream_oracle_engram_rows_match_the_mmap() {
    let files = files();
    let rd = EngramRowReader::new(files.clone()).expect("engram reader");
    let layers: Vec<usize> = rd.tables().iter().map(|t| t.layer).collect();
    println!("engram tables on layers {layers:?}");
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    for t in rd.tables() {
        assert_eq!(
            (t.row_bytes, t.head_dim),
            (84, 256),
            "one Q2_K block per row"
        );
        let ids: Vec<u64> = (0..24).map(|_| xorshift(&mut x) % t.rows as u64).collect();
        let mut got = vec![0u8; ids.len() * t.row_bytes];
        let t0 = Instant::now();
        rd.read_rows(t.layer, &ids, &mut got).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        for (i, &id) in ids.iter().enumerate() {
            let want = mmap_bytes(
                &files,
                t.shard,
                t.base + id * t.row_bytes as u64,
                t.row_bytes,
            );
            assert!(
                got[i * t.row_bytes..(i + 1) * t.row_bytes] == want[..],
                "engram layer {} row {id}: bytes differ from the mmap",
                t.layer
            );
        }
        println!(
            "  layer {:>2}: {} rows of {} ({:.2} GiB table) identical to the mmap; {ms:.1} ms by pread",
            t.layer,
            ids.len(),
            t.rows,
            (t.rows * t.row_bytes) as f64 / 1073741824.0
        );
        // out-of-range row is refused, not read
        assert!(
            rd.read_rows(t.layer, &[t.rows as u64], &mut got[..t.row_bytes])
                .is_err()
        );
    }
}

#[test]
#[ignore = "requires the on-disk DeepSeek-V4.1-Flash Q2_K shards"]
fn deepseek_v41_stream_one_token_gather() {
    let files = files();
    let map = ExpertSliceMap::new(files).expect("expert slice map");
    let lay = map.slot_layout();
    let per_layer = 6usize;
    let n_layers = map.layers().len();
    let instances = per_layer * n_layers;
    // enough for two full tokens, so the replay is a pure hit test
    let mut arena = HeapArena::new(2 * instances * lay.bytes);
    let mut lru = arena.lru(lay);
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let keys: Vec<(u32, u32)> = map
        .layers()
        .iter()
        .flat_map(|l| {
            let mut picked: Vec<u32> = Vec::new();
            while picked.len() < per_layer {
                let e = (xorshift(&mut x) % map.num_experts() as u64) as u32;
                if !picked.contains(&e) {
                    picked.push(e);
                }
            }
            picked.into_iter().map(move |e| (l.layer as u32, e))
        })
        .collect();
    assert_eq!(keys.len(), instances);
    for threads in [1usize, 8] {
        let mut arena2 = HeapArena::new(instances * lay.bytes);
        let mut lru2 = arena2.lru(lay);
        lru2.begin_token();
        let t0 = Instant::now();
        let slots = lru2.fetch_many(&map, &keys, threads).unwrap();
        let s = t0.elapsed().as_secs_f64();
        let st = lru2.stats();
        assert_eq!(slots.len(), instances);
        assert_eq!(st.misses as usize, instances);
        println!(
            "ONE TOKEN ({instances} instances = {per_layer} x {n_layers} layers): {:.2} GiB in {:.0} ms with {threads} reader thread(s) = {:.2} GB/s",
            st.bytes_read as f64 / 1073741824.0,
            s * 1e3,
            st.bytes_read as f64 / s / 1e9
        );
    }
    lru.begin_token();
    lru.fetch_many(&map, &keys, 8).unwrap();
    lru.reset_stats();
    lru.begin_token();
    let t0 = Instant::now();
    lru.fetch_many(&map, &keys, 8).unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let st = lru.stats();
    assert_eq!((st.hits as usize, st.misses), (instances, 0));
    println!(
        "  same token replayed: {instances}/{instances} hits, 0 bytes read, {ms:.2} ms; resident {} of {} slots",
        lru.resident(),
        lru.n_slots()
    );
}
