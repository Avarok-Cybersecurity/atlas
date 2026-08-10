// SPDX-License-Identifier: AGPL-3.0-only
//
// Tier-1 acceptance: the streamer delivers byte-identical bytes regardless of
// tier, so a forward pass CANNOT differ by tier. Proves, at the byte layer:
//   * PosixTier (bounce oracle) == UmaArenaTier (O_DIRECT zero-copy) across all
//     six NVFP4 sub-buffers + scale2/input_scale (foundation for the Stage-2
//     bit-identical logit gate);
//   * capped-arena slab reuse still matches the oracle (invariant C at bytes);
//   * one fetch touches only its own slot (byte-layer basis for invariant E's
//     local_expert_range scoping, fully tested at Stage 2).
//
// GPU-gated (each tier allocates CUDA memory) and O_DIRECT-gated (UMA reads need
// a block-backed fs — tmpfs returns EINVAL), so these are `#[ignore]` and run
// explicitly: `cargo test -p spark-storage --test expert_stream_parity -- --ignored`.
// The store is written under CARGO_TARGET_TMPDIR (target/, ext4) not /tmp.

use std::collections::HashMap;
use std::path::PathBuf;

use spark_storage::cuda_min::CudaCtx;
use spark_storage::expert::{ExpertKey, Proj};
use spark_storage::expert_tier::{ArenaSlot, ExpertTier, PosixTier, UmaArenaTier, read_device};
use spark_storage::expert_tier_rdma::RdmaTier;
use spark_storage::{ExpertFileWriter, ExpertIndex, ExpertRecordHeader, ProjData};

const LAYERS: u32 = 3;
const EXPERTS: u32 = 4;
const INTER: u64 = 64;
const HIDDEN: u64 = 128;
const GS: u64 = 16;

struct Expected {
    packed: [Vec<u8>; 3],
    scale: [Vec<u8>; 3],
    scale2: [f32; 3],
    input_scale: [Option<f32>; 3],
}

fn fill(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xABCD);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s & 0xFF) as u8
        })
        .collect()
}

fn proj_nk(p: Proj) -> (u64, u64) {
    match p {
        Proj::Gate | Proj::Up => (INTER, HIDDEN),
        Proj::Down => (HIDDEN, INTER),
    }
}

/// Write a synthetic store on an O_DIRECT-capable fs; return the expected bytes.
fn build_store(tag: &str) -> (PathBuf, HashMap<(u32, u32), Expected>) {
    // Compile-time env (target/, ext4) — not /tmp (tmpfs rejects O_DIRECT).
    let root = env!("CARGO_TARGET_TMPDIR");
    let dir = PathBuf::from(root).join(format!("xpr-parity-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);

    let index = ExpertIndex::new(INTER, HIDDEN, GS, 256, 4096, (0..LAYERS).collect(), EXPERTS);
    let spec = index.spec();
    let writer = ExpertFileWriter::create(&dir, index).expect("create store");
    let mut expected = HashMap::new();

    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            let mut packed: [Vec<u8>; 3] = Default::default();
            let mut scale: [Vec<u8>; 3] = Default::default();
            for p in Proj::ALL {
                let (n, k) = proj_nk(p);
                let seed = (layer as u64) << 24 | (expert as u64) << 8 | p as u64;
                packed[p as usize] = fill((n * k / 2) as usize, seed);
                scale[p as usize] = fill((n * k / GS) as usize, seed ^ 0x5555);
                assert_eq!(
                    packed[p as usize].len() as u64,
                    spec.proj_bytes(p).packed_bytes
                );
                assert_eq!(
                    scale[p as usize].len() as u64,
                    spec.proj_bytes(p).scale_bytes
                );
            }
            let scale2 = [
                1.0 + layer as f32,
                2.0 + expert as f32,
                0.5 + (layer + expert) as f32,
            ];
            let input_scale = [Some(0.25), None, Some(1.75)];
            let header = ExpertRecordHeader {
                layer,
                expert,
                inter: INTER as u32,
                hidden: HIDDEN as u32,
                group_size: GS as u32,
                scale2,
                input_scale,
            };
            let projs = [
                ProjData {
                    packed: &packed[0],
                    scale: &scale[0],
                },
                ProjData {
                    packed: &packed[1],
                    scale: &scale[1],
                },
                ProjData {
                    packed: &packed[2],
                    scale: &scale[2],
                },
            ];
            writer
                .write_record(ExpertKey::new(layer, expert), &header, &projs)
                .expect("write record");
            expected.insert(
                (layer, expert),
                Expected {
                    packed,
                    scale,
                    scale2,
                    input_scale,
                },
            );
        }
    }
    writer.finish().expect("finish store");
    (dir, expected)
}

/// Read a tier's residency back from the device and compare to `exp`.
fn assert_matches(
    tier: &mut dyn ExpertTier,
    key: ExpertKey,
    slot: ArenaSlot,
    stream: u64,
    exp: &Expected,
) {
    let r = tier.fetch(key, slot, stream).expect("fetch");
    assert_eq!(r.scale2, exp.scale2, "{:?} scale2 ({:?})", key, tier.kind());
    assert_eq!(
        r.input_scale,
        exp.input_scale,
        "{:?} input_scale ({:?})",
        key,
        tier.kind()
    );
    for p in Proj::ALL {
        let pl = exp.packed[p as usize].len();
        let sl = exp.scale[p as usize].len();
        let got_packed = read_device(r.packed_addr[p as usize], pl, stream).unwrap();
        let got_scale = read_device(r.scale_addr[p as usize], sl, stream).unwrap();
        assert_eq!(
            got_packed,
            exp.packed[p as usize],
            "{:?} {:?} packed via {:?}",
            key,
            p,
            tier.kind()
        );
        assert_eq!(
            got_scale,
            exp.scale[p as usize],
            "{:?} {:?} scale via {:?}",
            key,
            p,
            tier.kind()
        );
    }
}

#[test]
#[ignore = "requires GPU + O_DIRECT-capable fs"]
fn posix_equals_uma_byte_identical() {
    let ctx = CudaCtx::new(0).expect("cuda ctx");
    let (dir, expected) = build_store("full");
    // One slab per layer, one slot per expert — no reuse.
    let mut posix = PosixTier::open(&dir, LAYERS, EXPERTS).expect("posix tier");
    let mut uma = UmaArenaTier::open(&dir, LAYERS, EXPERTS).expect("uma tier");

    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            let key = ExpertKey::new(layer, expert);
            let slot = ArenaSlot::new(layer, expert);
            let exp = &expected[&(layer, expert)];
            // Each tier independently reproduces the source bytes...
            assert_matches(&mut posix, key, slot, ctx.stream, exp);
            assert_matches(&mut uma, key, slot, ctx.stream, exp);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires GPU + O_DIRECT-capable fs"]
fn capped_arena_eviction_parity() {
    // Only 2 slabs for a 3-layer store: layer 2 reuses slab 0, forcing eviction.
    let ctx = CudaCtx::new(0).expect("cuda ctx");
    let (dir, expected) = build_store("capped");
    let num_slabs = 2u32;
    let mut uma = UmaArenaTier::open(&dir, num_slabs, EXPERTS).expect("uma tier");
    let mut posix = PosixTier::open(&dir, num_slabs, EXPERTS).expect("posix tier");

    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            let key = ExpertKey::new(layer, expert);
            let slot = ArenaSlot::new(layer % num_slabs, expert); // reuse slabs
            let exp = &expected[&(layer, expert)];
            // Even under slab reuse, each fetch lands the correct record and
            // still matches the oracle — eviction doesn't corrupt bytes.
            assert_matches(&mut uma, key, slot, ctx.stream, exp);
            assert_matches(&mut posix, key, slot, ctx.stream, exp);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires GPU + O_DIRECT-capable fs"]
fn rdma_tier_matches_oracle_over_loopback() {
    // Stage 4 (Phase A): a peer serves the store over TCP; RdmaTier lands records
    // in the pinned arena and must be byte-identical to the Posix oracle — proving
    // the peer-as-tier abstraction with zero verbs risk.
    let ctx = CudaCtx::new(0).expect("cuda ctx");
    let (dir, expected) = build_store("rdma");
    let port = 20000 + (std::process::id() % 10000) as u16;
    let addr = format!("127.0.0.1:{port}");

    // Serve in a background thread (leaks on test exit — fine).
    let serve_dir = dir.clone();
    let serve_addr = addr.clone();
    std::thread::spawn(move || {
        let _ = spark_storage::expert_peer::serve(
            &serve_dir,
            serve_addr.as_str(),
            spark_storage::expert_peer::RdmaConfig::default(),
        );
    });
    // Wait for the listener to come up.
    let mut connected = None;
    for _ in 0..50 {
        if std::net::TcpStream::connect(&addr).is_ok() {
            connected = Some(());
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    connected.expect("peer never accepted a connection");

    let mut rdma = RdmaTier::connect(&addr, LAYERS, EXPERTS, false).expect("rdma connect");
    let mut posix = PosixTier::open(&dir, LAYERS, EXPERTS).expect("posix tier");
    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            let key = ExpertKey::new(layer, expert);
            let slot = ArenaSlot::new(layer, expert);
            let exp = &expected[&(layer, expert)];
            assert_matches(&mut rdma, key, slot, ctx.stream, exp);
            assert_matches(&mut posix, key, slot, ctx.stream, exp);
        }
    }
    assert!(rdma.healthy(), "rdma tier should still be healthy");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires GPU + O_DIRECT-capable fs"]
fn fetch_touches_only_its_slot() {
    // Byte-layer basis for invariant E: filling one expert's slot must not
    // disturb another slot's bytes (the install loop's per-expert scoping
    // relies on this). Full local_expert_range scoping is a Stage-2 test.
    let ctx = CudaCtx::new(0).expect("cuda ctx");
    let (dir, expected) = build_store("isolation");
    let mut uma = UmaArenaTier::open(&dir, 1, EXPERTS).expect("uma tier");

    let k0 = ExpertKey::new(0, 0);
    let k1 = ExpertKey::new(0, 1);
    let r0 = uma.fetch(k0, ArenaSlot::new(0, 0), ctx.stream).unwrap();
    // Fetch a different expert into a different slot...
    let _r1 = uma.fetch(k1, ArenaSlot::new(0, 1), ctx.stream).unwrap();
    // ...slot 0's bytes are unchanged.
    let e0 = &expected[&(0, 0)];
    for p in Proj::ALL {
        let got = read_device(
            r0.packed_addr[p as usize],
            e0.packed[p as usize].len(),
            ctx.stream,
        )
        .unwrap();
        assert_eq!(
            got, e0.packed[p as usize],
            "slot 0 {:?} disturbed by slot 1 fetch",
            p
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
