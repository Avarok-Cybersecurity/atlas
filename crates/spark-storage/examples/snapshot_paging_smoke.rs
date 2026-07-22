// SPDX-License-Identifier: AGPL-3.0-only
//
// WS-A live smoke: drive the paging tier end-to-end over real RDMA against an
// atlas-cache-peer started with --swap-dir. PUTs far more blobs than the RAM
// arena holds → forces the peer to spill the coldest to its NVMe swap file →
// GETs them all back and asserts byte-identical (a fault-from-disk on each key
// the peer evicted). Proves: connect_paging handshake, control channel
// (alloc/commit/get), one-sided RDMA data plane, and peer-side NVMe swap +
// rehydrate — the whole stack minus the model.
//
//   ATLAS_SNAP_PEER=host:port \
//   ATLAS_EXPERT_RDMA_DEV=roceP2p1s0f1 ATLAS_EXPERT_RDMA_GID=3 \
//   cargo run -p spark-storage --features cuda --example snapshot_paging_smoke
//
// Requires a GPU (pinned bounce) + rdma-core. Defaults to 127.0.0.1:9918 (start
// the peer: `atlas-cache-peer --listen 0.0.0.0:9918 --swap-dir /some/nvme/dir`).

#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
fn main() -> anyhow::Result<()> {
    use spark_storage::RdmaSnapshotArena;

    // The pinned RDMA bounce (cuMemAllocHost) needs a current CUDA context; the
    // model serve creates one, so a standalone client must too.
    let _cuda = spark_storage::cuda_min::CudaCtx::new(0)?;

    let addr = std::env::var("ATLAS_SNAP_PEER").unwrap_or_else(|_| "127.0.0.1:9918".into());
    let blob: usize = std::env::var("SMOKE_BLOB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(65536); // 16 × 4 KiB → O_DIRECT-aligned
    let slots: usize = std::env::var("SMOKE_SLOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let n: u64 = std::env::var("SMOKE_KEYS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32); // >> slots → guarantees disk spill

    // put | get | putget (default) — the SSM paging modes (v2, kind=0); or
    // kv | kv-isolation — the KV paging kind (PAGING_MAGIC_V2, kind=1)
    // driven through the real KvPagingBackend StorageBackend client; or
    // decode-isolation — two SSM decode clients (same model, different
    // per-process client salts) on one shared arena (v2, kind=0).
    let mode = std::env::var("SMOKE_MODE").unwrap_or_else(|_| "putget".into());
    if mode == "kv" || mode == "kv-isolation" {
        return kv_main(&addr, blob, slots, n, &mode);
    }
    if mode == "decode-isolation" {
        return decode_isolation_main(&addr, blob, slots, n);
    }

    let arena_bytes = (slots * blob) as u64;
    println!(
        "connecting paging tier @ {addr} [{mode}]: {slots}-slot RAM arena × {blob} B, {n} keys \
         (forces {} disk spills)",
        n.saturating_sub(slots as u64)
    );
    let arena = RdmaSnapshotArena::connect_paging(&addr, arena_bytes, blob)?;

    // Distinct, verifiable pattern per key.
    let pat = |k: u64| -> Vec<u8> {
        let mut v = vec![0u8; blob];
        for (i, b) in v.iter_mut().enumerate() {
            *b = (k as u8) ^ (i as u8).wrapping_mul(31);
        }
        v
    };

    let mut put_ms: Option<f64> = None;
    if mode != "get" {
        let t0 = std::time::Instant::now();
        for k in 0..n {
            arena.paging_put(k, &pat(k))?;
        }
        put_ms = Some(t0.elapsed().as_secs_f64() * 1e3);
    }
    if mode == "put" {
        println!("PUT-only done: {n} keys left resident+spilled in the shared peer arena");
        return Ok(());
    }

    let mut out = vec![0u8; blob];
    let t1 = std::time::Instant::now();
    for k in 0..n {
        let hit = arena.paging_get(k, &mut out)?;
        anyhow::ensure!(
            hit,
            "key {k} MISSING — peer dropped it, or (mode=get) not shared across connections"
        );
        anyhow::ensure!(
            out == pat(k),
            "key {k} CORRUPTED — spill/fault not byte-identical"
        );
    }
    let get_ms = t1.elapsed().as_secs_f64() * 1e3;
    if mode == "get" {
        println!(
            "CROSS-CONNECTION SHARING OK ✅  a SEPARATE client GET all {n} keys a prior `put` \
             run left in the shared peer — {:.1}ms/blob",
            get_ms / n as f64
        );
        return Ok(());
    }
    let put_ms = put_ms.unwrap_or(0.0);

    // Re-GET a definitely-evicted early key to time a cold fault-from-disk.
    let t2 = std::time::Instant::now();
    let _ = arena.paging_get(0, &mut out)?;
    let one_get_us = t2.elapsed().as_micros();

    println!(
        "PAGING SMOKE OK ✅  {n} blobs through {slots}-slot arena, ALL byte-identical after NVMe \
         spill+fault.  put {put_ms:.0}ms ({:.1}/blob)  get {get_ms:.0}ms ({:.1}/blob)  \
         single fault-from-disk {one_get_us}us",
        put_ms / n as f64,
        get_ms / n as f64,
    );
    Ok(())
}

/// SMOKE_MODE=decode-isolation — two SSM DECODE clients (same model
/// fingerprint, DIFFERENT per-process client salts) against ONE shared peer
/// paging arena (v2 handshake, kind=0). Decode cold keys are SLOT COORDINATES
/// (client-local `(seq, logical)` indices), so both clients derive
/// byte-identical pre-namespace keys — only the per-process client salt folded
/// into the decode namespace (`ATLAS_SSM_DECODE_CLIENT_ID` in the model serve)
/// keeps client B from faulting or deleting client A's rollback blobs.
///
/// Phase 1: client A PUTs `n` keys (n ≫ slots forces peer NVMe spills).
/// Phase 2: client B (salt 2) must MISS every key; its removes must not leak.
/// Phase 3: a THIRD connection with A's salt HITs all byte-identically — the
/// control proving phase 2's misses are isolation, not a dead connection.
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
fn decode_isolation_main(addr: &str, blob: usize, slots: usize, n: u64) -> anyhow::Result<()> {
    use atlas_tier::hash::mix64;
    use spark_storage::RdmaSnapshotArena;

    // Mirrors the SHAPE of spark-model's shipped derivation:
    //   ns(salt)  = mix64(mix64(fp, DECODE_DOMAIN), client_salt)
    //   wire(key) = mix64(cold_key, ns)
    // DECODE_DOMAIN_LIT is an illustrative transcription of
    // `atlas_kernels::DECODE_DOMAIN` (spark-storage has no atlas-kernels dep);
    // the property under test is salt isolation on one shared arena, not the
    // production constant's value.
    const DECODE_DOMAIN_LIT: u64 = 0xD3C0_DE12_A5B6_C7D8;
    const FP: u64 = 0xFEED_FACE_CAFE_BEEF; // fixed test "model" (as in kv_main)
    let ns = |salt: u64| mix64(mix64(FP, DECODE_DOMAIN_LIT), salt);
    let (ns_a, ns_b) = (ns(1), ns(2));

    let arena_bytes = (slots * blob) as u64;
    let pat = |k: u64| -> Vec<u8> {
        let mut v = vec![0u8; blob];
        for (i, x) in v.iter_mut().enumerate() {
            *x = (k as u8) ^ (i as u8).wrapping_mul(37) ^ 0x5A;
        }
        v
    };
    println!(
        "connecting SSM decode-isolation @ {addr}: {slots}-slot arena × {blob} B, {n} keys \
         (forces {} disk spills)",
        n.saturating_sub(slots as u64)
    );
    // Client A: PUT everything under its salted namespace.
    let a = RdmaSnapshotArena::connect_paging(addr, arena_bytes, blob)?;
    for k in 0..n {
        a.paging_put(mix64(k, ns_a), &pat(k))?;
    }
    // Client B (same model, different salt): every GET must MISS, and its
    // removes must not touch A's blobs (the cross-client delete hazard).
    let b = RdmaSnapshotArena::connect_paging(addr, arena_bytes, blob)?;
    let mut out = vec![0u8; blob];
    for k in 0..n {
        let hit = b.paging_get(mix64(k, ns_b), &mut out)?;
        anyhow::ensure!(
            !hit,
            "DECODE ISOLATION BROKEN: client B was served client A's rollback blob (key {k})"
        );
        b.paging_remove(mix64(k, ns_b))?;
    }
    // Control: a third connection with A's salt HITs all byte-identically.
    let a2 = RdmaSnapshotArena::connect_paging(addr, arena_bytes, blob)?;
    for k in 0..n {
        let hit = a2.paging_get(mix64(k, ns_a), &mut out)?;
        anyhow::ensure!(
            hit,
            "control MISS on key {k}: blob lost — dead connection, or B's removes \
             leaked across namespaces"
        );
        anyhow::ensure!(
            out == pat(k),
            "control CORRUPTED: key {k} not byte-identical"
        );
    }
    println!(
        "DECODE ISOLATION OK ✅  two client salts on one shared peer arena: B missed all \
         {n} of A's slot-coordinate keys (and B's removes did not leak); a same-salt \
         control connection restored all {n} byte-identical after peer NVMe spills"
    );
    Ok(())
}

/// SMOKE_MODE=kv — drive the KV paging kind end-to-end over real RDMA:
/// v2 handshake (kind=1), block-granular ALLOC/WRITE/COMMIT offloads of far
/// more blocks than the peer RAM arena holds (forcing NVMe spills), then
/// block-granular GET/READ restores into a DEVICE buffer, asserting
/// byte-identity (a fault-from-disk on every evicted block).
///
/// SMOKE_MODE=kv-isolation — two KV clients (same model fp, DIFFERENT client
/// salts) against ONE peer arena: client B must MISS client A's blocks (the
/// KV miss is a hard error naming --swap-cap-gb-kv), both round-trip their
/// own — the client-local-block-id cross-serve hazard, proven on hardware.
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
fn kv_main(addr: &str, blob: usize, slots: usize, n: u64, mode: &str) -> anyhow::Result<()> {
    use spark_storage::backend::BlockReadRequest;
    use spark_storage::backend::StorageBackend;
    use spark_storage::cuda_min::{DeviceBuffer, copy_d_to_h_async, stream_sync};
    use spark_storage::group::{GroupKey, GroupLayout, KvKind};
    use spark_storage::kv_paging::ns::derive_kv_ns;
    use spark_storage::kv_paging::{KvPagingBackend, KvPagingConnect};

    const NKV: u16 = 8;
    anyhow::ensure!(
        blob.is_multiple_of(2 * NKV as usize),
        "SMOKE_BLOB must be a multiple of {} (2·num_kv_heads)",
        2 * NKV
    );
    // A layout whose block_bytes == SMOKE_BLOB exactly (fields are pub; the
    // backend only derives block_bytes/group_id from them).
    let layout = GroupLayout {
        num_layers: 1,
        num_blocks: n as u32,
        num_kv_heads: NKV,
        group_stride: (blob / (2 * NKV as usize)) as u64,
        fs_block_size: 4096,
    };
    assert_eq!(layout.block_bytes() as usize, blob);
    let arena_bytes = (slots * blob) as u64;
    let fp: u64 = 0xFEED_FACE_CAFE_BEEF; // fixed test "model"
    let connect = |salt: u64| -> anyhow::Result<KvPagingBackend> {
        KvPagingBackend::connect(
            addr,
            layout,
            KvPagingConnect {
                arena_bytes,
                ns: derive_kv_ns(fp, &layout, 2, 16, 128, salt),
            },
        )
    };
    let pat = |b: u32, tag: u8| -> Vec<u8> {
        let mut v = vec![0u8; blob];
        for (i, x) in v.iter_mut().enumerate() {
            *x = (b as u8) ^ (i as u8).wrapping_mul(29) ^ tag;
        }
        v
    };
    let key = |b: u32| GroupKey::new(0, b, 0, KvKind::K);
    let dev = DeviceBuffer::new(blob)?;
    let mut host = vec![0u8; blob];
    let read_block = |be: &mut KvPagingBackend, b: u32, host: &mut [u8]| -> anyhow::Result<()> {
        be.read_blocks(
            &[BlockReadRequest {
                base_key: key(b),
                dst_dev_ptr: dev.ptr,
            }],
            0,
        )?;
        copy_d_to_h_async(host.as_mut_ptr() as *mut _, dev.ptr, blob, 0)?;
        stream_sync(0)
    };

    println!(
        "connecting KV paging tier @ {addr} [{mode}]: {slots}-slot arena × {blob} B blocks, \
         {n} blocks (forces {} disk spills)",
        n.saturating_sub(slots as u64)
    );
    if mode == "kv-isolation" {
        let mut a = connect(0x0000_0000_0000_0001)?;
        let mut b = connect(0x0000_0000_0000_0002)?;
        for blk in 0..n as u32 {
            a.write_block_from_host(key(blk), &pat(blk, 0xA0))?;
        }
        // Client B must MISS client A's blocks — the hard error, not bytes.
        let miss = read_block(&mut b, 0, &mut host);
        anyhow::ensure!(
            miss.is_err(),
            "ISOLATION BROKEN: client B was served client A's KV block"
        );
        let msg = format!("{:#}", miss.unwrap_err());
        anyhow::ensure!(
            msg.contains("unrecoverable"),
            "unexpected miss error: {msg}"
        );
        // Both clients round-trip their OWN block 0 on the shared arena.
        b.write_block_from_host(key(0), &pat(0, 0xB0))?;
        read_block(&mut b, 0, &mut host)?;
        anyhow::ensure!(host == pat(0, 0xB0), "client B corrupted");
        read_block(&mut a, 0, &mut host)?;
        anyhow::ensure!(host == pat(0, 0xA0), "client A corrupted by B's write");
        println!(
            "KV ISOLATION OK ✅  two salts on one shared peer arena: B missed A's blocks \
             (hard error), both round-tripped their own byte-identical"
        );
        return Ok(());
    }
    // mode == "kv": spill + rehydrate byte-identity through the real backend.
    let mut be = connect(0x5EED)?;
    let t0 = std::time::Instant::now();
    for blk in 0..n as u32 {
        be.write_block_from_host(key(blk), &pat(blk, 0))?;
    }
    let put_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = std::time::Instant::now();
    for blk in 0..n as u32 {
        read_block(&mut be, blk, &mut host)?;
        anyhow::ensure!(
            host == pat(blk, 0),
            "block {blk} CORRUPTED — spill/fault not byte-identical"
        );
    }
    let get_ms = t1.elapsed().as_secs_f64() * 1e3;
    // Overwrite-in-place (disk-id reuse) survives the peer round-trip.
    be.write_block_from_host(key(0), &pat(0, 0x77))?;
    read_block(&mut be, 0, &mut host)?;
    anyhow::ensure!(host == pat(0, 0x77), "overwrite-in-place corrupted");
    println!(
        "KV PAGING SMOKE OK ✅  {n} blocks through {slots}-slot arena, ALL byte-identical \
         after NVMe spill+fault (+ overwrite-in-place).  put {put_ms:.0}ms ({:.1}/blk)  \
         get {get_ms:.0}ms ({:.1}/blk)",
        put_ms / n as f64,
        get_ms / n as f64,
    );
    Ok(())
}

#[cfg(not(all(feature = "cuda", atlas_rdma_verbs)))]
fn main() {
    eprintln!("snapshot_paging_smoke needs --features cuda + rdma-core (atlas_rdma_verbs)");
    std::process::exit(1);
}
