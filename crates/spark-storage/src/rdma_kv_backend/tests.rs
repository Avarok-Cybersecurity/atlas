// SPDX-License-Identifier: AGPL-3.0-only
//
// Tests for the RDMA KV backend. Moved out of an inline `#[cfg(test)] mod tests`
// per the repo convention (tests live in their own files). Pure code motion.

use super::*;
use crate::cuda_min::CudaCtx;
use crate::group::KvKind;

// Read a UMA (pinned, host==dev VA) dst's bytes host-side — valid for both
// the bounce path (copy_h2d lands there) and zero-copy (RDMA lands there).
unsafe fn uma_bytes(buf: &PinnedBuffer, n: usize) -> &[u8] {
    unsafe { std::slice::from_raw_parts(buf.ptr as *const u8, n) }
}

#[test]
#[ignore = "requires GPU + live cache-peer at $ATLAS_KV_PEER"]
fn rdma_kv_round_trip() {
    let _ctx = CudaCtx::new(0).expect("cuda init");
    let peer = std::env::var("ATLAS_KV_PEER").expect("set ATLAS_KV_PEER=host:port");
    let layout = GroupLayout::new(2, 4, 2, 16, 128, 2, 4096);
    let bytes = layout.group_bytes() as usize;
    let mut be = RdmaKvBackend::connect(&peer, layout).expect("connect kv peer");
    let keys = [
        GroupKey::new(0, 0, 0, KvKind::K),
        GroupKey::new(0, 3, 1, KvKind::V),
        GroupKey::new(1, 2, 0, KvKind::V),
        GroupKey::new(1, 0, 1, KvKind::K),
    ];
    let pat = |i: usize| -> Vec<u8> { (0..bytes).map(|b| ((b + i * 37) & 0xFF) as u8).collect() };
    for (i, k) in keys.iter().enumerate() {
        be.write_from_host(*k, &pat(i)).expect("write_from_host");
    }
    // UMA dsts so the same test validates both the bounce and zero-copy paths.
    let devs: Vec<_> = keys
        .iter()
        .map(|_| PinnedBuffer::new(bytes).unwrap())
        .collect();
    let reqs: Vec<_> = keys
        .iter()
        .zip(&devs)
        .map(|(k, d)| ReadRequest {
            group: *k,
            dst_dev_ptr: d.device_ptr().unwrap(),
        })
        .collect();
    be.read(&reqs, _ctx.stream).expect("read");
    for (i, d) in devs.iter().enumerate() {
        let back = unsafe { uma_bytes(d, bytes) };
        assert_eq!(
            back,
            &pat(i)[..],
            "group {:?} corrupted through the RDMA blade",
            keys[i]
        );
    }
}

#[test]
#[ignore = "requires GPU + live cache-peer at $ATLAS_KV_PEER"]
fn rdma_kv_bandwidth() {
    let ctx = CudaCtx::new(0).expect("cuda init");
    let peer = std::env::var("ATLAS_KV_PEER").expect("set ATLAS_KV_PEER=host:port");
    let layout = GroupLayout::new(16, 64, 8, 64, 128, 2, 4096);
    let gbytes = layout.group_bytes() as usize;
    let mut be = RdmaKvBackend::connect(&peer, layout).expect("connect kv peer");
    let ngroups =
        (layout.num_layers as u64) * 2 * (layout.num_blocks as u64) * (layout.num_kv_heads as u64);
    let total = ngroups * gbytes as u64;
    let keys: Vec<GroupKey> = (0..layout.num_layers)
        .flat_map(|l| {
            (0..layout.num_blocks).flat_map(move |b| {
                (0..layout.num_kv_heads).flat_map(move |h| {
                    [
                        GroupKey::new(l, b, h, KvKind::K),
                        GroupKey::new(l, b, h, KvKind::V),
                    ]
                })
            })
        })
        .collect();
    let src = vec![0xABu8; gbytes];
    // UMA dst so zero-copy (ATLAS_KV_ZERO_COPY=1) can RDMA straight in.
    let dst = PinnedBuffer::new(gbytes).unwrap();
    let dptr = dst.device_ptr().unwrap();

    let t0 = std::time::Instant::now();
    for k in &keys {
        be.write_from_host(*k, &src).expect("write");
    }
    for rail in &mut be.rails {
        rail.drain(gbytes, 0).expect("drain");
    }
    let wdt = t0.elapsed().as_secs_f64();

    let reqs: Vec<_> = keys
        .iter()
        .map(|k| ReadRequest {
            group: *k,
            dst_dev_ptr: dptr,
        })
        .collect();
    let t1 = std::time::Instant::now();
    be.read(&reqs, ctx.stream).expect("read");
    let rdt = t1.elapsed().as_secs_f64();

    let gbps = |dt: f64| (total as f64) / dt / 1e9;
    println!(
        "\nRDMA KV tier ({} rail(s), {}, pipelined): {} groups × {} B = {:.0} MiB\n  \
         OFFLOAD (RDMA WRITE): {:.3}s => {:.2} GB/s\n  \
         RESTORE (RDMA READ): {:.3}s => {:.2} GB/s",
        be.rails.len(),
        if be.zero_copy {
            "zero-copy"
        } else {
            "bounce+h2d"
        },
        ngroups,
        gbytes,
        total as f64 / 1048576.0,
        wdt,
        gbps(wdt),
        rdt,
        gbps(rdt),
    );
}
