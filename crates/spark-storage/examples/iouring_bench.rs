// SPDX-License-Identifier: AGPL-3.0-only
//
// Phase-3 throughput bench for the IoUringBackend. Compares against the
// Phase-0 probe (which measured QD=1 io_uring on 64 KiB random reads at
// 178 MiB/s) by submitting batches of random group-sized reads at varying
// queue depths.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use spark_storage::backend::{IoUringBackend, ReadRequest, StorageBackend};
use spark_storage::cuda_min::{CudaCtx, DeviceBuffer};
use spark_storage::group::{GroupKey, GroupLayout, KvKind};
use spark_storage::layout::Layout;

fn parse_args() -> (PathBuf, u32, u32) {
    let mut dir: Option<PathBuf> = None;
    let mut block_size: u32 = 256; // 256 tokens × 128 dims × 2 bytes = 64 KiB / kv_head
    let mut head_dim: u32 = 128;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = Some(PathBuf::from(args.next().unwrap())),
            "--block-size" => block_size = args.next().unwrap().parse().unwrap(),
            "--head-dim" => head_dim = args.next().unwrap().parse().unwrap(),
            "--help" | "-h" => {
                eprintln!("usage: iouring-bench --dir <path> [--block-size N] [--head-dim N]");
                std::process::exit(0);
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    (dir.expect("--dir required"), block_size, head_dim)
}

fn main() -> Result<()> {
    let (dir, block_size, head_dim) = parse_args();
    std::fs::create_dir_all(&dir)?;
    let _ctx = CudaCtx::new(0)?;
    // Group layout sized to make each group ≈ 64 KiB (matching the Phase-0
    // random-64KiB workload).
    let nkv: u16 = 1;
    let num_blocks: u32 = 1024;
    let spec = GroupLayout::new(1, num_blocks, nkv, block_size, head_dim, 2, 4096);
    let group_bytes = spec.group_bytes() as usize;
    eprintln!("group_bytes = {group_bytes}, blocks = {num_blocks}");
    let layout = Layout::create(&dir, spec)?;

    // Pre-populate with a deterministic pattern via QD=1 backend (one-time).
    {
        let mut backend = IoUringBackend::new(layout, 1)?;
        let pat = vec![0xA5_u8; group_bytes];
        for blk in 0..num_blocks {
            backend.write_from_host(GroupKey::new(0, blk, 0, KvKind::K), &pat)?;
        }
    }

    // Reopen the layout (we consumed it) for each QD experiment.
    for &qd in &[1usize, 2, 4, 8, 16, 32] {
        let layout = Layout::open(&dir, spec)?;
        let mut backend = IoUringBackend::new(layout, qd)?;
        backend.drop_pagecache();

        // Synthesize 4 MiB of random reads as 64 of the 64 KiB groups.
        // For QD=1 this matches the Phase-0 random 64 KiB workload directly.
        let n_iters: usize = 1024;
        let dev = DeviceBuffer::new(group_bytes)?;
        let reqs: Vec<ReadRequest> = (0..n_iters)
            .map(|i| {
                let blk = ((i as u32).wrapping_mul(2_654_435_761)) % num_blocks;
                ReadRequest {
                    group: GroupKey::new(0, blk, 0, KvKind::K),
                    dst_dev_ptr: dev.ptr,
                }
            })
            .collect();

        let t = Instant::now();
        backend.read(&reqs, _ctx.stream)?;
        let dt = t.elapsed().as_secs_f64();
        let mib = (n_iters * group_bytes) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "qd={qd:>3}: {:>8.1} MiB/s ({n_iters} reads of {group_bytes}B in {dt:.3}s)",
            mib / dt
        );
    }

    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
