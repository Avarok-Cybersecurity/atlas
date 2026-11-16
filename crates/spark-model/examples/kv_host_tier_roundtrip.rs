// SPDX-License-Identifier: AGPL-3.0-only
//! GPU receipt for the host-RAM KV tier core (lane L26a, `spark_runtime::kv_host_tier`): the
//! tier on the real driver, over lane L25's pinned pool, with its own copy stream, events and
//! batched copies. No model is loaded: device blocks are carved from one allocation in the
//! paged-KV layout (per layer and side, block after block), filled with a per-block pattern,
//! stored, overwritten on the device, then restored into fresh blocks and compared byte for byte
//! against the CPU oracle (the pattern).
//!
//!   AVAROK_TARGET_HW=ada cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example kv_host_tier_roundtrip -- [--tokens 1024,4096,32768] [--block-tokens 16] \
//!       [--layers 8] [--kv-heads 4] [--head-dim 256] [--elem-bytes 2] [--budget-frac 1.0]
//!
//! Defaults are the Qwen3.5-4B attention shape with BF16 KV. Per length it gates:
//!
//! * G1 nothing was visible before its D2H copy was published, and every block the tier still
//!   holds restores byte-exact against the pattern;
//! * G2 the tier stays within its budget: with `--budget-frac 1` it holds every block (the
//!   leading run is the whole prompt); below 1 it evicted, and the blocks it holds are exactly
//!   its entries;
//! * G3 every host run went back to the pool after `shutdown`.
//!
//! It also prints the D2H (store) and H2D (restore) GB/s seen through event completion
//! (information; the tier's flip receipt is lane L26b's TTFT run). Exit code 1 on any FAIL.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::block_io::BlockRegions;
use spark_runtime::kv_host_tier::{
    BlockKey, KvHostTier, PoolArena, RestoreState, TierConfig, TierOptions, TierPolicy,
};
use spark_runtime::pinned_pool::PinnedPool;

struct Args {
    tokens: Vec<usize>,
    block_tokens: usize,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    elem_bytes: usize,
    budget_frac: f64,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        tokens: vec![1024, 4096, 32768],
        block_tokens: 16,
        layers: 8,
        kv_heads: 4,
        head_dim: 256,
        elem_bytes: 2,
        budget_frac: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--tokens" => {
                a.tokens = val()?
                    .split(',')
                    .map(|s| s.trim().parse::<usize>())
                    .collect::<std::result::Result<_, _>>()?
            }
            "--block-tokens" => a.block_tokens = val()?.parse()?,
            "--layers" => a.layers = val()?.parse()?,
            "--kv-heads" => a.kv_heads = val()?.parse()?,
            "--head-dim" => a.head_dim = val()?.parse()?,
            "--elem-bytes" => a.elem_bytes = val()?.parse()?,
            "--budget-frac" => a.budget_frac = val()?.parse()?,
            other => bail!("unknown flag {other}"),
        }
    }
    if a.tokens.is_empty() || a.block_tokens == 0 || a.layers == 0 {
        bail!("--tokens, --block-tokens and --layers must be non-empty and positive");
    }
    if !(a.budget_frac > 0.0 && a.budget_frac <= 1.0) {
        bail!("--budget-frac must be in (0, 1]");
    }
    Ok(a)
}

struct Gates {
    failed: usize,
}

impl Gates {
    fn check(&mut self, name: &str, ok: bool, detail: &str) {
        if ok {
            println!("PASS {name}: {detail}");
        } else {
            println!("FAIL {name}: {detail}");
            self.failed += 1;
        }
    }
}

/// The CPU oracle: byte `i` of block `b`'s image.
fn pattern(b: usize, i: usize) -> u8 {
    (b.wrapping_mul(2_654_435_761) ^ i.wrapping_mul(40_503) ^ (i >> 7)) as u8
}

/// `n` blocks carved from one allocation, paged-KV layout: region (layer, side) of block `b` at
/// `((layer * 2 + side) * n + b) * region` bytes.
fn carve(base: DevicePtr, n: usize, layers: usize, region: usize) -> Vec<BlockRegions> {
    (0..n)
        .map(|b| BlockRegions {
            regions: (0..layers * 2)
                .map(|ls| (DevicePtr(base.0 + ((ls * n + b) * region) as u64), region))
                .collect(),
        })
        .collect()
}

/// The device image laid out so that each block's regions (K then V per layer) read `pattern`.
fn device_image(n: usize, layers: usize, region: usize) -> Vec<u8> {
    let mut img = vec![0u8; n * layers * 2 * region];
    for b in 0..n {
        for ls in 0..layers * 2 {
            let off = (ls * n + b) * region;
            for j in 0..region {
                img[off + j] = pattern(b, ls * region + j);
            }
        }
    }
    img
}

fn read_block(g: &dyn GpuBackend, blk: &BlockRegions) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(blk.bytes());
    for &(p, bytes) in &blk.regions {
        let mut buf = vec![0u8; bytes];
        g.copy_d2h(p, &mut buf)?;
        out.extend_from_slice(&buf);
    }
    Ok(out)
}

fn gbps(bytes: usize, d: Duration) -> f64 {
    bytes as f64 / d.as_secs_f64().max(1e-9) / 1e9
}

fn run_length(gpu: &Arc<dyn GpuBackend>, a: &Args, tokens: usize, gates: &mut Gates) -> Result<()> {
    let g = gpu.as_ref();
    let n = tokens / a.block_tokens;
    let region = a.block_tokens * a.kv_heads * a.head_dim * a.elem_bytes;
    let block_bytes = region * a.layers * 2;
    let total = n * block_bytes;
    // `--budget-frac` is the share of the stored bytes that fits under the tier's high watermark
    // (eviction starts there), plus one block of slack.
    let high = TierOptions::default().high_watermark;
    let budget = ((total as f64 * a.budget_frac / high) as u64 + block_bytes as u64).max(1 << 20);
    println!("--- {tokens} tokens: {n} blocks of {block_bytes} B, budget {budget} B");
    let pool = PinnedPool::new(gpu.clone(), budget as usize, budget as usize, 64 << 20)?;
    let copy_stream = g.create_copy_stream()?;
    let cfg = TierConfig {
        budget_bytes: budget,
        writeback_bytes_per_step: u64::MAX,
        min_tokens: 1,
        policy: TierPolicy::Lru,
        restore_gate_ratio: 0.5,
    };
    let opts = TierOptions {
        register_metrics: false,
        namespace: 0x4B56_5449_4552,
        ..TierOptions::default()
    };
    let tier = KvHostTier::with_arena(
        cfg,
        Arc::new(PoolArena::new(pool.clone())),
        block_bytes,
        copy_stream,
        opts,
    );
    let src_base = g.alloc(total)?;
    let dst_base = g.alloc(total)?;
    g.copy_h2d(&device_image(n, a.layers, region), src_base)?;
    let src = carve(src_base, n, a.layers, region);
    let dst = carve(dst_base, n, a.layers, region);
    let toks: Vec<u32> = (0..(n * a.block_tokens) as u32).collect();
    let keys = BlockKey::chain_blocks(tier.options().namespace, 0, &toks, a.block_tokens);
    let chunks: Vec<&[u32]> = toks.chunks_exact(a.block_tokens).collect();

    // Store in calls under the staging cap, publishing as events complete.
    let t0 = Instant::now();
    let per_call = ((budget / 2) / block_bytes as u64).max(1) as usize;
    let mut stored = 0usize;
    let mut early_hits = 0usize;
    for start in (0..n).step_by(per_call) {
        let end = (start + per_call).min(n);
        stored += tier.store(g, &keys[start..end], &chunks[start..end], &src[start..end])?;
        early_hits += tier.lookup(&keys[start..end], &chunks[start..end]);
        let deadline = Instant::now() + Duration::from_secs(30);
        while tier.snapshot().staging_bytes > 0 && Instant::now() < deadline {
            tier.poll_writebacks(g);
        }
        tier.poll_writebacks(g);
    }
    let store_time = t0.elapsed();
    let s = tier.snapshot();
    println!(
        "info store: {stored} blocks started, {} published, {} evicted, {:.2} GB/s through completion",
        s.writeback_blocks,
        s.evictions,
        gbps(s.writeback_blocks as usize * block_bytes, store_time)
    );
    // Publication happens only in poll_writebacks, so a lookup right after a store call (before
    // any poll) can never see the blocks that call started.
    gates.check(
        "G1-visibility",
        early_hits == 0,
        &format!("{early_hits} hits on just-stored blocks before their copies were published"),
    );

    // The GPU copies are gone (blocks reused).
    g.memset(src_base, 0, total)?;
    let leading = tier.lookup(&keys, &chunks);
    let resident: Vec<usize> = (0..n)
        .filter(|&b| tier.lookup(&keys[b..b + 1], &chunks[b..b + 1]) == 1)
        .collect();
    let s = tier.snapshot();
    let fits = s.resident_bytes <= budget && resident.len() == s.entries;
    let shape_ok = if a.budget_frac < 1.0 {
        s.evictions > 0 && leading <= resident.len()
    } else {
        leading == n && resident.len() == n
    };
    gates.check(
        "G2-budget",
        fits && shape_ok,
        &format!(
            "{leading} leading hits, {} blocks held ({} B of {budget} B), {} evicted",
            resident.len(),
            s.resident_bytes,
            s.evictions
        ),
    );
    let mut restored_ok = true;
    let t1 = Instant::now();
    let mut done = 0usize;
    while done < resident.len() {
        let take = tier.max_restore_blocks().min(resident.len() - done).max(1);
        let idx = &resident[done..done + take];
        let k: Vec<BlockKey> = idx.iter().map(|&b| keys[b]).collect();
        let d: Vec<BlockRegions> = idx.iter().map(|&b| dst[b].clone()).collect();
        let t = tier.begin_restore(g, &k, &d)?;
        let deadline = Instant::now() + Duration::from_secs(30);
        let state = loop {
            match tier.poll(g, &t)? {
                RestoreState::Pending if Instant::now() < deadline => continue,
                other => break other,
            }
        };
        tier.release(t);
        if state != RestoreState::Ready {
            restored_ok = false;
            break;
        }
        done += take;
    }
    let restore_time = t1.elapsed();
    println!(
        "info restore: {done} blocks, {:.2} GB/s through completion",
        gbps(done * block_bytes, restore_time)
    );
    let mut mismatched = 0usize;
    for &b in resident.iter().take(done) {
        let got = read_block(g, &dst[b])?;
        if got.iter().enumerate().any(|(i, &v)| v != pattern(b, i)) {
            mismatched += 1;
        }
    }
    gates.check(
        "G1-bytes",
        restored_ok && mismatched == 0 && done == resident.len(),
        &format!(
            "{done} of {} held blocks restored, {mismatched} differ from the oracle",
            resident.len()
        ),
    );
    tier.shutdown(g);
    drop(tier);
    let used = pool.stats().used;
    gates.check(
        "G3-runs-returned",
        used == 0,
        &format!("{used} pool bytes still in use"),
    );
    g.free(src_base)?;
    g.free(dst_base)?;
    Ok(())
}

fn main() -> Result<()> {
    let a = parse_args()?;
    let backend: Arc<dyn GpuBackend> =
        Arc::new(AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?);
    println!(
        "kv_host_tier_roundtrip: layers={} kv_heads={} head_dim={} elem_bytes={} block_tokens={} \
         budget_frac={} supports_copy_batch={}",
        a.layers,
        a.kv_heads,
        a.head_dim,
        a.elem_bytes,
        a.block_tokens,
        a.budget_frac,
        backend.supports_copy_batch()
    );
    let mut gates = Gates { failed: 0 };
    for &t in &a.tokens {
        run_length(&backend, &a, t, &mut gates)?;
    }
    if gates.failed > 0 {
        println!("kv_host_tier_roundtrip: {} gate(s) FAILED", gates.failed);
        std::process::exit(1);
    }
    println!("kv_host_tier_roundtrip: all gates PASS");
    Ok(())
}
