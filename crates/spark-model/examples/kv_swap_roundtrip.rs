// SPDX-License-Identifier: AGPL-3.0-only
//! GPU receipt for the `kv_batched_swap` row (lane L25): the sequence swap image saved and
//! restored on the real driver through the per-region path (row off: `read_block` /
//! `write_block` per block per layer, blocking SSM copies, exactly what
//! `model/trait_impl/sequence/state_io.rs` does with the row off) and through the batched path
//! (row on: `spark_runtime::kv_cache::block_io::swap::{write_image, read_image}`, the functions
//! the row-on path calls), at several sequence lengths.
//!
//!   AVAROK_TARGET_HW=ada cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example kv_swap_roundtrip -- [--tokens 1024,4096,32768] [--dtype bf16|fp8] \
//!       [--layers 8] [--kv-heads 4] [--head-dim 256] [--ssm-layers 24] \
//!       [--ssm-h-mib 2] [--ssm-conv-kib 96] [--dir /tmp] [--reps 3]
//!
//! The defaults are the Qwen3.5-4B shape (8 full-attention layers, 4 KV heads of 256; 24 GDN
//! layers). Per length it prints the image bytes, the per-region and batched GB/s into a null
//! sink (copies only) and into a file under `--dir` (copies plus the file system), and gates:
//!
//! * G1 the two paths write the same file, byte for byte (planted control: one flipped byte
//!   must be detected);
//! * G2 the batched restore of the per-region file, and the per-region restore of the batched
//!   file, reproduce every KV and SSM byte;
//! * G3 (largest length, null sink, save and restore) batched >= 5x the per-region path and
//!   >= 8 GB/s, the row's flip receipt (master plan 3.5). G3 is reported, never waived.
//!
//! Exit code 1 on any FAIL.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::block_io::swap::{SwapStaging, read_image, write_image};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

struct Args {
    tokens: Vec<usize>,
    dtype: KvCacheDtype,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    ssm_layers: usize,
    ssm_h: usize,
    ssm_conv: usize,
    dir: PathBuf,
    reps: usize,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        tokens: vec![1024, 4096, 32768],
        dtype: KvCacheDtype::Bf16,
        layers: 8,
        kv_heads: 4,
        head_dim: 256,
        ssm_layers: 24,
        ssm_h: 2 << 20,
        ssm_conv: 96 << 10,
        dir: std::env::temp_dir(),
        reps: 3,
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
            "--dtype" => {
                a.dtype = match val()?.as_str() {
                    "bf16" => KvCacheDtype::Bf16,
                    "fp8" => KvCacheDtype::Fp8,
                    other => bail!("--dtype {other}: expected bf16 or fp8"),
                }
            }
            "--layers" => a.layers = val()?.parse()?,
            "--kv-heads" => a.kv_heads = val()?.parse()?,
            "--head-dim" => a.head_dim = val()?.parse()?,
            "--ssm-layers" => a.ssm_layers = val()?.parse()?,
            "--ssm-h-mib" => a.ssm_h = val()?.parse::<usize>()? << 20,
            "--ssm-conv-kib" => a.ssm_conv = val()?.parse::<usize>()? << 10,
            "--dir" => a.dir = PathBuf::from(val()?),
            "--reps" => a.reps = val()?.parse::<usize>()?.max(1),
            other => bail!("unknown flag {other}"),
        }
    }
    if a.tokens.is_empty() {
        bail!("--tokens is empty");
    }
    Ok(a)
}

struct Gates {
    failed: usize,
}

impl Gates {
    fn check(&mut self, name: &str, ok: bool, detail: &str) {
        println!("{} {name}: {detail}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            self.failed += 1;
        }
    }
}

fn pattern(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

/// One sequence on the card: its block table, KV contents and SSM buffers.
struct Seq {
    table: Vec<u32>,
    ssm: Vec<(DevicePtr, usize, DevicePtr, usize)>,
}

fn regions(kv: &PagedKvCache, s: &Seq) -> Result<Vec<(DevicePtr, usize)>> {
    let mut r: Vec<(DevicePtr, usize)> = Vec::new();
    for b in kv.table_regions(&s.table)? {
        r.extend(b.regions);
    }
    for &(h, hn, c, cn) in &s.ssm {
        r.push((h, hn));
        r.push((c, cn));
    }
    Ok(r)
}

/// The row-off writer: per block per layer `read_block`, then blocking SSM copies.
fn legacy_save(g: &dyn GpuBackend, kv: &PagedKvCache, s: &Seq, w: &mut dyn Write) -> Result<()> {
    let mut bufs = Vec::with_capacity(s.table.len() * kv.num_layers());
    for &b in &s.table {
        for l in 0..kv.num_layers() {
            bufs.push(kv.read_block(l, b, g)?);
        }
    }
    for (k, v) in &bufs {
        w.write_all(k)?;
        w.write_all(v)?;
    }
    for &(h, hn, c, cn) in &s.ssm {
        let mut hb = vec![0u8; hn];
        let mut cb = vec![0u8; cn];
        g.copy_d2h(h, &mut hb)?;
        g.copy_d2h(c, &mut cb)?;
        w.write_all(&hb)?;
        w.write_all(&cb)?;
    }
    w.flush()?;
    Ok(())
}

/// The row-off reader into the existing blocks of `s` (symmetric dtypes: K stride both sides).
fn legacy_restore(g: &dyn GpuBackend, kv: &PagedKvCache, s: &Seq, r: &mut dyn Read) -> Result<()> {
    let strides: Vec<usize> = (0..kv.num_layers())
        .map(|l| kv.block_stride_bytes_for_layer(l))
        .collect();
    let mut bufs = Vec::with_capacity(s.table.len() * strides.len());
    for _ in &s.table {
        for &st in &strides {
            let mut k = vec![0u8; st];
            let mut v = vec![0u8; st];
            r.read_exact(&mut k)?;
            r.read_exact(&mut v)?;
            bufs.push((k, v));
        }
    }
    let mut i = 0;
    for &b in &s.table {
        for l in 0..strides.len() {
            kv.write_block(l, b, &bufs[i].0, &bufs[i].1, g)?;
            i += 1;
        }
    }
    for &(h, hn, c, cn) in &s.ssm {
        let mut hb = vec![0u8; hn];
        let mut cb = vec![0u8; cn];
        r.read_exact(&mut hb)?;
        r.read_exact(&mut cb)?;
        g.copy_h2d(&hb, h)?;
        g.copy_h2d(&cb, c)?;
    }
    Ok(())
}

/// Every byte of `s` as one image (the oracle, via the per-region reads).
fn snapshot(g: &dyn GpuBackend, kv: &PagedKvCache, s: &Seq) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    legacy_save(g, kv, s, &mut out)?;
    Ok(out)
}

fn best_secs(reps: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f()?;
        best = best.min(t.elapsed().as_secs_f64());
    }
    Ok(best)
}

fn gbps(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs.max(1e-9) / 1e9
}

fn file_in(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("atlas-kv-swap-{}-{name}.bin", std::process::id()))
}

fn run_length(
    // `'static`: `PagedKvCache` releases through `ModelResource<dyn GpuBackend>`.
    g: &(dyn GpuBackend + 'static),
    a: &Args,
    tokens: usize,
    last: bool,
    gates: &mut Gates,
) -> Result<()> {
    let block = 16usize;
    let blocks = tokens.div_ceil(block);
    let cfg = KvCacheConfig {
        block_size: block,
        num_kv_heads: a.kv_heads,
        head_dim: a.head_dim,
        num_layers: a.layers,
        dtype: a.dtype,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let mut kv = PagedKvCache::new(cfg, 2 * blocks, g)?;
    let mk = |kv: &mut PagedKvCache, seed: u64| -> Result<Seq> {
        let table: Vec<u32> = (0..blocks)
            .map(|_| kv.alloc_block())
            .collect::<Result<_>>()?;
        let mut ssm = Vec::new();
        for i in 0..a.ssm_layers {
            let (h, c) = (g.alloc(a.ssm_h)?, g.alloc(a.ssm_conv)?);
            g.copy_h2d(&pattern(seed + 2 * i as u64, a.ssm_h), h)?;
            g.copy_h2d(&pattern(seed + 2 * i as u64 + 1, a.ssm_conv), c)?;
            ssm.push((h, a.ssm_h, c, a.ssm_conv));
        }
        Ok(Seq { table, ssm })
    };
    let src = mk(&mut kv, 1)?;
    for &b in &src.table {
        for l in 0..kv.num_layers() {
            let k = pattern(
                u64::from(b) * 97 + l as u64,
                kv.k_block_stride_bytes_for_layer(l),
            );
            let v = pattern(
                u64::from(b) * 97 + l as u64 + 50,
                kv.v_block_stride_bytes_for_layer(l),
            );
            kv.write_block(l, b, &k, &v, g)?;
        }
    }
    let dst = mk(&mut kv, 9_000)?;
    let staging = SwapStaging::default();
    let stream = g.default_stream();
    let src_regions = regions(&kv, &src)?;
    let dst_regions = regions(&kv, &dst)?;
    let want = snapshot(g, &kv, &src)?;
    let bytes = want.len();

    // G1: identical files, plus the planted control.
    let mut batched_file = Vec::with_capacity(bytes);
    write_image(g, &src_regions, &staging, &mut batched_file, stream)?;
    gates.check(
        &format!("G1 {tokens} tok same file"),
        batched_file == want,
        &format!("{bytes} B"),
    );
    let mut planted = want.clone();
    planted[bytes / 2] ^= 0x01;
    gates.check(
        &format!("G1 {tokens} tok planted control"),
        planted != batched_file,
        "a flipped byte is detected",
    );

    // G2: cross restores reproduce every byte.
    read_image(g, &dst_regions, &staging, &mut want.as_slice(), stream)?;
    gates.check(
        &format!("G2 {tokens} tok batched reads legacy"),
        snapshot(g, &kv, &dst)? == want,
        "KV + SSM",
    );
    legacy_restore(g, &kv, &dst, &mut batched_file.as_slice())?;
    gates.check(
        &format!("G2 {tokens} tok legacy reads batched"),
        snapshot(g, &kv, &dst)? == want,
        "KV + SSM",
    );
    drop(batched_file);

    // Timing into a null sink (copies only).
    let t_ls = best_secs(a.reps, || legacy_save(g, &kv, &src, &mut std::io::sink()))?;
    let t_bs = best_secs(a.reps, || {
        write_image(g, &src_regions, &staging, &mut std::io::sink(), stream).map(|_| ())
    })?;
    let t_lr = best_secs(a.reps, || {
        legacy_restore(g, &kv, &dst, &mut want.as_slice())
    })?;
    let t_br = best_secs(a.reps, || {
        read_image(g, &dst_regions, &staging, &mut want.as_slice(), stream).map(|_| ())
    })?;
    println!(
        "info {tokens} tok {bytes} B sink: save legacy {:.2} GB/s batched {:.2} GB/s ({:.1}x); \
         restore legacy {:.2} GB/s batched {:.2} GB/s ({:.1}x); copy_batch={}",
        gbps(bytes, t_ls),
        gbps(bytes, t_bs),
        t_ls / t_bs,
        gbps(bytes, t_lr),
        gbps(bytes, t_br),
        t_lr / t_br,
        g.supports_copy_batch()
    );

    // Timing through a file (information: includes the file system).
    let path = file_in(&a.dir, "legacy");
    let t_lf = best_secs(1, || {
        legacy_save(
            g,
            &kv,
            &src,
            &mut BufWriter::new(std::fs::File::create(&path)?),
        )
    })?;
    let path_b = file_in(&a.dir, "batched");
    let t_bf = best_secs(1, || {
        write_image(
            g,
            &src_regions,
            &staging,
            &mut BufWriter::new(std::fs::File::create(&path_b)?),
            stream,
        )
        .map(|_| ())
    })?;
    let t_brf = best_secs(1, || {
        read_image(
            g,
            &dst_regions,
            &staging,
            &mut BufReader::new(std::fs::File::open(&path_b)?),
            stream,
        )
        .map(|_| ())
    })?;
    let same = std::fs::read(&path)? == std::fs::read(&path_b)?;
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&path_b);
    gates.check(
        &format!("G1 {tokens} tok same file on disk"),
        same,
        &a.dir.display().to_string(),
    );
    println!(
        "info {tokens} tok file: save legacy {:.2} GB/s batched {:.2} GB/s; restore batched {:.2} GB/s",
        gbps(bytes, t_lf),
        gbps(bytes, t_bf),
        gbps(bytes, t_brf)
    );

    if last {
        let ok_save = t_ls / t_bs >= 5.0 && gbps(bytes, t_bs) >= 8.0;
        let ok_restore = t_lr / t_br >= 5.0 && gbps(bytes, t_br) >= 8.0;
        gates.check(
            &format!("G3 {tokens} tok save"),
            ok_save,
            &format!(
                "{:.1}x, {:.2} GB/s (need >= 5x and >= 8 GB/s)",
                t_ls / t_bs,
                gbps(bytes, t_bs)
            ),
        );
        gates.check(
            &format!("G3 {tokens} tok restore"),
            ok_restore,
            &format!(
                "{:.1}x, {:.2} GB/s (need >= 5x and >= 8 GB/s)",
                t_lr / t_br,
                gbps(bytes, t_br)
            ),
        );
    }

    staging.free(g);
    for s in [&src, &dst] {
        for &(h, _, c, _) in &s.ssm {
            g.free(h)?;
            g.free(c)?;
        }
    }
    avarok_core::scope::ModelResource::release(&mut kv, g)?;
    Ok(())
}

fn main() -> Result<()> {
    let a = parse_args()?;
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    println!(
        "kv_swap_roundtrip: dtype={} layers={} kv_heads={} head_dim={} ssm_layers={} \
         supports_copy_batch={}",
        a.dtype,
        a.layers,
        a.kv_heads,
        a.head_dim,
        a.ssm_layers,
        g.supports_copy_batch()
    );
    let mut gates = Gates { failed: 0 };
    let last = a.tokens.len() - 1;
    let mut sorted = a.tokens.clone();
    sorted.sort_unstable();
    for (i, &t) in sorted.iter().enumerate() {
        run_length(g, &a, t, i == last, &mut gates)?;
    }
    if gates.failed > 0 {
        println!("kv_swap_roundtrip: {} gate(s) FAILED", gates.failed);
        std::process::exit(1);
    }
    println!("kv_swap_roundtrip: all gates PASS");
    Ok(())
}
