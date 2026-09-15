// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//! S3 sizing on the real DeepSeek-V4.1 Flash Q2_K shards: the two things a
//! single-Spark generate must stream per token, measured, not estimated.
//!
//!   1. One routed expert = three slices of the stacked expert tensors
//!      (gate/up `[5120, 2304, 384]` Q2_K, down `[2304, 5120, 384]` Q3_K):
//!      bytes on disk, first-touch read from the mmap, CPU dequant to BF16.
//!   2. One token's engram rows: one Q2_K block (84 B) per row, `n_hash_cols`
//!      rows per table per token, gathered at random row indices.
//!
//! Gated `#[ignore]`. Run:
//!   ATLAS_SKIP_BUILD=1 cargo test -p spark-runtime -- --ignored deepseek_v41_stream --nocapture

use std::path::Path;
use std::time::Instant;

use super::container::{GgufFile, TensorInfo};
use super::dequant_cpu::{GgmlType as DqType, dequant_to_bf16};
use super::sidecar;
use crate::weights::{find_gguf, find_gguf_shards};

const MODEL_DIR: &str = "/home/rstesiak/models/dsv41-q2k";

struct Located {
    mmap: memmap2::Mmap,
    gguf: GgufFile,
    info: TensorInfo,
}

/// Find `name` across the shard set; returns the shard's mmap + parsed header + the tensor.
fn locate(name: &str) -> Located {
    let d = Path::new(MODEL_DIR);
    let first = find_gguf(d).expect("shard 0");
    let set = find_gguf_shards(&first).expect("shard set");
    for p in &set.paths {
        let (_f, mmap, gguf) = sidecar::open_gguf(p).expect("open shard");
        if let Some(t) = gguf.tensor(name) {
            let info = t.clone();
            return Located { mmap, gguf, info };
        }
    }
    panic!("{name} not found in any shard");
}

/// The CPU dequant's own type enum (it carries the id-42 group); 128 is irrelevant here.
fn dq_type(l: &Located) -> DqType {
    DqType::from_id(l.info.ggml_type.id(), 128).expect("ggml type")
}

fn block_geom(t: DqType) -> (usize, usize) {
    (t.block_size(), t.block_bytes().expect("block bytes"))
}

/// Raw bytes of expert `e` inside a stacked `[k, n, experts]` tensor (GGUF dims innermost-first).
fn expert_slice<'a>(l: &'a Located, e: usize) -> (&'a [u8], usize) {
    let dims = &l.info.dims;
    assert_eq!(dims.len(), 3, "{}: expected a stacked 3-D expert tensor", l.info.name);
    let (k, n, experts) = (dims[0], dims[1], dims[2]);
    assert!(e < experts);
    let (qk, bb) = block_geom(dq_type(l));
    let elems = k * n;
    assert!(elems.is_multiple_of(qk));
    let bytes = elems / qk * bb;
    let base = l.gguf.tensor_abs_offset(&l.info) + e * bytes;
    (&l.mmap[base..base + bytes], elems)
}

#[test]
#[ignore = "requires the on-disk DeepSeek-V4.1-Flash Q2_K shards"]
fn deepseek_v41_stream_one_expert() {
    let names = ["blk.3.ffn_gate_exps.weight", "blk.3.ffn_up_exps.weight", "blk.3.ffn_down_exps.weight"];
    let located: Vec<Located> = names.iter().map(|n| locate(n)).collect();
    let mut total_bytes = 0usize;
    let mut total_elems = 0usize;
    let mut first_touch_ms = 0f64;
    let mut second_touch_ms = 0f64;
    let mut dequant_ms = 0f64;
    // expert 200: not touched by anything before this test
    let e = 200usize;
    for l in &located {
        let (raw, elems) = expert_slice(l, e);
        total_bytes += raw.len();
        total_elems += elems;
        // first touch: page the slice in from the file
        let t0 = Instant::now();
        let mut acc = 0u64;
        for chunk in raw.chunks(4096) {
            acc = acc.wrapping_add(chunk[0] as u64);
        }
        first_touch_ms += t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        for chunk in raw.chunks(4096) {
            acc = acc.wrapping_add(chunk[1] as u64);
        }
        second_touch_ms += t1.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(acc);
        // CPU dequant to BF16
        let mut out = vec![0u16; elems];
        let t2 = Instant::now();
        dequant_to_bf16(dq_type(l), raw, elems, &mut out).expect("dequant");
        dequant_ms += t2.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&out);
        println!(
            "  {:<28} {:?} dims={:?} slice={} bytes ({:.2} MiB)",
            l.info.name,
            l.info.ggml_type,
            l.info.dims,
            raw.len(),
            raw.len() as f64 / 1048576.0
        );
    }
    let per_token_instances = 6 * 40;
    println!("ONE EXPERT (layer 3, expert {e}): {} bytes = {:.2} MiB on disk, {} elements", total_bytes, total_bytes as f64 / 1048576.0, total_elems);
    println!("  first touch (page-in) {:.1} ms  -> {:.2} GB/s", first_touch_ms, total_bytes as f64 / first_touch_ms / 1e6);
    println!("  second touch (cached) {:.1} ms", second_touch_ms);
    println!("  CPU dequant to BF16    {:.1} ms  -> {:.1} Melem/s single-thread", dequant_ms, total_elems as f64 / dequant_ms / 1e3);
    println!("PER TOKEN at 6 experts x 40 layers = {per_token_instances} instances:");
    println!("  bytes {:.2} GiB; page-in {:.2} s; single-thread dequant {:.1} s", total_bytes as f64 * per_token_instances as f64 / 1073741824.0, first_touch_ms * per_token_instances as f64 / 1e3, dequant_ms * per_token_instances as f64 / 1e3);
    assert_eq!(total_bytes, 46_080 * (84 + 84 + 110), "expert bytes: 46,080 blocks each of Q2_K, Q2_K, Q3_K");
}

#[test]
#[ignore = "requires the on-disk DeepSeek-V4.1-Flash Q2_K shards"]
fn deepseek_v41_stream_engram_rows() {
    let l = locate("blk.1.engram_embd.weight");
    let (qk, bb) = block_geom(dq_type(&l));
    let head_dim = l.info.dims[0];
    let rows = l.info.dims[1];
    assert_eq!((qk, bb, head_dim), (256, 84, 256), "one Q2_K block per engram row");
    let base = l.gguf.tensor_abs_offset(&l.info);
    println!("engram table layer 1: {} rows x {} = {:.2} GiB on disk, {:.2} GiB as BF16", rows, head_dim, (rows * bb) as f64 / 1073741824.0, (rows * head_dim * 2) as f64 / 1073741824.0);
    // deterministic pseudo-random rows, spread over the whole table
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut pick = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % rows as u64) as usize
    };
    let per_token_rows = 24usize; // (max_ngram 4 - 1) * 8 heads
    for &n in &[per_token_rows, 1000] {
        let ids: Vec<usize> = (0..n).map(|_| pick()).collect();
        let t0 = Instant::now();
        let mut out = vec![0u16; head_dim];
        let mut acc = 0u64;
        for &r in &ids {
            let blk = &l.mmap[base + r * bb..base + (r + 1) * bb];
            dequant_to_bf16(dq_type(&l), blk, head_dim, &mut out).expect("row dequant");
            acc = acc.wrapping_add(out[0] as u64);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(acc);
        println!("  gather+dequant {n:>5} random rows: {ms:.1} ms  ({:.3} ms/row, first touch)", ms / n as f64);
    }
}
