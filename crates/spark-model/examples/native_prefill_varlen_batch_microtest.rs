// SPDX-License-Identifier: AGPL-3.0-only
//! CHUNK-ZERO batched (varlen) prefill attention vs the per-stream path (#927).
//!
//! The H100 round-11 nsys trace showed a 16-way burst of 1193-token prompts
//! prefilled STRICTLY one prompt at a time — 64 forwards for 32 requests, with
//! `--max-prefill-tokens 8192` (six prompts would have fit) never binding. The
//! documented fix, `--prefill-varlen-batch`, could not engage because the
//! attention layer refused `seq_len_start == 0`: the wave that would fix the
//! serialisation was exactly the wave it declined.
//!
//! With that guard now reading the same predicate as admission, a co-arriving
//! burst of FRESH prompts goes through `inferspark_prefill_paged_batched` with
//! `q_offset = 0` and per-stream `cu_seqlens` / `kv_lens`. This microtest is the
//! receipt that doing so changes nothing per sequence.
//!
//! CONTRACT (gated, exit 1 on failure):
//!   1. BITS. For each of four RAGGED synthetic sequences at chunk 0, the
//!      batched kernel's output must equal the single-stream kernel's output on
//!      the same Q/K/V BYTE FOR BYTE. Both arms run the same
//!      `prefill_paged_compute.cuh` body at BR=32 with the same per-accumulator
//!      operand order; only `blockIdx.z` and the Q/O base offset differ. So the
//!      question is not "how close" — anything but `unequal = 0` is a bug.
//!      (Documented exception, none expected here: at `chunk_len >= 256` the
//!      dispatcher selects the BR=64 twin on both arms, which re-brackets the
//!      online-softmax rescale across a wider row tile. This test stays under
//!      256 so both arms are BR=32 and the comparison is exact.)
//!   2. KV. The K/V pool is byte-identical before and after both arms — the
//!      chunk-0 batched path READS the pages a fresh sequence just wrote, it
//!      must not write them again. A per-sequence digest is compared, so a
//!      cross-stream write lands on a different sequence's digest and is named.
//!   3. EXTENT. Guard bands either side of every output buffer are untouched,
//!      and stream `b`'s packed output region is written only by stream `b`
//!      (checked by running the batch twice with one stream's Q perturbed and
//!      asserting BOTH that that stream's rows move AND that no other stream's
//!      do). The perturbation is a whole head of the victim's LAST query row:
//!      the FIRST row attends to exactly one key under causal masking, so its
//!      softmax weight is 1.0 and its output is `V[0]` whatever Q holds — a
//!      control placed there cannot trip on any hardware, which is how round 13
//!      spent an H100 slot on `the harness is inert`.
//!
//! Run (H100 / any CUDA box with the kernels built):
//!   cargo run --release -p spark-model --features cuda,gpu-examples \
//!     --example native_prefill_varlen_batch_microtest

use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// Qwen3.8-27B attention geometry (`kernels/hopper/qwen3.8-27b/MODEL.toml`).
const NQ: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
/// KV paging granularity — the same 16 the serve uses, so the block tables the
/// two arms walk are the production shape.
const BS: usize = 16;
/// Four ragged fresh prompts. Deliberately not multiples of `BS` (a real burst
/// is never block-aligned) and all < 256 so both arms take the BR=32 kernel.
const LENS: [usize; 4] = [96, 163, 208, 49];
const GUARD: usize = 256; // bytes of sentinel either side of every buffer
const SENTINEL: u8 = 0x5a;

struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // [-1, 1), coarse on purpose: bf16 has 8 mantissa bits, so a finer
        // source would only be rounded away and would hide nothing.
        (((self.0 >> 40) as f32) / ((1u32 << 23) as f32)) * 2.0 - 1.0
    }
}

/// Allocate `bytes` with sentinel guard bands, returning the payload pointer.
fn alloc_guarded(gpu: &dyn GpuBackend, bytes: usize) -> Result<(DevicePtr, DevicePtr, usize)> {
    let total = bytes + 2 * GUARD;
    let base = gpu.alloc(total)?;
    gpu.memset(base, SENTINEL, total)?;
    Ok((base.offset(GUARD), base, total))
}

fn check_guards(gpu: &dyn GpuBackend, base: DevicePtr, total: usize, name: &str) -> Result<()> {
    let mut head = vec![0u8; GUARD];
    let mut tail = vec![0u8; GUARD];
    gpu.copy_d2h(base, &mut head)?;
    gpu.copy_d2h(base.offset(total - GUARD), &mut tail)?;
    ensure!(
        head.iter().all(|&b| b == SENTINEL) && tail.iter().all(|&b| b == SENTINEL),
        "{name}: a kernel wrote outside its buffer",
    );
    Ok(())
}

fn upload_bf16(gpu: &dyn GpuBackend, data: &[bf16]) -> Result<(DevicePtr, DevicePtr, usize)> {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    let (p, base, total) = alloc_guarded(gpu, bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok((p, base, total))
}

fn upload_u32(gpu: &dyn GpuBackend, data: &[u32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len().max(4))?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

fn upload_u64(gpu: &dyn GpuBackend, data: &[u64]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len().max(8))?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

fn download_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// FNV-1a over raw bytes — a digest, not a checksum with pretensions.
fn digest(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

struct Kernels {
    single: KernelHandle,
    batched: KernelHandle,
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let k = Kernels {
        single: gpu.kernel("prefill_paged", "inferspark_prefill_paged")?,
        batched: gpu.kernel(
            "inferspark_prefill_paged_batched",
            "inferspark_prefill_paged_batched",
        )?,
    };

    let n = LENS.len();
    let total_tokens: usize = LENS.iter().sum();
    let max_len = *LENS.iter().max().unwrap();
    let inv_sqrt_d = 1.0f32 / (HD as f32).sqrt();

    // ── Paged K/V pool ────────────────────────────────────────────────────
    // One shared pool, each sequence owning a disjoint run of blocks — the
    // layout a fresh burst gets from the block manager. Blocks are handed out
    // INTERLEAVED across sequences so a kernel that walks a block table
    // linearly (rather than through it) reads someone else's pages and the
    // comparison below says so.
    let blocks_per: Vec<usize> = LENS.iter().map(|&l| l.div_ceil(BS)).collect();
    let total_blocks: usize = blocks_per.iter().sum();
    let block_elems = BS * NKV * HD;
    let mut rng = Lcg(0x5eed_1234_5678_9abc);
    let k_pool_host: Vec<bf16> = (0..total_blocks * block_elems)
        .map(|_| bf16::from_f32(rng.next_f32()))
        .collect();
    let v_pool_host: Vec<bf16> = (0..total_blocks * block_elems)
        .map(|_| bf16::from_f32(rng.next_f32()))
        .collect();
    let (k_pool, k_base, k_total) = upload_bf16(&gpu, &k_pool_host)?;
    let (v_pool, v_base, v_total) = upload_bf16(&gpu, &v_pool_host)?;

    // Interleaved block assignment: the sequences take blocks round-robin, so
    // a sequence's pages are NOT contiguous. A kernel that walks the pool
    // linearly instead of through the block table then reads another
    // sequence's K/V and the bit comparison below names it.
    let mut tables: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut blk = 0u32;
    loop {
        let mut progressed = false;
        for (b, table) in tables.iter_mut().enumerate() {
            if table.len() < blocks_per[b] {
                table.push(blk);
                blk += 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    assert_eq!(blk as usize, total_blocks);

    let table_ptrs_host: Vec<u64> = tables
        .iter()
        .map(|t| upload_u32(&gpu, t).map(|p| p.0))
        .collect::<Result<Vec<_>>>()?;
    let table_devs: Vec<DevicePtr> = table_ptrs_host.iter().map(|&p| DevicePtr(p)).collect();
    let block_table_ptrs = upload_u64(&gpu, &table_ptrs_host)?;

    // ── Q, packed by cu_seqlens ───────────────────────────────────────────
    let q_host: Vec<bf16> = (0..total_tokens * NQ * HD)
        .map(|_| bf16::from_f32(rng.next_f32()))
        .collect();
    let (q, q_base, q_total) = upload_bf16(&gpu, &q_host)?;

    let mut cu = vec![0u32; n + 1];
    for b in 0..n {
        cu[b + 1] = cu[b] + LENS[b] as u32;
    }
    let cu_seqlens = upload_u32(&gpu, &cu)?;
    // Chunk 0: every token this sequence has is the chunk it just wrote, so
    // its KV extent IS its length.
    let kv_lens = upload_u32(&gpu, &LENS.map(|l| l as u32))?;

    let kv_before: Vec<u64> = (0..n)
        .map(|b| seq_kv_digest(&k_pool_host, &v_pool_host, &tables[b], block_elems))
        .collect();

    // ── Arm A: per-stream, one call per sequence ──────────────────────────
    let (out_single, os_base, os_total) = alloc_guarded(&gpu, total_tokens * NQ * HD * 2)?;
    for b in 0..n {
        let off = cu[b] as usize * NQ * HD * 2;
        ops::prefill_attention_paged(
            &gpu,
            k.single,
            q.offset(off),
            k_pool,
            v_pool,
            out_single.offset(off),
            table_devs[b],
            LENS[b] as u32,
            LENS[b] as u32,
            0, // q_offset — CHUNK ZERO, the case that used to be refused
            NQ as u32,
            NKV as u32,
            HD as u32,
            BS as u32,
            0, // sliding_window
            inv_sqrt_d,
            stream,
        )?;
    }
    gpu.synchronize(stream)?;

    // ── Arm B: one batched call over all four ─────────────────────────────
    let (out_batched, ob_base, ob_total) = alloc_guarded(&gpu, total_tokens * NQ * HD * 2)?;
    ops::prefill_attention_paged_batched(
        &gpu,
        k.batched,
        q,
        k_pool,
        v_pool,
        out_batched,
        block_table_ptrs,
        n as u32,
        cu_seqlens,
        kv_lens,
        max_len as u32, // q_len is the MAX: it bounds the grid's Q-tile dim only
        max_len as u32,
        0, // q_offset — chunk zero
        NQ as u32,
        NKV as u32,
        HD as u32,
        BS as u32,
        0,
        inv_sqrt_d,
        stream,
    )?;
    gpu.synchronize(stream)?;

    // ── 1. BITS ───────────────────────────────────────────────────────────
    let a = download_bf16(&gpu, out_single, total_tokens * NQ * HD)?;
    let b_out = download_bf16(&gpu, out_batched, total_tokens * NQ * HD)?;
    let mut failures = 0usize;
    for b in 0..n {
        let lo = cu[b] as usize * NQ * HD;
        let hi = cu[b + 1] as usize * NQ * HD;
        let unequal = (lo..hi).filter(|&i| a[i] != b_out[i]).count();
        let last_row = hi - NQ * HD;
        let last_unequal = (last_row..hi).filter(|&i| a[i] != b_out[i]).count();
        println!(
            "seq {b}: len={:4} blocks={:3}  unequal={unequal}/{}  last-position unequal={last_unequal}/{}",
            LENS[b],
            tables[b].len(),
            hi - lo,
            NQ * HD,
        );
        if unequal != 0 {
            failures += 1;
        }
    }
    ensure!(
        failures == 0,
        "batched chunk-0 prefill attention diverges from the per-stream path \
         on {failures}/{n} sequences — the two arms run the same BR=32 body, so \
         this is a bug, not a tolerance question",
    );

    // ── 2. KV untouched ───────────────────────────────────────────────────
    let k_after = download_bf16(&gpu, k_pool, k_pool_host.len())?;
    let v_after = download_bf16(&gpu, v_pool, v_pool_host.len())?;
    let k_after: Vec<bf16> = k_after.into_iter().map(bf16::from_bits).collect();
    let v_after: Vec<bf16> = v_after.into_iter().map(bf16::from_bits).collect();
    for b in 0..n {
        let after = seq_kv_digest(&k_after, &v_after, &tables[b], block_elems);
        ensure!(
            after == kv_before[b],
            "seq {b}: K/V pages changed across the prefill arms — the chunk-0 \
             batched path must READ the pages the sequence just wrote, never \
             rewrite them (a cross-stream write shows up here as the WRONG \
             sequence's digest moving)",
        );
    }
    println!("KV pool: {n}/{n} per-sequence digests unchanged");

    // ── 3. Extent ─────────────────────────────────────────────────────────
    check_guards(&gpu, k_base, k_total, "k_pool")?;
    check_guards(&gpu, v_base, v_total, "v_pool")?;
    check_guards(&gpu, q_base, q_total, "q")?;
    check_guards(&gpu, os_base, os_total, "out_single")?;
    check_guards(&gpu, ob_base, ob_total, "out_batched")?;

    // Stream isolation: perturb ONE sequence's Q and re-run the batch. Only
    // that sequence's packed rows may move. This is what catches a kernel that
    // indexes Q/O at `b * max_len` (the uniform layout) on a buffer packed by
    // `cu_seqlens` — the exact defect the varlen geometry exists to avoid, and
    // one that a same-length fixture cannot see.
    //
    // ⚠️ WHICH ROW IS PERTURBED IS THE WHOLE CONTROL. H100 round 13 ran this
    // example and it failed on `perturbing seq 1's Q changed nothing — the
    // harness is inert`, after every substantive assertion above had passed.
    // The reason was not the kernel: it perturbed element 0 of head 0 of the
    // victim's FIRST query row, and under causal masking that row attends to
    // exactly one key, so its softmax weight is identically 1.0 and its output
    // is `V[0]` REGARDLESS OF Q. The control could not trip on any hardware.
    //
    // So the perturbation lands on the victim's LAST row, which attends to all
    // `LENS[victim]` keys, and it moves a WHOLE HEAD by a full unit rather than
    // one element by an epsilon — Q is drawn from [-1, 1) and bf16 keeps 8
    // mantissa bits, so a per-element +1.0 across the head cannot be rounded
    // away inside the dot product. Both halves are then asserted: the victim's
    // last row MUST move, and no other sequence may.
    let victim = 1usize;
    let mut q_perturbed = q_host.clone();
    let victim_last_row = (cu[victim + 1] as usize - 1) * NQ * HD;
    for x in &mut q_perturbed[victim_last_row..victim_last_row + HD] {
        *x = bf16::from_f32(x.to_f32() + 1.0);
    }
    let bytes: Vec<u8> = q_perturbed
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect();
    gpu.copy_h2d(&bytes, q)?;
    ops::prefill_attention_paged_batched(
        &gpu,
        k.batched,
        q,
        k_pool,
        v_pool,
        out_batched,
        block_table_ptrs,
        n as u32,
        cu_seqlens,
        kv_lens,
        max_len as u32,
        max_len as u32,
        0,
        NQ as u32,
        NKV as u32,
        HD as u32,
        BS as u32,
        0,
        inv_sqrt_d,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let perturbed = download_bf16(&gpu, out_batched, total_tokens * NQ * HD)?;
    // The ARMING half, checked first and on the exact row that was touched: a
    // control that cannot trip is worth nothing, and "somewhere in the victim
    // moved" is a weaker statement than "the row whose Q changed moved".
    let victim_moved = (victim_last_row..victim_last_row + HD)
        .filter(|&i| perturbed[i] != b_out[i])
        .count();
    ensure!(
        victim_moved > 0,
        "perturbing head 0 of seq {victim}'s LAST query row (element {victim_last_row}, \
         attending to all {} keys) changed none of its {HD} outputs — the harness is \
         inert and nothing below it is evidence",
        LENS[victim],
    );
    println!(
        "stream isolation: seq {victim}'s last row moved in {victim_moved}/{HD} \
         outputs of the perturbed head"
    );
    // …and the ISOLATION half: every other sequence is byte-identical.
    for b in 0..n {
        let lo = cu[b] as usize * NQ * HD;
        let hi = cu[b + 1] as usize * NQ * HD;
        let moved = (lo..hi).filter(|&i| perturbed[i] != b_out[i]).count();
        if b == victim {
            ensure!(
                moved > 0,
                "perturbing seq {b}'s Q changed nothing — the harness is inert"
            );
        } else {
            ensure!(
                moved == 0,
                "perturbing seq {victim}'s Q moved {moved} of seq {b}'s outputs — the \
                 batched path is crossing stream boundaries in the packed layout",
            );
        }
    }
    println!("stream isolation: only seq {victim} moved");

    println!("\nALL PASS — chunk-0 varlen batched prefill matches the per-stream path");
    Ok(())
}

/// Digest of one sequence's K and V pages, in block-table order.
fn seq_kv_digest(k: &[bf16], v: &[bf16], table: &[u32], block_elems: usize) -> u64 {
    let mut bytes = Vec::with_capacity(table.len() * block_elems * 4);
    for &blk in table {
        let lo = blk as usize * block_elems;
        for x in &k[lo..lo + block_elems] {
            bytes.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        for x in &v[lo..lo + block_elems] {
            bytes.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }
    digest(&bytes)
}
