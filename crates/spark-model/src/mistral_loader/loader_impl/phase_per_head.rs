// SPDX-License-Identifier: AGPL-3.0-only

//! Phase B: per-head transpose of W_UK, W_UV; carve out wq_b_rope rows.

use anyhow::Result;

use super::super::gpu_alloc_or_managed;
use super::ctx::MistralLayerCtx;
use crate::weight_map::DenseWeight;

pub(crate) fn build_per_head_views(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    // Load-time phase timer. This phase and `phase_qk_absorbed` between them
    // own most of a LongCat layer's build time; the breakdown below is what
    // settled that. Kept permanently — one `Instant` per phase costs nothing
    // and the next person to ask "where does model load time go" reads it off
    // the log instead of re-deriving it.
    let t_phase = std::time::Instant::now();
    let n_kv = ctx.n_kv;
    let kv_lora = ctx.kv_lora;
    let nope = ctx.nope;
    let rope = ctx.rope;
    let v_dim = ctx.v_dim;
    let hd = ctx.hd;
    let q_lora = ctx.q_lora;
    let bf16 = ctx.bf16;
    let gpu = ctx.gpu;
    let stream = ctx.stream;
    let stride = nope + v_dim;
    let wkv_b = ctx.wkv_b.as_ref().expect("phase A must precede");
    let wq_b = ctx.wq_b.as_ref().expect("phase A must precede");

    // Q absorption: Q_absorbed[lkv] = sum_p(Q_nope[p] * wkv_b_k[lkv, p])
    // wkv_b has K_nope portion as [nope, kv_lora] per head — must
    // TRANSPOSE to [kv_lora, nope] for correct dot product. D2H the
    // relevant portion of wkv_b, transpose on CPU, upload.
    let wkv_b_total_rows = n_kv * stride;
    let wkv_b_bytes = wkv_b_total_rows * kv_lora * bf16;
    let mut wkv_b_host = vec![0u8; wkv_b_bytes];
    let t = std::time::Instant::now();
    gpu.copy_d2h(wkv_b.weight, &mut wkv_b_host)?;
    let t_d2h = t.elapsed();

    // Transpose K portion: [nope, kv_lora] → [kv_lora, nope] per head.
    let w_uk_per_head = kv_lora * nope * bf16;
    let mut w_uk_host = vec![0u8; n_kv * w_uk_per_head];
    let t = std::time::Instant::now();
    for head in 0..n_kv {
        for p in 0..nope {
            for lkv in 0..kv_lora {
                let src_off = ((head * stride + p) * kv_lora + lkv) * bf16;
                let dst_off = (head * kv_lora * nope + lkv * nope + p) * bf16;
                w_uk_host[dst_off..dst_off + bf16]
                    .copy_from_slice(&wkv_b_host[src_off..src_off + bf16]);
            }
        }
    }
    let t_transpose = t.elapsed();
    let t = std::time::Instant::now();
    let w_uk_t_ptr = gpu_alloc_or_managed(gpu, n_kv * w_uk_per_head)?;
    gpu.copy_h2d(&w_uk_host, w_uk_t_ptr)?;
    let t_h2d = t.elapsed();

    // W_UV[n, l, v]: bmm-friendly extraction layout. We need:
    // attn_latent[N, 1, Lkv] @ W_UV[N, Lkv, V] → [N, 1, V]
    // For now store as [N, v_dim, kv_lora] and use a transposed-convention
    // GEMV path in V extraction (TODO: GPU transpose kernel).
    //
    // ONE async copy PER HEAD, not per row. Within a head the source rows
    // (`head*stride + nope + v`, v ascending) and the destination rows are
    // both contiguous, so the whole head is a single linear run — see
    // `per_head_runs_cover_the_same_bytes`. The row-at-a-time version this
    // replaces issued `n_kv * v_dim` `copy_d2d` calls (LongCat: 8192 per
    // sublayer) and `copy_d2d` ends in a full `cuStreamSynchronize`, so it
    // was ~8k stream stalls to move 8 MB. Same bytes, same destination
    // layout; only the call granularity changed.
    let t = std::time::Instant::now();
    let w_uv_ptr = gpu_alloc_or_managed(gpu, n_kv * kv_lora * v_dim * bf16)?;
    let uv_run = v_dim * kv_lora * bf16;
    for head in 0..n_kv {
        let src = wkv_b.weight.offset((head * stride + nope) * kv_lora * bf16);
        let dst = w_uv_ptr.offset(head * uv_run);
        gpu.copy_d2d_async(src, dst, uv_run, stream)?;
    }
    let t_uv_d2d = t.elapsed();

    // Extract wq_b_rope: the rope portion of wq_b per head.
    // wq_b_rope[n*rope+r, l] = wq_b[n*hd+nope+r, l] for r in 0..rope.
    //
    // Same treatment: the `rope` source rows of a head are contiguous and so
    // are their destinations, so it is one run per head instead of
    // `n_kv * rope` synchronizing copies (LongCat: 2048 per sublayer).
    let t = std::time::Instant::now();
    let wqbr_ptr = gpu_alloc_or_managed(gpu, n_kv * rope * q_lora * bf16)?;
    let rope_run = rope * q_lora * bf16;
    for head in 0..n_kv {
        let src = wq_b.weight.offset((head * hd + nope) * q_lora * bf16);
        let dst = wqbr_ptr.offset(head * rope_run);
        gpu.copy_d2d_async(src, dst, rope_run, stream)?;
    }
    // One stall for all 2 * n_kv copies, restoring the ordering guarantee the
    // per-row `copy_d2d` gave (it synchronized after every single row).
    gpu.synchronize(stream)?;

    let t_rope_d2d = t.elapsed();

    tracing::info!(
        "MLA phase B (per-head views) L{}: total={:.1}ms | wkv_b d2h ({:.1} MB)={:.1}ms, \
         cpu-transpose ({} elems)={:.1}ms, w_uk h2d={:.1}ms, \
         w_uv d2d ({} runs)={:.1}ms, wq_b_rope d2d ({} runs)+sync={:.1}ms",
        ctx.layer_idx,
        t_phase.elapsed().as_secs_f64() * 1e3,
        wkv_b_bytes as f64 / 1e6,
        t_d2h.as_secs_f64() * 1e3,
        n_kv * nope * kv_lora,
        t_transpose.as_secs_f64() * 1e3,
        t_h2d.as_secs_f64() * 1e3,
        n_kv,
        t_uv_d2d.as_secs_f64() * 1e3,
        n_kv,
        t_rope_d2d.as_secs_f64() * 1e3,
    );

    ctx.wq_b_rope = Some(DenseWeight { weight: wqbr_ptr });
    ctx.w_uk_t = Some(DenseWeight { weight: w_uk_t_ptr });
    ctx.w_uv = Some(DenseWeight { weight: w_uv_ptr });
    ctx.w_uk_host = w_uk_host;
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Phase B extracts W_UV and wq_b_rope with ONE device copy per head
    /// instead of one per row. That is only correct because, within a head,
    /// both the source rows and the destination rows form a single contiguous
    /// run. This asserts exactly that: the byte ranges the per-head runs cover
    /// are identical to the ranges the original per-row loops covered, in the
    /// same source→destination correspondence.
    #[test]
    fn per_head_runs_cover_the_same_bytes() {
        // (n_kv, kv_lora, nope, rope, v_dim, hd, q_lora, bf16)
        for &(n_kv, kv_lora, nope, rope, v_dim, hd, q_lora) in &[
            (
                32usize, 512usize, 192usize, 64usize, 256usize, 256usize, 1536usize,
            ), // LongCat
            (8, 256, 128, 64, 128, 192, 1024), // Mistral-ish
            (3, 5, 7, 2, 11, 9, 13),           // ragged
        ] {
            let bf16 = 2usize;
            let stride = nope + v_dim;

            // W_UV: per-row reference vs per-head run.
            let mut want: Vec<(usize, usize, usize)> = Vec::new();
            for head in 0..n_kv {
                for v in 0..v_dim {
                    let src = (head * stride + nope + v) * kv_lora * bf16;
                    let dst = (head * v_dim * kv_lora + v * kv_lora) * bf16;
                    want.push((src, dst, kv_lora * bf16));
                }
            }
            let uv_run = v_dim * kv_lora * bf16;
            let mut got: Vec<(usize, usize, usize)> = Vec::new();
            for head in 0..n_kv {
                let src = (head * stride + nope) * kv_lora * bf16;
                let dst = head * uv_run;
                // Expand the run back into rows to compare like for like.
                for v in 0..v_dim {
                    got.push((
                        src + v * kv_lora * bf16,
                        dst + v * kv_lora * bf16,
                        kv_lora * bf16,
                    ));
                }
            }
            assert_eq!(
                got, want,
                "w_uv run/row mismatch (n_kv={n_kv} v_dim={v_dim})"
            );

            // wq_b_rope: same check.
            let mut want: Vec<(usize, usize, usize)> = Vec::new();
            for head in 0..n_kv {
                for r in 0..rope {
                    let src = (head * hd + nope + r) * q_lora * bf16;
                    let dst = (head * rope + r) * q_lora * bf16;
                    want.push((src, dst, q_lora * bf16));
                }
            }
            let rope_run = rope * q_lora * bf16;
            let mut got: Vec<(usize, usize, usize)> = Vec::new();
            for head in 0..n_kv {
                let src = (head * hd + nope) * q_lora * bf16;
                let dst = head * rope_run;
                for r in 0..rope {
                    got.push((
                        src + r * q_lora * bf16,
                        dst + r * q_lora * bf16,
                        q_lora * bf16,
                    ));
                }
            }
            assert_eq!(
                got, want,
                "wq_b_rope run/row mismatch (n_kv={n_kv} rope={rope})"
            );
        }
    }
}
