// SPDX-License-Identifier: AGPL-3.0-only

//! Phase C: precompute fused Q-absorption matrix W_QK_absorbed
//! [nq*kv_lora, q_lora] on the CPU, upload BF16.

use anyhow::Result;

use super::super::gpu_alloc_or_managed;
use super::ctx::MistralLayerCtx;
use crate::weight_map::DenseWeight;

pub(super) fn build_w_qk_absorbed(ctx: &mut MistralLayerCtx<'_>) -> Result<()> {
    let n_kv = ctx.n_kv;
    let n_heads = ctx.n_heads;
    let kv_lora = ctx.kv_lora;
    let q_lora = ctx.q_lora;
    let nope = ctx.nope;
    let hd = ctx.hd;
    let bf16 = ctx.bf16;
    let gpu = ctx.gpu;

    let wq_b = ctx.wq_b.as_ref().expect("phase A");
    let w_uk_t = ctx.w_uk_t.as_ref().expect("phase B");

    let wqk_size = n_kv * kv_lora * q_lora * bf16;
    let wqk_ptr = gpu_alloc_or_managed(gpu, wqk_size)?;
    {
        // Read wq_b[n_heads*hd, q_lora] from GPU.
        let wqb_bytes = n_heads * hd * q_lora * bf16;
        let mut wqb_buf = vec![0u8; wqb_bytes];
        gpu.copy_d2h(wq_b.weight, &mut wqb_buf)?;
        // Read W_UK[n_heads, kv_lora, nope] from GPU (transposed layout).
        let wuk_bytes = n_kv * kv_lora * nope * bf16;
        let mut wuk_buf = vec![0u8; wuk_bytes];
        gpu.copy_d2h(w_uk_t.weight, &mut wuk_buf)?;

        // Compute W_QK[n, kv_lora, q_lora] on CPU in FP32.
        let mut wqk_f32 = vec![0.0f32; n_kv * kv_lora * q_lora];
        let to_f32 = |buf: &[u8], idx: usize| -> f32 {
            let bits = u16::from_le_bytes([buf[idx * 2], buf[idx * 2 + 1]]);
            f32::from_bits((bits as u32) << 16)
        };
        for n in 0..n_kv {
            for lkv in 0..kv_lora {
                for l in 0..q_lora {
                    let mut sum = 0.0f32;
                    for p in 0..nope {
                        let wqb_val = to_f32(&wqb_buf, (n * hd + p) * q_lora + l);
                        let wuk_val = to_f32(&wuk_buf, n * kv_lora * nope + lkv * nope + p);
                        sum += wqb_val * wuk_val;
                    }
                    wqk_f32[(n * kv_lora + lkv) * q_lora + l] = sum;
                }
            }
        }
        let wqk_bf16: Vec<u8> = wqk_f32
            .iter()
            .flat_map(|&v| {
                let bits = (v.to_bits() >> 16) as u16;
                bits.to_le_bytes().to_vec()
            })
            .collect();
        gpu.copy_h2d(&wqk_bf16, wqk_ptr)?;
        if ctx.layer_idx == 0 {
            tracing::info!(
                "W_QK_absorbed: [{}, {}] ({:.1} MB per layer)",
                n_kv * kv_lora,
                q_lora,
                wqk_size as f64 / 1e6
            );
        }
    }
    ctx.w_qk_absorbed = Some(DenseWeight { weight: wqk_ptr });
    Ok(())
}
