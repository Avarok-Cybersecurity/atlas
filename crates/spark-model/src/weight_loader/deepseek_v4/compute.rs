// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::weight_map::DenseWeight;

fn alloc_or_managed(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    static USE_MANAGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if USE_MANAGED.load(std::sync::atomic::Ordering::Relaxed) {
        return gpu.alloc_managed(bytes);
    }
    match gpu.alloc(bytes) {
        Ok(p) => Ok(p),
        Err(_) => {
            USE_MANAGED.store(true, std::sync::atomic::Ordering::Relaxed);
            gpu.alloc_managed(bytes)
        }
    }
}

fn bf16() -> usize {
    2usize
}

fn to_f32(buf: &[u8], idx: usize) -> f32 {
    let bits = u16::from_le_bytes([buf[idx * 2], buf[idx * 2 + 1]]);
    f32::from_bits((bits as u32) << 16)
}

fn from_f32(v: f32) -> [u8; 2] {
    let bits = (v.to_bits() >> 16) as u16;
    bits.to_le_bytes()
}

/// Split wkv_b into W_UK and W_UV, transpose W_UK, extract wq_b_rope rows.
/// `wkv_b_shape` and `wq_b_shape` are the actual [n, k] dims from the checkpoint.
pub fn build_per_head_views(
    wkv_b: &DenseWeight,
    wkv_b_shape: &[usize],
    wq_b: &DenseWeight,
    wq_b_shape: &[usize],
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<(DenseWeight, DenseWeight, DenseWeight, Vec<u8>)> {
    let n_kv = config.num_key_value_heads;
    let kv_lora = config.kv_lora_rank;
    let nope = config.qk_nope_head_dim;
    let v_dim = config.v_head_dim;
    let q_lora = config.q_lora_rank;
    let hd = config.head_dim;
    let n_heads = config.num_attention_heads;
    let b = bf16();

    // Use actual tensor size from checkpoint to avoid over-read
    let wkvb_bytes = wkv_b_shape[0] * wkv_b_shape[1] * b;
    let mut wkvb_buf = vec![0u8; wkvb_bytes];
    gpu.copy_d2h(wkv_b.weight, &mut wkvb_buf)?;

    // W_UK: [n_kv, kv_lora, nope]
    let uk_size = n_kv * kv_lora * nope * b;
    let mut uk_host = vec![0u8; uk_size];
    for h in 0..n_kv {
        for lkv in 0..kv_lora {
            for p in 0..nope {
                let src = ((h * (nope + v_dim) + p) * kv_lora + lkv) * b;
                if src + b <= wkvb_buf.len() {
                    let dst = ((h * kv_lora + lkv) * nope + p) * b;
                    uk_host[dst..dst + b].copy_from_slice(&wkvb_buf[src..src + b]);
                }
            }
        }
    }
    let w_uk_t_ptr = alloc_or_managed(gpu, uk_size)?;
    gpu.copy_h2d(&uk_host, w_uk_t_ptr)?;

    // W_UV: [n_kv, v_dim, kv_lora]
    let uv_size = n_kv * v_dim * kv_lora * b;
    let mut uv_host = vec![0u8; uv_size];
    for h in 0..n_kv {
        for v in 0..v_dim {
            for lkv in 0..kv_lora {
                let src = ((h * (nope + v_dim) + nope + v) * kv_lora + lkv) * b;
                if src + b <= wkvb_buf.len() {
                    let dst = ((h * v_dim + v) * kv_lora + lkv) * b;
                    uv_host[dst..dst + b].copy_from_slice(&wkvb_buf[src..src + b]);
                }
            }
        }
    }
    let w_uv_ptr = alloc_or_managed(gpu, uv_size)?;
    gpu.copy_h2d(&uv_host, w_uv_ptr)?;

    // wq_b_rope: [n_heads*rope, q_lora]
    let rope = config.qk_rope_head_dim;
    let wq_bytes = wq_b_shape[0] * wq_b_shape[1] * b;
    let mut wq_buf = vec![0u8; wq_bytes];
    gpu.copy_d2h(wq_b.weight, &mut wq_buf)?;
    let rope_size = n_heads * rope * q_lora * b;
    let mut rope_host = vec![0u8; rope_size];
    for h in 0..n_heads {
        for r in 0..rope {
            for l in 0..q_lora {
                let src = ((h * hd + nope + r) * q_lora + l) * b;
                if src + b <= wq_buf.len() {
                    let dst = ((h * rope + r) * q_lora + l) * b;
                    rope_host[dst..dst + b].copy_from_slice(&wq_buf[src..src + b]);
                }
            }
        }
    }
    let wq_b_rope_ptr = alloc_or_managed(gpu, rope_size)?;
    gpu.copy_h2d(&rope_host, wq_b_rope_ptr)?;

    Ok((
        DenseWeight { weight: w_uk_t_ptr },
        DenseWeight { weight: w_uv_ptr },
        DenseWeight {
            weight: wq_b_rope_ptr,
        },
        uk_host,
    ))
}

/// W_QK_absorbed = wq_b_nope @ W_UK_T
pub fn build_w_qk_absorbed(
    wq_b: &DenseWeight,
    wq_b_shape: &[usize],
    w_uk_t: &DenseWeight,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let n_kv = config.num_key_value_heads;
    let kv_lora = config.kv_lora_rank;
    let q_lora = config.q_lora_rank;
    let nope = config.qk_nope_head_dim;
    let hd = config.head_dim;
    let b = bf16();

    let wqk_size = n_kv * kv_lora * q_lora * b;
    let wqk_ptr = alloc_or_managed(gpu, wqk_size)?;

    let wqb_bytes = wq_b_shape[0] * wq_b_shape[1] * b;
    let mut wqb_buf = vec![0u8; wqb_bytes];
    gpu.copy_d2h(wq_b.weight, &mut wqb_buf)?;
    let mut wuk_buf = vec![0u8; n_kv * kv_lora * nope * b];
    gpu.copy_d2h(w_uk_t.weight, &mut wuk_buf)?;

    let mut wqk_f32 = vec![0.0f32; n_kv * kv_lora * q_lora];
    for n in 0..n_kv {
        for lkv in 0..kv_lora {
            for l in 0..q_lora {
                let mut sum = 0.0f32;
                for p in 0..nope {
                    let wqb_idx = (n * hd + p) * q_lora + l;
                    let wuk_idx = n * kv_lora * nope + lkv * nope + p;
                    let wqb = if wqb_idx * 2 + 2 <= wqb_buf.len() {
                        to_f32(&wqb_buf, wqb_idx)
                    } else {
                        0.0
                    };
                    let wuk = if wuk_idx * 2 + 2 <= wuk_buf.len() {
                        to_f32(&wuk_buf, wuk_idx)
                    } else {
                        0.0
                    };
                    sum += wqb * wuk;
                }
                wqk_f32[(n * kv_lora + lkv) * q_lora + l] = sum;
            }
        }
    }
    let wqk_bf16: Vec<u8> = wqk_f32.iter().flat_map(|&v| from_f32(v)).collect();
    gpu.copy_h2d(&wqk_bf16, wqk_ptr)?;
    Ok(DenseWeight { weight: wqk_ptr })
}

/// Block-diagonal W_UK and W_UV for prefill batched GEMM.
pub fn build_block_diagonals(
    w_uk_host: &[u8],
    w_uv: &DenseWeight,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<(DenseWeight, DenseWeight)> {
    let n_kv = config.num_key_value_heads;
    let kv_lora = config.kv_lora_rank;
    let nope = config.qk_nope_head_dim;
    let v_dim = config.v_head_dim;
    let b = bf16();

    let bd_rows = n_kv * kv_lora;
    let bd_cols = n_kv * nope;
    let bd_size = bd_rows * bd_cols * b;
    let mut w_uk_bd = vec![0u8; bd_size];
    for head in 0..n_kv {
        for lkv in 0..kv_lora {
            for p in 0..nope {
                let src = (head * kv_lora * nope + lkv * nope + p) * b;
                let dst_row = head * kv_lora + lkv;
                let dst_col = head * nope + p;
                let dst = (dst_row * bd_cols + dst_col) * b;
                w_uk_bd[dst..dst + b].copy_from_slice(&w_uk_host[src..src + b]);
            }
        }
    }
    let w_uk_bd_ptr = alloc_or_managed(gpu, bd_size)?;
    gpu.copy_h2d(&w_uk_bd, w_uk_bd_ptr)?;

    let uv_bd_rows = n_kv * v_dim;
    let uv_bd_cols = n_kv * kv_lora;
    let uv_bd_size = uv_bd_rows * uv_bd_cols * b;
    let mut w_uv_host = vec![0u8; n_kv * v_dim * kv_lora * b];
    gpu.copy_d2h(w_uv.weight, &mut w_uv_host)?;
    let mut w_uv_bd = vec![0u8; uv_bd_size];
    for head in 0..n_kv {
        for v in 0..v_dim {
            for l in 0..kv_lora {
                let src = (head * v_dim * kv_lora + v * kv_lora + l) * b;
                let dst_row = head * v_dim + v;
                let dst_col = head * kv_lora + l;
                let dst = (dst_row * uv_bd_cols + dst_col) * b;
                w_uv_bd[dst..dst + b].copy_from_slice(&w_uv_host[src..src + b]);
            }
        }
    }
    let w_uv_bd_ptr = alloc_or_managed(gpu, uv_bd_size)?;
    gpu.copy_h2d(&w_uv_bd, w_uv_bd_ptr)?;

    Ok((
        DenseWeight {
            weight: w_uk_bd_ptr,
        },
        DenseWeight {
            weight: w_uv_bd_ptr,
        },
    ))
}

/// YaRN inv_freq — compute once, share across layers.
pub fn ensure_yarn_inv_freq(
    shared: &mut DevicePtr,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    if !shared.is_null() {
        return Ok(*shared);
    }
    let rope = config.qk_rope_head_dim;
    let factor = if config.yarn_factor > 0.0 {
        config.yarn_factor
    } else {
        16.0
    };
    let beta_fast = if config.yarn_beta_fast > 0.0 {
        config.yarn_beta_fast
    } else {
        32.0
    };
    let beta_slow = if config.yarn_beta_slow > 0.0 {
        config.yarn_beta_slow
    } else {
        1.0
    };
    let original_max_pos = if config.yarn_original_max_position_embeddings > 0 {
        config.yarn_original_max_position_embeddings as f32
    } else {
        65536.0
    };
    let dim_f = rope as f32;
    let theta_f = config.rope_theta as f32;
    let n_pairs = rope / 2;
    let mscale = crate::layers::qwen3_attention::helpers::yarn_rope_mscale(config);
    tracing::info!(
        "YaRN inv_freq: factor={:.1}, beta_fast={:.1}, beta_slow={:.1}, \
         original_max_pos={:.0}, theta={:.0}, rope_dim={}, n_pairs={}, mscale={:.6}",
        factor,
        beta_fast,
        beta_slow,
        original_max_pos,
        theta_f,
        rope,
        n_pairs,
        mscale,
    );

    let find_correction_dim = |num_rot: f32| -> f32 {
        (dim_f * (original_max_pos / (num_rot * 2.0 * std::f32::consts::PI)).ln())
            / (2.0 * theta_f.ln())
    };
    let low = find_correction_dim(beta_fast).floor().max(0.0);
    let high = find_correction_dim(beta_slow).ceil().min((rope - 1) as f32);
    let ramp_denom = if (high - low).abs() < 1e-6 {
        high - low + 0.001
    } else {
        high - low
    };

    let mut inv_freq = vec![0.0f32; n_pairs];
    for j in 0..n_pairs {
        let pos_freq = theta_f.powf((2 * j) as f32 / dim_f);
        let inv_freq_extrap = 1.0 / pos_freq;
        let inv_freq_interp = 1.0 / (factor * pos_freq);
        let ramp = ((j as f32 - low) / ramp_denom).clamp(0.0, 1.0);
        let extrap_factor = 1.0 - ramp;
        inv_freq[j] = inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor;
    }
    let bytes: Vec<u8> = inv_freq.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ptr = alloc_or_managed(gpu, bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    *shared = ptr;
    Ok(ptr)
}

/// Plain θ=10000 inv_freq (NO YaRN) for the raw-arm rope on `sliding_attention`
/// layers. The reference's "main" rope: θ=10000 (config `rope_theta`, which
/// Atlas's dispatch overrides to compress_rope_theta=160000 for the shared yarn
/// table — so the 10000 base is hardcoded here to match the reference). Cheap
/// (~32 floats); computed per compressor-less layer, no shared cache needed.
pub fn main_inv_freq(config: &ModelConfig, gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    const MAIN_ROPE_THETA: f32 = 10000.0; // reference rope_theta for sliding layers
    let rope = config.qk_rope_head_dim;
    let n_pairs = rope / 2;
    let dim_f = rope as f32;
    let mut inv_freq = vec![0.0f32; n_pairs];
    for (j, f) in inv_freq.iter_mut().enumerate() {
        *f = 1.0 / MAIN_ROPE_THETA.powf((2 * j) as f32 / dim_f);
    }
    let bytes: Vec<u8> = inv_freq.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ptr = alloc_or_managed(gpu, bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    Ok(ptr)
}
