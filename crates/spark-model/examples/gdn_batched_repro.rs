// SPDX-License-Identifier: AGPL-3.0-only
//! #110 standalone reproducer for the Q12 batched GDN prefill kernel.
//! Calls `gated_delta_rule_prefill_wy64_batched` directly with batch_size=2
//! and synthetic inputs, isolated from the 70GB model + scheduler timing, so
//! compute-sanitizer can pin the exact illegal access in seconds.
//!   compute-sanitizer --tool memcheck \
//!     cargo run -p spark-model --release --example gdn_batched_repro

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

// Real qwen3.6-35b-a3b GDN dims.
const NK: usize = 16;
const NV: usize = 32;
const KD: usize = 128;
const VD: usize = 128;

fn main() -> Result<()> {
    let batch_size: u32 = std::env::var("BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let seq_len: u32 = std::env::var("SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(96);
    let bs = batch_size as usize;
    let sl = seq_len as usize;

    let key_dim = NK * KD; // 2048
    let value_dim = NV * VD; // 4096
    let conv_dim = key_dim * 2 + value_dim; // 8192
    let gb_stride = NV * 2; // 64
    let h_numel = NV * KD * VD; // 524288 floats = 2 MiB per stream

    eprintln!(
        "gdn_batched_repro: batch={batch_size} seq_len={seq_len} key_dim={key_dim} \
         value_dim={value_dim} conv_dim={conv_dim} h_numel={h_numel}"
    );

    let set = atlas_kernels::ptx_for_config("qwen3_6_moe", 2048).expect("no ptx set");
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let k: KernelHandle = g.kernel(
        "gated_delta_rule_wy64_prefill",
        "gated_delta_rule_prefill_wy64_batched",
    )?;

    // Stacked qkv [bs*sl, conv_dim] bf16, packed [Q(key_dim)|K(key_dim)|V(value_dim)].
    let qkv_n = bs * sl * conv_dim;
    let qkv_host: Vec<u8> = (0..qkv_n)
        .flat_map(|i| {
            bf16::from_f32(((i % 17) as f32 - 8.0) * 0.01)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let qkv = g.alloc(qkv_host.len())?;
    g.copy_h2d(&qkv_host, qkv)?;

    // gate_beta [bs*sl, NV*2] f32, [gate(NV)|beta(NV)].
    let gb_n = bs * sl * gb_stride;
    let gb_host: Vec<u8> = (0..gb_n)
        .flat_map(|i| (0.5f32 + ((i % 7) as f32) * 0.01).to_le_bytes())
        .collect();
    let gate_beta = g.alloc(gb_host.len())?;
    g.copy_h2d(&gb_host, gate_beta)?;

    // output [bs*sl, value_dim] bf16.
    let out = g.alloc(bs * sl * value_dim * 2)?;
    g.memset(out, 0, bs * sl * value_dim * 2)?;

    // Per-stream h_state (2 MiB each, FP32), zeroed.
    let mut h_ptrs: Vec<u64> = Vec::with_capacity(bs);
    for _ in 0..bs {
        let h = g.alloc(h_numel * 4)?;
        g.memset(h, 0, h_numel * 4)?;
        h_ptrs.push(h.0);
    }
    // h_state_ptrs device array [bs] of float* (u64).
    let hp_host: Vec<u8> = h_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
    let h_state_ptrs = g.alloc(hp_host.len())?;
    g.copy_h2d(&hp_host, h_state_ptrs)?;

    // q/k/v pointers into qkv (byte offsets, matching the dispatcher).
    let q_ptr = qkv;
    let k_ptr = qkv.offset(key_dim * 2); // key_dim bf16 elements
    let v_ptr = qkv.offset(key_dim * 2 * 2);
    let gate_ptr = gate_beta;
    let beta_ptr = gate_beta.offset(NV * 4);

    let default_smem = (KD * VD * 4 + 32 * KD * 2 + 32 * KD * 2 + 32 * 32 * 4 + 256) as u32;
    let smem: u32 = std::env::var("SMEM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_smem);

    eprintln!(
        "launching gated_delta_rule_prefill_wy64_batched grid=[{NV},{batch_size},1] smem={smem}"
    );
    KernelLaunch::new(g, k)
        .grid([NV as u32, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state_ptrs)
        .arg_ptr(q_ptr)
        .arg_ptr(k_ptr)
        .arg_ptr(v_ptr)
        .arg_ptr(gate_ptr)
        .arg_ptr(beta_ptr)
        .arg_ptr(out)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32(conv_dim as u32)
        .arg_u32(conv_dim as u32)
        .arg_u32(gb_stride as u32)
        .launch(0)?;

    g.synchronize(0)?;
    eprintln!("gdn_batched_repro: kernel completed cleanly (no fault at batch={batch_size})");
    Ok(())
}
