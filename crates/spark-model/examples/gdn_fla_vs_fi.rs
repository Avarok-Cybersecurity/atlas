// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-impl A/B: Atlas FLA gdn_prefill_fla vs FlashInfer chunk_gated_delta_rule.
//! Loads the SAME q/k/v/g/beta the FI reference used (/tmp/gdn_ref/*.bin) + FI's output
//! (o_ref.bin), runs Atlas's full FLA scan on identical input, diffs the two outputs.
//! High cos (>0.99) => same math (e2e garbage is elsewhere). Low cos => the GDN math diverges.
use anyhow::Result;
use half::{bf16, f16};
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

fn rd(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|_| panic!("missing {p}"))
}
fn f16s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let (t, nk, nv, kd, vd) = (2048usize, 16usize, 32usize, 128usize, 128usize);
    let key_dim = nk * kd;
    let value_dim = nv * vd;
    let conv_dim = 2 * key_dim + value_dim;

    // Load FI-reference inputs (q,k,v fp16; g,beta fp32) and FI output (fp32).
    let q = f16s(&rd("/tmp/gdn_ref/q.bin"));
    let k = f16s(&rd("/tmp/gdn_ref/k.bin"));
    let v = f16s(&rd("/tmp/gdn_ref/v.bin"));
    let gate = f32s(&rd("/tmp/gdn_ref/g.bin"));
    let beta = f32s(&rd("/tmp/gdn_ref/beta.bin"));
    let o_ref = f32s(&rd("/tmp/gdn_ref/o_ref.bin"));

    // Pack into Atlas layout: qkv [t, conv_dim] bf16 ([Q|K|V]); gate_beta [t, 2*nv] f32.
    let mut qkv = vec![0u8; t * conv_dim * 2];
    {
        let put = |buf: &mut [u8], idx: usize, val: f32| {
            let b = bf16::from_f32(val).to_bits().to_le_bytes();
            buf[idx * 2] = b[0];
            buf[idx * 2 + 1] = b[1];
        };
        for tk in 0..t {
            for i in 0..key_dim {
                put(&mut qkv, tk * conv_dim + i, q[tk * key_dim + i]);
            }
            for i in 0..key_dim {
                put(&mut qkv, tk * conv_dim + key_dim + i, k[tk * key_dim + i]);
            }
            for i in 0..value_dim {
                put(
                    &mut qkv,
                    tk * conv_dim + 2 * key_dim + i,
                    v[tk * value_dim + i],
                );
            }
        }
    }
    let mut gb = vec![0u8; t * 2 * nv * 4];
    for tk in 0..t {
        for h in 0..nv {
            gb[(tk * 2 * nv + h) * 4..(tk * 2 * nv + h) * 4 + 4]
                .copy_from_slice(&gate[tk * nv + h].to_le_bytes());
            gb[(tk * 2 * nv + nv + h) * 4..(tk * 2 * nv + nv + h) * 4 + 4]
                .copy_from_slice(&beta[tk * nv + h].to_le_bytes());
        }
    }

    // Upload + scratch.
    let qkv_d = g.alloc(qkv.len())?;
    g.copy_h2d(&qkv, qkv_d)?;
    let gb_d = g.alloc(gb.len())?;
    g.copy_h2d(&gb, gb_d)?;
    let out_d = g.alloc(t * value_dim * 2)?;
    g.memset(out_d, 0, t * value_dim * 2)?;
    let h_state = g.alloc(nv * kd * vd * 4)?;
    g.memset(h_state, 0, nv * kd * vd * 4)?;
    let nt = t.div_ceil(64);
    let w_out = g.alloc(nt * nv * 64 * kd * 2)?;
    let u_out = g.alloc(nt * nv * 64 * vd * 2)?;
    let s_out = g.alloc(nt * nv * kd * vd * 2)?;
    let uc_out = g.alloc(nt * nv * 64 * vd * 2)?;
    let gc_out = g.alloc(nt * nv * 64 * 4)?;

    let q_ptr = qkv_d;
    let k_ptr = qkv_d.offset(key_dim * 2);
    let v_ptr = qkv_d.offset(2 * key_dim * 2);
    let gate_ptr = gb_d;
    let beta_ptr = gb_d.offset(nv * 4);

    let k_wu = g.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu")?;
    let k_dh = g.kernel(
        "gated_delta_rule_fla",
        "gated_delta_rule_chunk_delta_h_ksplit",
    )?;
    let k_fo = g.kernel("gated_delta_rule_fla", "gated_delta_rule_chunk_fwd_o")?;

    ops::gdn_prefill_fla(
        g,
        k_wu,
        k_dh,
        spark_runtime::gpu::KernelHandle(0),
        k_fo,
        h_state,
        q_ptr,
        k_ptr,
        v_ptr,
        gate_ptr,
        beta_ptr,
        out_d,
        w_out,
        u_out,
        s_out,
        uc_out,
        gc_out,
        1,
        t as u32,
        nt as u32,
        nk as u32,
        nv as u32,
        kd as u32,
        vd as u32,
        conv_dim as u32,
        conv_dim as u32,
        (nv * 2) as u32,
        false,
        DevicePtr::NULL,
        DevicePtr::NULL,
        false,
        false,
        0,
    )?;
    g.synchronize(0)?;

    // Read Atlas output (bf16) -> f32, compare to FI o_ref.
    let n = t * value_dim;
    let mut ob = vec![0u8; n * 2];
    g.copy_d2h(out_d, &mut ob)?;
    let o_fla: Vec<f32> = ob
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();

    let (mut maxe, mut dot, mut nf, mut nr, mut sf, mut sr) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let (a, b) = (o_fla[i] as f64, o_ref[i] as f64);
        maxe = maxe.max((a - b).abs());
        dot += a * b;
        nf += a * a;
        nr += b * b;
        sf += a.abs();
        sr += b.abs();
    }
    let cos = dot / (nf.sqrt() * nr.sqrt() + 1e-12);
    println!("=== Atlas FLA vs FlashInfer (same input) ===");
    println!(
        "|o_fla|mean={:.6}  |o_fi|mean={:.6}  norm_ratio(fla/fi)={:.4}",
        sf / n as f64,
        sr / n as f64,
        nf.sqrt() / (nr.sqrt() + 1e-12)
    );
    println!("cos={:.6}  max_abs_err={:.6}", cos, maxe);
    println!(
        "VERDICT: {}",
        if cos > 0.99 {
            "SAME MATH (divergence is elsewhere — runtime/gate-range/chunking)"
        } else if cos > 0.5 {
            "PARTIAL — likely scale/gate/beta convention diff"
        } else {
            "DIFFERENT MATH — formulation mismatch"
        }
    );
    Ok(())
}
