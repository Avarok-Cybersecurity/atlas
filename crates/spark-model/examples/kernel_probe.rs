// SPDX-License-Identifier: AGPL-3.0-only
//! Standalone probe: replicate the server's `ptx_for_config` kernel-load path
//! and report, per pipelined kernel, whether it is in the target's module set
//! and whether `gpu.kernel(module, func)` resolves — printing the exact error.
//! No model load; runs in seconds. Resolves "why is the pipelined handle 0".

use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

fn main() -> anyhow::Result<()> {
    // Server path: serve.rs uses ptx_for_config(model_type, hidden_size).
    // Qwen3.6-35B-A3B → model_type "qwen3_6_moe", hidden_size 2048.
    let set = atlas_kernels::ptx_for_config("qwen3_6_moe", 2048)
        .expect("no ptx set for (qwen3_6_moe, 2048)");
    eprintln!("target.model={} quant={} modules={}", set.target.model, set.target.quant, set.modules.len());

    // (a) membership: are the pipelined modules even in the server's set?
    for m in ["w8a16_gemm_pipelined", "gemm", "w8a16_gemm_t", "moe_fp8_grouped_gemm"] {
        let present = set.modules.iter().any(|(n, _)| *n == m);
        eprintln!("  module '{m}' in set: {present}");
    }

    // (b) resolution: init backend with the SAME modules, try each kernel.
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let probes = [
        ("w8a16_gemm_pipelined", "w8a16_gemm_pipelined"),
        ("gemm", "dense_gemm_bf16_pipelined"),
        ("w8a16_gemm_t", "w8a16_gemm_t_pipelined"),
        ("gemm", "dense_gemm_bf16"),          // known-good control
        ("w8a16_gemm_t", "w8a16_gemm_t"),     // known-good control (non-pipelined)
    ];
    for (m, f) in probes {
        match gpu.kernel(m, f) {
            Ok(h) => eprintln!("OK    {m}::{f} -> handle {}", h.0),
            Err(e) => eprintln!("ERR   {m}::{f} -> {e}"),
        }
    }
    Ok(())
}
