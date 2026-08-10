// SPDX-License-Identifier: AGPL-3.0-only

#![cfg(feature = "metal")]

use anyhow::Result;
use half::bf16;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::metal_backend::MetalGpuBackend;

#[test]
fn nllb_metal_kernels_smoke() -> Result<()> {
    let modules = atlas_kernels::metallib_modules();
    if modules.is_empty() {
        eprintln!(
            "metal kernel registry empty; run with ATLAS_TARGET_HW=metal \
             ATLAS_TARGET_MODEL=nllb-200-3.3b ATLAS_TARGET_QUANT=bf16"
        );
        return Ok(());
    }

    let backend = MetalGpuBackend::new(0, &modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.default_stream();

    run_linear_smoke(gpu, stream)?;
    run_layernorm_smoke(gpu, stream)?;
    run_attention_smoke(gpu, stream)?;
    run_bf16_batch_smoke(gpu, stream)?;
    Ok(())
}

fn run_linear_smoke(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    let linear = gpu.kernel("nllb_encoder", "nllb_linear")?;
    let no_bias = gpu.kernel("nllb_encoder", "nllb_linear_no_bias")?;

    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = [1.0f32, 0.0, 1.0, 0.0, 1.0, 1.0];
    let bias = [10.0f32, 20.0];
    let a_dev = upload(gpu, &a)?;
    let w_dev = upload(gpu, &w)?;
    let bias_dev = upload(gpu, &bias)?;
    let out = gpu.alloc(4 * 4)?;

    KernelLaunch::new(gpu, linear)
        .grid([1, 1, 1])
        .block([16, 16, 1])
        .arg_ptr(a_dev)
        .arg_ptr(w_dev)
        .arg_ptr(bias_dev)
        .arg_ptr(out)
        .arg_u32(2)
        .arg_u32(2)
        .arg_u32(3)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    assert_close(&download(gpu, out, 4)?, &[14.0, 25.0, 20.0, 31.0]);

    KernelLaunch::new(gpu, no_bias)
        .grid([1, 1, 1])
        .block([16, 16, 1])
        .arg_ptr(a_dev)
        .arg_ptr(w_dev)
        .arg_ptr(out)
        .arg_u32(2)
        .arg_u32(2)
        .arg_u32(3)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    assert_close(&download(gpu, out, 4)?, &[4.0, 5.0, 10.0, 11.0]);
    Ok(())
}

fn run_layernorm_smoke(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    let layernorm = gpu.kernel("nllb_encoder", "nllb_layernorm")?;
    let x = upload(gpu, &[1.0f32, 2.0])?;
    let w = upload(gpu, &[1.0f32, 1.0])?;
    let b = upload(gpu, &[0.0f32, 0.0])?;

    KernelLaunch::new(gpu, layernorm)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(b)
        .arg_u32(1)
        .arg_u32(2)
        .arg_f32(0.0)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    assert_close(&download(gpu, x, 2)?, &[-1.0, 1.0]);
    Ok(())
}

fn run_attention_smoke(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    let attn = gpu.kernel("nllb_encoder", "nllb_attn_kv")?;
    let q = upload(gpu, &[1.0f32, 0.0])?;
    let k = upload(gpu, &[1.0f32, 0.0])?;
    let v = upload(gpu, &[7.0f32, 9.0])?;
    let out = gpu.alloc(2 * 4)?;

    KernelLaunch::new(gpu, attn)
        .grid([1, 1, 1])
        .block([2, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(v)
        .arg_ptr(out)
        .arg_u32(1)
        .arg_u32(1)
        .arg_u32(1)
        .arg_u32(2)
        .arg_f32(1.0)
        .arg_u32(0)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    assert_close(&download(gpu, out, 2)?, &[7.0, 9.0]);
    Ok(())
}

fn run_bf16_batch_smoke(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    let scatter = gpu.kernel("nllb_encoder", "nllb_scatter_batched")?;
    let gather = gpu.kernel("nllb_encoder", "nllb_gather_batched")?;
    let gemv_batched = gpu.kernel("nllb_encoder", "nllb_gemv_batched_bf16")?;
    let topk = gpu.kernel("nllb_encoder", "nllb_topk_lse_bf16")?;

    let x = upload_bf16(gpu, &[1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])?;
    let w = upload_bf16(
        gpu,
        &[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 1.0, 0.0, //
            1.0, 1.0, 1.0, 1.0,
        ],
    )?;
    let bias = upload_bf16(gpu, &[10.0, 20.0, 30.0])?;
    let out = gpu.alloc(2 * 3 * 2)?;
    KernelLaunch::new(gpu, gemv_batched)
        .grid([3, 2, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(bias)
        .arg_ptr(out)
        .arg_u32(2)
        .arg_u32(3)
        .arg_u32(4)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    assert_close(
        &download_bf16(gpu, out, 6)?,
        &[11.0, 25.0, 40.0, 14.0, 25.0, 40.0],
    );

    let src = upload_bf16(gpu, &[1.0, 2.0, 3.0, 4.0])?;
    let cache = gpu.alloc(2 * 2 * 2 * 2)?;
    gpu.copy_h2d(bf16_bytes(&vec![bf16::from_f32(0.0); 8]), cache)?;
    KernelLaunch::new(gpu, scatter)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(cache)
        .arg_u32(1)
        .arg_u32(2)
        .arg_u32(2)
        .arg_u32(2)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    assert_close(
        &download_bf16(gpu, cache, 8)?,
        &[0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0],
    );

    let perm = upload_u32(gpu, &[1, 0])?;
    let gathered = gpu.alloc(2 * 2 * 2 * 2)?;
    KernelLaunch::new(gpu, gather)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(cache)
        .arg_ptr(gathered)
        .arg_ptr(perm)
        .arg_u32(2)
        .arg_u32(2)
        .arg_u32(2)
        .arg_u32(2)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    assert_close(
        &download_bf16(gpu, gathered, 8)?,
        &[0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0, 2.0],
    );

    let logits = upload_bf16(
        gpu,
        &[0.0, 3.0, 1.0, 8.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 10.0, -1.0],
    )?;
    let vals = gpu.alloc(10 * 4)?;
    let ids = gpu.alloc(10 * 4)?;
    let lse = gpu.alloc(4)?;
    KernelLaunch::new(gpu, topk)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(vals)
        .arg_ptr(ids)
        .arg_ptr(lse)
        .arg_u32(1)
        .arg_u32(12)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    assert_eq!(&download_u32(gpu, ids, 10)?[..4], &[10, 9, 3, 7]);
    assert_close(&download(gpu, vals, 4)?, &[10.0, 9.0, 8.0, 7.0]);
    let expected_lse = [
        -1.0f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    ]
    .iter()
    .map(|v| (*v - 10.0).exp())
    .sum::<f32>()
    .ln()
        + 10.0;
    let got_lse = download(gpu, lse, 1)?[0];
    assert!((got_lse - expected_lse).abs() < 1.0e-3);
    Ok(())
}

fn upload(gpu: &dyn GpuBackend, values: &[f32]) -> Result<spark_runtime::gpu::DevicePtr> {
    let ptr = gpu.alloc(std::mem::size_of_val(values))?;
    gpu.copy_h2d(f32_bytes(values), ptr)?;
    Ok(ptr)
}

fn upload_bf16(gpu: &dyn GpuBackend, values: &[f32]) -> Result<spark_runtime::gpu::DevicePtr> {
    let values: Vec<bf16> = values.iter().copied().map(bf16::from_f32).collect();
    let ptr = gpu.alloc(values.len() * 2)?;
    gpu.copy_h2d(bf16_bytes(&values), ptr)?;
    Ok(ptr)
}

fn upload_u32(gpu: &dyn GpuBackend, values: &[u32]) -> Result<spark_runtime::gpu::DevicePtr> {
    let ptr = gpu.alloc(std::mem::size_of_val(values))?;
    gpu.copy_h2d(u32_bytes(values), ptr)?;
    Ok(ptr)
}

fn download(
    gpu: &dyn GpuBackend,
    ptr: spark_runtime::gpu::DevicePtr,
    len: usize,
) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; len * 4];
    gpu.copy_d2h(ptr, &mut bytes)?;
    Ok(f32_slice(&bytes).to_vec())
}

fn download_bf16(
    gpu: &dyn GpuBackend,
    ptr: spark_runtime::gpu::DevicePtr,
    len: usize,
) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; len * 2];
    gpu.copy_d2h(ptr, &mut bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect())
}

fn download_u32(
    gpu: &dyn GpuBackend,
    ptr: spark_runtime::gpu::DevicePtr,
    len: usize,
) -> Result<Vec<u32>> {
    let mut bytes = vec![0u8; len * 4];
    gpu.copy_d2h(ptr, &mut bytes)?;
    Ok(u32_slice(&bytes).to_vec())
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1.0e-4,
            "idx {idx}: actual {a} != expected {e}"
        );
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn bf16_bytes(v: &[bf16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn u32_bytes(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn f32_slice(b: &[u8]) -> &[f32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len() / 4) }
}

fn u32_slice(b: &[u8]) -> &[u32] {
    unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<u32>(), b.len() / 4) }
}
