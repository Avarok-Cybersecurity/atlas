// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn values(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .flat_map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = ((state >> 32) as i32) as f32 / i32::MAX as f32;
            half::bf16::from_f32(x).to_bits().to_le_bytes()
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA kernels and a GPU"]
fn bf16_verify_out_matches_serial_decode() -> Result<()> {
    check(values(2560 * 6144, 0x1337))
}

#[test]
#[ignore = "requires CUDA and ATLAS_GDN_OUT_WEIGHT checkpoint BF16 weight"]
fn bf16_verify_checkpoint_out_matches_serial_decode() -> Result<()> {
    check(std::fs::read(std::env::var("ATLAS_GDN_OUT_WEIGHT")?)?)
}

fn check(weight_bytes: Vec<u8>) -> Result<()> {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let gemv = gpu.kernel("gemv", "dense_gemv_bf16")?;
    let gemm = gpu.kernel("gemm", "dense_gemm_bf16")?;
    let (hidden, value_dim) = (2560, 6144);
    assert_eq!(weight_bytes.len(), hidden * value_dim * 2);
    let weight = DenseWeight {
        weight: gpu.alloc(weight_bytes.len())?,
    };
    gpu.copy_h2d(&weight_bytes, weight.weight)?;
    let input_bytes = values(4 * value_dim, 0x5678);
    let input = gpu.alloc(input_bytes.len())?;
    gpu.copy_h2d(&input_bytes, input)?;
    let size = 4 * hidden * 2;
    let guard = gpu.alloc(size + 32)?;
    let output = guard.offset(16);
    let reference = gpu.alloc(size)?;
    let old = gpu.alloc(size)?;
    let mut differences = 0;
    for rows in [2, 3, 4] {
        for row in 0..rows {
            ops::dense_gemv(
                &gpu,
                gemv,
                input.offset(row * value_dim * 2),
                &weight,
                reference.offset(row * hidden * 2),
                hidden as u32,
                value_dim as u32,
                stream,
            )?;
        }
        gpu.copy_h2d(&vec![0xa5; size + 32], guard)?;
        project(
            &gpu, gemv, gemm, input, &weight, output, rows, hidden, value_dim, true, stream,
        )?;
        project(
            &gpu, gemv, gemm, input, &weight, old, rows, hidden, value_dim, false, stream,
        )?;
        let bytes = rows * hidden * 2;
        let mut want = vec![0; bytes];
        let mut got = vec![0; size + 32];
        let mut old_bytes = vec![0; bytes];
        gpu.copy_d2h(reference, &mut want)?;
        gpu.copy_d2h(guard, &mut got)?;
        gpu.copy_d2h(old, &mut old_bytes)?;
        differences += old_bytes
            .chunks_exact(2)
            .zip(want.chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            &got[16..16 + bytes],
            want,
            "K={rows} verify must match serial BF16 bits"
        );
        assert!(got[..16].iter().all(|&x| x == 0xa5));
        assert!(got[16 + bytes..].iter().all(|&x| x == 0xa5));
    }
    assert!(
        differences > 0,
        "old GEMM must expose the reduction mismatch"
    );
    eprintln!("BF16 GDN outproj K2/3/4 exact; old GEMM differs in {differences} BF16 values");
    Ok(())
}
