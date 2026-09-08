// SPDX-License-Identifier: AGPL-3.0-only

//! Real CUDA coverage of cached PLE FP8 rows, scales, and slot indexing.
use super::*;

#[derive(serde::Deserialize)]
struct Row {
    bytes: Vec<u8>,
    expected_bf16: Vec<u16>,
}

#[derive(serde::Deserialize)]
struct Fixture {
    scale: f32,
    rows: Vec<Row>,
}

fn check(f: Fixture) -> Result<()> {
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let stream = gpu.default_stream();
    let width = f.rows[0].bytes.len();
    let dir = std::env::temp_dir().join(format!("atlas-ple-fp8-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let mut segments = Vec::new();
    for (i, row) in f.rows.iter().enumerate() {
        assert_eq!(row.bytes.len(), width);
        assert_eq!(row.expected_bf16.len(), width);
        let path = dir.join(format!("{i}.bin"));
        // Nonzero offsets expose accidentally reading from the file start.
        let mut bytes = vec![0xa5; 37];
        bytes.extend_from_slice(&row.bytes);
        std::fs::write(&path, bytes)?;
        segments.push((path, 37));
    }
    let mut cache =
        spark_storage::NgramRowCache::open_segmented(&segments, 1, None, width, f.rows.len() + 2)?;
    cache.set_constant_scale(f.scale)?;
    let scale = DevicePtr(cache.scale_dev_va()?.expect("constant scales"));
    let dims = PleIdDims {
        ngram_size: 2,
        heads_per_ngram: 1,
        multipliers: vec![1, 3],
        head_vocab_sizes: vec![f.rows.len() as u64],
        head_offsets: vec![0],
        eos_token_id: 0,
    };
    let dw = || DenseWeight {
        weight: DevicePtr::NULL,
    };
    let weights = PleWeights {
        key_proj: dw(),
        value_proj: dw(),
        norm_key: dw(),
        norm_query: dw(),
        norm_conv: dw(),
        conv1d: dw(),
    };
    let mut layer = PleLayer::new(
        dims,
        width,
        width,
        1,
        2,
        1,
        1e-6,
        weights,
        NgramTable::Cached(Box::new(cache)),
        NgramRowFormat::Fp8 { scale },
        f.rows.len() + 2,
        &gpu,
    )?;
    // Guard both ends and exercise prefill, decode, repeated and reordered IDs.
    let size = (f.rows.len() + 2) * width * 2;
    let guarded = gpu.alloc(size + 32)?;
    gpu.free(layer.emb)?;
    layer.emb = guarded.offset(16);
    for ids in [
        (0..f.rows.len() as u64).rev().collect::<Vec<_>>(),
        vec![0],
        vec![f.rows.len() as u64 - 1, 0, 0],
    ] {
        gpu.copy_h2d(&vec![0xa5; size + 32], guarded)?;
        layer.gather(&ids, ids.len(), 1, &gpu, stream)?;
        let mut got = vec![0; size + 32];
        gpu.copy_d2h(guarded, &mut got)?;
        let want: Vec<u8> = ids
            .iter()
            .flat_map(|&i| {
                f.rows[i as usize]
                    .expected_bf16
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
            })
            .collect();
        assert_eq!(&got[16..16 + want.len()], want);
        assert!(got[..16].iter().all(|&x| x == 0xa5));
        assert!(got[16 + want.len()..].iter().all(|&x| x == 0xa5));
    }
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA kernels and a GPU"]
fn ple_fp8_cached_gather_matches_independent_oracle() -> Result<()> {
    let scale = 0.37109375;
    let mut rows = Vec::new();
    // Every finite E4M3 encoding, including signed zero and subnormals.
    let codes: Vec<u8> = (0..=255).filter(|x| x & 127 != 127).collect();
    for chunk in codes.chunks(127) {
        let expected_bf16 = chunk
            .iter()
            .map(|&b| {
                let exp = (b >> 3) & 15;
                let mantissa = b & 7;
                let value = if exp == 0 {
                    f32::from(mantissa) * 2f32.powi(-9)
                } else {
                    (1.0 + f32::from(mantissa) / 8.0) * 2f32.powi(i32::from(exp) - 7)
                };
                let signed = if b & 128 != 0 { -value } else { value };
                half::bf16::from_f32(signed * scale).to_bits()
            })
            .collect();
        rows.push(Row {
            bytes: chunk.to_vec(),
            expected_bf16,
        });
    }
    check(Fixture { scale, rows })
}

#[test]
#[ignore = "requires CUDA and ATLAS_PLE_FP8_FIXTURE checkpoint oracle"]
fn ple_fp8_checkpoint_gather_matches_oracle() -> Result<()> {
    let path = std::env::var("ATLAS_PLE_FP8_FIXTURE")?;
    check(serde_json::from_slice(&std::fs::read(path)?)?)
}
