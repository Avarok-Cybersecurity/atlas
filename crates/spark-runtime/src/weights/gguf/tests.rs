// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::mock::MockGpuBackend;
use crate::weights::WeightLoader;

fn push_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_str(b: &mut Vec<u8>, s: &str) {
    push_u64(b, s.len() as u64);
    b.extend_from_slice(s.as_bytes());
}

/// Minimal valid GGUF v3: one F32 1-D tensor + a `general.alignment` KV.
fn build_single_f32_gguf(name: &str, vals: &[f32]) -> Vec<u8> {
    let mut b = Vec::new();
    push_u32(&mut b, 0x4655_4747); // "GGUF"
    push_u32(&mut b, 3); // version
    push_u64(&mut b, 1); // tensor_count
    push_u64(&mut b, 1); // kv_count
    push_str(&mut b, "general.alignment");
    push_u32(&mut b, 4); // UINT32
    push_u32(&mut b, 32);
    push_str(&mut b, name);
    push_u32(&mut b, 1); // n_dims
    push_u64(&mut b, vals.len() as u64); // dims[0]
    push_u32(&mut b, 0); // ggml_type F32
    push_u64(&mut b, 0); // offset
    let pad = (32 - (b.len() % 32)) % 32;
    b.extend(std::iter::repeat_n(0u8, pad));
    for v in vals {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

#[test]
fn loads_single_tensor_cpu_fallback() {
    // Mock cannot execute kernels, so force the CPU reference dequant path.
    unsafe { std::env::set_var("ATLAS_GGUF_FORCE_CPU", "1") };

    let vals = [1.0f32, -2.0, 3.5, 0.0, 7.0, -0.25];
    let bytes = build_single_f32_gguf("token_embd.weight", &vals);

    let dir = std::env::temp_dir().join(format!("atlas_gguf_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.gguf"), &bytes).unwrap();

    let gpu = MockGpuBackend::new();
    let store = GgufLoader::new()
        .load(&dir, &gpu, 1024 * 1024)
        .expect("GGUF load");

    assert_eq!(store.len(), 1);
    assert!(store.contains("model.embed_tokens.weight"));
    let t = store.get("model.embed_tokens.weight").unwrap();
    assert_eq!(t.shape, vec![6]);
    assert_eq!(t.dtype, WeightDtype::BF16);

    let raw = gpu.read_alloc(t.ptr).expect("bf16 bytes present");
    assert_eq!(raw.len(), 6 * WeightDtype::BF16.byte_size());
    let got: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect();
    assert_eq!(got, vals.to_vec());

    std::fs::remove_dir_all(&dir).ok();
    unsafe { std::env::remove_var("ATLAS_GGUF_FORCE_CPU") };
}

#[test]
fn find_gguf_picks_first() {
    let dir = std::env::temp_dir().join(format!("atlas_gguf_find_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("b.gguf"), b"x").unwrap();
    std::fs::write(dir.join("a.gguf"), b"x").unwrap();
    std::fs::write(dir.join("notes.txt"), b"x").unwrap();
    let found = find_gguf(&dir).unwrap();
    assert_eq!(found.file_name().unwrap().to_str().unwrap(), "a.gguf");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn find_gguf_skips_mmproj_and_find_mmproj_pairs() {
    let dir = std::env::temp_dir().join(format!("atlas_gguf_mmproj_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // The mmproj sorts lexicographically FIRST ('B' < 'T'), so a naive
    // first-file pick would wrongly select the sidecar as the backbone.
    std::fs::write(dir.join("Bonsai-mmproj-Q8_0.gguf"), b"x").unwrap();
    std::fs::write(dir.join("Ternary-Bonsai-27B-Q2_0.gguf"), b"x").unwrap();

    let backbone = find_gguf(&dir).unwrap();
    assert_eq!(
        backbone.file_name().unwrap().to_str().unwrap(),
        "Ternary-Bonsai-27B-Q2_0.gguf"
    );
    let mmproj = sidecar::find_mmproj(&dir, &backbone).unwrap();
    assert_eq!(
        mmproj.file_name().unwrap().to_str().unwrap(),
        "Bonsai-mmproj-Q8_0.gguf"
    );

    // A text-only dir yields no sidecar.
    let dir2 = std::env::temp_dir().join(format!("atlas_gguf_textonly_{}", std::process::id()));
    std::fs::create_dir_all(&dir2).unwrap();
    std::fs::write(dir2.join("model-Q2_0.gguf"), b"x").unwrap();
    let bb2 = find_gguf(&dir2).unwrap();
    assert!(sidecar::find_mmproj(&dir2, &bb2).is_none());

    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&dir2).ok();
}
