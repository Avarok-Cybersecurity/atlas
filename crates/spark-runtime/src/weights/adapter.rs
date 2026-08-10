// SPDX-License-Identifier: AGPL-3.0-only

//! PEFT adapter loader: `adapter_model.safetensors` → [`WeightStore`].
//!
//! Not `SafetensorsLoader` because (a) that loader only probes
//! `model.safetensors*` names (weights/loader.rs) and (b)
//! `WeightDtype::from_safetensors` rejects F16 (weights.rs), the PEFT
//! default save dtype. F16 is converted to BF16 on the host here so no
//! F16 ever reaches a kernel or the WeightDtype whitelist.
//!
//! NOTE: the device copies made here become garbage once the adapter is
//! packed into the fixed-address LoRA pool and are never freed (no weight
//! dealloc anywhere in Atlas). Accepted leak at adapter scale (~MBs).

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::{bf16, f16};

use super::{WeightDtype, WeightStore, WeightTensor, evict_page_cache};
use crate::gpu::GpuBackend;

/// Load a PEFT adapter's `adapter_model.safetensors` from `adapter_dir`
/// onto the GPU. Mirrors the single-file path of `SafetensorsLoader`
/// (mmap → per-tensor alloc + copy_h2d → page-cache evict) with a
/// host-side F16→BF16 conversion branch added.
pub fn load_adapter_safetensors(
    adapter_dir: &Path,
    gpu: &dyn GpuBackend,
    oom_reserve_bytes: usize,
) -> Result<WeightStore> {
    let path = adapter_dir.join("adapter_model.safetensors");
    if !path.exists() {
        if adapter_dir.join("adapter_model.bin").exists() {
            bail!(
                "REJECT[pickle-adapter]: {} ships adapter_model.bin (torch pickle); \
                 re-save with safe_serialization=True",
                adapter_dir.display()
            );
        }
        bail!("No adapter_model.safetensors in {}", adapter_dir.display());
    }

    // Header-only preflight (no mmap): F16 counts 2 B/elem — identical to
    // its post-conversion BF16 footprint.
    let estimated = super::estimate_load_bytes(std::slice::from_ref(&path), &|_| false)?;
    let free = gpu.free_memory()?;
    if estimated + oom_reserve_bytes > free {
        bail!(
            "OOM pre-flight (LoRA adapter): {estimated} B adapter tensors + \
             {oom_reserve_bytes} B reserve exceeds {free} B free"
        );
    }

    let file = std::fs::File::open(&path)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;

    let mut weights = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let data = view.data();
        let (bytes, dtype): (Cow<'_, [u8]>, WeightDtype) = match view.dtype() {
            safetensors::Dtype::F16 => {
                // Host-side F16 -> BF16 (locked decision; `half = "2"` is
                // already a spark-runtime dep).
                let conv: Vec<u8> = data
                    .chunks_exact(2)
                    .flat_map(|c| {
                        bf16::from_f32(f16::from_le_bytes([c[0], c[1]]).to_f32()).to_le_bytes()
                    })
                    .collect();
                (Cow::Owned(conv), WeightDtype::BF16)
            }
            safetensors::Dtype::F32 => {
                // Host-side F32 -> BF16. PEFT's DEFAULT LoRA save dtype is F32
                // (the trainable adapter params stay fp32), so most real
                // adapters land here. The pool pack (`lora/mod.rs`, BF16_BYTES)
                // and `dense_gemv_bf16` assume BF16 unconditionally, so an F32
                // tensor left as-is is read at half stride = garbage delta
                // (silent: load succeeds, output corrupts). Convert here,
                // mirroring the F16 branch. The BF16 fixture never exercised
                // this path, which is why it hid.
                let conv: Vec<u8> = data
                    .chunks_exact(4)
                    .flat_map(|c| {
                        bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes()
                    })
                    .collect();
                (Cow::Owned(conv), WeightDtype::BF16)
            }
            other => (
                Cow::Borrowed(data),
                WeightDtype::from_safetensors(other)
                    .with_context(|| format!("LoRA adapter tensor '{name}'"))?,
            ),
        };
        let ptr = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(&bytes, ptr)?;
        weights.insert(name, WeightTensor { ptr, shape, dtype });
    }

    // Drop mmap before evicting page cache (GB10 unified memory).
    drop(tensors);
    drop(mmap);
    evict_page_cache(&file);

    Ok(WeightStore::from_map(weights))
}

// Gated on `feature = "cuda"`: the test constructs a real `AtlasCudaBackend`
// (a CUDA-only module), so the metal / no-CUDA build must not compile it.
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::load_adapter_safetensors;
    use crate::cuda_backend::AtlasCudaBackend;
    use crate::gpu::GpuBackend; // brings copy_d2h into scope
    use crate::weights::WeightDtype;
    use half::bf16;
    use safetensors::Dtype;
    use safetensors::serialize_to_file;
    use safetensors::tensor::TensorView;
    use std::collections::HashMap;

    /// Regression test for the PEFT-default-F32 → BF16 fix.
    ///
    /// PEFT saves LoRA adapters as F32 by default; before the fix
    /// `load_adapter_safetensors` left them F32 while the pool pack +
    /// `dense_gemv_bf16` read at BF16 stride = silent garbage delta. This
    /// builds a real F32 `adapter_model.safetensors` (two PEFT-style tensors,
    /// `lora_A` + `lora_B`), loads it on a live CUDA device, and asserts every
    /// returned tensor is BF16 and round-trips bit-exact.
    ///
    /// Gated `#[ignore]` (Atlas convention) because `AtlasCudaBackend::new`
    /// touches the CUDA driver; a GPU-less `cargo test` skips it. Opt in with
    /// `-- --ignored`.
    #[test]
    #[ignore = "requires a free CUDA device (GB10)"]
    fn f32_peft_adapter_loads_and_round_trips_as_bf16() {
        // Values all exactly representable in bf16 (≤8-bit mantissa), so the
        // F32 → BF16 conversion must round-trip bit-exact.
        let a_vals: [f32; 8] = [1.0, -2.0, 0.5, 0.25, 3.0, -1.5, 0.0, 8.0];
        let b_vals: [f32; 8] = [4.0, -0.75, 0.125, 16.0, -6.0, 2.0, 0.0625, -1.0];
        // lora_A [r=2, in=4]; lora_B [out=4, r=2]. Raw little-endian F32 bytes.
        let a_shape = vec![2usize, 4usize];
        let b_shape = vec![4usize, 2usize];
        let a_bytes: Vec<u8> = a_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Realistic PEFT keys (the loader does not parse names — the layer
        // allow-list lives downstream in spark-model — but keep them real).
        let a_key = "base_model.model.model.layers.3.self_attn.k_proj.lora_A.weight";
        let b_key = "base_model.model.model.layers.3.self_attn.k_proj.lora_B.weight";

        // Unique tempdir with no extra dep (spark-runtime has no tempfile
        // dev-dep): per-pid + per-thread subdir.
        let dir = std::env::temp_dir().join(format!(
            "atlas_adapter_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("adapter_model.safetensors");

        let a_view = TensorView::new(Dtype::F32, a_shape.clone(), &a_bytes).unwrap();
        let b_view = TensorView::new(Dtype::F32, b_shape.clone(), &b_bytes).unwrap();
        let mut map: HashMap<String, TensorView> = HashMap::new();
        map.insert(a_key.to_string(), a_view);
        map.insert(b_key.to_string(), b_view);
        serialize_to_file(map, None, &path).unwrap();

        // Real GPU backend. The loader only allocs + copies (launches no
        // kernel), but pass the codegen'd PTX set to mirror prod init.
        let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules()).unwrap();

        let store = load_adapter_safetensors(&dir, &gpu, 0).unwrap();

        // (1) dtype: the F32 adapter must load as BF16 — the core of the fix.
        for (key, shape, vals) in [(a_key, &a_shape, &a_vals), (b_key, &b_shape, &b_vals)] {
            let t = store.get(key).unwrap();
            assert_eq!(
                t.dtype,
                WeightDtype::BF16,
                "F32 adapter tensor must load as BF16"
            );
            assert_eq!(&t.shape, shape);

            // (2) values round-trip: read the 2-byte/elem BF16 back off device.
            let mut back = vec![0u8; vals.len() * 2];
            gpu.copy_d2h(t.ptr, &mut back).unwrap();
            for (i, chunk) in back.chunks_exact(2).enumerate() {
                let got = bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                assert_eq!(got, vals[i], "tensor '{key}' elem {i} F32->BF16 round-trip");
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
