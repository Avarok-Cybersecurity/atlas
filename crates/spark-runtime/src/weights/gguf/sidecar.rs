// SPDX-License-Identifier: AGPL-3.0-only

//! mmproj vision-tower sidecar support for the GGUF loader.
//!
//! Multimodal GGUF checkpoints (e.g. `prism-ml/Ternary-Bonsai-27B-gguf`) ship
//! the vision tower as a SEPARATE `*mmproj*.gguf` file (`general.architecture =
//! clip`) alongside the text backbone. This module holds the pieces that let the
//! loader detect that sidecar and fold it into the SAME [`WeightStore`] as the
//! backbone: the mmproj-aware file discovery, a shared per-file open helper, the
//! combined pre-flight footprint estimate, and the per-tensor [`load_pass`] loop
//! extracted from the backbone path so both files run through identical code.
//!
//! The clip file's tensor names live in a disjoint `v.*` / `mm.*` namespace and
//! translate to `model.visual.*` (see [`super::names`]), so merging into the
//! backbone's `model.layers.*` map is collision-free by construction. The one
//! special case is the temporal-split patch-embedding, fused in-pass via
//! [`super::value_transform::patch_embed_concat`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::{container, dequant_cpu, dequant_to_device, names, value_transform};
use crate::gpu::GpuBackend;
use crate::weights::{WeightDtype, WeightTensor};

/// True if `p` is an mmproj vision-tower sidecar (matched by filename, since the
/// projector's `general.architecture = clip` is only visible after parsing).
pub fn is_mmproj(p: &Path) -> bool {
    p.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.to_ascii_lowercase().contains("mmproj"))
        .unwrap_or(false)
}

/// Locate an mmproj vision-tower sidecar GGUF in `dir`, if present. Matches the
/// lexicographically-first `*mmproj*.gguf` that is not `backbone` (the
/// already-selected text model), so a text-only model dir returns `None` and a
/// multimodal dir returns the projector to load as a second pass.
pub fn find_mmproj(dir: &Path, backbone: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("gguf"))
        .filter(|p| is_mmproj(p) && p != backbone)
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

/// Open + mmap + parse one GGUF file. The returned [`memmap2::Mmap`] and
/// [`container::GgufFile`] are independent of the `File`, which is kept only so
/// the caller can `evict_page_cache` it after the pass.
pub fn open_gguf(path: &Path) -> Result<(std::fs::File, memmap2::Mmap, container::GgufFile)> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    // SAFETY: same mmap contract as the safetensors loader.
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let gguf = container::GgufFile::parse(&mmap)
        .with_context(|| format!("Failed to parse GGUF container: {}", path.display()))?;
    Ok((file, mmap, gguf))
}

/// Sum the BF16 footprint of every tensor a pass will actually keep (i.e. that
/// `names::translate` maps to a stored tensor, plus the clip patch-embed frames
/// that are fused rather than name-mapped), for the pre-flight OOM estimate.
pub fn est_bf16(gguf: &container::GgufFile, arch: &str) -> usize {
    let is_clip = value_transform::is_clip(arch);
    gguf.tensors
        .iter()
        .filter(|t| {
            (is_clip && value_transform::vision_patch_frame(&t.name).is_some())
                || !matches!(
                    names::translate(&t.name, arch),
                    None | Some(names::GgufName::Drop)
                )
        })
        .map(|t| t.num_elements() * WeightDtype::BF16.byte_size())
        .sum()
}

/// Load every recognized tensor from one already-parsed GGUF into `weights`,
/// dequantizing to BF16 on the way in. `arch` selects the name translation;
/// `gdn` (Some only for the qwen35 backbone) drives the GDN/RMSNorm value
/// fixups. Shared by the backbone pass and the `clip` mmproj sidecar pass so
/// both land tensors into the same [`WeightStore`] identically.
///
/// For a clip pass, `gdn` is `None` (its `arch` can never be qwen35-family) and
/// the file carries no `ExpertStack` names, so every tensor takes the plain
/// `dequant_to_device` path — except the temporal-split patch-embed weight
/// frames, which are dequantized, accumulated, and fused into the single
/// `model.visual.patch_embed.proj.weight` after the loop.
#[allow(clippy::too_many_arguments)]
pub fn load_pass(
    loader: &super::GgufLoader,
    gpu: &dyn GpuBackend,
    gguf: &container::GgufFile,
    mmap: &[u8],
    arch: &str,
    gdn: Option<value_transform::GdnDims>,
    force_cpu: bool,
    native_q2: bool,
    q2_group: usize,
    q2_variant: container::Q2Group,
    weights: &mut HashMap<String, WeightTensor>,
    skipped: &mut usize,
) -> Result<()> {
    // Patch-embed temporal-frame fan-in state (clip only).
    let vpatch = if value_transform::is_clip(arch) {
        value_transform::vision_patch_dims(gguf)
    } else {
        None
    };
    let mut patch_frames: Vec<Option<Vec<f32>>> = Vec::new();

    for tensor in &gguf.tensors {
        let num_elements = tensor.num_elements();
        let raw_len = gguf
            .tensor_byte_size(tensor, q2_variant)
            .with_context(|| format!("byte-len for tensor {}", tensor.name))?;
        let start = gguf.tensor_abs_offset(tensor);
        let raw = mmap
            .get(start..start + raw_len)
            .with_context(|| format!("tensor {} out of bounds in GGUF", tensor.name))?;
        let id = tensor.ggml_type.id();

        // Patch-embed temporal frames fan IN (many GGUF tensors → one HF tensor,
        // the mirror of ExpertStack's fan-out); dequant to F32 and stash for the
        // post-loop concat. Matched BEFORE names::translate because the `.1`
        // frame suffix breaks the default `.weight`/`.bias` name split.
        if let (Some(_), Some(frame)) = (vpatch, value_transform::vision_patch_frame(&tensor.name))
        {
            let gt = dequant_cpu::GgmlType::from_id(id, q2_group)
                .with_context(|| format!("ggml type for {}", tensor.name))?;
            let mut vals = vec![0f32; num_elements];
            dequant_cpu::dequant_to_f32(gt, raw, num_elements, &mut vals)
                .with_context(|| format!("CPU dequant patch frame {}", tensor.name))?;
            if patch_frames.len() <= frame {
                patch_frames.resize(frame + 1, None);
            }
            patch_frames[frame] = Some(vals);
            continue;
        }

        let target = match names::translate(&tensor.name, arch) {
            Some(names::GgufName::Drop) | None => continue,
            Some(t) => t,
        };

        // GGUF dims are ggml-order; Atlas/HF shape is the reverse.
        let hf_shape: Vec<usize> = tensor.dims.iter().rev().copied().collect();

        // ── Native keep-packed Q2_0 short-circuit ──
        // When `ATLAS_GGUF_NATIVE_Q2=1`, upload the raw `block_q2_0` bytes for
        // the big transform-free FFN projections UNCHANGED and tag them
        // `PackedQ2_0` so the model's `q2_0_gemv` decode path dequants in-kernel
        // (no BF16 expansion, no downstream NVFP4 requant). This is the whole
        // memory win. Excludes GDN reorder tensors via `!value_transform::needs`
        // (a column reorder would split blocks). Non-id-42 and non-FFN tensors,
        // and the flag-off default, are untouched below.
        if native_q2
            && id == 42
            && let names::GgufName::Direct(ref hf_name) = target
            && names::is_keep_packed_proj(hf_name)
            && !value_transform::needs(hf_name)
        {
            let ptr = gpu.alloc(raw.len())?;
            gpu.copy_h2d(raw, ptr)?;
            weights.insert(
                hf_name.clone(),
                WeightTensor {
                    ptr,
                    shape: hf_shape,
                    dtype: WeightDtype::PackedQ2_0 {
                        group: q2_group as u16,
                    },
                },
            );
            continue;
        }

        // ── Native keep-packed Q2_0 GDN row-permute short-circuit (Tier-1c) ──
        // The big GDN input projections (`in_proj_qkv` V-region, `in_proj_z`)
        // carry a value-head ROW reorder. Because that permutation moves whole
        // `value_head_dim`-row block-runs and one row is an integer number of
        // `block_q2_0` blocks, we can apply it directly to the PACKED bytes and
        // keep the weight 2-bit (byte-exact vs dequant→reorder→requant). The
        // within-row column reorder of `out_proj` is NOT handled here (stays on
        // its NVFP4/BF16 path). Runs BEFORE the CPU-dequant `needs()` branch so
        // these tensors never expand to BF16.
        if native_q2
            && id == 42
            && let names::GgufName::Direct(ref hf_name) = target
            && let Some(after_qk) = value_transform::packed_reorder_rows(hf_name)
            && let Some(dims) = gdn
        {
            let block_bytes = 2 + q2_group / 4; // fp16 scale + group/4 code bytes
            let permuted = value_transform::reorder_packed_rows(
                raw,
                &hf_shape,
                &dims,
                after_qk,
                q2_group,
                block_bytes,
            )
            .with_context(|| format!("packed GDN row-permute {hf_name}"))?;
            let ptr = gpu.alloc(permuted.len())?;
            gpu.copy_h2d(&permuted, ptr)?;
            weights.insert(
                hf_name.clone(),
                WeightTensor {
                    ptr,
                    shape: hf_shape,
                    dtype: WeightDtype::PackedQ2_0 {
                        group: q2_group as u16,
                    },
                },
            );
            continue;
        }

        // For qwen35 tensors whose VALUES need fixing, dequant on the CPU and
        // rewrite the BF16 host bytes before upload; everything else (all clip
        // tensors included) takes the prefer-GPU path.
        let bf16_ptr = match (&target, gdn) {
            (names::GgufName::Direct(hf_name), Some(dims)) if value_transform::needs(hf_name) => {
                let gt = dequant_cpu::GgmlType::from_id(id, q2_group)
                    .with_context(|| format!("ggml type for {}", tensor.name))?;
                let mut vals = vec![0f32; num_elements];
                dequant_cpu::dequant_to_f32(gt, raw, num_elements, &mut vals)
                    .with_context(|| format!("CPU dequant for qwen35 transform {}", tensor.name))?;
                value_transform::apply(hf_name, &mut vals, &hf_shape, &dims)
                    .with_context(|| format!("qwen35 value transform {hf_name}"))?;
                let host = value_transform::to_bf16_bytes(&vals);
                let ptr = gpu.alloc(host.len())?;
                gpu.copy_h2d(&host, ptr)?;
                ptr
            }
            _ => dequant_to_device(gpu, id, raw, num_elements, q2_group, force_cpu)
                .with_context(|| format!("dequant tensor {}", tensor.name))?,
        };

        match target {
            names::GgufName::Direct(hf_name) => {
                weights.insert(
                    hf_name,
                    WeightTensor {
                        ptr: bf16_ptr,
                        shape: hf_shape,
                        dtype: WeightDtype::BF16,
                    },
                );
            }
            names::GgufName::ExpertStack { layer, proj } => {
                loader.emit_experts(weights, bf16_ptr, &hf_shape, layer, proj, skipped)?;
            }
            names::GgufName::Drop => unreachable!("Drop filtered above"),
        }
    }

    // Post-loop: fuse the collected patch-embed temporal frames into the single
    // `[out_ch, in_ch·T·patch·patch]` linear weight the vision consumer reads.
    if let Some(dims) = vpatch
        && !patch_frames.is_empty()
    {
        let frames: Vec<&[f32]> = patch_frames
            .iter()
            .enumerate()
            .map(|(t, f)| {
                f.as_deref()
                    .with_context(|| format!("missing patch_embd temporal frame {t}"))
            })
            .collect::<Result<_>>()?;
        let fused = value_transform::patch_embed_concat(&frames, &dims)?;
        let host = value_transform::to_bf16_bytes(&fused);
        let ptr = gpu.alloc(host.len())?;
        gpu.copy_h2d(&host, ptr)?;
        weights.insert(
            value_transform::VISION_PATCH_EMBED_HF.to_string(),
            WeightTensor {
                ptr,
                shape: vec![
                    dims.out_ch,
                    dims.in_ch * frames.len() * dims.patch * dims.patch,
                ],
                dtype: WeightDtype::BF16,
            },
        );
    }

    Ok(())
}
