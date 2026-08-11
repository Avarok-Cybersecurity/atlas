// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3.5/3.6 (GDN-hybrid, `general.architecture = qwen35`) GGUF value
//! transforms.
//!
//! llama.cpp's `qwen35` converter stores three families of GDN / norm tensors
//! with a *different encoding* than the HF checkpoint Atlas's kernels expect.
//! The name map ([`super::names`]) already lands each tensor under its HF name;
//! this module fixes the VALUES (never re-quantizing — every transform runs on
//! the dequantized F32 host values, then rounds to BF16 once at the end, so
//! quant block boundaries are irrelevant and the near-1.0 norm offset keeps full
//! F32 precision). Gated on the `qwen35` arch by the caller; other archs never
//! see it.
//!
//! Three inverse transforms, diagnosed byte-for-byte against
//! `nvidia/Qwen3.6-27B-NVFP4`:
//!
//! 1. **RMSNorm +1 offset.** llama.cpp stores the raw norm weight; Atlas
//!    computes `x*(1+w)`, so it needs `w_hf = w_gguf - 1`. Applied to
//!    `input_layernorm`, `post_attention_layernorm`, `self_attn.q_norm`,
//!    `self_attn.k_norm` and the final `model.norm`. NOT the GDN `linear_attn.
//!    norm` (that one matches the reference untouched).
//!
//! 2. **SSM A recovery.** llama.cpp stores `ssm_a = -exp(A_log)`; Atlas's GDN
//!    wants the raw `A_log`, so `A_log = ln(-ssm_a)` (element-wise; `ssm_a` is
//!    negative). Applied to `linear_attn.A_log`.
//!
//! 3. **GDN value-head reorder.** llama.cpp's `_LinearAttentionVReorderBase`
//!    (conversion/qwen.py, `_reorder_v_heads`) reshapes the HF value-head axis
//!    `(num_k_heads, num_v_per_k, head_dim)` and swaps the first two dims to a
//!    tiled `(num_v_per_k, num_k_heads, head_dim)` order (so ggml can use a cheap
//!    broadcast repeat). We apply the INVERSE — gather HF head `i` from GGUF head
//!    `perm(i)` — to `dt_bias`, `A_log`, the V rows of `in_proj_qkv`, all rows of
//!    `in_proj_z` / `in_proj_a` / `in_proj_b`, the V channels of `conv1d`, and
//!    the V columns of `out_proj`.

use anyhow::{Result, ensure};

use super::container::GgufFile;

/// GDN head geometry, read from the GGUF `*.ssm.*` metadata keys.
#[derive(Clone, Copy, Debug)]
pub struct GdnDims {
    /// Linear-attention key heads (`ssm.group_count`).
    pub num_k_heads: usize,
    /// Linear-attention value heads (`ssm.time_step_rank`).
    pub num_v_heads: usize,
    /// Per-value-head dimension (`ssm.inner_size / num_v_heads`).
    pub value_head_dim: usize,
    /// Per-key-head dimension (`ssm.state_size`); sizes the Q|K partitions of
    /// the fused `in_proj_qkv` / `conv1d` so their V region can be located.
    pub key_head_dim: usize,
}

impl GdnDims {
    /// Value heads per key head (the reorder's inner group factor).
    fn num_v_per_k(&self) -> usize {
        self.num_v_heads / self.num_k_heads
    }

    /// Rows spanned by the fused Q and K partitions of `in_proj_qkv` / channels
    /// of `conv1d` (2 × key heads × key-head-dim). The V region follows.
    fn qk_rows(&self) -> usize {
        2 * self.key_head_dim * self.num_k_heads
    }

    /// Source GGUF value-head index for HF value-head `hf`. HF stores heads
    /// grouped by key head (`hf = k*num_v_per_k + r`); GGUF stores them tiled
    /// (`gguf = r*num_k_heads + k`). This inverts the converter's transpose.
    fn gguf_head(&self, hf: usize) -> usize {
        let vpk = self.num_v_per_k();
        let k = hf / vpk;
        let r = hf % vpk;
        r * self.num_k_heads + k
    }
}

/// Read GDN head geometry from GGUF metadata for architecture `arch`
/// (e.g. `"qwen35"`). Returns `None` if the SSM keys are absent or inconsistent.
pub fn gdn_dims(gguf: &GgufFile, arch: &str) -> Option<GdnDims> {
    let g = |suffix: &str| gguf.get_u64(&format!("{arch}.{suffix}"));
    let num_k_heads = g("ssm.group_count")? as usize;
    let num_v_heads = g("ssm.time_step_rank")? as usize;
    let inner_size = g("ssm.inner_size")? as usize;
    let key_head_dim = g("ssm.state_size")? as usize;
    if num_k_heads == 0
        || num_v_heads == 0
        || !num_v_heads.is_multiple_of(num_k_heads)
        || inner_size == 0
        || !inner_size.is_multiple_of(num_v_heads)
    {
        return None;
    }
    Some(GdnDims {
        num_k_heads,
        num_v_heads,
        value_head_dim: inner_size / num_v_heads,
        key_head_dim,
    })
}

/// True if `arch` is a Qwen3.5/3.6 GDN-hybrid architecture whose GGUF encoding
/// needs the value transforms in this module.
pub fn is_qwen35(arch: &str) -> bool {
    matches!(arch, "qwen35" | "qwen35moe" | "qwen3_5")
}

/// The per-tensor transform selected by (mapped) HF name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// Subtract 1.0 from every element (RMSNorm +1 offset).
    NormOffset,
    /// `A_log = ln(-ssm_a)` then value-head reorder along dim 0.
    ALog,
    /// Value-head reorder along the row axis. `after_qk` = the reordered region
    /// starts after the fused Q|K partition (`in_proj_qkv` / `conv1d`);
    /// otherwise the whole tensor is value heads. `head_dim_rows` = rows per
    /// value head (`false` → one, `true` → `value_head_dim`).
    ReorderRows { after_qk: bool, head_dim_rows: bool },
    /// Value-head reorder along the column (input) axis of `out_proj`.
    ReorderOutCols,
}

/// Classify a mapped HF tensor name into its transform, or `None` if untouched.
fn classify(hf: &str) -> Option<Op> {
    if hf == "model.norm.weight"
        || hf.ends_with(".input_layernorm.weight")
        || hf.ends_with(".post_attention_layernorm.weight")
        || hf.ends_with(".self_attn.q_norm.weight")
        || hf.ends_with(".self_attn.k_norm.weight")
    {
        return Some(Op::NormOffset);
    }
    if hf.ends_with(".linear_attn.A_log") {
        return Some(Op::ALog);
    }
    if hf.ends_with(".linear_attn.dt_bias")
        || hf.ends_with(".linear_attn.in_proj_a.weight")
        || hf.ends_with(".linear_attn.in_proj_b.weight")
    {
        return Some(Op::ReorderRows {
            after_qk: false,
            head_dim_rows: false,
        });
    }
    if hf.ends_with(".linear_attn.in_proj_z.weight") {
        return Some(Op::ReorderRows {
            after_qk: false,
            head_dim_rows: true,
        });
    }
    if hf.ends_with(".linear_attn.in_proj_qkv.weight") || hf.ends_with(".linear_attn.conv1d.weight")
    {
        return Some(Op::ReorderRows {
            after_qk: true,
            head_dim_rows: true,
        });
    }
    if hf.ends_with(".linear_attn.out_proj.weight") {
        return Some(Op::ReorderOutCols);
    }
    None
}

/// True if the loader must route this tensor through the CPU dequant + transform
/// path (i.e. its VALUES need fixing for the `qwen35` arch).
pub fn needs(hf_name: &str) -> bool {
    classify(hf_name).is_some()
}

/// For the native keep-packed Q2_0 path (`ATLAS_GGUF_NATIVE_Q2=1`): tensors whose
/// value transform is a *pure whole-row* value-head reorder ([`Op::ReorderRows`]
/// with `head_dim_rows = true`) can stay 2-bit — the permutation moves whole
/// `value_head_dim`-row blocks, and one row is an integer number of `block_q2_0`
/// blocks (`K / group`), so it never splits a group. [`reorder_packed_rows`]
/// applies the SAME permutation directly to the packed bytes, leaving the weight
/// HF-correct while packed. Returns `Some(after_qk)` for the two big GDN input
/// projections (`in_proj_qkv` → V-region only, `in_proj_z` → whole tensor),
/// `None` for everything else (which stays on the dequant→BF16 path):
///   * `in_proj_a`/`in_proj_b`/`dt_bias` reorder SINGLE rows (`head_dim_rows =
///     false`); tiny, left BF16.
///   * `out_proj` is a within-row COLUMN reorder ([`Op::ReorderOutCols`]) that
///     would need an intra-row block shuffle — deferred, stays on its NVFP4/BF16
///     path.
///   * `A_log` / norms are F32 scalar transforms — never packed.
pub fn packed_reorder_rows(hf_name: &str) -> Option<bool> {
    // Match by NAME, not just the `Op`: `conv1d` shares `in_proj_qkv`'s
    // `ReorderRows{after_qk,head_dim_rows}` classification but is a 3-D
    // recurrence tensor (`[d_inner,1,k]`) that must stay BF16 — never packed.
    if hf_name.ends_with(".linear_attn.in_proj_qkv.weight") {
        return Some(true); // reorder V-region rows only (after the Q|K partition)
    }
    if hf_name.ends_with(".linear_attn.in_proj_z.weight") {
        return Some(false); // whole tensor is value heads
    }
    None
}

/// Apply the value-head row reorder ([`packed_reorder_rows`]) directly to the raw
/// `block_q2_0` bytes of a keep-packed tensor. Byte-exact analog of
/// [`reorder_rows`]: because dequant is per-block and one HF row spans
/// `k / group` whole blocks (`row_bytes = (k/group) * block_bytes`), permuting
/// whole `value_head_dim`-row block-runs here is bit-identical to permuting the
/// dequantized rows. `raw` is `[n, k]` row-major packed; returns the reordered
/// copy.
pub fn reorder_packed_rows(
    raw: &[u8],
    hf_shape: &[usize],
    dims: &GdnDims,
    after_qk: bool,
    group: usize,
    block_bytes: usize,
) -> Result<Vec<u8>> {
    ensure!(
        hf_shape.len() == 2,
        "packed reorder expects 2-D [n,k], got {hf_shape:?}"
    );
    let (n, k) = (hf_shape[0], hf_shape[1]);
    ensure!(
        k % group == 0,
        "packed reorder: k={k} not a multiple of group={group}"
    );
    let row_bytes = (k / group) * block_bytes;
    ensure!(
        raw.len() == n * row_bytes,
        "packed reorder: raw len {} != n*row_bytes {}",
        raw.len(),
        n * row_bytes
    );
    let hd = dims.value_head_dim; // rows per value head
    let base_row = if after_qk { dims.qk_rows() } else { 0 };
    let region_rows = dims.num_v_heads * hd;
    ensure!(
        base_row + region_rows <= n,
        "packed reorder: V region rows [{base_row}..{}] exceed n={n}",
        base_row + region_rows
    );
    let mut out = raw.to_vec();
    let blk = hd * row_bytes; // bytes per value head
    for hf in 0..dims.num_v_heads {
        let g = dims.gguf_head(hf);
        let d = (base_row + hf * hd) * row_bytes;
        let s = (base_row + g * hd) * row_bytes;
        out[d..d + blk].copy_from_slice(&raw[s..s + blk]);
    }
    Ok(out)
}

/// Apply the qwen35 value transform (if any) to the dequantized F32 values
/// `buf` in place. `hf_shape` is the tensor's HF (row-major) shape. No-op for
/// names [`classify`] does not recognize.
pub fn apply(hf_name: &str, buf: &mut [f32], hf_shape: &[usize], dims: &GdnDims) -> Result<()> {
    let Some(op) = classify(hf_name) else {
        return Ok(());
    };
    match op {
        Op::NormOffset => buf.iter_mut().for_each(|x| *x -= 1.0),
        Op::ALog => {
            buf.iter_mut().for_each(|x| *x = (-*x).ln());
            reorder_rows(buf, hf_name, hf_shape, dims, false, false)?;
        }
        Op::ReorderRows {
            after_qk,
            head_dim_rows,
        } => reorder_rows(buf, hf_name, hf_shape, dims, after_qk, head_dim_rows)?,
        Op::ReorderOutCols => reorder_out_cols(buf, hf_name, hf_shape, dims)?,
    }
    Ok(())
}

/// Round an F32 slice to little-endian BF16 host bytes (the loader's upload
/// form). Round-to-nearest-even, NaN quieted.
pub fn to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
    }
    out
}

/// Elements produced by one row of a row-major tensor of shape `hf_shape` (the
/// product of every dim after the first; 1 for a 1-D tensor).
fn row_width(hf_shape: &[usize]) -> usize {
    hf_shape.iter().skip(1).product::<usize>().max(1)
}

/// Reorder value-head blocks along the outer (row) axis from GGUF tiled order to
/// HF grouped order. `after_qk` = the reordered region starts after the fused
/// Q|K partition (`in_proj_qkv` / `conv1d`); otherwise the whole tensor is value
/// heads. `head_dim_rows` = each value head spans `value_head_dim` rows (else 1).
fn reorder_rows(
    buf: &mut [f32],
    hf_name: &str,
    hf_shape: &[usize],
    dims: &GdnDims,
    after_qk: bool,
    head_dim_rows: bool,
) -> Result<()> {
    let rw = row_width(hf_shape);
    let hd = if head_dim_rows {
        dims.value_head_dim
    } else {
        1
    };
    let base = if after_qk { dims.qk_rows() * rw } else { 0 };
    let blk = hd * rw; // elements per value head
    let region = dims.num_v_heads * blk;
    ensure!(
        base + region <= buf.len(),
        "qwen35 reorder {hf_name}: V region [{base}..{}] exceeds {} elems",
        base + region,
        buf.len()
    );
    let src = buf[base..base + region].to_vec();
    for hf in 0..dims.num_v_heads {
        let g = dims.gguf_head(hf);
        let (d, s) = (base + hf * blk, g * blk);
        buf[d..d + blk].copy_from_slice(&src[s..s + blk]);
    }
    Ok(())
}

/// Reorder value-head column blocks (input dim) of `out_proj`, per output row.
/// `hf_shape = [out_dim, value_dim]`, `value_dim = num_v_heads * value_head_dim`.
fn reorder_out_cols(
    buf: &mut [f32],
    hf_name: &str,
    hf_shape: &[usize],
    dims: &GdnDims,
) -> Result<()> {
    ensure!(
        hf_shape.len() == 2,
        "qwen35 reorder {hf_name}: out_proj expects 2-D shape, got {hf_shape:?}"
    );
    let out_rows = hf_shape[0];
    let value_dim = hf_shape[1];
    ensure!(
        value_dim == dims.num_v_heads * dims.value_head_dim,
        "qwen35 reorder {hf_name}: value_dim {value_dim} != num_v_heads*value_head_dim {}",
        dims.num_v_heads * dims.value_head_dim
    );
    let cb = dims.value_head_dim; // elements per value-head column block
    ensure!(
        out_rows * value_dim <= buf.len(),
        "qwen35 reorder {hf_name}: {out_rows}x{value_dim} exceeds {} elems",
        buf.len()
    );
    let mut tmp = vec![0f32; value_dim];
    for row in 0..out_rows {
        let ro = row * value_dim;
        tmp.copy_from_slice(&buf[ro..ro + value_dim]);
        for hf in 0..dims.num_v_heads {
            let g = dims.gguf_head(hf);
            buf[ro + hf * cb..ro + hf * cb + cb].copy_from_slice(&tmp[g * cb..g * cb + cb]);
        }
    }
    Ok(())
}

// ── Vision (mmproj / `general.architecture = "clip"`) transforms ──
//
// The Qwen3-VL ViT tensors already land in Atlas's expected HF layout under the
// loader's normal GGUF→HF dim reversal: every `v.blk.*` projection (fused
// `attn_qkv` [1152,3456]→[3456,1152], `attn_out`, `ffn_up`/`ffn_down`), every
// LayerNorm + bias, `v.post_ln.*` (→ merger.norm — a plain BIASED LayerNorm, NO
// +1 offset) and `v.position_embd.weight` [1152,2304]→[2304,1152] is a plain
// copy. The ONE tensor needing custom shape work is the patch embedding:
// llama.cpp splits Qwen3-VL's Conv3d (temporal_patch_size = 2) into TWO per-frame
// Conv2d tensors that Atlas wants fused into a single
// [out_ch, in_ch·T·patch·patch] linear weight.

/// True if `arch` is a CLIP/mmproj vision tower (`general.architecture="clip"`).
pub fn is_clip(arch: &str) -> bool {
    arch == "clip"
}

/// The Atlas HF name the fused patch-embed weight is stored under (the name the
/// vision consumer `Qwen35WeightLoader::load_vision_encoder` reads).
pub const VISION_PATCH_EMBED_HF: &str = "model.visual.patch_embed.proj.weight";

/// Patch-embed geometry, read from the mmproj `clip.vision.*` metadata.
#[derive(Clone, Copy, Debug)]
pub struct VisionPatchDims {
    /// Output channels = ViT hidden (`clip.vision.embedding_length`, 1152).
    pub out_ch: usize,
    /// Input image channels (RGB → 3).
    pub in_ch: usize,
    /// Spatial patch side (`clip.vision.patch_size`, 16).
    pub patch: usize,
}

impl VisionPatchDims {
    /// Elements in one per-temporal-frame Conv2d row (`in_ch·patch·patch`).
    fn frame_row(&self) -> usize {
        self.in_ch * self.patch * self.patch
    }
}

/// Read [`VisionPatchDims`] from mmproj metadata; `None` if keys are missing.
pub fn vision_patch_dims(gguf: &GgufFile) -> Option<VisionPatchDims> {
    let out_ch = gguf.get_u64("clip.vision.embedding_length")? as usize;
    let patch = gguf.get_u64("clip.vision.patch_size")? as usize;
    if out_ch == 0 || patch == 0 {
        return None;
    }
    Some(VisionPatchDims {
        out_ch,
        in_ch: 3,
        patch,
    })
}

/// If `gguf_name` is a patch-embed temporal frame, return its frame index.
/// llama.cpp names frame 0 `v.patch_embd.weight` and each later frame
/// `v.patch_embd.weight.<n>` (`n` from 1). The trailing `.<n>` breaks the default
/// `.weight`/`.bias` split, so this MUST be matched before the name translator.
/// (`v.patch_embd.bias` returns `None` — it maps 1:1 through the name table.)
pub fn vision_patch_frame(gguf_name: &str) -> Option<usize> {
    let rest = gguf_name.strip_prefix("v.patch_embd.weight")?;
    if rest.is_empty() {
        return Some(0);
    }
    rest.strip_prefix('.').and_then(|n| n.parse::<usize>().ok())
}

/// Fuse the per-temporal-frame Conv2d patch-embed tensors into Atlas's
/// `[out_ch, in_ch·T·patch·patch]` linear weight (row-major F32).
///
/// `frames[t]` is one dequantized `v.patch_embd.weight{,.1,…}`, row-major in the
/// loader's reversed HF layout `[out_ch, in_ch, patch, patch]`. The output K axis
/// is ordered `c·(T·P·P) + t·(P·P) + ky·P + kx` — IDENTICAL to the pixel channel
/// order `vision_preprocess` writes (`off = c·(T·P·P)+t·(P·P)+py·P+px`), so the
/// patch-embed GEMM lines up channel-for-channel with the uploaded pixels.
///
/// LAYOUT ASSUMPTION (validate numerically on GB10): `frames` are in temporal
/// order — `v.patch_embd.weight` = t0, `v.patch_embd.weight.1` = t1. Swapping
/// them scrambles every patch embedding.
pub fn patch_embed_concat(frames: &[&[f32]], dims: &VisionPatchDims) -> Result<Vec<f32>> {
    let temporal = frames.len();
    ensure!(temporal > 0, "patch_embed_concat: no temporal frames");
    let blk = dims.patch * dims.patch; // P·P copied per (o,c,t)
    let frame_row = dims.frame_row(); // in_ch·P·P
    let dst_row = dims.in_ch * temporal * blk; // in_ch·T·P·P (= 1536)
    for (t, f) in frames.iter().enumerate() {
        ensure!(
            f.len() == dims.out_ch * frame_row,
            "patch_embed_concat: frame {t} has {} elems, expected {}",
            f.len(),
            dims.out_ch * frame_row
        );
    }
    let mut out = vec![0f32; dims.out_ch * dst_row];
    for o in 0..dims.out_ch {
        for c in 0..dims.in_ch {
            for (t, frame) in frames.iter().enumerate() {
                let src = o * frame_row + c * blk;
                let dst = o * dst_row + c * (temporal * blk) + t * blk;
                out[dst..dst + blk].copy_from_slice(&frame[src..src + blk]);
            }
        }
    }
    Ok(out)
}

/// f32 → bf16 bits, round-to-nearest-even (NaN quieted).
#[inline]
fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    if f.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x0000_7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

#[cfg(test)]
mod tests;
