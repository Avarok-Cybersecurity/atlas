// SPDX-License-Identifier: AGPL-3.0-only
//! Typed weight binding for the GLM-5.3-Flash KDA attention family.
//!
//! Every KDA `self_attn` block in the checkpoint binds through [`bind_kda_weights`], which is
//! **exhaustive and strict**: the tensor set must be exactly the 15 names below, every dtype and
//! shape is asserted, and any unrecognised `self_attn.*` tensor is a hard error. There is no
//! "skip what we don't know" path, silent or otherwise.
//!
//! ## Why this can be strict
//!
//! The family is structurally uniform. Audited across the checkpoint's **34** KDA layers:
//! **one** distinct (name, dtype, shape) signature, **0** quantisation artefacts, **0** missing
//! or unexpected tensors — while all 510 tensor hashes are distinct, so the blocks share
//! structure and nothing else. Layer 45 (MTP) is **DSA-shaped**, not KDA, and is not bindable
//! here; [`classify_attn_block`] separates the two from the tensor names alone.
//!
//! ## Traps this module exists to make impossible
//!
//! * The checkpoint stores **three** conv tensors of rank **3** (`[qkv, 1, kernel]`); HF holds
//!   one fused depthwise conv. Binding is `concat([q, k, v])` **in that order**, squeezed exactly
//!   once. Both the order and the squeeze are silent if wrong — the order because all three have
//!   identical shape, the squeeze because `[dim, 1, ks]` and `[dim, ks]` share their bytes.
//! * `A_log` is per **head** and F32; `dt_bias` is per **channel** and F32. Everything else is
//!   BF16. A loader that "helpfully" casts either to BF16 changes the gate.

use std::collections::BTreeSet;

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{Glm5NextKdaConfig, Glm5NextKdaWeights};
use crate::weight_map::DenseWeight;

/// The only two dtypes a KDA block contains.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KdaDtype {
    Bf16,
    F32,
}

impl KdaDtype {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "BF16" => Some(Self::Bf16),
            "F32" => Some(Self::F32),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }
}

/// One tensor as it sits in the checkpoint: dtype, shape and raw little-endian bytes.
pub struct RawTensor<'a> {
    pub dtype: KdaDtype,
    pub shape: Vec<usize>,
    pub bytes: &'a [u8],
}

/// A checkpoint slice scoped to ONE decoder layer. Names are layer-relative
/// (`self_attn.q_proj.weight`), so the same binder works for any layer index and any container.
pub trait KdaTensorSource {
    fn get(&self, name: &str) -> Option<RawTensor<'_>>;
    /// Every layer-relative name present, including non-attention ones.
    fn names(&self) -> Vec<String>;
}

/// The 15 `self_attn` tensors a KDA block has — and the complete list of what it may have.
///
/// Shapes are expressed against [`Glm5NextKdaConfig`] so a geometry change fails loudly here
/// rather than at launch. `H` = heads, `D` = head_dim, `Q` = H*D, `X` = hidden, `K` = conv kernel.
#[derive(Clone, Copy, Debug)]
pub struct TensorSpec {
    pub name: &'static str,
    pub dtype: KdaDtype,
    dims: &'static [Dim],
}

#[derive(Clone, Copy, Debug)]
enum Dim {
    Q,
    X,
    D,
    H,
    K,
    One,
}

use Dim::{D as DD, H as DH, K as DK, One as D1, Q as DQ, X as DX};

pub const KDA_TENSORS: &[TensorSpec] = &[
    TensorSpec {
        name: "self_attn.q_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, DX],
    },
    TensorSpec {
        name: "self_attn.k_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, DX],
    },
    TensorSpec {
        name: "self_attn.v_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, DX],
    },
    TensorSpec {
        name: "self_attn.q_conv1d.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, D1, DK],
    },
    TensorSpec {
        name: "self_attn.k_conv1d.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, D1, DK],
    },
    TensorSpec {
        name: "self_attn.v_conv1d.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, D1, DK],
    },
    TensorSpec {
        name: "self_attn.f_a_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DD, DX],
    },
    TensorSpec {
        name: "self_attn.f_b_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, DD],
    },
    TensorSpec {
        name: "self_attn.g_a_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DD, DX],
    },
    TensorSpec {
        name: "self_attn.g_b_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, DD],
    },
    TensorSpec {
        name: "self_attn.b_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DH, DX],
    },
    // F32 on disk, and F32 in the kernel signatures — no load-time conversion.
    TensorSpec {
        name: "self_attn.A_log",
        dtype: KdaDtype::F32,
        dims: &[DH],
    },
    TensorSpec {
        name: "self_attn.dt_bias",
        dtype: KdaDtype::F32,
        dims: &[DQ],
    },
    TensorSpec {
        name: "self_attn.o_norm.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DD],
    },
    TensorSpec {
        name: "self_attn.o_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DX, DQ],
    },
];

/// Names that identify a **DSA** (`deepseek_sparse_attention`) block, including the MTP layer.
/// Present so a caller can classify without guessing from the layer index.
pub const DSA_MARKERS: &[&str] = &[
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.indexer.wk.weight",
];

impl TensorSpec {
    pub fn expected_shape(&self, c: &Glm5NextKdaConfig) -> Vec<usize> {
        self.dims
            .iter()
            .map(|d| match d {
                Dim::Q => c.qkv_dim(),
                Dim::X => c.hidden,
                Dim::D => c.head_dim,
                Dim::H => c.heads,
                Dim::K => c.conv_kernel,
                Dim::One => 1,
            })
            .collect()
    }
}

/// What kind of attention block a layer's tensor names describe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttnBlockKind {
    Kda,
    /// `deepseek_sparse_attention`, and the MTP layer, which is DSA-shaped.
    Dsa,
    Unknown,
}

/// Classify from tensor names alone — never from the layer index, and never from `layer_types`,
/// which Slice 1 has to strip and rebuild.
pub fn classify_attn_block(names: &[String]) -> AttnBlockKind {
    let set: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    if DSA_MARKERS.iter().all(|m| set.contains(m)) {
        return AttnBlockKind::Dsa;
    }
    if KDA_TENSORS.iter().all(|t| set.contains(t.name)) {
        return AttnBlockKind::Kda;
    }
    AttnBlockKind::Unknown
}

/// Per-layer accounting, so "zero unknown, zero silent skips" is a reported number and not a
/// claim. `non_attn` is counted but deliberately NOT bound — FFN, norms and mHC are other slices.
#[derive(Clone, Debug, Default)]
pub struct KdaBindReport {
    pub layer_idx: usize,
    pub bound: usize,
    pub self_attn_seen: usize,
    pub non_attn_seen: usize,
    pub unknown_self_attn: Vec<String>,
    pub bytes: usize,
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

/// Bind one KDA block. Strict: exact tensor set, exact dtypes, exact shapes.
///
/// Weights are uploaded **verbatim** — BF16 stays BF16, F32 stays F32, nothing is converted,
/// requantised or dequantised, because nothing in a KDA block is quantised in the first place.
pub fn bind_kda_weights(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextKdaConfig,
    layer_idx: usize,
    src: &dyn KdaTensorSource,
) -> Result<(Glm5NextKdaWeights, KdaBindReport)> {
    cfg.validate()?;
    let names = src.names();
    let mut rep = KdaBindReport {
        layer_idx,
        ..Default::default()
    };

    let known: BTreeSet<&str> = KDA_TENSORS.iter().map(|t| t.name).collect();
    for n in &names {
        if n.starts_with("self_attn.") {
            rep.self_attn_seen += 1;
            if !known.contains(n.as_str()) {
                rep.unknown_self_attn.push(n.clone());
            }
        } else {
            rep.non_attn_seen += 1;
        }
    }
    if !rep.unknown_self_attn.is_empty() {
        bail!(
            "layer {layer_idx}: {} unrecognised self_attn tensor(s): {:?} — a KDA block has \
             exactly {} and this binder refuses to skip anything",
            rep.unknown_self_attn.len(),
            rep.unknown_self_attn,
            KDA_TENSORS.len()
        );
    }

    let mut fetch = |spec: &TensorSpec| -> Result<Vec<u8>> {
        let t = src
            .get(spec.name)
            .with_context(|| format!("layer {layer_idx}: missing {}", spec.name))?;
        if t.dtype != spec.dtype {
            bail!(
                "layer {layer_idx}: {} is {} but a KDA block requires {} — casting it would \
                 change the numerics",
                spec.name,
                t.dtype.name(),
                spec.dtype.name()
            );
        }
        let want = spec.expected_shape(cfg);
        if t.shape != want {
            bail!(
                "layer {layer_idx}: {} has shape {:?}, expected {want:?}",
                spec.name,
                t.shape
            );
        }
        let elem = match spec.dtype {
            KdaDtype::Bf16 => 2,
            KdaDtype::F32 => 4,
        };
        let expect_bytes = want.iter().product::<usize>() * elem;
        if t.bytes.len() != expect_bytes {
            bail!(
                "layer {layer_idx}: {} is {} B, shape {want:?} implies {expect_bytes} B",
                spec.name,
                t.bytes.len()
            );
        }
        rep.bound += 1;
        rep.bytes += t.bytes.len();
        Ok(t.bytes.to_vec())
    };

    let by_name = |n: &str| -> &TensorSpec { KDA_TENSORS.iter().find(|t| t.name == n).unwrap() };
    let mut raw = |n: &str| fetch(by_name(n));

    let q_proj = raw("self_attn.q_proj.weight")?;
    let k_proj = raw("self_attn.k_proj.weight")?;
    let v_proj = raw("self_attn.v_proj.weight")?;
    // 🪤 concat in q, k, v order — the same order as `cat([q_proj, k_proj, v_proj])`. The squeeze
    // of the singleton middle dim is validated above (rank 3, middle == 1) and is a SHAPE-only
    // operation: `[dim, 1, ks]` and `[dim, ks]` are the same row-major bytes, so nothing moves.
    let mut conv = raw("self_attn.q_conv1d.weight")?;
    conv.extend_from_slice(&raw("self_attn.k_conv1d.weight")?);
    conv.extend_from_slice(&raw("self_attn.v_conv1d.weight")?);
    debug_assert_eq!(conv.len(), cfg.conv_dim() * cfg.conv_kernel * 2);
    let f_a = raw("self_attn.f_a_proj.weight")?;
    let f_b = raw("self_attn.f_b_proj.weight")?;
    let g_a = raw("self_attn.g_a_proj.weight")?;
    let g_b = raw("self_attn.g_b_proj.weight")?;
    let b_proj = raw("self_attn.b_proj.weight")?;
    let a_log = raw("self_attn.A_log")?;
    let dt_bias = raw("self_attn.dt_bias")?;
    let o_norm = raw("self_attn.o_norm.weight")?;
    let o_proj = raw("self_attn.o_proj.weight")?;

    let dw = |b: &[u8]| -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: upload(gpu, b)?,
        })
    };
    let w = Glm5NextKdaWeights {
        q_proj: dw(&q_proj)?,
        k_proj: dw(&k_proj)?,
        v_proj: dw(&v_proj)?,
        conv: dw(&conv)?,
        f_a: dw(&f_a)?,
        f_b: dw(&f_b)?,
        dt_bias: upload(gpu, &dt_bias)?,
        a_log: upload(gpu, &a_log)?,
        b_proj: dw(&b_proj)?,
        g_a: dw(&g_a)?,
        g_b: dw(&g_b)?,
        o_norm: dw(&o_norm)?,
        o_proj: dw(&o_proj)?,
    };
    if rep.bound != KDA_TENSORS.len() {
        bail!(
            "layer {layer_idx}: bound {} of {} tensors",
            rep.bound,
            KDA_TENSORS.len()
        );
    }
    Ok((w, rep))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Glm5NextKdaConfig {
        Glm5NextKdaConfig {
            hidden: 4096,
            heads: 64,
            head_dim: 128,
            conv_kernel: 4,
            gate_lower_bound: -5.0,
            rms_norm_eps: 1e-5,
            l2_eps: 1e-6,
            chunk: 32,
        }
    }

    /// The spec table must reproduce the shapes measured off the real checkpoint. These are the
    /// audited values for GLM-5.3-Flash-NVFP4 @ 9e0d74e3, identical on all 34 KDA layers.
    #[test]
    fn tensor_spec_matches_the_audited_checkpoint_shapes() {
        let c = cfg();
        let want: &[(&str, &str, &[usize])] = &[
            ("self_attn.q_proj.weight", "BF16", &[8192, 4096]),
            ("self_attn.k_proj.weight", "BF16", &[8192, 4096]),
            ("self_attn.v_proj.weight", "BF16", &[8192, 4096]),
            ("self_attn.q_conv1d.weight", "BF16", &[8192, 1, 4]),
            ("self_attn.k_conv1d.weight", "BF16", &[8192, 1, 4]),
            ("self_attn.v_conv1d.weight", "BF16", &[8192, 1, 4]),
            ("self_attn.f_a_proj.weight", "BF16", &[128, 4096]),
            ("self_attn.f_b_proj.weight", "BF16", &[8192, 128]),
            ("self_attn.g_a_proj.weight", "BF16", &[128, 4096]),
            ("self_attn.g_b_proj.weight", "BF16", &[8192, 128]),
            ("self_attn.b_proj.weight", "BF16", &[64, 4096]),
            ("self_attn.A_log", "F32", &[64]),
            ("self_attn.dt_bias", "F32", &[8192]),
            ("self_attn.o_norm.weight", "BF16", &[128]),
            ("self_attn.o_proj.weight", "BF16", &[4096, 8192]),
        ];
        assert_eq!(
            KDA_TENSORS.len(),
            want.len(),
            "the KDA block has exactly 15 tensors"
        );
        for (n, dt, sh) in want {
            let s = KDA_TENSORS.iter().find(|t| &t.name == n).expect(n);
            assert_eq!(s.dtype.name(), *dt, "{n} dtype");
            assert_eq!(s.expected_shape(&c), sh.to_vec(), "{n} shape");
        }
    }

    /// A DSA block must never be mistaken for a KDA one. Layer 45 (MTP) is DSA-shaped, so this
    /// is what stops the MTP layer being fed to a KDA binder.
    #[test]
    fn dsa_and_mtp_blocks_do_not_classify_as_kda() {
        let dsa: Vec<String> = [
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.indexer.wk.weight",
            "self_attn.indexer.wq_b.weight",
            "self_attn.o_proj.weight",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(classify_attn_block(&dsa), AttnBlockKind::Dsa);

        let kda: Vec<String> = KDA_TENSORS.iter().map(|t| t.name.to_string()).collect();
        assert_eq!(classify_attn_block(&kda), AttnBlockKind::Kda);

        // A KDA block missing one tensor is UNKNOWN, never silently Kda.
        assert_eq!(classify_attn_block(&kda[1..]), AttnBlockKind::Unknown);
    }

    #[test]
    fn chunk_width_is_bounded_by_the_shared_memory_ceiling() {
        let mut c = cfg();
        assert!(c.validate().is_ok(), "C=32 must fit");
        assert!(c.smem_scan() <= SMEM_CEILING);
        c.chunk = 64;
        assert!(
            c.validate().is_err(),
            "C=64 needs 81920 B and must be rejected, not truncated"
        );
    }

    use super::super::SMEM_CEILING;
}
