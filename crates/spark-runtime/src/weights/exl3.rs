// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 (QTIP trellis-quantized) weight loading: reconstruct packed trellis
//! codes to original-basis weights.
//!
//! Format (see `.research/EXL3_DECODE_FINDINGS.md` and the snapshotted
//! ExLlamaV3 sources it cites — decode math is MIT-licensed, (c) 2025
//! turboderp):
//!
//!  * `{p}.trellis` — int16 `[in/16, out/16, 16*K]`: 16x16 weight tiles,
//!    K bits/weight, 256 codes packed contiguously per tile. Consecutive
//!    codes OVERLAP: code `t`'s 16-bit decode window is the stream bits
//!    `[(t+1)*K-16, (t+1)*K)` (mod `256*K`), i.e. K fresh bits on top of
//!    the previous window — that overlap is the "trellis"; decode itself
//!    is stateless.
//!  * `{p}.suh` — f16 `[in]`,  `{p}.svh` — f16 `[out]`: Hadamard sign/scale
//!    vectors. Reconstruction emits the ORIGINAL-basis weight
//!    `W = diag(suh) . H128 . W_hat . H128 . diag(svh)` (1/sqrt(128) per
//!    side, per 128x128 tile).
//!  * `{p}.mul1` / checkpoint metadata — codebook selector `cb`:
//!    0 = "3inst", 1 = "mcg", 2 = "mul1". The Qwen3.8-Flash-Next-exl3
//!    checkpoints use mul1 (the shipped `.mul1` scalar is the flag).
//!  * There is NO stored codebook: a 16-bit code is scrambled by 2-3
//!    integer ops and reinterpreted as fp16 lanes (see `decode_3inst_2`).
//!
//! Two implementations, deliberately independent:
//!  * GPU: `kernels/gb10/common/exl3_reconstruct.cu` (port of upstream's
//!    `reconstruct_had_slice`, bit-identical by construction), launched by
//!    [`reconstruct_had_bf16`].
//!  * CPU: [`cpu_ref`] — written from the format spec, NOT transcribed from
//!    the kernel's thread/shuffle structure. GPU-vs-CPU bit equality (the
//!    `exl3_reconstruct_parity` example) is therefore a real cross-check of
//!    both, same pattern as the GGUF `dequant_cpu` oracle.
//!
//! Both dims must be multiples of 128; the published checkpoints only
//! quantize such tensors (everything else stays f16/bf16).

use anyhow::{Context, Result, bail, ensure};

use crate::gpu::{DevicePtr, GpuBackend};
use crate::weights::{WeightDtype, WeightStore};

/// Kernel module holding the reconstruct + transpose instances.
pub(crate) const MODULE: &str = "exl3_reconstruct";

/// True for EXL3 f16 auxiliary tensors whose EXACT f16 bits are decode
/// inputs and must NOT take the loaders' default F16->BF16 conversion:
/// the Hadamard sign vectors. (`.trellis` is I16 and `.mul1` is I32, so
/// they never hit the F16 path.)
pub fn is_exl3_f16_aux(name: &str) -> bool {
    name.ends_with(".suh") || name.ends_with(".svh")
}

/// True if `store` holds an EXL3-quantized linear at `prefix` (i.e.
/// `{prefix}.trellis` + `.suh` + `.svh` are all present).
pub fn is_exl3_linear(store: &WeightStore, prefix: &str) -> bool {
    store.contains(&format!("{prefix}.trellis"))
        && store.contains(&format!("{prefix}.suh"))
        && store.contains(&format!("{prefix}.svh"))
}

/// True if any tensor name marks this store as an EXL3 checkpoint.
pub fn store_has_exl3(store: &WeightStore) -> bool {
    store.names().any(|n| n.ends_with(".trellis"))
}

/// One EXL3-quantized linear resolved from a [`WeightStore`] (tensors
/// already resident on the GPU via the normal load path).
///
/// `in_dim`/`out_dim` come from the trellis shape `[in/16, out/16, 16*K]`;
/// the codebook comes from the `.mul1` flag scalar (which stores the
/// codebook's multiplier constant — see [`Exl3Codebook::from_flag_scalar`]).
///
/// `Copy` on purpose: native serving carries this struct inside the `Copy`
/// `QuantWeight` enum (spark-model), exactly like `QuantizedWeight` /
/// `Fp8Weight`. All fields are plain device pointers + geometry.
#[derive(Debug, Clone, Copy)]
pub struct Exl3Weight {
    pub trellis: DevicePtr,
    pub suh: DevicePtr,
    pub svh: DevicePtr,
    pub in_dim: usize,
    pub out_dim: usize,
    pub k_bits: u32,
    pub cb: Exl3Codebook,
}

impl Exl3Weight {
    /// Resolve `{prefix}.{trellis,suh,svh,mul1}` from the store, validating
    /// shapes and dtypes. The `.mul1` scalar is read back from the GPU (4
    /// bytes) to pick the codebook.
    pub fn from_store(gpu: &dyn GpuBackend, store: &WeightStore, prefix: &str) -> Result<Self> {
        Self::from_store_inner(gpu, store, prefix, None)
    }

    /// Like [`Self::from_store`] but with a caller-known codebook — skips the
    /// 4-byte `.mul1` D2H readback (which is synchronous and, over the 73,728
    /// expert tensors of a 512-expert model, measurably slows load). Use ONLY
    /// for tensors whose codebook was already validated: the EXL3 materialize
    /// pass reads every expert's flag for its per-(layer, projection)
    /// uniformity check, so the expert loader reads ONE flag per (layer,
    /// projection) and passes it here for the siblings.
    pub fn from_store_with_cb(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        prefix: &str,
        cb: Exl3Codebook,
    ) -> Result<Self> {
        Self::from_store_inner(gpu, store, prefix, Some(cb))
    }

    fn from_store_inner(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        prefix: &str,
        known_cb: Option<Exl3Codebook>,
    ) -> Result<Self> {
        let trellis = store
            .get(&format!("{prefix}.trellis"))
            .with_context(|| format!("EXL3 linear {prefix}: missing .trellis"))?;
        let suh = store.get(&format!("{prefix}.suh"))?;
        let svh = store.get(&format!("{prefix}.svh"))?;
        ensure!(
            trellis.dtype == WeightDtype::UInt16,
            "EXL3 {prefix}.trellis dtype {:?}, expected UInt16 (from I16)",
            trellis.dtype
        );
        ensure!(
            suh.dtype == WeightDtype::F16 && svh.dtype == WeightDtype::F16,
            "EXL3 {prefix} sign vectors must stay F16 in the store (got {:?}/{:?}) — \
             the loader's F16->BF16 conversion must exempt .suh/.svh",
            suh.dtype,
            svh.dtype
        );
        ensure!(
            trellis.shape.len() == 3,
            "EXL3 {prefix}.trellis shape {:?}, expected [in/16, out/16, 16*K]",
            trellis.shape
        );
        let in_dim = trellis.shape[0] * 16;
        let out_dim = trellis.shape[1] * 16;
        let k_bits = k_bits_from_trellis_dim(trellis.shape[2])?;
        ensure!(
            suh.num_elements() == in_dim && svh.num_elements() == out_dim,
            "EXL3 {prefix}: suh/svh sizes {}/{} do not match trellis dims [{in_dim}, {out_dim}]",
            suh.num_elements(),
            svh.num_elements()
        );

        // Codebook flag: a 4-byte scalar holding the codebook's multiplier.
        let cb = match known_cb {
            Some(cb) => cb,
            None => match store.get(&format!("{prefix}.mul1")) {
                Ok(flag) => {
                    let mut bytes = [0u8; 4];
                    gpu.copy_d2h(flag.ptr, &mut bytes)?;
                    Exl3Codebook::from_flag_scalar(u32::from_le_bytes(bytes))?
                }
                // Absent flag = upstream's unflagged default.
                Err(_) => Exl3Codebook::Inst3,
            },
        };

        Ok(Self {
            trellis: trellis.ptr,
            suh: suh.ptr,
            svh: svh.ptr,
            in_dim,
            out_dim,
            k_bits,
            cb,
        })
    }

    /// Resident bytes of the packed representation:
    /// trellis (`in*out*K/8`) + suh/svh f16 vectors + the 4-byte flag scalar.
    pub fn packed_bytes(&self) -> usize {
        self.in_dim * self.out_dim * self.k_bits as usize / 8 + (self.in_dim + self.out_dim) * 2 + 4
    }

    /// What the same linear would cost as a runtime ModelOpt-style NVFP4
    /// triplet (packed E2M1 `[n, k/2]` + FP8 per-16 scales `[n, k/16]` +
    /// f32 scalar) — the materialize pass's expert fallback format. Used for
    /// the keep-native memory-savings log.
    pub fn nvfp4_equiv_bytes(&self) -> usize {
        self.in_dim * self.out_dim / 2 + self.in_dim * self.out_dim / 16 + 4
    }

    /// Materialize as Atlas-layout dense BF16 `[out, in]` (fresh buffer,
    /// caller owns). This is the GGUF-style "loading support" path: the
    /// result feeds the existing BF16 GEMMs or the runtime NVFP4/FP8
    /// requantizers, exactly like a BF16-checkpoint tensor.
    pub fn to_bf16(&self, gpu: &dyn GpuBackend) -> Result<DevicePtr> {
        reconstruct_had_bf16(
            gpu,
            self.trellis,
            self.suh,
            self.svh,
            self.in_dim,
            self.out_dim,
            self.k_bits,
            self.cb,
        )
    }
}

/// Codebook selector, matching upstream's `mcg`/`mul1` flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exl3Codebook {
    /// mul + add + lop3 ("3inst"), upstream default when no flag is set.
    Inst3 = 0,
    /// pure multiplicative congruential + lop3.
    Mcg = 1,
    /// mul + dp4a byte-sum + affine — what Qwen3.8-Flash-Next-exl3 ships.
    Mul1 = 2,
}

impl Exl3Codebook {
    /// The checkpoint self-describes each tensor's codebook by storing the
    /// codebook's own multiplier constant in the `.mul1` / `.mcg` scalar
    /// (verified against turboderp/Qwen3.8-Flash-Next-exl3: its `.mul1`
    /// scalars hold 0x83DCD12D). 0 means the flag is unset.
    pub fn from_flag_scalar(v: u32) -> Result<Self> {
        match v {
            0x83DC_D12D => Ok(Self::Mul1),
            0xCBAC_1FED => Ok(Self::Mcg),
            0 => Ok(Self::Inst3),
            other => bail!("unrecognized EXL3 codebook flag constant {other:#x}"),
        }
    }
}

/// K (bits/weight) of a trellis tensor from its innermost dim (`16*K`).
pub fn k_bits_from_trellis_dim(inner: usize) -> Result<u32> {
    ensure!(
        inner.is_multiple_of(16) && (1..=8).contains(&(inner / 16)),
        "EXL3 trellis inner dim {inner} is not 16*K for K in 1..=8"
    );
    Ok((inner / 16) as u32)
}

// Reconstruct launchers (allocating `reconstruct_had_*` forms and the
// stream-ordered `_into` forms the native dense prefill tier reuses) —
// sibling file, 500-LoC cap.
mod reconstruct_launch;
pub use reconstruct_launch::{
    reconstruct_had_bf16, reconstruct_had_f16_device, reconstruct_had_f16_into,
    transpose_f16_to_bf16_into,
};

#[cfg(test)]
mod store_tests {
    use std::collections::HashMap;

    use super::*;
    use crate::gpu::mock::MockGpuBackend;
    use crate::weights::WeightTensor;

    fn tensor(gpu: &MockGpuBackend, shape: Vec<usize>, dtype: WeightDtype) -> WeightTensor {
        let bytes: usize = shape.iter().product::<usize>() * dtype.byte_size().max(1);
        WeightTensor {
            ptr: gpu.alloc(bytes.max(4)).unwrap(),
            shape,
            dtype,
        }
    }

    fn exl3_store(gpu: &MockGpuBackend) -> WeightStore {
        let mut m = HashMap::new();
        // A real expert shape: [2560 -> 640] at K=4.
        m.insert(
            "l.gate_proj.trellis".to_string(),
            tensor(gpu, vec![160, 40, 64], WeightDtype::UInt16),
        );
        m.insert(
            "l.gate_proj.suh".to_string(),
            tensor(gpu, vec![2560], WeightDtype::F16),
        );
        m.insert(
            "l.gate_proj.svh".to_string(),
            tensor(gpu, vec![640], WeightDtype::F16),
        );
        m.insert(
            "l.gate_proj.mul1".to_string(),
            tensor(gpu, vec![], WeightDtype::Int32),
        );
        m.insert(
            "l.norm.weight".to_string(),
            tensor(gpu, vec![2560], WeightDtype::BF16),
        );
        WeightStore::from_map(m)
    }

    #[test]
    fn f16_aux_names() {
        assert!(is_exl3_f16_aux(
            "model.layers.0.mlp.experts.0.gate_proj.suh"
        ));
        assert!(is_exl3_f16_aux("a.svh"));
        assert!(!is_exl3_f16_aux("a.weight"));
        assert!(!is_exl3_f16_aux("a.suh.weight"));
    }

    // EP expert filtering parses the expert index from the name's `experts`
    // SEGMENT, so EXL3 suffixes (.trellis/.suh/.svh/.mul1) must filter
    // exactly like .weight — a rank must never load remote experts' trellis.
    #[test]
    fn ep_expert_index_parses_exl3_names() {
        use crate::weights::parse_expert_index;
        for sfx in ["trellis", "suh", "svh", "mul1", "weight"] {
            assert_eq!(
                parse_expert_index(&format!("model.layers.3.mlp.experts.42.gate_proj.{sfx}")),
                Some(42),
                ".{sfx} name must parse the expert index"
            );
        }
        assert_eq!(
            parse_expert_index("model.layers.3.mlp.shared_expert.up_proj.trellis"),
            None
        );
    }

    #[test]
    fn detection() {
        let gpu = MockGpuBackend::new();
        let store = exl3_store(&gpu);
        assert!(store_has_exl3(&store));
        assert!(is_exl3_linear(&store, "l.gate_proj"));
        assert!(!is_exl3_linear(&store, "l.norm"));
        assert!(!store_has_exl3(&WeightStore::empty()));
    }

    #[test]
    fn from_store_resolves_geometry() {
        let gpu = MockGpuBackend::new();
        let store = exl3_store(&gpu);
        let w = Exl3Weight::from_store(&gpu, &store, "l.gate_proj").unwrap();
        assert_eq!(w.in_dim, 2560);
        assert_eq!(w.out_dim, 640);
        assert_eq!(w.k_bits, 4);
        // Mock d2h reads back zeros -> flag 0 -> the unflagged default.
        assert_eq!(w.cb, Exl3Codebook::Inst3);
    }

    #[test]
    fn codebook_flag_constants() {
        assert_eq!(
            Exl3Codebook::from_flag_scalar(0x83DC_D12D).unwrap(),
            Exl3Codebook::Mul1
        );
        assert_eq!(
            Exl3Codebook::from_flag_scalar(0xCBAC_1FED).unwrap(),
            Exl3Codebook::Mcg
        );
        assert_eq!(
            Exl3Codebook::from_flag_scalar(0).unwrap(),
            Exl3Codebook::Inst3
        );
        assert!(Exl3Codebook::from_flag_scalar(1234).is_err());
    }
}

/// CPU reference implementation. Bit-exact to the GPU kernel — every fp16
/// operation replicates the kernel's op order and per-op rounding.
pub mod cpu_ref;
