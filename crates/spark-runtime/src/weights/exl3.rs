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
use crate::kernel_args::KernelLaunch;
use crate::weights::{WeightDtype, WeightStore};

const MODULE: &str = "exl3_reconstruct";

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
        let cb = match store.get(&format!("{prefix}.mul1")) {
            Ok(flag) => {
                let mut bytes = [0u8; 4];
                gpu.copy_d2h(flag.ptr, &mut bytes)?;
                Exl3Codebook::from_flag_scalar(u32::from_le_bytes(bytes))?
            }
            // Absent flag = upstream's unflagged default.
            Err(_) => Exl3Codebook::Inst3,
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

/// Reconstruct an EXL3 tensor to the upstream-native f16 `[in, out]`
/// row-major layout on the GPU (the reconstruct kernel's coalesced store
/// order). Returns a fresh f16 buffer of `in * out` elements (caller owns).
///
/// * `trellis` — device ptr to the packed `.trellis` int16 data
///   (`(in/16) * (out/16) * 16 * k_bits` u16s, uploaded by the caller).
/// * `suh` / `svh` — device ptrs to the f16 sign vectors (`in` / `out` f16s).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_had_f16_device(
    gpu: &dyn GpuBackend,
    trellis: DevicePtr,
    suh: DevicePtr,
    svh: DevicePtr,
    in_dim: usize,
    out_dim: usize,
    k_bits: u32,
    cb: Exl3Codebook,
) -> Result<DevicePtr> {
    ensure!(
        in_dim.is_multiple_of(128) && out_dim.is_multiple_of(128),
        "EXL3 reconstruct needs both dims divisible by 128, got [{in_dim}, {out_dim}]"
    );
    ensure!((1..=8).contains(&k_bits), "EXL3 K must be 1..=8, got {k_bits}");

    let name = format!("exl3_reconstruct_had_k{}_cb{}", k_bits, cb as u32);
    let kernel = match gpu.kernel(MODULE, &name) {
        Ok(k) => k,
        Err(e) => bail!("EXL3 kernel {name} unavailable on this target: {e}"),
    };
    let stream = gpu.default_stream();
    let f16_out = gpu.alloc(in_dim * out_dim * 2)?;
    let launch = KernelLaunch::new(gpu, kernel)
        .grid([(out_dim / 128) as u32, (in_dim / 128) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(f16_out)
        .arg_ptr(trellis)
        .arg_ptr(suh)
        .arg_ptr(svh)
        .arg_u32((out_dim / 16) as u32) // packed_blocks_n
        .arg_u32(0) // packed_n_offset
        .launch(stream);
    if let Err(e) = launch {
        gpu.free(f16_out).ok();
        return Err(e);
    }
    Ok(f16_out)
}

/// Reconstruct an EXL3 tensor to Atlas-layout BF16 `[out, in]` on the GPU.
/// Returns a fresh BF16 buffer of `out * in` elements (caller owns it).
///
/// Reconstructs to the f16 `[in, out]` layout first, then transposes to
/// Atlas's `[out, in]` row-major with a single f32-exact f16->bf16 rounding.
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_had_bf16(
    gpu: &dyn GpuBackend,
    trellis: DevicePtr,
    suh: DevicePtr,
    svh: DevicePtr,
    in_dim: usize,
    out_dim: usize,
    k_bits: u32,
    cb: Exl3Codebook,
) -> Result<DevicePtr> {
    let f16_tmp =
        reconstruct_had_f16_device(gpu, trellis, suh, svh, in_dim, out_dim, k_bits, cb)?;
    let transpose = match gpu.kernel(MODULE, "exl3_f16_to_bf16_t") {
        Ok(k) => k,
        Err(e) => {
            gpu.free(f16_tmp).ok();
            return Err(e);
        }
    };
    let stream = gpu.default_stream();
    let out = match gpu.alloc(out_dim * in_dim * 2) {
        Ok(p) => p,
        Err(e) => {
            gpu.free(f16_tmp).ok();
            return Err(e);
        }
    };
    let launch = KernelLaunch::new(gpu, transpose)
        .grid([(out_dim.div_ceil(32)) as u32, (in_dim.div_ceil(32)) as u32, 1])
        .block([32, 8, 1])
        .arg_ptr(f16_tmp)
        .arg_ptr(out)
        .arg_u32(in_dim as u32)
        .arg_u32(out_dim as u32)
        .launch(stream);
    gpu.synchronize(stream).ok();
    gpu.free(f16_tmp).ok();
    if let Err(e) = launch {
        gpu.free(out).ok();
        return Err(e);
    }
    Ok(out)
}

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
        assert!(is_exl3_f16_aux("model.layers.0.mlp.experts.0.gate_proj.suh"));
        assert!(is_exl3_f16_aux("a.svh"));
        assert!(!is_exl3_f16_aux("a.weight"));
        assert!(!is_exl3_f16_aux("a.suh.weight"));
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
        assert_eq!(Exl3Codebook::from_flag_scalar(0).unwrap(), Exl3Codebook::Inst3);
        assert!(Exl3Codebook::from_flag_scalar(1234).is_err());
    }
}

/// CPU reference implementation. Bit-exact to the GPU kernel — every fp16
/// operation replicates the kernel's op order and per-op rounding.
pub mod cpu_ref {
    use half::f16;

    use super::Exl3Codebook;

    const R_SCALE: f32 = 0.088_388_35_f32; // 0.08838834764831845f in the kernel

    fn f16b(bits: u16) -> f16 {
        f16::from_bits(bits)
    }

    /// f16 add with CUDA `__hadd` semantics (exact in f32, one rounding).
    fn hadd(a: f16, b: f16) -> f16 {
        f16::from_f32(a.to_f32() + b.to_f32())
    }
    fn hsub(a: f16, b: f16) -> f16 {
        f16::from_f32(a.to_f32() - b.to_f32())
    }
    fn hmul(a: f16, b: f16) -> f16 {
        f16::from_f32(a.to_f32() * b.to_f32())
    }
    /// f16 fused multiply-add with CUDA `__hfma` semantics: a*b+c computed
    /// exactly, ONE rounding. f64 holds the exact product+sum of f16 inputs.
    fn hfma(a: f16, b: f16, c: f16) -> f16 {
        f16::from_f64(a.to_f64() * b.to_f64() + c.to_f64())
    }

    /// Decode one 16-bit code window through codebook `cb`.
    /// (`decode_3inst` in upstream codebook.cuh; the lop3 imm 0x6a with
    /// those masks is `(x & 0x8fff8fff) ^ 0x3b603b60`.)
    pub fn decode_code(w: u16, cb: Exl3Codebook) -> f16 {
        let w = w as u32;
        match cb {
            Exl3Codebook::Inst3 => {
                let x = w
                    .wrapping_mul(89226354)
                    .wrapping_add(64248484);
                let x = (x & 0x8fff8fff) ^ 0x3b603b60;
                hadd(f16b(x as u16), f16b((x >> 16) as u16))
            }
            Exl3Codebook::Mcg => {
                let x = w.wrapping_mul(0xCBAC1FED);
                let x = (x & 0x8fff8fff) ^ 0x3b603b60;
                hadd(f16b(x as u16), f16b((x >> 16) as u16))
            }
            Exl3Codebook::Mul1 => {
                let x = w.wrapping_mul(0x83DCD12D);
                // __dp4a(x, 0x01010101, 0x6400): sum of the 4 bytes + bias.
                let sum: u32 = 0x6400
                    + (x & 0xff)
                    + ((x >> 8) & 0xff)
                    + ((x >> 16) & 0xff)
                    + ((x >> 24) & 0xff);
                let k_inv = f16b(0x1eee); //  0.00677 = 1/147.7
                let k_bias = f16b(0xc931); // -10.39
                hfma(f16b(sum as u16), k_inv, k_bias)
            }
        }
    }

    /// Decode one 16x16 tile (256 codes at `16*k` packed u16 words) into
    /// `tile[row][col]`.
    ///
    /// Bitstream: the u16 words pair into little-endian u32s
    /// (`u32[i] = (u16[2i+1] << 16) | u16[2i]`), and stream bit `x` is bit
    /// `31 - x%32` of u32 `x/32` — MSB-first WITHIN each u32, ascending
    /// u32 order. (Derived from the kernel's funnel-shift indexing:
    /// `s0 = (i1+1)*32 - b1` aligns the window END to the u32's LOW bits,
    /// which puts earlier stream bits at higher bit positions.)
    ///
    /// Code `t`'s decode window is stream bits `[(t+1)*k - 16, (t+1)*k)`
    /// mod `256*k` — K fresh bits below the previous window (the trellis
    /// overlap); the window value is read MSB-first: bit 15 of `w` is the
    /// OLDEST stream bit in the window.
    ///
    /// Position mapping `t -> (row, col)` follows the m16n8k16 B-fragment
    /// layout the packer wrote (verified against the GPU kernel by the
    /// parity example): with `l = t/8`, `j = t%8`:
    ///   row = (l%4)*2 + (j&1) + ((j>>1)&1)*8
    ///   col = ((l & !4)/8)*2 + ((l>>2)&1) + ((j>>2)&1)*8
    pub fn decode_tile(packed: &[u16], k: u32, cb: Exl3Codebook) -> [[f16; 16]; 16] {
        let k = k as usize;
        assert_eq!(packed.len(), 16 * k);
        let total_bits = 256 * k;
        let stream_bit = |idx: usize| -> u16 {
            let idx = idx % total_bits;
            let w32 = idx / 32;
            let bit = 31 - (idx % 32); // MSB-first within the u32
            let word = if bit >= 16 {
                packed[w32 * 2 + 1] // u32 high half = second u16
            } else {
                packed[w32 * 2]
            };
            (word >> (bit % 16)) & 1
        };
        let mut tile = [[f16::from_f32(0.0); 16]; 16];
        for t in 0..256 {
            let end = (t + 1) * k + total_bits; // + total_bits avoids underflow
            let mut w: u16 = 0;
            for b in 0..16 {
                // w bit 15 = oldest stream bit of the window
                w |= stream_bit(end - 16 + b) << (15 - b);
            }
            let l = t / 8;
            let j = t % 8;
            let row = (l % 4) * 2 + (j & 1) + ((j >> 1) & 1) * 8;
            let col = ((l & !4) / 8) * 2 + ((l >> 2) & 1) + ((j >> 2) & 1) * 8;
            tile[row][col] = decode_code(w, cb);
        }
        tile
    }

    /// One in-place FWHT butterfly stage in f16 over stride `s`, matching
    /// `shuffle_had_h2x32`: index with the bit clear gets `self + partner`,
    /// index with the bit set gets `partner - self`.
    fn fwht_stage_f16(v: &mut [f16; 128], group: usize, s: usize) {
        // `group` values per index share the transform (the 4-wide chunks);
        // s indexes in units of groups.
        let mut out = *v;
        for a in 0..(128 / group) {
            let p = a ^ s;
            for g in 0..group {
                let own = v[a * group + g];
                let partner = v[p * group + g];
                out[a * group + g] = if a & s == 0 {
                    hadd(own, partner)
                } else {
                    hsub(partner, own)
                };
            }
        }
        *v = out;
    }

    /// Full 128x128-block reconstruction with the both-side Hadamard, exactly
    /// replicating the GPU kernel's op order:
    ///  1. decode tiles -> W_hat
    ///  2. per column: 4-point butterfly over row groups (f16, then *rs),
    ///     then 5 FWHT stages over the 32 groups (f16)
    ///  3. per row: 4-point butterfly over col groups in f32 (exact), one
    ///     rounding to f16, *rs (f16), then 5 FWHT stages over groups (f16),
    ///     then *suh[row] then *svh[col] (two f16 muls)
    #[allow(clippy::needless_range_loop)]
    pub fn reconstruct_had_block(
        trellis_block: impl Fn(usize, usize) -> Vec<u16>, // (tile_r, tile_c) -> 16*k words
        suh: &[u16],  // 128 f16 bits for this block's rows
        svh: &[u16],  // 128 f16 bits for this block's cols
        k: u32,
        cb: Exl3Codebook,
    ) -> Vec<u16> {
        let rs = f16::from_f32(R_SCALE);
        // 1. decode
        let mut w = vec![[f16::from_f32(0.0); 128]; 128];
        for tr in 0..8 {
            for tc in 0..8 {
                let words = trellis_block(tr, tc);
                let tile = decode_tile(&words, k, cb);
                for r in 0..16 {
                    for c in 0..16 {
                        w[tr * 16 + r][tc * 16 + c] = tile[r][c];
                    }
                }
            }
        }
        // 2. column-direction (H . W): FWHT down each column over rows
        for c in 0..128 {
            let mut col = [f16::from_f32(0.0); 128];
            for r in 0..128 {
                col[r] = w[r][c];
            }
            // 4-point butterfly within each row group of 4, then *rs
            for a in 0..32 {
                let v0 = col[a * 4];
                let v1 = col[a * 4 + 1];
                let v2 = col[a * 4 + 2];
                let v3 = col[a * 4 + 3];
                let s0 = hadd(v0, v1);
                let d0 = hsub(v0, v1);
                let s1 = hadd(v2, v3);
                let d1 = hsub(v2, v3);
                col[a * 4] = hmul(hadd(s0, s1), rs);
                col[a * 4 + 1] = hmul(hadd(d0, d1), rs);
                col[a * 4 + 2] = hmul(hsub(s0, s1), rs);
                col[a * 4 + 3] = hmul(hsub(d0, d1), rs);
            }
            for s in [1usize, 2, 4, 8, 16] {
                fwht_stage_f16(&mut col, 4, s);
            }
            for r in 0..128 {
                w[r][c] = col[r];
            }
        }
        // 3. row-direction (W . H) + scales, fused as in the kernel store
        let mut out = vec![0u16; 128 * 128];
        for r in 0..128 {
            let mut row = [f16::from_f32(0.0); 128];
            // 4-point butterfly in f32 (exact), one rounding, then *rs in f16
            for g in 0..32 {
                let v0 = w[r][g * 4].to_f32();
                let v1 = w[r][g * 4 + 1].to_f32();
                let v2 = w[r][g * 4 + 2].to_f32();
                let v3 = w[r][g * 4 + 3].to_f32();
                let s0 = v0 + v1;
                let d0 = v0 - v1;
                let s1 = v2 + v3;
                let d1 = v2 - v3;
                row[g * 4] = hmul(f16::from_f32(s0 + s1), rs);
                row[g * 4 + 1] = hmul(f16::from_f32(d0 + d1), rs);
                row[g * 4 + 2] = hmul(f16::from_f32(s0 - s1), rs);
                row[g * 4 + 3] = hmul(f16::from_f32(d0 - d1), rs);
            }
            for s in [1usize, 2, 4, 8, 16] {
                fwht_stage_f16(&mut row, 4, s);
            }
            let su = f16b(suh[r]);
            for c in 0..128 {
                let v = hmul(hmul(row[c], su), f16b(svh[c]));
                out[r * 128 + c] = v.to_bits();
            }
        }
        out
    }

    /// Whole-tensor CPU reconstruction: trellis `[in/16, out/16, 16*k]` u16
    /// words -> f16 bits `[in, out]` row-major (the GPU kernel's pre-transpose
    /// layout, which is what the parity example compares against).
    #[allow(clippy::too_many_arguments)]
    pub fn reconstruct_had_f16(
        trellis: &[u16],
        suh: &[u16],
        svh: &[u16],
        in_dim: usize,
        out_dim: usize,
        k: u32,
        cb: Exl3Codebook,
    ) -> Vec<u16> {
        assert_eq!(in_dim % 128, 0);
        assert_eq!(out_dim % 128, 0);
        let kt = 16 * k as usize;
        assert_eq!(trellis.len(), (in_dim / 16) * (out_dim / 16) * kt);
        assert_eq!(suh.len(), in_dim);
        assert_eq!(svh.len(), out_dim);
        let tiles_n = out_dim / 16;
        let mut out = vec![0u16; in_dim * out_dim];
        for kb in 0..in_dim / 128 {
            for nb in 0..out_dim / 128 {
                let block = reconstruct_had_block(
                    |tr, tc| {
                        let tile_r = kb * 8 + tr;
                        let tile_c = nb * 8 + tc;
                        let base = (tile_r * tiles_n + tile_c) * kt;
                        trellis[base..base + kt].to_vec()
                    },
                    &suh[kb * 128..kb * 128 + 128],
                    &svh[nb * 128..nb * 128 + 128],
                    k,
                    cb,
                );
                for r in 0..128 {
                    let dst = (kb * 128 + r) * out_dim + nb * 128;
                    out[dst..dst + 128].copy_from_slice(&block[r * 128..r * 128 + 128]);
                }
            }
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // decode_code spot values, computed independently from the format
        // spec (u32 arithmetic + f16 reinterpretation done by hand):
        //   mul1(0): x = 0, dp4a sum = 0x6400 -> f16 1024.0;
        //            1024 * 0.006775 - 10.3906 = -3.4531 (one fma rounding)
        #[test]
        fn mul1_code_zero() {
            let v = decode_code(0, Exl3Codebook::Mul1);
            let expect = f16::from_f64(
                f16::from_bits(0x6400).to_f64() * f16::from_bits(0x1eee).to_f64()
                    + f16::from_bits(0xc931).to_f64(),
            );
            assert_eq!(v.to_bits(), expect.to_bits());
        }

        // The mcg decode of code 0 is exactly 0x3b60-as-f16 + 0x3b60-as-f16
        // (x=0 -> masked 0 -> xor pattern in both halves).
        #[test]
        fn mcg_code_zero() {
            let v = decode_code(0, Exl3Codebook::Mcg);
            let half = f16::from_bits(0x3b60);
            let expect = f16::from_f32(half.to_f32() + half.to_f32());
            assert_eq!(v.to_bits(), expect.to_bits());
        }

        // A trellis of all-zero words must decode every position to the same
        // value (every 16-bit window is 0), so after the symmetric Hadamard
        // sandwich with unit scales the block is rank-1-ish but crucially
        // FINITE everywhere. Sanity: no NaN/Inf anywhere at any K.
        #[test]
        fn zero_trellis_finite() {
            for k in 1..=8u32 {
                for cb in [Exl3Codebook::Inst3, Exl3Codebook::Mcg, Exl3Codebook::Mul1] {
                    let one = f16::from_f32(1.0).to_bits();
                    let out = reconstruct_had_f16(
                        &vec![0u16; (128 / 16) * (128 / 16) * 16 * k as usize],
                        &vec![one; 128],
                        &vec![one; 128],
                        128,
                        128,
                        k,
                        cb,
                    );
                    for &bits in &out {
                        let v = f16::from_bits(bits).to_f32();
                        assert!(v.is_finite(), "K={k} cb={cb:?} produced {v}");
                    }
                }
            }
        }

        // Bit-window extraction must match the kernel's aligned-K=4 fast
        // path (`dq8_aligned_4bits`), whose first-lane extraction reduces
        // to: window(t=7) = u32word0 & 0xffff = u16[0], and
        // window(t=3) = u32word0 >> 16 = u16[1] (derived by hand from the
        // funnel-shift indexing; the GPU parity example is the ground
        // truth).
        #[test]
        fn window_alignment_k4() {
            let k = 4usize;
            let mut words = vec![0u16; 16 * k];
            words[0] = 0xABCD; // u32 word0 low half
            words[1] = 0x1234; // u32 word0 high half
            let total = 256 * k;
            let stream_bit = |idx: usize| -> u16 {
                let idx = idx % total;
                let w32 = idx / 32;
                let bit = 31 - (idx % 32);
                let word = if bit >= 16 { words[w32 * 2 + 1] } else { words[w32 * 2] };
                (word >> (bit % 16)) & 1
            };
            let get = |t: usize| {
                let end = (t + 1) * k + total;
                let mut w = 0u16;
                for b in 0..16 {
                    w |= stream_bit(end - 16 + b) << (15 - b);
                }
                w
            };
            assert_eq!(get(3), 0x1234);
            assert_eq!(get(7), 0xABCD);
        }
    }
}
