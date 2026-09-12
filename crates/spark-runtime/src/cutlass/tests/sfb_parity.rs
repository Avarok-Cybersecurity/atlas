// SPDX-License-Identifier: AGPL-3.0-only

//! Numeric parity for the SFB weight-scale swizzle (`pack_weight_sfb`).
//!
//! ## Why this exists
//!
//! `pack_weight_sfb` has two source-layout modes and its doc comment claims
//! "Output layout is identical either way":
//!
//! * `src_n_major = false` — Atlas-transposed `[K/16, N]`, indexed
//!   `scales[group * n + col]`
//! * `src_n_major = true`  — checkpoint-native `[N, K/16]`, indexed
//!   `scales[col * (k/16) + group]`
//!
//! That claim is a PROPERTY, not a fixture: for any scale matrix `A` of shape
//! `[N, K/16]`, packing `A` n-major must byte-match packing `Aᵀ` k-major.
//! Both are supposed to be the same swizzle over the same logical values, so
//! no golden data is needed to test it.
//!
//! It matters because **qwen4_exp is the only model that takes the n-major
//! fallback** — Holo and qwen35 build the transposed `gate_ptrs_t` and take
//! the other branch — and enabling CUTLASS grouped MoE on qwen4_exp measured
//! +30% prefill but destabilised generation (not bit-exact; a controlled
//! agentic A/B went from 6/6 passing to not completing, with the model
//! hallucinating that its own output was being corrupted). A rarely-exercised
//! swizzle branch is the leading suspect, and before this file the branch had
//! no test coverage at all.
//!
//! ## What a failure here means, and what it does not
//!
//! These tests validate the packer's INTERNAL consistency: that the two source
//! indexings address the same logical element. They deliberately build a DENSE
//! `[N, K/16]` source, so they do **not** prove that qwen4_exp's real
//! checkpoint scales are dense with row stride exactly `k/16`. A padded or
//! strided scale tensor would make the n-major read wrong in production while
//! these tests still pass. If everything here is green, that stride assumption
//! against a real checkpoint is the next thing to check — the packer takes a
//! bare pointer and is told only `n` and `k`, so it cannot detect the
//! difference itself.
//!
//! GPU tests: `#[ignore]` per repo convention.
//! ```text
//! cargo test -p spark-runtime --release sfb_ -- --ignored --nocapture
//! ```

use super::*;

/// Bytes in the swizzled SFB atom for a given `(n, k)`.
///
/// Mirrors `MoeLayer::build_cutlass_grouped_sfb`'s `sfb_len`: the atom is
/// padded to 128 in N and to 4 in the K/16 scale-group axis. Kept as a
/// separate copy on purpose — if the two ever disagree, the allocation and
/// the kernel's writes disagree, and this test should be the thing that says
/// so rather than a silent heap overwrite in production.
fn sfb_len(n: usize, k: usize) -> usize {
    n.div_ceil(128) * 128 * (k / 16).div_ceil(4) * 4
}

/// A deterministic, non-symmetric E4M3 byte for logical position `(row, col)`.
///
/// Non-symmetric in the two indices on purpose: a packer that transposed its
/// source by mistake would still produce matching output for any pattern that
/// is symmetric under `(row, col) -> (col, row)`, so such a pattern could not
/// detect the very bug this file exists to find.
///
/// Restricted to `0x00..=0x7E`: `0x7F` is NaN in E4M3, and the packer's output
/// type is `float_ue4m3_t` — UNSIGNED — so negative inputs are outside the
/// meaningful domain for a weight scale and would only test clamping.
fn scale_byte(row: usize, col: usize, salt: u64) -> u8 {
    let h = (row as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((col as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(salt);
    ((h >> 24) % 0x7F) as u8
}

/// Device buffer that frees itself, so an assertion failure cannot leak it.
struct DevBuf(*mut c_void);

impl DevBuf {
    fn alloc(bytes: usize) -> Self {
        let mut p: *mut c_void = std::ptr::null_mut();
        cuda_check(unsafe { cudaMalloc(&mut p, bytes) }, "cudaMalloc");
        Self(p)
    }

    fn upload(src: &[u8]) -> Self {
        let b = Self::alloc(src.len());
        cuda_check(
            unsafe {
                cudaMemcpy(
                    b.0,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "cudaMemcpy H2D",
        );
        b
    }

    fn zeroed(bytes: usize) -> Self {
        Self::upload(&vec![0u8; bytes])
    }

    fn download(&self, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        cuda_check(
            unsafe {
                cudaMemcpy(
                    out.as_mut_ptr() as *mut c_void,
                    self.0,
                    bytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "cudaMemcpy D2H",
        );
        out
    }

    fn addr(&self) -> u64 {
        self.0 as u64
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe { cudaFree(self.0) };
    }
}

/// Pack one `[N, K/16]` scale matrix through BOTH source layouts and return
/// the two SFB atoms.
///
/// `a_n_major` is row-major `[N, K/16]`. The k-major arm transposes it on the
/// HOST into `[K/16, N]` — the same logical matrix, the layout the Atlas
/// transposed path would have produced — and packs that with
/// `src_n_major = false`.
///
/// The output buffers are zero-filled first: the SFB atom is padded (N to 128,
/// K/16 to 4) and the kernel only writes the live region, so comparing raw
/// buffers would otherwise compare uninitialised padding and fail for reasons
/// that have nothing to do with the swizzle.
fn pack_both_ways(n: usize, k: usize, salt: u64) -> (Vec<u8>, Vec<u8>) {
    let groups = k / 16;
    let len = sfb_len(n, k);

    let mut a_n_major = vec![0u8; n * groups];
    for row in 0..n {
        for g in 0..groups {
            a_n_major[row * groups + g] = scale_byte(row, g, salt);
        }
    }
    // [N, K/16] -> [K/16, N]
    let mut a_k_major = vec![0u8; groups * n];
    for row in 0..n {
        for g in 0..groups {
            a_k_major[g * n + row] = a_n_major[row * groups + g];
        }
    }

    let src_n = DevBuf::upload(&a_n_major);
    let src_k = DevBuf::upload(&a_k_major);
    let out_n = DevBuf::zeroed(len);
    let out_k = DevBuf::zeroed(len);

    pack_weight_sfb(src_n.addr(), out_n.addr(), n as u32, k as u32, true, 0)
        .expect("pack_weight_sfb (n-major)");
    pack_weight_sfb(src_k.addr(), out_k.addr(), n as u32, k as u32, false, 0)
        .expect("pack_weight_sfb (k-major)");
    cuda_check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize");

    (out_n.download(len), out_k.download(len))
}

/// Compare two SFB atoms and panic with a diagnosis rather than just a count.
///
/// The mismatch COUNT distinguishes the likely causes on its own: a handful of
/// differing bytes points at an edge/padding case, while "almost every byte"
/// means the two arms addressed different elements throughout — a transposed
/// or wrongly-strided read.
fn assert_sfb_eq(n: usize, k: usize, got_n: &[u8], got_k: &[u8]) {
    assert_eq!(got_n.len(), got_k.len(), "SFB length disagreement");
    let diffs: Vec<usize> = got_n
        .iter()
        .zip(got_k.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    if diffs.is_empty() {
        return;
    }
    let nonzero = got_n.iter().filter(|b| **b != 0).count();
    let show: Vec<String> = diffs
        .iter()
        .take(8)
        .map(|&i| format!("[{i}] n-major={:#04x} k-major={:#04x}", got_n[i], got_k[i]))
        .collect();
    panic!(
        "SFB parity FAILED for n={n} k={k}: {} of {} bytes differ \
         ({} bytes are live in the n-major atom).\n  {}\n\
         The two source layouts are supposed to be the same swizzle over the \
         same values, so any difference means one of the two indexings in \
         pack_weight_sfb_group addresses the wrong element. qwen4_exp is the \
         only model on the n-major path.",
        diffs.len(),
        got_n.len(),
        nonzero,
        show.join("\n  ")
    );
}

/// qwen4_exp gate/up projection: `(n, k) = (moe_intermediate, hidden)`.
///
/// Read from the shipping checkpoint's `config.json`
/// (Qwen3.8-Flash-Next-NVFP4): `hidden_size = 2560`,
/// `moe_intermediate_size = 640`, 512 experts over 48 layers. These are the
/// exact dims the grouped MoE runs, and the ones the CUTLASS enablement
/// measured +30% prefill on before generation destabilised.
///
/// The shared expert has `shared_expert_intermediate_size = 640` — the same
/// shape — so it needs no separate case.
#[test]
#[ignore = "requires GPU"]
fn sfb_layout_parity_qwen4exp_gate_up() {
    let (n, k) = (640, 2560);
    let (a, b) = pack_both_ways(n, k, 0xA1);
    assert_sfb_eq(n, k, &a, &b);
}

/// qwen4_exp down projection: dims swap to `(hidden, moe_intermediate)`.
///
/// Its own case rather than a loop entry because `down` is where the two axes
/// trade places — N becomes 2560 and K/16 becomes 40 — so a bug that depends
/// on their relative magnitude shows here and not in gate/up.
#[test]
#[ignore = "requires GPU"]
fn sfb_layout_parity_qwen4exp_down() {
    let (n, k) = (2560, 640);
    let (a, b) = pack_both_ways(n, k, 0xD0);
    assert_sfb_eq(n, k, &a, &b);
}

/// The other production model whose loader builds SFB atoms:
/// holo-3.1-35b-a3b (`hidden_size = 2048`, `moe_intermediate_size = 512`).
///
/// Holo reaches the packer through the TRANSPOSED path, not the n-major
/// fallback, so parity here is the control: if qwen4_exp's shapes were to fail
/// while these pass, the fault is specific to the shapes rather than to the
/// n-major indexing, and vice versa.
///
/// ★ Every shape in this file is fully aligned in both axes, because every
/// real one is: N is a multiple of 128 (640, 2560, 512, 2048) and K/16 is a
/// multiple of 4 (160, 40, 128, 32). The SFB atom's padding path — N rounded
/// to 128, K/16 rounded to 4 — is therefore NOT exercised by any shipping
/// model, and consequently not by these tests. That is a deliberate
/// consequence of testing real shapes only; a synthetic unaligned case would
/// cover it, at the cost of asserting behaviour no model depends on.
#[test]
#[ignore = "requires GPU"]
fn sfb_layout_parity_holo35b() {
    for &(n, k) in &[
        (512, 2048), // gate/up: (moe_intermediate, hidden)
        (2048, 512), // down:    (hidden, moe_intermediate)
    ] {
        let (a, b) = pack_both_ways(n, k, n as u64 * 31 + k as u64);
        assert_sfb_eq(n, k, &a, &b);
    }
}

/// The swizzle must be a permutation of the source values, not a
/// value-transforming pass.
///
/// Parity alone cannot catch a fault that corrupts BOTH arms identically — two
/// wrongs agreeing still pass `assert_sfb_eq`. This checks the packed atom
/// against the source as a multiset: every source scale must appear in the
/// output the same number of times, with the padding accounted for separately.
///
/// Values round-trip `E4M3 byte -> float -> ue4m3 -> byte`. That is
/// bit-preserving for the non-negative finite domain used here (both types
/// share the E4M3 field layout for positive values), so an exact multiset
/// compare is the right assertion rather than an approximate one.
#[test]
#[ignore = "requires GPU"]
fn sfb_swizzle_preserves_values() {
    let (n, k) = (640, 2560);
    let groups = k / 16;
    let len = sfb_len(n, k);

    let mut src = vec![0u8; n * groups];
    for row in 0..n {
        for g in 0..groups {
            src[row * groups + g] = scale_byte(row, g, 0x5E);
        }
    }
    let d_src = DevBuf::upload(&src);
    let d_out = DevBuf::zeroed(len);
    pack_weight_sfb(d_src.addr(), d_out.addr(), n as u32, k as u32, true, 0)
        .expect("pack_weight_sfb");
    cuda_check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize");
    let out = d_out.download(len);

    let mut want = [0usize; 256];
    for b in &src {
        want[*b as usize] += 1;
    }
    let mut got = [0usize; 256];
    for b in &out {
        got[*b as usize] += 1;
    }
    // The atom is larger than the live region; the surplus is the zero fill.
    let padding = len - n * groups;
    got[0] = got[0].saturating_sub(padding);

    for v in 0..256usize {
        assert_eq!(
            got[v], want[v],
            "value {v:#04x}: swizzle emitted {} occurrences, source had {} \
             (n={n} k={k}). The SFB pack must PERMUTE the scales, so any \
             change in the multiset means values are being dropped, \
             duplicated, or altered — which parity between the two source \
             layouts cannot detect on its own.",
            got[v], want[v]
        );
    }
}
