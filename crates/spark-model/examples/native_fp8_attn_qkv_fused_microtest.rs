// SPDX-License-Identifier: AGPL-3.0-only
//! The FUSED attention Q/K/V decode GEMM against the three GEMMs it replaces,
//! at Qwen3.8-27B shapes (#927).
//!
//! WHY. nsys `--cuda-graph-trace=node`, 1xH100 80GB HBM3,
//! `Qwen/Qwen3.8-27B-FP8` @ `3717cb05e`, round 13 cell V, median `n = 16`
//! decode step **19.887 ms** busy (`h100-r13-attribution.md` §§C.2–C.4):
//! `q_proj` is 16 graph nodes, 460.0 µs = 28.75 µs/node (`K = 5120`,
//! `N = 12288`, 2 189 GB/s = 65.3 % of HBM), and `k_proj` + `v_proj` are 32
//! nodes, 510.9 µs = **15.97 µs/node** for a **5.24 MB** weight read —
//! **328 GB/s = 9.8 % of HBM**, the worst-utilised GEMM in the step by a
//! factor of six. One N=1024 node is 8 tiles of 128 columns on 132 SMs;
//! appended onto `q_proj` they ride a wave that is already running. Rank 5 of
//! the round-13 decode table, **428 µs/step (2.2 %)**.
//!
//! It answers four questions and guesses none:
//!
//!   1. BITS. Concatenating along N gives INDEPENDENT output columns over the
//!      same K with the same block scales, so this is a BYTE-equality gate and
//!      not a tolerance: fused columns `[0, 12288)` must equal `q_proj`,
//!      `[12288, 13312)` `k_proj` and `[13312, 14336)` `v_proj`, over the full
//!      padded row extent. Three KNOWN_BAD controls prove it can go red.
//!   2. THE CONSUMERS. The fused output IS the `[n, per_seq_qkv]` slot layout —
//!      the whole layout argument — so the gate is not the GEMM alone:
//!      `deinterleave_qg` and the KV-cache write on BOTH KV dtypes (BF16, and
//!      the FP8 storage the #919 calibrated path uses) must produce the same
//!      bytes from either arm.
//!   3. THE PHANTOM ROWS. cuBLASLt is handed `ceil16(M) = 16` at every rung of
//!      this band and WRITES rows `m..16`, so both arms are compared over the
//!      padded extent and every buffer carries guard bands.
//!   4. TIME. Per `M`, sync'd over `REPS`: µs and weight-GB/s for both arms
//!      plus the k/v pair alone — the 328 GB/s row the lever deletes.
//!
//! Run (H100): `ATLAS_TARGET_HW=hopper cargo run --release -p spark-model
//! --features cuda,gpu-examples --example native_fp8_attn_qkv_fused_microtest`.
//! ★ The target matters: `fp8_scale_transpose.cu` is a HOPPER-owned source
//! (`[kernels] overrides`), so a GB10 build fails the lookup by name. cuBLASLt
//! is called directly, so `ATLAS_CUBLAS_GEMM` is not needed.

use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_model::weight_map::{Fp8Weight, WeightQuantFormat};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

const H: usize = 5120; // hidden / contraction width
const NQ: usize = 24; // q heads
const NKV: usize = 4; // kv heads
const HD: usize = 256; // head dim
const Q_N: usize = NQ * HD * 2; // gated [Q|gate] = 12288
const KV_N: usize = NKV * HD; // 1024
const FUSED_N: usize = Q_N + 2 * KV_N; // 14336 == per_seq_qkv / 2
const MAX_M: usize = 16; // the cuBLASLt M pad across the whole band
const ROWS: [usize; 3] = [5, 8, 16];
const REPS: usize = 20;
const GUARD: usize = 64; // sentinel bytes either side of every buffer
const SENTINEL: u8 = 0x5a;
const BF16: usize = 2;
const BLOCK_SIZE: usize = 16; // KV pool page
const SLOTS: usize = 64; // KV pool slots, enough for MAX_M writes

struct Rng(u64);

impl Rng {
    fn bits(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// FP8 E4M3 bytes, sign kept, exponents clipped off the NaN encodings.
    fn fp8(&mut self, elems: usize) -> Vec<u8> {
        (0..elems)
            .map(|_| {
                let x = self.bits();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect()
    }
    /// One FP32 scale per 128x128 weight block.
    fn scales(&mut self, blocks: usize) -> Vec<u8> {
        (0..blocks)
            .flat_map(|_| (((self.bits() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect()
    }
    fn acts(&mut self, elems: usize) -> Vec<u8> {
        (0..elems)
            .flat_map(|_| {
                bf16::from_f32(((self.bits() % 2049) as f32 - 1024.0) / 1024.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    }
}

/// Report one comparison and tally a failure rather than aborting, so a red
/// run names EVERY slice that moved and not just the first.
fn check(name: &str, r: Result<()>, failures: &mut usize) {
    match r {
        Ok(()) => println!("  {name}: BYTE-IDENTICAL"),
        Err(e) => {
            *failures += 1;
            println!("  {name}: FAIL {e}");
        }
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// A guarded device buffer: `GUARD` sentinel bytes, `len` payload, `GUARD`
/// sentinel bytes. `ptr` is the payload; `base` is what gets re-armed.
struct Guarded {
    base: DevicePtr,
    ptr: DevicePtr,
    sentinel: Vec<u8>,
    len: usize,
}

impl Guarded {
    fn new(gpu: &dyn GpuBackend, len: usize) -> Result<Self> {
        let sentinel = vec![SENTINEL; len + 2 * GUARD];
        let base = upload(gpu, &sentinel)?;
        Ok(Self {
            base,
            ptr: base.offset(GUARD),
            sentinel,
            len,
        })
    }
    fn arm(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.sentinel, self.base)
    }
    /// Payload bytes, after checking both guard bands survived.
    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut host = vec![0_u8; self.sentinel.len()];
        gpu.copy_d2h(self.base, &mut host)?;
        ensure!(
            host[..GUARD].iter().all(|&b| b == SENTINEL)
                && host[GUARD + self.len..].iter().all(|&b| b == SENTINEL),
            "a launch wrote outside its output extent (guard band clobbered)"
        );
        Ok(host[GUARD..GUARD + self.len].to_vec())
    }
}

/// The oracle: `observed` must equal `reference` byte for byte over `spans`
/// (byte offset, byte length). Also refuses anything non-finite inside those
/// spans when the payload is BF16.
fn equal_bytes(
    observed: &[u8],
    reference: &[u8],
    spans: &[(usize, usize)],
    check_finite: bool,
) -> Result<()> {
    for &(o, len) in spans {
        let a = &observed[o..o + len];
        let b = &reference[o..o + len];
        ensure!(
            !check_finite
                || a.chunks_exact(2)
                    .all(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).is_finite()),
            "nonfinite value in the fused output"
        );
        if a != b {
            let bad = a.iter().zip(b).filter(|(x, y)| x != y).count();
            anyhow::bail!("{bad} of {len} bytes differ (byte equality is the gate)");
        }
    }
    Ok(())
}

/// `out[m, n] = a_fp8[m, k] @ w[n, k]ᵀ` at output row pitch `ldc` — the exact
/// call the serve's `ops::decode_w8a8_gemm` makes, including the K-major
/// activation-scale layout cuBLASLt documents.
#[allow(clippy::too_many_arguments)]
fn gemm(
    gpu: &dyn GpuBackend,
    kmajor_k: KernelHandle,
    a_fp8: DevicePtr,
    a_scale: DevicePtr,
    a_kmajor: DevicePtr,
    w: &Fp8Weight,
    out: DevicePtr,
    m: usize,
) -> Result<()> {
    let m_pad = ops::cublas_fp8_m_pad(m as u32);
    ops::fp8_act_scale_to_kmajor(
        gpu, kmajor_k, a_scale, a_kmajor, m as u32, m_pad, H as u32, 0,
    )?;
    spark_runtime::cublaslt::fp8_gemm_act_weight_t_blkscaled_ldc(
        a_fp8.0,
        a_kmajor.0,
        w.weight.0,
        w.row_scale.0,
        out.0,
        m_pad,
        w.n,
        w.k,
        FUSED_N as u32,
        0,
    )
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let quant = ops::Fp8ActQuant::resolve(&gpu);
    let kmajor = gpu.kernel("fp8_scale_transpose", "fp8_act_scale_to_kmajor")?;
    let deint = gpu.kernel("ssm_preprocess", "deinterleave_qg")?;
    let cache_bf16 = gpu.kernel("reshape_and_cache", "reshape_and_cache_flash")?;
    let cache_fp8 = gpu.kernel("reshape_and_cache", "reshape_and_cache_flash_fp8")?;
    println!(
        "gate: BYTE EQUALITY (not a tolerance). Splitting N gives independent \
         output columns over the same K with the same block scales, so the \
         fused slices must reproduce the three GEMMs exactly — and so must \
         both consumers that read them."
    );
    println!("shapes: K={H} q_proj N={Q_N} k/v N={KV_N} fused N={FUSED_N}");

    let mut rng = Rng(0x0927_2026_0A77_0001);

    // ONE weight allocation holding `[q|k|v]` along N, with the three as VIEWS
    // inside it — the loader's contract reproduced byte for byte
    // (`weight_loader/qwen35_dense.rs`), so any difference the comparison sees
    // is the GEMM's and not the operands'.
    let w_bytes = rng.fp8(FUSED_N * H);
    let kb = H / 128;
    let s_bytes = rng.scales((FUSED_N / 128) * kb);
    let fused_w = upload(&gpu, &w_bytes)?;
    let fused_s = upload(&gpu, &s_bytes)?;
    let view = |n_off: usize, n: usize| Fp8Weight {
        weight: fused_w.offset(n_off * H),
        row_scale: fused_s.offset((n_off / 128) * kb * 4),
        n: n as u32,
        k: H as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    let q_w = view(0, Q_N);
    let k_w = view(Q_N, KV_N);
    let v_w = view(Q_N + KV_N, KV_N);
    let fused = view(0, FUSED_N);

    let act = upload(&gpu, &rng.acts(MAX_M * H))?;
    let a_fp8 = gpu.alloc(MAX_M * H)?;
    let a_scale = gpu.alloc(MAX_M * kb * 4)?;
    let a_kmajor = gpu.alloc(MAX_M * kb * 4)?;

    // Both arms write the SAME `[16, 14336]` slot layout — that is the claim.
    let three_out = Guarded::new(&gpu, MAX_M * FUSED_N * BF16)?;
    let fused_out = Guarded::new(&gpu, MAX_M * FUSED_N * BF16)?;

    // KV pool + slot mapping for the cache-write gate.
    let pool_elems = SLOTS * NKV * HD;
    let slots: Vec<u8> = (0..MAX_M as i64).flat_map(|i| i.to_le_bytes()).collect();
    let slot_mapping = upload(&gpu, &slots)?;
    let pool = |elem: usize| Guarded::new(&gpu, pool_elems * elem);
    let (kc_bf, vc_bf, kc_bf2, vc_bf2) = (pool(BF16)?, pool(BF16)?, pool(BF16)?, pool(BF16)?);
    let (kc_f8, vc_f8, kc_f82, vc_f82) = (pool(1)?, pool(1)?, pool(1)?, pool(1)?);

    // The whole padded extent: cuBLASLt writes rows `m..16`, so every row is
    // compared, phantom rows included. A phantom row that differed between the
    // arms would be a real difference in a serve, where those rows land in
    // decode slots not in the step.
    let slice_spans = |col_off: usize, width: usize| -> Vec<(usize, usize)> {
        (0..MAX_M)
            .map(|r| ((r * FUSED_N + col_off) * BF16, width * BF16))
            .collect()
    };
    let q_spans = slice_spans(0, Q_N);
    let k_spans = slice_spans(Q_N, KV_N);
    let v_spans = slice_spans(Q_N + KV_N, KV_N);

    let mut failures = 0usize;
    let mut controls_done = false;

    for m in ROWS {
        println!("\n=== M = {m} (cuBLASLt pad = {MAX_M}) ===");
        // Quantize ONCE — q, k and v read the same `normed` rows, which is why
        // the serve splits `decode_w8a8_quant_act` from the GEMM.
        ops::per_token_group_quant_fp8(&gpu, quant, act, a_fp8, a_scale, m as u32, H as u32, 0)?;
        let m_pad = ops::cublas_fp8_m_pad(m as u32) as usize;
        if m_pad > m {
            gpu.memset_async(a_fp8.offset(m * H), 0, (m_pad - m) * H, 0)?;
        }

        // The three arms, each as ONE closure so the correctness pass and the
        // timing loop below cannot drift apart.
        let proj = |w: &Fp8Weight, col: usize| {
            gemm(
                &gpu,
                kmajor,
                a_fp8,
                a_scale,
                a_kmajor,
                w,
                three_out.ptr.offset(col * BF16),
                m,
            )
        };
        let three_gemms = || {
            proj(&q_w, 0)?;
            proj(&k_w, Q_N)?;
            proj(&v_w, Q_N + KV_N)
        };
        let kv_gemms = || {
            proj(&k_w, Q_N)?;
            proj(&v_w, Q_N + KV_N)
        };
        let one_gemm = || {
            gemm(
                &gpu,
                kmajor,
                a_fp8,
                a_scale,
                a_kmajor,
                &fused,
                fused_out.ptr,
                m,
            )
        };

        // ── the three ──
        three_out.arm(&gpu)?;
        three_gemms()?;
        gpu.synchronize(0)?;
        let three_host = three_out.read(&gpu)?;

        // ── the fused arm ──
        fused_out.arm(&gpu)?;
        one_gemm()?;
        gpu.synchronize(0)?;
        let fused_host = fused_out.read(&gpu)?;

        for (name, spans) in [
            ("Q slice", &q_spans),
            ("K slice", &k_spans),
            ("V slice", &v_spans),
        ] {
            check(
                name,
                equal_bytes(&fused_host, &three_host, spans, true),
                &mut failures,
            );
        }

        // ── consumer 1: deinterleave_qg, in place over the Q columns, at the
        // SAME row stride both arms produce ──
        for buf in [three_out.ptr, fused_out.ptr] {
            let (nq, hd, ld) = (NQ as u32, HD as u32, FUSED_N as u32);
            ops::deinterleave_qg(&gpu, deint, buf, m as u32, nq, hd, ld, 0)?;
        }
        gpu.synchronize(0)?;
        let (r, o) = (three_out.read(&gpu)?, fused_out.read(&gpu)?);
        let live: Vec<(usize, usize)> = (0..m)
            .map(|row| (row * FUSED_N * BF16, FUSED_N * BF16))
            .collect();
        check(
            "deinterleave_qg",
            equal_bytes(&o, &r, &live, true),
            &mut failures,
        );

        // ── consumer 2: the KV-cache write, both dtypes ──
        // K lives at column 12288 and V at 13312 with row stride 14336 — the
        // strides the kernel already takes. Nothing about the write changes;
        // this proves it.
        for (dtype, kk, (kc_a, vc_a), (kc_b, vc_b)) in [
            ("KV bf16", cache_bf16, (&kc_bf, &vc_bf), (&kc_bf2, &vc_bf2)),
            ("KV fp8", cache_fp8, (&kc_f8, &vc_f8), (&kc_f82, &vc_f82)),
        ] {
            let arms = [(three_out.ptr, kc_a, vc_a), (fused_out.ptr, kc_b, vc_b)];
            for (src, kc, vc) in arms {
                kc.arm(&gpu)?;
                vc.arm(&gpu)?;
                let (key, value) = (src.offset(Q_N * BF16), src.offset((Q_N + KV_N) * BF16));
                if dtype == "KV bf16" {
                    ops::reshape_and_cache(
                        &gpu,
                        kk,
                        key,
                        value,
                        kc.ptr,
                        vc.ptr,
                        slot_mapping,
                        m as u32,
                        NKV as u32,
                        HD as u32,
                        BLOCK_SIZE as u32,
                        FUSED_N as u32,
                        FUSED_N as u32,
                        0,
                        0,
                    )?;
                } else {
                    ops::reshape_and_cache_fp8(
                        &gpu,
                        kk,
                        key,
                        value,
                        kc.ptr,
                        vc.ptr,
                        slot_mapping,
                        m as u32,
                        NKV as u32,
                        HD as u32,
                        BLOCK_SIZE as u32,
                        1.0,
                        1.0,
                        FUSED_N as u32,
                        FUSED_N as u32,
                        (BLOCK_SIZE * NKV * HD) as u64,
                        0,
                    )?;
                }
            }
            gpu.synchronize(0)?;
            let whole = vec![(0usize, kc_a.len)];
            for (what, b, a) in [("k_cache", kc_b, kc_a), ("v_cache", vc_b, vc_a)] {
                let r = equal_bytes(&b.read(&gpu)?, &a.read(&gpu)?, &whole, false);
                check(&format!("{dtype} {what}"), r, &mut failures);
            }
        }

        if !controls_done {
            // A green run has to be able to go red.
            for control in ["wrong slice", "one byte", "nonfinite"] {
                let mut bad = fused_host.clone();
                let spans = match control {
                    // The layout error this gate exists to catch: reading V
                    // where K belongs — what a wrong N order or a wrong row
                    // stride would produce.
                    "wrong slice" => {
                        let ((k_off, len), (v_off, _)) = (k_spans[0], v_spans[0]);
                        bad.copy_within(v_off..v_off + len, k_off);
                        &k_spans
                    }
                    "one byte" => {
                        bad[q_spans[MAX_M - 1].0 + 2] ^= 1;
                        &q_spans
                    }
                    _ => {
                        let o = v_spans[0].0;
                        bad[o..o + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
                        &v_spans
                    }
                };
                let err = equal_bytes(&bad, &three_host, spans, true)
                    .expect_err("known-bad output was admitted by the real oracle");
                println!("  KNOWN_BAD {control}: refused: {err}");
            }
            controls_done = true;
        }

        // ── time ──
        let bytes = (FUSED_N * H) as f64; // all three, read once either way
        let kv_bytes = (2 * KV_N * H) as f64;
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            f()?;
            gpu.synchronize(0)?;
            let t = Instant::now();
            for _ in 0..REPS {
                f()?;
            }
            gpu.synchronize(0)?;
            Ok(t.elapsed().as_secs_f64() / REPS as f64)
        };
        let three = time(&three_gemms)?;
        let kv_only = time(&kv_gemms)?;
        let one = time(&one_gemm)?;
        println!(
            "  three GEMMs: {:8.2} us {:7.0} GB/s  |  k+v alone: {:8.2} us {:7.0} GB/s  \
             |  fused: {:8.2} us {:7.0} GB/s  ({:.2}x)",
            three * 1e6,
            bytes / three / 1e9,
            kv_only * 1e6,
            kv_bytes / kv_only / 1e9,
            one * 1e6,
            bytes / one / 1e9,
            three / one,
        );
    }

    println!(
        "\nserve spelling: the arm is `[defaults] attn_qkv_fused` (hopper \
         `true`); `ATLAS_ATTN_QKV_FUSED=0` restores the three-GEMM arm. \
         Round-13 receipt and the round-17 prediction: \
         ATTN-QKV-FUSION-ATTRIBUTION.md"
    );
    ensure!(failures == 0, "{failures} byte-equality gate(s) failed");
    println!("OK");
    Ok(())
}
