// SPDX-License-Identifier: AGPL-3.0-only

//! GPU EQUALITY: the GQA-packed paged-decode kernels against the unpacked ones.
//!
//! `paged_decode_attn_{fp8,bf16}_gqa` claim BIT-IDENTITY with
//! `paged_decode_attn_fp8` / `paged_decode_attn`: per q head the same warp
//! partition of `[window_start, seq_len)`, the same `BC = 4` batched block
//! loop, the same dot-product term order, the same 5-step `__shfl_xor_sync`
//! butterfly and the same 8-warp tree merge, with only the K/V loads hoisted
//! above the per-head loop. That claim was argued at SOURCE level and compiled;
//! until this file ran, nothing had ever EXECUTED it.
//!
//! This test does not re-derive the argument. It runs both kernels over the
//! same Q, the same KV pool, the same block table and the same seq lens into
//! two separate output buffers and compares them BYTE for byte, no tolerance —
//! the claim is bit-identity, not closeness.
//!
//! # Why the vacuity guards are half this file
//!
//! A parity test that passes because NEITHER kernel ran, or because both arms
//! ran the SAME kernel, is worse than no test. Three independent positives are
//! asserted before any comparison is believed:
//!
//! 1. **Distinct entry points.** Both handles resolve non-zero and differ.
//! 2. **The real route accepts this shape.** [`splitk_dispatch::gqa_pack_kernel`]
//!    — the production conjunction, lever included — must hand back the packed
//!    handle. `gqa_pack_enabled()` is a process-wide `OnceLock` that resolves
//!    `false` by default, so this test ASSERTS it armed rather than
//!    short-circuiting to a green nothing; run it with
//!    `AVAROK_ATTN_DECODE_GQA_PACK=1`.
//! 3. **Each arm wrote every head it owns.** Both output buffers start at
//!    `0xA5`. The packed grid is `(num_kv_heads, num_seqs)` = 4 CTAs per
//!    sequence for 24 q heads, so a run that left ANY of the 24 head slices
//!    still all-`0xA5` is a run in which the unpacked kernel — one head per CTA
//!    — was launched under the packed grid. Only a kernel that writes
//!    `PD_GQA = 6` heads per CTA can fill the buffer.
//!
//! `#[ignore]` per repo convention (needs a GPU and a built PTX set). Run with:
//! ```text
//! AVAROK_ATTN_DECODE_GQA_PACK=1 cargo test -p spark-model --release \
//!   gqa_packed_decode_is_byte_identical -- --ignored --nocapture
//! ```

use avarok_kernels::attn_splitk;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::gqa_pack_fixture::{
    BANDS, BLOCK_SIZE, Band, CASES, Case, Fixture, HD, K_SCALE, NKV, NQ, UNWRITTEN, V_SCALE,
    all_finite, bf16_bits, fixture, fp8_byte, heads_written, uniform,
};
use super::splitk_dispatch;
use crate::layers::ops;

#[test]
#[ignore]
#[allow(clippy::too_many_lines)]
fn gqa_packed_decode_is_byte_identical_to_unpacked() {
    const NEEDED: [&str; 4] = [
        "paged_decode",
        "paged_decode_fp8",
        "paged_decode_attn_bf16_gqa",
        "paged_decode_attn_fp8_gqa",
    ];
    let set = avarok_kernels::all_ptx_sets()
        .into_iter()
        .find(|t| {
            NEEDED
                .iter()
                .all(|m| t.modules.iter().any(|(name, _)| name == m))
        })
        .expect(
            "no built PTX target carries paged_decode + paged_decode_fp8 + both \
             GQA-packed twins; build the GB10 kernels first",
        );
    println!(
        "target {}/{} arch {}",
        set.target.model, set.target.quant, set.ptx_arch
    );

    // ── Vacuity guard 1: the shape gate, stated against the predicate the
    // dispatch uses, not against these constants.
    assert_eq!(
        NQ / NKV,
        attn_splitk::DECODE_GQA_PACK_WIDTH,
        "nq/nkv must be the pack width"
    );
    assert_eq!(HD, attn_splitk::DECODE_GQA_PACK_HEAD_DIM);
    assert!(
        attn_splitk::gqa_pack_shape_ok(NQ, NKV, HD),
        "nq={NQ} nkv={NKV} head_dim={HD} is not a shape the packed kernels serve",
    );

    // ── Vacuity guard 2: the lever. `gqa_pack_enabled()` is a process-wide
    // OnceLock resolving `false` by default; a test that let it answer `false`
    // would find `gqa_pack_kernel` returning None at its first line and would
    // go green having measured nothing.
    assert!(
        attn_splitk::gqa_pack_enabled(),
        "AVAROK_ATTN_DECODE_GQA_PACK is not armed in this process — re-run with \
         AVAROK_ATTN_DECODE_GQA_PACK=1. Without it the packed kernel is never \
         the one production would pick and this test would be measuring a route \
         nobody takes.",
    );

    let gpu =
        spark_runtime::cuda_backend::AvarokCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();

    let k_fp8 = g
        .kernel("paged_decode_fp8", "paged_decode_attn_fp8")
        .unwrap();
    let k_fp8_gqa = g
        .kernel("paged_decode_attn_fp8_gqa", "paged_decode_attn_fp8_gqa")
        .unwrap();
    let k_bf16 = g.kernel("paged_decode", "paged_decode_attn").unwrap();
    let k_bf16_gqa = g
        .kernel("paged_decode_attn_bf16_gqa", "paged_decode_attn_bf16_gqa")
        .unwrap();

    // ── Vacuity guard 3: four DISTINCT, resolved entry points.
    for (n, h) in [
        ("paged_decode_fp8::paged_decode_attn_fp8", k_fp8),
        (
            "paged_decode_attn_fp8_gqa::paged_decode_attn_fp8_gqa",
            k_fp8_gqa,
        ),
        ("paged_decode::paged_decode_attn", k_bf16),
        (
            "paged_decode_attn_bf16_gqa::paged_decode_attn_bf16_gqa",
            k_bf16_gqa,
        ),
    ] {
        assert!(h.0 != 0, "{n} resolved to handle 0");
        println!("resolved {n} -> handle {}", h.0);
    }
    assert_ne!(k_fp8.0, k_fp8_gqa.0, "FP8 arms resolved the SAME kernel");
    assert_ne!(k_bf16.0, k_bf16_gqa.0, "BF16 arms resolved the SAME kernel");

    // ── Vacuity guard 4: the PRODUCTION route, lever and all, hands back the
    // packed handle for this shape — so the kernel launched below is the one
    // `run_paged_decode` would launch, not one only this test can reach.
    for (arm, h) in [("fp8", k_fp8_gqa), ("bf16", k_bf16_gqa)] {
        let routed = splitk_dispatch::gqa_pack_kernel(Some(h), NQ, NKV, HD);
        assert_eq!(
            routed.map(|r| r.0),
            Some(h.0),
            "{arm}: gqa_pack_kernel refused the packed handle at the production shape",
        );
        // And it refuses a ratio the kernel cannot index, which is the check
        // the guard above would otherwise be indistinguishable from.
        assert!(
            splitk_dispatch::gqa_pack_kernel(Some(h), NQ + 1, NKV, HD).is_none(),
            "{arm}: gqa_pack_kernel accepted nq={} over nkv={NKV}",
            NQ + 1,
        );
    }

    let upload = |b: &[u8]| -> DevicePtr {
        let p = g.alloc(b.len().max(256)).unwrap();
        g.copy_h2d_async(b, p, stream).unwrap();
        p
    };
    let out_pair = |bytes: usize| -> (DevicePtr, DevicePtr) {
        let a = g.alloc(bytes).unwrap();
        let b = g.alloc(bytes).unwrap();
        g.memset_async(a, UNWRITTEN, bytes, stream).unwrap();
        g.memset_async(b, UNWRITTEN, bytes, stream).unwrap();
        (a, b)
    };

    let mut compared = 0usize;
    for (ci, case) in CASES.iter().enumerate() {
        for (bi, band) in BANDS.iter().enumerate() {
            let seed = 0x9E37_79B9_0000_0001 ^ ((ci as u64) << 8) ^ (bi as u64);
            let inv_sqrt_d = 1.0f32 / (HD as f32).sqrt();
            let cache_stride = u64::from(BLOCK_SIZE * NKV * HD);

            // ── FP8 KV ──────────────────────────────────────────────────────
            let (lo, hi) = band.fp8_exp;
            let f = fixture(case, seed, 1, |s| vec![fp8_byte(s, lo, hi)]);
            let (q, kp, vp, bt, sl) = (
                upload(&f.q),
                upload(&f.k),
                upload(&f.v),
                upload(&f.block_table),
                upload(&f.seq_lens),
            );
            let (o_ref, o_gqa) = out_pair(f.out_bytes);
            ops::paged_decode_attn_fp8(
                g,
                k_fp8,
                q,
                kp,
                vp,
                o_ref,
                bt,
                sl,
                f.max_blocks_per_seq,
                f.num_seqs,
                NQ,
                NKV,
                HD,
                BLOCK_SIZE,
                inv_sqrt_d,
                K_SCALE,
                V_SCALE,
                NQ * HD,
                cache_stride,
                case.sliding,
                stream,
            )
            .unwrap();
            ops::paged_decode_attn_fp8_gqa(
                g,
                k_fp8_gqa,
                q,
                kp,
                vp,
                o_gqa,
                bt,
                sl,
                f.max_blocks_per_seq,
                f.num_seqs,
                NQ,
                NKV,
                HD,
                BLOCK_SIZE,
                inv_sqrt_d,
                K_SCALE,
                V_SCALE,
                NQ * HD,
                cache_stride,
                case.sliding,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            compare("fp8", case, band, &f, g, o_ref, o_gqa);
            compared += 1;
            for p in [q, kp, vp, bt, sl, o_ref, o_gqa] {
                g.free(p).unwrap();
            }

            // ── BF16 KV ─────────────────────────────────────────────────────
            let mag = band.bf16_mag;
            let f = fixture(case, seed ^ 0xBF16, 2, |s| {
                bf16_bits(uniform(s, mag)).to_le_bytes().to_vec()
            });
            let (q, kp, vp, bt, sl) = (
                upload(&f.q),
                upload(&f.k),
                upload(&f.v),
                upload(&f.block_table),
                upload(&f.seq_lens),
            );
            let (o_ref, o_gqa) = out_pair(f.out_bytes);
            ops::paged_decode_attn_bf16(
                g,
                k_bf16,
                q,
                kp,
                vp,
                o_ref,
                bt,
                sl,
                f.max_blocks_per_seq,
                f.num_seqs,
                NQ,
                NKV,
                HD,
                BLOCK_SIZE,
                inv_sqrt_d,
                NQ * HD,
                case.sliding,
                stream,
            )
            .unwrap();
            ops::paged_decode_attn_bf16_gqa(
                g,
                k_bf16_gqa,
                q,
                kp,
                vp,
                o_gqa,
                bt,
                sl,
                f.max_blocks_per_seq,
                f.num_seqs,
                NQ,
                NKV,
                HD,
                BLOCK_SIZE,
                inv_sqrt_d,
                NQ * HD,
                case.sliding,
                stream,
            )
            .unwrap();
            g.synchronize(stream).unwrap();
            compare("bf16", case, band, &f, g, o_ref, o_gqa);
            compared += 1;
            for p in [q, kp, vp, bt, sl, o_ref, o_gqa] {
                g.free(p).unwrap();
            }
        }
    }
    println!("{compared} (case, band, dtype) launches compared byte-for-byte");
    assert_eq!(
        compared,
        CASES.len() * BANDS.len() * 2,
        "not every cell ran"
    );
}

/// Download both buffers, prove both arms wrote, then compare bytes.
fn compare(
    dtype: &str,
    case: &Case,
    band: &Band,
    f: &Fixture,
    g: &dyn GpuBackend,
    o_ref: DevicePtr,
    o_gqa: DevicePtr,
) {
    let mut a = vec![0u8; f.out_bytes];
    let mut b = vec![0u8; f.out_bytes];
    g.copy_d2h(o_ref, &mut a).unwrap();
    g.copy_d2h(o_gqa, &mut b).unwrap();
    let tag = format!("{dtype} {} {}", case.label, band.label);

    heads_written(&a, f.num_seqs)
        .unwrap_or_else(|e| panic!("{tag}: the UNPACKED arm did not run: {e}"));
    heads_written(&b, f.num_seqs).unwrap_or_else(|e| {
        panic!(
            "{tag}: the PACKED arm did not write every head: {e}. Under \
             grid=(nkv={NKV}, num_seqs={}) only a kernel writing PD_GQA heads per \
             CTA can fill this buffer — an unpacked kernel launched here would \
             leave heads {NKV}..{NQ} untouched.",
            f.num_seqs,
        )
    });
    all_finite(&a).unwrap_or_else(|e| panic!("{tag}: {e}"));

    if let Some((i, (x, y))) = a
        .iter()
        .zip(b.iter())
        .enumerate()
        .find(|(_, (x, y))| x != y)
    {
        panic!(
            "{tag}: output differs at byte {i} (unpacked 0x{x:02X} vs packed 0x{y:02X}) — \
             the packed kernel is NOT bit-identical",
        );
    }
    println!("{tag}: {} output bytes byte-identical", f.out_bytes);
}
