// SPDX-License-Identifier: AGPL-3.0-only

//! GPU parity for the PLE gate and conv against the reference.
//!
//! `ple.cu` was written from `Qwen4ExpTextPLELayer.forward`. Three details in
//! it are invisible when wrong — the offset-from-1 norms, the GROUPED norm,
//! and the gate's SIGNED SQUARE ROOT — so it gets held to a number before it
//! is wired into the highway, exactly as the mHC kernel was in phase A.
//!
//! Fixtures: `bench/qwen4_exp/ple_golden.py --bin-dir <dir>`, whose
//! intermediates come from the reference's OWN submodules on the real
//! checkpoint weights.
//!
//! GPU test: `#[ignore]` per repo convention. Run with
//! ```text
//! ATLAS_PLE_TEST_DATA=/tank/atlas-testdata/qwen4exp_ple \
//!   cargo test -p spark-model --release ple_kernels -- --ignored --nocapture
//! ```

use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

struct Fx {
    dir: String,
    hc: usize,
    h: usize,
    t: usize,
    eps: f32,
    k_size: usize,
    dilation: usize,
}

impl Fx {
    fn load() -> Self {
        let dir = std::env::var("ATLAS_PLE_TEST_DATA").expect(
            "set ATLAS_PLE_TEST_DATA — generate with \
             `python3 -u bench/qwen4_exp/ple_golden.py --bin-dir <dir>`",
        );
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json")).unwrap())
                .unwrap();
        // All three PLE norms are `Qwen4ExpTextRMSNorm`, the offset-from-1
        // form. A fixture regenerated under the other convention would
        // quietly retune the tolerance instead of failing.
        assert_eq!(
            m["norm_convention"].as_str().unwrap(),
            "normed * (1.0 + weight)"
        );
        Self {
            dir,
            hc: m["hc_count"].as_u64().unwrap() as usize,
            h: m["hidden_size"].as_u64().unwrap() as usize,
            t: m["num_tokens"].as_u64().unwrap() as usize,
            eps: m["rms_norm_eps"].as_f64().unwrap() as f32,
            k_size: m["conv_kernel_size"].as_u64().unwrap() as usize,
            dilation: m["conv_dilation"].as_u64().unwrap() as usize,
        }
    }

    fn bytes(&self, n: &str) -> Vec<u8> {
        let p = format!("{}/{n}.bin", self.dir);
        std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
    }

    fn f32s(&self, n: &str) -> Vec<f32> {
        self.bytes(n)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

fn upload(g: &dyn GpuBackend, b: &[u8]) -> DevicePtr {
    let p = g.alloc(b.len().max(256)).unwrap();
    g.copy_h2d_async(b, p, g.default_stream()).unwrap();
    p
}

fn dl_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 4];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn compare(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut max_abs = 0.0f32;
    let (mut dot, mut ng, mut nw) = (0.0f64, 0.0f64, 0.0f64);
    let mut worst = 0usize;
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        dot += a as f64 * b as f64;
        ng += a as f64 * a as f64;
        nw += b as f64 * b as f64;
    }
    let cos = dot / (ng.sqrt() * nw.sqrt()).max(1e-30);
    let rms = (nw / want.len() as f64).sqrt();
    println!(
        "  {label:<16} max|diff|={max_abs:.4e} cos={cos:.9} ref_rms={rms:.4e} \
         worst[{worst}] got={:.6} want={:.6}",
        got[worst], want[worst]
    );
    // RELATIVE tolerance, floored at the RMS. A bound scaled to the global
    // RMS alone is wrong for a heavy-tailed tensor: `gated_normed` has RMS
    // 0.82 but peaks at 12.6, so an RMS-scaled bound demands 15x more
    // precision at the peak than BF16 can store. Dividing by each element's
    // own magnitude — floored at the RMS so near-zero elements do not blow
    // the ratio up — asks for the same relative accuracy everywhere.
    //
    // BF16 carries ~8 mantissa bits (2^-8 = 0.39% per store) and the kernel
    // accumulates a 2560-term reduction in a different order than torch, so
    // 2% is a real bound, not a fitted one: the defects this test exists to
    // catch (a plain-`w` norm, a global RMS, a missing signed sqrt) all miss
    // by whole multiples, not by fractions of a percent.
    let mut max_rel = 0.0f32;
    for (&a, &b) in got.iter().zip(want) {
        let denom = b.abs().max(rms as f32);
        max_rel = max_rel.max((a - b).abs() / denom);
    }
    println!("  {label:<16} max_rel={max_rel:.4e}");
    assert!(
        max_rel <= 0.02,
        "{label}: max relative error {max_rel:.4e} > 2%"
    );
    assert!(cos > 0.9999, "{label}: cosine {cos:.9}");
}

#[test]
#[ignore]
fn ple_kernels_match_reference() {
    let f = Fx::load();
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let (t, h, hc) = (f.t, f.h, f.hc);
    let c = hc * h;
    println!(
        "T={t} hidden={h} hc={hc} channels={c} k={} dilation={} eps={:e}",
        f.k_size, f.dilation, f.eps
    );

    let k_gate = g.kernel("ple", "ple_gate").unwrap();
    let k_conv = g.kernel("ple", "ple_conv").unwrap();
    for (n, k) in [("ple_gate", k_gate), ("ple_conv", k_conv)] {
        assert!(k.0 != 0, "{n} resolved to handle 0");
    }

    // ── gate ──
    let hidden = upload(g, &f.bytes("hidden"));
    let key = upload(g, &f.bytes("key_proj_out"));
    let value = upload(g, &f.bytes("value_proj_out"));
    let nq = upload(g, &f.bytes("w_norm_query"));
    let nk = upload(g, &f.bytes("w_norm_key"));
    let nc = upload(g, &f.bytes("w_norm_conv"));
    let gated = g.alloc(t * c * 4).unwrap();
    // FP32: the conv reads this, and BF16 here costs the output 2.7%.
    let gated_normed = g.alloc(t * c * 4).unwrap();
    ops::ple_gate(
        g,
        k_gate,
        hidden,
        key,
        value,
        nq,
        nk,
        nc,
        gated,
        gated_normed,
        t as u32,
        h as u32,
        hc as u32,
        f.eps,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    // Per-token max|diff|. Kept because it is what localized the one real
    // failure here: a fixture that dropped the batch dimension made token 0
    // match and every later token diverge, which reads exactly like a
    // time-dependence bug in the kernel and is not one.
    {
        let got = dl_f32(g, gated, t * c);
        let want = f.f32s("gated");
        let worst = (0..t)
            .map(|tok| {
                (0..c)
                    .map(|i| (got[tok * c + i] - want[tok * c + i]).abs())
                    .fold(0.0f32, f32::max)
            })
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        println!("  worst token: {} (max|diff| {:.3e})", worst.0, worst.1);
    }
    compare("gated", &dl_f32(g, gated, t * c), &f.f32s("gated"));
    compare(
        "gated_normed",
        &dl_f32(g, gated_normed, t * c),
        &f.f32s("gated_normed"),
    );

    // ── conv + silu + add ──
    // State is zero at a sequence start, which is what the reference's
    // `F.pad(..., (state_len, 0))` amounts to on the first call.
    let state_len = (f.k_size - 1) * f.dilation;
    let state = upload(g, &vec![0u8; state_len * c * 4]);
    let w_conv = upload(g, &f.bytes("w_conv1d"));
    let out = g.alloc(t * c * 4).unwrap();
    ops::ple_conv(
        g,
        k_conv,
        gated_normed,
        gated,
        w_conv,
        state,
        out,
        t as u32,
        c as u32,
        f.k_size as u32,
        f.dilation as u32,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    compare("output", &dl_f32(g, out, t * c), &f.f32s("output"));

    // ── the dilation is load-bearing: prove a wrong one fails ──
    // Running the same conv with dilation 1 reads timesteps t-3..t instead of
    // t-9,t-6,t-3,t. If that ALSO matched, this test would be proving nothing.
    let state2 = upload(g, &vec![0u8; state_len * c * 4]);
    let out_d1 = g.alloc(t * c * 4).unwrap();
    ops::ple_conv(
        g,
        k_conv,
        gated_normed,
        gated,
        w_conv,
        state2,
        out_d1,
        t as u32,
        c as u32,
        f.k_size as u32,
        1,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    let d1 = dl_f32(g, out_d1, t * c);
    let want = f.f32s("output");
    let worst = d1
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("  dilation=1 control: max|diff| vs reference = {worst:.4e}");
    assert!(
        worst > 1e-3,
        "dilation is not affecting the result — the conv is not reading the \
         taps this test claims it is"
    );
}

/// The SEGMENTED row cache and the gather, isolated from everything else.
///
/// This is the arm that separates "PLE's math is wrong" from "PLE is reading
/// the wrong rows" — and the second failure is invisible downstream, because
/// every row in a 320M-row embedding table is a plausible embedding.
///
/// Needs the checkpoint (it reads real rows off NVMe):
/// ```text
/// ATLAS_PLE_TEST_DATA=/tank/atlas-testdata/qwen4exp_ple \
/// ATLAS_PLE_CKPT=/path/to/snapshot \
///   cargo test -p spark-model --release ple_gather -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn ple_gather_reads_the_right_rows() {
    let f = Fx::load();
    let snap = match std::env::var("ATLAS_PLE_CKPT") {
        Ok(s) => s,
        Err(_) => {
            println!("ATLAS_PLE_CKPT unset — skipping");
            return;
        }
    };
    let head_dim = 160usize;
    let ids: Vec<u64> = f
        .bytes("ids")
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let want = f.f32s("embeddings");
    println!("ids={} want={} floats", ids.len(), want.len());

    // Rebuild the segmented cache exactly as the loader does, straight from
    // the safetensors header — no model load.
    let (path, bases, rows_per) =
        crate::weight_loader::qwen4_exp::ple_shard_layout(&snap).expect("PLE shard layout");
    println!(
        "shards={} rows_per={rows_per} file={}",
        bases.len(),
        path.display()
    );
    // CUDA FIRST: the cache's arena is pinned, GPU-addressable memory, so it
    // needs a live context. The loader gets one for free; a bare test does not.
    let set = atlas_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4").unwrap();
    let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).unwrap();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let k = g.kernel("embed_from_argmax", "batched_embed").unwrap();

    let mut cache = spark_storage::NgramRowCache::open_segmented(
        &path,
        bases,
        rows_per,
        None,
        head_dim * 2,
        ids.len().next_power_of_two(),
    )
    .expect("segmented cache");

    let mut slots = Vec::new();
    cache.resolve(&ids, &mut slots).expect("resolve");
    assert_eq!(slots.len(), ids.len());

    let slot_bytes: Vec<u8> = slots.iter().flat_map(|s| s.to_le_bytes()).collect();
    let slots_dev = upload(g, &slot_bytes);
    let out = g.alloc(ids.len() * head_dim * 2).unwrap();
    ops::batched_embed(
        g,
        k,
        slots_dev,
        DevicePtr(cache.table_dev_va().unwrap()),
        out,
        ids.len() as u32,
        head_dim as u32,
        stream,
    )
    .unwrap();
    g.synchronize(stream).unwrap();
    cache.end_batch();

    let got = dl_bf16_local(g, out, ids.len() * head_dim);
    let (hits, misses, evictions) = cache.stats();
    println!("cache: {hits} hits, {misses} misses, {evictions} evictions");
    let nan = got.iter().filter(|v| !v.is_finite()).count();
    println!("non-finite gathered values: {nan} / {}", got.len());
    compare("embeddings", &got, &want);
}

fn dl_bf16_local(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut raw = vec![0u8; n * 2];
    g.copy_d2h(p, &mut raw).unwrap();
    raw.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}
