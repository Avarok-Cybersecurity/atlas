// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//! The production engram against its oracles:
//!   * the hasher against the reference `NgramHashState`, exact, all regimes;
//!   * the GPU projection + gate against the golden's `engram_out` captures
//!     (tiny graph, rows and weights regenerated as the reference does), within bf16;
//!   * real-table rows through the device Q2_K dequant, bit-identical to the
//!     CPU decoder (ignored, needs the shards).

use std::sync::Arc;

use super::*;
use crate::layers::deepseek_v41_ref::engram::{
    EngramTables, NgramHashState, embed_rows, regen_bf16_matrix, regen_bf16_qk, regen_table,
};
use crate::layers::deepseek_v41_ref::testutil::*;
use crate::layers::deepseek_v41_ref::{Golden, fixed_int};

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    (b.wrapping_add(0x7FFF + lsb) >> 16) as u16
}

/// The production tables from the fixture's, flattened the way the GGUF
/// metadata is.
fn tables_from_ref(t: &EngramTables) -> Arc<EngramHashTables> {
    let flat_u64 = |v: &[i64]| -> Vec<u64> { v.iter().map(|&x| x as u64).collect() };
    let mults: Vec<i64> = t.multipliers.iter().flatten().copied().collect();
    let primes: Vec<i64> = t.primes.iter().flatten().flatten().copied().collect();
    let offs: Vec<i64> = t.offsets.iter().flatten().copied().collect();
    // the fixture's pad_id is already in the compressed space; find a token that maps to it
    let pad_tok = t
        .token_map
        .iter()
        .position(|&m| m == t.pad_id)
        .expect("a token mapping to pad") as u32;
    Arc::new(
        EngramHashTables::from_flat(
            t.layer_ids.clone(),
            t.max_ngram,
            t.n_heads,
            pad_tok,
            t.token_map.clone(),
            &flat_u64(&mults),
            &flat_u64(&primes),
            &flat_u64(&offs),
        )
        .unwrap(),
    )
}

struct Fx {
    g: Golden,
    t: EngramTables,
    regimes: Vec<String>,
    ids: Vec<u32>,
    prefill: usize,
    max_seq: usize,
    dim: usize,
    hc: usize,
    eps: f32,
}

fn fx() -> Fx {
    let g = Golden::load();
    let t = EngramTables::from_golden(&g);
    let regimes = g.regimes();
    let vocab = g.fixture_u64("vocab_size");
    let prefill = g.fixture_u64("prefill_len") as usize;
    let ids: Vec<u32> = (0..(prefill + 2) as u64)
        .map(|i| fixed_int("input_ids", i, vocab) as u32)
        .collect();
    let max_seq = g.fixture_u64("max_seq_len") as usize;
    let dim = g.fixture_u64("dim") as usize;
    let hc = g.fixture_u64("hc_mult") as usize;
    let eps = g.fixture_f64("norm_eps") as f32;
    Fx {
        g,
        t,
        regimes,
        ids,
        prefill,
        max_seq,
        dim,
        hc,
        eps,
    }
}

/// `(regime, start, len)` for prefill and the two decode steps.
fn regime_spans(f: &Fx) -> Vec<(String, usize, usize)> {
    let p = f.prefill;
    vec![
        (f.regimes[0].clone(), 0, p),
        (f.regimes[1].clone(), p, 1),
        (f.regimes[2].clone(), p + 1, 1),
    ]
}

#[test]
fn hasher_matches_the_reference_exactly() {
    let f = fx();
    let tables = tables_from_ref(&f.t);
    assert_eq!(tables.pad_id, f.t.pad_id);
    assert_eq!(tables.n_hash_cols(), f.t.n_hash_cols());
    let mut ours = EngramHasher::new(tables, f.max_seq);
    let mut theirs = NgramHashState::new(&f.t, 1, f.max_seq);
    for (r, start, len) in regime_spans(&f) {
        let got = ours.hash(&f.ids[start..start + len], start).unwrap();
        let ids_i64: Vec<i64> = f.ids[start..start + len]
            .iter()
            .map(|&x| x as i64)
            .collect();
        let want = theirs.forward(&ids_i64, len, start);
        assert_eq!(got, want, "{r}: hash ids differ from the reference");
        // and the golden itself, where the reference was already checked
        let gt = f.g.tensor(&r, "engram_hashes");
        let got_f: Vec<f64> = got.iter().map(|&v| v as f64).collect();
        check_capture(&format!("{r}.engram_hashes"), &got_f, &gt, 0.0, 0.0);
    }
}

#[test]
fn layer_row_ids_are_token_major() {
    let f = fx();
    let tables = tables_from_ref(&f.t);
    let (nl, cols) = (tables.n_layers(), tables.n_hash_cols());
    let hasher = EngramHasher::new(tables, f.max_seq);
    let tokens = 3;
    let hashes: Vec<i64> = (0..tokens * nl * cols).map(|i| i as i64).collect();
    for li in 0..nl {
        let ids = hasher.layer_row_ids(&hashes, tokens, li);
        for s in 0..tokens {
            for c in 0..cols {
                assert_eq!(ids[s * cols + c], ((s * nl + li) * cols + c) as u64);
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn backend() -> spark_runtime::cuda_backend::AtlasCudaBackend {
    let set = atlas_kernels::ptx_for_exact_target("deepseek-v4-flash", "nvfp4")
        .expect("deepseek-v4-flash/nvfp4 not in this build");
    spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &set.modules).expect("CUDA backend")
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
fn gpu_engram_matches_the_golden_within_bf16() {
    use spark_runtime::gpu::GpuBackend;
    let f = fx();
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let tables = tables_from_ref(&f.t);
    let cols = tables.n_hash_cols();
    let hd = f.t.head_dim;
    let in_f = cols * hd;
    let out_f = f.dim * (f.hc + 1);
    let mut eng = EngramV41::new(g, f.dim, f.hc, hd, cols, f.eps, f.prefill + 2).unwrap();
    for (li, &lid) in f.t.layer_ids.iter().enumerate() {
        let w = regen_bf16_matrix(&format!("layers.{lid}.engram.wkv.weight"), out_f, in_f);
        let q = regen_bf16_qk(&format!("layers.{lid}.engram.q_weight"), f.hc * f.dim);
        let k = regen_bf16_qk(&format!("layers.{lid}.engram.k_weight"), f.hc * f.dim);
        let w_bytes: Vec<u8> = w.iter().flat_map(|&v| bf16_bits(v).to_le_bytes()).collect();
        let wkv = g.alloc(w_bytes.len()).unwrap();
        g.copy_h2d(&w_bytes, wkv).unwrap();
        let qk = EngramV41::upload_qk(g, &q, &k).unwrap();
        eng.add_layer(EngramLayerWeights {
            layer: lid,
            wkv,
            qk,
        });
        let _ = li;
    }
    let mut hasher = EngramHasher::new(tables.clone(), f.max_seq);
    for (r, start, len) in regime_spans(&f) {
        let hashes = hasher.hash(&f.ids[start..start + len], start).unwrap();
        for (li, &lid) in f.t.layer_ids.iter().enumerate() {
            let table = regen_table(&f.t, li);
            let ids: Vec<i64> = hasher
                .layer_row_ids(&hashes, len, li)
                .iter()
                .map(|&v| v as i64)
                .collect();
            let emb = embed_rows(&table, hd, &ids);
            let rows: Vec<u16> = emb.iter().map(|&v| bf16_bits(v)).collect();
            eng.rows_from_bf16(g, &rows, len * cols).unwrap();
            let xin = f.g.tensor(&r, &format!("L{lid}.engram_in"));
            assert_eq!(xin.stride, 1, "engram_in must be stored at full resolution");
            let x: Vec<f32> = xin.data.iter().map(|&v| v as f32).collect();
            assert_eq!(x.len(), len * f.hc * f.dim);
            let x_bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            let streams = g.alloc(x_bytes.len()).unwrap();
            g.copy_h2d(&x_bytes, streams).unwrap();
            eng.apply(g, lid, streams, len, stream).unwrap();
            g.synchronize(stream).unwrap();
            let mut out = vec![0u8; x_bytes.len()];
            g.copy_d2h(streams, &mut out).unwrap();
            g.free(streams).unwrap();
            let got: Vec<f64> = out
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect();
            let gt = f.g.tensor(&r, &format!("L{lid}.engram_out"));
            check_capture(
                &format!("{r}.L{lid}.engram_out (gpu)"),
                &got,
                &gt,
                bf16_tol(&gt),
                BF16_CK_REL,
            );
            println!(
                "  {r} L{lid}: {} values within bf16 of engram_out",
                got.len()
            );
        }
    }
    eng.free(g).unwrap();
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA GB10 + the on-disk DeepSeek-V4.1-Flash Q2_K shards"]
fn gpu_engram_rows_from_the_real_tables_match_the_cpu_decoder() {
    use spark_runtime::gpu::GpuBackend;
    use spark_runtime::weights::dequant_cpu::{GgmlType, dequant_to_bf16};
    use spark_runtime::weights::expert_stream::{EngramRowReader, ShardFiles};
    let files = Arc::new(
        ShardFiles::open_dir(std::path::Path::new("/home/rstesiak/models/dsv41-q2k"))
            .expect("shards"),
    );
    let rd = EngramRowReader::new(files).unwrap();
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let stream = g.default_stream();
    let cols = 24usize;
    let eng = EngramV41::new(g, 5120, 4, 256, cols, 1e-6, 4).unwrap();
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let q2k = GgmlType::from_id(10, 128).unwrap();
    for t in rd.tables() {
        let n_rows = 2 * cols;
        let ids: Vec<u64> = (0..n_rows)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x % t.rows as u64
            })
            .collect();
        let mut raw = vec![0u8; n_rows * ENGRAM_ROW_BYTES];
        rd.read_rows(t.layer, &ids, &mut raw).unwrap();
        eng.rows_from_q2k(g, &raw, n_rows, stream).unwrap();
        g.synchronize(stream).unwrap();
        let mut got = vec![0u8; n_rows * 256 * 2];
        g.copy_d2h(eng.rows_ptr(), &mut got).unwrap();
        let got: Vec<u16> = got
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let mut want = vec![0u16; n_rows * 256];
        dequant_to_bf16(q2k, &raw, n_rows * 256, &mut want).unwrap();
        assert_eq!(
            got, want,
            "engram layer {}: GPU rows differ from the CPU decoder",
            t.layer
        );
        println!(
            "  layer {:>2}: {n_rows} real rows through the device Q2_K dequant, bit-identical to the CPU decoder",
            t.layer
        );
    }
    eng.free(g).unwrap();
}
