// SPDX-License-Identifier: AGPL-3.0-only

//! Pack-time fusion of the four RAW GDN input-projection LoRA tensors into
//! the two runtime modules, mirroring the base-weight loader's own fusions:
//!
//! - `in_proj_qkv` [2k+v, h] and `in_proj_z` [v, h] are row-CONCATENATED at
//!   load into one sequential `[Q|K|V|Z]` weight (`gpu_concat_rows`), so their
//!   deltas fuse into ONE block-diagonal rank-2r pair over the full qkvz
//!   width: A = [A_qkv; A_z] (2r rows), B[0..qkv, 0..r] = B_qkv,
//!   B[qkv.., r..2r] = B_z. One `apply_lora_delta` on the contiguous
//!   deinterleaved buffer then reproduces both deltas exactly.
//!
//! - `in_proj_b` / `in_proj_a` [nv, h] are INTERLEAVED at load per key-head
//!   group (`interleave_ba`: group g of vpg heads → vpg beta rows then vpg
//!   alpha rows), so their fused B rows are PERMUTED into that layout:
//!   A = [A_b; A_a], B[dst_beta(h), 0..r] = B_b[h], B[dst_alpha(h), r..2r] =
//!   B_a[h]. The raw fused delta is then in the exact row order the BA-gates
//!   kernels consume pre-transform.
//!
//! An adapter targeting only SOME of the four raws still fuses correctly:
//! the missing half's rows/cols stay zero (zeros contribute nothing).
//!
//! The fused rank is ALWAYS `2 * r` (even when one half is absent) so the
//! pool's `max_rank` requirement is a function of the adapter config alone.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use atlas_core::config::{ModelConfig, PeftAdapterConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::target::GdnProj;
use super::types::LoraModule;

/// PEFT `target_modules` vocabulary of the raw GDN input projections
/// (what `validate_peft_config` and the audit reverse-check match against).
pub(crate) const RAW_PEFT_NAMES: [&str; 4] =
    ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"];

/// Audited raw-tensor coverage: `(layer, proj) -> [a_key, b_key]`.
pub(crate) type GdnMap = BTreeMap<(usize, GdnProj), [Option<String>; 2]>;

/// Fused host-side pairs ready for the pool pack:
/// `(layer, fused module) -> (A bytes [2r, h], B bytes [out, 2r], 2r)`.
/// PEFT-layout (unpadded); `pack_slot` pads B's row stride to `max_rank`
/// exactly as it does for a native tensor of rank 2r.
pub(crate) type FusedGdnMap = BTreeMap<(usize, LoraModule), (Vec<u8>, Vec<u8>, usize)>;

const BF16: usize = 2;

fn read_host(store: &WeightStore, key: &str, bytes: usize, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
    let t = store.get(key)?;
    let mut host = vec![0u8; bytes];
    gpu.copy_d2h(t.ptr, &mut host)
        .with_context(|| format!("gdn_fuse: d2h of '{key}'"))?;
    Ok(host)
}

/// Copy `src` ([rows, src_cols] BF16, col offset `dst_col_off`) into `dst`
/// ([?, dst_cols] BF16) at row offset `dst_row_of(row)`.
fn scatter_rows(
    dst: &mut [u8],
    dst_cols: usize,
    src: &[u8],
    rows: usize,
    src_cols: usize,
    dst_col_off: usize,
    dst_row_of: impl Fn(usize) -> usize,
) {
    for r in 0..rows {
        let d = (dst_row_of(r) * dst_cols + dst_col_off) * BF16;
        let s = r * src_cols * BF16;
        dst[d..d + src_cols * BF16].copy_from_slice(&src[s..s + src_cols * BF16]);
    }
}

/// Build the fused host pairs for one adapter from its audited raw coverage.
pub(crate) fn fuse_gdn_pairs(
    store: &WeightStore,
    gdn: &GdnMap,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<FusedGdnMap> {
    let mut out: FusedGdnMap = BTreeMap::new();
    if gdn.is_empty() {
        return Ok(out);
    }
    let r = peft.r;
    let fused_r = 2 * r;
    let h = cfg.hidden_size;
    let qkv = GdnProj::Qkv.out_dim(cfg);
    let qkvz = cfg.ssm_qkvz_size();
    let ba = cfg.ssm_ba_size();
    let nv = cfg.linear_num_value_heads;
    let nk = cfg.linear_num_key_heads;
    let vpg = nv / nk;

    // Group raw entries per (layer, fused module).
    let layers: std::collections::BTreeSet<usize> = gdn.keys().map(|(l, _)| *l).collect();
    for layer in layers {
        // ── GdnQkvz: [A_qkv; A_z] block-diagonal over [0..qkv | qkv..qkvz] ──
        let has_qkv = gdn.contains_key(&(layer, GdnProj::Qkv));
        let has_z = gdn.contains_key(&(layer, GdnProj::Z));
        if has_qkv || has_z {
            let mut a_host = vec![0u8; fused_r * h * BF16];
            let mut b_host = vec![0u8; qkvz * fused_r * BF16];
            if let Some([Some(a_key), Some(b_key)]) = gdn.get(&(layer, GdnProj::Qkv)) {
                let a = read_host(store, a_key, r * h * BF16, gpu)?;
                a_host[..a.len()].copy_from_slice(&a); // rows 0..r
                let b = read_host(store, b_key, qkv * r * BF16, gpu)?;
                scatter_rows(&mut b_host, fused_r, &b, qkv, r, 0, |row| row);
            }
            if let Some([Some(a_key), Some(b_key)]) = gdn.get(&(layer, GdnProj::Z)) {
                let a = read_host(store, a_key, r * h * BF16, gpu)?;
                a_host[r * h * BF16..].copy_from_slice(&a); // rows r..2r
                let b = read_host(store, b_key, (qkvz - qkv) * r * BF16, gpu)?;
                scatter_rows(&mut b_host, fused_r, &b, qkvz - qkv, r, r, |row| qkv + row);
            }
            out.insert((layer, LoraModule::GdnQkvz), (a_host, b_host, fused_r));
        }

        // ── GdnBa: [A_b; A_a] with B rows permuted to the interleave ──
        let has_a = gdn.contains_key(&(layer, GdnProj::A));
        let has_b = gdn.contains_key(&(layer, GdnProj::B));
        if has_a || has_b {
            let mut a_host = vec![0u8; fused_r * h * BF16];
            let mut b_host = vec![0u8; ba * fused_r * BF16];
            // interleave_ba ground truth (weight_map/fp8_lut.rs): group g of
            // vpg value heads → dst rows g*2vpg+{0..vpg} = BETA heads
            // g*vpg+{0..vpg}, dst rows g*2vpg+vpg+{0..vpg} = ALPHA heads.
            let dst_beta = |head: usize| (head / vpg) * 2 * vpg + (head % vpg);
            let dst_alpha = |head: usize| (head / vpg) * 2 * vpg + vpg + (head % vpg);
            if let Some([Some(a_key), Some(b_key)]) = gdn.get(&(layer, GdnProj::B)) {
                let a = read_host(store, a_key, r * h * BF16, gpu)?;
                a_host[..a.len()].copy_from_slice(&a); // rows 0..r (beta)
                let b = read_host(store, b_key, nv * r * BF16, gpu)?;
                scatter_rows(&mut b_host, fused_r, &b, nv, r, 0, dst_beta);
            }
            if let Some([Some(a_key), Some(b_key)]) = gdn.get(&(layer, GdnProj::A)) {
                let a = read_host(store, a_key, r * h * BF16, gpu)?;
                a_host[r * h * BF16..].copy_from_slice(&a); // rows r..2r (alpha)
                let b = read_host(store, b_key, nv * r * BF16, gpu)?;
                scatter_rows(&mut b_host, fused_r, &b, nv, r, r, dst_alpha);
            }
            out.insert((layer, LoraModule::GdnBa), (a_host, b_host, fused_r));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    use super::*;
    use crate::layers::ops::lora_delta::{
        LoraKernels, LoraPair, apply_lora_delta, compute_lora_delta_raw,
    };
    use crate::weight_map::DenseWeight;

    fn bf16_bits(v: f32) -> u16 {
        ((v.to_bits() + 0x8000) >> 16) as u16
    }
    fn bf16_val(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// GPU parity: the FUSED block-diagonal pairs built by `fuse_gdn_pairs`,
    /// applied through the very kernels the serving path uses, must reproduce
    /// the per-projection reference deltas: `scale*B_qkv(A_qkv x)` on rows
    /// 0..qkv, `scale*B_z(A_z x)` on the z rows, and the interleaved
    /// beta/alpha layout for the BA delta. This is the exact-replay parity
    /// harness the in_proj family was held back for.
    #[test]
    #[ignore] // GPU (CI is CPU-only): cargo test -p spark-model gdn_fuse -- --ignored
    fn fused_gdn_pairs_match_reference_on_gpu() {
        let cfg = crate::lora::test_support::cfg();
        let gpu = spark_runtime::cuda_backend::AtlasCudaBackend::new(
            0,
            &atlas_kernels::ptx_modules(),
        )
        .expect("CUDA backend");
        let g: &dyn spark_runtime::gpu::GpuBackend = &gpu;
        let kernels = LoraKernels::new(g).unwrap();

        let h = cfg.hidden_size;
        let qkv = GdnProj::Qkv.out_dim(&cfg);
        let qkvz = cfg.ssm_qkvz_size();
        let ba = cfg.ssm_ba_size();
        let nv = cfg.linear_num_value_heads;
        let vpg = nv / cfg.linear_num_key_heads;
        let r = 4usize;
        let fused_r = 2 * r;
        let scale = 32.0f32 / r as f32;

        // Deterministic pseudo-random BF16 host tensors.
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let mk = |rows: usize, cols: usize, rnd: &mut dyn FnMut() -> f32| -> Vec<u16> {
            (0..rows * cols).map(|_| bf16_bits(rnd())).collect()
        };
        let a_qkv = mk(r, h, &mut rnd);
        let b_qkv = mk(qkv, r, &mut rnd);
        let a_z = mk(r, h, &mut rnd);
        let b_z = mk(qkvz - qkv, r, &mut rnd);
        let a_a = mk(r, h, &mut rnd);
        let b_a = mk(nv, r, &mut rnd);
        let a_b = mk(r, h, &mut rnd);
        let b_b = mk(nv, r, &mut rnd);
        let x = mk(1, h, &mut rnd);

        // Synthetic device WeightStore holding the eight raw tensors.
        let up = |host: &[u16]| -> spark_runtime::gpu::DevicePtr {
            let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
            let p = g.alloc(bytes.len()).unwrap();
            g.copy_h2d(&bytes, p).unwrap();
            p
        };
        let mut map = HashMap::new();
        let mut put = |name: &str, host: &[u16], shape: Vec<usize>| {
            map.insert(
                name.to_string(),
                WeightTensor {
                    ptr: up(host),
                    shape,
                    dtype: WeightDtype::BF16,
                },
            );
        };
        put("aq", &a_qkv, vec![r, h]);
        put("bq", &b_qkv, vec![qkv, r]);
        put("az", &a_z, vec![r, h]);
        put("bz", &b_z, vec![qkvz - qkv, r]);
        put("aa", &a_a, vec![r, h]);
        put("ba", &b_a, vec![nv, r]);
        put("ab", &a_b, vec![r, h]);
        put("bb", &b_b, vec![nv, r]);
        let store = WeightStore::from_map(map);

        let peft = atlas_core::config::PeftAdapterConfig {
            r,
            lora_alpha: 32.0,
            target_modules: Vec::new(),
            target_modules_pattern: None,
            use_rslora: false,
            layers_to_transform: None,
            trainable_token_indices: Vec::new(),
            modules_to_save: Vec::new(),
            lora_embedding: false,
        };
        let mut gdn: GdnMap = BTreeMap::new();
        let e = |a: &str, b: &str| [Some(a.to_string()), Some(b.to_string())];
        gdn.insert((0, GdnProj::Qkv), e("aq", "bq"));
        gdn.insert((0, GdnProj::Z), e("az", "bz"));
        gdn.insert((0, GdnProj::A), e("aa", "ba"));
        gdn.insert((0, GdnProj::B), e("ab", "bb"));
        let fused = fuse_gdn_pairs(&store, &gdn, &peft, &cfg, g).unwrap();

        // Upload fused pairs (max_rank = fused_r: no extra padding to test).
        let mk_pair = |m: LoraModule, out: usize| -> LoraPair {
            let (a_host, b_host, fr) = fused.get(&(0, m)).unwrap();
            assert_eq!(*fr, fused_r);
            let a_ptr = g.alloc(a_host.len()).unwrap();
            g.copy_h2d(a_host, a_ptr).unwrap();
            let b_ptr = g.alloc(b_host.len()).unwrap();
            g.copy_h2d(b_host, b_ptr).unwrap();
            LoraPair {
                a: DenseWeight { weight: a_ptr },
                b: DenseWeight { weight: b_ptr },
                rank: fused_r as u32,
                k_in: h as u32,
                n_out: out as u32,
                scale,
                max_rank: fused_r as u32,
            }
        };
        let qkvz_pair = mk_pair(LoraModule::GdnQkvz, qkvz);
        let ba_pair = mk_pair(LoraModule::GdnBa, ba);

        // CPU reference per raw projection, BF16-FAITHFUL at the stage
        // boundaries the engine has: the shrink output xa is STORED BF16, and
        // the expand output delta is STORED BF16 before the scale fold
        // (accumulation inside each GEMV is f32).
        let refd = |a: &[u16], b: &[u16], out: usize| -> Vec<f32> {
            let xa: Vec<f32> = (0..r)
                .map(|i| {
                    let acc: f32 = (0..h)
                        .map(|k| bf16_val(a[i * h + k]) * bf16_val(x[k]))
                        .sum();
                    bf16_val(bf16_bits(acc))
                })
                .collect();
            (0..out)
                .map(|o| {
                    let d: f32 = (0..r).map(|i| bf16_val(b[o * r + i]) * xa[i]).sum();
                    scale * bf16_val(bf16_bits(d))
                })
                .collect()
        };
        let want_qkv = refd(&a_qkv, &b_qkv, qkv);
        let want_z = refd(&a_z, &b_z, qkvz - qkv);
        let want_a = refd(&a_a, &b_a, nv);
        let want_b = refd(&a_b, &b_b, nv);

        // CONTROL: a plain (unfused) rank-r pair through the same kernels —
        // separates "fusion is wrong" from "engine-vs-reference numerics".
        let x_dev = up(&x);
        {
            let a_ptr = up(&a_qkv);
            // Pad B [qkv, r] -> [qkv, fused_r] row stride like the pool does.
            let mut b_pad = vec![0u16; qkv * fused_r];
            for o in 0..qkv {
                for i in 0..r {
                    b_pad[o * fused_r + i] = b_qkv[o * r + i];
                }
            }
            let b_ptr = up(&b_pad);
            // Pad A rows r -> fused_r with zeros (contiguous head).
            let mut a_pad = vec![0u16; fused_r * h];
            a_pad[..r * h].copy_from_slice(&a_qkv);
            let a_ptr2 = up(&a_pad);
            let _ = a_ptr;
            let plain = LoraPair {
                a: DenseWeight { weight: a_ptr2 },
                b: DenseWeight { weight: b_ptr },
                rank: r as u32,
                k_in: h as u32,
                n_out: qkv as u32,
                scale,
                max_rank: fused_r as u32,
            };
            let out_c = g.alloc(qkv * 2).unwrap();
            g.memset(out_c, 0, qkv * 2).unwrap();
            let xa_c = g.alloc(fused_r * 2).unwrap();
            let d_c = g.alloc(qkv * 2).unwrap();
            apply_lora_delta(
                g,
                &kernels,
                &plain,
                x_dev,
                out_c,
                1,
                xa_c,
                d_c,
                g.default_stream(),
            )
            .unwrap();
            g.synchronize(g.default_stream()).unwrap();
            let mut gotc = vec![0u8; qkv * 2];
            g.copy_d2h(out_c, &mut gotc).unwrap();
            let gc = |i: usize| bf16_val(u16::from_le_bytes([gotc[i * 2], gotc[i * 2 + 1]]));
            let mut worst = 0f32;
            for o in 0..qkv {
                worst = worst.max((gc(o) - want_qkv[o]).abs());
            }
            println!("CONTROL plain pair worst abs err = {worst}");
        }

        // Engine: apply the fused qkvz delta into a ZEROED [1, qkvz] target.
        let out_dev = g.alloc(qkvz * 2).unwrap();
        g.memset(out_dev, 0, qkvz * 2).unwrap();
        let xa_s = g.alloc(fused_r * 2).unwrap();
        let d_s = g.alloc(qkvz * 2).unwrap();
        apply_lora_delta(
            g,
            &kernels,
            &qkvz_pair,
            x_dev,
            out_dev,
            1,
            xa_s,
            d_s,
            g.default_stream(),
        )
        .unwrap();
        g.synchronize(g.default_stream()).unwrap();
        let mut got = vec![0u8; qkvz * 2];
        g.copy_d2h(out_dev, &mut got).unwrap();
        let got_f = |i: usize| bf16_val(u16::from_le_bytes([got[i * 2], got[i * 2 + 1]]));
        let tol = 0.02f32;
        for o in 0..qkv {
            assert!(
                (got_f(o) - want_qkv[o]).abs() <= tol + 0.05 * want_qkv[o].abs(),
                "qkv row {o}: got {} want {}",
                got_f(o),
                want_qkv[o]
            );
        }
        for o in 0..(qkvz - qkv) {
            assert!(
                (got_f(qkv + o) - want_z[o]).abs() <= tol + 0.05 * want_z[o].abs(),
                "z row {o}: got {} want {}",
                got_f(qkv + o),
                want_z[o]
            );
        }

        // Engine: RAW ba delta (unscaled) then scale in the comparison —
        // exactly how the gates kernels consume it.
        let ba_dev = g.alloc(ba * 2).unwrap();
        compute_lora_delta_raw(
            g,
            &kernels,
            &ba_pair,
            x_dev,
            ba_dev,
            1,
            xa_s,
            g.default_stream(),
        )
        .unwrap();
        g.synchronize(g.default_stream()).unwrap();
        let mut got_ba = vec![0u8; ba * 2];
        g.copy_d2h(ba_dev, &mut got_ba).unwrap();
        let got_ba_f =
            |i: usize| scale * bf16_val(u16::from_le_bytes([got_ba[i * 2], got_ba[i * 2 + 1]]));
        for head in 0..nv {
            let dst_b = (head / vpg) * 2 * vpg + (head % vpg);
            let dst_a = (head / vpg) * 2 * vpg + vpg + (head % vpg);
            assert!(
                (got_ba_f(dst_b) - want_b[head]).abs() <= tol + 0.05 * want_b[head].abs(),
                "beta head {head} (row {dst_b}): got {} want {}",
                got_ba_f(dst_b),
                want_b[head]
            );
            assert!(
                (got_ba_f(dst_a) - want_a[head]).abs() <= tol + 0.05 * want_a[head].abs(),
                "alpha head {head} (row {dst_a}): got {} want {}",
                got_ba_f(dst_a),
                want_a[head]
            );
        }
        println!("fused GDN parity: qkv+z rows and interleaved beta/alpha all match");
    }

    #[test]
    fn interleave_row_maps_match_loader() {
        // qwen3.8-27B: nk=16, nv=48, vpg=3. Loader layout: group g → rows
        // g*6+{0,1,2} beta heads g*3+{0,1,2}, rows g*6+{3,4,5} alpha heads.
        let vpg = 3usize;
        let dst_beta = |head: usize| (head / vpg) * 2 * vpg + (head % vpg);
        let dst_alpha = |head: usize| (head / vpg) * 2 * vpg + vpg + (head % vpg);
        assert_eq!(dst_beta(0), 0);
        assert_eq!(dst_beta(2), 2);
        assert_eq!(dst_alpha(0), 3);
        assert_eq!(dst_alpha(2), 5);
        assert_eq!(dst_beta(3), 6);
        assert_eq!(dst_alpha(3), 9);
        assert_eq!(dst_beta(47), 15 * 6 + 2);
        assert_eq!(dst_alpha(47), 15 * 6 + 5);
        // Bijection over 2*nv rows.
        let nv = 48;
        let mut seen = vec![false; 2 * nv];
        for hd in 0..nv {
            for d in [dst_beta(hd), dst_alpha(hd)] {
                assert!(!seen[d], "row {d} hit twice");
                seen[d] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }
}
