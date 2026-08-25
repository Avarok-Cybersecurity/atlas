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
