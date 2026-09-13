// SPDX-License-Identifier: AGPL-3.0-only

//! Applying `super::tp::KdaTpPlan` — the shard COPIES, at last.
//!
//! # Why this is an adapter and not a change to the binder
//!
//! `super::binding::bind_kda_weights` is proven: exact tensor set, exact dtypes, exact
//! shapes, gated numerically on real weights. Teaching it about TP would mean rewriting its
//! validation to check on-disk (full) shapes against a per-rank config — i.e. editing the one
//! piece of this lane that has never been wrong.
//!
//! So the slicing happens *upstream*. `KdaShardedSource` wraps any `KdaTensorSource` and
//! hands the binder bytes that are **already this rank's**, with **local** shapes. The binder
//! then validates them against the local config exactly as it always has and cannot tell the
//! difference — which is the point: **TP=1 and TP=2 run the same binder code path**, so there
//! is no "works at TP=1, silently differs at TP=2" seam for a bug to live in.
//!
//! # 🪤 Two slicing shapes, and the wrong one is not a shape error
//!
//! * **Row slicing** (`HeadRows` / `ChannelRows`) is a contiguous byte range: this rank's rows
//!   sit next to each other on disk.
//! * **Column slicing** (`ChannelCols`, i.e. `o_proj` alone) is a STRIDED gather — every row
//!   contributes its own middle slice. Row-slicing `o_proj` instead yields a well-formed
//!   `[hidden/tp, heads*head_dim]` tensor of real numbers and a wrong output.
//!
//! Both produce the same LOCAL element count at `tp_size = 2` when the tensor is square-ish,
//! so a size check cannot separate them. The test below separates them by value.

use anyhow::{Result, bail};

use super::binding::{KdaTensorSource, RawTensor};
use super::tp::{KdaShard, KdaTensorPlan, KdaTpPlan};

/// Plan name for a checkpoint tensor name: `self_attn.q_proj.weight` → `q_proj`.
///
/// 🪤 `A_log` and `dt_bias` carry no `.weight` suffix; stripping unconditionally would
/// miss them.
fn plan_name(checkpoint_name: &str) -> &str {
    let n = checkpoint_name
        .strip_prefix("self_attn.")
        .unwrap_or(checkpoint_name);
    n.strip_suffix(".weight").unwrap_or(n)
}

/// This rank's bytes for one tensor, given its plan entry.
///
/// `full` is the whole on-disk tensor. Returns a fresh buffer because a column slice is not
/// contiguous and cannot be borrowed.
pub fn shard_bytes(plan: &KdaTensorPlan, full: &[u8]) -> Result<Vec<u8>> {
    let e = plan.elem_bytes;
    let expect = plan.full_rows * plan.full_row_elems * e;
    if full.len() != expect {
        bail!(
            "{}: {} B on disk, the plan's full shape [{}, {}] x {e} B implies {expect} B",
            plan.name,
            full.len(),
            plan.full_rows,
            plan.full_row_elems
        );
    }
    Ok(match plan.kind {
        KdaShard::Replicated => full.to_vec(),
        // Contiguous: this rank's rows are adjacent on disk.
        KdaShard::HeadRows | KdaShard::ChannelRows => {
            let row = plan.full_row_elems * e;
            let start = plan.src_row_offset * row;
            full[start..start + plan.local_rows * row].to_vec()
        }
        // 🪤 STRIDED. Every row keeps only its own `[src_col_offset, +local_row_elems)` slice.
        KdaShard::ChannelCols => {
            let row = plan.full_row_elems * e;
            let lo = plan.src_col_offset * e;
            let width = plan.local_row_elems * e;
            let mut out = Vec::with_capacity(plan.local_rows * width);
            for r in 0..plan.local_rows {
                let base = r * row + lo;
                out.extend_from_slice(&full[base..base + width]);
            }
            out
        }
    })
}

/// The local shape the binder should see, in the on-disk rank order.
///
/// Mirrors the on-disk rank rather than the plan's flat `[rows, row_elems]` view: the conv
/// tensors are rank-3 `[dim, 1, kernel]` on disk and the binder validates that rank.
fn local_shape(plan: &KdaTensorPlan, disk_shape: &[usize]) -> Vec<usize> {
    let mut s = disk_shape.to_vec();
    match plan.kind {
        KdaShard::Replicated => {}
        KdaShard::HeadRows | KdaShard::ChannelRows => {
            if let Some(first) = s.first_mut() {
                *first = plan.local_rows;
            }
        }
        KdaShard::ChannelCols => {
            if let Some(last) = s.last_mut() {
                *last = plan.local_row_elems;
            }
        }
    }
    s
}

/// A [`KdaTensorSource`] that yields **this rank's slice** of every KDA tensor.
///
/// Wrap the full-checkpoint source in this and hand it to `bind_kda_weights` unchanged.
pub struct KdaShardedSource<'a> {
    inner: &'a dyn KdaTensorSource,
    plan: &'a KdaTpPlan,
    /// Sliced bytes, materialised up front: `KdaTensorSource::get` returns a borrow, so the
    /// buffers have to outlive the call.
    sliced: std::collections::BTreeMap<String, (super::binding::KdaDtype, Vec<usize>, Vec<u8>)>,
}

impl<'a> KdaShardedSource<'a> {
    pub fn new(inner: &'a dyn KdaTensorSource, plan: &'a KdaTpPlan) -> Result<Self> {
        let mut sliced = std::collections::BTreeMap::new();
        for name in inner.names() {
            // Only `self_attn.*` is a KDA tensor; the binder counts everything else as
            // `non_attn_seen` and never fetches it, so passing it through untouched keeps
            // that census honest.
            let Some(p) = plan.get(plan_name(&name)) else {
                continue;
            };
            let Some(t) = inner.get(&name) else { continue };
            sliced.insert(
                name.clone(),
                (t.dtype, local_shape(p, &t.shape), shard_bytes(p, t.bytes)?),
            );
        }
        Ok(Self {
            inner,
            plan,
            sliced,
        })
    }

    pub fn plan(&self) -> &KdaTpPlan {
        self.plan
    }
}

impl KdaTensorSource for KdaShardedSource<'_> {
    fn get(&self, name: &str) -> Option<RawTensor<'_>> {
        match self.sliced.get(name) {
            Some((dtype, shape, bytes)) => Some(RawTensor {
                dtype: *dtype,
                shape: shape.clone(),
                bytes,
            }),
            // Not a planned KDA tensor — pass through, so the binder's "unrecognised
            // self_attn tensor" refusal still fires on anything unexpected.
            None => self.inner.get(name),
        }
    }

    fn names(&self) -> Vec<String> {
        self.inner.names()
    }
}

#[cfg(test)]
mod tests {
    use super::super::binding::KdaDtype;
    use super::*;

    /// GLM-5.3's real KDA geometry at TP=2.
    fn plan(rank: usize) -> KdaTpPlan {
        KdaTpPlan::new(rank, 2, 4096, 128, 64, 4, 128).unwrap()
    }

    #[test]
    fn plan_names_strip_the_prefix_but_not_a_missing_suffix() {
        assert_eq!(plan_name("self_attn.q_proj.weight"), "q_proj");
        assert_eq!(plan_name("self_attn.q_conv1d.weight"), "q_conv1d");
        // 🪤 no `.weight` on these two.
        assert_eq!(plan_name("self_attn.A_log"), "A_log");
        assert_eq!(plan_name("self_attn.dt_bias"), "dt_bias");
        // Every plan entry must be reachable from some checkpoint name.
        for t in &plan(0).tensors {
            let candidates = [
                format!("self_attn.{}.weight", t.name),
                format!("self_attn.{}", t.name),
            ];
            assert!(
                candidates.iter().any(|c| plan_name(c) == t.name),
                "plan entry {} is unreachable from a checkpoint name",
                t.name
            );
        }
    }

    /// Row slicing is contiguous; the two ranks partition the tensor exactly.
    #[test]
    fn channel_rows_partition_the_tensor() {
        // GLM's real tensors are too big for a test. The smallest LEGAL stand-in: the plan
        // enforces `2 * local_heads * head_dim % 256 == 0` (the fused conv+L2 kernel hardcodes
        // 2 heads per 256-thread block), so heads=4 head_dim=64 at tp=2 is the floor.
        let p = KdaTpPlan::new(0, 2, 4, 64, 4, 2, 4).unwrap();
        let q = p.get("q_proj").unwrap().clone();
        assert_eq!(q.kind, KdaShard::ChannelRows);
        // [full_ch = 256, hidden = 4] BF16, byte value = row index.
        let full: Vec<u8> = (0..=255u8).flat_map(|r| [r; 8]).collect();

        let r0 = shard_bytes(&q, &full).unwrap();
        let mut q1 = q.clone();
        q1.src_row_offset = q.local_rows;
        let r1 = shard_bytes(&q1, &full).unwrap();

        assert_eq!(
            r0.len() + r1.len(),
            full.len(),
            "the two ranks partition it"
        );
        let mut rejoined = r0.clone();
        rejoined.extend_from_slice(&r1);
        assert_eq!(rejoined, full, "and rejoin to the original, in order");
        assert!(r0.iter().all(|b| *b < 128), "rank 0 keeps the low rows");
        assert!(r1.iter().all(|b| *b >= 128), "rank 1 keeps the high rows");
    }

    /// 🔴 The discriminating test. `o_proj` is column-sliced; a row slice of the same tensor
    /// has the SAME LENGTH and different contents, so only a value check separates them.
    #[test]
    fn o_proj_is_column_sliced_not_row_sliced() {
        let p = KdaTpPlan::new(1, 2, 4, 64, 4, 2, 4).unwrap();
        let o = p.get("o_proj").unwrap();
        assert_eq!(o.kind, KdaShard::ChannelCols);
        assert_eq!(o.full_rows, 4, "hidden");
        assert_eq!(o.full_row_elems, 256, "heads * head_dim");
        assert_eq!(o.local_rows, 4, "every row is kept");
        assert_eq!(o.local_row_elems, 128, "half the input dim");

        // [4, 256] BF16 where each element's low byte is its column index (mod 256).
        let full: Vec<u8> = (0..4)
            .flat_map(|_| (0..=255u8).flat_map(|c| [c, 0]))
            .collect();
        let got = shard_bytes(o, &full).unwrap();

        // rank 1 keeps columns 128..256 of EVERY row.
        let want: Vec<u8> = (0..4)
            .flat_map(|_| (128..=255u8).flat_map(|c| [c, 0]))
            .collect();
        assert_eq!(got, want);

        // A row slice of the same byte count would be rows 2..4 — same length, different
        // bytes. This is the failure a size check cannot catch.
        let row_sliced = &full[full.len() / 2..];
        assert_eq!(row_sliced.len(), got.len());
        assert_ne!(row_sliced, got.as_slice());
    }

    /// `o_norm` is `[head_dim]` — within-head, so it must survive TP untouched. A
    /// "shard everything that looks per-head" rule corrupts 256 B and nothing says so.
    #[test]
    fn o_norm_and_the_low_rank_down_projections_are_replicated() {
        let p = plan(1);
        for n in ["o_norm", "f_a_proj", "g_a_proj"] {
            let t = p.get(n).unwrap();
            assert_eq!(t.kind, KdaShard::Replicated, "{n}");
            let full = vec![0xABu8; t.full_bytes()];
            assert_eq!(shard_bytes(t, &full).unwrap(), full, "{n} must be verbatim");
        }
    }

    /// 🪤 `A_log` is per-HEAD `[64]`, `dt_bias` per-CHANNEL `[8192]`. Same block, different
    /// granularity — slicing `dt_bias` by head count keeps 1/128th of the right data.
    #[test]
    fn a_log_and_dt_bias_shard_at_different_granularity() {
        let p = plan(1);
        let a = p.get("A_log").unwrap();
        let d = p.get("dt_bias").unwrap();
        assert_eq!((a.full_rows, a.local_rows), (64, 32), "A_log is per-head");
        assert_eq!(
            (d.full_rows, d.local_rows),
            (8192, 4096),
            "dt_bias is per-channel"
        );
        // Rank 1's offsets differ by a factor of head_dim.
        assert_eq!(d.src_row_offset, a.src_row_offset * 128);
    }

    /// At `tp_size = 1` the plan is the identity, so the sharded source is a pass-through —
    /// which is what lets TP=1 and TP=2 share one binder code path.
    #[test]
    fn tp1_is_the_identity() {
        let p = KdaTpPlan::new(0, 1, 4096, 128, 64, 4, 128).unwrap();
        for t in &p.tensors {
            assert_eq!(t.local_bytes(), t.full_bytes(), "{}", t.name);
            let full = vec![0x5Au8; t.full_bytes()];
            assert_eq!(shard_bytes(t, &full).unwrap(), full, "{}", t.name);
        }
        assert!(!p.needs_output_all_reduce());
    }

    /// A tensor whose on-disk size disagrees with the plan is refused, not silently
    /// truncated — the shard would otherwise read a valid range of the wrong tensor.
    #[test]
    fn a_size_mismatch_is_refused() {
        let p = plan(0);
        let q = p.get("q_proj").unwrap();
        let err = shard_bytes(q, &vec![0u8; q.full_bytes() - 2]).unwrap_err();
        assert!(err.to_string().contains("on disk"), "{err}");
    }

    /// The conv tensors are rank-3 `[dim, 1, kernel]` on disk; the local shape must keep that
    /// rank, because the binder validates it.
    #[test]
    fn local_shape_preserves_the_on_disk_rank() {
        let p = plan(1);
        let c = p.get("q_conv1d").unwrap();
        assert_eq!(local_shape(c, &[8192, 1, 4]), vec![4096, 1, 4]);
        let o = p.get("o_proj").unwrap();
        assert_eq!(local_shape(o, &[4096, 8192]), vec![4096, 4096]);
        let n = p.get("o_norm").unwrap();
        assert_eq!(local_shape(n, &[128]), vec![128]);
    }

    /// End to end through the adapter: the binder-facing view is this rank's slice, with a
    /// local shape, and unplanned names still pass through so the binder's refusal survives.
    #[test]
    fn the_adapter_presents_local_slices_and_passes_the_rest_through() {
        struct Src(Vec<(String, KdaDtype, Vec<usize>, Vec<u8>)>);
        impl KdaTensorSource for Src {
            fn get(&self, name: &str) -> Option<RawTensor<'_>> {
                self.0
                    .iter()
                    .find(|(n, ..)| n == name)
                    .map(|(_, d, s, b)| RawTensor {
                        dtype: *d,
                        shape: s.clone(),
                        bytes: b,
                    })
            }
            fn names(&self) -> Vec<String> {
                self.0.iter().map(|(n, ..)| n.clone()).collect()
            }
        }
        let p = KdaTpPlan::new(1, 2, 4, 64, 4, 2, 4).unwrap();
        let q = p.get("q_proj").unwrap();
        let src = Src(vec![
            (
                "self_attn.q_proj.weight".into(),
                KdaDtype::Bf16,
                vec![256, 4],
                (0..=255u8).flat_map(|r| [r; 8]).collect(),
            ),
            (
                "input_layernorm.weight".into(),
                KdaDtype::Bf16,
                vec![4],
                vec![0xEE; 8],
            ),
        ]);
        let sh = KdaShardedSource::new(&src, &p).unwrap();

        let t = sh.get("self_attn.q_proj.weight").unwrap();
        assert_eq!(t.shape, vec![128, 4], "local rows, full row width");
        assert_eq!(t.bytes.len(), q.local_bytes());
        assert!(
            t.bytes.iter().all(|b| *b >= 128),
            "rank 1 keeps the high rows"
        );

        // Not a KDA tensor: untouched.
        let n = sh.get("input_layernorm.weight").unwrap();
        assert_eq!(n.bytes, vec![0xEE; 8]);
        assert_eq!(sh.names().len(), 2, "the census still sees everything");
    }
}
