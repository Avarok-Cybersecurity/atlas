// SPDX-License-Identifier: AGPL-3.0-only

//! The [`super::plan_dtype`] cast as the binders actually see it: through
//! [`KdaTensorSource::get`]. The unit tests next door prove the byte math; these
//! prove the wiring — that `get` hands out the plan-dtype copy when one exists
//! and the checkpoint's own bytes when one does not.

use super::*;

/// Build a `LayerSource` the way `collect` would, without a GPU.
fn source(entries: &[(&str, WeightDtype, Vec<usize>, Vec<u8>)]) -> LayerSource {
    let mut names = Vec::new();
    let mut tensors = std::collections::BTreeMap::new();
    let mut plan_cast = std::collections::BTreeMap::new();
    for (rel, dt, shape, bytes) in entries {
        names.push((*rel).to_string());
        if let Some(cast) = plan_dtype::cast_to_plan_dtype(rel, *dt, bytes).unwrap() {
            plan_cast.insert((*rel).to_string(), cast);
        }
        tensors.insert((*rel).to_string(), (*dt, shape.clone(), bytes.clone()));
    }
    LayerSource {
        names,
        tensors,
        plan_cast,
    }
}

fn f32_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16_blob(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect()
}

/// `nvidia/GLM-5.3-Flash-NVFP4` stores `q_conv1d` at F32. The binder must see
/// BF16 at the plan's byte width, with the checkpoint's shape intact — the
/// shape is what `KdaTpPlan` slices against.
#[test]
fn an_f32_conv1d_reaches_the_binder_as_plan_bf16() {
    let vals = [1.0f32, -2.5, 0.0, 7.75];
    let src = source(&[(
        "self_attn.q_conv1d.weight",
        WeightDtype::FP32,
        vec![4, 1, 1],
        f32_blob(&vals),
    )]);
    let t = src.get("self_attn.q_conv1d.weight").expect("present");
    assert_eq!(t.dtype, KdaDtype::Bf16);
    assert_eq!(t.shape, vec![4, 1, 1]);
    assert_eq!(t.bytes, bf16_blob(&vals).as_slice());
    // The byte count the sharding check in `tp_bind::shard_bytes` computes.
    assert_eq!(t.bytes.len(), 4 * 2);
    // `f32` still reads the checkpoint's own F32 — the DSA absorb math and
    // the F32 `ape` upload depend on that.
    assert_eq!(src.f32("self_attn.q_conv1d.weight").unwrap(), vals.to_vec());
}

/// A BF16 checkpoint allocates no cast copy and `get` borrows its bytes —
/// the `LibertAIDAI` load is unchanged.
#[test]
fn a_bf16_conv1d_is_handed_over_untouched() {
    let raw = bf16_blob(&[1.0, -2.5, 0.0, 7.75]);
    let src = source(&[(
        "self_attn.q_conv1d.weight",
        WeightDtype::BF16,
        vec![4, 1, 1],
        raw.clone(),
    )]);
    assert!(src.plan_cast.is_empty(), "BF16 must allocate no cast copy");
    let t = src.get("self_attn.q_conv1d.weight").expect("present");
    assert_eq!(t.dtype, KdaDtype::Bf16);
    assert_eq!(t.bytes, raw.as_slice());
}

/// `A_log` is F32 in the plan and F32 in both exports: no cast copy, and the
/// binder still sees 4-byte elements.
#[test]
fn a_log_stays_f32_through_get() {
    let raw = f32_blob(&[1.0, -2.5]);
    let src = source(&[("self_attn.A_log", WeightDtype::FP32, vec![2], raw.clone())]);
    assert!(src.plan_cast.is_empty());
    let t = src.get("self_attn.A_log").expect("present");
    assert_eq!(t.dtype, KdaDtype::F32);
    assert_eq!(t.bytes, raw.as_slice());
}
