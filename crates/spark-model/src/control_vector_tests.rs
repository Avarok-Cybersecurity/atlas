// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-only tests for the control-vector host load. CI has no GPU, so these
//! exercise [`build_table`] — the parse, the validation and the
//! unit-normalize/scale-fold — against real GGUF bytes. The kernel itself is
//! proven by `examples/cvec_projection_microtest.rs` on a GPU box.

use super::*;

const GGUF_MAGIC: u32 = 0x4655_4747;

fn push_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

/// Build a control-vector GGUF holding `dirs` as F32 `direction.<il>` tensors.
/// `type_id` and `n_dims` are overridable so the rejection tests can produce a
/// file that is well-formed GGUF but wrong for a control vector.
fn gguf_with(dirs: &[(usize, Vec<f32>)], name: fn(usize) -> String, type_id: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&(dirs.len() as u64).to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes()); // one metadata key

    push_str(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes()); // STRING
    push_str(&mut b, "controlvector");

    let mut off = 0u64;
    for (il, v) in dirs {
        push_str(&mut b, &name(*il));
        b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(&type_id.to_le_bytes());
        b.extend_from_slice(&off.to_le_bytes());
        off += (v.len() * 4) as u64;
    }
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    for (_, v) in dirs {
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    b
}

fn gguf(dirs: &[(usize, Vec<f32>)]) -> Vec<u8> {
    gguf_with(dirs, |il| format!("direction.{il}"), 0)
}

/// A unit vector of `hidden` dims whose direction depends on `seed`.
fn unit(hidden: usize, seed: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..hidden)
        .map(|i| (((i * 2654435761 + seed * 40503) % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let n = v
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    for x in v.iter_mut() {
        *x = (*x as f64 / n) as f32;
    }
    v
}

fn spec(start: usize, end: usize, scale: f32, mode: CvecMode) -> ControlVectorSpec {
    ControlVectorSpec {
        path: PathBuf::from("<test>"),
        scale,
        layer_start: start,
        layer_end: end,
        mode,
        // No model here: these run CPU-only in CI, so the hint check has
        // nothing to compare against and is skipped.
        model_type: None,
    }
}

fn norm(row: &[f32]) -> f64 {
    row.iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn unit_vectors_at_scale_one_give_scale_one() {
    let h = 16;
    let dirs: Vec<_> = (1..=4).map(|il| (il, unit(h, il))).collect();
    let (table, scales) =
        build_table(&gguf(&dirs), &spec(1, 4, 1.0, CvecMode::Project), h, 8).unwrap();

    for il in 1..=4 {
        assert!(
            (scales[il] - 1.0).abs() < 1e-6,
            "layer {il} scale {} != 1.0",
            scales[il]
        );
        assert!((norm(&table[il * h..(il + 1) * h]) - 1.0).abs() < 1e-5);
    }
    // Layer 0 never carries a direction.
    assert_eq!(scales[0], 0.0);
    assert!(table[0..h].iter().all(|x| *x == 0.0));
}

#[test]
fn project_mode_folds_the_norm_into_the_scale() {
    // llama.cpp stores the unit direction and treats |v| as the per-layer
    // scale, so a file scaled by 3 behaves as scale 3 rather than as a longer
    // vector. Mirroring that is what makes a scaled file portable both ways.
    let h = 16;
    let scaled: Vec<f32> = unit(h, 1).iter().map(|x| x * 3.0).collect();
    let (table, scales) = build_table(
        &gguf(&[(1, scaled)]),
        &spec(1, 1, 1.0, CvecMode::Project),
        h,
        4,
    )
    .unwrap();

    assert!((scales[1] - 3.0).abs() < 1e-5, "scale was {}", scales[1]);
    assert!((norm(&table[h..2 * h]) - 1.0).abs() < 1e-5);
}

#[test]
fn user_scale_multiplies_the_folded_norm() {
    let h = 16;
    let scaled: Vec<f32> = unit(h, 1).iter().map(|x| x * 2.0).collect();
    let (_, scales) = build_table(
        &gguf(&[(1, scaled)]),
        &spec(1, 1, 0.5, CvecMode::Project),
        h,
        4,
    )
    .unwrap();
    assert!((scales[1] - 1.0).abs() < 1e-5, "scale was {}", scales[1]);
}

#[test]
fn add_mode_keeps_the_raw_vector_and_the_user_scale() {
    let h = 16;
    let raw: Vec<f32> = unit(h, 1).iter().map(|x| x * 4.0).collect();
    let (table, scales) = build_table(
        &gguf(&[(1, raw.clone())]),
        &spec(1, 1, 0.1, CvecMode::Add),
        h,
        4,
    )
    .unwrap();

    assert!((scales[1] - 0.1).abs() < 1e-6);
    // NOT normalized: the additive arm's magnitude is part of the vector.
    assert!((norm(&table[h..2 * h]) - 4.0).abs() < 1e-4);
    for (got, want) in table[h..2 * h].iter().zip(raw.iter()) {
        assert!((got - want).abs() < 1e-6);
    }
}

#[test]
fn rows_outside_the_active_range_stay_zero() {
    let h = 16;
    let dirs: Vec<_> = (1..=6).map(|il| (il, unit(h, il))).collect();
    let (table, scales) =
        build_table(&gguf(&dirs), &spec(3, 4, 1.0, CvecMode::Project), h, 8).unwrap();

    for il in 0..8 {
        let active = (3..=4).contains(&il);
        assert_eq!(
            scales[il] != 0.0,
            active,
            "layer {il}: scale {} but active={active}",
            scales[il]
        );
        assert_eq!(
            table[il * h..(il + 1) * h].iter().any(|x| *x != 0.0),
            active,
            "layer {il}: row non-zero but active={active}"
        );
    }
}

#[test]
fn rejects_a_hidden_size_mismatch() {
    // The commonest real mistake: a vector built for another model.
    let e = build_table(
        &gguf(&[(1, unit(32, 1))]),
        &spec(1, 1, 1.0, CvecMode::Project),
        16,
        4,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("dims"), "{e}");
    assert!(e.contains("16"), "{e}");
}

#[test]
fn rejects_a_non_direction_tensor() {
    let e = gguf_with(&[(1, unit(16, 1))], |il| format!("blk.{il}.weight"), 0);
    let e = build_table(&e, &spec(1, 1, 1.0, CvecMode::Project), 16, 4)
        .unwrap_err()
        .to_string();
    assert!(e.contains("direction"), "{e}");
}

#[test]
fn rejects_a_non_f32_tensor() {
    // type 1 = F16.
    let b = gguf_with(&[(1, unit(16, 1))], |il| format!("direction.{il}"), 1);
    let e = build_table(&b, &spec(1, 1, 1.0, CvecMode::Project), 16, 4)
        .unwrap_err()
        .to_string();
    assert!(e.contains("F32"), "{e}");
}

#[test]
fn rejects_a_zero_vector_in_project_mode() {
    // A zero direction projects nothing — it would be a per-layer silent
    // no-op inside a range the operator believes is active.
    let e = build_table(
        &gguf(&[(1, vec![0.0; 16])]),
        &spec(1, 1, 1.0, CvecMode::Project),
        16,
        4,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("zero vector"), "{e}");
}

#[test]
fn rejects_a_range_past_the_last_layer() {
    let e = build_table(
        &gguf(&[(1, unit(16, 1))]),
        &spec(1, 9, 1.0, CvecMode::Project),
        16,
        4,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("exceeds"), "{e}");
}

#[test]
fn rejects_an_inverted_range() {
    let e = build_table(
        &gguf(&[(1, unit(16, 1))]),
        &spec(3, 1, 1.0, CvecMode::Project),
        16,
        4,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("empty"), "{e}");
}

#[test]
fn rejects_a_range_that_selects_no_direction() {
    // The file covers layers 1..2 but the operator asked for 5..6. Applying
    // nothing at all is exactly the silent no-op this guard exists to stop.
    let dirs: Vec<_> = (1..=2).map(|il| (il, unit(16, il))).collect();
    let e = build_table(&gguf(&dirs), &spec(5, 6, 1.0, CvecMode::Project), 16, 8)
        .unwrap_err()
        .to_string();
    assert!(e.contains("no direction falls inside"), "{e}");
}

#[test]
fn rejects_a_non_finite_scale() {
    let e = build_table(
        &gguf(&[(1, unit(16, 1))]),
        &spec(1, 1, f32::NAN, CvecMode::Project),
        16,
        4,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("finite"), "{e}");
}

#[test]
fn rejects_a_layer_zero_direction() {
    let e = build_table(
        &gguf(&[(0, unit(16, 0))]),
        &spec(0, 1, 1.0, CvecMode::Project),
        16,
        4,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("outside the model's layers"), "{e}");
}

#[test]
fn rejects_a_non_gguf_file() {
    let e = build_table(
        b"not a gguf at all",
        &spec(1, 1, 1.0, CvecMode::Project),
        16,
        4,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("control-vector GGUF"), "{e}");
}

/// Like [`gguf`] but with an explicit `general.architecture` and an optional
/// `controlvector.model_hint`, so the identity checks can be exercised.
fn gguf_ident(dirs: &[(usize, Vec<f32>)], arch: &str, hint: Option<&str>) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(dirs.len() as u64).to_le_bytes());
    b.extend_from_slice(&(if hint.is_some() { 2u64 } else { 1u64 }).to_le_bytes());

    push_str(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes());
    push_str(&mut b, arch);
    if let Some(h) = hint {
        push_str(&mut b, "controlvector.model_hint");
        b.extend_from_slice(&8u32.to_le_bytes());
        push_str(&mut b, h);
    }

    let mut off = 0u64;
    for (il, v) in dirs {
        push_str(&mut b, &format!("direction.{il}"));
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&(v.len() as u64).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&off.to_le_bytes());
        off += (v.len() * 4) as u64;
    }
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    for (_, v) in dirs {
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    b
}

fn spec_for(start: usize, end: usize, mode: CvecMode, model: Option<&str>) -> ControlVectorSpec {
    ControlVectorSpec {
        path: PathBuf::from("<test>"),
        scale: 1.0,
        layer_start: start,
        layer_end: end,
        mode,
        model_type: model.map(str::to_string),
    }
}

#[test]
fn rejects_a_file_that_is_not_a_control_vector() {
    // Geometry alone does not establish identity; a model shard passed by
    // mistake should fail here rather than at a confusing tensor-name error.
    let e = build_table(
        &gguf_ident(&[(1, unit(16, 0))], "llama", None),
        &spec_for(1, 1, CvecMode::Project, None),
        16,
        4,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("not a control vector"), "{e}");
}

#[test]
fn rejects_a_vector_derived_for_another_model() {
    // Two unrelated models can share a hidden size, and a direction from one
    // applied to the other loads and steers with no symptom but worse output.
    let e = build_table(
        &gguf_ident(&[(1, unit(16, 0))], "controlvector", Some("llama3")),
        &spec_for(1, 1, CvecMode::Project, Some("qwen4_exp")),
        16,
        4,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("model_hint"), "{e}");
}

#[test]
fn the_model_hint_comparison_ignores_naming_convention() {
    // THE case that makes this a normalisation and not an equality check: the
    // published refusal projection says `qwen4exp` (llama.cpp convention) and
    // this engine says `qwen4_exp`. Demanding equality would reject the one
    // artifact the feature shipped for.
    build_table(
        &gguf_ident(&[(1, unit(16, 0))], "controlvector", Some("qwen4exp")),
        &spec_for(1, 1, CvecMode::Project, Some("qwen4_exp")),
        16,
        4,
    )
    .expect("qwen4exp and qwen4_exp are the same model");
}

#[test]
fn a_missing_model_hint_is_tolerated() {
    // Hand-built and older vectors predate the convention. Absent is not a
    // mismatch, and refusing those would be a regression with no safety gain.
    build_table(
        &gguf_ident(&[(1, unit(16, 0))], "controlvector", None),
        &spec_for(1, 1, CvecMode::Project, Some("qwen4_exp")),
        16,
        4,
    )
    .expect("no hint means nothing to disagree with");
}

#[test]
fn rejects_a_unit_file_in_add_mode() {
    // The fail-open this exists for. `add` applies the row verbatim under one
    // global scale, so unit rows make the dose meaningless — usually far too
    // small to do anything, which reads as "the vector has no effect" rather
    // than as a misconfiguration.
    let dirs: Vec<(usize, Vec<f32>)> = (1..4).map(|il| (il, unit(16, il))).collect();
    let e = build_table(&gguf(&dirs), &spec(1, 3, 1.0, CvecMode::Add), 16, 8)
        .unwrap_err()
        .to_string();
    assert!(e.contains("UNIT-normalised"), "{e}");
    assert!(e.contains("--magnitude"), "{e}");
}

#[test]
fn accepts_a_raw_magnitude_file_in_add_mode() {
    // The same directions scaled to genuinely varying norms — which is what a
    // raw-magnitude file looks like, since |mean-diff| tracks the stream norm
    // and that grows with depth.
    let dirs: Vec<(usize, Vec<f32>)> = (1..4)
        .map(|il| {
            let k = 0.2 * il as f32;
            (il, unit(16, il).iter().map(|x| x * k).collect::<Vec<f32>>())
        })
        .collect();
    let (_, scales) = build_table(&gguf(&dirs), &spec(1, 3, 1.0, CvecMode::Add), 16, 8)
        .expect("varying per-layer norms are a raw file");
    // `add` keeps the user scale as-is; the magnitude lives in the row.
    assert_eq!(scales[1], 1.0);
}

#[test]
fn a_unit_file_is_still_correct_in_project_mode() {
    // Same file as the rejected `add` case: unit rows are exactly what
    // projection wants, so the guard must not leak into the other operator.
    let dirs: Vec<(usize, Vec<f32>)> = (1..4).map(|il| (il, unit(16, il))).collect();
    let (_, scales) = build_table(&gguf(&dirs), &spec(1, 3, 1.0, CvecMode::Project), 16, 8)
        .expect("unit rows are what project mode is for");
    assert!((scales[1] - 1.0).abs() < 1e-5, "{}", scales[1]);
}
