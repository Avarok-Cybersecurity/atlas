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
