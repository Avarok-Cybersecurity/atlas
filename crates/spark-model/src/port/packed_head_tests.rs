// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the packed LM head (port lane L24): the
//! chunking over the weight's decode rows, the row independence the logits
//! rely on, the load check that keeps packed bytes away from every BF16 reader,
//! and the refusal on a build without lane L19's Ada modules.

use std::collections::HashMap;
use std::sync::Arc;

use avarok_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::*;
use crate::port::w4::fake::{Call, FakeW4, log};

const VOCAB: usize = 40;
const HIDDEN: usize = 16;

fn bf16_rows(m: usize, k: usize) -> Vec<u8> {
    (0..m * k)
        .flat_map(|i| half::bf16::from_f32(((i * 7 % 23) as f32 - 11.0) / 4.0).to_le_bytes())
        .collect()
}

#[test]
fn a_prompt_is_served_in_chunks_of_the_weights_decode_rows() {
    let gpu = MockGpuBackend::new();
    let calls = log();
    let w: W4Box = Arc::new(FakeW4::new("head", VOCAB, HIDDEN, 8, &calls));
    let head = PackedHead::from_linear(&gpu, w.clone()).unwrap();
    assert!(!head.scratch.is_null());
    assert_eq!(head.scratch_bytes, w.scratch_bytes(8));
    let hidden = gpu.alloc(19 * HIDDEN * 2).unwrap();
    let logits = gpu.alloc(19 * VOCAB * 2).unwrap();
    head.run(&gpu, hidden, 19, logits, 0).unwrap();
    let got: Vec<Call> = calls.lock().iter().map(|(_, c)| c.clone()).collect();
    let want: Vec<Call> = [(0usize, 8usize), (8, 8), (16, 3)]
        .iter()
        .map(|&(done, rows)| Call::Decode {
            x: hidden.offset(done * HIDDEN * 2),
            rows,
            out: logits.offset(done * VOCAB * 2),
        })
        .collect();
    assert_eq!(got, want);
}

/// Logits of a prompt row equal the same row decoded alone: the W4Linear
/// contract (row r's bits never depend on the rows beside it) carried through
/// the chunking.
#[test]
fn every_row_equals_the_same_row_decoded_alone() {
    let gpu = MockGpuBackend::new();
    let calls = log();
    let w: W4Box = Arc::new(FakeW4::new("head", VOCAB, HIDDEN, 8, &calls).computing(7));
    let head = PackedHead::from_linear(&gpu, w).unwrap();
    let m = 11usize;
    let x = bf16_rows(m, HIDDEN);
    let hidden = gpu.alloc(x.len()).unwrap();
    gpu.copy_h2d(&x, hidden).unwrap();
    let all = gpu.alloc(m * VOCAB * 2).unwrap();
    head.run(&gpu, hidden, m, all, 0).unwrap();
    let mut whole = vec![0u8; m * VOCAB * 2];
    gpu.copy_d2h(all, &mut whole).unwrap();
    let one = gpu.alloc(VOCAB * 2).unwrap();
    for r in 0..m {
        head.run(&gpu, hidden.offset(r * HIDDEN * 2), 1, one, 0)
            .unwrap();
        let mut row = vec![0u8; VOCAB * 2];
        gpu.copy_d2h(one, &mut row).unwrap();
        assert_eq!(row, &whole[r * VOCAB * 2..(r + 1) * VOCAB * 2], "row {r}");
    }
}

#[test]
fn the_ada_head_refuses_a_build_without_the_keep_packed_modules() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(crate::port::gguf_q4::launch::MMVQ_MODULE);
    let blocks = gpu.alloc(VOCAB * 256 / 256 * 210).unwrap();
    let err = PackedHead::new(&gpu, blocks, VOCAB, 256).unwrap_err();
    assert!(format!("{err:#}").contains("not in this build"), "{err:#}");
    // No lookup was issued against the absent module.
    assert!(
        gpu.kernel_lookups_snapshot()
            .iter()
            .all(|(m, _)| m != crate::port::gguf_q4::launch::MMVQ_MODULE)
    );
}

fn store_with_head(gpu: &MockGpuBackend, dtype: Option<WeightDtype>) -> WeightStore {
    let mut m = HashMap::new();
    if let Some(dtype) = dtype {
        let ptr = gpu.alloc(64).unwrap();
        m.insert(
            "lm_head.weight".to_string(),
            WeightTensor {
                ptr,
                shape: vec![VOCAB, 256],
                dtype,
            },
        );
    }
    WeightStore::from_map(m)
}

fn config(head_bf16: Option<bool>, fp8: bool) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3_5".to_string();
    c.lm_head_bf16_override = head_bf16;
    c.lm_head_fp8 = fp8;
    c
}

#[test]
fn without_the_flag_the_check_changes_nothing() {
    let gpu = MockGpuBackend::new();
    let store = store_with_head(&gpu, Some(WeightDtype::Q6K));
    // Even the combination the flag refuses passes: the flag is the gate.
    check_load_with(false, &store, &config(Some(false), true), true).unwrap();
}

#[test]
fn a_packed_head_with_the_drafter_or_a_requantization_is_refused() {
    let gpu = MockGpuBackend::new();
    let store = store_with_head(&gpu, Some(WeightDtype::Q6K));
    let spec = check_load_with(true, &store, &config(Some(true), false), true).unwrap_err();
    assert!(format!("{spec:#}").contains("--speculative"), "{spec:#}");
    for c in [config(Some(false), false), config(Some(true), true)] {
        let e = check_load_with(true, &store, &c, false).unwrap_err();
        assert!(format!("{e:#}").contains("--lm-head-dtype bf16"), "{e:#}");
    }
    check_load_with(true, &store, &config(Some(true), false), false).unwrap();
}

#[test]
fn a_head_that_is_not_packed_is_served_as_before() {
    let gpu = MockGpuBackend::new();
    for dtype in [None, Some(WeightDtype::BF16), Some(WeightDtype::FP8E4M3)] {
        let store = store_with_head(&gpu, dtype);
        // Not an error even with the drafter on: nothing is packed.
        check_load_with(true, &store, &config(None, false), true).unwrap();
    }
}
