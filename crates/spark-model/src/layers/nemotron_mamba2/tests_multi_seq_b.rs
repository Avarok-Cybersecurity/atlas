// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone-B mock-dispatch tests for the Nemotron-H Mamba-2 decode paths:
//! the strided conv/scan arm, the arm-aware projection rung, the slot-hole
//! decline, and the K-token MTP verify body.
//!
//! Split out of `tests_multi_seq.rs` (which keeps the milestone-A cases and
//! all the shared fixtures) purely for the repo's 500-LoC file cap — this is
//! a CHILD module of it, so `use super::*` picks up `lightning_config`,
//! `mk_layer`, `run_multi_seq`, `pool_states`, `count` and `grids`.
//!
//! Geometry legend is in the parent module's header.

use super::*;

/// Milestone B: the whole point of splitting the gate. At n=4 on the BF16
/// arm — BELOW `MAMBA2_PROJ_MIN_BF16` — the PROJECTIONS must stay per-row
/// (that arm's batched twin is a tile GEMM, MEASURED not bit-exact by
/// `examples/bf16_batch_bitparity_microtest.rs`), while the
/// conv/scan still collapses into ONE strided launch pair and the norms
/// still batch. Fails without the split: milestone A delegated the whole
/// layer to the default loop here, so all four phases were per-row.
#[test]
fn mamba2_multi_seq_n4_bf16_batches_conv_scan_but_not_projections() {
    let config = lightning_config();
    let gpu = MockGpuBackend::new();
    let mut layer = mk_layer(&gpu, &config);
    let dw = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    layer.set_bf16_weights(
        dw(config.mamba2_in_proj_size() * config.hidden_size * 2),
        dw(config.hidden_size * config.mamba2_d_inner() * 2),
    );
    assert_eq!(layer.proj_batch_min(), super::MAMBA2_PROJ_MIN_BF16);

    let mut owned = pool_states(&gpu, &layer, 4);
    let mut states: Vec<&mut (dyn LayerState + 'static)> =
        owned.iter_mut().map(|b| &mut **b).collect();
    run_multi_seq(&gpu, &config, &layer, &mut states).unwrap();

    let seen = grids(&gpu);
    // Projections stay per-row: no batched dense GEMM at either shape.
    assert_eq!(
        count(&seen, [81, 1, 1], [256, 1, 1]),
        0,
        "batched in_proj GEMM launched below the BF16 projection rung; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [21, 1, 1], [256, 1, 1]),
        0,
        "batched out_proj GEMM launched below the BF16 projection rung; grids: {seen:?}"
    );
    // ... but conv + scan are ONE strided launch each over all 4 rows.
    assert_eq!(
        count(&seen, [24, 4, 1], [256, 1, 1]),
        1,
        "expected ONE strided conv1d over 4 rows; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [64, 4, 1], [128, 1, 1]),
        1,
        "expected ONE strided mamba2_ssm over 4 rows; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [24, 1, 1], [256, 1, 1]),
        0,
        "per-row conv1d launched with a fully dense slot prefix; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [64, 1, 1], [128, 1, 1]),
        0,
        "per-row mamba2_ssm launched with a fully dense slot prefix; grids: {seen:?}"
    );
    // Norms batch (bit-identical per row), so grid.x == 4, twice.
    assert_eq!(
        count(&seen, [4, 1, 1], [1024, 1, 1]),
        2,
        "expected batched rms_norm_residual + gated_rms_norm at grid.x=4; grids: {seen:?}"
    );
}

/// The FP8 arm — the one Lightning takes — batches its projections from
/// rung 2, because `w8a16_gemv_batch4` is byte-identical to M x
/// `w8a16_gemv` (proven by `examples/w8a16_batch_bitparity_microtest.rs`).
/// At n=2 that means everything batches: one GEMV-batch4 per projection,
/// one strided conv, one strided scan.
#[test]
fn mamba2_multi_seq_n2_fp8_batches_everything() {
    let config = lightning_config();
    let gpu = MockGpuBackend::new();
    let mut layer = mk_layer(&gpu, &config);
    let fp8 = |n: usize, k: usize| Fp8Weight {
        weight: gpu.alloc(n * k).unwrap(),
        row_scale: gpu.alloc(n.div_ceil(128) * k.div_ceil(128) * 4).unwrap(),
        n: n as u32,
        k: k as u32,
        scale_format: crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
    };
    layer
        .set_fp8_weights(
            Some(fp8(config.mamba2_in_proj_size(), config.hidden_size)),
            Some(fp8(config.hidden_size, config.mamba2_d_inner())),
            true,
        )
        .unwrap();
    assert_eq!(layer.proj_batch_min(), 2, "FP8 arm must batch from rung 2");

    let mut owned = pool_states(&gpu, &layer, 2);
    let mut states: Vec<&mut (dyn LayerState + 'static)> =
        owned.iter_mut().map(|b| &mut **b).collect();
    run_multi_seq(&gpu, &config, &layer, &mut states).unwrap();

    let seen = grids(&gpu);
    // w8a16_gemv_batch4 geometry: (ceil(N/4), 1, 1) / (256,1,1). in_proj
    // N=10304 -> 2576; out_proj N=2688 -> 672. Exactly one launch each: the
    // per-row loop would have issued two of each.
    assert_eq!(
        count(&seen, [2576, 1, 1], [256, 1, 1]),
        1,
        "expected ONE batched in_proj FP8 GEMV at n=2; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [672, 1, 1], [256, 1, 1]),
        1,
        "expected ONE batched out_proj FP8 GEMV at n=2; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [24, 2, 1], [256, 1, 1]),
        1,
        "expected ONE strided conv1d over 2 rows; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [64, 2, 1], [128, 1, 1]),
        1,
        "expected ONE strided mamba2_ssm over 2 rows; grids: {seen:?}"
    );
}

/// The NVFP4 arm — the one Nano-30B takes — also batches from rung 2, now
/// that `w4a16_gemv_batch4/8/16` is byte-identical to M x `w4a16_gemv`
/// (proven by `examples/w4a16_batch_bitparity_microtest.rs`). `mk_layer`
/// installs no BF16/FP8 weights, so this is the NVFP4 arm. At n=2 the
/// projections must be ONE `w4a16_gemv_batch4` launch each, not two GEMVs.
#[test]
fn mamba2_multi_seq_n2_nvfp4_batches_projections() {
    let config = lightning_config();
    let gpu = MockGpuBackend::new();
    let layer = mk_layer(&gpu, &config);
    assert_eq!(
        layer.proj_batch_min(),
        2,
        "NVFP4 arm must batch from rung 2"
    );

    let mut owned = pool_states(&gpu, &layer, 2);
    let mut states: Vec<&mut (dyn LayerState + 'static)> =
        owned.iter_mut().map(|b| &mut **b).collect();
    run_multi_seq(&gpu, &config, &layer, &mut states).unwrap();

    let seen = grids(&gpu);
    // w4a16_gemv_batch4 shares the single-GEMV geometry (ceil(N/4),1,1) /
    // (256,1,1), so the tell is the COUNT: one launch per projection, where
    // the per-row loop below the rung would issue two.
    assert_eq!(
        count(&seen, [2576, 1, 1], [256, 1, 1]),
        1,
        "expected ONE batched in_proj NVFP4 GEMV at n=2; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [672, 1, 1], [256, 1, 1]),
        1,
        "expected ONE batched out_proj NVFP4 GEMV at n=2; grids: {seen:?}"
    );
}

/// Slot fragmentation and pad rows are the SAME failure mode for the strided
/// arm: a row whose pool slot is not at `base + i * slot_bytes` cannot be
/// covered by an inferred state stride, and covering it anyway would write
/// one sequence's recurrent state into another's slot. Rows `0..p` must go
/// strided and rows `p..n` must fall back to the per-row form.
#[test]
fn mamba2_multi_seq_declines_strided_past_a_slot_hole() {
    let config = lightning_config();
    let gpu = MockGpuBackend::new();
    let mut layer = mk_layer(&gpu, &config);
    let dw = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    layer.set_bf16_weights(
        dw(config.mamba2_in_proj_size() * config.hidden_size * 2),
        dw(config.hidden_size * config.mamba2_d_inner() * 2),
    );

    let mut owned = pool_states(&gpu, &layer, 8);
    // Row 3 lands somewhere else entirely — a reclaimed slot, or the shared
    // pool dummy slot every PAD row carries.
    let stray = gpu.alloc(layer.h_state_bytes).unwrap();
    owned[3]
        .as_any_mut()
        .downcast_mut::<SsmLayerState>()
        .unwrap()
        .h_state = stray;
    let mut states: Vec<&mut (dyn LayerState + 'static)> =
        owned.iter_mut().map(|b| &mut **b).collect();
    run_multi_seq(&gpu, &config, &layer, &mut states).unwrap();

    let seen = grids(&gpu);
    assert_eq!(
        count(&seen, [24, 3, 1], [256, 1, 1]),
        1,
        "expected ONE strided conv1d over the 3-row dense prefix; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [64, 3, 1], [128, 1, 1]),
        1,
        "expected ONE strided mamba2_ssm over the 3-row dense prefix; grids: {seen:?}"
    );
    // Rows 3..8 keep the per-row form — 5 of each, batch=1 geometry.
    assert_eq!(
        count(&seen, [24, 1, 1], [256, 1, 1]),
        5,
        "expected 5 per-row conv1d launches after the hole; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [64, 1, 1], [128, 1, 1]),
        5,
        "expected 5 per-row mamba2_ssm launches after the hole; grids: {seen:?}"
    );
    // The strided launch must NEVER cover all 8 — that is the state-bleed bug.
    assert_eq!(
        count(&seen, [64, 8, 1], [128, 1, 1]),
        0,
        "strided scan covered a fragmented batch — would corrupt row 3's slot"
    );
}

/// MTP verify (`decode_batched`) batches its stateless phases across the K
/// verify tokens while keeping conv+scan sequential in `t` — the K rows are
/// time steps of ONE sequence, so they can never share a scan launch. Fails
/// without the change: milestone A ran K full `decode()` calls, i.e. K
/// separate projection sweeps.
#[test]
fn mamba2_decode_batched_k_batches_projections_not_the_scan() {
    let config = lightning_config();
    let gpu = MockGpuBackend::new();
    let mut layer = mk_layer(&gpu, &config);
    let fp8 = |n: usize, k: usize| Fp8Weight {
        weight: gpu.alloc(n * k).unwrap(),
        row_scale: gpu.alloc(n.div_ceil(128) * k.div_ceil(128) * 4).unwrap(),
        n: n as u32,
        k: k as u32,
        scale_format: crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
    };
    layer
        .set_fp8_weights(
            Some(fp8(config.mamba2_in_proj_size(), config.hidden_size)),
            Some(fp8(config.hidden_size, config.mamba2_d_inner())),
            true,
        )
        .unwrap();

    let buffers = BufferArena::new(&config, 64, 4096, 16, 32, &gpu).unwrap();
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let ctx = ForwardContext {
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        gpu: &gpu,
        config: &config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    };
    let mut owned = pool_states(&gpu, &layer, 1);
    let k = 2usize;
    layer
        .decode_batched_k(
            buffers.hidden_states(),
            buffers.residual(),
            k,
            owned[0].as_mut(),
            &ctx,
            0,
        )
        .unwrap();

    let seen = grids(&gpu);
    assert_eq!(
        count(&seen, [2576, 1, 1], [256, 1, 1]),
        1,
        "expected ONE batched in_proj sweep for K=2 verify rows; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [672, 1, 1], [256, 1, 1]),
        1,
        "expected ONE batched out_proj sweep for K=2 verify rows; grids: {seen:?}"
    );
    // Conv + scan stay per-token: token t+1's state depends on token t.
    assert_eq!(
        count(&seen, [24, 1, 1], [256, 1, 1]),
        k,
        "conv1d must run once per verify token; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [64, 1, 1], [128, 1, 1]),
        k,
        "mamba2_ssm must run once per verify token; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [2, 1, 1], [1024, 1, 1]),
        2,
        "expected batched rms_norm + gated_rms_norm over the K rows; grids: {seen:?}"
    );
}
