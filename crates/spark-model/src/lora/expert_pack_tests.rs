// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 expert-pack structural tests (GPU-free): the `present` gate and the
//! sizing key-lists derived from an audit. The device pack + shape audit run on
//! hardware / against a real WeightStore.

use std::collections::BTreeMap;
use std::collections::HashMap;

use atlas_core::config::PeftAdapterConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{ExpertMap, RouterMap, key_lists, pack_into, packed_stride, present};
use crate::lora::test_support::cfg;
use crate::lora::{ExpertProj, LoraLayerWeights, expert_router_bytes};

fn empty() -> (RouterMap, ExpertMap) {
    (BTreeMap::new(), BTreeMap::new())
}

#[test]
fn present_is_false_only_when_both_empty() {
    let (r, e) = empty();
    assert!(!present(&r, &e));

    let (mut r, e) = empty();
    r.insert(3, [Some("a".into()), Some("b".into())]);
    assert!(present(&r, &e));

    let (r, mut e) = empty();
    e.insert((7, 2, ExpertProj::Down), [None, None]);
    assert!(present(&r, &e));
}

#[test]
fn key_lists_projects_layers_and_projs() {
    let mut router: RouterMap = BTreeMap::new();
    router.insert(3, [None, None]);
    router.insert(7, [None, None]);
    let mut experts: ExpertMap = BTreeMap::new();
    experts.insert((7, 0, ExpertProj::Gate), [None, None]);
    experts.insert((7, 5, ExpertProj::Down), [None, None]);

    let (ek, rl) = key_lists(&router, &experts);
    assert_eq!(ek, vec![(7, ExpertProj::Gate), (7, ExpertProj::Down)]);
    assert_eq!(rl, vec![3, 7]);
}

// ── Item-2 (PR #335 gate): uint4-aligned pool stride ────────────────────────

#[test]
fn packed_stride_rounds_up_to_multiple_of_8() {
    // Positive: a non-multiple-of-8 rank pads UP to the next uint4 boundary.
    assert_eq!(packed_stride(12), 16);
    assert_eq!(packed_stride(1), 8);
    assert_eq!(packed_stride(9), 16);
    // Negative: an already-aligned rank is NOT inflated.
    assert_eq!(packed_stride(8), 8);
    assert_eq!(packed_stride(16), 16);
    assert_eq!(packed_stride(0), 0);
}

fn peft_r(r: usize) -> PeftAdapterConfig {
    PeftAdapterConfig {
        r,
        lora_alpha: 2.0 * r as f64, // scale = 2.0 either way (rank-independent)
        target_modules: vec!["gate_proj".into()],
        target_modules_pattern: None,
        use_rslora: false,
        layers_to_transform: None,
        trainable_token_indices: Vec::new(),
        modules_to_save: Vec::new(),
        lora_embedding: false,
    }
}

/// Upload a patterned BF16 tensor to the mock device; byte `i` = `i % 251`.
fn up_tensor(gpu: &MockGpuBackend, shape: Vec<usize>) -> WeightTensor {
    let bytes: Vec<u8> = (0..shape.iter().product::<usize>() * 2)
        .map(|i| (i % 251) as u8)
        .collect();
    let ptr = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(&bytes, ptr).unwrap();
    WeightTensor {
        ptr,
        shape,
        dtype: WeightDtype::BF16,
    }
}

/// Drive the REAL pack entry (`pack_into`) at r=12 through the mock GPU and
/// assert the uint4-alignment contract end to end:
///   - every packed A/B base offset is 16-byte aligned (uint4-loadable),
///   - `LoraPair.rank` stays the LOGICAL 12 while `max_rank` is the padded 16,
///   - the B repack lands rows at the PADDED stride with zeroed pad columns,
///   - bytes consumed == `expert_router_bytes` (sizing/packing SSOT).
#[test]
fn pack_into_r12_is_uint4_aligned_and_matches_sizing() {
    let gpu = MockGpuBackend::new();
    let cfg = cfg(); // hidden 2048, moe_inter 512
    let peft = peft_r(12);

    // Router pair on layer 3 ([512, 2048] base) + one Down expert on layer 7.
    let (r_out, r_in) = super::router_dims(&cfg); // (512, 2048)
    let (e_out, e_in) = ExpertProj::Down.dims(&cfg, 7); // (2048, 512)
    let mut map: HashMap<String, WeightTensor> = HashMap::new();
    map.insert("r.a".into(), up_tensor(&gpu, vec![peft.r, r_in]));
    map.insert("r.b".into(), up_tensor(&gpu, vec![r_out, peft.r]));
    map.insert("e.a".into(), up_tensor(&gpu, vec![peft.r, e_in]));
    map.insert("e.b".into(), up_tensor(&gpu, vec![e_out, peft.r]));
    let store = WeightStore::from_map(map);

    let mut router: RouterMap = BTreeMap::new();
    router.insert(3, [Some("r.a".into()), Some("r.b".into())]);
    let mut experts: ExpertMap = BTreeMap::new();
    experts.insert(
        (7, 0, ExpertProj::Down),
        [Some("e.a".into()), Some("e.b".into())],
    );

    // Pool sized EXACTLY like loading.rs does: from the audited key set.
    let (ek, rl) = key_lists(&router, &experts);
    let sized = expert_router_bytes(&cfg, &ek, &rl, peft.r);
    let pool = gpu.alloc(sized).unwrap();
    gpu.memset(pool, 0, sized).unwrap();

    let mut layers: Vec<Option<LoraLayerWeights>> = (0..48).map(|_| None).collect();
    let mut off = 0usize;
    let packed = pack_into(
        &mut layers,
        &store,
        &peft,
        &router,
        &experts,
        &cfg,
        &gpu,
        pool,
        peft.r, // callers pass the RAW rank cap; padding is derived inside
        &mut off,
    )
    .unwrap();
    assert_eq!(packed, 2);
    // Sizing/packing SSOT: the pack consumed exactly what the estimator sized.
    assert_eq!(
        off, sized,
        "pack_into offsets must match expert_router_bytes"
    );

    let rp = layers[3].as_ref().unwrap().router.as_ref().unwrap();
    let ep = layers[7].as_ref().unwrap().experts.as_ref().unwrap();
    let ep = ep.pairs.get(&(0, ExpertProj::Down)).unwrap();
    for (what, p) in [("router", rp), ("expert-down", ep)] {
        // Logical rank vs derived padded stride — never conflated.
        assert_eq!(p.rank, 12, "{what}: logical rank");
        assert_eq!(p.max_rank, 16, "{what}: padded uint4 stride");
        // uint4 (16 B) alignment of both packed bases relative to the pool.
        assert_eq!((p.a.weight.0 - pool.0) % 16, 0, "{what}: A base alignment");
        assert_eq!((p.b.weight.0 - pool.0) % 16, 0, "{what}: B base alignment");
    }
    // Region layout: router A at pool+0 sized stride*in, B right after.
    assert_eq!(rp.a.weight.0, pool.0);
    assert_eq!(rp.b.weight.0, pool.0 + (16 * r_in * 2) as u64);

    // B repack: row 1 must start at the PADDED stride (16 elems = 32 B) and
    // carry the source's row-1 bytes, with pad columns 12..16 zero.
    let mut b_row1 = vec![0u8; 16 * 2];
    gpu.copy_d2h(
        spark_runtime::gpu::DevicePtr(rp.b.weight.0 + 16 * 2),
        &mut b_row1,
    )
    .unwrap();
    let src_row1: Vec<u8> = (12 * 2..24 * 2).map(|i| (i % 251) as u8).collect();
    assert_eq!(&b_row1[..12 * 2], &src_row1[..], "row 1 at padded stride");
    assert_eq!(&b_row1[12 * 2..], &[0u8; 8], "pad columns stay zero");
}

/// Negative arm: an already-aligned rank (16) packs at its own stride — the
/// padding path must be a no-op, byte-identical to the raw-rank layout.
#[test]
fn pack_into_r16_layout_is_unpadded() {
    let gpu = MockGpuBackend::new();
    let cfg = cfg();
    let peft = peft_r(16);
    let (r_out, r_in) = super::router_dims(&cfg);
    let mut map: HashMap<String, WeightTensor> = HashMap::new();
    map.insert("r.a".into(), up_tensor(&gpu, vec![peft.r, r_in]));
    map.insert("r.b".into(), up_tensor(&gpu, vec![r_out, peft.r]));
    let store = WeightStore::from_map(map);
    let mut router: RouterMap = BTreeMap::new();
    router.insert(3, [Some("r.a".into()), Some("r.b".into())]);
    let experts: ExpertMap = BTreeMap::new();

    let sized = expert_router_bytes(&cfg, &[], &[3], peft.r);
    // Raw-rank formula: no padding may be introduced at an aligned rank.
    assert_eq!(sized, (16 * r_in + r_out * 16) * 2);
    let pool = gpu.alloc(sized).unwrap();
    gpu.memset(pool, 0, sized).unwrap();
    let mut layers: Vec<Option<LoraLayerWeights>> = (0..48).map(|_| None).collect();
    let mut off = 0usize;
    pack_into(
        &mut layers,
        &store,
        &peft,
        &router,
        &experts,
        &cfg,
        &gpu,
        pool,
        peft.r,
        &mut off,
    )
    .unwrap();
    assert_eq!(off, sized);
    let rp = layers[3].as_ref().unwrap().router.as_ref().unwrap();
    assert_eq!((rp.rank, rp.max_rank), (16, 16));
    let mut packed_b = vec![0u8; r_out * peft.r * 2];
    gpu.copy_d2h(rp.b.weight, &mut packed_b).unwrap();
    let source_b: Vec<u8> = (0..packed_b.len()).map(|i| (i % 251) as u8).collect();
    assert_eq!(packed_b, source_b, "aligned B must be byte-identical");
}
