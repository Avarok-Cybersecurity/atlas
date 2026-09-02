// SPDX-License-Identifier: AGPL-3.0-only

//! Materialize-pass tests on a store whose EXL3 quartets live in a
//! fast-loader arena (`WeightStore::adopt_arena` + `.offset()` views).
//! Child of `exl3_materialize_tests.rs` (shares its helpers).

use super::*;

/// A quartet the fast loader POOLED (arena + `.offset()` views) that this
/// pass then materializes (gates off): the removed views must not be
/// `gpu.free`d (the mock, like CUDA, rejects interior pointers), the bytes
/// are reported as stranded, the arena stays live, and release later frees
/// it exactly once.
#[test]
fn pooled_quartet_materializes_without_interior_free() {
    use atlas_core::scope::ModelResource;
    use spark_runtime::weights::WeightArena;
    let gpu = MockGpuBackend::new();
    let p = "model.layers.0.mlp.experts.0.gate_proj";
    // One arena holding the whole quartet at 256-B slots, file order.
    let sizes = [
        (
            format!("{p}.trellis"),
            vec![160usize, 40, 32],
            WeightDtype::UInt16,
        ),
        (format!("{p}.suh"), vec![2560], WeightDtype::F16),
        (format!("{p}.svh"), vec![640], WeightDtype::F16),
        (format!("{p}.mul1"), vec![], WeightDtype::Int32),
    ];
    let slot = |n: usize| n.div_ceil(256) * 256;
    let arena_bytes: usize = sizes
        .iter()
        .map(|(_, s, d)| slot(s.iter().product::<usize>() * d.byte_size()))
        .sum();
    let base = gpu.alloc(arena_bytes).unwrap();
    let mut m = HashMap::new();
    let mut off = 0usize;
    let mut pooled_bytes = 0usize;
    for (name, shape, dtype) in sizes {
        let bytes = shape.iter().product::<usize>() * dtype.byte_size();
        m.insert(
            name,
            WeightTensor {
                ptr: base.offset(off),
                shape,
                dtype,
            },
        );
        off += slot(bytes);
        pooled_bytes += bytes;
    }
    // A per-tensor bystander that the pass must still free normally.
    exl3_linear(&gpu, &mut m, "model.layers.0.linear_attn.in_proj_qkv", 4);
    let mut store = WeightStore::from_map(m);
    store.adopt_arena(WeightArena {
        base,
        bytes: arena_bytes,
        label: "test",
    });
    stamp_mul1(&gpu, &store, p);
    let before = gpu.alloc_count(); // arena + 4 bystander tensors

    let stats = materialize_exl3_impl(&gpu, &mut store, false, false, OFF).unwrap();
    assert_eq!(stats.quantized, 1);
    assert_eq!(stats.bf16, 1);
    assert_eq!(stats.stranded_pooled_bytes, pooled_bytes);
    for sfx in ["trellis", "suh", "svh", "mul1"] {
        assert!(!store.contains(&format!("{p}.{sfx}")), "{p}.{sfx} removed");
    }
    assert!(store.contains(&format!("{p}.weight")));
    // Bystander's 4 sources freed; arena still live; triplet (3) + BF16 (1)
    // + the pass's own allocations are the new entries.
    assert!(gpu.alloc_count() >= before - 4);
    assert_eq!(store.arena_count(), 1);
    assert!(store.is_pooled(base));

    store.release(&gpu).unwrap();
    assert_eq!(
        gpu.alloc_count(),
        0,
        "arena freed once with everything else"
    );
}

/// Gates ON: the pooled quartet is kept and nothing about it moves.
#[test]
fn pooled_quartet_kept_native_stays_pooled() {
    use spark_runtime::weights::WeightArena;
    let gpu = MockGpuBackend::new();
    let mut m = HashMap::new();
    exl3_linear(&gpu, &mut m, "lm_head", 4);
    let mut store = WeightStore::from_map(m);
    // Re-home the quartet into one arena (copy the bytes, swap the ptrs).
    let names: Vec<String> = ["trellis", "suh", "svh", "mul1"]
        .iter()
        .map(|s| format!("lm_head.{s}"))
        .collect();
    let total: usize = names
        .iter()
        .map(|n| store.get(n).unwrap().byte_size().max(4).div_ceil(256) * 256)
        .sum();
    let base = gpu.alloc(total).unwrap();
    let mut off = 0;
    for n in &names {
        let old = store.remove(n).unwrap();
        let mut buf = vec![0u8; old.byte_size().max(4)];
        gpu.copy_d2h(old.ptr, &mut buf).unwrap();
        gpu.copy_h2d(&buf, base.offset(off)).unwrap();
        gpu.free(old.ptr).unwrap();
        let slot = old.byte_size().max(4).div_ceil(256) * 256;
        store.insert(
            n.clone(),
            WeightTensor {
                ptr: base.offset(off),
                shape: old.shape,
                dtype: old.dtype,
            },
        );
        off += slot;
    }
    store.adopt_arena(WeightArena {
        base,
        bytes: total,
        label: "test",
    });
    stamp_mul1(&gpu, &store, "lm_head");
    let stats = materialize_exl3_impl(&gpu, &mut store, true, false, OFF).unwrap();
    assert_eq!(stats.kept_native, 1);
    assert_eq!(stats.stranded_pooled_bytes, 0);
    for n in &names {
        assert!(
            store.is_pooled(store.get(n).unwrap().ptr),
            "{n} still pooled"
        );
    }
    assert_eq!(gpu.alloc_count(), 1, "just the arena");
}
