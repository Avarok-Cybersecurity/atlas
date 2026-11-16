// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the shared quantize-then-free install: the order of quantizes and frees (the
//! load peak), the ledger as the owner of every kept buffer, and what a failure part-way leaves.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::cell::RefCell;
use std::collections::HashMap;

fn site<'a>(gpu: &'a MockGpuBackend, store: &'a WeightStore) -> Site<'a> {
    Site {
        gpu,
        store,
        stream: 0,
        lp: "model.layers.0",
    }
}

#[test]
fn each_source_is_freed_before_the_next_projection_is_quantized() {
    let log: RefCell<Vec<String>> = RefCell::new(Vec::new());
    let out = quantize_then_free(
        3,
        |i| {
            log.borrow_mut().push(format!("q{i}"));
            Ok(i * 10)
        },
        |i| {
            log.borrow_mut().push(format!("f{i}"));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(out, [0, 10, 20]);
    let order = log.into_inner();
    assert_eq!(
        order,
        ["q0", "f0", "q1", "f1", "q2", "f2"],
        "peak = checkpoint + one projection"
    );

    // A failed quantize stops the loop: its source and every later one stay with the caller.
    let log: RefCell<Vec<String>> = RefCell::new(Vec::new());
    let r = quantize_then_free(
        3,
        |i| {
            log.borrow_mut().push(format!("q{i}"));
            if i == 1 {
                Err(anyhow::anyhow!("non-finite group"))
            } else {
                Ok(i)
            }
        },
        |i| {
            log.borrow_mut().push(format!("f{i}"));
            Ok(())
        },
    );
    assert!(r.is_err());
    assert_eq!(log.into_inner(), ["q0", "f0", "q1"]);
    // A failed free also stops it: the next projection is not quantized over a live source.
    let r = quantize_then_free(2, Ok, |_| Err(anyhow::anyhow!("free failed")));
    assert!(r.is_err());
}

#[test]
fn every_kept_buffer_is_owned_by_the_ledger_under_its_role() {
    for (role, label) in [
        (W4Role::Ffn, "ffn affine from bf16"),
        (W4Role::Gdn, "ssm affine from bf16"),
        (W4Role::Attn, "attn affine from bf16"),
    ] {
        let gpu = MockGpuBackend::new();
        let store = WeightStore::from_map(HashMap::new());
        let src = gpu.alloc(128 * 256 * 2).unwrap();
        let p = [Projection {
            src,
            n: 128,
            k: 256,
            name: "w",
        }];
        let w = install_projections(role, &W4Opts::default(), site(&gpu, &store), &p).unwrap();
        assert_eq!(labels(role)[0], label);
        let rows = store.derived().by_label();
        assert_eq!(rows.len(), 4, "{role:?}: codes, scales, biases, tickets");
        assert!(
            rows.iter().all(|r| labels(role).contains(&r.0)),
            "{role:?}: {rows:?}"
        );
        assert_eq!(store.derived().bytes(), w[0].device_bytes());
        assert!(gpu.read_alloc(src).is_none());
        store.derived().release(&gpu).unwrap();
        assert_eq!(
            gpu.live_alloc_count(),
            0,
            "{role:?}: nothing leaks, nothing freed twice"
        );
    }
}

#[test]
fn a_concat_held_by_the_ledger_leaves_it_before_it_is_freed() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    let concat = gpu.alloc(128 * 256 * 2).unwrap();
    store
        .derived()
        .adopt("ssm qkvz bf16 concat", concat, 128 * 256 * 2);
    let p = [Projection {
        src: concat,
        n: 128,
        k: 256,
        name: "qkvz",
    }];
    let w = install_projections(W4Role::Gdn, &W4Opts::default(), site(&gpu, &store), &p).unwrap();
    assert!(gpu.read_alloc(concat).is_none(), "freed");
    assert_eq!(
        store.derived().bytes(),
        w[0].device_bytes(),
        "only the weight's buffers remain"
    );
    store
        .derived()
        .release(&gpu)
        .expect("no double free at teardown");
}

#[test]
fn a_failure_part_way_keeps_the_done_weights_owned_and_the_rest_allocated() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    let a = gpu.alloc(128 * 256 * 2).unwrap();
    let b = gpu.alloc(128 * 96 * 2).unwrap();
    // The second projection's K (96) is not a whole 64-group: the quantizer refuses it.
    let p = [
        Projection {
            src: a,
            n: 128,
            k: 256,
            name: "first",
        },
        Projection {
            src: b,
            n: 128,
            k: 96,
            name: "second",
        },
    ];
    let e = install_projections(W4Role::Attn, &W4Opts::default(), site(&gpu, &store), &p)
        .expect_err("K 96 at group 64");
    let text = format!("{e:#}");
    assert!(text.contains("second [128x96]"), "{text}");
    assert!(
        text.contains("AVAROK_ATTN_AFFINE4_FROM_BF16=0 keeps BF16"),
        "{text}"
    );
    assert!(
        gpu.read_alloc(a).is_none(),
        "the first source went after its quantize"
    );
    assert!(
        gpu.read_alloc(b).is_some(),
        "the failed one stays the caller's"
    );
    assert_eq!(
        store.derived().len(),
        4,
        "the first weight is owned, so teardown frees it"
    );
    store.derived().release(&gpu).unwrap();
    gpu.free(b).unwrap();
    assert_eq!(gpu.live_alloc_count(), 0);
}

#[test]
fn the_int8_prompt_gemm_is_prepared_on_the_load_path_and_the_install_still_owns_everything() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    let src = gpu.alloc(128 * 256 * 2).unwrap();
    let p = [Projection {
        src,
        n: 128,
        k: 256,
        name: "w",
    }];
    let int8 = W4Opts {
        int8_prefill: true,
        ..W4Opts::default()
    };
    let gemm = crate::port::affine::prefill::launch::GEMM_MODULE;
    let w = install_projections(W4Role::Ffn, &int8, site(&gpu, &store), &p).unwrap();
    let l = gpu.kernel_lookups_snapshot();
    assert!(
        l.iter().any(|(m, _)| m == gemm),
        "entries resolved at load: {l:?}"
    );
    assert_eq!(store.derived().bytes(), w[0].device_bytes());
    assert!(gpu.read_alloc(src).is_none());
    // Without the row the prompt GEMM's entries are not looked up by the install.
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    let src = gpu.alloc(128 * 256 * 2).unwrap();
    let p = [Projection {
        src,
        n: 128,
        k: 256,
        name: "w",
    }];
    install_projections(W4Role::Ffn, &W4Opts::default(), site(&gpu, &store), &p).unwrap();
    let l = gpu.kernel_lookups_snapshot();
    assert!(l.iter().all(|(m, _)| m != gemm), "{l:?}");
}

#[test]
fn a_foreign_w4_linear_has_no_affine_buffers() {
    let log = crate::port::w4::fake::log();
    let fake = crate::port::w4::fake::FakeW4::new("foreign", 64, 128, 16, &log);
    assert!(
        buffers_of(&fake).is_empty(),
        "nothing to adopt, so the install's ensure! refuses it"
    );
}
