// SPDX-License-Identifier: AGPL-3.0-only

//! The native-FP8 dense loader's decision table (#915).
//!
//! Every case is a route the H100 serve can actually be put into, expressed
//! through the SAME `GemmDispatch` the forward pass carries — not through the
//! process environment, which a `OnceLock` resolver would pin for the whole
//! test binary.

use super::*;
use crate::layers::ops::GemmDispatch;

/// The shape a native-FP8 dense serve boots in with no `ATLAS_*` set: FP8
/// overlays on both the FFN and attention, block-scaled prefill on, the W8A8
/// kernels present.
fn h100_default() -> DenseFp8Inputs {
    DenseFp8Inputs {
        ffn_fp8: true,
        attn_fp8: true,
        keep_nvfp4: false,
        dispatch: GemmDispatch::defaults(),
        w8a8_kernels: true,
        attn_w4a4: false,
        attn_prefill_q_t: false,
    }
}

/// K and V, and nothing else: the measured default route.
const KV_ONLY: Fp8TwinSet = Fp8TwinSet {
    q: false,
    k: true,
    v: true,
    o: false,
};

#[test]
fn default_native_fp8_route_builds_no_nvfp4_and_only_the_kv_fp8_twins() {
    let p = DenseFp8Plan::resolve(h100_default());
    assert_eq!(
        p,
        DenseFp8Plan {
            ffn_nvfp4: false,
            attn_nvfp4: false,
            attn_fp8_twins: KV_ONLY,
        },
        "the dense FFN and the attention NVFP4 fallbacks are unreachable; K/V \
         FP8 twins are NOT (cache_skip_qkv.rs has no W8A8 arm)"
    );
}

#[test]
fn the_kv_fp8_twins_survive_every_default_route_variation() {
    // `cache_skip_qkv.rs:218`/`:235` dereference them on the FIRST prefill
    // chunk of every request, with no W8A8 arm ahead of them. No lever below
    // may take them away — this is the NULL-pointer-launch regression guard.
    for dispatch in [
        GemmDispatch::defaults(),
        GemmDispatch {
            fp8_blockscaled_prefill: false,
            ..GemmDispatch::defaults()
        },
    ] {
        for w8a8_kernels in [true, false] {
            for attn_prefill_q_t in [true, false] {
                let p = DenseFp8Plan::resolve(DenseFp8Inputs {
                    dispatch,
                    w8a8_kernels,
                    attn_prefill_q_t,
                    ..h100_default()
                });
                assert!(p.attn_fp8_twins.k && p.attn_fp8_twins.v, "{p:?}");
            }
        }
    }
}

#[test]
fn a_non_fp8_layer_keeps_every_nvfp4_copy_and_gets_no_fp8_twins() {
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        ffn_fp8: false,
        attn_fp8: false,
        ..h100_default()
    });
    assert!(p.ffn_nvfp4, "no FP8 overlay -> NVFP4 is the only weight");
    assert!(p.attn_nvfp4);
    assert_eq!(
        p.attn_fp8_twins,
        Fp8TwinSet::NONE,
        "there are no FP8 weights to transpose on a non-FP8 layer"
    );
}

#[test]
fn the_ffn_and_attention_overlays_are_decided_independently() {
    // `proj_is_native_fp8` is checked per projection family, so a checkpoint
    // can ship a native-FP8 FFN beside BF16-dequant attention.
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        attn_fp8: false,
        ..h100_default()
    });
    assert!(!p.ffn_nvfp4);
    assert!(p.attn_nvfp4);

    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        ffn_fp8: false,
        ..h100_default()
    });
    assert!(p.ffn_nvfp4);
    assert!(!p.attn_nvfp4);
}

#[test]
fn single_scale_prefill_brings_the_q_and_o_fp8_twins_back() {
    // ATLAS_FP8_SINGLE_SCALE clears `fp8_blockscaled_prefill`, which makes the
    // W8A8 arms in `paged_qkv.rs:220` / `paged_oproj.rs:94` decline — prefill
    // then lands on `w8a16_gemm_n128_m128`, which reads `weight_t`/`scale_t`.
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        dispatch: GemmDispatch {
            fp8_blockscaled_prefill: false,
            ..GemmDispatch::defaults()
        },
        ..h100_default()
    });
    assert_eq!(p.attn_fp8_twins, Fp8TwinSet::ALL);
    assert!(
        !p.ffn_nvfp4,
        "the dense FFN still never reads NVFP4: `w8_gemm!` binds its \
         transposed operand to a literal None on every rung"
    );
}

#[test]
fn a_target_without_the_w8a8_kernels_keeps_every_fp8_twin() {
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        w8a8_kernels: false,
        ..h100_default()
    });
    assert_eq!(p.attn_fp8_twins, Fp8TwinSet::ALL);
}

#[test]
fn the_q_transpose_lever_keeps_only_the_q_twin() {
    // `cache_skip_qkv.rs:142` reads ATLAS_ATTN_PREFILL_Q_T per prefill and is
    // NOT memoised, so the loader has to mirror it or free a twin a later
    // getenv re-enables.
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        attn_prefill_q_t: true,
        ..h100_default()
    });
    assert_eq!(
        p.attn_fp8_twins,
        Fp8TwinSet {
            q: true,
            k: true,
            v: true,
            o: false,
        }
    );
}

#[test]
fn every_cutlass_nvfp4_attention_lever_keeps_the_nvfp4_attention_copies() {
    for set in [
        |d: &mut GemmDispatch| d.cutlass_nvfp4_gemm = true,
        |d: &mut GemmDispatch| d.cutlass_nvfp4_attn_q = true,
        |d: &mut GemmDispatch| d.cutlass_nvfp4_attn_kv = true,
        |d: &mut GemmDispatch| d.cutlass_nvfp4_attn_o = true,
    ] {
        let mut dispatch = GemmDispatch::defaults();
        set(&mut dispatch);
        let p = DenseFp8Plan::resolve(DenseFp8Inputs {
            dispatch,
            ..h100_default()
        });
        assert!(
            p.attn_nvfp4,
            "a CUTLASS NVFP4 attention lever reads the transposed NVFP4 twin: {dispatch:?}"
        );
    }
}

#[test]
fn the_umbrella_nvfp4_flag_builds_no_fp8_twins() {
    // `transpose_fp8_for_prefill` itself early-returns under this flag
    // (`prefill_weights.rs:357`); the plan must agree or the two drift.
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        dispatch: GemmDispatch {
            cutlass_nvfp4_gemm: true,
            ..GemmDispatch::defaults()
        },
        ..h100_default()
    });
    assert_eq!(p.attn_fp8_twins, Fp8TwinSet::NONE);
    assert!(p.attn_nvfp4);
}

#[test]
fn attn_w4a4_keeps_the_nvfp4_o_proj() {
    // `prefill/paged_oproj.rs:38-42` builds its W4A4 arm with NO weight-type
    // predicate and feeds it `&self.attn.o_proj`, so a NULL o_proj under this
    // lever is a NULL-pointer kernel launch. (The QKV side at
    // `paged_qkv.rs:51` does check `as_nvfp4()` and is already closed.)
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        attn_w4a4: true,
        ..h100_default()
    });
    assert!(p.attn_nvfp4);
}

#[test]
fn an_ssm_only_lever_does_not_resurrect_the_attention_copies() {
    // `cutlass_nvfp4_ssm_out` is deliberately not implied by the umbrella flag
    // and does not touch attention — see `ops/dispatch_config.rs`.
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        dispatch: GemmDispatch {
            cutlass_nvfp4_ssm_out: true,
            ..GemmDispatch::defaults()
        },
        ..h100_default()
    });
    assert!(!p.attn_nvfp4);
}

#[test]
fn attn_nvfp4_does_not_depend_on_the_w8a8_kernels() {
    // `RouteEnv::attn_nvfp4` is asked BEFORE the layer exists, so it passes a
    // placeholder for the layer-local handle. That is only sound while this
    // holds.
    for w8a8_kernels in [true, false] {
        for attn_fp8 in [true, false] {
            let a = DenseFp8Plan::resolve(DenseFp8Inputs {
                attn_fp8,
                w8a8_kernels,
                ..h100_default()
            });
            let b = DenseFp8Plan::resolve(DenseFp8Inputs {
                attn_fp8,
                w8a8_kernels: !w8a8_kernels,
                ..h100_default()
            });
            assert_eq!(a.attn_nvfp4, b.attn_nvfp4);
        }
    }
}

#[test]
fn the_escape_hatch_restores_the_pre_915_loader() {
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        keep_nvfp4: true,
        ..h100_default()
    });
    assert_eq!(
        p,
        DenseFp8Plan {
            ffn_nvfp4: true,
            attn_nvfp4: true,
            attn_fp8_twins: Fp8TwinSet::ALL,
        },
        "ATLAS_DENSE_FP8_KEEP_NVFP4 must reproduce the old footprint exactly, \
         or it is useless for bisecting a suspected gap in this table"
    );
}

#[test]
fn the_escape_hatch_cannot_invent_fp8_twins_on_a_non_fp8_layer() {
    let p = DenseFp8Plan::resolve(DenseFp8Inputs {
        keep_nvfp4: true,
        attn_fp8: false,
        ..h100_default()
    });
    assert_eq!(
        p.attn_fp8_twins,
        Fp8TwinSet::NONE,
        "there is no FP8 weight to transpose"
    );
}

/// The Qwen3.8-27B-FP8 numbers from the H100 ledger, reproduced from the
/// shapes. If these drift, the attribution in the module docs is wrong.
#[test]
fn the_h100_ledger_rows_reproduce_from_the_model_shapes() {
    const H: usize = 5120;
    const INTER: usize = 17408;
    const MIB: usize = 1024 * 1024;

    // `loaders_fp8.rs:229` + `quantized.rs:261` (packed) and `:230` + `:262`
    // (scales), 8,960 MB + 1,120 MB each across 256 allocations: 64 layers x 3
    // FFN projections plus 16 attention layers x 4.
    let ffn = dense_ffn_nvfp4_bytes(H, INTER);
    assert_eq!(
        ffn / MIB,
        3 * 2 * (42 * 2 + 5) + 3,
        "≈127.5 MiB x2 per layer"
    );
    assert_eq!(64 * ffn / MIB, 2 * (8160 + 1020));

    // 16 attention layers: q_n = 10240 (gated), kv_n = 2560, o_k = 5120.
    let attn = attn_nvfp4_bytes(10240, 2560, 5120, H);
    // Base + twin for q/k/v/o is 800 MB + 100 MB over the 16 layers; the fused
    // q|k|v twin is the remainder and is NOT part of the ledger's 8,960 row.
    let paired = 2 * (nvfp4_bytes(10240, H) + 2 * nvfp4_bytes(2560, H) + nvfp4_bytes(H, 5120));
    assert_eq!(16 * paired / MIB, 2 * (800 + 100));
    assert!(attn > paired, "the fused twin is extra");

    // `quantized.rs:643`, 1,600 MB x64 = the four FP8 prefill twins x16 layers.
    let twins = attn_fp8_twin_bytes(Fp8TwinSet::ALL, 10240, 2560, 5120, H);
    assert_eq!(16 * (twins / MIB), 1600, "100 MiB of FP8 twins per layer");

    // What the default route now declines: the whole FFN NVFP4 family, the
    // whole attention NVFP4 family, and the Q+O FP8 twins.
    let declined =
        64 * ffn + 16 * attn + 16 * (twins - attn_fp8_twin_bytes(KV_ONLY, 10240, 2560, 5120, H));
    assert!(
        (22.5..23.5).contains(&(declined as f64 / 1e9)),
        "expected ~23.1 GB of the sweep's 28.01 GB not to be built, got {} GB",
        declined as f64 / 1e9
    );
}

#[test]
fn the_summary_line_names_the_twins_it_built() {
    let mut r = DerivedResidency::default();
    r.keep(3_840 * 1024 * 1024);
    r.skip(20_000 * 1024 * 1024);
    r.free(1_024 * 1024 * 1024);
    r.twins.ssm_fp8_concat = true;
    let line = r.summary(28_747 * 1024 * 1024);
    assert!(
        line.starts_with("native FP8 dense residency: weights "),
        "{line}"
    );
    assert!(line.contains("(twins: ssm-qkvz-fp8)"), "{line}");
    assert!(line.contains("not built "), "{line}");
}

#[test]
fn no_twins_reads_as_none_not_as_an_empty_list() {
    assert_eq!(TwinsBuilt::default().describe(), "none");
    assert_eq!(
        TwinsBuilt {
            ffn_nvfp4: true,
            attn_fp8: true,
            ..TwinsBuilt::default()
        }
        .describe(),
        "ffn-nvfp4+t, attn-fp8-t"
    );
}
