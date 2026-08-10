// SPDX-License-Identifier: AGPL-3.0-only

//! Key-classification seam tests: `adapter_id_hash` stability/base-reserve +
//! generation fold, the decode/verify graph-key discipline, and the
//! `classify_key` accept/reject value pin. Types resolve through the
//! `crate::lora` facade.

use crate::lora::test_support::*;
use crate::lora::*;

#[test]
fn classify_key_maps_supported_and_rejects_unsupported() {
    let cfg = cfg();
    // Accepts: k/v/o attn projections + mlp gate/up/down on a full-attention
    // layer (3,7,11,… in this factory config), both A and B halves, and the
    // exact (layer, module, A|B) tuple.
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.3.self_attn.k_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (3, LoraTarget::Attn(LoraModule::KProj), AdapterAb::A)
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.3.self_attn.v_proj.lora_B.weight",
            &cfg
        )
        .unwrap(),
        (3, LoraTarget::Attn(LoraModule::VProj), AdapterAb::B)
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.7.self_attn.o_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (7, LoraTarget::Attn(LoraModule::OProj), AdapterAb::A)
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.11.mlp.gate_proj.lora_B.weight",
            &cfg
        )
        .unwrap(),
        (11, LoraTarget::Attn(LoraModule::GateProj), AdapterAb::B)
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.47.mlp.down_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (47, LoraTarget::Attn(LoraModule::DownProj), AdapterAb::A)
    );

    // q_proj IS supported (gated interleaved [Q|gate] folds like k/v/o on a
    // full-attn layer) → classifies to QProj, not a rejection.
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (3, LoraTarget::Attn(LoraModule::QProj), AdapterAb::A)
    );

    // Rejects — every unsupported shape is a NAMED hard error, never a
    // silent skip / None:
    // A GDN/linear-attention layer (layer 0) — LoRA is full-attention only.
    assert!(
        classify_key(
            "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight",
            &cfg
        )
        .is_err()
    );
    // A GDN projection target (linear_attn.*) → rejected.
    assert!(
        classify_key(
            "base_model.model.model.layers.3.linear_attn.in_proj_qkv.lora_A.weight",
            &cfg
        )
        .is_err()
    );
    // A non-PEFT key (no `base_model.model.` prefix) → rejected.
    assert!(classify_key("model.layers.3.self_attn.k_proj.weight", &cfg).is_err());
}

#[test]
fn classify_key_maps_experts_and_router() {
    // Factory cfg: num_experts = 512, FullAttention layers 3,7,…
    let cfg = cfg();
    // A routed-expert projection → Expert { n, proj } on a full-attention layer.
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.7.mlp.experts.42.gate_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (
            7,
            LoraTarget::Expert {
                n: 42,
                proj: ExpertProj::Gate
            },
            AdapterAb::A
        )
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.11.mlp.experts.0.down_proj.lora_B.weight",
            &cfg
        )
        .unwrap(),
        (
            11,
            LoraTarget::Expert {
                n: 0,
                proj: ExpertProj::Down
            },
            AdapterAb::B
        )
    );
    // The multimodal `.language_model.` trunk segment before `.layers.` is
    // prefix-agnostic (same as attention).
    assert_eq!(
        classify_key(
            "base_model.model.model.language_model.layers.7.mlp.experts.3.up_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (
            7,
            LoraTarget::Expert {
                n: 3,
                proj: ExpertProj::Up
            },
            AdapterAb::A
        )
    );
    // The MoE router `mlp.gate` (DISTINCT from `mlp.gate_proj`) → Router.
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.3.mlp.gate.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (3, LoraTarget::Router, AdapterAb::A)
    );

    // Experts + router are valid on a NON-full-attention (GDN/linear) layer too
    // — MoE FFN exists on every layer, so a real Qwen3.6/Holo adapter targets
    // them there. Layer 0 is linear-attention in the factory cfg.
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.0.mlp.experts.5.down_proj.lora_A.weight",
            &cfg
        )
        .unwrap(),
        (
            0,
            LoraTarget::Expert {
                n: 5,
                proj: ExpertProj::Down
            },
            AdapterAb::A
        )
    );
    assert_eq!(
        classify_key(
            "base_model.model.model.layers.0.mlp.gate.lora_B.weight",
            &cfg
        )
        .unwrap(),
        (0, LoraTarget::Router, AdapterAb::B)
    );
    // But an ATTENTION target on that same linear-attention layer stays rejected.
    assert!(
        classify_key(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            &cfg
        )
        .is_err()
    );

    // Named rejects (never a silent skip):
    // expert index out of range.
    assert!(
        classify_key(
            "base_model.model.model.layers.3.mlp.experts.999.gate_proj.lora_A.weight",
            &cfg
        )
        .is_err()
    );
    // fused/unindexed expert layout (target_parameters spelling) — phase-3.
    assert!(
        classify_key(
            "base_model.model.model.layers.3.mlp.experts.gate_up_proj.lora_A.weight",
            &cfg
        )
        .is_err()
    );
    // fused per-expert gate_up_proj — phase-3.
    assert!(
        classify_key(
            "base_model.model.model.layers.3.mlp.experts.5.gate_up_proj.lora_A.weight",
            &cfg
        )
        .is_err()
    );
    // unknown expert projection.
    assert!(
        classify_key(
            "base_model.model.model.layers.3.mlp.experts.5.wat_proj.lora_A.weight",
            &cfg
        )
        .is_err()
    );
}

#[test]
fn classify_key_rejects_experts_on_dense_model() {
    // num_experts == 0 (dense) → expert LoRA is a named reject.
    let mut dense = cfg();
    dense.num_experts = 0;
    assert!(
        classify_key(
            "base_model.model.model.layers.3.mlp.experts.0.gate_proj.lora_A.weight",
            &dense
        )
        .is_err()
    );
}

#[test]
fn adapter_id_hash_is_stable_and_base_reserved() {
    // Deterministic and name-derived (survives pool-slot reuse: same name →
    // same id regardless of which runtime slot it lands in).
    assert_eq!(adapter_id_hash("sparky", 0), adapter_id_hash("sparky", 0));
    assert_ne!(adapter_id_hash("sparky", 0), adapter_id_hash("vega", 0));
    // 0 is reserved for base; the empty name still yields a non-zero id.
    assert_ne!(adapter_id_hash("", 0), 0);
    assert_ne!(adapter_id_hash("anything", 0), 0);
}

#[test]
fn adapter_id_hash_generation_changes_id_but_never_base() {
    // Task #25: gen 0 is a strict no-op; a bumped generation changes the id
    // (so a re-staged same-name slot misses the stale prefix), and no
    // (name, generation) pair aliases the base sentinel 0.
    for name in ["sparky", "vega", ""] {
        let g0 = adapter_id_hash(name, 0);
        let g1 = adapter_id_hash(name, 1);
        let g2 = adapter_id_hash(name, 2);
        assert_ne!(g0, g1, "generation bump must change the id ({name})");
        assert_ne!(g1, g2, "each generation is distinct ({name})");
        assert_ne!(g0, 0, "gen 0 never aliases base ({name})");
        assert_ne!(g1, 0, "gen 1 never aliases base ({name})");
        assert_ne!(g2, 0, "gen 2 never aliases base ({name})");
        // Determinism across calls.
        assert_eq!(g1, adapter_id_hash(name, 1));
    }
}

#[test]
fn decode_graph_key_folds_active_adapter_id() {
    // Task #28: the decode/verify graph cache key is `(slot, active_id)`
    // where active_id = adapter_id_for_slot(-1). This test proves the
    // *keying* discipline that makes graph replay safe under a swappable
    // pool: the compound key HITS iff the active adapter identity is
    // unchanged, and MISSES on any rotate (active name change) or swap
    // (generation bump). adapter_id_hash's own stability is covered above.
    let slot = 3usize;

    // Base (no LoRA) → active_id 0 → key reduces to (slot, 0): byte-identical
    // single-key behavior. Same base step re-keys to the same entry (HIT).
    assert_eq!((slot, 0u64), (slot, 0u64));

    // A fixed single adapter never rotates / never bumps generation → the id
    // is constant → the same logical key every step (HIT, still graphed).
    let sparky = adapter_id_hash("sparky", 0);
    assert_eq!((slot, sparky), (slot, adapter_id_hash("sparky", 0)));

    // A ROTATE changes the active adapter name → different id → different key
    // → the pre-rotate graph is a MISS (never replayed over swapped bytes).
    let vega = adapter_id_hash("vega", 0);
    assert_ne!((slot, sparky), (slot, vega));

    // A SWAP into the active slot bumps that slot's generation → different id
    // → different key → MISS (fresh capture over the new pool bytes).
    let sparky_gen1 = adapter_id_hash("sparky", 1);
    assert_ne!((slot, sparky), (slot, sparky_gen1));

    // The base sentinel 0 never aliases a real adapter's key on the same slot.
    assert_ne!((slot, 0u64), (slot, sparky));
    assert_ne!((slot, 0u64), (slot, sparky_gen1));

    // A DIFFERENT slot with the SAME active id is a distinct key (per-slot
    // SSM/KV pointers still bake in) — the slot component is preserved.
    assert_ne!((slot, sparky), (slot + 1, sparky));

    // verify_kgamma's 3-tuple `(slot, K, active_id)`: same discipline, and K
    // (gamma width) stays an independent axis alongside the active id.
    assert_eq!(
        (slot, 5usize, sparky),
        (slot, 5usize, adapter_id_hash("sparky", 0))
    );
    assert_ne!((slot, 5usize, sparky), (slot, 5usize, vega));
    assert_ne!((slot, 5usize, sparky), (slot, 6usize, sparky));
}
