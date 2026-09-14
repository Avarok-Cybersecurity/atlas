# RST session — C4/C5/C6 CPU fixtures (not rental-green)

CHARTER
-----------------------------------------------
Find whether hybrid state, AttnRes mix, and LatentMoE frozen-gate mix are self-consistent on the CPU graph. Not HF, not GPU.

KNOWN-BAD
C4: wrong KDA conv slot / wrong MLA kv row after prefix hit.
C5: mix=0 vs recorded.
C6: force expert 0.

TEST NOTES
`2c0191cd6`. Mac `kimi_k3` **50 passed**. Tests:
- `c4_hybrid_state_prefix_hit_matches_cold_prefill_{tiny,small}`
- `c4_wrong_kda_conv_slot_after_prefix_hit_diverges`
- `c4_wrong_mla_kv_row_after_prefix_hit_diverges`
- `c5_attnres_mix_matches_recorded_fixture` / `c5_zero_mix_weights_diverges_from_recorded`
- `c6_frozen_gates_topk_and_mix_match_recorded` / `c6_force_expert_zero_diverges`

TEST NOTES (spark2, 2026-09-13, `K3_TWIN` 0.40B, `83c0d34`)
- `c4_hybrid_state_prefix_hit_matches_cold_prefill_twin` **ok** — aviation prefix + first id 1459. Known-bad: wrong KDA conv slot and wrong MLA kv row after prefix hit both diverged.
- `c5_twin_mix0_is_skip_and_diverges_from_mix1` **ok** — mix=1 first generated id **1459** (C1). mix=0 moved the first token. Not the one-hot stub.
- `c6_twin_force_expert_zero_diverges` **ok** — frozen-gate top-1 is expert 1; force expert 0 changes the mix. Greedy-8 `force_expert` did **not** move aviation ids; mix-level mutant is the instrument.

STOP
C4–C6 **green** on the C1 twin. GPU kernel TODO remains; not required to check these boxes.
