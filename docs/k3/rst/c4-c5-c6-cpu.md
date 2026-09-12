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

STOP
CPU fixtures complete. Umbrella C4–C6 stay unchecked until they run on a C1-matching / GPU path.
