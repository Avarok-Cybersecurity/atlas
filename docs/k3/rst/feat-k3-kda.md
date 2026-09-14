# RST session — feat/k3-kda (extracted)

CHARTER
-----------------------------------------------
Find whether the extracted KDA CPU ref + unique `kda_decode` stem are the FLA-unbounded backend C1 needed, and that they are **not** a GDN/Mamba copy.

AREAS
feat/k3-kda
kda-ref
kda_decode.cu
C1 (oracle lives on the umbrella twin run)

ORACLE
- Unit: `bounded_gate(..., None)` is `-exp(A_log)*softplus` (FLA default). Twin JSON omits `gate_lower_bound`.
- Twin C1: `docs/k3/rst/c1-engine.md` on the umbrella (0.40B greedy vs HF). This PR extracts the KDA math that made first-token 1459 hold.
- CUDA: module stem `kda_decode`, entries `k3_kda_conv_update_f32` + `k3_kda_recurrent_step_f32`. Host launch looks them up.

KNOWN-BAD (instrument actually failed, then the formula was kept)
- `omitted_lower_bound_is_neg_exp_a_softplus`: `Some(-5)` vs `None` max-abs > 0.1. Guessing `-5` on the twin was a C1 close-race.
- `prefix_hit_wrong_slot_diverges`: restore conv/recurrent from the previous token, next decode moves.
- `beta_zero_is_sigmoid_half_not_zero`: raw beta=0 still writes delta (`sigmoid(0)=0.5`).

TEST NOTES
`cargo test -p atlas-core --lib -- kimi_k3::kda --nocapture`
`cargo test -p atlas-core --lib -- parse_kimi_k3_0_40b_twin parse_kimi_k3_official_config`
Twin parse: `linear_gate_lower_bound == 0.0` (omitted key, not `-5`). Official: `-5.0`.

Do not copy GDN/Mamba into this directory. `kda_decode.cu` is a new stem.

BUGS
#N/A this slice. Full C1 until-EOS remains the umbrella twin run.

STOP
Charter complete for the extracted KDA module. Hybrid cache / MLA / AttnRes / LatentMoE / TP are later feat/k3-* PRs. Do not merge the umbrella.
