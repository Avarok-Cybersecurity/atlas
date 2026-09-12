# RST session — C1 CPU engine vs HF goldens

CHARTER
-----------------------------------------------
Find whether the atlas-core CPU graph, loaded from the 0.40B BF16 twin, matches HF greedy token ids. It is a token-identity oracle, not a quality oracle.

#AREAS
C1
cpu-forward
kda-ref

START
2026-09-12

ORACLE
`docs/k3/goldens/kimi-k3-0.40b-greedy.json` (HF CUDA greedy). Known-bad: AttnRes mix=0 / force expert 0 must not match goldens.

KNOWN-BAD (instrument)
`rst_mutant_fails_golden_compare` **passed** on spark2 (mix=0 ≠ golden). Weak once the clean path also mismatches, but the lever still moves tokens.

TEST NOTES
spark2, `K3_TWIN=.../Kimi-K3-0.40B`, `cargo test -p atlas-core --lib kimi_k3::c1` (210 s).

Prompt 0 prefix **matches** HF tokenizer:
`[18805, 308, 799, 5624, 12524, 318, 57195, 11]` plus next id `1459`.
First mismatch is the following token: ours `13` vs HF `387`.
Our continuation then collapses to a `261/667` loop. HF stays on Bee Movie paste.

`rst_attnres_mix0_or_force_expert0_diverges` ok on synthetic. `load_bf16_twin_binds_layers` 11/11 spark-model.

BUGS
#BUG
C1 **not green**. CPU KDA/MLA/AttnRes graph does not match HF even at the first generated token. Degenerate argmax loop after ~20 tokens. Do not call this a K3 decode.

STOP
Charter complete for "does the CPU engine lock to HF". Answer: no. Next: first-token logit dump vs HF `language_model`, then KDA recurrence vs `fla.ops.kda`.
