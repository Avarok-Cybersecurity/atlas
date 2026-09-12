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

TEST NOTES (2026-09-12, after `e0aab722c`)
spark2 `c1_prompt0_first_token_top8` + `c1_prompt0_first_eight_generated`: **PASS** (28.5 s).
- last-pos argmax=**1459** matches HF first generated; 387 is second (top-8 includes 387 at rank 8).
- first-8 generated `[1459, 387, 1495, 2189, 261, 56207, 1765, 413]` match goldens.

Full 8×128 greedy (`c1_greedy_vs_hf_goldens_skip_if_missing`) **FAIL** at index 54 (46 generated tokens). Shared prefix through `..., 19392, 13, 163585` then ours `39058` vs HF `163585` (HF double-emits 163585 and loops Bee Movie). Ours does not collapse to 261/667 anymore.

TEST NOTES (EOS)
163585 is `[EOS]`. Every golden hits EOS inside the 128 cap (gen index 9–45). The previous 128-token FAIL was the token *after* the first EOS (HF emits a second `[EOS]` and loops; we emit `Prepare`). That is not a greedy oracle.

C1 protocol: exact match **through first EOS inclusive**, max_new=128 as a cap. Re-run all 8 prompts on spark2.

STOP
Waiting on spark2 8-prompt until-EOS. Do not check C2–C7 until that is green.
