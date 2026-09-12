# RST session — S1 graph (in flight)

CHARTER
-----------------------------------------------
Find whether a Kimi K3 decoder layer can be assembled from KDA, gated NoPE MLA, AttnRes, SiTU-GLU, and LatentMoE without copying GDN/Mamba kernels, and whether a planted graph mutant diverges from HF on the 0.40B twin.

#AREAS
S1
C1
kda-mla-attnres-situ-moe

ORACLE
HF `inference-optimization/Kimi-K3-0.40B` greedy 128 tok × 8 prompts (C1) — goldens file not in-tree yet; C1 skeleton skips if missing and `#[ignore]`s the engine compare.
CPU closed-form / self-consistency on Mac: `cargo test -p atlas-core --lib kimi_k3`.
Known-bad (instrument failed first, then green):
- SiTU β=0 / SwiGLU mutant diverges from β=4/25 closed form
- AttnRes mix=0 is identity skip; mix=0 vs mix=1 diverges
- KDA prefix-hit then wrong conv/recurrent slot diverges
- LatentMoE force expert 0 diverges from true top-k

#NOTES
- Graph from official parser: 0.40B → 6 KDA + last MLA (layers 0–2 KDA, 3 MLA, 4–6 KDA, 7 MLA); layer 0 dense. Official 93 / 69 KDA / 24 MLA.
- KDA is a new CPU backend (full-rank `g_proj`, bound −5, conv 4, head_dim 128). No `.cu` under `kernels/gb10/kimi-k3`.
- Loader BF16 twin bind is C1 (`docs/k3/rst/c1-cpu-forward.md`). Numeric refs in `atlas-core` (Mac) and re-exported from `spark-model/src/kimi_k3/`.
- Observed 2026-09-12 workstation (S1): 22 passed, 1 ignored (C1 engine), 0 failed.

#BUGS
#N/A this slice (CPU graph only). C1 token-exact is blocked on missing `docs/k3/goldens/kimi-k3-0.40b-greedy.json` and a bound forward.

STOP
Charter complete for S1 CPU graph. C1 engine compare parked until goldens + weights.
