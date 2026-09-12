# RST session — C1 0.40B BF16 CPU forward

CHARTER
-----------------------------------------------
Find whether the 0.40B BF16 twin can be bound and greedy-decoded on the S1 CPU refs (embed → 8 layers → norm → lm_head) without copying GDN/Mamba CUDA, and whether a planted graph mutant fails the golden compare.

#AREAS
C1
KimiK3WeightLoader
atlas-core CPU greedy

ORACLE
HF `inference-optimization/Kimi-K3-0.40B` greedy 128 tok × 8 prompts — goldens file `docs/k3/goldens/kimi-k3-0.40b-greedy.json` (skip if missing).
CPU self-consistency on Mac: `cargo test -p atlas-core --lib kimi_k3`.
Known-bad (instrument failed first):
- AttnRes mix=0 changes greedy tokens vs mix=1 on the synthetic twin
- Force expert 0 changes greedy tokens vs true top-k
- `weight_packed` in the store → `S5 MXFP4 not this slice` (not a silent BF16 bind)

#NOTES
- Loader `load_layers` / embed / final_norm / lm_head bind unpacked `.weight` for the twin. Official packed experts stay S5.
- GPU `TransformerLayer::decode` still refuses (C1 is the atlas-core CPU graph). spark-model tests need Linux (`posix_fallocate`).
- Goldens file is in-tree (`docs/k3/goldens/kimi-k3-0.40b-greedy.json`, 8×128). Engine compare vs HF is skipped without `K3_TWIN` safetensors→`K3CpuModel` ingest (not this slice).
- Twin names: `language_model.model.layers.*`, 8 experts, no `weight_packed`.

#BUGS
#N/A this slice for synthetic RST. Token-exact vs HF still blocked on goldens + weight ingest.

STOP
Charter complete for C1 CPU bind + greedy + known-bad. HF token-exact parked until goldens + twin weights on the host.
