# RST session — feat/k3-hybrid-cache (extracted)

CHARTER
-----------------------------------------------
Find whether HybridCache slots follow the twin mixer map (6 KDA + 2 MLA, last MLA), and whether prefix-clone mutants on KDA state and MLA KV rows actually move the blob.

AREAS
feat/k3-hybrid-cache
C2/C3/C4 (full twin logits live on umbrella `c2-c3-cpu.md` / `c4-c5-c6-cpu.md`)

ORACLE
- Twin JSON → `K3Graph` + `HybridCache::from_graph`: layers 0-2,4-6 KDA; 3 and 7 MLA.
- Roundtrip `LayerCache::to_bytes` / `from_bytes` for KDA and MLA.
- Umbrella twin C2/C3/C4 (spark2 `K3_TWIN`) remain the full-graph oracle.

KNOWN-BAD (instrument actually failed)
- `trash_kda_state_after_prefix_clone_diverges`: fill conv+recurrent with 7 after clone → blob ≠ prefix.
- `wrong_mla_kv_row_after_append_diverges`: swap last packed K/V row with row 0 → tensors move.

TEST NOTES
`cargo test -p avarok-core --lib -- kimi_k3::cache kimi_k3::layer`

Prefill-then-decode logits (C2) still need `cpu_forward` on the umbrella until that graph is extracted.

BUGS
#N/A this slice.

TEST NOTES (review 1077)
`DeviceHybridCache::from_host` seeds GPU buffers once (H2D). Decode must use those ptrs. Host `HybridCache` remains prefix snapshot/restore.

STOP
Charter complete for hybrid cache types + slot mutants + device-resident seed. MLA decode / AttnRes / MoE are later PRs.
