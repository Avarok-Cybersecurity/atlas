# RST session — feat/k3-mla-gated (extracted)

CHARTER
-----------------------------------------------
Find whether gated NoPE MLA is a distinct mixer (RoPE slots exist but are not rotated when `mla_use_nope`), and whether the official layer map ends with two MLA layers.

AREAS
feat/k3-mla-gated
C1/C4 (twin logits on umbrella)

ORACLE
- `MlaConfig::twin_0_40b` / `production` geometry.
- Official graph: 24 MLA, last two (0-based 91 and 92) MLA.
- CUDA stem `mla_decode` (`k3_mla_*`). Default on; `K3_CUDA_MLA=0` CPU.

KNOWN-BAD
- `output_gate_mutates`: gate off == identity; gate on diverges.
- `maybe_rope` with `use_nope=true` leaves q/k unchanged (covered by `gated_nope_path_runs` vs rotate).

TEST NOTES
`cargo test -p avarok-core --lib -- kimi_k3::mla official_graph_census`

BUGS
#N/A this slice.

TEST NOTES (review 1078)
Host launch still D2H q/k and re-uploads KV (CPU oracle). Certified tok/s: `MlaDeviceKv` + `launch_k3_mla_decode_token_on_device` (one D2H: output). Known-bad: `device_kv_second_token_d2h_is_output_only`.

STOP
Charter complete for gated NoPE MLA. AttnRes / LatentMoE / TP later.
