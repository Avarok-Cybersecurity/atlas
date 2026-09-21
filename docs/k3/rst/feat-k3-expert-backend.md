# RST session — feat/k3-expert-backend (mmap / prefetch)

CHARTER
-----------------------------------------------
Find whether routed experts can live as per-id NVMe packs (`e{id}.mxfp4`) so official MXFP4 is not a VRAM floor, and whether a missing pack fails closed.

AREAS
K3_EXPERT_BACKEND
mmap
prefetch
C0 96-shard map (names only; no download)

ORACLE
Dummy pack magic `K3E1` + id + payload round-trips through `MmapExpertStore`.
Unset env is `resident` (twin).

KNOWN-BAD
`missing_pack_does_not_silent_zero` — get(1) when only e0 exists errors with `no silent host F32`.

TEST NOTES
`cargo test -p avarok-core --lib -- kimi_k3::expert_backend`
Do not download moonshotai/Kimi-K3.

BUGS
#N/A this slice.

STOP
Charter complete for dummy mmap packs. Prefetch is last-router hint only (no IO thread yet).
