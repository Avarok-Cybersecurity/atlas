# RST session — feat/k3-attnres (extracted)

CHARTER
-----------------------------------------------
Find whether mix=0 is identity skip and mix=1 matches the recorded softmax fixture, and that they diverge.

AREAS
feat/k3-attnres
C5 (twin greedy mix=0 vs 1459 lives on umbrella `c4-c5-c6-cpu.md`)

ORACLE
Recorded mix=1 vector (atol 1e-5). mix=0 == skip source.

KNOWN-BAD
`mix_zero_vs_mix_one_diverges` — mix=0 ≠ mix=1, max-abs > 0.5.

TEST NOTES
`cargo test -p atlas-core --lib -- kimi_k3::attnres`
Umbrella twin: mix=1 first id 1459, mix=0 moves it.

BUGS
#N/A this slice.

TEST NOTES (review 1079)
`AttnResHub::decode` drops the key on `Err` (umbrella `940bd4eeb`). Last-layer success still `remove`s. Known-bad: `hub_drops_entry_on_decode_err`.

STOP
Charter complete for AttnRes mix + hub error path. Twin greedy remains umbrella until cpu_forward extract.
