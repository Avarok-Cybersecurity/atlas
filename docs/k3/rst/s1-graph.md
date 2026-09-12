# RST session — S1 graph (in flight)

CHARTER
-----------------------------------------------
Find whether a Kimi K3 decoder layer can be assembled from KDA, gated NoPE MLA, AttnRes, SiTU-GLU, and LatentMoE without copying GDN/Mamba kernels, and whether a planted graph mutant diverges from HF on the 0.40B twin.

#AREAS
S1
C1
kda-mla-attnres-situ-moe

ORACLE
HF `inference-optimization/Kimi-K3-0.40B` greedy 128 tok × 8 prompts (C1).
Known-bad: zero AttnRes mix / force expert 0 / skip one KDA conv must change tokens vs HF.

STOP
Not yet. Graph implementer + HF goldens running.
