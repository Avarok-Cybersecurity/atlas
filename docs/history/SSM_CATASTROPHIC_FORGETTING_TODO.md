# SSM Catastrophic Forgetting — Mitigations TODO

**Problem**: Hybrid SSM+attention models (Qwen3.5, Jamba, etc.) suffer from SSM state saturation on long conversations. The recurrent hidden state has finite capacity — once "full", the model converges to a fixed-point that produces repetition loops or empty responses.

**Observed symptoms in Atlas**:
- Model generates the same text repeatedly ("I see the issue - cargo is not in the PATH. Let me try running it directly:")
- Model produces 2-token empty responses (`<think>` + EOS) after long conversation context
- Model generates `<system-reminder>` tag loops (regurgitating prompt metadata)
- Degradation correlates with conversation length, not prompt complexity

**Root cause**: "Stuffed Mamba" (arXiv 2410.07145, ACL 2025) — SSM state channels exhibit **state explosion** when processing sequences longer than a threshold determined by state size. The model "fails to forget earlier tokens when there is more information than it can remember."

**Architecture context**: Qwen3.5-122B has 48 layers: 36 GDN/SSM + 12 full attention (3:1 ratio). The 12 attention layers can still attend to full KV cache, but 36 SSM layers dominate and their state saturates first.

---

## Mitigation 1: Repetition Penalty (Low effort)

**What**: Apply a small repetition penalty (1.05-1.1) to decode sampling when tools are active in the Anthropic handler. Claude Code doesn't set one, leaving the model unguarded against repetition.

**Where**: `crates/spark-server/src/anthropic.rs` — set `repetition_penalty: 1.1` instead of `1.0` when `tools_active`.

**Expected impact**: Breaks token-level repetition loops. Does not fix root cause (state saturation) but prevents the most visible symptom.

**References**: Standard technique, used by all inference frameworks.

---

## Mitigation 2: SSM State Normalization (Medium effort)

**What**: Clamp the recurrent state norm after each decode step. When any channel's magnitude exceeds a threshold, normalize it back. Prevents state explosion from cascading.

**Where**: `crates/spark-model/src/model.rs` — after each SSM layer forward pass in the decode path, add a state norm check and clamp.

**Expected impact**: Prevents state explosion entirely. May slightly degrade short-context quality if threshold is too aggressive.

**References**: "Stuffed Mamba" Section 4.1 — "State Normalization: cap the state norm after each recurrent update"

---

## Mitigation 3: Increased Decay at Inference Time (Medium effort)

**What**: Scale the SSM decay parameter (alpha_t / A_log) at inference time to force faster forgetting. Reduces insertion strength (B_t) or increases decay rate so old information is discarded more aggressively.

**Where**: `crates/spark-model/src/model.rs` — modify the SSM recurrence to apply a decay multiplier during long-context decode.

**Expected impact**: Prevents state saturation by ensuring old context is forgotten. Trade-off: model loses ability to reference very old conversation context (but attention layers still can).

**References**: "Stuffed Mamba" Section 4.2 — "Forget More, Remember Less"

---

## Mitigation 4: LongMamba Token Filtering (High effort)

**What**: Classify SSM hidden channels as "local" (short receptive field) vs "global" (long receptive field). For global channels, filter out unimportant tokens before state accumulation — only "critical" tokens update the global state.

**Where**: New module in `crates/spark-model/` — requires analyzing channel receptive fields and implementing a token importance scorer.

**Expected impact**: Significant improvement in long-context coherence. Prevents unimportant tokens (boilerplate, repetitive tool results) from consuming state capacity.

**References**: "LongMamba: Enhancing Mamba's Long-Context Capabilities" (ICLR 2025)

---

## Mitigation 5: Sliding Window via State Difference (High effort)

**What**: Maintain offset SSM states to simulate a windowed view. Store state snapshots at regular intervals and compute the "state difference" to effectively create a sliding window over the recurrent state.

**Where**: `crates/spark-model/src/model.rs` + `crates/spark-runtime/src/` — extends Marconi snapshot infrastructure.

**Expected impact**: Prevents state saturation by bounding the effective context the SSM state encodes. Attention layers handle retrieval beyond the window.

**References**: "Stuffed Mamba" Section 4.3 — "Sliding Window via State Difference"

---

## Mitigation 6: Periodic State Reset During Decode (Low effort, crude)

**What**: Reset SSM recurrent state to zeros every N decode tokens. Crude but effective — the model essentially "forgets" SSM context periodically and relies on attention layers for continuity.

**Where**: `crates/spark-server/src/scheduler.rs` — in the decode loop, after every N tokens, zero out the SSM state for the active sequence.

**Expected impact**: Breaks repetition loops immediately. Causes brief quality degradation at reset boundaries. Only suitable as a last-resort safety net.

**References**: Not formally studied but implied by sliding window approaches.

---

## Mitigation 7: Complex-Valued SSM States (Architecture change, very high effort)

**What**: Replace real-valued diagonal SSM with complex-valued exponential-trapezoidal SSM (Mamba-3 approach). Complex eigenvalues can represent rotational dynamics, effectively doubling state capacity without increasing state size.

**Where**: Would require new CUDA kernels in `crates/atlas-ssm/` — fundamental architecture change.

**Expected impact**: Near-perfect state tracking (100% vs 0.9% on parity tasks). Would require model retraining.

**References**: "Mamba-3: Improved Sequence Modeling using State Space Principles" (arXiv 2603.15569, ICLR 2026)

---

## Priority Order

1. **Repetition penalty** — immediate, zero-risk
2. **SSM state normalization** — medium effort, high impact, no quality trade-off if tuned well
3. **Increased decay** — medium effort, good for long conversations
4. **Periodic state reset** — low effort safety net for production
5. **LongMamba token filtering** — high effort but most principled fix
6. **Sliding window state** — high effort, extends Marconi infrastructure
7. **Complex-valued SSM** — architecture change, requires retraining

---

## Key Papers

- [Stuffed Mamba: State Collapse and State Capacity](https://arxiv.org/abs/2410.07145) — ACL 2025
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) — ICLR 2025
- [Mamba-3: Improved Sequence Modeling](https://arxiv.org/abs/2603.15569) — ICLR 2026
- [LongMamba: Training-Free Receptive Field Enlargement](https://proceedings.iclr.cc/paper_files/paper/2025/file/ab5d50d269e52f8eed497062311ff173-Paper-Conference.pdf) — ICLR 2025
- [Rethinking Long-Range Dependency in Mamba/SSM](https://arxiv.org/abs/2509.04226)
- [Characterizing SSM and Hybrid Model Performance with Long Context](https://arxiv.org/abs/2507.12442)
