# BF16 ground-truth result: the forward pass is NOT the bug

Method: downloaded the original BF16 base `Qwen/Qwen3.6-35B-A3B` (full precision,
same weights the FP8 checkpoint quantizes), ran a forward (layer-truncated to 16
layers to fit unified memory — layer N depends only on 0..N, so L0..15 are
bit-exact), op-hooked the last-token hidden states, and compared BOTH Atlas-FP8
and vLLM-FP8 against this ground truth (same 149 tokens). `threeway_vs_bf16_truth.py`.

## Result — Atlas is CLOSER to truth than vLLM at every layer and op

Per-layer residual rel_l2 vs BF16 truth (lower = more accurate):
  L3  Atlas 0.046  vLLM 0.065      L7  Atlas 0.041  vLLM 0.052
  L11 Atlas 0.049  vLLM 0.059      L15 Atlas 0.052  vLLM 0.062
  -> ATLAS closer at ALL 16 layers.
o_proj (attention output) rel_l2 vs truth:
  L3 Atlas 0.059 / vLLM 0.071 ; L11 Atlas 0.083 / vLLM 0.102 -> ATLAS closer.
Atlas q/k/v proj + k-norm vs truth: 0.014 / 0.034 / 0.039 / 0.036 (all tight).

## Conclusion

Atlas's forward pass — attention, RoPE, q/k-norm ((1+w), matches HF
Qwen3_5MoeRMSNorm), projections (W8A16), MoE, SSM — is FAITHFUL and MORE accurate
than vLLM's W8A8. The Atlas-vs-vLLM divergence measured earlier (and localized to
the attention path) was vLLM's lower precision, NOT an Atlas bug. There is no
kernel/numerical bug.

The tool-calling failure is therefore POST-logits: the thinking->tool-call
transition is a knife-edge decision where ~0.05-level differences (inherent to any
FP8 engine; present in both Atlas and vLLM) flip a low-margin token. Atlas lands in
the narrate-then-malformed basin more often than vLLM, despite being more accurate.

## Fix

Not a kernel change. Constrain the output structure so a flipped low-margin token
cannot malform: force the qwen3_coder tool grammar to engage right after `</think>`
on tool-armed turns (non-triggered grammar). Keeps thinking on. thinking_in_tools=
false works for the same reason (skips the brittle transition) but is the blunt version.

Earlier KERNEL_LOCALIZATION.md conclusion ("full-attention kernel is the divergent
op") is SUPERSEDED by this: that divergence was vLLM-relative; vs ground truth Atlas
is the more accurate engine.
