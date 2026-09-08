#!/usr/bin/env python3
"""GLM-5.3-Flash KDA golden-vector generator.

Runs the REAL HuggingFace ``transformers`` 5.16.1 ``glm5_next`` implementation on a tiny
deterministic fixture and emits every intermediate tensor as JSON, so an Atlas-side reference
implementation can be checked sub-op by sub-op without a GPU.

Source of truth
---------------
``transformers/models/glm5_next/modeling_glm5_next.py`` from the PyPI wheel
``transformers-5.16.1-py3-none-any.whl``.
  modeling_glm5_next.py   sha256 2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b
  configuration_glm5_next.py sha256 b62936c9f5d74b09ea511f3211abde65f97a9be1cb3c8f10fdad7c879850fa72

Nothing here re-derives an equation. Every number comes from calling HF's own
``Glm5NextTextForgetGate``, ``Glm5NextTextRMSNormGated``, ``l2norm``,
``recurrent_kimi_delta_attention`` and ``chunk_kimi_delta_attention``.

Config stub
-----------
``Glm5NextTextForgetGate.__init__`` reads exactly four config attributes
(``linear_head_dim``, ``linear_num_heads``, ``hidden_size``, ``linear_lower_bound``), so a
SimpleNamespace stub is used instead of a full ``Glm5NextTextConfig``. This avoids dragging in the
45-layer production geometry and is verifiable by reading the constructor.

Determinism
-----------
No RNG. Every input tensor is built by ``_fixed`` from an integer index, and every input is written
into the golden file alongside the outputs, so the consumer never has to reproduce the generator's
arithmetic.
"""

import hashlib
import json
import math
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    Glm5NextTextForgetGate,
    Glm5NextTextRMSNormGated,
    chunk_kimi_delta_attention,
    l2norm,
    recurrent_kimi_delta_attention,
)

# ---------------------------------------------------------------- fixture geometry
# Mirrors the production shape RELATIONSHIPS at toy size:
#   production: hidden=4096, num_heads=64, head_dim=128, low-rank width == head_dim
#   fixture:    hidden=8,    num_heads=2,  head_dim=4,   low-rank width == head_dim
HIDDEN = 8
H = 2  # linear_num_heads
D = 4  # linear_head_dim  (== low-rank width for f_a/f_b and g_a/g_b)
T = 6  # tokens -- >1 so the recurrent state transition is exercised, not just one step
B = 1
LOWER_BOUND = -5.0  # config linear_attn_config.gate_lower_bound
RMS_EPS = 1e-5  # config rms_norm_eps
CHUNK = 2  # chunk_size for the prefill path; T=6 => 3 chunks

torch.use_deterministic_algorithms(True)


def _fixed(shape, salt):
    """Deterministic, RNG-free, well-conditioned filler in roughly [-0.9, 0.9]."""
    n = 1
    for s in shape:
        n *= s
    vals = [math.sin(0.7 * i + 1.3 * salt) * 0.9 for i in range(n)]
    return torch.tensor(vals, dtype=torch.float32).reshape(shape)


def _dump(t):
    return t.detach().to(torch.float32).flatten().tolist()


def _entry(t):
    return {"shape": list(t.shape), "data": _dump(t)}


cfg = SimpleNamespace(
    hidden_size=HIDDEN,
    linear_head_dim=D,
    linear_num_heads=H,
    linear_lower_bound=LOWER_BOUND,
)

golden = {"fixture": {"hidden": HIDDEN, "heads": H, "head_dim": D, "tokens": T,
                      "lower_bound": LOWER_BOUND, "rms_eps": RMS_EPS, "chunk_size": CHUNK}}

# ---------------------------------------------------------------- inputs
hidden_states = _fixed((B, T, HIDDEN), 0)

# Post-conv q/k/v. The short conv is deliberately OUT of scope here: it is classed REUSE against
# Atlas's existing fused conv+SiLU+L2 kernel, and folding it in would couple two independent checks.
q_in = _fixed((B, T, H, D), 1)
k_in = _fixed((B, T, H, D), 2)
v_in = _fixed((B, T, H, D), 3)

W_f_a = _fixed((D, HIDDEN), 4)          # f_a_proj.weight  [out=D, in=HIDDEN]
W_f_b = _fixed((H * D, D), 5)           # f_b_proj.weight  [out=H*D, in=D]
dt_bias = _fixed((H * D,), 6)           # per-CHANNEL, length H*D -- not per-head
A_log = _fixed((H,), 7)                 # per-HEAD, length H
W_b = _fixed((H, HIDDEN), 8)            # b_proj.weight    [out=H, in=HIDDEN]
W_g_a = _fixed((D, HIDDEN), 9)          # g_a_proj.weight
W_g_b = _fixed((H * D, D), 10)          # g_b_proj.weight
o_norm_w = _fixed((D,), 11)             # o_norm.weight
W_o = _fixed((HIDDEN, H * D), 12)       # o_proj.weight

golden["inputs"] = {
    "hidden_states": _entry(hidden_states),
    "q_in": _entry(q_in), "k_in": _entry(k_in), "v_in": _entry(v_in),
    "W_f_a": _entry(W_f_a), "W_f_b": _entry(W_f_b), "dt_bias": _entry(dt_bias),
    "A_log": _entry(A_log), "W_b": _entry(W_b),
    "W_g_a": _entry(W_g_a), "W_g_b": _entry(W_g_b),
    "o_norm_w": _entry(o_norm_w), "W_o": _entry(W_o),
}

# ---------------------------------------------------------------- 1. q/k L2 normalisation
# HF: l2norm(x) = x / sqrt(sum(x^2) + 1e-6)   (NOT max(norm, eps) -- matches the FLA triton form)
q_l2 = l2norm(q_in.float(), dim=-1, eps=1e-6)
k_l2 = l2norm(k_in.float(), dim=-1, eps=1e-6)

# ---------------------------------------------------------------- 2. q scale = 1/sqrt(d)
scale = 1.0 / (D**0.5)
q_scaled = q_l2 * scale

# ---------------------------------------------------------------- 3-6. forget gate (HF module)
fg = Glm5NextTextForgetGate(cfg)
with torch.no_grad():
    fg.f_a_proj.weight.copy_(W_f_a)
    fg.f_b_proj.weight.copy_(W_f_b)
    fg.dt_bias.copy_(dt_bias)
    fg.A_log.copy_(A_log)
    gate = fg(hidden_states)  # [B, T, H, D] -- bounded gate, per (head, channel)

# Intermediates re-expressed from the same weights so each sub-op is separately checkable.
g_lowrank = (hidden_states @ W_f_a.T) @ W_f_b.T            # 3. f_a/f_b low-rank projection
g_biased = (g_lowrank.float() + dt_bias.float()).view(B, T, H, D)  # 4. per-channel dt_bias
decay = torch.exp(A_log.float())                           # 5. exp(A_log), per head

# ---------------------------------------------------------------- 7. beta
beta = torch.sigmoid(hidden_states @ W_b.T)                # [B, T, H]

# ---------------------------------------------------------------- 8. recurrent state update
# Fed the RAW q/k/v: the HF kernels do the l2norm + scale internally
# (use_qk_l2norm_in_kernel=True, scale = 1/sqrt(d_q)).
core_rec, state_rec = recurrent_kimi_delta_attention(
    q_in.clone(), k_in.clone(), v_in.clone(),
    g=gate, beta=beta, initial_state=None, output_final_state=True,
    use_qk_l2norm_in_kernel=True,
)

# Prefill formulation over the same fixture -- must agree with the recurrent one.
core_chunk, state_chunk = chunk_kimi_delta_attention(
    q_in.clone(), k_in.clone(), v_in.clone(),
    g=gate, beta=beta, chunk_size=CHUNK, initial_state=None,
    output_final_state=True, use_qk_l2norm_in_kernel=True,
)

# Same prefill with chunk_size=4: T=6 is not a multiple of 4, so this exercises the pad-to-chunk
# path (pad_size=2) that chunk_size=2 never touches.
core_chunk_c4, state_chunk_c4 = chunk_kimi_delta_attention(
    q_in.clone(), k_in.clone(), v_in.clone(),
    g=gate, beta=beta, chunk_size=4, initial_state=None,
    output_final_state=True, use_qk_l2norm_in_kernel=True,
)

# Two-segment continuation: prefill 4 tokens, then decode tokens 4 and 5 one at a time off the
# carried state. Proves the state transition, not just a single step.
core_pre, state_pre = chunk_kimi_delta_attention(
    q_in[:, :4].clone(), k_in[:, :4].clone(), v_in[:, :4].clone(),
    g=gate[:, :4], beta=beta[:, :4], chunk_size=CHUNK, initial_state=None,
    output_final_state=True, use_qk_l2norm_in_kernel=True,
)
carry = state_pre
step_outs = []
for t in range(4, T):
    o_t, carry = recurrent_kimi_delta_attention(
        q_in[:, t : t + 1].clone(), k_in[:, t : t + 1].clone(), v_in[:, t : t + 1].clone(),
        g=gate[:, t : t + 1], beta=beta[:, t : t + 1],
        initial_state=carry, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    step_outs.append(o_t)
core_split = torch.cat([core_pre] + step_outs, dim=1)

# ---------------------------------------------------------------- 10. low-rank output gate
out_gate = ((hidden_states @ W_g_a.T) @ W_g_b.T).view(B, T, H, D)

# ---------------------------------------------------------------- 9. RMSNormGated + sigmoid
o_norm = Glm5NextTextRMSNormGated(D, eps=RMS_EPS)
with torch.no_grad():
    o_norm.weight.copy_(o_norm_w)
    normed = o_norm(core_rec, out_gate)

layer_out = normed.reshape(B, T, H * D) @ W_o.T  # o_proj

golden["outputs"] = {
    "q_l2": _entry(q_l2),
    "k_l2": _entry(k_l2),
    "q_scaled": _entry(q_scaled),
    "g_lowrank": _entry(g_lowrank),
    "g_biased": _entry(g_biased),
    "decay": _entry(decay),
    "gate": _entry(gate),
    "beta": _entry(beta),
    "core_recurrent": _entry(core_rec),
    "state_recurrent": _entry(state_rec),
    "core_chunked": _entry(core_chunk),
    "state_chunked": _entry(state_chunk),
    "core_chunked_c4_padded": _entry(core_chunk_c4),
    "state_chunked_c4_padded": _entry(state_chunk_c4),
    "core_split_prefill_then_decode": _entry(core_split),
    # HF-produced carried state after the 4-token prefill. Lets a decode-only kernel be
    # driven from HF's own state rather than from a reproduction of it.
    "state_after_prefill4": _entry(state_pre),
    "core_prefill4": _entry(core_pre),
    "out_gate": _entry(out_gate),
    "o_norm_out": _entry(normed),
    "layer_out": _entry(layer_out),
}

# Internal consistency: HF's own prefill and decode formulations must already agree.
golden["hf_self_checks"] = {
    "chunk_vs_recurrent_max_abs": float((core_chunk - core_rec).abs().max()),
    "chunk_vs_recurrent_state_max_abs": float((state_chunk - state_rec).abs().max()),
    "split_vs_full_max_abs": float((core_split - core_rec).abs().max()),
    "chunk_c4_padded_vs_recurrent_max_abs": float((core_chunk_c4 - core_rec).abs().max()),
}

blob = json.dumps(golden, indent=1, sort_keys=True)
with open("/w/kda_golden.json", "w") as fh:
    fh.write(blob)

print("transformers", __import__("transformers").__version__)
print("torch", torch.__version__)
print("golden sha256", hashlib.sha256(blob.encode()).hexdigest())
print("hf_self_checks", golden["hf_self_checks"])
