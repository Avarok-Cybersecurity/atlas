#!/usr/bin/env python3
"""GLM-5.3-Flash KDA gate golden at PRODUCTION geometry (H=64, D=128).

Slice 2's `kda_golden.json` binds every KDA sub-op to HuggingFace, but at a toy
geometry (H=2, D=4). The gate kernel's remaining risk at production scale is purely
INDEXING — `dt_bias` is per (head, channel) `[H*D]` while `A_log` is per head `[H]`,
and a kernel that collapsed either axis would still pass the toy fixture if H and D
were small enough to alias. This file removes that gap.

Source of truth: the real `transformers` 5.16.1 `Glm5NextTextForgetGate`. No equation
is re-derived here.

Determinism: inputs come from an integer LCG mapped to float32 by exact division by
2^24, so there is no transcendental in the generator and no RNG. Inputs are stored in
the file regardless, so the consumer never reproduces this arithmetic.

Adversarial by construction:
  * `dt_bias` varies across BOTH d and h, with a large per-channel ramp, so a kernel
    that broadcast one bias per head produces visibly wrong numbers.
  * `A_log` is distinct per head and spans a wide range, so a kernel that used head 0's
    decay everywhere is caught.
  * `g_raw` is scaled per head so that some rows sit deep in sigmoid saturation
    (both tails) while others stay in the linear region.
"""

import hashlib
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    Glm5NextTextForgetGate,
)

H = 64  # linear_num_heads      (production)
D = 128  # linear_head_dim      (production)
T = 2  # >1 so token-major row indexing (row = t*H + h) is exercised
HIDDEN = 8  # unused by the gate itself; ForgetGate only needs it to size f_a_proj
LOWER_BOUND = -5.0


class Lcg:
    """Integer LCG -> float32 in [-1, 1). No transcendentals, no platform libm."""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next_unit(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((self.s >> 40) / (1 << 24)) * 2.0 - 1.0  # exact in binary32

    def tensor(self, *shape):
        n = 1
        for s in shape:
            n *= s
        return torch.tensor([self.next_unit() for _ in range(n)], dtype=torch.float32).reshape(shape)


rng = Lcg(0x5EED_0053)

# g_raw: per-head amplitude ramp -> heads 0..63 span linear region through deep saturation.
g_raw = rng.tensor(T, H, D)
head_amp = torch.tensor([0.5 + 1.6 * h for h in range(H)], dtype=torch.float32)
g_raw = g_raw * head_amp.view(1, H, 1)

# dt_bias: per (head, channel). Ramps across d AND h so neither axis can be collapsed.
dt_bias = rng.tensor(H, D) * 0.3
dt_bias = dt_bias + torch.tensor(
    [[(d - D / 2) * 0.02 + h * 0.01 for d in range(D)] for h in range(H)], dtype=torch.float32
)

# A_log: distinct per head, wide spread. exp(A_log) in roughly [0.37, 2.7].
A_log = torch.tensor([-1.0 + 2.0 * h / (H - 1) for h in range(H)], dtype=torch.float32)

cfg = SimpleNamespace(
    hidden_size=HIDDEN,
    linear_head_dim=D,
    linear_num_heads=H,
    linear_lower_bound=LOWER_BOUND,
)

# `Glm5NextTextForgetGate.forward` computes g from hidden_states via f_a/f_b. Those two
# projections are already bound elementwise by the Slice 2 fixture, so here f_b is set to
# identity-by-rows and f_a is bypassed: hidden_states is fed the already-projected g_raw.
# Simpler and exact: call the gate arithmetic through the module by installing weights that
# make the projection a no-op is fragile, so instead reproduce ONLY the post-projection part
# using the module's own parameters and forward on a shim.
fg = Glm5NextTextForgetGate(cfg)
with torch.no_grad():
    fg.dt_bias.copy_(dt_bias.reshape(-1))
    fg.A_log.copy_(A_log)


class _IdentityFb(torch.nn.Module):
    """Makes `f_b_proj(f_a_proj(x))` return x unchanged, so `forward` receives g_raw."""

    def forward(self, x):
        return x


with torch.no_grad():
    fg.f_a_proj = _IdentityFb()
    fg.f_b_proj = _IdentityFb()
    gate = fg(g_raw.reshape(T, 1, H * D)).reshape(T, H, D)

# Independent restatement of HF's documented law, as a cross-check on the shim above.
check = LOWER_BOUND * torch.sigmoid(
    torch.exp(A_log).view(1, H, 1) * (g_raw.float() + dt_bias.view(1, H, D))
)
shim_err = float((gate - check).abs().max())
assert shim_err == 0.0, f"identity-projection shim changed the result: {shim_err}"


def arr(t):
    vals = t.flatten().tolist()
    return "[" + ",".join(f"{v:.9g}" for v in vals) + "]"


def entry(t):
    return f'{{"shape":{list(t.shape)},"data":{arr(t)}}}'


sat_lo = float((gate <= LOWER_BOUND * 0.999999).sum())
sat_hi = float((gate >= -1e-30).sum())

body = (
    "{\n"
    f' "fixture":{{"heads":{H},"head_dim":{D},"tokens":{T},"lower_bound":{LOWER_BOUND}}},\n'
    f' "coverage":{{"saturated_at_lower_bound":{int(sat_lo)},"saturated_at_zero":{int(sat_hi)},'
    f'"total":{T * H * D}}},\n'
    ' "inputs":{\n'
    f'  "g_raw":{entry(g_raw)},\n'
    f'  "dt_bias":{entry(dt_bias)},\n'
    f'  "A_log":{entry(A_log)}\n'
    " },\n"
    ' "outputs":{\n'
    f'  "gate":{entry(gate)}\n'
    " }\n"
    "}\n"
)

with open("/w/kda_gate_prod_golden.json", "w") as fh:
    fh.write(body)

print("transformers", __import__("transformers").__version__)
print("torch", torch.__version__)
print("bytes", len(body))
print("sha256", hashlib.sha256(body.encode()).hexdigest())
print(f"saturated at lower_bound: {int(sat_lo)} / {T * H * D}")
print(f"saturated at zero:        {int(sat_hi)} / {T * H * D}")
print("gate min/max", float(gate.min()), float(gate.max()))
