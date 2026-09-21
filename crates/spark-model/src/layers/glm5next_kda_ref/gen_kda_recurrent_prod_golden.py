#!/usr/bin/env python3
"""GLM-5.3-Flash KDA recurrent DECODE golden at PRODUCTION geometry (H=64, D=128).

Slice 2's `kda_golden.json` binds the recurrence to HuggingFace elementwise, but at H=2,
D=4. At production geometry the full recurrent state is 64*128*128 = 1,048,576 floats per
step — far too large to commit. This file therefore commits:

  * the per-step OUTPUT `o[H,D]` in full (8192 floats/step), and
  * a strided SAMPLE of the recurrent state (prime stride 251, ~4178 floats/step), and
  * an fp64 index-weighted checksum over the FULL state, which no indexing or ordering
    error can survive.

Inputs are NOT committed. They are produced by an integer LCG that is reproduced exactly
in Rust (`Lcg` in `kda_recurrent_microtest.rs`): pure 64-bit integer ops plus an exact
division by 2^24, so there is no transcendental and no platform libm in the generator.
`lcg_probe` pins the first 8 draws so a divergence in that reproduction fails loudly
rather than silently changing the fixture.

Source of truth: HF `transformers` 5.16.1 `recurrent_kimi_delta_attention`, called
directly. No equation is re-derived here.

The initial state is NON-ZERO, so step 1 already exercises carried state; the zero-state
path is covered by the H=2/D=4 fixture in `kda_golden.json`.
"""

import hashlib
import sys

import torch

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    recurrent_kimi_delta_attention,
)

H = 64
D = 128
STEPS = 3
LOWER_BOUND = -5.0
SAMPLE_STRIDE = 251  # prime, so the sample cannot alias the H/D/step periods


class Lcg:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next_unit(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((self.s >> 40) / (1 << 24)) * 2.0 - 1.0

    def tensor(self, *shape):
        n = 1
        for s in shape:
            n *= s
        return torch.tensor([self.next_unit() for _ in range(n)], dtype=torch.float32).reshape(shape)


probe_rng = Lcg(0x5EED_2EC0)
lcg_probe = [probe_rng.next_unit() for _ in range(8)]

rng = Lcg(0x5EED_2EC0)
# Drawn in exactly this order; the Rust side must draw identically.
state0 = rng.tensor(H, D, D) * 0.05  # non-zero carried state
q_steps = [rng.tensor(1, 1, H, D) for _ in range(STEPS)]
k_steps = [rng.tensor(1, 1, H, D) for _ in range(STEPS)]
v_steps = [rng.tensor(1, 1, H, D) for _ in range(STEPS)]
gate_steps = [LOWER_BOUND * torch.sigmoid(rng.tensor(1, 1, H, D) * 3.0) for _ in range(STEPS)]
beta_steps = [torch.sigmoid(rng.tensor(1, 1, H)) for _ in range(STEPS)]

# HF's kernel L2-normalises q/k internally. The Atlas kernel consumes ALREADY-normalised
# q/k (its conv fuses the L2), so the golden records the normalised tensors that Atlas
# will be fed, and HF is driven with use_qk_l2norm_in_kernel=False on those same values.
def l2(x):
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


outs, samples, checksums = [], [], []
carry = state0.unsqueeze(0).clone()  # HF wants [B, H, K, V]
qn_steps, kn_steps = [], []
for i in range(STEPS):
    qn, kn = l2(q_steps[i]), l2(k_steps[i])
    qn_steps.append(qn)
    kn_steps.append(kn)
    o, carry = recurrent_kimi_delta_attention(
        qn.clone(),
        kn.clone(),
        v_steps[i].clone(),
        g=gate_steps[i],
        beta=beta_steps[i],
        initial_state=carry,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,  # already normalised above
    )
    outs.append(o.reshape(H, D))
    flat = carry.reshape(-1)
    samples.append(flat[::SAMPLE_STRIDE].contiguous())
    idx = torch.arange(flat.numel(), dtype=torch.float64)
    checksums.append(float((flat.to(torch.float64) * (idx + 1.0)).sum()))


def arr(t):
    return "[" + ",".join(f"{v:.9g}" for v in t.flatten().tolist()) + "]"


def entry(t):
    return f'{{"shape":{list(t.shape)},"data":{arr(t)}}}'


body = (
    "{\n"
    f' "fixture":{{"heads":{H},"head_dim":{D},"steps":{STEPS},'
    f'"lower_bound":{LOWER_BOUND},"sample_stride":{SAMPLE_STRIDE},"seed":"0x5EED2EC0"}},\n'
    f' "lcg_probe":[{",".join(f"{v:.9g}" for v in lcg_probe)}],\n'
    ' "outputs":{\n'
    + ",\n".join(f'  "o_step{i}":{entry(outs[i])}' for i in range(STEPS))
    + ",\n"
    + ",\n".join(f'  "state_sample{i}":{entry(samples[i])}' for i in range(STEPS))
    + "\n },\n"
    f' "state_checksums":[{",".join(repr(c) for c in checksums)}]\n'
    "}\n"
)

with open("/w/kda_recurrent_prod_golden.json", "w") as fh:
    fh.write(body)

print("transformers", __import__("transformers").__version__)
print("torch", torch.__version__)
print("bytes", len(body))
print("sha256", hashlib.sha256(body.encode()).hexdigest())
for i in range(STEPS):
    print(
        f"step{i}: |o|max={float(outs[i].abs().max()):.6g} "
        f"state|max={float(carry.abs().max()):.6g} checksum={checksums[i]:.10g}"
    )
