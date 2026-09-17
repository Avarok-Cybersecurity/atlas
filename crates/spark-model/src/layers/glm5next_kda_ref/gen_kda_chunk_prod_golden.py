#!/usr/bin/env python3
"""GLM-5.3-Flash KDA CHUNKED PREFILL golden at PRODUCTION geometry (H=64, D=128).

T is deliberately small (5) and deliberately NOT a multiple of any tested chunk size, so the
committed golden covers the ragged tail at production H/D while staying ~500 KB. Larger
prefills are covered GPU-side by the chunk-vs-recurrent invariant, which needs no golden.

The initial state is NON-ZERO, so inter-chunk state propagation is exercised from chunk 0.

Chunk size here is 2 (three chunks: 2,2,1 -> pad 1). The chunked algorithm is algebraically
equal to the recurrence for ANY chunk size — HF's own self-check has chunk=2, chunk=4-padded
and the recurrence agreeing to <1e-7 — so this single golden is a valid target for the GPU
kernel at every chunk size it supports.

Source of truth: HF `transformers` 5.16.1 `chunk_kimi_delta_attention`, called directly.
Inputs come from the same integer LCG the Rust microtest reproduces bit for bit.
"""

import hashlib
import sys

import torch

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    chunk_kimi_delta_attention,
    recurrent_kimi_delta_attention,
)

H = 64
D = 128
T = 5
CHUNK = 2
LOWER_BOUND = -5.0
SAMPLE_STRIDE = 251


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


probe_rng = Lcg(0x5EED_C400)
lcg_probe = [probe_rng.next_unit() for _ in range(8)]

rng = Lcg(0x5EED_C400)
state0 = rng.tensor(H, D, D) * 0.05
q = rng.tensor(1, T, H, D)
k = rng.tensor(1, T, H, D)
v = rng.tensor(1, T, H, D)
gate = LOWER_BOUND * torch.sigmoid(rng.tensor(1, T, H, D) * 3.0)
beta = torch.sigmoid(rng.tensor(1, T, H))


def l2(x):
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


# The Atlas kernels consume ALREADY-normalised q/k (the conv fuses the L2 upstream), so the
# golden is driven with use_qk_l2norm_in_kernel=False on the normalised tensors.
qn, kn = l2(q), l2(k)

out, state = chunk_kimi_delta_attention(
    qn.clone(), kn.clone(), v.clone(), g=gate, beta=beta, chunk_size=CHUNK,
    initial_state=state0.unsqueeze(0).clone(), output_final_state=True,
    use_qk_l2norm_in_kernel=False,
)

# HF's own chunk-vs-recurrent invariant on this exact fixture, recorded so the Rust side can
# assert the oracle was self-consistent before trusting it.
rec_out, rec_state = recurrent_kimi_delta_attention(
    qn.clone(), kn.clone(), v.clone(), g=gate, beta=beta,
    initial_state=state0.unsqueeze(0).clone(), output_final_state=True,
    use_qk_l2norm_in_kernel=False,
)
hf_chunk_vs_recurrent = float((out - rec_out).abs().max())
hf_chunk_vs_recurrent_state = float((state - rec_state).abs().max())

flat = state.reshape(-1)
idx = torch.arange(flat.numel(), dtype=torch.float64)
state_checksum = float((flat.to(torch.float64) * (idx + 1.0)).sum())
sample = flat[::SAMPLE_STRIDE].contiguous()


def arr(t):
    return "[" + ",".join(f"{x:.9g}" for x in t.flatten().tolist()) + "]"


def entry(t):
    return f'{{"shape":{list(t.shape)},"data":{arr(t)}}}'


body = (
    "{\n"
    f' "fixture":{{"heads":{H},"head_dim":{D},"tokens":{T},"chunk":{CHUNK},'
    f'"lower_bound":{LOWER_BOUND},"sample_stride":{SAMPLE_STRIDE},"seed":"0x5EEDC400"}},\n'
    f' "lcg_probe":[{",".join(f"{x:.9g}" for x in lcg_probe)}],\n'
    f' "hf_self_checks":{{"chunk_vs_recurrent_max_abs":{hf_chunk_vs_recurrent!r},'
    f'"chunk_vs_recurrent_state_max_abs":{hf_chunk_vs_recurrent_state!r}}},\n'
    ' "outputs":{\n'
    f'  "out":{entry(out.reshape(T, H, D))},\n'
    f'  "state_sample":{entry(sample)}\n'
    " },\n"
    f' "state_checksum":{state_checksum!r}\n'
    "}\n"
)

with open("/w/kda_chunk_prod_golden.json", "w") as fh:
    fh.write(body)

print("transformers", __import__("transformers").__version__)
print("torch", torch.__version__)
print("bytes", len(body))
print("sha256", hashlib.sha256(body.encode()).hexdigest())
print("HF chunk vs recurrent:", hf_chunk_vs_recurrent, "state:", hf_chunk_vs_recurrent_state)
print("out |max|", float(out.abs().max()), "state |max|", float(state.abs().max()))
