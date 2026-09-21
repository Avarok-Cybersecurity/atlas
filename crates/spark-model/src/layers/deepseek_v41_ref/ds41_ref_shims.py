# SPDX-License-Identifier: AGPL-3.0-only
# provenance-id: 526f6e616c6420522e205374657369616b
"""Pure-torch stand-ins for DeepSeek-V4.1's `inference/kernel.py` (tilelang).

tilelang is not installed on the DGX Spark, and the reference model calls five of its kernels
unconditionally even in `dtype="bf16"` mode. Each function here is written from the tilelang
source in `kernel.py` (deepseek-ai/DeepSeek-V4.1-Flash @ dba1be0a), line-referenced, so that the
golden this produces is the reference's arithmetic in fp32, not a re-derivation.

Install before importing the reference:  `sys.modules["kernel"] = ds41_ref_shims`
"""
import torch
import torch.nn.functional as F

_FP8_MAX = 448.0
_FP4_MAX = 6.0
# float4_e2m1fn values with their 3-bit codes; ties round to the EVEN code (RNE).
_E2M1_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _pow2_ceil_log2(v: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(v)) for positive normal fp32, via the IEEE fields exactly as
    `fast_log2_ceil` / `fast_pow2` (kernel.py:22-35): exponent-127, +1 iff mantissa != 0."""
    v = v.float().contiguous()
    bits = v.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127
    man = bits & ((1 << 23) - 1)
    n = e + (man != 0).to(torch.int32)
    return ((n + 127) << 23).view(torch.float32)


def _to_e2m1_rne(y: torch.Tensor) -> torch.Tensor:
    """Round |y| onto the e2m1 grid with ties-to-even on the code, keep the sign."""
    a = y.abs()
    grid = _E2M1_VALS.to(y.device)
    # index of the largest grid value <= a
    lo = torch.searchsorted(grid, a.reshape(-1), right=True).reshape(a.shape) - 1
    lo = lo.clamp(0, grid.numel() - 1)
    hi = (lo + 1).clamp(max=grid.numel() - 1)
    vlo, vhi = grid[lo], grid[hi]
    dlo, dhi = a - vlo, vhi - a
    pick_hi = dhi < dlo
    tie = dhi == dlo
    # on a tie choose the even code
    pick_hi = pick_hi | (tie & (hi % 2 == 0) & (lo % 2 == 1))
    out = torch.where(pick_hi, vhi, vlo)
    return out.copysign(y)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """kernel.py:40-126. Block-wise fp8 e4m3 over the last dim. scale = amax/448, or the
    power-of-two ceiling of that when scale_fmt is set. inplace: quant+dequant back into x."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.float().contiguous().view(-1, N // block_size, block_size)
    amax = z.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil_log2(amax / _FP8_MAX) if scale_fmt is not None else amax / _FP8_MAX
    q = (z / s).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        y = (q.float() * s).view(x.shape).to(x.dtype)
        x.copy_(y)
        return x
    return q.view(x.shape), s.squeeze(-1).view(*x.shape[:-1], N // block_size).to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """kernel.py:127-206. Block-wise e2m1. e8m0 scales: power-of-two ceiling of amax/6 with a
    2^-126 floor; e4m3 scales (compressed KV): amax/6 rounded through e4m3, amax floored at 6*2^-9."""
    N = x.size(-1)
    assert N % block_size == 0
    z = x.float().contiguous().view(-1, N // block_size, block_size)
    amax = z.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0**-9)
        s = (amax / _FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0**-126)
        s = _pow2_ceil_log2(amax / _FP4_MAX)
    q = _to_e2m1_rne((z / s).clamp(-_FP4_MAX, _FP4_MAX))
    if not inplace:
        raise NotImplementedError("ds41_ref_shims: fp4 packed output not needed in bf16 mode")
    x.copy_((q * s).view(x.shape).to(x.dtype))
    return x


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.py:310-405. Per (b, s): gather the kv rows named by topk_idxs (-1 = absent), softmax
    over [scores, attn_sink] with the sink contributing to the denominator only. All-absent rows
    return zero, matching the kernel's finite -1e30 floor."""
    b, s, h, d = q.shape
    qf = q.float()
    kvf = kv.float()
    idx = topk_idxs.long()
    valid = idx >= 0
    safe = idx.clamp_min(0)
    g = kvf[torch.arange(b, device=kv.device)[:, None, None], safe]  # [b, s, topk, d]
    scores = torch.einsum("bshd,bstd->bsht", qf, g) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    full = torch.cat([scores, sink], dim=-1)
    p = torch.softmax(full, dim=-1)[..., :-1]
    p = torch.nan_to_num(p, nan=0.0)
    o = torch.einsum("bsht,bstd->bshd", p, g)
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """kernel.py:406-476. pre = sigmoid(m*s0+b)+eps; post = 2*sigmoid(m*s1+b); comb = softmax rows
    + eps, then column-normalise with +eps, then (iters-1) alternating row/col normalisations."""
    hc = hc_mult
    m = mixes.float()
    b0, b1, b2 = hc_base[:hc].float(), hc_base[hc : 2 * hc].float(), hc_base[2 * hc :].float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0].float() + b0) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1].float() + b1)
    comb = (m[..., 2 * hc :] * hc_scale[2].float() + b2).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def fp8_gemm(*a, **k):
    raise NotImplementedError("ds41_ref_shims: fp8 weights are not used in bf16 mode")


def fp4_gemm(*a, **k):
    raise NotImplementedError("ds41_ref_shims: fp4 weights are not used in bf16 mode")
