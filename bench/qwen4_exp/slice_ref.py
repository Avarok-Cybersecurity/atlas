"""Layer-by-layer bisect against the reference — PLAN.md phase E.

Every PIECE of this port is pinned to the reference: the n-gram ids are
bit-exact, the mHC kernels and the PLE gate/conv match to cosine 0.99999, the
NVMe gather is bit-exact. The model still does not produce coherent text. That
combination says the fault is in the COMPOSITION, which per-kernel probes
cannot see.

So reproduce the same taps Atlas writes (`ATLAS_QWEN4EXP_DUMP`) and diff.

WHAT MAKES THIS AFFORDABLE. The obvious blocker is the 512-expert MoE on every
layer. Two things get around it:

  * Atlas taps the highway at the SUB-LAYER boundary — after a block's
    `hc_post`, before the next `hc_pre`. Reproducing `L00_post_gdn` therefore
    needs layer 0's GDN projections and NOTHING ELSE. No experts at all.
  * Where experts are unavoidable, top-10 routing over a short prompt touches
    a few dozen of the 512, and only those need loading.

So the ladder runs cheapest-first and stops at the first divergence:

    embed -> hc_expand -> L00_in -> L00_post_gdn -> L00_post_moe -> L01 ...

Usage:
    python3 -u bench/qwen4_exp/slice_ref.py --dump-dir /tmp/qwen4dump
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

DEFAULT_SNAP = (
    '/tank/hf/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/snapshots/'
    '129972269565f7f4f664fdf8dd42268d3bbda9fd'
)


def load(snap: str, index: dict, name: str) -> torch.Tensor:
    with safe_open(os.path.join(snap, index[name]), framework='pt') as fh:
        return fh.get_tensor(name)


def compare(label: str, got: np.ndarray, want: np.ndarray) -> bool:
    """Report and return whether this tap matches."""
    got = got.reshape(-1).astype(np.float64)
    want = want.reshape(-1).astype(np.float64)
    n = min(len(got), len(want))
    if len(got) != len(want):
        print(f'  {label:<18} LENGTH MISMATCH got={len(got)} want={len(want)}')
        return False
    got, want = got[:n], want[:n]
    diff = np.abs(got - want)
    denom = max(np.sqrt((want ** 2).mean()), 1e-12)
    cos = float(got @ want / max(np.linalg.norm(got) * np.linalg.norm(want), 1e-30))
    ok = cos > 0.999
    print(
        f'  {label:<18} cos={cos:.9f}  max|diff|={diff.max():.4e}  '
        f'ref_rms={denom:.4e}  {"OK" if ok else "<<< DIVERGES"}'
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', default=os.environ.get('QWEN4EXP_PATH', DEFAULT_SNAP))
    ap.add_argument('--dump-dir', required=True,
                    help='directory ATLAS_QWEN4EXP_DUMP wrote')
    ap.add_argument('--tokens', default='',
                    help='comma-separated prompt token ids (must match the serve request)')
    args = ap.parse_args()

    snap = args.snapshot
    raw = json.load(open(os.path.join(snap, 'config.json')))
    tc = raw['text_config']
    index = json.load(
        open(os.path.join(snap, 'model.safetensors.index.json')))['weight_map']

    h = tc['hidden_size']
    hc = tc['hc_count']
    hc_dim = hc * h
    pfx = 'model.language_model'

    tokens = [int(t) for t in args.tokens.split(',') if t.strip()]
    if not tokens:
        print('--tokens is required: the ids the server actually prefilled.')
        return 2
    t = len(tokens)
    print(f'tokens={tokens}  T={t}  hidden={h}  hc={hc}')

    def tap(name: str) -> np.ndarray | None:
        p = os.path.join(args.dump_dir, name)
        if not os.path.exists(p):
            print(f'  (missing tap {name})')
            return None
        dt = np.float32 if name.endswith('.bin') and '.bf16.' not in name else None
        buf = np.fromfile(p, dtype=np.float32 if dt else np.uint16)
        if dt is None:  # bf16 -> f32
            buf = (buf.astype(np.uint32) << 16).view(np.float32)
        return buf

    # ── 1. embedding ──
    embed_w = load(snap, index, f'{pfx}.embed_tokens.weight').float()
    ids = torch.tensor(tokens, dtype=torch.long)
    embed = embed_w[ids]                                   # [T, H]
    print(f'embed |x|={embed.norm():.4f}')

    # ── 2. hc_expand: broadcast to hc identical streams ──
    highway = embed.unsqueeze(1).expand(t, hc, h).reshape(t, hc_dim).contiguous()
    got = tap('L00_in.bin')
    if got is not None:
        ok = compare('L00_in (expand)', got, highway.numpy())
        if not ok:
            print('\nDIVERGES AT THE EMBEDDING / hc_expand — nothing after this '
                  'is worth reading. Check embed_tokens and the broadcast.')
            return 1

    # ── 3. layer 0 GDN sublayer: hc_pre -> GDN -> hc_post ──
    # Needs ONLY layer 0's mHC attn site and its GDN projections.
    lp = f'{pfx}.layers.0'
    eps = tc['rms_norm_eps']

    def grouped_rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        g = x.unflatten(-1, (hc, h))
        r = torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + eps)
        return (g * r).flatten(-2) * (1.0 + w)

    hc_norm = load(snap, index, f'{lp}.attn_hyper_connection.hc_norm.weight').float()
    down = load(snap, index, f'{lp}.attn_hyper_connection.input_mix_weight_down.weight').float()
    up = load(snap, index, f'{lp}.attn_hyper_connection.input_mix_weight_up.weight').float()
    inject = load(snap, index, f'{lp}.attn_hyper_connection.block_inject_weight.weight').float()

    normed = grouped_rms(highway, hc_norm)
    w = torch.nn.functional.silu(normed @ down.T / hc)
    w = torch.sigmoid(w @ up.T).unflatten(-1, (hc, h))
    mixed = (w * normed.unflatten(-1, (hc, h))).mean(dim=-2)     # [T, H]
    inj = 2 * torch.sigmoid(normed @ inject.T / hc)              # [T, hc]
    print(f'L00 hc_pre mixed |x|={mixed.norm():.4f}  inj=[{inj.min():.4f},{inj.max():.4f}]')

    # Split the sublayer three ways: hc_pre's two outputs, then the block.
    # `L00_post_gdn` diverges at cosine 0.80 with 2.6x the reference
    # magnitude, and that combination — right direction, wrong scale —
    # points at the injection vector rather than the block math.
    got = tap('L00_hc_pre_mixed.bf16.bin')
    if got is not None:
        compare('L00 hc_pre mixed', got, mixed.numpy())
    got = tap('L00_hc_pre_inj.bin')
    if got is not None:
        print(f'    atlas inj = {np.round(got[:hc], 6).tolist()}')
        print(f'    ref   inj = {np.round(inj[0].numpy(), 6).tolist()}  (token 0)')
        compare('L00 hc_pre inj', got, inj.numpy())

    # GDN block on `mixed` — run the REFERENCE MODULE, not a transcription.
    # Same principle as the PLE golden: the thing under test is our port, so
    # the other side of the comparison has to be `modeling_qwen4_exp.py`
    # itself. Layer 0's GDN needs 9 tensors and no experts.
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpTextConfig,
    )
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextGatedDeltaNet,
    )

    config = Qwen4ExpTextConfig(**{
        k: v for k, v in tc.items()
        if k not in ('architectures', 'model_type', 'dtype', 'torch_dtype')
    })
    gdn = Qwen4ExpTextGatedDeltaNet(config, layer_idx=0).to(torch.float32).eval()
    with torch.no_grad():
        for attr, name in (
            ('in_proj_qkv', 'in_proj_qkv.weight'),
            ('in_proj_z', 'in_proj_z.weight'),
            ('in_proj_b', 'in_proj_b.weight'),
            ('in_proj_a', 'in_proj_a.weight'),
            ('out_proj', 'out_proj.weight'),
        ):
            getattr(gdn, attr).weight.copy_(
                load(snap, index, f'{lp}.linear_attn.{name}').float())
        gdn.conv1d.weight.copy_(
            load(snap, index, f'{lp}.linear_attn.conv1d.weight').float())
        gdn.A_log.copy_(load(snap, index, f'{lp}.linear_attn.A_log').float())
        gdn.dt_bias.copy_(load(snap, index, f'{lp}.linear_attn.dt_bias').float())
        gdn.norm.weight.copy_(
            load(snap, index, f'{lp}.linear_attn.norm.weight').float())
        # Record the reference's pre-out_proj value (the gated RMSNorm's
        # output) by wrapping the module rather than reimplementing the
        # recurrence. That splits the block one more time:
        #   projection [OK] -> conv/gates/recurrence -> gated norm -> out_proj
        captured = {}
        _real_norm_fwd = gdn.norm.forward

        def _rec(hidden_states, gate=None):
            out = _real_norm_fwd(hidden_states, gate)
            captured['pre_out_proj'] = out.detach().clone()
            return out

        gdn.norm.forward = _rec
        block_out = gdn(mixed.unsqueeze(0))
        gdn.norm.forward = _real_norm_fwd
    if isinstance(block_out, tuple):
        block_out = block_out[0]
    block_out = block_out[0]
    # Split the block. The projection is a pure GEMM; reproducing it needs
    # only in_proj_qkv and in_proj_z, so if it matches the fault is downstream
    # in conv / recurrence / gated-norm / out_proj.
    got = tap('L00_qkvz_preconv.bf16.bin')
    if got is not None:
        qkv_w = load(snap, index, f'{lp}.linear_attn.in_proj_qkv.weight').float()
        z_w = load(snap, index, f'{lp}.linear_attn.in_proj_z.weight').float()
        # Atlas stores the concat as sequential [Q|K|V|Z].
        want_qkvz = torch.cat([mixed @ qkv_w.T, mixed @ z_w.T], dim=-1)
        compare('L00 qkvz preconv', got, want_qkvz.detach().numpy())

    # Conv alone: pad, depthwise k=4 dilation=1, SiLU (the reference applies
    # the activation inside `causal_conv1d_fn`). Covers the first conv_dim
    # channels only — `z` is NOT convolved.
    got = tap('L00_post_conv.bf16.bin')
    if got is not None:
        qkv_w = load(snap, index, f'{lp}.linear_attn.in_proj_qkv.weight').float()
        conv_w = load(snap, index, f'{lp}.linear_attn.conv1d.weight').float()
        mq = (mixed @ qkv_w.T)
        k_size = conv_w.shape[-1]
        xx = torch.nn.functional.pad(mq.T.unsqueeze(0), (k_size - 1, 0))
        cc = torch.nn.functional.conv1d(
            xx, conv_w.squeeze(1).unsqueeze(1), groups=mq.shape[-1])
        want_conv = torch.nn.functional.silu(cc).squeeze(0).T[:t]
        ok = compare('L00 post_conv', got, want_conv.detach().numpy())
        print('    -> conv1d is the fault' if not ok
              else '    -> conv is RIGHT; fault is gates / recurrence / gated-norm')

    # Gates: [g(nv), beta(nv)] FP32 per token, as the recurrence reads them.
    #   beta = sigmoid(b)          g = exp(-exp(A_log) * softplus(a + dt_bias))
    got = tap('L00_gates.bin')
    if got is not None:
        a_w = load(snap, index, f'{lp}.linear_attn.in_proj_a.weight').float()
        b_w = load(snap, index, f'{lp}.linear_attn.in_proj_b.weight').float()
        a_log = load(snap, index, f'{lp}.linear_attn.A_log').float()
        dt_b = load(snap, index, f'{lp}.linear_attn.dt_bias').float()
        a_raw = mixed @ a_w.T                       # [T, nv]
        b_raw = mixed @ b_w.T
        beta_ref = torch.sigmoid(b_raw)
        g_ref = torch.exp(-a_log.exp() * torch.nn.functional.softplus(a_raw + dt_b))
        want_gates = torch.cat([g_ref, beta_ref], dim=-1)   # [T, 2*nv]
        ok = compare('L00 gates', got, want_gates.detach().numpy())
        print('    -> BA GEMM / gate transforms are the fault' if not ok
              else '    -> gates RIGHT; the delta-rule recurrence kernel is the fault')

    got = tap('L00_pre_out_proj.bf16.bin')
    if got is not None and 'pre_out_proj' in captured:
        ref_pre = captured['pre_out_proj'].reshape(-1)
        print(f'    ref pre_out_proj shape={tuple(captured["pre_out_proj"].shape)} '
              f'|x|={ref_pre.norm():.4f}')
        ok = compare('L00 pre_out_proj', got, ref_pre.numpy())
        if not ok:
            print('    -> conv1d / gates / delta-rule recurrence / gated norm')
        else:
            print('    -> everything up to the gated norm is RIGHT; out_proj is wrong')

    print(f'L00 GDN block_out |x|={block_out.norm():.4f}')
    got = tap('L00_block_out.bf16.bin')
    if got is not None:
        compare('L00 block_out', got, block_out.detach().numpy())

    # hc_post: residual[t, s*H+d] += block_out[t,d] * inj[t,s]
    post = (highway.unflatten(-1, (hc, h))
            + block_out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
    got = tap('L00_post_gdn.bin')
    if got is not None:
        ok = compare('L00_post_gdn', got, post.detach().numpy())
        if not ok:
            print("\nDIVERGES IN LAYER 0's GDN SUBLAYER. `L00_in` matched, so "
                  "the embedding and hc_expand are fine; the fault is in "
                  "hc_pre, the GDN block, or hc_post. hc_pre and hc_post "
                  "already have kernel probes that PASS, which points at the "
                  "block.")
            return 1

    for name in ('L00_post_gdn.bin', 'L00_post_moe.bin', 'L01_in.bin'):
        g = tap(name)
        if g is not None:
            print(f'  (have {name}: |x|={np.linalg.norm(g):.4f}, '
                  f'{np.isfinite(g).all() and "finite" or "NON-FINITE"})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
