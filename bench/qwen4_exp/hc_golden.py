"""Golden for the mHC low-rank residual — Avarok #753 item B, PLAN.md phase A.

`kernels/gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu` and
`ops/hyper_connection_lowrank.rs` were written from the reference and compared
to NOTHING. This runs the real `Qwen4ExpTextGatedResidual` on real checkpoint
weights and dumps its outputs, so the kernel can be held against a number
before it is wired into 48 layers.

Two things this is built to catch, both of which produce plausible
activations when wrong:

  * `hc_norm` is a GROUPED RMSNorm (`group_size = hidden_size`) — the four
    2560-wide streams normalize INDEPENDENTLY inside the 10240 vector. A
    single RMS over all 10240 is a different function of the same shape.
  * `Qwen4ExpTextRMSNorm` applies `normed * (1.0 + weight)`, not
    `normed * weight`. See `norm_convention` in the emitted npz.

The module runs on CPU in float32; only ~54 MB of weights are touched, so no
GPU and no full-model load.

Usage:
    python3 -u bench/qwen4_exp/hc_golden.py [--out hc_golden.npz]

Requires `transformers >= 5.16` (which ships `qwen4_exp` natively — verified
byte-identical to `ref/modeling_qwen4_exp.py`), torch, safetensors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import (
    Qwen4ExpTextConfig,
)
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedResidual,
)

DEFAULT_SNAP = (
    '/tank/hf/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/snapshots/'
    '129972269565f7f4f664fdf8dd42268d3bbda9fd'
)
NUM_TOKENS = 8

# Every site the low-rank collapse appears at, and what it exercises.
SITES = {
    # per-layer, use_combine=True -> mixed_input + injection_weights
    'attn': 'model.language_model.layers.0.attn_hyper_connection',
    'mlp': 'model.language_model.layers.0.mlp_hyper_connection',
    # model-level, use_combine=False -> mixed_input only. This one IS the
    # model's final norm: the checkpoint has no `model.norm.weight`.
    'head': 'model.language_model.hyper_connection_mixer',
}


def load_site(snap: str, index: dict, prefix: str) -> dict[str, torch.Tensor]:
    """Pull one hyper-connection block's tensors out of the shards."""
    names = {
        'hc_norm.weight': 'hc_norm',
        'input_mix_weight_down.weight': 'down',
        'input_mix_weight_up.weight': 'up',
        'block_inject_weight.weight': 'inject',
    }
    out: dict[str, torch.Tensor] = {}
    for suffix, short in names.items():
        full = f'{prefix}.{suffix}'
        shard = index.get(full)
        if shard is None:
            continue  # `hyper_connection_mixer` has no block_inject_weight
        with safe_open(os.path.join(snap, shard), framework='pt') as fh:
            out[short] = fh.get_tensor(full)
    return out


def build(config: Qwen4ExpTextConfig, w: dict[str, torch.Tensor]):
    use_combine = 'inject' in w
    mod = Qwen4ExpTextGatedResidual(config, use_combine=use_combine)
    mod = mod.to(torch.float32).eval()
    with torch.no_grad():
        mod.hc_norm.weight.copy_(w['hc_norm'].float())
        mod.input_mix_weight_down.weight.copy_(w['down'].float())
        mod.input_mix_weight_up.weight.copy_(w['up'].float())
        if use_combine:
            mod.block_inject_weight.weight.copy_(w['inject'].float())
    return mod, use_combine


def bf16_bytes(a: np.ndarray) -> bytes:
    """BF16 the way the kernel reads it: raw u16, round-to-nearest-even."""
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return t.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()


def write_bins(out_dir: str, dump: dict, weights: dict) -> None:
    """Raw sidecar for the Rust probe.

    The npz is the human-readable artifact; Rust reads flat little-endian
    buffers instead, in exactly the dtypes the kernel's parameters declare.
    Getting a dtype wrong here would show up as a parity failure blamed on
    the kernel, so each file's dtype is spelled out in `meta.json`.
    """
    os.makedirs(out_dir, exist_ok=True)
    files: dict[str, str] = {}

    def put(name: str, data: bytes, dtype: str) -> None:
        with open(os.path.join(out_dir, name + '.bin'), 'wb') as fh:
            fh.write(data)
        files[name] = dtype

    # FP32 highway, FP32 expectations — `streams` is the model's residual.
    put('streams', dump['hyper_input'].astype(np.float32).tobytes(), 'f32')
    put('post_expected', dump['post_expected'].astype(np.float32).tobytes(),
        'f32')
    # BF16: everything the kernel's signature declares as __nv_bfloat16.
    put('post_block_out', bf16_bytes(dump['post_block_out']), 'bf16')
    for site in SITES:
        if f'{site}_mixed_input' not in dump:
            continue
        put(f'{site}_mixed', dump[f'{site}_mixed_input'].astype(np.float32)
            .tobytes(), 'f32')
        if f'{site}_injection_weights' in dump:
            put(f'{site}_inj', dump[f'{site}_injection_weights']
                .astype(np.float32).tobytes(), 'f32')
        for short in ('hc_norm', 'down', 'up', 'inject'):
            key = f'{site}_w_{short}'
            if key in weights:
                put(f'{site}_w_{short}', bf16_bytes(weights[key]), 'bf16')

    meta = {
        'hc_count': int(dump['hc_count']),
        'hidden_size': int(dump['hidden_size']),
        'hc_lowrank': int(dump['hc_lowrank']),
        'rms_norm_eps': float(dump['rms_norm_eps']),
        'num_tokens': int(dump['num_tokens']),
        'norm_convention': dump['norm_convention'].decode(),
        'files': files,
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print(f'wrote {out_dir}/ ({len(files)} bins + meta.json) for the Rust probe')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin-dir', default=os.environ.get('ATLAS_HC_TEST_DATA'),
                    help='also emit raw .bin + meta.json for the Rust probe')
    ap.add_argument('--snapshot', default=os.environ.get('QWEN4EXP_PATH',
                                                         DEFAULT_SNAP))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__),
                                                  'hc_golden.npz'))
    ap.add_argument('--tokens', type=int, default=NUM_TOKENS)
    args = ap.parse_args()

    snap = args.snapshot
    cfg_raw = json.load(open(os.path.join(snap, 'config.json')))
    text_raw = cfg_raw.get('text_config', cfg_raw)
    config = Qwen4ExpTextConfig(**{
        k: v for k, v in text_raw.items()
        if k not in ('architectures', 'model_type', 'dtype', 'torch_dtype')
    })
    index = json.load(
        open(os.path.join(snap, 'model.safetensors.index.json')))['weight_map']

    hc = config.hc_count
    h = config.hidden_size
    hc_dim = hc * h
    print(f'hc_count={hc} hidden={h} hc_dim={hc_dim} '
          f'lowrank={config.hc_lowrank} eps={config.rms_norm_eps}')

    # A fixed, reproducible highway. Scaled to ~unit RMS per stream so the
    # grouped norm is actually exercised — with equal per-stream scales a
    # single global RMS would agree with the grouped one and the bug this is
    # built to catch would hide. Give each stream a DIFFERENT scale.
    g = torch.Generator().manual_seed(0xC0FFEE)
    hyper = torch.randn(1, args.tokens, hc_dim, generator=g,
                        dtype=torch.float32)
    stream_scales = torch.tensor([0.25, 1.0, 4.0, 16.0])[:hc]
    hyper = (hyper.unflatten(-1, (hc, h)) * stream_scales[:, None]).flatten(-2)
    print('per-stream scales (deliberately unequal, to expose a global RMS): '
          f'{stream_scales.tolist()}')

    dump: dict[str, np.ndarray] = {
        'hyper_input': hyper[0].numpy(),
        'hc_count': np.int32(hc),
        'hidden_size': np.int32(h),
        'hc_lowrank': np.int32(config.hc_lowrank),
        'rms_norm_eps': np.float32(config.rms_norm_eps),
        'num_tokens': np.int32(args.tokens),
        # Recorded so the Rust side cannot silently assume the other form.
        'norm_convention': np.bytes_(b'normed * (1.0 + weight)'),
    }
    # The weights go in a SEPARATE file. They are ~80 MB of float32 pulled
    # verbatim out of the checkpoint, so they are regenerable and gitignored;
    # the contract worth version-controlling is the inputs and the expected
    # outputs, which are under a megabyte.
    weights: dict[str, np.ndarray] = {}

    for site, prefix in SITES.items():
        w = load_site(snap, index, prefix)
        if 'hc_norm' not in w:
            print(f'{site}: MISSING at {prefix} — skipping')
            continue
        mod, use_combine = build(config, w)
        with torch.no_grad():
            out = mod(hyper)
        if use_combine:
            mixed, passthrough, inj = out
            dump[f'{site}_injection_weights'] = inj[0].numpy()
            assert torch.equal(passthrough, hyper), \
                'hyper_input must pass through untouched'
        else:
            mixed = out
        dump[f'{site}_mixed_input'] = mixed[0].numpy()
        for short, t in w.items():
            weights[f'{site}_w_{short}'] = t.float().numpy()

        m = mixed[0]
        print(f'{site:>5}: mixed_input |x|={m.norm():.4f} '
              f'mean={m.mean():+.5f} std={m.std():.5f} '
              f'shape={tuple(m.shape)}'
              + ('' if not use_combine else
                 f'  inj range=[{inj.min():.4f}, {inj.max():.4f}]'))

        # The load-bearing check, run here so the golden itself is verified
        # rather than merely dumped: recompute the grouped norm by hand and
        # confirm the module agrees, then show what the two wrong readings
        # would have produced instead.
        if site == 'attn':
            x = hyper[0].float()
            grouped = x.unflatten(-1, (hc, h))
            rms = torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True)
                              + config.rms_norm_eps)
            normed = (grouped * rms).flatten(-2) * (1.0 + w['hc_norm'].float())
            ref_normed = mod.hc_norm(x)
            print(f'       grouped-norm hand-check: max|diff|='
                  f'{(normed - ref_normed).abs().max():.3e}')
            global_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True)
                                     + config.rms_norm_eps)
            wrong_global = x * global_rms * (1.0 + w['hc_norm'].float())
            wrong_offset = (grouped * rms).flatten(-2) * w['hc_norm'].float()
            print(f'       if a GLOBAL rms were used: max|diff|='
                  f'{(wrong_global - ref_normed).abs().max():.3e}')
            print(f'       if the +1 offset were dropped: max|diff|='
                  f'{(wrong_offset - ref_normed).abs().max():.3e}')
            dump['attn_hc_normed'] = ref_normed.detach().numpy()

    # hc_post: inject a block output back into every stream.
    #   out[t, s*H + d] = residual[t, s*H + d] + block_out[t, d] * inj[t, s]
    # The kernel reads `block_out` as BF16, so round first and compute the
    # expectation from the rounded values — otherwise the tolerance is
    # measuring the fixture's dtype rather than the kernel.
    g2 = torch.Generator().manual_seed(0xBEEF)
    block_out = torch.randn(args.tokens, h, generator=g2, dtype=torch.float32)
    block_out = block_out.to(torch.bfloat16).float()
    inj = torch.from_numpy(dump['attn_injection_weights'])
    post = (hyper[0].unflatten(-1, (hc, h))
            + block_out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
    dump['post_block_out'] = block_out.numpy()
    dump['post_expected'] = post.numpy()
    print(f' post: |x|={post.norm():.4f} (block_out rounded to BF16 first)')

    np.savez(args.out, **dump)
    w_out = args.out.replace('.npz', '_weights.npz')
    np.savez(w_out, **weights)
    if args.bin_dir:
        write_bins(args.bin_dir, dump, weights)
    print(f'\nwrote {args.out} '
          f'({os.path.getsize(args.out) / 1e6:.2f} MB, {len(dump)} arrays)')
    print(f'wrote {w_out} '
          f'({os.path.getsize(w_out) / 1e6:.1f} MB, {len(weights)} arrays) '
          f'— regenerable, gitignored')
    return 0


if __name__ == '__main__':
    sys.exit(main())
