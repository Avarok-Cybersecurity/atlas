"""Golden for PLE n-gram injection — Avarok #753 item C, PLAN.md phase D.

PLE is the top correctness risk in this port, for two reasons that both fail
SILENTLY:

  * The ID computation does NOT transfer from LongCat (#746). LongCat uses a
    polynomial rolling hash over token ids; Qwen multiplies by SplitMix64-
    derived odd multipliers and XORs. A wrong hash returns VALID rows from a
    320M-row table — every lookup succeeds, every shape checks out, and the
    embeddings are simply the wrong rows.
  * The fusion is cross-attention into the n-gram embedding, not an additive
    embedding, and its gate carries a SIGNED SQUARE ROOT that nobody would
    guess. Omit it and the gate distribution is wrong but finite.

So this dumps both: the raw ids (for a bit-exactness test that needs no GPU
and no weights) and the full layer output.

The reference's own `forward` computes the ids — this file does not
re-transcribe the hash. `ngram_embedding` is swapped for a recorder, so the
authority stays `modeling_qwen4_exp.py`.

Usage:
    python3 -u bench/qwen4_exp/ple_golden.py [--bin-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import (
    Qwen4ExpTextConfig,
)
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextNGramEmbedding,
    Qwen4ExpTextPLELayer,
)

DEFAULT_SNAP = (
    '/tank/hf/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/snapshots/'
    '129972269565f7f4f664fdf8dd42268d3bbda9fd'
)
# A prompt with structure the hash should react to: a repeat (so two positions
# share an n-gram), and an EOS in the middle (so `_shift_right_ignore_eos`
# actually has a segment boundary to respect — with no EOS that branch is
# never exercised and a wrong implementation of it passes).
TOKENS = [9707, 11, 1879, 9707, 11, 248044, 785, 6722, 315, 9625, 374]


class IdRecorder(torch.nn.Module):
    """Stands in for the 320M-row embedding: records ids, returns zeros.

    Letting the reference's own forward compute the ids is the point — the
    hash is the thing under test, so re-deriving it here would test this file
    against itself.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.ids: torch.Tensor | None = None
        self.rows: torch.Tensor | None = None

    @property
    def weight(self) -> torch.Tensor:  # the reference reads `.weight.device`
        return torch.zeros(1, self.dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self.ids = ids.clone()
        if self.rows is not None:
            return self.rows
        return torch.zeros(*ids.shape, self.dim)


def gather_rows(snap: str, index: dict, ids: np.ndarray, dim: int) -> np.ndarray:
    """Read exactly the rows `ids` names out of the 128 shards.

    Shard `s` holds global rows `[s*R, (s+1)*R)` for a uniform `R` — checked
    here rather than assumed, because an uneven split would silently shift
    every row past the first shard.
    """
    shards = sorted(
        (k for k in index if '.ngram_embedding.shard_' in k),
        key=lambda k: int(re.search(r'shard_(\d+)', k).group(1)),
    )
    with safe_open(os.path.join(snap, index[shards[0]]), framework='pt') as fh:
        rows_per_shard, shard_dim = fh.get_slice(shards[0]).get_shape()
    assert shard_dim == dim, f'shard dim {shard_dim} != {dim}'

    out = np.zeros((len(ids), dim), dtype=np.float32)
    want: dict[int, list[tuple[int, int]]] = {}
    for i, gid in enumerate(ids):
        s, local = divmod(int(gid), rows_per_shard)
        assert s < len(shards), f'id {gid} past the last shard'
        want.setdefault(s, []).append((i, local))

    by_file: dict[str, list[int]] = {}
    for s in want:
        by_file.setdefault(index[shards[s]], []).append(s)
    for fname, ss in by_file.items():
        with safe_open(os.path.join(snap, fname), framework='pt') as fh:
            for s in ss:
                sl = fh.get_slice(shards[s])
                for i, local in want[s]:
                    out[i] = sl[local:local + 1].float().numpy()[0]
    return out


def bf16_bytes(a: np.ndarray) -> bytes:
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    return t.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()


def write_bins(out_dir: str, dump: dict) -> None:
    """Raw sidecar for the Rust probe, in the dtypes the kernel declares.

    `hidden` stays FP32 because it IS the mHC highway; everything else is the
    BF16 the projections and norms actually ship.
    """
    os.makedirs(out_dir, exist_ok=True)
    files: dict[str, str] = {}

    def put(name: str, data: bytes, dtype: str) -> None:
        with open(os.path.join(out_dir, name + '.bin'), 'wb') as fh:
            fh.write(data)
        files[name] = dtype

    put('hidden', dump['hidden'].astype(np.float32).tobytes(), 'f32')
    for n in ('key_proj_out', 'value_proj_out', 'w_norm_key', 'w_norm_query',
              'w_norm_conv', 'w_conv1d'):
        put(n, bf16_bytes(dump[n]), 'bf16')
    for n in ('gated', 'gated_normed', 'output'):
        put(n, dump[n].astype(np.float32).tobytes(), 'f32')
    put('gate_sigmoid', dump['gate_sigmoid'].astype(np.float32).tobytes(), 'f32')
    # Ids + the rows they name, so the Rust side can test the SEGMENTED row
    # cache and the gather in isolation from everything downstream.
    put('ids', dump['ids'].astype(np.uint64).tobytes(), 'u64')
    put('embeddings', dump['embeddings'].astype(np.float32).tobytes(), 'f32')
    # FP32 copy of the value projection, for the probe's bisect arithmetic.
    put('value_proj_out_f32',
        dump['value_proj_out'].astype(np.float32).tobytes(), 'f32')

    meta = {k: (int(dump[k]) if dump[k].dtype.kind in 'iu' else float(dump[k]))
            for k in ('hc_count', 'hidden_size', 'head_dim', 'ngram_heads',
                      'conv_kernel_size', 'conv_dilation', 'rms_norm_eps')}
    meta['num_tokens'] = int(dump['hidden'].shape[0])
    meta['norm_convention'] = dump['norm_convention'].decode()
    meta['files'] = files
    with open(os.path.join(out_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    print(f'wrote {out_dir}/ ({len(files)} bins + meta.json) for the Rust probe')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', default=os.environ.get('QWEN4EXP_PATH',
                                                         DEFAULT_SNAP))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__),
                                                  'ple_golden.npz'))
    ap.add_argument('--bin-dir', default=os.environ.get('ATLAS_PLE_TEST_DATA'))
    args = ap.parse_args()

    snap = args.snapshot
    raw = json.load(open(os.path.join(snap, 'config.json')))
    text_raw = raw.get('text_config', raw)
    config = Qwen4ExpTextConfig(**{
        k: v for k, v in text_raw.items()
        if k not in ('architectures', 'model_type', 'dtype', 'torch_dtype')
    })
    index = json.load(
        open(os.path.join(snap, 'model.safetensors.index.json')))['weight_map']

    # ple_layer_ids is 1-INDEXED: `ple_layer_ids.index(layer_idx + 1)`, so
    # [2] selects layer_idx == 1. The checkpoint agrees — the tensors live at
    # `layers.1.ple.*` with nothing under `layers.2`.
    layer_idx = config.ple_layer_ids[0] - 1
    ple_layer_index = 0
    lp = f'model.language_model.layers.{layer_idx}.ple'
    hc, h = config.hc_count, config.hidden_size
    hc_dim = hc * h
    print(f'PLE at MODEL LAYER {layer_idx} (ple_layer_ids={config.ple_layer_ids},'
          f' 1-indexed); ngram_size={config.ngram_size} '
          f'heads_per_ngram={config.heads_per_ngram} '
          f'ple_embed_dim={config.ple_embed_dim} eos={config.eos_token_id}')

    def get(name: str) -> torch.Tensor:
        full = f'{lp}.{name}'
        with safe_open(os.path.join(snap, index[full]), framework='pt') as fh:
            return fh.get_tensor(full)

    # ── ids: the reference computes them; we only record ──
    # The constructor allocates `nn.Embedding(padded_vocab, 160)` — 320M rows,
    # ~205 GB in fp32. Stub it for the duration: we never use the real module,
    # only the id arithmetic around it, and the rows come from the shards.
    import torch.nn as _nn
    _real_embedding = _nn.Embedding

    class _StubEmbedding(_nn.Module):
        def __init__(self, num, dim, *a, **k):
            super().__init__()
            self.num_embeddings, self.embedding_dim = num, dim
            self.weight = torch.zeros(1, dim)

    _nn.Embedding = _StubEmbedding
    try:
        ng = Qwen4ExpTextNGramEmbedding(
            config, config.ple_embed_dim, layer_idx, ple_layer_index)
    finally:
        _nn.Embedding = _real_embedding
    print(f'  padded table {ng.ngram_embedding.num_embeddings} x '
          f'{ng.ngram_embedding.embedding_dim} (stubbed; rows come from shards)')
    head_dim = config.ple_embed_dim // ng.ngram_heads
    rec = IdRecorder(head_dim)
    ng.ngram_embedding = rec

    # The checkpoint SHIPS the multipliers and the per-head vocab/offsets.
    # Read them and use them; `_build_layer_multipliers` is a cross-check,
    # never the source. Disagreement here would mean the derivation drifted.
    ck_mult = get('ple_embedding.layer_multipliers')
    ck_vocab = get('ple_embedding.ngram_heads_vocab_sizes')
    ck_off = get('ple_embedding.ngram_heads_offsets')
    for name, derived, shipped in (
        ('layer_multipliers', ng.layer_multipliers, ck_mult),
        ('ngram_heads_vocab_sizes', ng.ngram_heads_vocab_sizes, ck_vocab),
        ('ngram_heads_offsets', ng.ngram_heads_offsets, ck_off),
    ):
        same = torch.equal(derived.cpu(), shipped.cpu())
        print(f'  {name:<26} derived == shipped: {same}')
        if not same:
            print(f'    derived  {derived.tolist()[:4]}...')
            print(f'    shipped  {shipped.tolist()[:4]}...')
    ng.layer_multipliers = torch.nn.Buffer(ck_mult.clone())
    ng.ngram_heads_vocab_sizes = torch.nn.Buffer(ck_vocab.clone())
    ng.ngram_heads_offsets = torch.nn.Buffer(ck_off.clone())

    input_ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        ng(input_ids, None)
    ids = rec.ids[0].numpy().astype(np.uint64)   # [T, ngram_heads]
    t, n_heads = ids.shape
    print(f'  ids {ids.shape} range [{ids.min()}, {ids.max()}]  '
          f'heads={n_heads} head_dim={head_dim}')

    # ── real rows for exactly those ids ──
    flat = ids.reshape(-1)
    rows = gather_rows(snap, index, flat, head_dim)
    print(f'  gathered {len(flat)} rows from the 128-shard table')
    # Keep the BATCH dim. Without it `ng.forward` returns [T, 2560] instead
    # of [1, T, 2560], and `value_proj(emb)[0]` then dumps ONE TOKEN while the
    # expected tensors — which broadcast back to [1,T,...] — stay correct. The
    # probe then matches on token 0 and diverges everywhere else, which reads
    # exactly like a time-dependence bug in the kernel. It was the fixture.
    rec.rows = torch.from_numpy(rows.reshape(1, t, n_heads, head_dim))

    # ── the full layer ──
    # Same stub: `Qwen4ExpTextPLELayer` builds its own NGramEmbedding, and so
    # its own 320M-row table.
    _nn.Embedding = _StubEmbedding
    try:
        ple = Qwen4ExpTextPLELayer(config, layer_idx, ple_layer_index)
    finally:
        _nn.Embedding = _real_embedding
    ple = ple.to(torch.float32).eval()
    with torch.no_grad():
        ple.key_proj.weight.copy_(get('key_proj.weight').float())
        ple.value_proj.weight.copy_(get('value_proj.weight').float())
        ple.norm_key.weight.copy_(get('norm_key.weight').float())
        ple.norm_query.weight.copy_(get('norm_query.weight').float())
        ple.norm_conv.weight.copy_(get('norm_conv.weight').float())
        ple.conv1d.weight.copy_(get('conv1d.weight').float())
    # `ple.ple_embedding` is the whole NGramEmbedding (it takes
    # `(input_ids, past_key_values)`), not the inner table. Swap in the one
    # built above, whose recorder now returns the REAL rows for these ids —
    # so the reference's full chain runs, table lookup included.
    ple.ple_embedding = ng

    g = torch.Generator().manual_seed(0x9E3779B9)
    hidden = torch.randn(1, t, hc_dim, generator=g, dtype=torch.float32)
    with torch.no_grad():
        embeddings = ng(input_ids, None)
        out = ple(hidden, input_ids, None)
    print(f'  embeddings |x|={embeddings.norm():.4f}  '
          f'output {tuple(out.shape)} |x|={out.norm():.4f} '
          f'mean={out.mean():+.5f}')

    dump = {
        'tokens': np.array(TOKENS, dtype=np.uint32),
        'ids': ids,
        'embeddings': embeddings.reshape(t, -1).numpy(),
        'hidden': hidden[0].numpy(),
        'output': out[0].detach().numpy(),
        'layer_multipliers': ck_mult.numpy(),
        'ngram_heads_vocab_sizes': ck_vocab.numpy(),
        'ngram_heads_offsets': ck_off.numpy(),
        'eos_token_id': np.int64(config.eos_token_id if not isinstance(
            config.eos_token_id, list) else config.eos_token_id[0]),
        'ngram_size': np.int32(config.ngram_size),
        'heads_per_ngram': np.int32(config.heads_per_ngram),
        'ngram_heads': np.int32(n_heads),
        'head_dim': np.int32(head_dim),
        'hc_count': np.int32(hc),
        'hidden_size': np.int32(h),
        'layer_idx': np.int32(layer_idx),
        'conv_kernel_size': np.int32(config.ple_conv_kernel_size),
        'conv_dilation': np.int32(config.ngram_size),
        'rms_norm_eps': np.float32(config.rms_norm_eps),
        'norm_convention': np.bytes_(b'normed * (1.0 + weight)'),
    }
    # Intermediates for the Rust kernel probe. Produced by the reference's OWN
    # submodules on the reference's own weights — this is not a second
    # implementation, it is the same one, observed one stage earlier.
    with torch.no_grad():
        key_out = ple.key_proj(embeddings)                       # [1,T,hc*H]
        value_out = ple.value_proj(embeddings)                   # [1,T,H]
        key_n = ple.norm_key(key_out).unflatten(-1, (hc, h))
        query_n = ple.norm_query(hidden).unflatten(-1, (hc, h))
        gate = (key_n * query_n).sum(-1, keepdim=True) / (h ** 0.5)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value_out.unsqueeze(-2)
        gated_flat = gated.flatten(-2)
        gated_normed = ple.norm_conv(gated_flat)
    dump['key_proj_out'] = key_out[0].numpy()
    dump['value_proj_out'] = value_out[0].numpy()
    dump['gate_sigmoid'] = torch.sigmoid(gate)[0, :, :, 0].numpy()
    dump['gated'] = gated_flat[0].numpy()
    dump['gated_normed'] = gated_normed[0].numpy()
    dump['w_norm_key'] = get('norm_key.weight').float().numpy()
    dump['w_norm_query'] = get('norm_query.weight').float().numpy()
    dump['w_norm_conv'] = get('norm_conv.weight').float().numpy()
    dump['w_conv1d'] = get('conv1d.weight').float().numpy()
    print(f'  key_proj |x|={key_out.norm():.4f}  value_proj |x|={value_out.norm():.4f}')
    print(f'  gate sigmoid range [{torch.sigmoid(gate).min():.4f}, '
          f'{torch.sigmoid(gate).max():.4f}]  gated |x|={gated_flat.norm():.4f}')

    np.savez(args.out, **dump)
    if args.bin_dir:
        write_bins(args.bin_dir, dump)
    print(f'\nwrote {args.out} '
          f'({os.path.getsize(args.out) / 1e6:.2f} MB, {len(dump)} arrays)')

    # A JSON fixture for the Rust id test: pure token arithmetic, so it needs
    # neither GPU nor checkpoint and can run in CI.
    fx = os.path.join(os.path.dirname(args.out), 'ple_id_fixtures.json')
    json.dump({
        'tokens': TOKENS,
        'eos_token_id': int(dump['eos_token_id']),
        'ngram_size': int(config.ngram_size),
        'heads_per_ngram': int(config.heads_per_ngram),
        'layer_multipliers': [int(v) for v in ck_mult.tolist()],
        'ngram_heads_vocab_sizes': [int(v) for v in ck_vocab.tolist()],
        'ngram_heads_offsets': [int(v) for v in ck_off.tolist()],
        'expected_ids': [[int(v) for v in row] for row in ids.tolist()],
    }, open(fx, 'w'), indent=1)
    print(f'wrote {fx} (id bit-exactness fixture, no GPU/checkpoint needed)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
