"""Golden for the QSA indexer — Avarok #753 phase G.

Runs the real `Qwen4ExpTextQSAIndexer` (transformers >= 5.16 ships qwen4_exp
natively, byte-identical to ref/modeling_qwen4_exp.py) on real checkpoint
weights from the FIRST full-attention layer (model layer 3), over a T=2200
random sequence. 2200 tokens = 550 complete 4-token blocks > block_topk=512,
so selection actively PRUNES — the regime the kernel exists for.

What the npz carries:
  * hidden        [T, H] bf16-faithful input (stored as u16)
  * selected_mask [T, T] bool — the module's end-to-end output, squeezed.
    Row i is query i's visible set AFTER selection; the last row is exactly
    the decode-step contract.
  * q_post        [T, n_heads, 128] — q after q_layernorm + partial rope
    (child modules + the reference `apply_rotary_pos_emb`, not transcriptions)
  * raw_keys      [T, 128] — the un-normed per-token indexer keys (what the
    engine caches)
  * block_keys    [n_blocks, 128] — pooled -> k_layernorm -> rope at block
    start, for the FULL-prefix visibility of the LAST query (causal ⇒ every
    earlier query's blocks are a prefix of these)
  * scores_last   [n_blocks] — the last query's block scores
    (relu(q·k).sum(heads)/sqrt(128)), for kernel-level debugging; the SET in
    selected_mask is the contract (float ties at the 512th block are the only
    legitimate divergence and are ~impossible with random input).

Semantics this pins down (all from running reference code, not reading it):
  * rope on the indexer q/k covers dims [0, 64) (head_dim 256 x partial 0.25
    from the MODEL config — not the indexer's 128), nope tail [64, 128)
  * text-only mrope (equal position grids) with interleaved sections
  * pooled keys are meaned RAW (pre-norm), THEN k_layernorm, THEN roped at
    the block's FIRST token position
  * the always-visible tail is the last (visible % 4) tokens

Usage:
    /home/ms/qwen4ref-venv/bin/python -u bench/qwen4_exp/qsa_golden.py \
        [--out bench/qwen4_exp/qsa_golden.npz]
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen4_exp.configuration_qwen4_exp import (
    Qwen4ExpTextConfig,
)
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextQSAIndexer,
    Qwen4ExpTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

DEFAULT_SNAP = (
    '/tank/hf/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/snapshots/'
    '129972269565f7f4f664fdf8dd42268d3bbda9fd'
)
LAYER = 3  # first full-attention layer (layer_types is 3:1 GDN:attn)
T = 2200
SEED = 0


def bf16_roundtrip(a: torch.Tensor) -> torch.Tensor:
    """Round to BF16 and back — the input as the engine will actually see it."""
    return a.to(torch.bfloat16).to(torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', default=os.environ.get('QWEN4EXP_PATH', DEFAULT_SNAP))
    ap.add_argument('--out', default='bench/qwen4_exp/qsa_golden.npz')
    args = ap.parse_args()

    cfg_raw = json.load(open(os.path.join(args.snapshot, 'config.json')))
    tcfg = cfg_raw.get('text_config', cfg_raw)
    config = Qwen4ExpTextConfig(**tcfg)
    idx_json = json.load(
        open(os.path.join(args.snapshot, 'model.safetensors.index.json')))
    index = idx_json['weight_map']

    prefix = f'model.language_model.layers.{LAYER}.self_attn.indexer'
    names = ['index_qk_proj.weight', 'q_layernorm.weight', 'k_layernorm.weight']
    weights: dict[str, torch.Tensor] = {}
    for n in names:
        full = f'{prefix}.{n}'
        shard = index[full]
        with safe_open(os.path.join(args.snapshot, shard), framework='pt') as fh:
            weights[n] = fh.get_tensor(full)

    mod = Qwen4ExpTextQSAIndexer(config, layer_idx=LAYER).to(torch.float32).eval()
    with torch.no_grad():
        mod.index_qk_proj.weight.copy_(weights['index_qk_proj.weight'].float())
        mod.q_layernorm.weight.copy_(weights['q_layernorm.weight'].float())
        mod.k_layernorm.weight.copy_(weights['k_layernorm.weight'].float())

    torch.manual_seed(SEED)
    hidden = bf16_roundtrip(torch.randn(1, T, config.hidden_size) * 0.5)

    rot = Qwen4ExpTextRotaryEmbedding(config)
    position_ids = torch.arange(T).unsqueeze(0)
    cos, sin = rot(hidden, position_ids)

    causal = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)

    with torch.no_grad():
        mask = mod(hidden, (cos, sin), causal, None)
    # bool 4D [1,1,T,T]
    selected = mask.squeeze(0).squeeze(0).to(torch.bool)

    # Intermediates via the module's own children + the reference rope fn.
    with torch.no_grad():
        hd = mod.index_head_dim
        qk = mod.index_qk_proj(hidden)
        q, token_k = torch.split(
            qk, [mod.index_n_heads * hd, mod.index_kv_heads * hd], dim=-1)
        q = q.reshape(1, T, -1, hd)
        raw_keys = token_k.reshape(1, T, -1, hd).squeeze(2)
        q = mod.q_layernorm(q)
        q = apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=2)

        ratio = mod.compress_ratio
        n_blocks = T // ratio
        block_tok = torch.arange(n_blocks * ratio).view(n_blocks, ratio)
        key_groups = raw_keys[0].index_select(0, block_tok.flatten())
        key_groups = key_groups.view(n_blocks, ratio, hd)
        pooled = key_groups.float().mean(dim=1)
        pooled = mod.k_layernorm(pooled)
        starts = block_tok[:, 0]
        block_keys = apply_rotary_pos_emb(
            pooled.unsqueeze(1),
            cos=cos[0].index_select(0, starts),
            sin=sin[0].index_select(0, starts),
        ).squeeze(1)

        scores_last = torch.matmul(
            q[0, T - 1].float(), block_keys.float().transpose(-1, -2)
        ).transpose(-1, -2)
        scores_last = torch.relu(scores_last).sum(dim=-1) / math.sqrt(hd)

    sel_counts = selected.sum(dim=-1)
    print(f'T={T} blocks={n_blocks} budget={mod.token_budget} '
          f'topk_blocks={mod.block_topk}')
    print(f'selected-count first over-budget query (i=2051): {sel_counts[2051]}')
    print(f'selected-count last query: {sel_counts[-1]} '
          f'(expect {mod.token_budget + (T % ratio if T % ratio else 0)})')
    assert sel_counts[100].item() == 101, 'inert region must be all-visible'
    assert sel_counts[-1].item() <= mod.token_budget + ratio - 1

    np.savez_compressed(
        args.out,
        hidden_u16=hidden.squeeze(0).to(torch.bfloat16).view(torch.uint16).numpy(),
        selected_mask=selected.numpy(),
        q_post=q.squeeze(0).numpy().astype(np.float32),
        raw_keys=raw_keys.squeeze(0).numpy().astype(np.float32),
        block_keys=block_keys.numpy().astype(np.float32),
        scores_last=scores_last.numpy().astype(np.float32),
        meta=np.array([T, n_blocks, mod.token_budget, mod.block_topk,
                       mod.compress_ratio, mod.index_n_heads, hd, LAYER],
                      dtype=np.int64),
    )
    print(f'wrote {args.out}')

    # Raw-bin mirror for the Rust GPU parity harness (hc_golden convention:
    # one .bin per tensor + meta.json; ATLAS_QSA_TEST_DATA points here).
    bin_dir = args.out.rsplit('.', 1)[0] + '_bins'
    os.makedirs(bin_dir, exist_ok=True)

    def wbin(name: str, t: torch.Tensor, bf16: bool) -> None:
        a = t.detach().contiguous()
        raw = (a.to(torch.bfloat16).view(torch.uint16) if bf16
               else a.to(torch.float32).view(torch.int32)).numpy().tobytes()
        open(os.path.join(bin_dir, f'{name}.bin'), 'wb').write(raw)

    wbin('hidden', hidden.squeeze(0), bf16=True)
    wbin('q_post', q.squeeze(0), bf16=False)
    wbin('raw_keys', raw_keys.squeeze(0), bf16=False)
    wbin('block_keys', block_keys, bf16=False)
    wbin('scores_last', scores_last, bf16=False)
    wbin('w_qk_proj', weights['index_qk_proj.weight'], bf16=True)
    wbin('w_q_norm', weights['q_layernorm.weight'], bf16=True)
    wbin('w_k_norm', weights['k_layernorm.weight'], bf16=True)
    # cos/sin for all T positions, rotary width (64): what the engine's rope
    # tables hold. FP32.
    wbin('rope_cos', cos.squeeze(0), bf16=False)
    wbin('rope_sin', sin.squeeze(0), bf16=False)
    # The LAST query's selected token set, sorted, as i32 — the decode
    # contract (row T-1 of the mask).
    last_sel = torch.nonzero(selected[T - 1], as_tuple=False).flatten().to(torch.int32)
    open(os.path.join(bin_dir, 'selected_last.bin'), 'wb').write(
        last_sel.numpy().tobytes())
    # Stage 2: the FULL mask for every selective row (pos >= budget+ratio-1),
    # row-major u8 [n_sel_rows, T] — the prefill-selection set contract.
    bound = mod.token_budget + mod.compress_ratio - 1
    sel_rows = selected[bound:].to(torch.uint8)
    open(os.path.join(bin_dir, 'mask_sel_rows.bin'), 'wb').write(
        sel_rows.numpy().tobytes())
    json.dump(
        {
            'num_tokens': T, 'n_blocks': n_blocks,
            'token_budget': mod.token_budget, 'block_topk': mod.block_topk,
            'compress_ratio': mod.compress_ratio,
            'index_n_heads': mod.index_n_heads, 'index_head_dim': hd,
            'hidden_size': config.hidden_size,
            'rms_norm_eps': config.rms_norm_eps,
            'rotary_dim': int(cos.shape[-1]),
            'layer': LAYER, 'seed': SEED,
            'norm_convention': 'normed * (1.0 + weight)',
            'selected_last_count': int(last_sel.numel()),
        },
        open(os.path.join(bin_dir, 'meta.json'), 'w'), indent=1)
    print(f'wrote {bin_dir}')


if __name__ == '__main__':
    main()
