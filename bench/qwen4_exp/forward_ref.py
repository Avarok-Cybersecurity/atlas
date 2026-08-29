#!/usr/bin/env python3
"""Streaming reference forward for Qwen3.8-Flash-Next — full-model logits
golden for the sampled-quality (KL) question.

Unlike the longcat numpy port, this DRIVES THE REAL transformers modules
(transformers 5.16.1 ships qwen4_exp): one decoder layer is instantiated,
loaded and run at a time, so peak RAM is one layer (~5 GB, the 512
experts) instead of 102 GB. Three checkpoint-specific adaptations:

  * routed experts are NVFP4 (packed e2m1 U8 + per-16 F8_E4M3 scales +
    global F32) — dequantized exactly to f32 on load;
  * the PLE n-gram embedding is 51 B params across 128 shards — the
    nn.Embedding is swapped for a row-on-demand lookup that reads exactly
    the rows the fixture hashes to (ple_golden.py's gather_rows);
  * everything else is BF16 and loads directly.

Output npz: fixture tokens + f32 logits at EVERY position + per-layer
hyper-stream RMS (debug trail). Compare against the engine with
ATLAS_DUMP_LOGITS_PATH rows (vocab width = model.vocab_size() = 248077 —
NOT config vocab_size 248320; see b7b7bcf3).

Usage: forward_ref.py [SNAPSHOT_DIR] [OUT_NPZ]
  env LAYER_LIMIT=n  run only the first n layers (smoke)
"""
import ctypes
import gc
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  (module forwards use it)
from safetensors import safe_open

torch.set_grad_enabled(False)
torch.set_num_threads(max(8, os.cpu_count() // 2))

SNAP = sys.argv[1] if len(sys.argv) > 1 else (
    '/tank/hf/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4/snapshots/'
    '129972269565f7f4f664fdf8dd42268d3bbda9fd')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'qwen4exp_forward_golden.npz')
LAYER_LIMIT = int(os.environ.get('LAYER_LIMIT', '0'))

IDX = json.load(open(os.path.join(SNAP, 'model.safetensors.index.json')))[
    'weight_map']
HANDLES = {}


def raw_tensor(name):
    """One tensor from the shards, as torch (BF16 kept, U8 kept, F32 kept)."""
    path = os.path.join(SNAP, IDX[name])
    if path not in HANDLES:
        HANDLES[path] = safe_open(path, framework='pt')
    return HANDLES[path].get_tensor(name)


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_nvfp4(prefix, out=None):
    """`prefix`.{weight,weight_scale,weight_scale_2} -> exact f32 [O, I].

    `out`: preallocated destination (a slice of a persistent fused buffer) —
    dequantizing 1536 experts/layer through fresh allocations fragments
    glibc arenas until RSS hits ~115 GB and earlyoom SIGTERMs the run
    (observed three times at the same spot); writing in place keeps the
    process flat."""
    packed = raw_tensor(prefix + '.weight')            # U8 [O, I/2]
    scale = raw_tensor(prefix + '.weight_scale')       # F8_E4M3 [O, I/16]
    scale2 = raw_tensor(prefix + '.weight_scale_2').float()  # scalar
    lo = packed & 0x0F
    hi = packed >> 4
    def nib(n):
        return torch.where((n & 8).bool(), -E2M1[(n & 7).long()],
                           E2M1[(n & 7).long()])
    o, half = packed.shape
    w = out if out is not None else torch.empty(o, half * 2,
                                                dtype=torch.float32)
    w[:, 0::2] = nib(lo)
    w[:, 1::2] = nib(hi)
    w *= scale.float().repeat_interleave(16, dim=1)
    w *= scale2
    return w


def is_nvfp4(prefix):
    return (prefix + '.weight_scale') in IDX


def site_state_dict(ckpt_prefix):
    """All params/buffers under `ckpt_prefix.` as an f32 state dict with the
    prefix stripped; NVFP4 groups collapsed to their dequantized weight;
    n-gram embedding shards excluded (the lookup shim owns them)."""
    plen = len(ckpt_prefix) + 1
    out = {}
    seen_nvfp4 = set()
    for k in IDX:
        if not k.startswith(ckpt_prefix + '.'):
            continue
        local = k[plen:]
        if '.ngram_embedding.shard_' in local:
            continue
        if local.startswith('mlp.experts.'):
            continue  # fused separately into persistent buffers
        m = re.match(r'(.*)\.(weight|weight_scale|weight_scale_2|input_scale)$',
                     local)
        if m and is_nvfp4(ckpt_prefix + '.' + m.group(1)):
            base = m.group(1)
            if base in seen_nvfp4:
                continue
            seen_nvfp4.add(base)
            out[base + '.weight'] = dequant_nvfp4(ckpt_prefix + '.' + base)
            continue
        t = raw_tensor(k)
        out[local] = t.float() if t.is_floating_point() else t
    return out


class RowLookup(torch.nn.Module):
    """nn.Embedding stand-in reading exactly the requested rows from the
    128 n-gram shards (51 B params never materialize)."""

    def __init__(self, ckpt_prefix, dim):
        super().__init__()
        self.shards = sorted(
            (k for k in IDX if k.startswith(ckpt_prefix)
             and '.ngram_embedding.shard_' in k),
            key=lambda k: int(re.search(r'shard_(\d+)', k).group(1)))
        with safe_open(os.path.join(SNAP, IDX[self.shards[0]]),
                       framework='pt') as fh:
            self.rows_per_shard, d = fh.get_slice(self.shards[0]).get_shape()
        assert d == dim, (d, dim)
        # `.weight.device` is read by the reference forward.
        self.weight = torch.nn.Parameter(torch.zeros(1, dim),
                                         requires_grad=False)
        self.dim = dim

    def forward(self, ids):
        flat = ids.reshape(-1)
        uniq, inverse = torch.unique(flat, return_inverse=True)
        rows = torch.empty(len(uniq), self.dim, dtype=torch.float32)
        for j, gid in enumerate(uniq.tolist()):
            s, local = divmod(int(gid), self.rows_per_shard)
            name = self.shards[s]
            path = os.path.join(SNAP, IDX[name])
            if path not in HANDLES:
                HANDLES[path] = safe_open(path, framework='pt')
            rows[j] = HANDLES[path].get_slice(name)[local:local + 1].float()
        return rows[inverse].reshape(*ids.shape, self.dim)


FUSED = {}


def fuse_experts(ckpt_prefix, sd, tc):
    """Dequantize per-expert checkpoint tensors straight into TWO
    persistent fused buffers matching the reference module's grouped
    layout: gate_up_proj [E, 2I, H] (gate rows then up rows — the forward
    `chunk(2)`s the output), down_proj [E, H, I]. The buffers are reused
    for every layer (the layer is loaded with assign=True and dropped
    before the next fill), so the 10 GB is allocated exactly once."""
    if not is_nvfp4(ckpt_prefix + '.mlp.experts.0.gate_proj'):
        return
    e_n, i_n, h_n = tc.num_experts, tc.moe_intermediate_size, tc.hidden_size
    if not FUSED:
        FUSED['gate_up'] = torch.empty(e_n, 2 * i_n, h_n)
        FUSED['down'] = torch.empty(e_n, h_n, i_n)
    for e in range(e_n):
        pfx = f'{ckpt_prefix}.mlp.experts.{e}'
        dequant_nvfp4(pfx + '.gate_proj', out=FUSED['gate_up'][e, :i_n])
        dequant_nvfp4(pfx + '.up_proj', out=FUSED['gate_up'][e, i_n:])
        dequant_nvfp4(pfx + '.down_proj', out=FUSED['down'][e])
    sd['mlp.experts.gate_up_proj'] = FUSED['gate_up']
    sd['mlp.experts.down_proj'] = FUSED['down']


def main():
    from transformers import AutoTokenizer
    from transformers.masking_utils import (create_causal_mask,
                                            create_recurrent_attention_mask)
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpTextConfig)
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual,
        Qwen4ExpTextRotaryEmbedding)

    # The checkpoint's nested model_type (qwen3_8_flash_next_text) does not
    # match the registered qwen4_exp_text, so from_pretrained silently drops
    # the dict and returns DEFAULTS (hidden 2048 != 2560). Build the text
    # config straight from the json instead.
    tc_json = json.load(open(os.path.join(SNAP, 'config.json')))['text_config']
    tc_json.pop('model_type', None)
    tc = Qwen4ExpTextConfig(**tc_json)
    tok = AutoTokenizer.from_pretrained(SNAP)

    msgs = [{'role': 'user', 'content': 'Hello, what model are you?'}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True,
                                     reasoning_effort='low')
    ids = tok(prompt, return_tensors='pt').input_ids
    # QWEN4EXP_EXTRA_IDS: comma-separated token ids appended to the fixture
    # (the engine's greedy continuation) so reference positions past the
    # prompt align one-to-one with the engine's decode-step dump rows.
    extra = os.environ.get('QWEN4EXP_EXTRA_IDS', '')
    if extra:
        more = torch.tensor([[int(t) for t in extra.split(',')]])
        ids = torch.cat([ids, more], dim=1)
    print(f'fixture: {ids.shape[1]} tokens', flush=True)

    # ── model-level orchestration, transcribed from Qwen4ExpTextModel.forward ──
    embed_w = raw_tensor('model.language_model.embed_tokens.weight')
    inputs_embeds = embed_w[ids[0]].float().unsqueeze(0)
    del embed_w

    T = ids.shape[1]
    position_ids = torch.arange(T).view(1, 1, -1).expand(4, 1, -1)
    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]

    mask_kwargs = {
        'config': tc, 'inputs_embeds': inputs_embeds, 'attention_mask': None,
        'past_key_values': None, 'position_ids': text_position_ids,
        'allow_is_causal_skip': False,
    }
    # create_causal_mask can still return None (the is-causal fast path)
    # even with allow_is_causal_skip=False on this transformers version —
    # and the QSA indexer dereferences the mask (`attention_mask == 0`).
    # Build the additive causal mask explicitly; `== 0` marks visibility.
    full_mask = create_causal_mask(**mask_kwargs)
    if full_mask is None:
        tril = torch.tril(torch.ones(T, T, dtype=torch.bool))
        full_mask = torch.zeros(1, 1, T, T)
        full_mask.masked_fill_(~tril, float('-inf'))
    causal_mask_mapping = {
        'full_attention': full_mask,
        'linear_attention': create_recurrent_attention_mask(**mask_kwargs),
    }
    conv_mask = causal_mask_mapping.get('linear_attention')
    ple_input_ids = ids
    if tc.ple_layer_ids and conv_mask is not None:
        eos = tc.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        ple_input_ids = torch.where(conv_mask.bool(), ple_input_ids, eos)

    rotary = Qwen4ExpTextRotaryEmbedding(tc)
    position_embeddings = rotary(inputs_embeds, position_ids)
    hidden = inputs_embeds.repeat(1, 1, tc.hc_count)

    n_layers = LAYER_LIMIT or tc.num_hidden_layers
    layer_rms = []
    saved_highway = {}

    # The PLE layer's CONSTRUCTOR materializes the full n-gram nn.Embedding
    # (billions of rows, ~90 GB) before any swap can happen — earlyoom killed
    # three runs at exactly this point. Stub any huge Embedding at
    # construction; RowLookup replaces it right after.
    real_embedding = torch.nn.Embedding

    class _TinyEmbedding(torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim, **kw):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.zeros(1, embedding_dim), requires_grad=False)

    def guarded_embedding(num_embeddings, embedding_dim, **kw):
        if num_embeddings > 10_000_000:
            return _TinyEmbedding(num_embeddings, embedding_dim)
        return real_embedding(num_embeddings, embedding_dim, **kw)

    for i in range(n_layers):
        torch.nn.Embedding = guarded_embedding
        try:
            layer = Qwen4ExpTextDecoderLayer(tc, i)
        finally:
            torch.nn.Embedding = real_embedding
        if layer.ple is not None:
            pfx = f'model.language_model.layers.{i}'
            layer.ple.ple_embedding.ngram_embedding = RowLookup(
                pfx, tc.ple_embed_dim // layer.ple.ple_embedding.ngram_heads)
        pfx = f'model.language_model.layers.{i}'
        sd = site_state_dict(pfx)
        fuse_experts(pfx, sd, tc)
        missing, unexpected = layer.load_state_dict(sd, strict=False,
                                                    assign=True)
        real_missing = [k for k in missing
                        if 'ngram_embedding.weight' not in k]
        assert not real_missing, f'layer {i} missing: {real_missing[:8]}'
        assert not unexpected, f'layer {i} unexpected: {unexpected[:8]}'
        layer = layer.float().eval()
        hidden = layer(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask_mapping['full_attention'],
            conv_mask=conv_mask,
            past_key_values=None,
            ple_input_ids=ple_input_ids,
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        rms = hidden.pow(2).mean().sqrt().item()
        layer_rms.append(rms)
        print(f'layer {i:2d}: hyper rms {rms:.4f}', flush=True)
        # QWEN4EXP_SAVE_HIGHWAY=n: keep the full post-layer highway tensors
        # for the first n layers (diffable against the engine's
        # ATLAS_QWEN4EXP_DUMP taps: post-layer-i == L{i}_post_moe ==
        # L{i+1}_in).
        if i < int(os.environ.get('QWEN4EXP_SAVE_HIGHWAY', '0')):
            saved_highway[f'highway_L{i:02d}'] = (
                hidden[0].float().numpy().astype(np.float32))
        del layer, sd
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0)

    mixer = Qwen4ExpTextGatedResidual(tc, use_combine=False)
    msd = site_state_dict('model.language_model.hyper_connection_mixer')
    mixer.load_state_dict(msd, strict=True)
    final_hidden = mixer.float().eval()(hidden)

    lm_w = raw_tensor('lm_head.weight').float()
    logits = final_hidden[0] @ lm_w.T
    print(f'logits: {tuple(logits.shape)}  '
          f'last-pos top1 logit {logits[-1].max().item():.3f}', flush=True)
    p = torch.softmax(logits[-1].double(), dim=-1)
    top = torch.topk(p, 5)
    for v, j in zip(top.values.tolist(), top.indices.tolist()):
        print(f'  ref last-pos: {tok.decode([j])!r} p={v:.4f}', flush=True)

    np.savez_compressed(
        OUT,
        tokens=ids[0].numpy(),
        logits=logits.numpy().astype(np.float32),
        layer_rms=np.array(layer_rms, dtype=np.float32),
        n_layers=np.array([n_layers]),
        **saved_highway,
    )
    print(f'wrote {OUT}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
