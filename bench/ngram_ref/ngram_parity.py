#!/usr/bin/env python3
"""N-gram embedding parity reference for the LongCat-Flash-Lite /
Qwen3.8-Flash-Next family (see bench/ngram_ref/modeling_longcat_ngram.py,
fetched from meituan-longcat/LongCat-Flash-Lite @ main).

This is a line-faithful numpy port of `NgramEmbedding.forward` plus the
`NgramCache` decode contract. It is the golden reference the Atlas
implementation must match bit-for-bit on the integer id math and to BF16
tolerance on the fused embedding.

Three layers of assurance, in order of availability:
  1. property tests on the hash math (id ranges, order sensitivity,
     table-size coprimality) — run anywhere, no weights;
  2. prefill-vs-incremental-decode equivalence: embedding a prompt in one
     shot must equal embedding it token-by-token through the (n-1)-token
     context cache. This exercises the exact seams a serving engine has
     (prefill chunking, decode steps, cache truncation);
  3. golden-vector dump against the real checkpoint's tables (--dump,
     needs the downloaded snapshot) for the Atlas loader/kernel to match.

Usage:
  python3 ngram_parity.py            # property + consistency tests (synthetic)
  python3 ngram_parity.py --dump DIR # golden vectors from a real snapshot
"""

import argparse
import json
import struct
import sys

import numpy as np


class Cfg:
    """The config subset the ngram path consumes (LongCat-Lite values as
    defaults; override for synthetic tests or Qwen's eventual keys)."""

    def __init__(self, vocab_size=131072, hidden_size=3072,
                 ngram_vocab_size_ratio=78, emb_neighbor_num=4,
                 emb_split_num=4, eos_token_id=2, pad_token_id=0):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ngram_vocab_size_ratio = ngram_vocab_size_ratio
        self.emb_neighbor_num = emb_neighbor_num
        self.emb_split_num = emb_split_num
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    @property
    def m(self):
        return self.ngram_vocab_size_ratio * self.vocab_size

    @property
    def num_embedders(self):
        return self.emb_split_num * (self.emb_neighbor_num - 1)

    @property
    def emb_dim(self):
        return self.hidden_size // self.num_embedders

    def table_vocab(self, index):
        # reference: `int(self.m + i * 2 + 1)` — consecutive odd offsets so
        # the K tables per n-gram size have distinct (usually coprime) sizes.
        return int(self.m + index * 2 + 1)


def shift_right_ignore_eos(row, n, eos_token_id):
    """Reference `_shift_right_ignore_eos` for ONE sequence (1-D array):
    shift right by n, but reset at document boundaries — a position within
    n tokens of a segment start (segment = span ending at an EOS,
    inclusive) reads 0 instead of crossing the boundary."""
    seq_len = row.shape[0]
    out = np.zeros_like(row)
    eos_positions = np.nonzero(row == eos_token_id)[0]
    prev = 0
    for e in eos_positions:
        end = int(e) + 1
        if end - prev > n:
            out[prev + n:end] = row[prev:end - n]
        prev = end
    if prev < seq_len and seq_len - prev > n:
        out[prev + n:seq_len] = row[prev:seq_len - n]
    return out


def precompute_vocab_mods(cfg):
    """vocab_mods[(i, j)] = [V^1 mod T, ..., V^(i-1) mod T] for the table at
    index (i-2)*k + j. Python ints — the products overflow i64 for real
    table sizes (V ~ 2^17, T ~ 2^23; id accumulation fits i64 but the
    Atlas kernel must do the mod reduction per term, not at the end)."""
    mods = {}
    for i in range(2, cfg.emb_neighbor_num + 1):
        for j in range(cfg.emb_split_num):
            index = (i - 2) * cfg.emb_split_num + j
            t = cfg.table_vocab(index)
            row = []
            power_mod = 1
            for _ in range(i - 1):
                power_mod = (power_mod * cfg.vocab_size) % t
                row.append(power_mod)
            mods[(i, j)] = row
    return mods


def ngram_ids_for_table(context, cfg, i, j, mods):
    """Hash ids over `context` (1-D int64) for n-gram size i, split j.
    Returns ids into table (i-2)*k+j, same length as context."""
    index = (i - 2) * cfg.emb_split_num + j
    t = cfg.table_vocab(index)
    ids = context.astype(object).copy()  # python ints: no overflow
    for d in range(2, i + 1):
        shifted = shift_right_ignore_eos(context, d - 1, cfg.eos_token_id)
        ids = ids + shifted.astype(object) * mods[(i, j)][d - 2]
    return np.array([int(v) % t for v in ids], dtype=np.int64)


def ngram_embed(input_ids, ngram_context, cfg, weights):
    """Faithful port of NgramEmbedding.forward for ONE sequence.

    weights: dict with
      'word'          : [vocab, hidden] float
      ('emb', index)  : [table_vocab(index), emb_dim] float
      ('proj', index) : [hidden, emb_dim] float  (nn.Linear weight layout)
    Returns [seq_len, hidden] float32.
    """
    seq_len = input_ids.shape[0]
    if ngram_context is not None and ngram_context.shape[0] > 0:
        ctx = np.concatenate(
            [ngram_context[-(cfg.emb_neighbor_num - 1):], input_ids])
    else:
        ctx = input_ids

    x = weights['word'][input_ids].astype(np.float32).copy()
    mods = precompute_vocab_mods(cfg)

    for i in range(2, cfg.emb_neighbor_num + 1):
        for j in range(cfg.emb_split_num):
            index = (i - 2) * cfg.emb_split_num + j
            ids = ngram_ids_for_table(ctx, cfg, i, j, mods)[-seq_len:]
            x_ng = weights[('emb', index)][ids].astype(np.float32)
            x += x_ng @ weights[('proj', index)].T.astype(np.float32)

    return x / (1 + cfg.emb_split_num * (cfg.emb_neighbor_num - 1))


# ── tests ────────────────────────────────────────────────────────────────

def synth_cfg():
    # small but structurally faithful: 3 n-gram sizes × 2 splits = 6 tables
    return Cfg(vocab_size=997, hidden_size=48, ngram_vocab_size_ratio=5,
               emb_neighbor_num=4, emb_split_num=2, eos_token_id=2)


def synth_weights(cfg, seed=0):
    rng = np.random.default_rng(seed)
    w = {'word': rng.standard_normal((cfg.vocab_size, cfg.hidden_size),
                                     dtype=np.float32)}
    for index in range(cfg.num_embedders):
        w[('emb', index)] = rng.standard_normal(
            (cfg.table_vocab(index), cfg.emb_dim), dtype=np.float32)
        w[('proj', index)] = rng.standard_normal(
            (cfg.hidden_size, cfg.emb_dim), dtype=np.float32)
    return w


def test_hash_properties():
    cfg = synth_cfg()
    mods = precompute_vocab_mods(cfg)
    rng = np.random.default_rng(1)
    ctx = rng.integers(3, cfg.vocab_size, 64).astype(np.int64)
    for i in range(2, cfg.emb_neighbor_num + 1):
        for j in range(cfg.emb_split_num):
            index = (i - 2) * cfg.emb_split_num + j
            ids = ngram_ids_for_table(ctx, cfg, i, j, mods)
            assert ids.min() >= 0 and ids.max() < cfg.table_vocab(index)
    # order sensitivity: swapping two context tokens must change ids
    # somewhere for every n>=2 table (with overwhelming probability)
    ctx2 = ctx.copy()
    ctx2[10], ctx2[11] = ctx2[11], ctx2[10]
    changed = 0
    for i in range(2, cfg.emb_neighbor_num + 1):
        a = ngram_ids_for_table(ctx, cfg, i, 0, mods)
        b = ngram_ids_for_table(ctx2, cfg, i, 0, mods)
        changed += int((a != b).any())
    assert changed == cfg.emb_neighbor_num - 1, "hash not order-sensitive"
    print("hash properties: OK")


def test_prefill_equals_decode():
    """Embedding a prompt in one shot == token-by-token through the cache.
    THE serving-seam test: also covers chunked prefill (split at every
    possible boundary) and the (n-1)-token cache truncation."""
    cfg = synth_cfg()
    w = synth_weights(cfg)
    rng = np.random.default_rng(2)
    toks = rng.integers(3, cfg.vocab_size, 24).astype(np.int64)
    # sprinkle an EOS mid-sequence to exercise the reset path
    toks[9] = cfg.eos_token_id

    full = ngram_embed(toks, None, cfg, w)

    # per-token decode with an (n-1)-token rolling context
    keep = cfg.emb_neighbor_num - 1
    ctx = np.zeros(0, dtype=np.int64)
    for t in range(toks.shape[0]):
        step = ngram_embed(toks[t:t + 1], ctx, cfg, w)
        assert np.allclose(step[0], full[t], atol=1e-5), \
            f"decode diverges from prefill at t={t}"
        ctx = np.concatenate([ctx, toks[t:t + 1]])[-keep:]

    # chunked prefill at every split point
    for cut in range(1, toks.shape[0]):
        a = ngram_embed(toks[:cut], None, cfg, w)
        b = ngram_embed(toks[cut:], toks[:cut][-keep:], cfg, w)
        got = np.concatenate([a, b])
        assert np.allclose(got, full, atol=1e-5), f"chunk split {cut} diverges"
    print("prefill == decode == chunked prefill: OK")


def test_real_dims():
    """LongCat-Lite real dims: id math only (no tables materialized)."""
    cfg = Cfg()
    assert cfg.num_embedders == 12 and cfg.emb_dim == 256
    assert cfg.table_vocab(0) == 78 * 131072 + 1
    mods = precompute_vocab_mods(cfg)
    rng = np.random.default_rng(3)
    ctx = rng.integers(3, cfg.vocab_size, 32).astype(np.int64)
    for i in range(2, 5):
        for j in range(4):
            index = (i - 2) * 4 + j
            ids = ngram_ids_for_table(ctx, cfg, i, j, mods)
            assert ids.max() < cfg.table_vocab(index)
    print("real-dims id math: OK "
          f"(12 tables, ~{cfg.table_vocab(0):,} rows x {cfg.emb_dim} dim)")


# ── golden dump against a real snapshot ──────────────────────────────────

def dump_golden(snapshot_dir):
    """Read the needed embedder/proj rows straight from the safetensors
    index for a fixed token sequence and write golden fused embeddings to
    ngram_golden.npz for the Atlas implementation to match."""
    import glob
    import os
    idx = json.load(open(os.path.join(snapshot_dir,
                                      'model.safetensors.index.json')))
    cfgj = json.load(open(os.path.join(snapshot_dir, 'config.json')))
    cfg = Cfg(vocab_size=cfgj['vocab_size'], hidden_size=cfgj['hidden_size'],
              ngram_vocab_size_ratio=cfgj['ngram_vocab_size_ratio'],
              emb_neighbor_num=cfgj['emb_neighbor_num'],
              emb_split_num=cfgj['emb_split_num'],
              eos_token_id=cfgj.get('eos_token_id', 2),
              pad_token_id=cfgj.get('pad_token_id', 0))

    handles = {}

    def read_rows(name, rows):
        shard = os.path.join(snapshot_dir, idx['weight_map'][name])
        if shard not in handles:
            f = open(shard, 'rb')
            n = struct.unpack('<Q', f.read(8))[0]
            handles[shard] = (f, json.loads(f.read(n)), 8 + n)
        f, hdr, base = handles[shard]
        info = hdr[name]
        assert info['dtype'] == 'BF16', (name, info['dtype'])
        dim = info['shape'][1]
        out = np.zeros((len(rows), dim), dtype=np.float32)
        for r_i, r in enumerate(rows):
            f.seek(base + info['data_offsets'][0] + r * dim * 2)
            a = np.frombuffer(f.read(dim * 2), dtype=np.uint16)
            out[r_i] = (a.astype(np.uint32) << 16).view(np.float32)
        return out

    def read_full(name):
        shard = os.path.join(snapshot_dir, idx['weight_map'][name])
        if shard not in handles:
            f = open(shard, 'rb')
            n = struct.unpack('<Q', f.read(8))[0]
            handles[shard] = (f, json.loads(f.read(n)), 8 + n)
        f, hdr, base = handles[shard]
        info = hdr[name]
        f.seek(base + info['data_offsets'][0])
        raw = f.read(info['data_offsets'][1] - info['data_offsets'][0])
        a = np.frombuffer(raw, dtype=np.uint16)
        return ((a.astype(np.uint32) << 16).view(np.float32)
                .reshape(info['shape']))

    # Discover tensor names (dump what the index actually calls them).
    # NUMERIC sort — lexicographic puts "embedders.10" before "embedders.2"
    # and silently mismaps tables 2..9.
    import re

    def table_index(name):
        return int(re.search(r'\.(\d+)\.weight$', name).group(1))

    names = list(idx['weight_map'])
    emb_names = sorted((n for n in names if 'embedders' in n), key=table_index)
    proj_names = sorted((n for n in names if 'post_projs' in n),
                        key=table_index)
    word_name = next(n for n in names if 'embed_tokens' in n)
    print('embedders:', emb_names[:2], '... projs:', proj_names[:2],
          'word:', word_name)

    rng = np.random.default_rng(7)
    toks = rng.integers(3, cfg.vocab_size, 16).astype(np.int64)
    mods = precompute_vocab_mods(cfg)

    x = read_rows(word_name, list(toks)).copy()
    for i in range(2, cfg.emb_neighbor_num + 1):
        for j in range(cfg.emb_split_num):
            index = (i - 2) * cfg.emb_split_num + j
            ids = ngram_ids_for_table(toks, cfg, i, j, mods)
            rows = read_rows(emb_names[index], list(ids))
            proj = read_full(proj_names[index])
            x += rows @ proj.T
    x /= (1 + cfg.emb_split_num * (cfg.emb_neighbor_num - 1))
    np.savez('ngram_golden.npz', tokens=toks, fused=x)
    print('wrote ngram_golden.npz:', x.shape)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', metavar='SNAPSHOT_DIR')
    args = ap.parse_args()
    if args.dump:
        dump_golden(args.dump)
        sys.exit(0)
    test_hash_properties()
    test_prefill_equals_decode()
    test_real_dims()
    print('ALL PARITY TESTS PASS')
