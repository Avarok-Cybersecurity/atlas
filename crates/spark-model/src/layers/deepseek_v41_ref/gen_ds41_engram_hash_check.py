#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# provenance-id: 526f6e616c6420522e205374657369616b
"""Tier-2 check: the engram hash parameters shipped in the Q2_K GGUF are what DeepSeek's
`engram.py` computes from the real V4.1 tokenizer.

Recomputes, from `deepseek-ai/DeepSeek-V4.1-Flash @ dba1be0a` `inference/engram.py` and the
published tokenizer: the compressed token map, the per-(layer, n-gram, head) bucket primes, the
per-slot offsets, and the per-(layer, lookback) multipliers. Compares each to the
`deepseek41.engram.*` metadata in the vcruz305 Q2_K shard 0. No model forward, no GPU.

  python3 crates/spark-model/src/layers/deepseek_v41_ref/gen_ds41_engram_hash_check.py
"""
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np

REF = "/home/rstesiak/code/ds4-0731-dspark/reference/deepseek-v41"
GGUF = "/home/rstesiak/models/dsv41-q2k/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf"
sys.path.insert(0, REF + "/inference")
sys.path.insert(0, "/home/rstesiak/code/ds4-0731-dspark/tools")
import engram as E  # noqa: E402
import gguf_manifest as G  # noqa: E402


def read_engram_meta(path):
    """Read every deepseek41.engram.* KV in full (the manifest tool elides big arrays)."""
    want = {}
    with open(path, "rb") as f:
        r = G.R(f)
        assert f.read(4) == b"GGUF"
        r.u32(); r.u64(); nkv = r.u64()
        for _ in range(nkv):
            k = r.st(); t = r.u32()
            if t == 9:
                et = r.u32(); n = r.u64()
                if k.startswith("deepseek41.engram."):
                    fmt = {4: "<%dI", 5: "<%di", 10: "<%dQ", 11: "<%dq"}[et]
                    sz = {4: 4, 5: 4, 10: 8, 11: 8}[et]
                    want[k] = list(struct.unpack(fmt % n, f.read(sz * n)))
                    continue
                sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(et)
                if sz:
                    f.seek(n * sz, 1)
                else:
                    for _ in range(n):
                        r.val(et)
            else:
                v = r.val(t)
                if k.startswith("deepseek41.engram."):
                    want[k] = v
    return want


def cmp(name, got, want):
    got, want = list(got), list(want)
    if got == want:
        print(f"  PASS  {name:32s} n={len(got)}")
        return True
    if len(got) != len(want):
        print(f"  FAIL  {name:32s} len {len(got)} vs GGUF {len(want)}")
        return False
    i = next(j for j in range(len(got)) if got[j] != want[j])
    print(f"  FAIL  {name:32s} first mismatch at [{i}]: computed {got[i]} vs GGUF {want[i]}")
    return False


def main():
    t0 = time.time()
    meta = read_engram_meta(GGUF)
    print("GGUF engram keys:", sorted(meta))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REF, trust_remote_code=False)
    print(f"tokenizer loaded: {len(tok)} ids  ({time.time()-t0:.1f}s)")

    args = SimpleNamespace(
        engram_layer_ids=tuple(meta["deepseek41.engram.layer_ids"]),
        engram_max_ngram_size=meta["deepseek41.engram.max_ngram_size"],
        engram_n_heads=meta["deepseek41.engram.head_count"],
        engram_head_dim=meta["deepseek41.engram.key_length"],
        engram_vocab_size=16_000_000,      # HF config; NOT in the GGUF, see note below
        engram_num_embeddings=(384006168, 384016682),
        engram_pad_id=meta["deepseek41.engram.pad_id"],
        engram_compressed_vocab_size=99092,
    )
    print(f"layout args: layers={args.engram_layer_ids} ngram={args.engram_max_ngram_size} "
          f"heads={args.engram_n_heads} pad_id={args.engram_pad_id}")

    token_map, vocab = E.build_compressed_token_map(tok)
    print(f"compressed vocab computed: {vocab}   ({time.time()-t0:.1f}s)")
    ok = vocab == args.engram_compressed_vocab_size
    print(f"  {'PASS' if ok else 'FAIL'}  compressed_vocab_size          {vocab} vs config 99092")
    ok &= cmp("token_map", token_map, meta["deepseek41.engram.token_map"])
    print(f"  info  compressed pad id = token_map[{args.engram_pad_id}] = {token_map[args.engram_pad_id]}")

    layout = E.EngramLayout.from_args(args)
    flat_primes = [p for layer in layout.primes for per_ngram in layer for p in per_ngram]
    ok &= cmp("primes", flat_primes, meta["deepseek41.engram.primes"])
    offsets = []
    for layer in layout.primes:
        sizes = [p for per_ngram in layer for p in per_ngram]
        offsets += [int(x) for x in np.cumsum([0, *sizes[:-1]])]
    ok &= cmp("offsets", offsets, meta["deepseek41.engram.offsets"])
    mult = E.compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab)
    ok &= cmp("multipliers", [int(x) for x in mult.flatten().tolist()],
              meta["deepseek41.engram.multipliers"])
    rows_needed = [sum(p for per_ngram in layer for p in per_ngram) for layer in layout.primes]
    print(f"  info  rows addressable per layer (sum of primes) = {rows_needed}; "
          f"tensor rows = {list(args.engram_num_embeddings)}")
    print(f"\n{'ALL PASS' if ok else 'MISMATCH'}  ({time.time()-t0:.1f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
