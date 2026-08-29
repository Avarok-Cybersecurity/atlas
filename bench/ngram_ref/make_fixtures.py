#!/usr/bin/env python3
"""Emit cross-language n-gram id fixtures (consumed by the Rust unit test
in spark-model's ngram module). Run from bench/ngram_ref/."""
import json

import numpy as np

from ngram_parity import Cfg, precompute_vocab_mods, ngram_ids_for_table

fixtures = []
for name, cfg, toks in [
    ("synth", Cfg(vocab_size=997, hidden_size=48, ngram_vocab_size_ratio=5,
                  emb_neighbor_num=4, emb_split_num=2, eos_token_id=2),
     [901, 15, 371, 2, 88, 990, 41, 7, 640, 3, 55, 2, 118, 42]),
    ("longcat_lite", Cfg(),  # real LongCat-Lite dims
     [131071, 5, 88231, 2, 42, 77777, 130000, 999, 1, 65536, 2, 31337]),
]:
    mods = precompute_vocab_mods(cfg)
    ctx = np.array(toks, dtype=np.int64)
    tables = {}
    for i in range(2, cfg.emb_neighbor_num + 1):
        for j in range(cfg.emb_split_num):
            index = (i - 2) * cfg.emb_split_num + j
            ids = ngram_ids_for_table(ctx, cfg, i, j, mods)
            tables[str(index)] = [int(v) for v in ids]
    fixtures.append({
        "name": name,
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "ngram_vocab_size_ratio": cfg.ngram_vocab_size_ratio,
        "emb_neighbor_num": cfg.emb_neighbor_num,
        "emb_split_num": cfg.emb_split_num,
        "eos_token_id": cfg.eos_token_id,
        "tokens": toks,
        "expected_ids": tables,
    })
json.dump(fixtures, open('ngram_id_fixtures.json', 'w'), indent=1)
print('fixtures written:', [f['name'] for f in fixtures])
