#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# provenance-id: 526f6e616c6420522e205374657369616b
"""DeepSeek-V4.1 Flash tiny-graph golden generator (stage S2 harness, #1059 / #970).

Runs DeepSeek's OWN reference `inference/model.py` (deepseek-ai/DeepSeek-V4.1-Flash @ dba1be0a)
on a tiny synthetic model that keeps every V4.1 structural feature:

  * hyper-connections with hc_mult=4, delayed mixes (attention uses the previous FFN's pre_mix)
  * engram on two layers, DeepSeek's n-gram hash (`engram.py`), fp8 table rows
  * shared compressed attention: two kv/index source layers, three consumers, ratios (0,0,2,2,1,1)
  * the layer-2 candidate prefilter, top-k indexer, sliding window, attention sinks
  * sqrt-softplus routing with correction bias, route_scale 1.5, swiglu clamp 10, shared expert

and emits every intermediate an Atlas implementation must reproduce. Nothing here re-derives an
equation: every value comes from the reference module, with `ds41_ref_shims` standing in for the
five tilelang kernels (pure torch, written from the tilelang source).

Determinism (RNG-free, order-free)
----------------------------------
Weights and inputs come from `fixed(name, i)`:  salt = fnv1a64(name);
z = splitmix64(salt XOR (i * 0x9E3779B97F4A7C15 mod 2^64));  u = (z >> 40) / 2^24  in [0,1);
value = f32( (2u - 1) * scale + offset ), f64 arithmetic. The Rust side reproduces this bit for
bit, so the golden commits only OUTPUTS: each capture as {shape, n, stride, ck, data} where `data`
is a prime-strided sample and `ck` the fp64 index-weighted checksum over the WHOLE tensor.

Run from the workspace root:   python3 crates/spark-model/src/layers/deepseek_v41_ref/gen_ds41_tiny_golden.py
"""
import functools
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REF = "/home/rstesiak/code/ds4-0731-dspark/reference/deepseek-v41/inference"
sys.path.insert(0, HERE)
sys.path.insert(0, REF)
import ds41_ref_shims  # noqa: E402

sys.modules["kernel"] = ds41_ref_shims
import engram as engram_mod  # noqa: E402
import model as M  # noqa: E402

torch.use_deterministic_algorithms(True)
MASK64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15

# ---------------------------------------------------------------- fixture geometry
# Every shape RELATIONSHIP of the production model holds; only the sizes are toy. Block-size
# asserts in the reference fix head_dim, index_head_dim and engram_head_dim at multiples of 32.
FIX = dict(
    vocab_size=64, dim=64, n_layers=6, n_heads=4, head_dim=32, rope_head_dim=8,
    q_lora_rank=16, o_lora_rank=16, o_groups=2,
    moe_inter_dim=24, n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
    score_func="sqrtsoftplus", route_scale=1.5, norm_topk_prob=True, swiglu_limit=10.0,
    window_size=8, compress_ratios=(0, 0, 2, 2, 1, 1),
    # Mirrors the production topology: the candidate source and ALL its consumers share a
    # compress ratio (production: source 20 and consumers 24/28/32/36 are every one ratio 1).
    # A ratio-2 source feeding a ratio-1 consumer would give the mask and the scores different
    # widths; the reference never does it and neither does this fixture.
    kv_source_layers=(2, 4), index_source_layers=(2, 4, 5),
    candidate_source_layer=4, candidate_topk_blocks=3, candidate_block_size=2,
    index_n_heads=2, index_head_dim=32, index_topk=6,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=(1, 4), engram_max_ngram_size=4, engram_vocab_size=64,
    engram_n_heads=2, engram_head_dim=32, engram_pad_id=2, engram_compressed_vocab_size=64,
    # rows = sum of that layer's bucket primes (the 6 primes above 63 per layer: 462, 630).
    # Asserted against EngramLayout below; the Tier-2 check proves the same identity holds for
    # the shipped Q2_K file (384006168 / 384016682).
    engram_num_embeddings=(462, 630),
    rope_theta=10000.0, compress_rope_theta=40000.0, rope_factor=2.0, original_seq_len=32,
    beta_fast=32, beta_slow=1, norm_eps=1e-20, gate_temp=1.0,
    vision_n_layers=0, dtype="bf16", expert_dtype=None, n_mtp_layers=0, dspark_block_size=0,
    max_batch_size=1, max_seq_len=64, temperature=0.0,
)
SOURCE_LAYERS = set(FIX["kv_source_layers"]) | set(FIX["index_source_layers"]) | {FIX["candidate_source_layer"]}
PREFILL = 12  # > window_size (8) and 12 % 8 != 0, so the ring buffer's cutoff branch runs
B = 1

# ---------------------------------------------------------------- RNG-free filler
def splitmix64(z: int) -> int:
    z = (z + GOLD) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def fnv1a64(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & MASK64
    return h


def fixed(name: str, n: int, scale: float, offset: float = 0.0) -> torch.Tensor:
    salt = np.uint64(fnv1a64(name))
    i = np.arange(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = salt ^ (i * np.uint64(GOLD))
        z = z + np.uint64(GOLD)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(40)).astype(np.float64) / float(1 << 24)
    v = (2.0 * u - 1.0) * scale + offset
    return torch.from_numpy(v.astype(np.float32))


def fixed_ints(name: str, n: int, modulus: int) -> torch.Tensor:
    salt = fnv1a64(name)
    return torch.tensor([splitmix64(salt ^ ((i * GOLD) & MASK64)) % modulus for i in range(n)])


# ---------------------------------------------------------------- weight rule
def rule(name: str, p: torch.Tensor):
    """(scale, offset, kind). kind: f32 | fp8 | e8m0. The rule is by NAME so it is reproducible
    from the golden's `weights_meta` alone."""
    if name.endswith("embed.scale"):
        return 1.0, 1.0, "e8m0"
    if name.endswith("embed.weight") and "engram" in name:
        return 0.5, 0.0, "fp8"
    if name == "embed.weight":
        return 0.5, 0.0, "f32"
    if name.endswith("attn_sink"):
        return 0.1, 0.0, "f32"
    if name.endswith(("hc_attn_scale", "hc_ffn_scale")):
        return 0.05, 1.0, "f32"
    if name.endswith(("hc_attn_base", "hc_ffn_base")):
        return 0.1, 0.0, "f32"
    if name.endswith(("q_weight", "k_weight")) and "engram" in name:
        return 0.05, 1.0, "f32"
    if name.endswith("gate.bias") or name.endswith("gate.bias_vl"):
        return 0.05, 0.0, "f32"
    if "norm" in name.split(".")[-2:][0] or name.endswith("norm.weight"):
        return 0.05, 1.0, "f32"
    if p.dim() >= 2:
        return float(p.shape[-1]) ** -0.5, 0.0, "f32"
    return 0.05, 0.0, "f32"


def init_all(model: torch.nn.Module) -> dict:
    meta = {}
    for name, p in model.named_parameters():
        scale, offset, kind = rule(name, p)
        n = p.numel()
        v = fixed(name, n, scale, offset).view(p.shape)
        with torch.no_grad():
            if kind == "fp8":
                p.copy_(v.to(torch.float8_e4m3fn))
            elif kind == "e8m0":
                p.copy_(v.to(torch.float8_e8m0fnu))
            else:
                p.copy_(v.to(p.dtype))
        meta[name] = dict(shape=list(p.shape), n=n, scale=scale, offset=offset, kind=kind,
                          dtype=str(p.dtype).removeprefix("torch."), ck=ck(p.detach().float()))
    return meta


# ---------------------------------------------------------------- emit
def ck(t: torch.Tensor) -> float:
    f = t.detach().reshape(-1).to(torch.float64)
    i = torch.arange(f.numel(), dtype=torch.float64)
    return float((f * (i + 1.0)).sum())


def _next_prime(k: int) -> int:
    k = max(2, k)
    while any(k % d == 0 for d in range(2, int(k**0.5) + 1)):
        k += 1
    return k


def stride_for(n: int, is_int: bool) -> int:
    """`ck` covers the whole tensor; the strided `data` is a diagnostic sample capped at 12
    floats (64 for ints, which are exact and cheap). Prime stride, so the sample never aliases a
    row or head period."""
    cap = 64 if is_int else 12
    return 1 if n <= cap else _next_prime(-(-n // cap))


# Captures a sub-module test needs as its INPUT are stored at full resolution (stride 1), so the
# sub-module can be checked before the layers that would otherwise have to produce that input.
FULL_CAPTURES = {
    "L1.engram_in", "L4.engram_in",
    # layer 0 (plain: no engram, no compression): every input the block scaffold, the MoE and
    # the ratio-0 attention need, so each can be tested on its own
    "L0.h_in", "L0.pre_mix_in", "L0.attn_in", "L0.attn_out", "L0.ffn_in", "L0.ffn_out",
}


def emit_tensor(t: torch.Tensor, is_int: bool, name: str = "") -> dict:
    n = t.numel()
    stride = 1 if name in FULL_CAPTURES else stride_for(n, is_int)
    flat = t.detach().reshape(-1)
    if is_int:
        data = [int(x) for x in flat.tolist()[::stride]]
        c = ck(flat.to(torch.float64))
    else:
        # full round-trip precision: json.dump writes repr(), so an exact bf16 value like
        # -0.002197265625 survives, where 9 significant figures would not
        data = [float(x) for x in flat.float().tolist()[::stride]]
        c = ck(flat.float())
    return dict(shape=list(t.shape), n=n, stride=stride, ck=c, data=data)


# ---------------------------------------------------------------- capture
REC: dict[str, dict[str, tuple[torch.Tensor, bool]]] = {}
CUR = dict(regime=None, layer=None, mix_call=0)


def rec(name: str, t: torch.Tensor, is_int: bool = False):
    REC[CUR["regime"]][name] = (t.detach().clone().cpu(), is_int)


def L(name: str) -> str:
    return f"L{CUR['layer']}.{name}"


def patch(cls, attr, wrapper):
    orig = getattr(cls, attr)
    setattr(cls, attr, functools.wraps(orig)(lambda *a, **k: wrapper(orig, *a, **k)))


def install_hooks():
    def block_forward(orig, self, x, start_pos, pre_mix, image_mask, *attn_args):
        CUR["layer"], CUR["mix_call"] = self.layer_id, 0
        rec(L("h_in"), x)
        rec(L("pre_mix_in"), pre_mix)
        out, ffn_pre = orig(self, x, start_pos, pre_mix, image_mask, *attn_args)
        rec(L("h_out"), out)
        rec(L("pre_mix_out"), ffn_pre)
        # the shared slots only change after a source layer writes them; caches are recorded
        # over their USED prefix so an off-by-one cannot hide behind zero-filled tail rows
        if self.layer_id in SOURCE_LAYERS:
            sa = M.shared_attn
            seqlen = x.size(1)
            used = (start_pos + seqlen) // FIX["compress_ratios"][self.layer_id]
            for k in ("compress_kv", "index_k", "topk_idxs", "candidates"):
                v = getattr(sa, k)
                if v is None:
                    continue
                if k in ("compress_kv", "index_k"):
                    v = v[:, :used]
                rec(f"shared.L{self.layer_id}.{k}", v.to(torch.int32) if v.dtype == torch.bool else v,
                    is_int=v.dtype in (torch.bool, torch.int32, torch.int64))
        return out, ffn_pre

    def hc_mixes(orig, self, x, hc_fn, hc_scale, hc_base):
        pre, post, comb = orig(self, x, hc_fn, hc_scale, hc_base)
        site = "attn" if CUR["mix_call"] == 0 else "ffn"
        CUR["mix_call"] += 1
        rec(L(f"{site}_pre"), pre)
        rec(L(f"{site}_post"), post)
        rec(L(f"{site}_comb"), comb)
        return pre, post, comb

    def attn_forward(orig, self, x, start_pos):
        rec(L("attn_in"), x)
        y = orig(self, x, start_pos)
        rec(L("attn_out"), y)
        return y

    def moe_forward(orig, self, x, image_mask=None):
        rec(L("ffn_in"), x)
        y = orig(self, x, image_mask)
        rec(L("ffn_out"), y)
        return y

    def gate_forward(orig, self, x, image_mask=None):
        w, idx = orig(self, x, image_mask)
        rec(L("moe_weights"), w)
        rec(L("moe_indices"), idx, is_int=True)
        return w, idx

    def engram_forward(orig, self, x, hash_ids, token_mask=None):
        # Transformer.forward runs `layer.engram(h)` BEFORE `layer(h)`, so CUR["layer"] still
        # names the previous block here; label by the engram's own layer_id instead.
        lid = self.layer_id
        rec(f"L{lid}.engram_in", x)
        rec(f"L{lid}.engram_hash_ids", hash_ids, is_int=True)
        y = orig(self, x, hash_ids, token_mask)
        rec(f"L{lid}.engram_out", y)
        return y

    def sparse_attn(orig, q, kv, attn_sink, topk_idxs, softmax_scale):
        rec(L("sa_q"), q)
        rec(L("sa_kv"), kv)
        rec(L("sa_topk_idxs"), topk_idxs, is_int=True)
        o = orig(q, kv, attn_sink, topk_idxs, softmax_scale)
        rec(L("sa_o"), o)
        return o

    def embed_forward(orig, self, x):
        y = orig(self, x)
        rec("embed", y)
        return y

    def hash_forward(orig, self, input_ids, start_pos, token_mask=None):
        h = orig(self, input_ids, start_pos, token_mask)
        rec("engram_hashes", h, is_int=True)
        return h

    def head_forward(orig, self, x, full_logits=False):
        rec("head_in", x)
        rec("logits_full", orig(self, x, True))
        return orig(self, x, full_logits)

    def tf_forward(orig, self, input_ids, start_pos=0, images=None, token_types=None):
        rec("input_ids", input_ids, is_int=True)
        return orig(self, input_ids, start_pos, images, token_types)

    patch(M.Block, "forward", block_forward)
    patch(M.Block, "hc_mixes", hc_mixes)
    patch(M.Attention, "forward", attn_forward)
    patch(M.MoE, "forward", moe_forward)
    patch(M.Gate, "forward", gate_forward)
    patch(M.Engram, "forward", engram_forward)
    patch(M.ParallelEmbedding, "forward", embed_forward)
    patch(M.ParallelHead, "forward", head_forward)
    patch(M.Transformer, "forward", tf_forward)
    patch(engram_mod.NgramHashState, "forward", hash_forward)
    M.sparse_attn = functools.wraps(ds41_ref_shims.sparse_attn)(
        lambda *a: sparse_attn(ds41_ref_shims.sparse_attn, *a))


# ---------------------------------------------------------------- main
def main():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cpu")
    # the tiny vocab has no real tokenizer: identity compressed map of the same size
    V = FIX["vocab_size"]
    engram_mod.build_compressed_token_map = lambda tok: (list(range(V)), V)

    args = M.ModelArgs(**FIX)
    model = M.Transformer(args, tokenizer=None)
    model.eval()
    rows = [sum(p for per_ngram in layer for p in per_ngram) for layer in model.engram_layout.primes]
    assert rows == list(FIX["engram_num_embeddings"]), (rows, FIX["engram_num_embeddings"])
    weights_meta = init_all(model)
    # the final-norm input is the collapsed stream; hook the instance
    model.norm.register_forward_hook(lambda m, i, o: rec("h_final", i[0]))
    # mid-chain engram captures so a mismatch localises to hash / table / projection / gate
    for layer in model.layers:
        if layer.engram is None:
            continue
        lid = layer.layer_id
        layer.engram.embed.register_forward_hook(
            lambda m, i, o, lid=lid: rec(f"L{lid}.engram_embed", o))
        layer.engram.wkv.register_forward_hook(
            lambda m, i, o, lid=lid: rec(f"L{lid}.engram_kv", o))
    install_hooks()

    total = PREFILL + 2
    ids = fixed_ints("input_ids", B * total, V).view(B, total)

    regimes = []
    pf = f"prefill{PREFILL}"
    CUR["regime"] = pf; REC[pf] = {}
    model(ids[:, :PREFILL], 0); regimes.append(pf)
    for pos in (PREFILL, PREFILL + 1):
        name = f"decode{pos}"
        CUR["regime"] = name; REC[name] = {}
        model(ids[:, pos : pos + 1], pos); regimes.append(name)

    out = {
        "fixture": {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in FIX.items()},
            "batch": B, "prefill_len": PREFILL, "regimes": regimes,
            "reference": "deepseek-ai/DeepSeek-V4.1-Flash inference/model.py @ dba1be0a40aa45a94ad051997016db3960a90277",
            "kernels": "ds41_ref_shims.py (pure torch, from kernel.py tilelang source)",
            "torch": torch.__version__,
            "filler": "salt=fnv1a64(name); z=splitmix64(salt ^ (i*0x9E3779B97F4A7C15)); u=(z>>40)/2^24; v=f32((2u-1)*scale+offset)",
            "input_ids_rule": "fixed_ints('input_ids', batch*(prefill_len+2), vocab_size)[b*(prefill_len+2)+t]",
            "lcg_probe": [splitmix64(fnv1a64("probe") ^ ((i * GOLD) & MASK64)) for i in range(8)],
            # The hash tables are register_buffers (not parameters), and in production they are
            # READ from the GGUF, never recomputed. Shipped here so the Rust engram consumes given
            # tables exactly as the loader will; recomputing `multipliers` would mean reproducing
            # numpy's PCG64 bounded draw, a dependency the real path never has.
            "engram": {
                "layer_ids": list(model.engram_layout.layer_ids),
                "max_ngram_size": model.engram_layout.max_ngram_size,
                "n_heads": model.engram_layout.n_heads,
                "head_dim": model.engram_layout.head_dim,
                "n_hash_cols": (model.engram_layout.max_ngram_size - 1) * model.engram_layout.n_heads,
                "num_embeddings": list(model.engram_layout.num_embeddings),
                "pad_id_compressed": int(model.engram_hash.pad_id),
                "token_map": model.engram_hash.token_map.tolist(),
                "multipliers": model.engram_hash.multipliers.tolist(),   # [layers][ngram]
                "primes": model.engram_hash.primes.tolist(),             # [layers][ngram-1][heads]
                "offsets": model.engram_hash.offsets.tolist(),           # [layers][(ngram-1)*heads]
                # per-block e8m0 dequant scales of each fp8 table, as floats (all powers of two).
                # Shipped because production dequantises rows from Q2_K, a different format: the
                # Rust reference's contract is "dequantised rows in", and the e8m0 rounding rule is
                # pinned by its own test rather than assumed.
                "scale_block": 32,
                "scales": [layer.engram.embed.scale.float().reshape(-1).tolist()
                           for layer in model.layers if layer.engram is not None],
            },
        },
        "weights_meta": weights_meta,
    }
    for r in regimes:
        out[r] = {k: emit_tensor(t, is_int, k) for k, (t, is_int) in REC[r].items()}

    path = os.path.join(HERE, "ds41_tiny_golden.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    n_caps = sum(len(out[r]) for r in regimes)
    print(f"wrote {path}: {os.path.getsize(path)/1024:.1f} KiB, {len(weights_meta)} params, "
          f"{n_caps} captures across {regimes}")
    lg = REC[pf]["logits_full"][0]
    print(f"prefill logits_full shape {list(lg.shape)}  finite={bool(torch.isfinite(lg).all())}  "
          f"argmax(last)={int(lg[0,-1].argmax())}")


if __name__ == "__main__":
    main()
