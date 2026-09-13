#!/usr/bin/env python3
"""Slice 8 Gate 4 — GLM-5.3-Flash kpool INDEXER golden, HF transformers 5.16.1.

The indexer is proven BEFORE the MLA on purpose: a wrong top-k still produces plausible
attention output, so an MLA test built on a broken selection would pass and poison everything
downstream.

Every equation comes from HF's own `Glm5NextTextIndexer` methods — `get_visible_tokens`,
`get_pooled_states`, `append_visible_tail` — driven either through the real `forward` (prefill)
or through a driver that mirrors it for the decode case (HF's decode path needs the Cache
machinery; the driver reuses the same three methods, so no equation is re-derived here).

Regimes, all at PRODUCTION dimensions (index_topk=2048, index_kpool=4, n_heads=32, head_dim=128):
  short7      S=7    — far below pool/topk capacity: 1 complete pool, tail exercised
  medium64    S=64   — 16 pools, still below select_k
  ragged      S=13 with LEADING padding — the [P,P,A,B,...] case pooling must skip
  longsparse  S=2560 — 640 pools vs select_k=512, so 128 pools are genuinely DROPPED
  decode      q_len=1 over the S=2560 state — must equal longsparse's LAST query row

Weights are the REAL layer tensors; nothing in a DSA block is quantised (audited: all BF16),
so this golden IS the production numerics.
"""
import hashlib, json, struct, sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/w/hf5161_pkg")
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer  # noqa: E402

LAYERS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["3"])]
PACKET = "/w/dsa_layer%d.safetensors"
CONFIG = "/w/config.json"
OUT = "/w/dsa_indexer_golden.json"
STRIDE = 251
STRIDE_BIG = 1009

_ST = {"BF16": (torch.bfloat16, np.uint16), "F32": (torch.float32, np.float32)}


def load_packet(path):
    fh = open(path, "rb")
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    base = 8 + n
    out = {}
    for k, m in hdr.items():
        a, b = m["data_offsets"]
        fh.seek(base + a)
        arr = np.frombuffer(fh.read(b - a), dtype=_ST[m["dtype"]][1]).copy()
        t = torch.from_numpy(arr)
        if m["dtype"] == "BF16":
            t = t.view(torch.bfloat16)
        out[k] = t.reshape(m["shape"])
    return out


class Lcg:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((self.s >> 40) / (1 << 24)) * 2.0 - 1.0

    def t(self, *shape):
        n = 1
        for s in shape:
            n *= s
        return torch.tensor([self.u() for _ in range(n)], dtype=torch.float32).reshape(shape)


raw_cfg = json.load(open(CONFIG))["text_config"]
cfg = Glm5NextTextConfig(**raw_cfg)
HID, NH, HD = cfg.hidden_size, cfg.index_n_heads, cfg.index_head_dim
KPOOL, TOPK, QLORA = cfg.index_kpool, cfg.index_topk, cfg.q_lora_rank
SELECT_K_MAX = TOPK // KPOOL
OUT_WIDTH = TOPK + (KPOOL - 1 if cfg.index_kpool_always_select_tail else 0)
assert (NH, HD, KPOOL, TOPK, QLORA) == (32, 128, 4, 2048, 1536)


def build(dtype, W):
    m = Glm5NextTextIndexer(cfg, 3).to(dtype).eval()
    sd = {
        "wq_b.weight": W["self_attn.indexer.wq_b.weight"],
        "wk.weight": W["self_attn.indexer.wk.weight"],
        "k_norm.weight": W["self_attn.indexer.k_norm.weight"],
        "k_norm.bias": W["self_attn.indexer.k_norm.bias"],
        "weights_proj.weight": W["self_attn.indexer.weights_proj.weight"],
        "index_kpool_compress_ape": W["self_attn.indexer.index_kpool_compress_ape"],
        "index_kpool_compress_gate": W["self_attn.indexer.index_kpool_compress_gate"],
    }
    m.load_state_dict({k: v.to(dtype) for k, v in sd.items()}, strict=True)
    return m


def run_indexer(m, hidden, q_resid, mask, q_len=None):
    """Mirror of `Glm5NextTextIndexer.forward`, stage by stage.

    `q_len=None` -> full prefill (identical to calling forward with no cache).
    `q_len=k`    -> the LAST k queries only, with the full packed state as the KV -- what a
                    decode step sees. Every equation still comes from HF's own methods.
    """
    B, S = hidden.shape[:2]
    st = {}
    q_all = m.wq_b(q_resid).view(B, S, -1, HD)
    k = m.k_norm(m.wk(hidden)).view(B, S, -1, HD).squeeze(2)
    gate_scores = F.linear(hidden, m.index_kpool_compress_gate)
    valid_channel = mask.to(k.dtype)[..., None]
    packed = torch.cat([k, gate_scores, valid_channel], dim=-1)

    st["k_normed"] = k[0].float().clone()
    st["gate_scores"] = gate_scores[0].float().clone()

    q = q_all if q_len is None else q_all[:, -q_len:]
    q_rows = S if q_len is None else q_len
    q_mask = mask if q_len is None else mask[:, -q_len:]

    valid_keys = packed[..., -1].bool()
    visible = m.get_visible_tokens(valid_keys=valid_keys, q_length=q_rows, current_length=S)
    pool_keys, pool_indices, pool_valid = m.get_pooled_states(packed_states=packed)
    st["pool_keys"] = pool_keys[0].float().clone()
    st["pool_indices"] = pool_indices[0].to(torch.int64).clone()
    st["pool_valid"] = pool_valid[0].to(torch.int32).clone()

    scores_raw = torch.matmul(q.float(), pool_keys.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores_raw * m.softmax_scale)
    # Fraction of PER-HEAD (query, head, pool) entries the ReLU actually clamps. Measured here
    # and not on `index_scores`: index_scores is a 32-head weighted SUM, so it is zero only when
    # every head clamped — which would report "ReLU never fires" even when it fires constantly.
    relu_clamped = float(scores_raw.le(0.0).float().mean())
    weights = m.weights_proj(hidden.to(m.weights_proj.weight.dtype)).float() * (NH**-0.5)
    w = weights if q_len is None else weights[:, -q_len:]
    index_scores = torch.matmul(w.unsqueeze(-2), scores).squeeze(-2)

    pool_end = pool_indices[..., -1].clamp(0, S - 1)
    pool_visible = visible.gather(-1, index=pool_end[:, None, :].expand(B, q_rows, -1))
    valid_candidates = pool_visible & pool_valid[:, None]
    index_scores = index_scores.masked_fill(~valid_candidates, torch.finfo(index_scores.dtype).min)
    st["index_scores"] = index_scores[0].clone()
    st["valid_candidates"] = valid_candidates[0].to(torch.int32).clone()

    select_k = min(TOPK // KPOOL, index_scores.shape[-1])
    selected = index_scores.topk(select_k, dim=-1).indices
    st["selected_pools"] = selected[0].to(torch.int64).clone()

    batch_idx = torch.arange(B)[:, None, None]
    selected_valid = valid_candidates.gather(-1, selected)
    selected_indices = pool_indices[batch_idx, selected]
    topk_indices = selected_indices.flatten(-2)
    topk_indices = topk_indices.masked_fill(
        ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1
    )
    width = TOPK
    if m.index_kpool_always_select_tail:
        topk_indices = m.append_visible_tail(topk_indices, visible, valid_keys)
        width += KPOOL - 1
    topk_indices = F.pad(topk_indices, (0, width - topk_indices.shape[-1]), value=-1)[..., :width]
    topk_indices = topk_indices.masked_fill(~q_mask[..., None], -1)
    topk = topk_indices.to(torch.int32)[0]

    st["topk_indices"] = topk.to(torch.int64).clone()
    # 🔴 Order-canonical form. The consumer scatters these into a boolean mask, so ORDER is
    # unobservable downstream — but `topk`'s order among equal scores is implementation-defined
    # AND a 1-ulp score difference reorders adjacent ranks. Comparing raw positions therefore
    # fails on a correct implementation. Sorting each row makes a strided positional comparison
    # a genuine SET comparison.
    st["topk_sorted"] = torch.sort(topk.to(torch.int64), dim=-1).values.clone()
    # Derived facts a kernel must reproduce exactly, checked independently of the raw array.
    tk64 = topk.to(torch.int64)
    valid_tk = tk64.ge(0)
    st_meta = {
        # 🔴 Order-independent, EXACT per-row set digests. A positional comparison of the sorted
        # row amplifies a single pool swap into dozens of shifted positions, so it cannot
        # distinguish "one near-tie resolved differently" from "the selection is wrong". Sum +
        # count over the valid entries is cheap, exact per row, and insensitive to order.
        "row_sum": (tk64 * valid_tk).sum(-1).clone(),
        "row_count": valid_tk.sum(-1).to(torch.int64).clone(),
        "select_k": select_k,
        "n_pools": int(pool_keys.shape[1]),
        "out_width": width,
        "valid_per_row": topk.ge(0).sum(-1).to(torch.int64).clone(),
        "max_index": int(topk.max()),
        "min_index": int(topk.min()),
    }
    # ── Selection ambiguity, measured where it actually matters ────────────────────
    # A tie anywhere in the ranking is harmless; a tie ACROSS THE CUTOFF is not — it makes
    # `topk` order implementation-defined, so the reference's own pool identities stop being
    # a legal target. ReLU clamps every negative score to exactly 0.0, so the tail of the
    # ranking is a large block of exact zeros and the cutoff can land inside it.
    sc = index_scores[0]
    n_valid = valid_candidates[0].sum(-1)
    cutoff_tie, boundary_score_zero = 0, 0
    for r in range(sc.shape[0]):
        row = sc[r][valid_candidates[0, r]]
        if row.numel() <= select_k:
            continue
        srt = torch.sort(row, descending=True).values
        if float(srt[select_k - 1]) == float(srt[select_k]):
            cutoff_tie += 1
            if float(srt[select_k - 1]) == 0.0:
                boundary_score_zero += 1
    st_meta["cutoff_tie_rows"] = cutoff_tie
    st_meta["cutoff_tie_rows_at_zero"] = boundary_score_zero
    st_meta["rows_with_more_pools_than_select_k"] = int((n_valid > select_k).sum())
    st_meta["relu_clamped_fraction"] = relu_clamped
    st_meta["zero_index_score_fraction"] = float(
        (sc.eq(0.0) & valid_candidates[0]).sum() / valid_candidates[0].sum().clamp(min=1)
    )
    st_meta["tie_rows"] = int(
        sum(
            1
            for r in range(sc.shape[0])
            if len(torch.unique(sc[r][valid_candidates[0, r]]))
            != int(valid_candidates[0, r].sum())
        )
    )
    return st, st_meta


def emit(st, meta):
    parts = []
    for name, t in st.items():
        t = t.flatten()
        stride = 1 if t.numel() <= 4096 else (STRIDE if t.numel() <= 4_000_000 else STRIDE_BIG)
        s = t[::stride]
        if t.dtype in (torch.int64, torch.int32):
            data = ",".join(str(int(x)) for x in s.tolist())
            ck = float(sum((i + 1) * int(v) for i, v in enumerate(t.tolist())))
        else:
            data = ",".join(f"{x:.9g}" for x in s.tolist())
            f = t.to(torch.float64)
            ck = float((f * (torch.arange(f.numel(), dtype=torch.float64) + 1.0)).sum())
        parts.append(
            '    "%s":{"n":%d,"stride":%d,"ck":%r,"data":[%s]}' % (name, t.numel(), stride, ck, data)
        )
    for k, v in meta.items():
        if torch.is_tensor(v):
            parts.append(
                '    "%s":{"n":%d,"stride":1,"ck":0,"data":[%s]}'
                % (k, v.numel(), ",".join(str(int(x)) for x in v.flatten().tolist()))
            )
        else:
            parts.append('    "%s":%r' % (k, v))
    return "{\n" + ",\n".join(parts) + "\n   }"


# (name, seq_len, leading_pad, q_len_for_decode, negate_q)
# `relu_probe` exists because the natural fixture produces only positive pool scores, which
# leaves `F.relu(scores * softmax_scale)` as dead code. Negating the query flips roughly half
# the dot products negative, so the ReLU actually clamps — and the resulting block of exact
# zeros is what can put a TIE ACROSS THE SELECTION CUTOFF, making top-k order ambiguous.
REGIMES = [("short7", 7, 0, None, False), ("medium64", 64, 0, None, False),
           ("ragged13", 13, 5, None, False), ("relu_probe", 2560, 0, None, True),
           ("longsparse", 2560, 0, None, False), ("decode", 2560, 0, 1, False)]

blocks, diag = [], []
for LAYER in LAYERS:
    W = load_packet(PACKET % LAYER)
    per_dt = []
    for dt_name, dt in (("f32", torch.float32), ("bf16", torch.bfloat16)):
        mod = build(dt, W)
        for rname, S, pad, q_len, neg_q in REGIMES:
            rng = Lcg(0x0D5A_C0DE)
            hidden = rng.t(1, S, HID) * 0.5
            q_resid = rng.t(1, S, QLORA) * 0.5
            if neg_q:
                q_resid = -q_resid
            mask = torch.ones(1, S, dtype=torch.bool)
            if pad:
                mask[:, :pad] = False  # LEADING padding: pooling must start at the first real token
            with torch.no_grad():
                st, meta = run_indexer(mod, hidden.to(dt), q_resid.to(dt), mask, q_len)
            per_dt.append(f'   "{dt_name}__{rname}":' + emit(st, meta))
            diag.append((LAYER, dt_name, rname, S, meta["n_pools"], meta["select_k"],
                         int(meta["valid_per_row"].max()), meta["tie_rows"],
                         meta["cutoff_tie_rows"], meta["cutoff_tie_rows_at_zero"],
                         meta["relu_clamped_fraction"],
                         meta["zero_index_score_fraction"]))
    blocks.append(f'  "{LAYER}":{{\n' + ",\n".join(per_dt) + "\n  }")

body = ("{\n"
        f' "fixture":{{"hidden":{HID},"index_n_heads":{NH},"index_head_dim":{HD},'
        f'"index_kpool":{KPOOL},"index_topk":{TOPK},"q_lora_rank":{QLORA},'
        f'"select_k_max":{SELECT_K_MAX},"out_width":{OUT_WIDTH},"always_select_tail":true,'
        f'"k_norm_eps":1e-06,"softmax_scale":{HD ** -0.5!r},"seed":"0x0D5AC0DE",'
        f'"checkpoint":"LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3","layers":{LAYERS}}},\n'
        ' "by_layer":{\n' + ",\n".join(blocks) + "\n }\n}\n")
open(OUT, "w").write(body)

print("transformers", __import__("transformers").__version__, "torch", torch.__version__)
print("bytes", len(body))
print("sha256", hashlib.sha256(body.encode()).hexdigest())
print(f"{'L':>3} {'dt':5} {'regime':11} {'S':>5} {'pools':>6} {'sel_k':>6} {'valid':>6} "
      f"{'ties':>5} {'cutTie':>7} {'@zero':>6} {'reluClamp':>10} {'zeroIdxSc':>10}")
for r in diag:
    print("%3d %-5s %-11s %5d %6d %6d %6d %5d %7d %6d %10.4f %10.4f" % r)
