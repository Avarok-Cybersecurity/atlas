"""Slice 11 gate 1 — COMPLETE GLM-5.3 checkpoint accounting, from the real index.

Every tensor classified, exact bytes per family, nothing inferred from a formula.
"""
import json, re, struct, collections, sys
S = ("/home/cluster/hf-glm53/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/"
     "snapshots/9e0d74e3cef17f634e84fb8e2223707e02616290")
idx = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(S + "/config.json"))
tc = cfg["text_config"]
NL, NMTP = tc["num_hidden_layers"], tc["num_nextn_predict_layers"]
DENSE = set(range(tc["first_k_dense_replace"]))
KDA = set(tc["linear_attn_config"]["kda_layers"])

_h = {}
def meta(k):
    sh = idx[k]
    if sh not in _h:
        fh = open(S + "/" + sh, "rb"); n = struct.unpack("<Q", fh.read(8))[0]
        _h[sh] = json.loads(fh.read(n))
    m = _h[sh][k]
    a, b = m["data_offsets"]
    return m["dtype"], tuple(m["shape"]), b - a

def fam(k):
    if k == "lm_head.weight": return "lm_head"
    if k == "model.language_model.embed_tokens.weight": return "embeddings"
    if k == "model.language_model.norm.weight": return "final_norm"
    if k.startswith("model.visual.") or k.startswith("model.vision"): return "vision(NOT LOADED)"
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)$", k)
    if not m: return "UNKNOWN"
    L, r = int(m.group(1)), m.group(2)
    mtp = L >= NL
    if r in ("eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight"):
        return "mtp_head(NOT LOADED)"
    if r.startswith("hc_"):
        return "mHC"
    if r in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        return "layer_norms(MTP,NOT LOADED)" if mtp else "layer_norms"
    if r.startswith("self_attn."):
        if mtp: return "attn_DSA(MTP,NOT LOADED)"
        return "attn_KDA" if L in KDA else "attn_DSA"
    if r.startswith("mlp."):
        sfx = "(MTP,NOT LOADED)" if mtp else ""
        if r.startswith("mlp.experts."):
            return "moe_experts_NVFP4" + sfx
        if r.startswith("mlp.shared_experts."): return "moe_shared_BF16" + sfx
        if r.startswith("mlp.gate."): return "moe_router" + sfx
        return "dense_ffn" + sfx
    return "UNKNOWN"

cnt = collections.Counter(); byt = collections.Counter()
dt = collections.defaultdict(collections.Counter); unknown = []
for k in idx:
    f = fam(k)
    if f == "UNKNOWN": unknown.append(k); continue
    d, sh, n = meta(k)
    cnt[f] += 1; byt[f] += n; dt[f][d] += 1

print("total tensors in checkpoint:", len(idx))
print("UNKNOWN:", len(unknown), unknown[:5])
print()
print("%-34s %8s %14s %10s  dtypes" % ("family", "tensors", "bytes", "GiB"))
tot_t = tot_b = 0
text_b = 0
for f in sorted(cnt, key=lambda x: -byt[x]):
    print("%-34s %8d %14d %10.3f  %s"
          % (f, cnt[f], byt[f], byt[f] / 2**30, dict(dt[f])))
    tot_t += cnt[f]; tot_b += byt[f]
    if "NOT LOADED" not in f: text_b += byt[f]
print("%-34s %8d %14d %10.3f" % ("TOTAL", tot_t, tot_b, tot_b / 2**30))
print("%-34s %8s %14d %10.3f" % ("TEXT MODEL (loaded)", "", text_b, text_b / 2**30))
assert tot_t == len(idx), (tot_t, len(idx))

# EP=2 split: routed experts shard, everything else replicates.
exp_b = byt["moe_experts_NVFP4"]
rep_b = text_b - exp_b
print()
print("EP=2 per-rank (routed experts sharded 144/144, everything else replicated):")
print("  routed experts / 2 : %10.3f GiB" % (exp_b / 2 / 2**30))
print("  replicated         : %10.3f GiB" % (rep_b / 2**30))
print("  PER-RANK TOTAL     : %10.3f GiB" % ((exp_b / 2 + rep_b) / 2**30))
print("  single-node total  : %10.3f GiB   (EP=1)" % (text_b / 2**30))
