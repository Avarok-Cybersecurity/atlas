# Slice 9 gate 1 — extract the mHC (hyper-connection) parameter packets.
#
# Pulls `hc_{attn,ffn}_{fn,base,scale}` for the requested layers out of the
# 120-shard NVFP4 checkpoint into one standalone safetensors packet per layer.
# Read-only on the checkpoint. Diagnostics to stderr, nothing to stdout.
import json, struct, hashlib, sys, os

S = ("/home/cluster/hf-glm53/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/"
     "snapshots/9e0d74e3cef17f634e84fb8e2223707e02616290")
OUT = sys.argv[1]
LAYERS = [int(x) for x in sys.argv[2].split(",")]
SITES = ("attn", "ffn")
PARTS = ("fn", "base", "scale")

index = json.load(open(S + "/model.safetensors.index.json"))["weight_map"]
os.makedirs(OUT, exist_ok=True)

# One open handle per shard we touch; the hc tensors of a layer are contiguous
# in practice but we do not rely on it.
handles, headers = {}, {}


def shard(name):
    if name not in handles:
        fh = open(S + "/" + name, "rb")
        n = struct.unpack("<Q", fh.read(8))[0]
        headers[name] = (json.loads(fh.read(n)), 8 + n)
        handles[name] = fh
    return handles[name], headers[name]


for layer in LAYERS:
    pfx = "model.language_model.layers.%d." % layer
    blobs, out_hdr, off = [], {}, 0
    for site in SITES:
        for part in PARTS:
            key = pfx + "hc_%s_%s" % (site, part)
            if key not in index:
                raise SystemExit("MISSING %s — refusing to write a partial packet" % key)
            fh, (hdr, base) = shard(index[key])
            m = hdr[key]
            a, b = m["data_offsets"]
            fh.seek(base + a)
            raw = fh.read(b - a)
            assert len(raw) == b - a
            short = key[len(pfx):]
            out_hdr[short] = {"dtype": m["dtype"], "shape": m["shape"],
                              "data_offsets": [off, off + len(raw)]}
            off += len(raw)
            blobs.append(raw)
            print("  L%-2d %-14s %-5s %-14s %8d B  sha256 %s"
                  % (layer, short, m["dtype"], str(m["shape"]), len(raw),
                     hashlib.sha256(raw).hexdigest()[:32]), file=sys.stderr)
    out_hdr["__metadata__"] = {
        "source_repo": "LibertAIDAI/GLM-5.3-Flash-NVFP4",
        "source_revision": "9e0d74e3cef17f634e84fb8e2223707e02616290",
        "layer": str(layer), "prefix": pfx,
    }
    hj = json.dumps(out_hdr).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    path = "%s/mhc_layer%d.safetensors" % (OUT, layer)
    with open(path, "wb") as w:
        w.write(struct.pack("<Q", len(hj)))
        w.write(hj)
        for b_ in blobs:
            w.write(b_)
    print("  -> %s  (%d B payload)" % (path, off), file=sys.stderr)
