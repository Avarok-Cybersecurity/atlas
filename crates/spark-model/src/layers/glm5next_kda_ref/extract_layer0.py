# Extract the layer-0 KDA packet from shard 1/120 into a standalone safetensors file.
# Binary goes to stdout; all diagnostics to stderr. Read-only on the checkpoint.
import json, struct, hashlib, sys
S = "/home/cluster/hf-glm53/hub/models--LibertAIDAI--GLM-5.3-Flash-NVFP4/snapshots/9e0d74e3cef17f634e84fb8e2223707e02616290"
f = S + "/model-00001-of-00120.safetensors"
fh = open(f, "rb")
n = struct.unpack("<Q", fh.read(8))[0]
hdr = json.loads(fh.read(n))
base = 8 + n
want = sorted(k for k in hdr if ".layers.0.self_attn." in k)
want.append("model.language_model.layers.0.input_layernorm.weight")

blobs, out_hdr, off = [], {}, 0
for k in want:
    m = hdr[k]
    a, b = m["data_offsets"]
    fh.seek(base + a)
    raw = fh.read(b - a)
    assert len(raw) == b - a
    short = k.split("layers.0.")[1]
    out_hdr[short] = {"dtype": m["dtype"], "shape": m["shape"],
                      "data_offsets": [off, off + len(raw)]}
    off += len(raw)
    blobs.append(raw)
    print("  %-34s %-6s %-14s %9d B  sha256 %s"
          % (short, m["dtype"], str(m["shape"]), len(raw),
             hashlib.sha256(raw).hexdigest()[:32]), file=sys.stderr)
out_hdr["__metadata__"] = {
    "source_repo": "LibertAIDAI/GLM-5.3-Flash-NVFP4",
    "source_revision": "9e0d74e3cef17f634e84fb8e2223707e02616290",
    "source_shard": "model-00001-of-00120.safetensors",
    "layer": "0", "prefix": "model.language_model.layers.0.",
}
hj = json.dumps(out_hdr).encode()
hj += b" " * ((8 - len(hj) % 8) % 8)
w = sys.stdout.buffer
w.write(struct.pack("<Q", len(hj))); w.write(hj)
for b_ in blobs:
    w.write(b_)
w.flush()
print("  TOTAL packet bytes: %d (%.1f MiB)" % (off, off / 1048576), file=sys.stderr)
