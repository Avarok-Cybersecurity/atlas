# SPDX-License-Identifier: AGPL-3.0-only
"""Build a bounded synthetic MXFP4 expert fixture from the FP32 K3 twin; not a production quantizer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def pack(weight):
    import numpy as np
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % 32 or not np.isfinite(w).all():
        raise ValueError('expert matrix must be finite rank-2 with K divisible by 32')
    groups = w.reshape(w.shape[0], -1, 32)
    maxima = np.max(np.abs(groups), axis=-1)
    exponent = np.ceil(np.log2(np.maximum(maxima, np.finfo(np.float32).tiny)/6))
    exponent = np.clip(exponent, -127, 127).astype(np.int32)
    exponent[maxima == 0] = 0
    scale = np.exp2(exponent.astype(np.float32))
    normalized = groups / scale[..., None]
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    codes = np.argmin(np.abs(np.abs(normalized)[..., None] - levels), axis=-1).astype(np.uint8)
    codes |= (np.signbit(normalized).astype(np.uint8) << 3)
    codes = codes.reshape(w.shape)
    return (codes[:, ::2] | (codes[:, 1::2] << 4)), (exponent + 127).astype(np.uint8)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy as np
    from safetensors.numpy import load_file, save_file
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-source-bytes', type=int, required=True)
    args = p.parse_args()
    source = args.source/'model.safetensors'
    if args.max_source_bytes <= 0 or source.stat().st_size > args.max_source_bytes:
        p.error('source exceeds explicit small-fixture size limit')
    args.output.mkdir(parents=True, exist_ok=False)
    tensors = load_file(source)
    count = 0
    for name in list(tensors):
        if '.block_sparse_moe.experts.' in name and name.endswith(('.w1.weight', '.w2.weight', '.w3.weight')):
            if tensors[name].dtype != np.float32:
                raise ValueError('this fixture builder only accepts FP32 twin experts')
            packed, scale = pack(tensors.pop(name))
            tensors[name.removesuffix('.weight')+'.weight_packed'] = packed
            tensors[name.removesuffix('.weight')+'.weight_scale'] = scale
            count += 1
    if not count:
        raise ValueError('no routed expert matrices found')
    destination = args.output/'model.safetensors'
    save_file(tensors, destination)
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'tiktoken.model'):
        if (args.source/name).is_file():
            shutil.copy2(args.source/name, args.output/name)
    config_path=args.output/'config.json'
    config=json.loads(config_path.read_text())
    text=config.get('text_config',config)
    text['quantization_config']={'quant_method':'compressed-tensors','format':'mxfp4-pack-quantized'}
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    receipt = {'purpose': 'synthetic TP correctness fixture; not official K3 quantization',
               'expert_matrices': count, 'group_size': 32,
               'source_sha256': sha256(source), 'fixture_sha256': sha256(destination)}
    (args.output/'fixture.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
