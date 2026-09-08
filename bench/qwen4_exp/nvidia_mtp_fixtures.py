# SPDX-License-Identifier: AGPL-3.0-only
"""Extract bounded GPU test fixtures from a local NVIDIA Qwen3.8 snapshot."""
import argparse
import json
from pathlib import Path
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('snapshot', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
index = json.loads((args.snapshot / 'model.safetensors.index.json').read_text())['weight_map']


def tensor(name, row=None):
    with (args.snapshot / index[name]).open('rb') as source:
        header_size = struct.unpack('<Q', source.read(8))[0]
        meta = json.loads(source.read(header_size))[name]
        begin, end = meta['data_offsets']
        if row is None:
            offset, size = 0, end - begin
        else:
            assert meta['dtype'] == 'F8_E4M3' and meta['shape'][1] == 160
            assert 0 <= row < meta['shape'][0]
            offset, size = row * 160, 160
        source.seek(8 + header_size + begin + offset)
        data = source.read(size)
        assert len(data) == size
        return meta, data


prefix = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.'
meta, raw_scale = tensor(prefix + 'weight_scale')
assert meta['dtype'] == 'BF16' and meta['shape'] == [1]
scale = struct.unpack('<f', b'\0\0' + raw_scale)[0]
assert scale > 0


def expected_bf16(byte):
    """Independent finite E4M3 -> FP32 multiplication -> BF16 ties-to-even."""
    assert byte & 127 != 127, 'Fixture contains an FP8 NaN'
    sign = -1 if byte & 128 else 1
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    value = mantissa * 2 ** -9 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)
    bits = struct.unpack('<I', struct.pack('<f', sign * value * scale))[0]
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 65535


rows = []
for shard in [0, 1, 63, 127]:
    for row in [0, 1, 123456, 2500011]:
        _, data = tensor(prefix + f'shard_{shard}.weight', row)
        rows.append(dict(shard=shard, row=row, bytes=list(data),
                         expected_bf16=list(map(expected_bf16, data))))
fixture = dict(checkpoint=str(args.snapshot), scale=scale, head_dim=160, rows=rows)
(args.output / 'ple-fp8-checkpoint.json').write_text(json.dumps(fixture))
meta, weight = tensor('model.language_model.layers.0.linear_attn.out_proj.weight')
assert meta['dtype'] == 'BF16' and meta['shape'] == [2560, 6144]
(args.output / 'gdn-out-weight.bin').write_bytes(weight)
(args.output / 'gdn-out-weight.json').write_text(json.dumps(meta))
print(f'Wrote PLE and GDN fixtures to {args.output}')
