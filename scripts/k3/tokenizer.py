#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Derive and differential-test K3's byte-BPE tokenizer; never execute model code."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import random


def read_pattern(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pat_str' for t in node.targets):
            value = node.value
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Constant)
                    and value.func.value.value == '|' and value.func.attr == 'join'
                    and len(value.args) == 1 and not value.keywords):
                parts = ast.literal_eval(value.args[0])
                if isinstance(parts, list) and all(isinstance(p, str) for p in parts):
                    return '|'.join(parts)
    raise ValueError('Expected a literal pat_str list joined with |')


def read_ranks(path):
    ranks = {}
    for line in path.read_text().splitlines():
        token, rank = line.split()
        token, rank = base64.b64decode(token, validate=True), int(rank)
        if not token or token in ranks:
            raise ValueError('Empty or duplicate byte token')
        ranks[token] = rank
    if sorted(ranks.values()) != list(range(len(ranks))):
        raise ValueError('Ranks must be unique and contiguous')
    return ranks


def byte_alphabet():
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    result = {b: chr(b) for b in visible}
    for b in range(256):
        if b not in result:
            result[b] = chr(256 + len(result) - len(visible))
    return result


def build(ranks, pattern, config):
    from tokenizers import Tokenizer, Regex, AddedToken, models, pre_tokenizers, decoders
    alphabet = byte_alphabet()
    encode = lambda value: ''.join(alphabet[b] for b in value)
    vocab = {encode(token): rank for token, rank in ranks.items()}
    merges = []
    for token, rank in sorted(ranks.items(), key=lambda item: item[1]):
        splits = [(token[:i], token[i:]) for i in range(1, len(token))
                  if token[:i] in ranks and token[i:] in ranks
                  and ranks[token[:i]] < rank and ranks[token[i:]] < rank]
        splits.sort(key=lambda pair: (ranks[pair[0]], ranks[pair[1]]))
        merges.extend((encode(a), encode(b)) for a, b in splits)
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    specials = {}
    for index in range(len(ranks), len(ranks) + 256):
        entry = config['added_tokens_decoder'].get(str(index), {})
        token = entry.get('content', f'<|reserved_token_{index}|>')
        specials[token] = index
        tokenizer.add_tokens([AddedToken(token, normalized=False, special=entry.get('special', False))])
        if tokenizer.token_to_id(token) != index:
            raise ValueError(f'Special token ID mismatch at {index}')
    return tokenizer, specials


def validate(tokenizer, ranks, pattern, specials, config, oracle_path=None):
    import tiktoken
    reference = tiktoken.Encoding(name='k3-reference', pat_str=pattern,
                                 mergeable_ranks=ranks, special_tokens=specials)
    cases = ['', 'Hello world!', '中文汉字 English 日本語 한글', "I'm CAN'T naïve café e\u0301",
             '123456789012345\n\r\n\t   ', 'def f(x):\n    return {"x": x + 1}\n',
             '👩🏽‍💻🙂🌍', '<|open|>message role="assistant"<|sep|>analysis',
             '\x00\x01\x7f\u2028\u00a0'] + list(specials)
    rng = random.Random(3107)
    alphabet = list(' abcXYZ123\n\t_<>|"éß中漢文🙂') + [chr(i) for i in range(32, 127)]
    cases += [''.join(rng.choices(alphabet, k=rng.randrange(1, 300))) for _ in range(2000)]
    # Every vocabulary entry proves decoding, including invalid standalone UTF-8 bytes.
    for raw, index in ranks.items():
        if tokenizer.decode([index], skip_special_tokens=False) != reference.decode([index]):
            raise ValueError(f'Decode mismatch at {index}')
    skip_ids = {int(i) for i, entry in config['added_tokens_decoder'].items() if entry['special']}
    oracle = []
    for index, text in enumerate(cases):
        expected = reference.encode(text, allowed_special='all')
        actual = tokenizer.encode(text, add_special_tokens=False).ids
        oracle.append({'text': text, 'ids': expected, 'decoded': reference.decode(expected), 'decoded_skip_special': reference.decode([i for i in expected if i not in skip_ids])})
        if tokenizer.decode(actual, skip_special_tokens=True) != oracle[-1]['decoded_skip_special']:
            raise ValueError(f'Skip-special decoding mismatch at case {index}')
        if actual != expected:
            raise ValueError(f'Encode mismatch at case {index}: {text!r}: {actual} != {expected}')
        if tokenizer.decode(actual, skip_special_tokens=False) != reference.decode(expected):
            raise ValueError(f'Roundtrip mismatch at case {index}')
    if oracle_path is not None:
        oracle_path.write_text(json.dumps(oracle, ensure_ascii=False) + '\n')
    return {'encode_cases': len(cases), 'individual_byte_token_decodes': len(ranks),
            'scope': 'Raw completion text with special markers enabled; not segmented XTML chat'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New derived directory; source remains immutable')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output must not exist')
    names = ['tiktoken.model', 'tokenization_kimi.py', 'tokenizer_config.json']
    sources = {name: args.source / name for name in names}
    ranks = read_ranks(sources['tiktoken.model'])
    if any(bytes([b]) not in ranks for b in range(256)):
        raise ValueError('Missing byte fallback token')
    pattern = read_pattern(sources['tokenization_kimi.py'].read_text())
    config = json.loads(sources['tokenizer_config.json'].read_text())
    tokenizer, specials = build(ranks, pattern, config)
    result = validate(tokenizer, ranks, pattern, specials, config)
    args.output.mkdir(parents=True)
    target = args.output / 'tokenizer.json'
    tokenizer.save(str(target))
    from tokenizers import Tokenizer
    validate(Tokenizer.from_file(str(target)), ranks, pattern, specials, config, args.output / 'tokenizer-oracle.json')
    import tokenizers, tiktoken
    receipt = {'schema': 1, 'sources': {name: hashlib.sha256(path.read_bytes()).hexdigest()
               for name, path in sources.items()}, 'derived_tokenizer_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'versions': {'tokenizers': tokenizers.__version__, 'tiktoken': tiktoken.__version__}, **result}
    (args.output / 'derived-tokenizer.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
