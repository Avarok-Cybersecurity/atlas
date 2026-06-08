#!/usr/bin/env python3
"""Scan Qwen3.6 tokenizer for all whitespace-bearing tokens.

Produces three groups:
  Group 1: pure-whitespace tokens (only spaces/tabs/newlines/CR)
  Group 2: short (len<=4) leading-whitespace tokens with a NON-ALPHA tail
           (punctuation / digit / bracket / quote — the kinds that pair
           with `.trim()` to defeat the param-body guard).
  Group 3: digit-with-leading-space tokens (` 0`..` 9`) — turns out Qwen3.6
           has NONE; documented for completeness so callers know to also
           mask token 220 (the bare space) before a digit-continuation.

Output: qwen36_whitespace_tokens.json
"""
import json
from tokenizers import Tokenizer

TK_PATH = '/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0/tokenizer.json'
OUT_PATH = '/workspace/atlas-mtp/bench/fp8_dgx2_drift/qwen36_whitespace_tokens.json'

WS_CHARS = set(' \t\n\r\x0b\x0c   ')

def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def main():
    tk = Tokenizer.from_file(TK_PATH)
    vocab_size = tk.get_vocab_size(with_added_tokens=True)
    vocab = tk.get_vocab(with_added_tokens=True)
    id_to_tok = {v: k for k, v in vocab.items()}

    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    def bytelevel_decode(s):
        try:
            byts = bytes([u2b[c] for c in s])
            return byts.decode('utf-8', errors='replace')
        except KeyError:
            return None  # likely a special token, skip

    group1 = []
    group2 = []
    group3 = []

    for tid in range(vocab_size):
        tok_str = id_to_tok.get(tid)
        if tok_str is None:
            continue
        decoded = bytelevel_decode(tok_str)
        if decoded is None or decoded == '':
            continue

        if all(c in WS_CHARS for c in decoded):
            group1.append({'id': tid, 'decoded': decoded, 'repr': repr(decoded), 'category': 'pure_ws'})
            continue

        # Short leading-ws (len <= 4) with non-ws tail.
        if len(decoded) <= 4 and decoded[0] in WS_CHARS:
            tail = decoded.lstrip()
            if tail and not tail[0].isalpha():
                # Non-alphabetic tail: punctuation / digit / bracket / quote.
                # These are the dangerous ones for tool-body `.trim()` escapes.
                group2.append({
                    'id': tid,
                    'decoded': decoded,
                    'repr': repr(decoded),
                    'tail': tail,
                    'tail_class': 'digit' if tail[0].isdigit() else 'punct',
                    'category': 'short_leading_ws_nonalpha',
                })
                if decoded.startswith(' ') and len(tail) == 1 and tail.isdigit():
                    group3.append({
                        'id': tid, 'decoded': decoded, 'repr': repr(decoded),
                        'digit': tail, 'category': 'space_digit',
                    })

    out = {
        'tokenizer_path': TK_PATH,
        'vocab_size': vocab_size,
        'counts': {
            'pure_ws': len(group1),
            'short_leading_ws_nonalpha': len(group2),
            'space_digit': len(group3),
        },
        'group1_pure_ws': sorted(group1, key=lambda x: x['id']),
        'group2_short_leading_ws_nonalpha': sorted(group2, key=lambda x: x['id']),
        'group3_space_digit': sorted(group3, key=lambda x: x['id']),
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"vocab_size                = {vocab_size}")
    print(f"pure_ws                   = {len(group1)}")
    print(f"short_leading_ws_nonalpha = {len(group2)}")
    print(f"space_digit               = {len(group3)}")
    print(f"Wrote: {OUT_PATH}")

    print("\nFirst 40 pure_ws:")
    for e in group1[:40]:
        print(f"  {e['id']:6d}  {e['repr']}")
    print(f"\nAll {len(group2)} short_leading_ws_nonalpha:")
    for e in group2:
        print(f"  {e['id']:6d}  {e['repr']}  (tail={e['tail_class']})")
    print(f"\nspace_digit (count={len(group3)}):")
    for e in group3:
        print(f"  {e['id']:6d}  {e['repr']}")

if __name__ == '__main__':
    main()
