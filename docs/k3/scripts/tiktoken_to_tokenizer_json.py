#!/usr/bin/env python3
from transformers.convert_slow_tokenizer import TikTokenConverter

pat = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)
c = TikTokenConverter(
    vocab_file="/home/pidtom/k3-lab/refs/Kimi-K3-0.40B/tiktoken.model",
    pattern=pat,
)
print("converting", flush=True)
tok = c.converted()
print("n_vocab", tok.get_vocab_size(), flush=True)
out = "/home/pidtom/k3-lab/refs/Kimi-K3-0.40B/tokenizer.json"
tok.save(out)
print("saved", out, flush=True)
ids = tok.encode("According to all known laws of aviation,").ids
print("aviation", ids, flush=True)
want = [18805, 308, 799, 5624, 12524, 318, 57195, 11]
print("match", ids == want, flush=True)
