#!/usr/bin/env python3
"""Phantom-mangling marker scanner + normaliser, for the prefix-cache A/B.

A MARKER is the model asserting a difference between two strings that are
BYTE-IDENTICAL in the transcript, or narrating a corruption/rename that did not
occur.  Byte-identity is verified programmatically: every structural pattern
captures BOTH operands and the hit is emitted only when
`left == right` (or NFKC(left) == NFKC(right), to catch a genuine homoglyph
claim being made about characters that normalise together).

Scope: ONLY the model's own generated prose -- the `[reasoning]` and `[text]`
sections of the trajectory.  Tool results (`[result ...]`) are the environment
talking and `[call ...]` payloads are JSON-escaped source, and including either
would let a large tool dump move the denominator without moving the numerator.
That same assistant-prose byte count is the normaliser, so numerator and
denominator are drawn from exactly the same text.

Tiers:
  STRICT  -- "A instead of B", "A not B", "A rather than B", "A became B",
             "A should be B", with A == B.  These phrasings only make sense as
             a difference claim, so a byte-identical pair is a marker.
  ARROW   -- "A -> B" / "A => B" / "A -> B" with A == B.  Reported separately
             because Rust signatures ("fn f(x: u16) -> u16") produce genuine
             false positives; lines that look like a signature or a use-path
             are dropped up front and the residue is hand-audited.
  NARRATE -- corruption / mangling / "the environment is renaming" narration.
             No operand pair to verify, so this tier is hand-audited in full.

Usage: scan.py <tag> <trajectory.txt>...
"""
import re
import sys
import json
import unicodedata

ANSI = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

# ---------------------------------------------------------------- operand
# A term is a backticked span, a double-quoted span, or a bare identifier run.
BARE = r'[A-Za-z0-9_.:!\[\]#]{2,60}'


def term(g):
    return ('(?:`(?P<' + g + 'b>[^`\n]{1,60})`'
            r'|"(?P<' + g + 'd>[^"\n]{1,60})"'
            "|'(?P<" + g + "s>[^'\n]{1,60})'"
            '|(?P<' + g + 'p>' + BARE + '))')


def pair(mid):
    return re.compile(term('a') + r'\s*(?:' + mid + r')\s*' + term('b'))


def operands(m):
    left = m.group('ab') or m.group('ad') or m.group('as') or m.group('ap')
    right = m.group('bb') or m.group('bd') or m.group('bs') or m.group('bp')
    return left, right


STRICT = [
    ('INSTEAD', pair(r'instead of')),
    ('NOT',     pair(r',?\s*not')),
    ('RATHER',  pair(r'rather than')),
    ('BECAME',  pair(r'became|was changed to|turned into|'
                     r'(?:got |was |been )?renamed to|is being (?:rendered|written) as|'
                     r'came out as|was replaced (?:by|with)')),
    ('SHOULDBE', pair(r'should (?:be|have been|say|read)')),
    ('VS',      pair(r'vs\.?|versus')),
]
ARROW = [('ARROW', pair(r'→|->|=>|⟶'))]

# Lines where "->" is Rust syntax, not a difference claim.
SIGNATURE = re.compile(r'\bfn\s|\bimpl\b|\bwhere\b|\bReturns?\b.*->|^\s*(?://|#)|'
                       r'\bmatch\b|=>\s*\{|\|_\||\bclosure\b')

NARRATE = re.compile(
    r'mangl|corrupt|garbl|byte-identical|looks? identical|identical but|'
    r'environment (?:seems|is|appears|might be|may be)|'
    r'(?:seems|appears) to be (?:mangling|renaming|corrupting|mapping|substituting)|'
    r'silently (?:changed|replaced|rewrote|renamed)|'
    r'somehow (?:changed|different|replaced|became)|'
    r'homoglyph|zero-width|invisible char|'
    r'(?:mapping|renaming) crate names|'
    r'rendering in my messages|display(?:ed)? (?:wrong|incorrectly)|'
    r'not what I (?:wrote|typed)|different from what I wrote|'
    r'keep(?:s)? (?:writing|typing) .{0,40}instead',
    re.I)


def nfkc(s):
    return unicodedata.normalize('NFKC', s)


def parse(path):
    """-> (assistant_spans, all_lines).  A span is (start_line, [lines])."""
    lines = [ANSI.sub('', l).rstrip('\n')
             for l in open(path, encoding='utf-8', errors='replace')]
    spans, cur, start = [], None, 0
    for i, ln in enumerate(lines, 1):
        m = re.match(r'^\[(reasoning|text|result [^\]]*|call [^\]]*|finish|prompt)\]', ln)
        if m:
            if cur is not None:
                spans.append((start, cur))
                cur = None
            kind = m.group(1)
            if kind in ('reasoning', 'text'):
                cur, start = [], i + 1
            continue
        if re.match(r'^── turn \d+ ─', ln):
            if cur is not None:
                spans.append((start, cur))
                cur = None
            continue
        if cur is not None:
            cur.append(ln)
    if cur is not None:
        spans.append((start, cur))
    return spans, lines


def scan(path):
    spans, _ = parse(path)
    prose_bytes = 0
    hits = []
    for start, body in spans:
        for off, line in enumerate(body):
            prose_bytes += len(line.encode('utf-8')) + 1
            lineno = start + off
            for tier, table in (('STRICT', STRICT), ('ARROW', ARROW)):
                for name, rx in table:
                    for m in rx.finditer(line):
                        a, b = operands(m)
                        if a is None or b is None:
                            continue
                        if not (a == b or nfkc(a) == nfkc(b)):
                            continue
                        if tier == 'ARROW' and SIGNATURE.search(line):
                            continue
                        hits.append(dict(tier=tier, pat=name, file=path,
                                         line=lineno, a=a, b=b,
                                         text=line.strip()[:400]))
            if NARRATE.search(line):
                hits.append(dict(tier='NARRATE', pat='NARRATE', file=path,
                                 line=lineno, a=None, b=None,
                                 text=line.strip()[:400]))
    # de-duplicate: one marker per (file, line, tier)
    seen, dedup = set(), []
    for h in hits:
        k = (h['file'], h['line'], h['tier'])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(h)
    return prose_bytes, dedup


def main():
    tag = sys.argv[1]
    total_bytes, all_hits, per_file = 0, [], {}
    for p in sys.argv[2:]:
        b, h = scan(p)
        total_bytes += b
        all_hits += h
        per_file[p] = (b, len(h))
    print(f"### ARM {tag}")
    for h in sorted(all_hits, key=lambda x: (x['file'], x['line'])):
        op = f" [{h['a']!r} == {h['b']!r}]" if h['a'] is not None else ""
        print(f"{h['file']}:{h['line']}: {h['tier']}/{h['pat']}{op}: {h['text']}")
    print("\n--- per transcript (assistant prose bytes, markers) ---")
    for p, (b, n) in sorted(per_file.items()):
        print(f"{b:9d}  {n:4d}  {p}")
    tiers = {}
    for h in all_hits:
        tiers[h['tier']] = tiers.get(h['tier'], 0) + 1
    kb = total_bytes / 1024.0
    print(f"\nassistant prose      : {total_bytes} bytes = {kb:.2f} KB")
    print(f"markers total        : {len(all_hits)}   by tier {tiers}")
    print(f"markers per KB prose : {len(all_hits)/kb:.4f}" if kb else "n/a")
    json.dump(dict(tag=tag, prose_bytes=total_bytes, hits=all_hits,
                   per_file={k: v for k, v in per_file.items()}),
              open(f"/home/ms/.claude/jobs/5a7bd33d/tmp/mangle2/scan.{tag}.json", "w"),
              indent=1)


main()
