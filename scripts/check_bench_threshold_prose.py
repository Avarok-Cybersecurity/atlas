#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Threshold-prose honesty check: the comment must not contradict the bound.

Every `kernels/{hw}/{model}/BENCH.toml` threshold is annotated with prose that
says what the number is and where it came from. That prose is what a reader
consults to decide whether a measurement is healthy -- and NOTHING made it
track the value beside it. It drifted, in exactly the way you would expect:

    commit c9a682734f, "gate(thresholds): recalibrate ..., TOML-only"
        -min = 27.0
        +min = 21.0
    ... with the six comment lines directly above it untouched, so the file
    went on saying "Floor 27.0 = mean minus ~1 tok/s" while the gate in force
    was 21.0 (effective 20.5 after `noise`). One line changed; the paragraph
    explaining it did not.

This script fails CI when prose that annotates a bound states a number that
contradicts the bound. It is DELIBERATELY NARROW -- see "WHAT THIS CANNOT SEE"
below, which is as much the point as the rules are. A checker that fired on
every number in a paragraph would fire on sigmas, sample counts, superseded
values and other gates' floors, and would be switched off within a week.

RULES. Each fires only on a phrase that ASSERTS THE BOUND ITSELF.

  Scope A -- the inline comment block between a `[benchmarks.metrics.<name>]`
  header and its first key. That block annotates exactly one metric, so a
  number in a bound-assertion position in it is a claim about that metric.

    A1  "Floor <N>" / "the floor of <N>"      -> must equal `min`
    A2  "Ceiling <N>" / "the ceiling of <N>"  -> must equal `max`
        Both fire ONLY when the word is sentence-initial AND capitalised,
        or is directly preceded by the/a/an/this with nothing in between. A
        QUALIFIED floor names a different bound -- "vacuity floor 160", "the
        MLPerf floor 83.64", "the serial floor" -- and is correctly left
        alone. That single restriction is what keeps A1 from crying wolf on
        this corpus; the GREEN self-test fixture pins it.
    A3  "noise <N>"                           -> must equal `noise`
    A4  a comment line beginning "<N> = ..."  -> must equal `min` or `max`
        (the derivation form: "1800 = the 250-turn ceiling x 7.22 s/turn").
        A1/A2/A4 cannot tell a claim from a QUOTATION of a dead claim, and
        there is deliberately NO escape-hatch comment -- an escape hatch is
        how prose starts drifting again. Convention instead: write the LIVE
        bound as "the floor of N" so A1 checks it, and a dead one in any
        other shape ("advertising N as the bar"). Being made to mark which is
        which is the point; it is the distinction the seed defect lost.
    A5  "EXACT pin" / "pinned at exactly"     -> `min` must equal `max`
        The adjacent class: a bound documented as a pin whose enforcement is
        a one-sided floor pins nothing. (`scoring::compare` treats min==max
        as the pin arm; a lone `min` is a floor.)

  Scope B -- any comment line or `note` line that names a metric declared in
  the SAME [[benchmarks]] entry and attaches a number to it. Entry scoping is
  not a detail: `sum_wall_s` is declared twice in one file (agentic 1800,
  ssm-poison 1200) and `geometry_matched` three times, so a file-wide lookup
  scores prose against whichever entry happened to be last. A shared header
  block sitting above an entry counts as that entry's preamble.

    B1  "<name> = <N>", "<name> >= <N>", "<name> <= <N>",
        "<name> is pinned at [exactly] <N>", "<name> must be <N>"
        -> compared against `min`/`max` as the operator implies.
        The metric name is an exact token from the file, which is why this
        rule is safe to run over free prose where A1-A5 would not be. It
        fires only where the name is BACKTICKED or opens the line, i.e. is
        referred to rather than used as a unit -- without that guard it read
        "40 turns x 10 iterations = 400 turns" as a claim about the
        `iterations` bound. The GREEN fixture pins that case.

WHAT THIS CANNOT SEE. Recall is low and that is stated rather than hidden:

  * FREE PROSE THAT NAMES NO METRIC. The concurrency-sweep note said "the five
    committed floors (24/43/63/94/94) ... stand untouched as placeholders" for
    weeks after those five were replaced by eight different numbers. No rule
    here fires on it, and writing one that did would mean matching that
    sentence, not that class. A human sweep found it; a human sweep is still
    the only tool for it.
  * SYMBOLIC ALIASES. "Sigma-wall <= 1300 s" annotates `sum_wall_s`, but B1
    keys on the declared metric name, so the alias is invisible.
  * DERIVATIONS. "mean minus ~1 tok/s", "the slower tier + ~20% headroom",
    "best-ever minus 0.05" are arithmetic over numbers this script does not
    hold. A1 catches the seed defect because the prose ALSO stated the result
    ("Floor 27.0"); had it said only "mean minus ~1 tok/s" nothing would fire.
  * CROSS-FILE AND CROSS-LANGUAGE restatements. `decode_floor/mod.rs` and
    `gate/coverage.rs` both describe this same floor in doc comments; neither
    is scanned. Only BENCH.toml is.
  * WHETHER THE NUMBER IS RIGHT. This checks prose against value, never value
    against reality. A bound and a comment that agree and are both wrong pass.

PRECISION, on the tree at the time of writing (4 BENCH.toml, 83 bounds): 28
assertions -- A1 1, A2 0, A3 1, A4 1, A5 1, B1 24 -- one real contradiction
(the seed), zero false alarms. Zero only because the article/sentence-initial
and backtick/line-initial guards were added AFTER the first cut cried wolf on
real prose; both are pinned by GREEN fixtures so they cannot be loosened
silently.

RECALL IS THE WEAK SIDE AND IS NOT ESTIMATED. 28 assertions over 83 bounds
means most bounds have nothing asserted about them. This finds the shapes
above and nothing else; a green run does not mean the prose in this tree is
honest.

Stdlib only, no GPU, no network. Exit 1 on any violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

NUM = r"(\d+(?:\.\d+)?)"

# A1/A2 fire in exactly two shapes, and the narrowness IS the rule:
#
#   ARTICLE  "the floor of 1", "a ceiling of 8.5" -- the article must sit
#            IMMEDIATELY before the word. "the MLPerf floor 83.64" therefore
#            does not match, because "MLPerf" is in between and names a
#            different bound.
#   INITIAL  "Floor 27.0" at the start of a comment line or a sentence.
#            Capitalised and sentence-initial is how this corpus writes a
#            claim about the bound in hand; "(vacuity floor 160)" mid-sentence
#            and lowercase is how it writes a claim about something else.
#
# Loosening either shape reintroduces the false alarms the GREEN fixture
# 'qualified floors name other bounds' exists to pin.
RE_FLOOR = [
    re.compile(rf"\b(?:[Tt]he|[Aa]n?|[Tt]his)\s+floor(?:\s+of)?\s+{NUM}"),
    re.compile(rf"(?:^|(?<=\. ))Floor\s+{NUM}"),
]
RE_CEIL = [
    re.compile(rf"\b(?:[Tt]he|[Aa]n?|[Tt]his)\s+ceiling(?:\s+of)?\s+{NUM}"),
    re.compile(rf"(?:^|(?<=\. ))Ceiling\s+{NUM}"),
]
RE_NOISE = re.compile(rf"(?<![\w_])noise\s+{NUM}")
RE_LEADING_DERIVATION = re.compile(rf"^{NUM}\s*=\s*\S")
RE_PIN_CLAIM = re.compile(r"EXACT pin|exact pin|[Pp]inned (?:at|to) exactly")

# B1 operators, longest first so ">=" is not read as ">".
B1_OPS = [
    ("is pinned at exactly", "pin"),
    ("is pinned to exactly", "pin"),
    ("is PINNED at exactly", "pin"),
    ("is PINNED to exactly", "pin"),
    ("is pinned at", "pin"),
    ("is PINNED at", "pin"),
    ("must be", "either"),
    (">=", "min"),
    ("≥", "min"),
    ("<=", "max"),
    ("≤", "max"),
    ("==", "either"),
    ("=", "either"),
]

TOL = 1e-9


class Bound:
    def __init__(self, name: str, line: int, entry: int) -> None:
        self.name = name
        self.line = line
        # Which [[benchmarks]] table this bound belongs to. Load-bearing: one
        # file declares `sum_wall_s` twice (agentic 1800, ssm-poison 1200) and
        # `geometry_matched` three times. A file-wide name lookup silently
        # scored prose against the LAST entry's number.
        self.entry = entry
        self.min: float | None = None
        self.max: float | None = None
        self.noise: float | None = None
        self.comment: list[tuple[int, str]] = []

    def is_pin(self) -> bool:
        return self.min is not None and self.max is not None and abs(self.min - self.max) < TOL

    def has(self, value: float, which: str) -> bool:
        """Does `value` match this bound under the operator's reading?"""
        cands: list[float | None]
        if which == "min":
            cands = [self.min]
        elif which == "max":
            cands = [self.max]
        elif which == "pin":
            cands = [self.min] if self.is_pin() else [None]
        else:  # "either" -- an `=` says nothing about which arm it means
            cands = [self.min, self.max]
        return any(c is not None and abs(c - value) < TOL for c in cands)

    def render(self) -> str:
        parts = [f"{k} = {v}" for k, v in
                 (("min", self.min), ("max", self.max), ("noise", self.noise)) if v is not None]
        return ", ".join(parts) if parts else "<no bound>"


def parse(text: str) -> list[Bound]:
    """Collect every [benchmarks.metrics.*] table with its inline comment block."""
    bounds: list[Bound] = []
    cur: Bound | None = None
    pending: list[tuple[int, str]] = []
    seen_key = False
    entry = -1
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line == "[[benchmarks]]":
            entry += 1
            cur = None
            continue
        header = re.fullmatch(r"\[benchmarks\.metrics\.([A-Za-z0-9_]+)\]", line)
        if header:
            cur = Bound(header.group(1), i, entry)
            bounds.append(cur)
            pending = []
            seen_key = False
            continue
        if line.startswith("["):
            cur = None
            continue
        if cur is None:
            continue
        kv = re.fullmatch(rf"(min|max|noise)\s*=\s*{NUM}", line)
        if kv:
            setattr(cur, kv.group(1), float(kv.group(2)))
            # Comments accumulated BEFORE the first key annotate this bound.
            # Comments after it belong to whatever comes next.
            if not seen_key:
                cur.comment = pending
                seen_key = True
            pending = []
            continue
        if line.startswith("#"):
            if not seen_key:
                pending.append((i, line.lstrip("#").strip()))
            continue
        if line:
            cur = None
    return bounds


def scope_a(path: Path, bounds: list[Bound]) -> list[str]:
    out = []
    for b in bounds:
        joined = " ".join(t for _, t in b.comment)
        for ln, txt in b.comment:
            for rx in RE_FLOOR:
                for m in rx.finditer(txt):
                    v = float(m.group(1))
                    if not b.has(v, "min"):
                        out.append(f"{path}:{ln}: [A1] {b.name}: prose says floor {v}, "
                                   f"but the bound in force is {b.render()} "
                                   f"(declared at line {b.line})")
            for rx in RE_CEIL:
                for m in rx.finditer(txt):
                    v = float(m.group(1))
                    if not b.has(v, "max"):
                        out.append(f"{path}:{ln}: [A2] {b.name}: prose says ceiling {v}, "
                                   f"but the bound in force is {b.render()}")
            for m in RE_NOISE.finditer(txt):
                v = float(m.group(1))
                if b.noise is None or abs(b.noise - v) > TOL:
                    out.append(f"{path}:{ln}: [A3] {b.name}: prose says noise {v}, "
                               f"but the bound in force is {b.render()}")
            m = RE_LEADING_DERIVATION.match(txt)
            if m:
                v = float(m.group(1))
                if not b.has(v, "either"):
                    out.append(f"{path}:{ln}: [A4] {b.name}: prose derives {v}, "
                               f"but the bound in force is {b.render()}")
        if RE_PIN_CLAIM.search(joined) and not b.is_pin():
            out.append(f"{path}:{b.line}: [A5] {b.name}: documented as an EXACT pin, but "
                       f"enforcement is {b.render()} — min == max is the only arm "
                       f"scoring::compare treats as a pin")
    return out


def owning_entry(lines: list[str]) -> list[int]:
    """Per line, the index of the [[benchmarks]] entry that line documents.

    A line inside an entry belongs to it. A trailing run of comment/blank
    lines immediately before the NEXT `[[benchmarks]]` header belongs to that
    next entry instead -- that is where this tree puts a gate's shared header
    block (the vision-fidelity `integrity_passed = 10` table, for one), and
    attributing it backwards scored it against the previous gate.
    """
    n = len(lines)
    owner = [-1] * n
    cur = -1
    for i, raw in enumerate(lines):
        if raw.strip() == "[[benchmarks]]":
            cur += 1
        owner[i] = cur
    # Walk backwards: a comment/blank run that ends at an entry header is a
    # preamble for that header, not a tail of what came before.
    nxt = None
    for i in range(n - 1, -1, -1):
        line = lines[i].strip()
        if line == "[[benchmarks]]":
            nxt = owner[i]
            continue
        if line.startswith("#") or line == "":
            if nxt is not None:
                owner[i] = nxt
            continue
        nxt = None
    return owner


def scope_b(path: Path, text: str, bounds: list[Bound]) -> list[str]:
    if not bounds:
        return []
    per_entry: dict[int, dict[str, Bound]] = {}
    for b in bounds:
        per_entry.setdefault(b.entry, {})[b.name] = b
    names = sorted({b.name for b in bounds}, key=len, reverse=True)
    name_re = re.compile(r"`?\b(" + "|".join(re.escape(n) for n in names) + r")\b`?")
    lines = text.splitlines()
    owner = owning_entry(lines)
    out = []
    for i, raw in enumerate(lines, start=1):
        for nm in name_re.finditer(raw):
            # The name must be REFERRED TO, not merely used as a unit. A
            # backtick makes it a reference; so does starting the line. What
            # this excludes is arithmetic that happens to end in the word:
            # "40 turns x 10 iterations = 400 turns" is not a claim that the
            # `iterations` bound is 400, and B1 fired on it before this guard.
            backticked = nm.group(0).startswith("`")
            line_initial = raw[: nm.start()].strip(" \t#-*") == ""
            if not (backticked or line_initial):
                continue
            # Resolve inside the owning entry only. A name this entry does not
            # declare is a cross-reference to another gate, which this rule
            # deliberately does not judge -- it has no way to know which one.
            b = per_entry.get(owner[i - 1], {}).get(nm.group(1))
            if b is None:
                continue
            rest = raw[nm.end():]
            stripped = rest.lstrip()
            for op, which in B1_OPS:
                if not stripped.startswith(op):
                    continue
                tail = stripped[len(op):]
                m = re.match(rf"\s*{NUM}", tail)
                if not m:
                    break
                v = float(m.group(1))
                if not b.has(v, which):
                    out.append(f"{path}:{i}: [B1] {b.name}: prose says "
                               f"'{b.name} {op} {m.group(1)}', but the bound in force is "
                               f"{b.render()} (declared at line {b.line})")
                break
    return out


def check_file(path: Path, label: Path | None = None) -> list[str]:
    """Scan `path`; report it as `label` (a repo-relative path) if given.

    The split matters: an earlier cut passed the RELATIVE path to both, so the
    read resolved against the process CWD instead of the repo root and the
    script silently scanned a different tree than the one it printed. It was
    caught by the negative control reading clean on a tree that provably
    contained the defect -- which is the whole reason that control is run.
    """
    text = path.read_text(encoding="utf-8")
    bounds = parse(text)
    shown = label if label is not None else path
    return scope_a(shown, bounds) + scope_b(shown, text, bounds)


# ---------------------------------------------------------------------------
# Self-test: every rule proven able to FAIL, then proven able to pass.
#
# A rule that has only ever been seen green is indistinguishable from one that
# cannot fire. The fixtures are the negative control and run in the same
# invocation as the real scan, so a rule broken into inertness by a future
# edit is caught here rather than by nobody. They live in a sibling module
# only because of the 500-line ceiling, and the import is HARD: a missing
# fixture file must stop the run, never quietly reduce this to a scan with no
# control behind it.
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_bench_threshold_prose_fixtures import GREEN, RED  # noqa: E402

RULES = ("[A1]", "[A2]", "[A3]", "[A4]", "[A5]", "[B1]")
_uncovered = set(RULES) - {rule for _, _, rule in RED}
assert not _uncovered, (
    f"rules with no RED fixture: {sorted(_uncovered)} — each is unproven and "
    "may be unable to fire at all"
)


def self_test() -> int:
    failures = []
    for name, toml, rule in RED:
        p = Path(f"<red:{name}>")
        found = scope_a(p, parse(toml)) + scope_b(p, toml, parse(toml))
        hits = [f for f in found if rule in f]
        if not hits:
            failures.append(f"RED fixture '{name}' did NOT fire {rule} — "
                            f"that rule cannot fail and therefore checks nothing. "
                            f"(got: {found or 'no findings at all'})")
        else:
            print(f"  red   {rule} {name}\n          -> {hits[0].split(': ', 1)[1]}")
    for name, toml in GREEN:
        p = Path(f"<green:{name}>")
        found = scope_a(p, parse(toml)) + scope_b(p, toml, parse(toml))
        if found:
            failures.append(f"GREEN fixture '{name}' produced a FALSE ALARM: {found}")
        else:
            print(f"  green      {name}")
    if failures:
        print("\nself-test FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"self-test OK: {len(RED)} rules proven able to fail, "
          f"{len(GREEN)} no-alarm cases held")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--self-test" in argv:
        return self_test()

    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "kernels").glob("*/*/BENCH.toml"))
    if not files:
        # A checker that silently scans nothing is the failure mode this whole
        # script exists to prevent.
        print(f"threshold prose: NO BENCH.toml found under {root / 'kernels'}")
        return 1

    rc = self_test()
    if rc:
        return rc

    violations: list[str] = []
    bounds_seen = 0
    for path in files:
        bounds_seen += len(parse(path.read_text(encoding="utf-8")))
        violations.extend(check_file(path, path.relative_to(root)))

    if violations:
        print(f"\nthreshold prose: {len(violations)} contradiction(s) "
              f"across {len(files)} file(s), {bounds_seen} bound(s)")
        for v in violations:
            print(f"  {v}")
        print("\nFix the PROSE to match the value in force. Changing a measured "
              "bound is a certification event and needs its own campaign.")
        return 1

    print(f"\nthreshold prose: OK — {bounds_seen} bound(s) in {len(files)} file(s), "
          "no comment contradicts the value it annotates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
