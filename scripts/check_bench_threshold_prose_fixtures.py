# SPDX-License-Identifier: AGPL-3.0-only
"""Fixtures for check_bench_threshold_prose.py -- the negative control.

Split from the checker at the repo's 500-line ceiling; an exact piecewise
copy, nothing rewritten. DATA, but load-bearing data: the checker imports it
HARD (a missing file stops the run) and refuses to start unless every rule it
implements has a RED case here. A rule with no red fixture has never been seen
to fail, and is indistinguishable from one that cannot.

Each RED entry is (name, BENCH.toml fragment, expected rule id) and MUST
produce that rule id. Each GREEN entry is (name, fragment) and must produce
nothing -- those are the cry-wolf cases, drawn from real prose in the tree.
"""

RED = [
    ("A1 floor contradicts min", """
[benchmarks.metrics.server_decode_tok_s]
# Judged against the 12-run calibration: mean 28.03, sigma ~0.05.
# Floor 27.0 = mean minus ~1 tok/s; noise 0.5 covers the run-to-run band.
min = 21.0
noise = 0.5
""", "[A1]"),
    ("A2 ceiling contradicts max", """
[benchmarks.metrics.median_ms]
# The ceiling of 554.46 is the dgx1 baseline x +3%.
max = 900.0
""", "[A2]"),
    ("A3 noise contradicts noise", """
[benchmarks.metrics.overall_accuracy]
# Bar set at the measured value less the documented noise 0.4 band.
min = 83.82
noise = 0.9
""", "[A3]"),
    ("A4 derivation contradicts max", """
[benchmarks.metrics.sum_wall_s]
# 1800 = the 250-turn degeneracy ceiling x the 7.22 s/turn worst tier.
max = 5000.0
""", "[A4]"),
    ("A5 pin claim over a one-sided floor", """
[benchmarks.metrics.runs]
# EXACT pin, like the BFCL draw size: a median over a different n is not
# comparable to this floor.
min = 3.0
""", "[A5]"),
    ("B1 metric-name restatement in a header block", """
# geometry_matched  = 14  the CORRECTNESS floor.
[benchmarks.metrics.geometry_matched]
min = 12.0
""", "[B1]"),
    ("B1 restatement inside a note string", '''
[[benchmarks]]
note = """\\
`iterations` is PINNED at exactly 10, the N the two bounds are written for."""

[benchmarks.metrics.iterations]
min = 12.0
max = 12.0
''', "[B1]"),
]

RED.append((
    "B1 resolves within the entry, not the file",
    """
[[benchmarks]]
gate = "agentic"
note = """ + '"""' + """\\
`sum_wall_s` <= 1800 is the blowup ceiling."""  + '"""' + """

[benchmarks.metrics.sum_wall_s]
max = 1400.0

[[benchmarks]]
gate = "poison"

[benchmarks.metrics.sum_wall_s]
max = 1800.0
""",
    "[B1]",
))

GREEN = [
    ("qualified floors name other bounds, not this one", """
[benchmarks.metrics.min_completion_tokens]
# Observed 174/187/200 at osl 200 (vacuity floor 160); 164 sits below the
# minimum with margin and above the 80%-of-osl rule. Do not confuse it with
# the MLPerf floor 83.64 or the serial floor a thinking-on serve measures.
min = 164.0
"""),
    ("an honest floor, an honest noise, an honest pin", """
[benchmarks.metrics.server_decode_tok_s]
# The floor of 21.0 is the in-gate n=3 basis; noise 0.5 covers the band.
min = 21.0
noise = 0.5

[benchmarks.metrics.runs]
# EXACT pin: a median over a different n is not comparable.
min = 3.0
max = 3.0
"""),
    ("sample counts and sigmas in prose are not bound claims", """
[benchmarks.metrics.c8_aggregate_tok_s]
# C=8's sigma is 4.21 across three back-to-back reps, so 3*sigma is 12.6 and
# the floor lands 21% under the mean of 60.6.
min = 48.0
noise = 2.3
"""),
    ("a metric name used as a unit inside arithmetic is not a claim", """
[benchmarks.metrics.iterations]
# Reachable, so not vacuous: the per-iteration cap is 40 turns
# x 10 iterations = 400 turns.
min = 10.0
max = 10.0
"""),
]
