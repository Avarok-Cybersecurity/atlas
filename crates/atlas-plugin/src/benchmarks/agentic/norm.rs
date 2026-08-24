// SPDX-License-Identifier: AGPL-3.0-only

//! Normalising shell output so that identical work reads identically.
//!
//! **Why this exists.** Gate A's bar is an exact 10-of-10 on two counts, and a
//! gate that samples cannot tell a regression from a draw. Greedy decoding
//! (`agent::TEMPERATURE`) fixes only half of it: the agent's context is not the
//! prompt alone, it is the prompt *plus every byte of tool output fed back into
//! it*. A `cargo build` that reports `in 0.00s` on one run and `in 1.23s` on the
//! next hands the model two different token sequences for the same work, and
//! from there two greedy trajectories diverge legitimately. Removing that
//! variance is what makes a repeat run a repeat.
//!
//! **Normalising is not hiding.** Every rule below is anchored on a shape that
//! only tool *bookkeeping* has — a cargo status word at the start of a line, a
//! `time(1)` row, curl's progress meter, an ISO timestamp. Nothing that carries
//! signal is touched:
//!
//! * compiler diagnostics, panics, `error[E….]`, and every `warning:` are
//!   passed through untouched — they are the reason a run fails and the model's
//!   only route back;
//! * **ports are never scrubbed.** `ATLAS_HARNESS_PORT` (default 3001) is part
//!   of the prompt's contract, and a model that cannot read the port back out
//!   of `fuser`/`curl` output cannot tear its server down. Only the PID column
//!   of `fuser`'s output is rewritten, never the `<port>/tcp` it is keyed by;
//! * exit statuses stay: the digit-run rules start at four digits precisely so
//!   `(exit status: 101)` and `test result: … 2 passed` survive;
//! * file contents are not routed through here at all — `read`, `grep` and
//!   `glob` results bypass it, so an `edit` whose `oldString` came from a
//!   previous `read` still matches the bytes on disk.
//!
//! Applied at exactly one place, `agent::run_shell`, and *before* truncation:
//! normalising afterwards would leave the elision counts (and therefore the cut
//! points) varying with the raw byte lengths this module exists to remove.
//!
//! **What this cannot reach.** The agent runs real shell on a real box, so it
//! can see the box. `ps aux | grep <its crate>` — a command this model issues
//! unprompted — returns the COMMAND line of every matching process, including
//! anything *else* running that happens to mention the benchmark. The numeric
//! columns are normalised (see `ps_line` below); the command text of a stranger's
//! process is data we must not invent. So a tier is only repeatable on a box
//! that is quiet: no second benchmark, and no polling loop of your own whose
//! command line the agent's `grep` would match. Ordinary timing (a build that
//! finishes on one side of a `sleep 5` and not the other) is likewise outside
//! what any text rule can fix.

/// Lines that report *progress*, not results. Dropping them is what makes a
/// cold build and a warm one read the same: `Compiling …` appears only when a
/// unit is stale, and the shared warm target dir guarantees the second tier
/// finds units the first tier left behind. `Updating`/`Downloading`/`Locking`
/// are the same story for the registry cache. The `Finished` line — the one
/// that says whether the build worked — is deliberately not in this list.
const PROGRESS: [&str; 7] = [
    "Compiling ",
    "Fresh ",
    "Downloading ",
    "Downloaded ",
    "Updating ",
    "Locking ",
    "Blocking waiting",
];

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Longest first: `ms` must win over `s`.
const UNITS: [&str; 6] = ["ns", "µs", "us", "ms", "s", "m"];

/// Normalise one shell result. Idempotent: every placeholder it writes is
/// digit-free, so no rule can match its own output.
pub fn normalize(text: &str) -> String {
    let mut lines: Vec<String> = text
        .split('\n')
        .filter(|line| !is_noise(line))
        .map(scrub)
        .collect();
    sort_test_results(&mut lines);
    lines.join("\n")
}

/// libtest prints one `test <name> ... ok` line per test **as it finishes**,
/// from a thread pool sized to the machine, so two runs of the same green suite
/// routinely emit those lines in different orders. Sorting each consecutive
/// block imposes the order libtest never had; nothing is added, removed or
/// rewritten, and the summary line (`test result: …`, which does not match the
/// shape) keeps the counts. A failing test's `---- <name> stdout ----` block is
/// a different shape and is never reordered.
fn sort_test_results(lines: &mut [String]) {
    let mut i = 0;
    while i < lines.len() {
        if !is_test_result(&lines[i]) {
            i += 1;
            continue;
        }
        let mut end = i;
        while end < lines.len() && is_test_result(&lines[end]) {
            end += 1;
        }
        lines[i..end].sort();
        i = end;
    }
}

fn is_test_result(line: &str) -> bool {
    line.starts_with("test ") && line.contains(" ... ")
}

/// A line with no content at all — pure progress reporting.
fn is_noise(line: &str) -> bool {
    let head = line.trim_start();
    PROGRESS.iter().any(|p| head.starts_with(p))
        // curl's progress meter, which prints transfer *rates*. The two header
        // rows plus every meter row (all of which carry a `--:--:--` clock).
        // The response body is on stdout and is untouched.
        || line.contains("--:--:--")
        || head.starts_with("% Total")
        || head.starts_with("Dload")
}

fn scrub(line: &str) -> String {
    let line = replace_spans(line);
    let line = trailing_duration(&line);
    let line = time_builtin(&line);
    pids(&line)
}

// ── span rules ─────────────────────────────────────────────────────

/// Rewrite every timestamp-shaped span, at word boundaries only so that a run
/// of digits inside an identifier (a cargo metadata hash, a version) is never
/// mistaken for one.
fn replace_spans(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < line.len() {
        let rest = &line[i..];
        let boundary = !line[..i].chars().next_back().is_some_and(identifier_char);
        if boundary && let Some((len, replacement)) = span_at(rest) {
            let ends_at_boundary = !rest[len..].chars().next().is_some_and(identifier_char);
            if ends_at_boundary {
                out.push_str(replacement);
                i += len;
                continue;
            }
        }
        let c = rest.chars().next().unwrap_or('\0');
        out.push(c);
        i += c.len_utf8();
    }
    out
}

fn identifier_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn span_at(s: &str) -> Option<(usize, &'static str)> {
    if let Some(n) = timestamp(s) {
        return Some((n, "<timestamp>"));
    }
    if let Some(n) = listing_date(s) {
        return Some((n, "<date>"));
    }
    // `curl: (7) Failed to connect to localhost port 3001 after 0 ms: …` — the
    // port stays, the connect attempt's duration does not.
    if let Some(rest) = s.strip_prefix("after")
        && let Some(n) = duration(rest.as_bytes(), 0)
    {
        return Some((5 + n, "after <elapsed>"));
    }
    None
}

/// `2026-08-06T21:09:12.345Z`, and the space-separated form a `tracing`
/// subscriber prints. A bare date with no clock is left alone: it does not move
/// during a run and could be data.
fn timestamp(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut i = digits(b, 0, 4, 4)?;
    i = byte(b, i, b'-')?;
    i = digits(b, i, 2, 2)?;
    i = byte(b, i, b'-')?;
    i = digits(b, i, 2, 2)?;
    if !matches!(b.get(i), Some(b'T') | Some(b' ')) {
        return None;
    }
    let mut j = clock(b, i + 1)?;
    if b.get(j) == Some(&b'.') {
        j = digits(b, j + 1, 1, 9)?;
    }
    match b.get(j) {
        Some(b'Z') => j += 1,
        Some(b'+') | Some(b'-') => {
            if let Some(k) = digits(b, j + 1, 2, 2)
                .and_then(|k| byte(b, k, b':'))
                .and_then(|k| digits(b, k, 2, 2))
            {
                j = k;
            }
        }
        _ => {}
    }
    Some(j)
}

/// `ls -l`'s mtime column and `date`'s own output: `Aug  6 21:09`,
/// `Aug  6 21:09:12`, `Aug  6  2025`.
fn listing_date(s: &str) -> Option<usize> {
    let month = MONTHS.iter().find(|m| s.starts_with(**m))?;
    let b = s.as_bytes();
    let mut i = spaces(b, month.len())?;
    i = digits(b, i, 1, 2)?;
    let j = spaces(b, i)?;
    clock(b, j).or_else(|| digits(b, j, 4, 4))
}

fn clock(b: &[u8], i: usize) -> Option<usize> {
    let mut j = digits(b, i, 2, 2)?;
    j = byte(b, j, b':')?;
    j = digits(b, j, 2, 2)?;
    if b.get(j) == Some(&b':') {
        j = digits(b, j + 1, 2, 2)?;
    }
    Some(j)
}

// ── duration rules ─────────────────────────────────────────────────

/// A line that *ends* in `… in <duration>`: cargo's `Finished` line, the test
/// harness's `test result: … finished in 0.00s`. Anchoring on the end of the
/// line is what keeps this away from prose that happens to contain " in ".
fn trailing_duration(line: &str) -> String {
    let trimmed = line.trim_end();
    let Some(cut) = trimmed.rfind(" in ") else {
        return line.to_string();
    };
    let tail = &trimmed[cut + 4..];
    match whole_duration(tail) {
        true => format!("{} in <elapsed>", &trimmed[..cut]),
        false => line.to_string(),
    }
}

/// `time(1)`'s three rows — `real\t0m1.234s`.
fn time_builtin(line: &str) -> String {
    let head = line.trim_start();
    let Some(word) = ["real", "user", "sys"]
        .iter()
        .find(|w| head.starts_with(**w))
    else {
        return line.to_string();
    };
    let rest = &head[word.len()..];
    let value = rest.trim_start();
    if value.len() == rest.len() || !whole_duration(value.trim_end()) {
        return line.to_string();
    }
    format!("{}{word}\t<elapsed>", &line[..line.len() - head.len()])
}

fn whole_duration(s: &str) -> bool {
    duration(s.as_bytes(), 0) == Some(s.len()) && !s.is_empty()
}

/// One or more `<number><unit>` groups, each optionally preceded by a space:
/// `0.42s`, `1m 03s`, `0m1.234s`, `0 ms`.
fn duration(b: &[u8], start: usize) -> Option<usize> {
    let (mut i, mut any) = (start, false);
    loop {
        let mut j = i;
        if b.get(j) == Some(&b' ') {
            j += 1;
        }
        let Some(n) = number(b, j) else { break };
        let mut k = n;
        if b.get(k) == Some(&b' ') {
            k += 1;
        }
        let Some(end) = unit(b, k) else { break };
        i = end;
        any = true;
    }
    any.then_some(i)
}

/// A unit only counts when a letter does not follow it, so `3 more` is not
/// "3 minutes" and `2 seconds` is not `2 s` + `econds`.
fn unit(b: &[u8], i: usize) -> Option<usize> {
    let rest = std::str::from_utf8(b.get(i..)?).ok()?;
    let u = UNITS.iter().find(|u| rest.starts_with(**u))?;
    let end = i + u.len();
    match b.get(end) {
        Some(c) if c.is_ascii_alphabetic() => None,
        _ => Some(end),
    }
}

fn number(b: &[u8], i: usize) -> Option<usize> {
    let j = digits(b, i, 1, 12)?;
    match b.get(j) {
        Some(b'.') => digits(b, j + 1, 1, 12),
        _ => Some(j),
    }
}

// ── process ids ────────────────────────────────────────────────────

/// Rewrite process ids, and only process ids.
///
/// Three anchored cases, in order of confidence:
///
/// 1. `fuser`'s `3001/tcp:  12345` — the port is the key and stays; every
///    number after the colon is a pid by definition of the format.
/// 2. a line that consists of nothing but numbers — `echo $!`, `pgrep`,
///    `lsof -t -i:3001`.
/// 3. a line that *names* process bookkeeping (`kill`, `pid`, `process`) —
///    `kill: (12345) - No such process`, `Killed process 12345`.
///
/// Cases 2 and 3 start at **four** digits. That floor is what keeps
/// `(exit status: 101)`, `test result: 2 passed`, and an HTTP status out of it,
/// and it costs nothing here: this box's pids are six and seven digits.
///
/// One residual is accepted knowingly and is harmless: a four-digit byte count
/// alone on a line (`… | wc -c`). `ps` output is NOT covered beyond [`ps_line`]:
/// its pid column has no anchor word, so a run that inspects processes by hand
/// still varies.
fn pids(line: &str) -> String {
    if let Some(rewritten) = ps_line(line).or_else(|| fuser_line(line)) {
        return rewritten;
    }
    // A line that already carries a placeholder has been through here; re-
    // anchoring on the word inside `<pid>` would let a second pass reach digits
    // the first pass deliberately kept (a port in a `ps` COMMAND column).
    let named = !line.contains("<pid>")
        && line
            .split(|c: char| !c.is_ascii_alphabetic())
            .any(|w| NAMES_A_PID.iter().any(|k| w.eq_ignore_ascii_case(k)));
    match named || bare_numbers(line) {
        true => digit_runs(line, 4, 7),
        false => line.to_string(),
    }
}

/// Whole words, not substrings. `contains("process")` also fired on
/// "Processing", and `contains("kill")` on "skill" — and a match here licenses
/// rewriting every four-digit number on the line, so `Processing 1000 records`
/// (the model's own program, printing its own output) came back as
/// `Processing <pid> records`. These seven are every form the genuine cases
/// take: `kill: (12345) - No such process`, `Killed process 12345`, `pid=12345`.
const NAMES_A_PID: [&str; 7] = [
    "kill",
    "killed",
    "killing",
    "pid",
    "pids",
    "process",
    "processes",
];

/// `ps aux`'s bookkeeping columns.
///
/// Not a hypothetical: the model reaches for `ps aux | grep <name>` to check
/// whether its server came up, and pid, %CPU, %MEM, VSZ, RSS, START and TIME
/// all differ between two runs of identical work. What it asked for — the
/// COMMAND tail, plus USER, TTY and STAT — is kept verbatim.
///
/// Recognised by shape, because `| grep` strips the header: eleven or more
/// fields, an all-digit pid, two `N.N` percentages, two integer sizes, and a
/// colon-separated TIME column. Prose does not have that shape.
fn ps_line(line: &str) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.len() < 11 {
        return None;
    }
    let int = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
    let decimal = |s: &str| s.split_once('.').is_some_and(|(a, b)| int(a) && int(b));
    let elapsed = |s: &str| s.contains(':') && s.split(':').all(int);
    let shaped =
        int(f[1]) && decimal(f[2]) && decimal(f[3]) && int(f[4]) && int(f[5]) && elapsed(f[9]);
    shaped.then(|| {
        format!(
            "{} <pid> <cpu> <mem> <vsz> <rss> {} {} <start> <time> {}",
            f[0],
            f[6],
            f[7],
            f[10..].join(" ")
        )
    })
}

fn fuser_line(line: &str) -> Option<String> {
    let head = line.trim_start();
    let n = digits(head.as_bytes(), 0, 1, 5)?;
    let rest = ["/tcp:", "/udp:"]
        .iter()
        .find_map(|p| head[n..].strip_prefix(*p))?;
    let keep = &line[..line.len() - rest.len()];
    Some(format!("{keep}{}", digit_runs(rest, 1, 9)))
}

fn bare_numbers(line: &str) -> bool {
    let mut fields = line.split_whitespace().peekable();
    fields.peek().is_some()
        && fields.all(|f| (4..=7).contains(&f.len()) && f.bytes().all(|c| c.is_ascii_digit()))
}

/// Replace every *free-standing* run of `min..=max` digits.
///
/// A run that touches [`GLUED`] on either side is part of something else and is
/// left alone — `pingpong-9a8b7c6d` (a cargo metadata hash), `axum-0.8.4` (a
/// version), `0.0.0.0:3001` and `3001/tcp` (an address and a port). That guard
/// is why the port survives even on a line that also says "kill".
fn digit_runs(s: &str, min: usize, max: usize) -> String {
    let (b, mut out, mut i) = (s.as_bytes(), String::with_capacity(s.len()), 0);
    while i < s.len() {
        if b[i].is_ascii_digit() {
            let end = digits(b, i, 1, usize::MAX).unwrap_or(i);
            let glued_before = i > 0 && glued(b[i - 1]);
            let glued_after = b.get(end).copied().is_some_and(glued);
            let free = !(glued_before || glued_after || after_port_word(s, i));
            match free && (min..=max).contains(&(end - i)) {
                true => out.push_str("<pid>"),
                false => out.push_str(&s[i..end]),
            }
            i = end;
            continue;
        }
        let c = s[i..].chars().next().unwrap_or('\0');
        out.push(c);
        i += c.len_utf8();
    }
    out
}

/// A number the line has just called a port is a port, and this module's header
/// promises ports are never scrubbed. The prompt's own teardown instruction is
/// "kill whatever is listening on its port", so the model writes that sentence
/// into its script and its comments: `# kill whatever is on port 3001` names a
/// pid word and carries a free-standing four-digit number, and came back with
/// the port replaced. `3001/tcp` and `0.0.0.0:3001` are already safe by
/// [`GLUED`]; this is the spelled-out case.
fn after_port_word(s: &str, i: usize) -> bool {
    let head = s[..i].trim_end_matches([' ', '\t', '=', ':']);
    head.rsplit(|c: char| !c.is_ascii_alphabetic())
        .next()
        .is_some_and(|w| w.eq_ignore_ascii_case("port"))
}

/// Neighbours that mean "this number is part of a larger token".
const GLUED: [u8; 4] = [b'.', b':', b'/', b'-'];

fn glued(c: u8) -> bool {
    c.is_ascii_alphanumeric() || GLUED.contains(&c)
}

// ── byte-level helpers ─────────────────────────────────────────────

fn digits(b: &[u8], i: usize, min: usize, max: usize) -> Option<usize> {
    let mut end = i;
    while end < b.len() && b[end].is_ascii_digit() && end - i < max {
        end += 1;
    }
    (end - i >= min).then_some(end)
}

fn byte(b: &[u8], i: usize, want: u8) -> Option<usize> {
    (b.get(i) == Some(&want)).then_some(i + 1)
}

fn spaces(b: &[u8], i: usize) -> Option<usize> {
    let mut end = i;
    while b.get(end) == Some(&b' ') {
        end += 1;
    }
    (end > i).then_some(end)
}

#[cfg(test)]
#[path = "norm_tests.rs"]
mod tests;
