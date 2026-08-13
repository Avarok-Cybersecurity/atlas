// SPDX-License-Identifier: AGPL-3.0-only

//! Best-effort scrubbing and size-budgeting for the log tail an issue report
//! attaches — pure functions, so every rule is testable without a filesystem
//! or a network.
//!
//! # What this can and cannot do
//!
//! The attachment is posted to a PUBLIC issue tracker. What is reliably
//! machine-recognizable gets redacted here: credential shapes (GitHub, Hugging
//! Face, `sk-`, AWS key IDs, `Authorization`/`Bearer`/`token=` values), the
//! user's home directory, username and hostname, and non-loopback IP literals.
//! What is NOT redacted — prompts or chat text that reached the logs, model
//! names, free-text error messages — cannot be recognized by shape, and
//! claiming otherwise would train users to skip the preview that is the real
//! mitigation. The UI words its promise to match this file, not the reverse.
//!
//! False positives are accepted by design: over-redacting a phrase that merely
//! looks like a key costs a little context in a bug report; under-redacting a
//! real key costs a credential.

/// What a recognized secret is replaced with.
pub const REDACTED: &str = "«redacted»";

/// What a non-loopback IP literal is replaced with.
pub const REDACTED_IP: &str = "«ip»";

/// GitHub rejects issue bodies over this many characters
/// ("Body is too long (maximum is 65536 characters)").
pub const GITHUB_BODY_LIMIT: usize = 65_536;

/// The budget this module actually fills to. Under the GitHub limit on
/// purpose: GitHub's counting of astral/emoji content is not documented to
/// match `chars().count()`, and a body assembled exactly at the limit would
/// turn that ambiguity into a 422 after the user already pressed send.
pub const BODY_BUDGET: usize = 60_000;

/// The identity strings to scrub, resolved once by the caller so the scrubbing
/// itself stays pure — a function that reaches into the environment per line
/// cannot be tested without mutating process globals.
pub struct RedactCtx {
    pub home: Option<String>,
    pub user: Option<String>,
    pub host: Option<String>,
}

impl RedactCtx {
    /// Resolve from the live environment. Values of one character are dropped:
    /// substituting every single letter that happens to match would shred the
    /// log instead of anonymizing it.
    pub fn from_env() -> Self {
        let keep = |s: String| {
            let t = s.trim().to_string();
            (t.len() > 1).then_some(t)
        };
        Self {
            home: std::env::var("HOME").ok().and_then(keep),
            user: std::env::var("USER").ok().and_then(keep),
            host: std::fs::read_to_string("/proc/sys/kernel/hostname")
                .ok()
                .or_else(|| std::env::var("HOSTNAME").ok())
                .and_then(keep),
        }
    }
}

/// Scrub one log line: credential shapes, then identity, then IPs — in that
/// order, so an identity substring inside a token that was already redacted
/// cannot resurrect part of it.
pub fn redact_line(line: &str, ctx: &RedactCtx) -> String {
    let s = redact_credentials(line);
    let s = redact_identity(&s, ctx);
    redact_ips(&s)
}

/// A recognizable credential prefix and what its tail looks like.
struct Shape {
    prefix: &'static str,
    tail: fn(char) -> bool,
    min_tail: usize,
}

fn alnum(c: char) -> bool {
    c.is_ascii_alphanumeric()
}
fn alnum_underscore(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}
fn keyish(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}
fn upper_alnum(c: char) -> bool {
    c.is_ascii_uppercase() || c.is_ascii_digit()
}

/// The shapes this build recognizes. `gh?_` covers our own GitHub tokens as
/// defence-in-depth — they never enter logs by construction, but a rule that
/// depends on every other rule holding is not a rule.
const SHAPES: [Shape; 9] = [
    Shape {
        prefix: "ghp_",
        tail: alnum,
        min_tail: 36,
    },
    Shape {
        prefix: "gho_",
        tail: alnum,
        min_tail: 36,
    },
    Shape {
        prefix: "ghu_",
        tail: alnum,
        min_tail: 36,
    },
    Shape {
        prefix: "ghs_",
        tail: alnum,
        min_tail: 36,
    },
    Shape {
        prefix: "ghr_",
        tail: alnum,
        min_tail: 36,
    },
    Shape {
        prefix: "github_pat_",
        tail: alnum_underscore,
        min_tail: 22,
    },
    Shape {
        prefix: "hf_",
        tail: alnum,
        min_tail: 30,
    },
    Shape {
        prefix: "sk-",
        tail: keyish,
        min_tail: 20,
    },
    Shape {
        prefix: "AKIA",
        tail: upper_alnum,
        min_tail: 16,
    },
];

fn redact_credentials(line: &str) -> String {
    let mut s = line.to_string();
    for sh in &SHAPES {
        s = redact_shape(&s, sh);
    }
    // Header and key=value forms, where the secret has no recognizable shape
    // of its own — the surrounding syntax is the signal. `token=` is a
    // substring match, so `access_token=`/`refresh_token=` are covered too.
    s = redact_after(&s, "authorization:", true);
    for key in [
        "bearer ",
        "token=",
        "secret=",
        "api_key=",
        "apikey=",
        "password=",
    ] {
        s = redact_after(&s, key, false);
    }
    s
}

fn redact_shape(s: &str, sh: &Shape) -> String {
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while let Some(off) = s[i..].find(sh.prefix) {
        let start = i + off;
        // The char before the prefix must not be alphanumeric: "risk-…" is
        // prose, not an `sk-` key mid-word.
        let bounded = s[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_ascii_alphanumeric());
        let after = start + sh.prefix.len();
        // Tail chars are all ASCII, so the char count is the byte count.
        let run = s[after..].chars().take_while(|&c| (sh.tail)(c)).count();
        if bounded && run >= sh.min_tail {
            out.push_str(&s[i..start]);
            out.push_str(REDACTED);
            i = after + run;
        } else {
            out.push_str(&s[i..after]);
            i = after;
        }
    }
    out.push_str(&s[i..]);
    out
}

/// ASCII-case-insensitive substring search. Hand-rolled rather than
/// `to_lowercase()` + `find`, because non-ASCII lowercasing can change byte
/// lengths and the returned index must be valid in the ORIGINAL string.
fn find_ci(hay: &str, needle: &str, from: usize) -> Option<usize> {
    let h = hay.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || from + n.len() > h.len() {
        return None;
    }
    (from..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

/// Redact whatever follows `key` — the rest of the line for headers, the next
/// token otherwise. The key itself is kept: the reader should still see THAT
/// a token was sent, only not WHICH.
fn redact_after(s: &str, key: &str, to_eol: bool) -> String {
    const DELIMS: &[char] = &[' ', '\t', '"', '\'', '&', ',', ';', ')', ']', '}'];
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while let Some(pos) = find_ci(s, key, i) {
        let vstart = pos + key.len();
        out.push_str(&s[i..vstart]);
        if to_eol {
            if s[vstart..].trim().is_empty() {
                return out + &s[vstart..];
            }
            out.push(' ');
            out.push_str(REDACTED);
            return out;
        }
        let rest = &s[vstart..];
        let skip: usize = rest
            .chars()
            .take_while(|&c| c == ' ')
            .map(char::len_utf8)
            .sum();
        let val: usize = rest[skip..]
            .chars()
            .take_while(|c| !DELIMS.contains(c))
            .map(char::len_utf8)
            .sum();
        if val == 0 {
            i = vstart;
            continue;
        }
        out.push_str(&rest[..skip]);
        out.push_str(REDACTED);
        i = vstart + skip + val;
    }
    out.push_str(&s[i..]);
    out
}

fn redact_identity(s: &str, ctx: &RedactCtx) -> String {
    let mut s = s.to_string();
    // Home before username: `$HOME` usually CONTAINS the username, and
    // replacing the name first would leave `/home/«user»` fragments the home
    // pass no longer matches.
    if let Some(h) = &ctx.home {
        s = s.replace(h.as_str(), "~");
    }
    if let Some(u) = &ctx.user {
        s = s.replace(&format!("/home/{u}"), "~");
        s = replace_word(&s, u, "«user»");
    }
    if let Some(h) = &ctx.host {
        s = replace_word(&s, h, "«host»");
    }
    s
}

/// Whole-word replacement — a username that happens to be an English word
/// must not be substituted mid-token.
fn replace_word(s: &str, word: &str, with: &str) -> String {
    let boundary = |c: Option<char>| c.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while let Some(off) = s[i..].find(word) {
        let start = i + off;
        let end = start + word.len();
        if boundary(s[..start].chars().next_back()) && boundary(s[end..].chars().next()) {
            out.push_str(&s[i..start]);
            out.push_str(with);
        } else {
            out.push_str(&s[i..end]);
        }
        i = end;
    }
    out.push_str(&s[i..]);
    out
}

/// Replace non-loopback IP literals. Tokens are maximal runs of the IP
/// alphabet, then parsed with `std::net` — the parser, not a pattern, decides
/// what an address is, so `12:34:56` timestamps and `3.3.0` versions survive.
fn redact_ips(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut token = String::new();
    for c in s.chars() {
        if c.is_ascii_hexdigit() || c == ':' || c == '.' {
            token.push(c);
        } else {
            push_ip_token(&mut out, &token);
            token.clear();
            out.push(c);
        }
    }
    push_ip_token(&mut out, &token);
    out
}

fn push_ip_token(out: &mut String, token: &str) {
    if token.is_empty() {
        return;
    }
    let core = token.trim_end_matches(['.', ':']);
    let suffix = &token[core.len()..];
    if let Ok(ip) = core.parse::<std::net::IpAddr>() {
        if keep_ip(&ip) {
            out.push_str(token);
        } else {
            out.push_str(REDACTED_IP);
            out.push_str(suffix);
        }
        return;
    }
    // `10.10.10.1:8000` — an IPv4 with a port is not parseable whole. The
    // port is configuration, not identity; only the address is replaced.
    if let Some((head, tail)) = core.split_once(':')
        && let Ok(ip) = head.parse::<std::net::Ipv4Addr>()
    {
        if keep_ip(&std::net::IpAddr::V4(ip)) {
            out.push_str(token);
        } else {
            out.push_str(REDACTED_IP);
            out.push(':');
            out.push_str(tail);
            out.push_str(suffix);
        }
        return;
    }
    out.push_str(token);
}

/// Loopback and unspecified addresses stay: `127.0.0.1` and `0.0.0.0` say
/// nothing about the box and everything about how the server was bound —
/// which is often the bug being reported.
fn keep_ip(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => v4.is_loopback() || v4.is_unspecified(),
        std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.is_unspecified(),
    }
}

// ── Size budget ──

/// A log tail trimmed to fit a character budget.
pub struct TrimmedLog {
    pub text: String,
    pub included: usize,
    pub total: usize,
}

/// The line that stands in for what was dropped — visible in the preview and
/// in the posted issue, so nobody mistakes a trimmed log for a complete one.
pub fn omission_marker(omitted: usize, tee_path: Option<&str>) -> String {
    format!(
        "— {omitted} earlier lines omitted to fit GitHub's 65,536-character limit; full log: {} —",
        tee_path.unwrap_or("(tee file unavailable)")
    )
}

/// Keep the newest lines that fit `budget` characters (counting one `\n` per
/// line). Oldest-first trimming, because the newest lines are the ones that
/// describe the failure being reported.
pub fn trim_to_budget(lines: &[String], budget: usize, tee_path: Option<&str>) -> TrimmedLog {
    let total = lines.len();
    let full: usize = lines.iter().map(|l| l.chars().count() + 1).sum();
    if full <= budget {
        return TrimmedLog {
            text: lines.join("\n"),
            included: total,
            total,
        };
    }
    // Reserve marker room using the worst-case omission count — the count can
    // only shrink as more lines fit, so the reserve never under-counts.
    let mut used = omission_marker(total, tee_path).chars().count() + 1;
    let mut keep = 0;
    for l in lines.iter().rev() {
        let c = l.chars().count() + 1;
        if used + c > budget {
            break;
        }
        used += c;
        keep += 1;
    }
    let mut text = omission_marker(total - keep, tee_path);
    for l in &lines[total - keep..] {
        text.push('\n');
        text.push_str(l);
    }
    TrimmedLog {
        text,
        included: keep,
        total,
    }
}

/// The code fence wide enough that `content` cannot close it. Logs can carry
/// ``` sequences; per CommonMark a fence is closed only by a run at least as
/// long, so one more backtick than the longest embedded run keeps the log
/// inside its block instead of rendering as markdown in the public issue.
pub fn fence_for(content: &str) -> String {
    let mut longest = 0usize;
    let mut run = 0usize;
    for c in content.chars() {
        if c == '`' {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    "`".repeat((longest + 1).max(3))
}

#[cfg(test)]
#[path = "redact_tests.rs"]
mod tests;
