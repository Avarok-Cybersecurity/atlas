// SPDX-License-Identifier: AGPL-3.0-only

//! Text helpers for the dense-GEMM source-contract tests.
//!
//! Deliberately dumb: brace/paren matching and comment stripping, no parser
//! dependency. Split out of `gemm_dense_tests.rs` to keep both files under the
//! repo's 500-line cap.
//!
//! ★ Every extractor here returns `Option`/an empty `Vec` when it cannot find
//! what it was asked for. The CALLERS assert that they found something, so a
//! kernel or launcher that changes shape produces a LOUD failure telling you to
//! re-read the guard — never a silently vacuous pass.

use std::path::{Path, PathBuf};

/// Every `.cu` file in the repo's kernel tree.
pub fn cu_files() -> Vec<PathBuf> {
    fn visit(d: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                visit(&p, out);
            } else if p.extension().is_some_and(|x| x == "cu") {
                out.push(p);
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels");
    let mut files = Vec::new();
    visit(&root, &mut files);
    assert!(
        files.len() > 100,
        "kernel tree not found at {}",
        root.display()
    );
    files.sort();
    files
}

/// `kernels/…`-rooted path, for readable assertion messages.
pub fn rel(p: &Path) -> String {
    let s = p.to_string_lossy().into_owned();
    match s.split_once("kernels/") {
        Some((_, tail)) => format!("kernels/{tail}"),
        None => s,
    }
}

/// Drop `//` line comments. CUDA signatures carry commas INSIDE comments
/// (`// [K/2, N] transposed`), which is exactly how a naive parameter count
/// reads 11 where the compiler sees 9.
pub fn strip_line_comments(s: &str) -> String {
    s.lines()
        .map(|l| l.split_once("//").map_or(l, |(head, _)| head))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Index of the delimiter matching the one at `open` (which must be at `open`).
fn match_delim(s: &str, open: usize, o: char, c: char) -> Option<usize> {
    let mut depth = 0usize;
    for (i, ch) in s[open..].char_indices() {
        if ch == o {
            depth += 1;
        } else if ch == c {
            depth -= 1;
            if depth == 0 {
                return Some(open + i);
            }
        }
    }
    None
}

/// `(signature, body)` of the `__global__` kernel `name`, or `None` if this
/// file does not define it. `signature` spans `void NAME(` through its `)`;
/// `body` spans the matching braces.
pub fn kernel_sig_body(src: &str, name: &str) -> Option<(String, String)> {
    let pat = format!("void {name}(");
    let mut from = 0usize;
    while let Some(off) = src[from..].find(&pat) {
        let at = from + off;
        from = at + pat.len();
        // `extern "C" __global__` — or `__global__ __launch_bounds__(N, M)` —
        // immediately precedes. Anything further away is a call, not a decl.
        let mut lo = at.saturating_sub(160);
        while lo > 0 && !src.is_char_boundary(lo) {
            lo -= 1; // kernel headers are full of em-dashes; do not split one
        }
        if !src[lo..at].contains("__global__") {
            continue;
        }
        let open = at + pat.len() - 1;
        let close = match_delim(src, open, '(', ')')?;
        let sig = src[at..=close].to_string();
        let bopen = close + src[close..].find('{')?;
        let bclose = match_delim(src, bopen, '{', '}')?;
        return Some((sig, src[bopen..=bclose].to_string()));
    }
    None
}

/// Number of parameters in a kernel signature produced by `kernel_sig_body`.
pub fn param_count(sig: &str) -> usize {
    let sig = strip_line_comments(sig);
    let open = sig.find('(').unwrap_or(0);
    let close = sig.rfind(')').unwrap_or(sig.len());
    split_top_level(&sig[open + 1..close]).len()
}

/// Split on commas that sit at nesting depth zero, dropping blank trailing
/// entries (a trailing comma before `)` is idiomatic Rust).
fn split_top_level(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in s.chars() {
        match ch {
            '(' | '[' | '{' | '<' => {
                depth += 1;
                cur.push(ch);
            }
            ')' | ']' | '}' | '>' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut cur));
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out.retain(|a| !a.trim().is_empty());
    out
}

/// Every index expression `arr[...]` appearing in `body`, brackets stripped.
pub fn indexed_exprs(body: &str, arr: &str) -> Vec<String> {
    let body = strip_line_comments(body);
    let pat = format!("{arr}[");
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(off) = body[from..].find(&pat) {
        let open = from + off + pat.len() - 1;
        let Some(close) = match_delim(&body, open, '[', ']') else {
            break;
        };
        out.push(
            body[open + 1..close]
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
        );
        from = close + 1;
    }
    out
}

/// Does `expr` multiply by the bare identifier `N`? `* N_TILE` and `* NUM_x`
/// must NOT count — only the matrix width itself.
pub fn multiplies_by_bare_n(expr: &str) -> bool {
    let b = expr.as_bytes();
    for (i, &ch) in b.iter().enumerate() {
        if ch != b'*' {
            continue;
        }
        let mut j = i + 1;
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if b.get(j) != Some(&b'N') {
            continue;
        }
        let after = b.get(j + 1).copied().unwrap_or(b' ');
        if !after.is_ascii_alphanumeric() && after != b'_' {
            return true;
        }
    }
    false
}

/// Body of the Rust `fn name` in `src` (braces included), or `None`.
pub fn fn_body(src: &str, name: &str) -> Option<String> {
    let pat = format!("fn {name}(");
    let at = src.find(&pat)?;
    let params_open = at + pat.len() - 1;
    let params_close = match_delim(src, params_open, '(', ')')?;
    let bopen = params_close + src[params_close..].find('{')?;
    let bclose = match_delim(src, bopen, '{', '}')?;
    Some(src[bopen..=bclose].to_string())
}

/// How many kernel arguments a launcher packs — one per `.arg_*` builder call.
pub fn launcher_arg_count(src: &str, name: &str) -> Option<usize> {
    Some(fn_body(src, name)?.matches(".arg_").count())
}

/// Arguments of the first parenthesised group at/after the start of `s`.
pub fn call_args(s: &str) -> Vec<String> {
    let s = strip_line_comments(s);
    let Some(open) = s.find('(') else {
        return Vec::new();
    };
    let Some(close) = match_delim(&s, open, '(', ')') else {
        return Vec::new();
    };
    split_top_level(&s[open + 1..close])
}
