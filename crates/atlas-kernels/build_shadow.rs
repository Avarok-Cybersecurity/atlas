// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel entry-point resolution for the shadow-drift detector.
//!
//! Shadowing is whole-file: a model's `kernels/{hw}/{model}/{quant}/foo.cu`
//! replaces `kernels/{hw}/common/foo.cu` entirely. When `common/` later gains a
//! kernel the fork never picked up, that kernel silently vanishes for that
//! model. Comparing the two files' entry points is what turns that into a
//! reported defect instead of a handle-0 lookup miss at runtime.
//!
//! ## Why this is not a `__global__` grep
//!
//! It used to be, and the grep had a hole big enough to hide the exact bug the
//! detector exists to catch. Twenty-one `common/*.cu` files contain no literal
//! `__global__` at all — they are configuration headers that bind a name and
//! delegate:
//!
//! ```cuda
//! #define KERNEL_NAME inferspark_prefill_paged_fp8
//! #include "prefill_paged_compute.cuh"     // declares KERNEL_NAME and KERNEL_NAME##_64
//! ```
//!
//! A text scan of that file returns the EMPTY SET, so every comparison against
//! it reported "drops nothing" — indistinguishable, in the build log, from a
//! shadow that really was complete. Dropping one of those kernels was
//! undetectable by construction.
//!
//! So this resolves what the file actually declares. It follows quoted
//! `#include`s, tracks `#define`s, and expands the name expression in each
//! declaration through them, covering the three indirections in the tree:
//!
//! * object-like binding — `#define KERNEL_NAME foo` read by an included
//!   header;
//! * argument paste — `CONCAT(KERNEL_NAME, _64)`, which the paged-prefill
//!   headers use to emit a second BR=64 entry point beside the primary one;
//! * instantiation macro — `#define WYN(K) ... __global__ void name##K(...)`
//!   invoked as `WYN(5)`, which declares one kernel per invocation.
//!
//! It also understands Metal's `kernel void name(...)`, so the 83 entry points
//! under `kernels/metal/common/` stop resolving to nothing the moment a Metal
//! model dir grows its first shadow.
//!
//! It is still a source-level scan — it runs BEFORE nvcc and cannot depend on
//! compiled artifacts — but it resolves the directives that decide which entry
//! points a file declares, rather than assuming they are absent.
//!
//! Deliberately NOT implemented: conditional compilation. `#if`/`#ifdef` are
//! ignored and both arms contribute. That over-reports rather than under-
//! reports — a kernel listed as dropped that is actually compiled shows up as
//! a build warning to investigate, whereas a kernel missed is the silent
//! failure this file exists to prevent.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// How deep quoted `#include`s are followed. The real chains are two levels
/// (`shadow.cu` -> `common/foo.cu` -> `compute.cuh`); the cap only bounds a
/// pathological tree, since revisits are already suppressed.
const MAX_INCLUDE_DEPTH: usize = 8;

/// How many times a name is rewritten through `#define` before giving up, so a
/// mutually recursive pair of defines cannot spin.
const MAX_MACRO_HOPS: usize = 8;

/// A `__global__` declaration's name as written, before macro expansion.
#[derive(Debug, PartialEq, Eq)]
enum NameExpr {
    /// `__global__ void foo(` or `__global__ void KERNEL_NAME(`.
    Plain(String),
    /// `__global__ void CONCAT(KERNEL_NAME, _64)(` — a function-like macro
    /// whose arguments are pasted together. Only the arguments matter; which
    /// concat macro spelled the paste does not.
    Paste(Vec<String>),
}

/// Accumulated preprocessor state for one root file and everything it includes.
#[derive(Default)]
struct Scan {
    /// Object-like `#define NAME VALUE`. First definition wins, which matches
    /// the real order: a `.cu` binds `KERNEL_NAME` and only then includes the
    /// header that reads it.
    defines: BTreeMap<String, String>,
    /// Unexpanded declaration names, in encounter order.
    decls: Vec<NameExpr>,
}

/// A function-like `#define NAME(a, b) <body>`, with line continuations joined.
/// Only macros whose body declares a kernel are kept — the rest bind nothing
/// this resolver needs and there are a great many of them.
struct FnMacro {
    params: Vec<String>,
    body: String,
}

/// How the source dialect of a ROOT kernel file spells an entry-point
/// declaration.
///
/// The two must not be merged into one keyword list. `__global__` is a reserved
/// specifier that appears nowhere else, so it can be matched anywhere in the
/// text. `kernel` is an ordinary English word that shows up constantly in CUDA
/// comments ("the kernel dequantizes ..."), and matching it there invents entry
/// points named after the next word. So it is recognised only in `.metal`
/// sources, and only at the start of a line, which is how all 83 Metal entry
/// points are written.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dialect {
    /// CUDA / HIP: `extern "C" __global__ void name(...)`, matched anywhere.
    Cuda,
    /// Metal: `kernel void name(...)`, matched at line start only.
    Metal,
}

impl Dialect {
    fn of(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("metal") => Dialect::Metal,
            _ => Dialect::Cuda,
        }
    }

    fn keyword(self) -> &'static str {
        match self {
            Dialect::Cuda => "__global__",
            Dialect::Metal => "kernel",
        }
    }
}

/// The kernel entry points `file` declares, following quoted includes.
///
/// Returns the empty set for an unreadable file, matching the previous
/// behaviour: a missing common namesake means "nothing to drop", not a panic
/// in a build script.
pub fn entry_points(file: &Path) -> BTreeSet<String> {
    let mut scan = Scan::default();
    let mut seen = BTreeSet::new();
    walk(file, Dialect::of(file), 0, &mut seen, &mut scan);
    let mut out = BTreeSet::new();
    for decl in &scan.decls {
        if let Some(name) = resolve(decl, &scan.defines) {
            out.insert(name);
        }
    }
    out
}

/// Every kernel entry point `model_file` drops relative to `common_file`.
/// Sorted, for stable build warnings.
pub fn shadowed_missing_symbols(common_file: &Path, model_file: &Path) -> Vec<String> {
    entry_points(common_file)
        .difference(&entry_points(model_file))
        .cloned()
        .collect()
}

/// Read one file into `scan` and recurse into its quoted includes.
fn walk(
    path: &Path,
    dialect: Dialect,
    depth: usize,
    seen: &mut BTreeSet<PathBuf>,
    scan: &mut Scan,
) {
    if depth > MAX_INCLUDE_DEPTH {
        return;
    }
    // Canonicalize so a relative `../../common/x.cu` and a direct path to the
    // same file are one entry. Fall back to the path as given when the file
    // does not exist — `read_to_string` below then fails and this is a no-op.
    let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if !seen.insert(key) {
        return;
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        return;
    };
    let dir = path.parent().unwrap_or(Path::new("."));

    let mut includes = Vec::new();
    let mut fn_macros: BTreeMap<String, FnMacro> = BTreeMap::new();
    let mut logical = String::new();
    for line in text.lines() {
        // Join `\`-continued lines into one logical directive before parsing:
        // an instantiation macro's whole body is one `#define`.
        logical.push_str(line.trim_end().trim_end_matches('\\'));
        if line.trim_end().ends_with('\\') {
            logical.push('\n');
            continue;
        }
        let directive = std::mem::take(&mut logical);
        let trimmed = directive.trim_start();
        let Some(rest) = trimmed.strip_prefix('#') else {
            continue;
        };
        let rest = rest.trim_start();
        if let Some(body) = rest.strip_prefix("define") {
            record_define(body, dialect, &mut scan.defines, &mut fn_macros);
        } else if let Some(body) = rest.strip_prefix("include")
            && let Some(rel) = quoted_include(body)
        {
            includes.push(dir.join(rel));
        }
    }

    collect_decls(&text, dialect, &mut scan.decls);
    collect_instantiations(&text, dialect, &fn_macros, &mut scan.decls);

    for inc in includes {
        // The dialect is the ROOT file's, not the include's: a `.cu` pulls in
        // `.cuh`, a `.metal` pulls in `.h`, and an extension-based guess would
        // read a Metal header as CUDA and find nothing in it.
        walk(&inc, dialect, depth + 1, seen, scan);
    }
}

/// Kernels declared inside a function-like macro, one per invocation site.
///
/// `#define WYN(K) extern "C" __global__ void gated_delta_rule_wy##K(...)`
/// followed by `WYN(5)` declares `gated_delta_rule_wy5`. Scanning the raw text
/// finds the `__global__` but its name still carries the `##K` paste, so it
/// resolves to nothing — the same "returns empty, looks thorough" shape as the
/// `#define KERNEL_NAME` case. Substituting the invocation's arguments and
/// dropping the paste operators makes the name concrete.
fn collect_instantiations(
    text: &str,
    dialect: Dialect,
    fn_macros: &BTreeMap<String, FnMacro>,
    out: &mut Vec<NameExpr>,
) {
    for (name, mac) in fn_macros {
        for args in invocations(text, name) {
            if args.len() != mac.params.len() {
                continue;
            }
            let mut body = mac.body.clone();
            for (param, arg) in mac.params.iter().zip(&args) {
                body = replace_word(&body, param, arg);
            }
            collect_decls(&body.replace("##", ""), dialect, out);
        }
    }
}

/// Argument lists of every `NAME(...)` call in `text` that is not the `#define`
/// or `#undef` of `NAME` itself.
fn invocations(text: &str, name: &str) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for (idx, _) in text.match_indices(name) {
        if idx > 0 && is_ident_char(text.as_bytes()[idx - 1]) {
            continue;
        }
        let line_start = text[..idx].rfind('\n').map_or(0, |n| n + 1);
        let head = text[line_start..idx].trim_start();
        if head.starts_with("#define") || head.starts_with("#undef") || head.starts_with('#') {
            continue;
        }
        let after = idx + name.len();
        if !text[after..].starts_with('(') {
            continue;
        }
        let Some(close) = balanced_end(text, after) else {
            continue;
        };
        out.push(
            text[after + 1..close - 1]
                .split(',')
                .map(|a| a.trim().to_string())
                .collect(),
        );
    }
    out
}

/// Replace whole-word occurrences of `word` in `text`.
fn replace_word(text: &str, word: &str, with: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    for (idx, _) in text.match_indices(word) {
        let before_ok = idx == 0 || !is_ident_char(bytes[idx - 1]);
        let end = idx + word.len();
        let after_ok = end >= bytes.len() || !is_ident_char(bytes[end]);
        if before_ok && after_ok {
            out.push_str(&text[last..idx]);
            out.push_str(with);
            last = end;
        }
    }
    out.push_str(&text[last..]);
    out
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Record one `#define`. Object-like defines become name bindings;
/// function-like ones are kept only when their body declares a kernel, which is
/// the instantiation-macro case `collect_instantiations` expands.
fn record_define(
    body: &str,
    dialect: Dialect,
    defines: &mut BTreeMap<String, String>,
    fn_macros: &mut BTreeMap<String, FnMacro>,
) {
    let body = body.strip_prefix([' ', '\t']).unwrap_or(body);
    let name_len = body
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .unwrap_or(body.len());
    if name_len == 0 {
        return;
    }
    let (name, rest) = body.split_at(name_len);
    if rest.starts_with('(') {
        let Some(close) = balanced_end(rest, 0) else {
            return;
        };
        let macro_body = rest[close..].to_string();
        if !macro_body.contains(dialect.keyword()) {
            return;
        }
        let params = rest[1..close - 1]
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        fn_macros.insert(
            name.to_string(),
            FnMacro {
                params,
                body: macro_body,
            },
        );
        return;
    }
    let value = rest.trim().to_string();
    defines.entry(name.to_string()).or_insert(value);
}

/// The path out of `#include "foo/bar.cuh"`. Angle-bracket includes are system
/// headers and are not followed.
fn quoted_include(body: &str) -> Option<&str> {
    let body = body.trim_start();
    let rest = body.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(&rest[..end])
}

/// Every entry-point declaration's name expression, as written.
///
/// `kernel` is a plain word, so unlike `__global__` it needs a whole-word check
/// to avoid firing on `kernel_handle` or the word in a comment. Requiring the
/// following token to be a type keeps it to declarations.
fn collect_decls(text: &str, dialect: Dialect, out: &mut Vec<NameExpr>) {
    let bytes = text.as_bytes();
    let keyword = dialect.keyword();
    for (idx, _) in text.match_indices(keyword) {
        let end = idx + keyword.len();
        if idx > 0 && is_ident_char(bytes[idx - 1]) {
            continue;
        }
        let rest = &text[end..];
        if !rest.starts_with([' ', '\t', '\n']) {
            continue;
        }
        if dialect == Dialect::Metal && !text[..idx].trim_end_matches([' ', '\t']).ends_with('\n') {
            continue;
        }
        if let Some(expr) = name_expr(rest) {
            out.push(expr);
        }
    }
}

/// Parse the declarator that follows `__global__`, stepping over an interposed
/// `__launch_bounds__(...)` and recognising a token-pasting macro call.
fn name_expr(tail: &str) -> Option<NameExpr> {
    let mut pos = 0usize;
    for _ in 0..MAX_MACRO_HOPS {
        let open = pos + tail.get(pos..)?.find('(')?;
        let ident = last_ident(&tail[pos..open])?;
        let close = balanced_end(tail, open)?;
        if ident == "__launch_bounds__" {
            pos = close;
            continue;
        }
        // `NAME(...)(`  -> the first paren list is a macro argument list and
        // the second is the kernel's parameter list, so NAME is a paste macro.
        // `NAME(...)`   -> NAME is the kernel and that list is its parameters.
        if tail[close..].trim_start().starts_with('(') {
            let args = tail[open + 1..close - 1]
                .split(',')
                .map(|a| a.trim().to_string())
                .filter(|a| !a.is_empty())
                .collect::<Vec<_>>();
            return (!args.is_empty()).then_some(NameExpr::Paste(args));
        }
        return Some(NameExpr::Plain(ident.to_string()));
    }
    None
}

/// The trailing identifier of a declarator head such as `void *` + name.
fn last_ident(head: &str) -> Option<&str> {
    let token = head.split_whitespace().last()?;
    let token = token.trim_start_matches('*');
    is_ident(token).then_some(token)
}

/// Index just past the `)` matching the `(` at `open`.
fn balanced_end(s: &str, open: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    for (i, b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Expand a name expression to the identifier nvcc will actually emit.
fn resolve(expr: &NameExpr, defines: &BTreeMap<String, String>) -> Option<String> {
    match expr {
        NameExpr::Plain(name) => expand(name, defines),
        NameExpr::Paste(args) => {
            let mut out = String::new();
            for arg in args {
                out.push_str(&expand(arg, defines)?);
            }
            is_ident(&out).then_some(out)
        }
    }
}

/// Rewrite `token` through the object-like defines until it is not one.
///
/// A paste argument such as `_64` is not a macro and not a valid identifier on
/// its own, so validity is checked by the caller for pastes and here only for
/// the whole-name case.
fn expand(token: &str, defines: &BTreeMap<String, String>) -> Option<String> {
    let mut cur = token.to_string();
    for _ in 0..MAX_MACRO_HOPS {
        match defines.get(&cur) {
            Some(next) if next != &cur => cur = next.trim().to_string(),
            _ => break,
        }
    }
    (!cur.is_empty() && cur.chars().all(|c| c.is_alphanumeric() || c == '_')).then_some(cur)
}

/// A C identifier: non-empty, alphanumeric/underscore, not digit-initial.
fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
        && !s.starts_with(|c: char| c.is_ascii_digit())
}
