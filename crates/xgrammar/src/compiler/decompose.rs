// SPDX-License-Identifier: AGPL-3.0-only
//
// WGRAMMAR static/dynamic decomposition (Tier 3c).
//
// Implements the compile-time half of the "WGRAMMAR" technique
// (arXiv:2507.16768): a tool-call JSON-schema grammar is mostly FIXED
// SCAFFOLDING — the literal property keys, the structural punctuation
// (`{`, `"`, `:`, `,`, `}`), the skeleton — interleaved with a few
// CONTEXT-DEPENDENT value slots (the string / number / enum values the
// model fills in). The scaffolding is byte-for-byte identical for every
// request against a given schema, so the work of resolving it can be
// done ONCE, at compile time, instead of per-token at decode.
//
// INTRA-RULE DECOMPOSITION — WHY WHOLE-RULE IS NOT ENOUGH
// -------------------------------------------------------
// After grammar optimization a JSON-schema `root` rule is a SINGLE
// `Sequence` that interleaves both kinds of element, e.g.
//
//     root ::= "{" [ \n\t]* "\"x\"" [ \n\t]* ":" [ \n\t]* basic_integer …
//
// The scaffolding (`"{"`, `"\"x\""`, `":"`, …) and the value slots
// (the `[ \n\t]*` whitespace classes, the `basic_integer` rule ref)
// are NOT separate rules — they are siblings inside one body. So a
// whole-rule static/dynamic verdict would mark this rule "dynamic" and
// miss ~99% of it. WGRAMMAR's real granularity is the SEGMENT: a run
// of consecutive static `ByteString` elements inside a rule body is a
// static segment with a precomputed literal; everything between is a
// dynamic segment. This module decomposes each rule body that way.
//
// WHAT THIS ADDS OVER TIERS 2 / 3b — BE PRECISE
// ---------------------------------------------
// Tier 3b (`coalesce.rs` + `matcher/coalesce.rs`) already detects, at
// DECODE time, that a position is forced (`analyze_bitmask`) and walks
// the forced chain. Tier 2 (the JIT `mask_cache` + cross-grammar
// `RuleLevelCache`) already memoizes a per-state `AdaptiveTokenMask`
// the first time the matcher touches that state. So masks and chains
// are NOT recomputed per request once warm.
//
// What is still done LAZILY, and only when the matcher happens to reach
// a state, is the *discovery* of which spans are pure static
// scaffolding. Tier 3b answers "is THIS state forced?" one state at a
// time, on the hot path. WGRAMMAR's genuine delta is to answer, at
// COMPILE time and for the whole grammar at once, "which spans of which
// rule bodies are nothing but a fixed literal?" — and to precompute
// their literal byte sequence then. This module is that pass:
// `decompose_static_regions`.
//
// It is deliberately NOT a second masking mechanism. It computes byte
// sequences, not token masks; the matcher's actual forced-token
// decisions still flow through Tier 3b's `analyze_bitmask` over the
// authoritative `AdaptiveTokenMask`. The precomputed sequence is a
// compile-time INDEX of the scaffolding — it lets a scheduler know,
// before the first decode step, exactly which byte spans of the output
// are fixed, rather than rediscovering them token by token.
//
// CORRECTNESS
// -----------
// A segment is `Static` ONLY when it is a run of `ByteString` /
// `EmptyStr` elements — each admits exactly one byte string, so the
// precomputed bytes are read straight off the `ByteString` payloads,
// the same bytes the FSM scan would consume. Any element that is a
// `CharacterClass`, `CharacterClassStar`, `Choices`, `RuleRef`,
// `Repeat` or `TagDispatch` opens a `Dynamic` segment. The pass is
// pure and side-effect-free; it reads the already-optimized grammar
// and never mutates it.

use crate::grammar::{GrammarData, GrammarExprType};

/// One decomposed span of a rule body under WGRAMMAR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// A run of consecutive fixed-literal elements — pure scaffolding.
    /// The bytes are precomputed: this span emits exactly these bytes,
    /// in this order, every request. It is a forced-token chain known
    /// at compile time (Tier 3b would rediscover it lazily at decode).
    Static {
        /// The literal bytes this span emits, start to finish.
        bytes: Vec<u8>,
    },
    /// A genuine value slot — a character class, a number/string
    /// sub-grammar, a repeat, or a tag dispatch. Its continuation
    /// depends on the model's output and is not precomputed.
    Dynamic,
    /// A choice point: the model picks one branch. The branches are
    /// each decomposed in turn — a branch is itself mostly scaffolding
    /// (the non-empty-object branch of a JSON object opens with the
    /// fixed `{"key":` literal). The choice ITSELF is dynamic, but the
    /// per-branch scaffolding is still precomputed inside each
    /// `branches[i]`. This is WGRAMMAR's recursive decomposition: the
    /// static structure is exposed even when it sits behind a choice.
    Choice {
        /// One decomposition per branch, in grammar order.
        branches: Vec<RuleDecomposition>,
    },
}

impl Segment {
    /// True for a fixed-literal (scaffolding) segment.
    #[must_use]
    pub fn is_static(&self) -> bool {
        matches!(self, Segment::Static { .. })
    }

    /// The precomputed literal bytes, if this segment is a flat static
    /// run. A [`Segment::Choice`] returns `None` even though its
    /// branches may carry static spans — descend into `branches` for
    /// those.
    #[must_use]
    pub fn static_bytes(&self) -> Option<&[u8]> {
        match self {
            Segment::Static { bytes } => Some(bytes),
            Segment::Dynamic | Segment::Choice { .. } => None,
        }
    }
}

/// The decomposition of one rule body into alternating static / dynamic
/// segments, in body order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RuleDecomposition {
    segments: Vec<Segment>,
}

impl RuleDecomposition {
    /// The body's segments, in order. Adjacent segments always differ
    /// in kind (consecutive static elements are coalesced into one
    /// `Static` segment).
    #[must_use]
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// True when the whole rule body is a single fixed literal — the
    /// rule is pure scaffolding with no value slot or choice.
    #[must_use]
    pub fn is_fully_static(&self) -> bool {
        matches!(self.segments.as_slice(), [Segment::Static { .. }]) || self.segments.is_empty()
    }

    /// True when the rule body contains at least one static segment,
    /// recursing into [`Segment::Choice`] branches.
    #[must_use]
    pub fn has_static(&self) -> bool {
        self.segments.iter().any(|s| match s {
            Segment::Static { .. } => true,
            Segment::Dynamic => false,
            Segment::Choice { branches } => branches.iter().any(RuleDecomposition::has_static),
        })
    }

    /// Total precomputed static scaffolding bytes in this rule body,
    /// recursing into [`Segment::Choice`] branches.
    #[must_use]
    pub fn static_bytes_len(&self) -> usize {
        self.segments
            .iter()
            .map(|s| match s {
                Segment::Static { bytes } => bytes.len(),
                Segment::Dynamic => 0,
                Segment::Choice { branches } => branches
                    .iter()
                    .map(RuleDecomposition::static_bytes_len)
                    .sum(),
            })
            .sum()
    }

    /// Count of static / dynamic segments in this body, recursing into
    /// [`Segment::Choice`] branches. A choice contributes its branches'
    /// segments (the choice node itself is not counted as either).
    fn segment_counts(&self) -> (usize, usize) {
        let mut stat = 0;
        let mut dyn_ = 0;
        for s in &self.segments {
            match s {
                Segment::Static { .. } => stat += 1,
                Segment::Dynamic => dyn_ += 1,
                Segment::Choice { branches } => {
                    for b in branches {
                        let (bs, bd) = b.segment_counts();
                        stat += bs;
                        dyn_ += bd;
                    }
                }
            }
        }
        (stat, dyn_)
    }
}

/// The result of the WGRAMMAR static/dynamic decomposition pass over a
/// whole grammar — one [`RuleDecomposition`] per rule, indexed by
/// `rule_id`.
///
/// This is the compile-time "index of the scaffolding": it records,
/// before any decode step, which byte spans of which rule bodies are
/// fixed literals and what those literals are. It composes with Tier 3b
/// — every `Static` segment is exactly a forced-token chain — and with
/// Tier 2 — the JIT cache still owns the actual `AdaptiveTokenMask`s.
/// It adds no parallel masking path; it is a classification + byte
/// precompute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrammarDecomposition {
    rules: Vec<RuleDecomposition>,
}

impl GrammarDecomposition {
    /// The decomposition of rule `rule_id`'s body.
    #[must_use]
    pub fn rule_region(&self, rule_id: i32) -> &RuleDecomposition {
        &self.rules[rule_id as usize]
    }

    /// Every rule's decomposition, in `rule_id` order.
    #[must_use]
    pub fn regions(&self) -> &[RuleDecomposition] {
        &self.rules
    }

    /// Number of rules whose body is entirely fixed scaffolding.
    #[must_use]
    pub fn fully_static_rule_count(&self) -> usize {
        self.rules.iter().filter(|r| r.is_fully_static()).count()
    }

    /// Number of static segments across every rule body — the count of
    /// distinct scaffolding spans WGRAMMAR precomputes. Recurses into
    /// [`Segment::Choice`] branches.
    #[must_use]
    pub fn static_segment_count(&self) -> usize {
        self.rules.iter().map(|r| r.segment_counts().0).sum()
    }

    /// Number of dynamic segments across every rule body — the count of
    /// genuine value slots. Recurses into [`Segment::Choice`] branches.
    #[must_use]
    pub fn dynamic_segment_count(&self) -> usize {
        self.rules.iter().map(|r| r.segment_counts().1).sum()
    }

    /// Total precomputed static scaffolding bytes across the grammar —
    /// the byte count the matcher never has to discover at decode.
    #[must_use]
    pub fn total_static_bytes(&self) -> usize {
        self.rules
            .iter()
            .map(RuleDecomposition::static_bytes_len)
            .sum()
    }

    /// The fraction of segments that are static scaffolding, in
    /// `[0.0, 1.0]`. WGRAMMAR's premise is that the *bytes* are
    /// overwhelmingly scaffolding; the segment count is a coarser proxy.
    /// Returns `0.0` for a grammar with no segments.
    #[must_use]
    pub fn static_segment_fraction(&self) -> f64 {
        let total = self.static_segment_count() + self.dynamic_segment_count();
        if total == 0 {
            return 0.0;
        }
        self.static_segment_count() as f64 / total as f64
    }
}

/// Run the WGRAMMAR static/dynamic decomposition over an optimized
/// grammar.
///
/// For every rule, decompose its body into alternating
/// [`Segment::Static`] spans (fixed literals — the scaffolding, with
/// their bytes precomputed here, once) and [`Segment::Dynamic`] spans
/// (value slots). The grammar is only read; it is not mutated. The
/// result is the compile-time decomposition index.
///
/// This pass runs once per `compile_*` call, after grammar
/// optimization. It is cheap — a single linear walk of the AST — and
/// its output is stored on the [`super::CompiledGrammar`] so a
/// scheduler / matcher can consult the static/dynamic split without
/// any per-token work.
#[must_use]
pub fn decompose_static_regions(grammar: &GrammarData) -> GrammarDecomposition {
    let num_rules = grammar.num_rules();
    let mut rules = Vec::with_capacity(num_rules as usize);
    for rule_id in 0..num_rules {
        let body_id = grammar.rule(rule_id).body_expr_id;
        rules.push(decompose_rule_body(grammar, body_id));
    }
    GrammarDecomposition { rules }
}

/// Decompose one rule body into segments.
///
/// The body is one of: a `ByteString` / `EmptyStr` (one static
/// segment), a `Sequence` (decomposed element-by-element), a `Choices`
/// with a single branch (decompose that branch — degenerate but
/// possible post-optimization), or anything else (one dynamic segment).
fn decompose_rule_body(grammar: &GrammarData, body_id: i32) -> RuleDecomposition {
    let mut builder = SegmentBuilder::default();
    builder.push_expr(grammar, body_id, 0);
    RuleDecomposition {
        segments: builder.finish(),
    }
}

/// Maximum expression-tree depth the decomposition will descend. A
/// well-formed optimized grammar nests only a handful deep; this bound
/// is a defensive guard against a pathological / cyclic `Sequence`
/// nesting and never trips for real schemas. A span that would exceed
/// it is conservatively classified `Dynamic`.
const MAX_DECOMPOSE_DEPTH: u32 = 64;

/// Accumulates segments while walking a rule body, coalescing a run of
/// consecutive static elements into one [`Segment::Static`].
#[derive(Default)]
struct SegmentBuilder {
    segments: Vec<Segment>,
    /// Bytes of the static run currently being accumulated, if any.
    pending_static: Option<Vec<u8>>,
}

impl SegmentBuilder {
    /// Append a literal byte run, extending the pending static segment.
    fn push_static(&mut self, bytes: &[u8]) {
        self.pending_static
            .get_or_insert_with(Vec::new)
            .extend_from_slice(bytes);
    }

    /// Close any pending static run, then append one dynamic segment.
    fn push_dynamic(&mut self) {
        self.flush_static();
        self.segments.push(Segment::Dynamic);
    }

    /// Emit the pending static run as a `Static` segment, if non-empty.
    fn flush_static(&mut self) {
        if let Some(bytes) = self.pending_static.take() {
            self.segments.push(Segment::Static { bytes });
        }
    }

    /// Walk one expression, classifying it (and, for a `Sequence`, each
    /// of its children) into the running segment list.
    fn push_expr(&mut self, grammar: &GrammarData, expr_id: i32, depth: u32) {
        if depth >= MAX_DECOMPOSE_DEPTH {
            self.push_dynamic();
            return;
        }
        let expr = grammar.expr(expr_id);
        match expr.kind {
            // `EmptyStr` contributes nothing — it leaves the pending
            // static run open so neighbours still coalesce.
            GrammarExprType::EmptyStr => {}
            GrammarExprType::ByteString => {
                let bytes: Vec<u8> = expr.data.iter().map(|&b| b as u8).collect();
                self.push_static(&bytes);
            }
            GrammarExprType::Sequence => {
                for &child in expr.data {
                    self.push_expr(grammar, child, depth + 1);
                }
            }
            // A `Choices` is a divergence point. A single branch is
            // degenerate — descend into it. With several branches the
            // choice itself is dynamic, but each branch is decomposed
            // recursively so its own internal scaffolding (the fixed
            // `{"key":` prefix of a JSON object's non-empty branch) is
            // still exposed as static spans inside the branch. This is
            // WGRAMMAR's recursive decomposition.
            GrammarExprType::Choices => {
                if expr.data.len() == 1 {
                    self.push_expr(grammar, expr.data[0], depth + 1);
                } else {
                    self.flush_static();
                    let branches = expr
                        .data
                        .iter()
                        .map(|&branch| {
                            let mut sub = SegmentBuilder::default();
                            sub.push_expr(grammar, branch, depth + 1);
                            RuleDecomposition {
                                segments: sub.finish(),
                            }
                        })
                        .collect();
                    self.segments.push(Segment::Choice { branches });
                }
            }
            // Value slots: a character class, a sub-grammar reference,
            // a repeat, or a tag dispatch — each one dynamic segment.
            GrammarExprType::CharacterClass
            | GrammarExprType::CharacterClassStar
            | GrammarExprType::RuleRef
            | GrammarExprType::Repeat
            | GrammarExprType::TagDispatch => self.push_dynamic(),
        }
    }

    /// Finalize: close any trailing static run and return the segments.
    fn finish(mut self) -> Vec<Segment> {
        self.flush_static();
        self.segments
    }
}

#[cfg(test)]
#[path = "decompose_unit_tests.rs"]
mod tests;
