// SPDX-License-Identifier: AGPL-3.0-only
//
// FSMEdge — the edge type of a finite-state machine.
// Port of `struct FSMEdge` from xgrammar `cpp/fsm.h`.

/// Sentinel edge-type values stored in `FSMEdge::min`.
///
/// When `min >= 0` the edge is a character range `[min, max]`.
/// Negative values select a special edge type.
pub mod edge_type {
    /// `min >= 0` — a character range. This is the boundary value.
    pub const CHAR_RANGE: i16 = 0;
    /// Epsilon transition (no input consumed).
    pub const EPSILON: i16 = -1;
    /// Reference to another rule; `max` holds the rule id.
    pub const RULE_REF: i16 = -2;
    /// Accepts an end-of-sequence token.
    pub const EOS: i16 = -3;
    /// Repeated rule reference; `max` indexes into `edge_aux_data`.
    pub const REPEAT_REF: i16 = -4;
}

/// The largest character value an edge range may carry.
pub const MAX_CHAR: i16 = 255;

/// An edge of a finite-state machine.
///
/// `min`/`max` encode the edge kind (see [`edge_type`]); `target` is the
/// destination state id. The C++ struct is `alignas(8)` and packs
/// `(i16, i16, i32)` — this layout is preserved so the compact
/// representation matches the C++ memory layout byte-for-byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FsmEdge {
    /// Edge-kind discriminant / range lower bound. See [`edge_type`].
    pub min: i16,
    /// Range upper bound, rule id, or aux-data index depending on `min`.
    pub max: i16,
    /// Destination state id.
    pub target: i32,
}

impl FsmEdge {
    /// Construct an edge. Panics if a char-range edge has `min > max`,
    /// matching the C++ `XGRAMMAR_DCHECK`.
    pub fn new(min: i16, max: i16, target: i32) -> Self {
        debug_assert!(
            !(min >= edge_type::CHAR_RANGE && min > max),
            "Invalid FsmEdge: min > max. min={min}, max={max}"
        );
        Self { min, max, target }
    }

    /// True if this is a character-range edge.
    pub fn is_char_range(&self) -> bool {
        self.min >= edge_type::CHAR_RANGE
    }

    /// True if this is an epsilon transition.
    pub fn is_epsilon(&self) -> bool {
        self.min == edge_type::EPSILON
    }

    /// True if this is a rule reference.
    pub fn is_rule_ref(&self) -> bool {
        self.min == edge_type::RULE_REF
    }

    /// True if this is an EOS transition.
    pub fn is_eos(&self) -> bool {
        self.min == edge_type::EOS
    }

    /// True if this is a repeat reference.
    pub fn is_repeat_ref(&self) -> bool {
        self.min == edge_type::REPEAT_REF
    }

    /// True if the edge uses `edge_aux_data` (currently only repeat refs).
    pub fn is_aux_edge(&self) -> bool {
        self.is_repeat_ref()
    }

    /// Referenced rule id, or `-1` if this is not a rule reference.
    pub fn ref_rule_id(&self) -> i32 {
        if self.is_rule_ref() {
            self.max as i32
        } else {
            -1
        }
    }

    /// Index into `edge_aux_data`, or `-1` if not a repeat reference.
    pub fn aux_index(&self) -> i16 {
        if self.is_repeat_ref() { self.max } else { -1 }
    }
}

// FSMEdge's C++ `operator<` sorts by the tuple (min, max, target).
impl PartialOrd for FsmEdge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FsmEdge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.min, self.max, self.target).cmp(&(other.min, other.max, other.target))
    }
}

/// A view into `edge_aux_data` for a repeat edge.
///
/// Layout in the backing slice is `[rule_id, lower, upper]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepeatEdgeRef {
    /// The rule being repeated.
    pub rule_id: i16,
    /// Minimum repeat count.
    pub lower: i32,
    /// Maximum repeat count (`-1` for unbounded).
    pub upper: i32,
}

impl RepeatEdgeRef {
    /// Build a [`RepeatEdgeRef`] from the 3-element aux slice starting at `idx`.
    pub fn from_aux(data: &[i32], idx: usize) -> Self {
        Self {
            rule_id: data[idx] as i16,
            lower: data[idx + 1],
            upper: data[idx + 2],
        }
    }
}

/// Compare two edges by `(min, max)` only — used to deduplicate ranges.
/// Mirrors the C++ `FSMEdgeRangeComparator`.
pub fn cmp_edge_range(lhs: &FsmEdge, rhs: &FsmEdge) -> std::cmp::Ordering {
    (lhs.min, lhs.max).cmp(&(rhs.min, rhs.max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_range_predicates() {
        let e = FsmEdge::new(b'a' as i16, b'z' as i16, 3);
        assert!(e.is_char_range());
        assert!(!e.is_epsilon());
        assert!(!e.is_rule_ref());
        assert!(!e.is_eos());
        assert!(!e.is_repeat_ref());
        assert!(!e.is_aux_edge());
        assert_eq!(e.ref_rule_id(), -1);
        assert_eq!(e.aux_index(), -1);
    }

    #[test]
    fn epsilon_edge() {
        let e = FsmEdge::new(edge_type::EPSILON, 0, 7);
        assert!(e.is_epsilon());
        assert!(!e.is_char_range());
    }

    #[test]
    fn rule_ref_edge() {
        let e = FsmEdge::new(edge_type::RULE_REF, 5, 2);
        assert!(e.is_rule_ref());
        assert_eq!(e.ref_rule_id(), 5);
    }

    #[test]
    fn eos_edge() {
        let e = FsmEdge::new(edge_type::EOS, 0, 1);
        assert!(e.is_eos());
    }

    #[test]
    fn repeat_ref_edge() {
        let e = FsmEdge::new(edge_type::REPEAT_REF, 9, 4);
        assert!(e.is_repeat_ref());
        assert!(e.is_aux_edge());
        assert_eq!(e.aux_index(), 9);
    }

    #[test]
    fn ordering_is_tuple_lexicographic() {
        let a = FsmEdge::new(1, 2, 3);
        let b = FsmEdge::new(1, 2, 4);
        let c = FsmEdge::new(1, 3, 0);
        assert!(a < b);
        assert!(b < c);
        let mut v = vec![c, b, a];
        v.sort();
        assert_eq!(v, vec![a, b, c]);
    }

    #[test]
    fn range_comparator_ignores_target() {
        let a = FsmEdge::new(1, 2, 99);
        let b = FsmEdge::new(1, 2, 0);
        assert_eq!(cmp_edge_range(&a, &b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn repeat_edge_ref_from_aux() {
        let data = vec![0, 0, 0, 7, 2, 5];
        let r = RepeatEdgeRef::from_aux(&data, 3);
        assert_eq!(r.rule_id, 7);
        assert_eq!(r.lower, 2);
        assert_eq!(r.upper, 5);
    }
}
