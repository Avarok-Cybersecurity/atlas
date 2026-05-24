// SPDX-License-Identifier: AGPL-3.0-only
//
// Speculative fast-accept calculation for the mask generator. Port of
// `GrammarMatcherForTokenMaskCache::GetSpeculativeCalculation` and the
// inline fast-accept branch of `GetTokenMaskWithFirstCharacterCheck`
// from `cpp/grammar_compiler.cc`.
//
// For self-recursive rules (e.g. JSON-string content) and TagDispatch
// rules, a token consisting only of "safe" bytes can be accepted
// without running it through the parser.

use crate::grammar::GrammarExprType;

use super::MaskGenerator;

impl MaskGenerator<'_> {
    /// Decide whether speculative fast-accept applies for the seeded
    /// state, and which first bytes are "safe".
    ///
    /// Returns `(applicable, safe_byte_mask)`. Port of
    /// `GetSpeculativeCalculation`.
    pub(super) fn speculative_calculation(&self) -> (bool, [bool; 256]) {
        let mut mask = [false; 256];
        let body_id = self.grammar.rule(self.init_rule_id).body_expr_id;
        let fsm = self.grammar.per_rule_fsms[self.init_rule_id as usize]
            .as_ref()
            .expect("rule must have a per-rule FSM");

        if self.grammar.expr(body_id).kind == GrammarExprType::TagDispatch {
            // TagDispatch: bytes that transit back to the AC start.
            for edge in fsm.fsm().edges(self.init_state.element_id as usize) {
                if edge.target as usize != fsm.start() || !edge.is_char_range() {
                    continue;
                }
                for c in edge.min..=edge.max {
                    mask[c as usize] = true;
                }
            }
            return (true, mask);
        }

        // Non-tag rule: detect a self-recursive-like initial state.
        let elem = self.init_state.element_id;
        let mut applicable = false;
        for edge in fsm.fsm().edges(elem as usize) {
            if !edge.is_char_range() {
                continue;
            }
            // Case A: a self-loop char edge.
            if edge.target == elem {
                applicable = true;
                for c in edge.min..=edge.max {
                    mask[c as usize] = true;
                }
                continue;
            }
            // Case B: from the start state, an edge into a state that
            // recursively calls this same rule.
            if fsm.start() as i32 == elem {
                for next in fsm.fsm().edges(edge.target as usize) {
                    let recurses = (next.is_rule_ref() && next.ref_rule_id() == self.init_rule_id)
                        || (next.is_repeat_ref()
                            && fsm.fsm().repeat_edge_info(next.aux_index()).rule_id as i32
                                == self.init_rule_id);
                    if recurses {
                        applicable = true;
                        for c in edge.min..=edge.max {
                            mask[c as usize] = true;
                        }
                        break;
                    }
                }
            }
        }
        (applicable, mask)
    }

    /// Try to fast-accept `token` (sorted-vocab index `index`) without
    /// the parser. Returns whether it was accepted and pushed.
    ///
    /// Port of the speculative branch inside the scan loop.
    pub(super) fn try_speculative_accept(
        &mut self,
        token: &[u8],
        index: i32,
        spec_mask: &[bool; 256],
        definite_bitset: Option<&[bool]>,
    ) -> bool {
        if let Some(bitset) = definite_bitset {
            // TagDispatch optimization.
            if token.is_empty() {
                self.accepted.push(index);
                return true;
            }
            if spec_mask[token[0] as usize] && bitset[index as usize] {
                self.accepted.push(index);
                return true;
            }
            false
        } else {
            // Self-recursive rule: every byte must be a safe ASCII byte.
            let all_safe = token.iter().all(|&b| b.is_ascii() && spec_mask[b as usize]);
            if all_safe {
                self.accepted.push(index);
                return true;
            }
            false
        }
    }
}
