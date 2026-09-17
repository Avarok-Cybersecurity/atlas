// SPDX-License-Identifier: AGPL-3.0-only

//! Token-level trie for constraining decoding to an enumerable set of
//! symbol strings (B.1, 2026-04-25).
//!
//! Reference: PackMonitor (arXiv:2602.20717). When the set of legal
//! values at a decode position is finite — declared MCP tool names,
//! known workspace file paths, public function symbols from
//! rust-analyzer — building a DFA over their **token-level** encodings
//! and AND-ing its vocab mask with XGrammar's structural mask
//! eliminates 100% of out-of-vocabulary fabrications. The model
//! cannot emit `axum::route_with_layer` if that token sequence isn't a
//! valid prefix in the trie.
//!
//! ## Scope
//!
//! This module ships the **runtime primitive**: build a trie from a
//! list of strings + a tokenizer, then query `allowed_next_tokens` to
//! get a vocab mask per position. Wiring into request paths is opt-in
//! per future client extensions:
//!
//!   - Tool-name enforcement is already covered by XGrammar's
//!     structural-tag `<function={NAME}>` begin-pattern, so the trie
//!     overlaps there. It's most useful for **argument values** typed
//!     as path/symbol where XGrammar treats them as plain strings.
//!   - Workspace symbol enforcement requires the client to declare
//!     the allowed set (no standard API today). Atlas exposes the
//!     primitive so MCP servers / opencode / Claude Code can adopt it
//!     once a convention emerges.
//!
//! ## Invariants
//!
//! - The trie operates on **token IDs**, not characters. Strings are
//!   tokenised once at construction.
//! - `terminal` flag marks positions where a complete symbol ends —
//!   needed because some symbols are prefixes of others (e.g., `read`
//!   vs `read_file`).
//! - Empty input list → trie that allows nothing (caller should skip
//!   the AND-mask in that case to avoid blocking generation).
//! - `allowed_next_tokens` is `O(branching_factor)` per call; the
//!   number of children at any node is bounded by tokenizer vocab,
//!   typically ≤ 50 in practice.

use std::collections::HashMap;

/// A token-level trie over a closed set of symbol strings.
#[derive(Debug, Clone)]
pub struct SymbolTrie {
    nodes: Vec<TrieNode>,
}

#[derive(Debug, Clone)]
struct TrieNode {
    /// Children by next token ID.
    children: HashMap<u32, usize>,
    /// True if a complete symbol ends at this node.
    terminal: bool,
}

/// Public construction interface — the caller supplies the tokeniser.
pub trait Tokeniser {
    /// Encode `s` to a sequence of token IDs. Implementations must
    /// match the model's tokeniser (incl. BPE merges + BOS/EOS rules)
    /// EXACTLY — a mismatch silently allows the model to emit
    /// "valid" tokens that don't decode to a trie-legal symbol.
    fn encode(&self, s: &str) -> Vec<u32>;
}

impl SymbolTrie {
    /// Build a trie from the given symbol strings. Returns `None` when
    /// the symbol list is empty (caller should not engage the AND-mask).
    pub fn build<T: Tokeniser>(symbols: &[&str], tokeniser: &T) -> Option<Self> {
        if symbols.is_empty() {
            return None;
        }
        let mut nodes = vec![TrieNode {
            children: HashMap::new(),
            terminal: false,
        }];
        for sym in symbols {
            let toks = tokeniser.encode(sym);
            if toks.is_empty() {
                continue;
            }
            let mut cur = 0usize;
            for t in &toks {
                let next = nodes[cur].children.get(t).copied();
                cur = match next {
                    Some(idx) => idx,
                    None => {
                        let new_idx = nodes.len();
                        nodes.push(TrieNode {
                            children: HashMap::new(),
                            terminal: false,
                        });
                        nodes[cur].children.insert(*t, new_idx);
                        new_idx
                    }
                };
            }
            nodes[cur].terminal = true;
        }
        if nodes.len() == 1 {
            // Every symbol tokenised to empty — degenerate.
            return None;
        }
        Some(Self { nodes })
    }

    /// Return `Some(node_id)` after walking `prefix_tokens` from root.
    /// Returns `None` if the prefix has fallen off the trie (no more
    /// valid extensions). Callers walking a streaming generation pass
    /// this as the running state.
    pub fn walk_prefix(&self, prefix_tokens: &[u32]) -> Option<usize> {
        let mut cur = 0usize;
        for &t in prefix_tokens {
            cur = *self.nodes[cur].children.get(&t)?;
        }
        Some(cur)
    }

    /// Compute the vocab mask of allowed next tokens at `node_id`.
    /// Returns a sorted, deduped Vec — the caller intersects this
    /// with XGrammar's mask via AND.
    ///
    /// If the node is terminal AND has no children, returns an empty
    /// vec (the symbol is complete; downstream grammar must close
    /// the structural tag).
    pub fn allowed_next_tokens(&self, node_id: usize) -> Vec<u32> {
        let node = &self.nodes[node_id];
        let mut out: Vec<u32> = node.children.keys().copied().collect();
        out.sort_unstable();
        out
    }

    /// Check whether the given prefix marks a complete symbol —
    /// caller uses this to decide whether emitting the structural
    /// close tag is currently valid.
    pub fn is_terminal(&self, node_id: usize) -> bool {
        self.nodes[node_id].terminal
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whitespace-splitting test tokeniser: each space-separated word
    /// hashes to one synthetic token id (1-indexed). Stable across
    /// calls so different strings sharing a word share a token.
    struct WordTokeniser;
    impl Tokeniser for WordTokeniser {
        fn encode(&self, s: &str) -> Vec<u32> {
            // Map each word to an id derived from its bytes.
            s.split_whitespace()
                .map(|w| {
                    let mut h: u32 = 1;
                    for b in w.bytes() {
                        h = h.wrapping_mul(31).wrapping_add(b as u32);
                    }
                    h.max(1) // 0 reserved as "no token"
                })
                .collect()
        }
    }

    #[test]
    fn empty_symbols_returns_none() {
        let trie = SymbolTrie::build(&[], &WordTokeniser);
        assert!(trie.is_none());
    }

    #[test]
    fn single_symbol_walks_to_terminal() {
        let trie = SymbolTrie::build(&["foo bar baz"], &WordTokeniser).unwrap();
        let toks = WordTokeniser.encode("foo bar baz");
        let node = trie.walk_prefix(&toks).expect("walk hits terminal");
        assert!(trie.is_terminal(node));
        let next = trie.allowed_next_tokens(node);
        assert!(next.is_empty(), "no extensions past terminal");
    }

    #[test]
    fn two_symbols_share_prefix() {
        let trie = SymbolTrie::build(&["read file", "read dir"], &WordTokeniser).unwrap();
        let read_toks = WordTokeniser.encode("read");
        let read_node = trie.walk_prefix(&read_toks).expect("walk hits 'read'");
        // Not terminal — both symbols extend.
        assert!(!trie.is_terminal(read_node));
        let allowed = trie.allowed_next_tokens(read_node);
        assert_eq!(allowed.len(), 2, "two extensions: file + dir");
    }

    #[test]
    fn prefix_overlap_keeps_terminal_at_inner_node() {
        // "read" itself is a symbol AND a prefix of "read file".
        let trie = SymbolTrie::build(&["read", "read file"], &WordTokeniser).unwrap();
        let read_toks = WordTokeniser.encode("read");
        let read_node = trie.walk_prefix(&read_toks).expect("walk hits 'read'");
        assert!(trie.is_terminal(read_node), "'read' alone is terminal");
        let next = trie.allowed_next_tokens(read_node);
        assert_eq!(next.len(), 1, "only 'file' extends");
    }

    #[test]
    fn off_trie_prefix_returns_none() {
        let trie = SymbolTrie::build(&["foo bar"], &WordTokeniser).unwrap();
        let bogus = WordTokeniser.encode("foo wrong");
        let res = trie.walk_prefix(&bogus);
        assert!(res.is_none(), "off-trie prefix has no node");
    }

    #[test]
    fn allowed_next_tokens_is_sorted() {
        let trie = SymbolTrie::build(&["zebra", "alpha", "delta"], &WordTokeniser).unwrap();
        let root_allowed = trie.allowed_next_tokens(0);
        // Should be sorted ascending — caller can binary-search the AND-mask.
        for window in root_allowed.windows(2) {
            assert!(window[0] < window[1], "tokens must be sorted ascending");
        }
    }
}
