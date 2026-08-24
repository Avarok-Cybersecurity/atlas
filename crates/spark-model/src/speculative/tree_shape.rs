// SPDX-License-Identifier: AGPL-3.0-only

//! Spine+hedge draft-tree shapes for tree speculative decoding (Phase 1 of
//! the tree-spec plan).
//!
//! A tree is a **spine** (the drafter's top-1 chain, exactly today's chain
//! draft) plus **hedge leaves**: rank-2..k siblings of spine nodes with no
//! children. The constraint is load-bearing — every node's ancestors are
//! spine nodes, so per-row attention visibility is "committed prefix +
//! spine(1..d-1) + self", the drafter KV stays a linear chain, and GDN
//! verification decomposes into one spine pass + one 1-token pass per hedge.
//!
//! Shape notation (`ATLAS_TREE_SHAPE="1,2,2,2"`): per-depth node counts;
//! depth d contributes 1 spine node + (count_d - 1) hedges. Verify width
//! M = 1 (root row) + total nodes.
//!
//! Row layout (fixed): row 0 = root (last committed token, the bonus row),
//! rows 1..=L = spine in depth order, then hedges in (depth, rank) order.
//! The spine being a contiguous prefix is what lets the GDN layer run one
//! existing wy-kernel pass over rows 0..=L unchanged.

use anyhow::{Result, bail};

/// Maximum candidate rank per position (drafter shadow/top-k width).
pub const MAX_RANK: usize = 4;
/// Maximum tree nodes (excl. root row); M = nodes + 1 <= 8 keeps every
/// projection on the batched-GEMV path once the batch8 kernel lands.
pub const MAX_NODES: usize = 7;

/// Static tree shape: per-depth node counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeShape {
    /// counts[d-1] = number of candidate nodes at depth d (1 = spine only).
    pub counts: Vec<u8>,
}

impl TreeShape {
    pub fn parse(s: &str) -> Result<Self> {
        let counts: Vec<u8> = s
            .split(',')
            .map(|t| t.trim().parse::<u8>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("ATLAS_TREE_SHAPE parse: {e}"))?;
        let shape = Self { counts };
        shape.validate()?;
        Ok(shape)
    }

    pub fn validate(&self) -> Result<()> {
        if self.counts.is_empty() {
            bail!("tree shape: empty");
        }
        if self.counts.iter().any(|&c| c == 0 || c as usize > MAX_RANK) {
            bail!("tree shape: per-depth count must be 1..={MAX_RANK}");
        }
        if self.nodes() > MAX_NODES {
            bail!("tree shape: {} nodes > max {MAX_NODES}", self.nodes());
        }
        Ok(())
    }

    pub fn spine_len(&self) -> usize {
        self.counts.len()
    }

    /// Total tree nodes (spine + hedges), excluding the root row.
    pub fn nodes(&self) -> usize {
        self.counts.iter().map(|&c| c as usize).sum()
    }

    /// Verify width M = root row + nodes.
    pub fn verify_width(&self) -> usize {
        self.nodes() + 1
    }

    /// Stable id for CUDA-graph keying: counts packed 4 bits per depth.
    pub fn shape_id(&self) -> u64 {
        self.counts
            .iter()
            .fold(0u64, |acc, &c| (acc << 4) | (c as u64))
    }

    /// A pure chain (no hedges) — must reproduce the chain drafter exactly.
    pub fn is_chain(&self) -> bool {
        self.counts.iter().all(|&c| c == 1)
    }
}

/// One hedge leaf: the drafter's rank-`rank` candidate at `depth`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HedgeNode {
    pub depth: usize,
    /// Candidate rank (2-based: rank 2 = drafter's second choice).
    pub rank: usize,
    pub token: u32,
}

/// A proposed draft tree: spine tokens + hedge leaves for one verify step.
#[derive(Debug, Clone)]
pub struct TreeDraft {
    pub shape: TreeShape,
    /// Spine tokens, spine[d-1] = top-1 draft at depth d.
    pub spine: Vec<u32>,
    /// Hedges sorted by (depth, rank) — the row-layout order.
    pub hedges: Vec<HedgeNode>,
}

/// One verify row: (token, depth, parent_row). Row 0 is the root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeRow {
    pub token: u32,
    pub depth: usize,
    pub parent_row: usize,
}

impl TreeDraft {
    /// Verify-row layout: [root, spine..., hedges...]. `root_token` is the
    /// last committed token (the chain paths' `a.last_token`).
    pub fn rows(&self, root_token: u32) -> Vec<TreeRow> {
        let l = self.spine.len();
        let mut rows = Vec::with_capacity(1 + l + self.hedges.len());
        rows.push(TreeRow {
            token: root_token,
            depth: 0,
            parent_row: 0,
        });
        for (i, &t) in self.spine.iter().enumerate() {
            // spine row d's parent is spine row d-1 (row 0 = root).
            rows.push(TreeRow {
                token: t,
                depth: i + 1,
                parent_row: i,
            });
        }
        for h in &self.hedges {
            // spine+hedge invariant: a depth-d hedge's parent is the spine
            // node at depth d-1 (= row d-1 in this layout).
            rows.push(TreeRow {
                token: h.token,
                depth: h.depth,
                parent_row: h.depth - 1,
            });
        }
        rows
    }

    /// Longest root-to-leaf accepted path given per-row target argmaxes
    /// `v[row]` (the target's next token after that row). Returns the
    /// accepted rows in order plus the bonus token (the target argmax at
    /// the last accepted row — row 0 if nothing accepted). Byte-identical
    /// to greedy by induction: children of a row hold distinct tokens, so
    /// at most one matches `v[parent]`.
    pub fn accept_path(&self, rows: &[TreeRow], v: &[u32]) -> (Vec<usize>, u32) {
        let l = self.spine.len();
        let mut path = Vec::with_capacity(l);
        let mut cur = 0usize; // root row
        for d in 1..=l {
            let want = v[cur];
            // Children of `cur` (a spine row at depth d-1): the spine row at
            // depth d and every hedge at depth d.
            let spine_row = d; // rows[1..=l] are the spine
            let mut next = None;
            if rows[spine_row].token == want {
                next = Some(spine_row);
            } else {
                for (i, h) in self.hedges.iter().enumerate() {
                    if h.depth == d && h.token == want {
                        next = Some(1 + l + i);
                        break;
                    }
                }
            }
            match next {
                Some(r) => {
                    path.push(r);
                    cur = r;
                    // A hedge is a leaf — the path cannot extend below it.
                    if r > l {
                        break;
                    }
                }
                None => break,
            }
        }
        (path, v[cur])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(shape: &str, spine: &[u32], hedges: &[(usize, usize, u32)]) -> TreeDraft {
        TreeDraft {
            shape: TreeShape::parse(shape).unwrap(),
            spine: spine.to_vec(),
            hedges: hedges
                .iter()
                .map(|&(depth, rank, token)| HedgeNode { depth, rank, token })
                .collect(),
        }
    }

    #[test]
    fn parse_and_validate() {
        let s = TreeShape::parse("1,2,2,2").unwrap();
        assert_eq!(s.spine_len(), 4);
        assert_eq!(s.nodes(), 7);
        assert_eq!(s.verify_width(), 8);
        assert!(!s.is_chain());
        assert!(TreeShape::parse("1,1").unwrap().is_chain());
        assert!(TreeShape::parse("").is_err());
        assert!(TreeShape::parse("1,0").is_err());
        assert!(TreeShape::parse("5").is_err()); // rank > MAX_RANK
        assert!(TreeShape::parse("2,2,2,2").is_err()); // 8 nodes > 7
        assert_ne!(
            TreeShape::parse("1,2").unwrap().shape_id(),
            TreeShape::parse("2,1").unwrap().shape_id()
        );
    }

    #[test]
    fn row_layout_spine_prefix() {
        let t = draft("2,2", &[10, 20], &[(1, 2, 11), (2, 2, 21)]);
        let rows = t.rows(5);
        // [root, spine1, spine2, hedge(d1), hedge(d2)]
        assert_eq!(rows.len(), 5);
        assert_eq!(
            rows[0],
            TreeRow {
                token: 5,
                depth: 0,
                parent_row: 0
            }
        );
        assert_eq!(
            rows[1],
            TreeRow {
                token: 10,
                depth: 1,
                parent_row: 0
            }
        );
        assert_eq!(
            rows[2],
            TreeRow {
                token: 20,
                depth: 2,
                parent_row: 1
            }
        );
        assert_eq!(
            rows[3],
            TreeRow {
                token: 11,
                depth: 1,
                parent_row: 0
            }
        );
        assert_eq!(
            rows[4],
            TreeRow {
                token: 21,
                depth: 2,
                parent_row: 1
            }
        );
    }

    #[test]
    fn accept_full_spine() {
        let t = draft("1,1", &[10, 20], &[]);
        let rows = t.rows(5);
        // v[root]=10 (spine1 ok), v[spine1]=20 (spine2 ok), v[spine2]=99 bonus
        let (path, bonus) = t.accept_path(&rows, &[10, 20, 99]);
        assert_eq!(path, vec![1, 2]);
        assert_eq!(bonus, 99);
    }

    #[test]
    fn accept_hedge_rescue_is_terminal() {
        // spine [10, 20], hedge at depth1 = 11. Its bonus deliberately
        // equals spine2, proving the accepted hedge remains a leaf.
        let t = draft("2,1", &[10, 20], &[(1, 2, 11)]);
        let rows = t.rows(5);
        // v[root]=11 → spine1(10) misses, hedge(11) rescues (row 3).
        // Hedge is a leaf: path ends there; bonus = v[hedge row]=20.
        let (path, bonus) = t.accept_path(&rows, &[11, 55, 66, 20]);
        assert_eq!(path, vec![3]);
        assert_eq!(bonus, 20);
    }

    #[test]
    fn accept_hedge_after_spine_prefix_is_terminal() {
        // The first spine token is accepted before a depth-2 hedge rescues
        // the path. This reaches hedge lookup after `cur` has advanced.
        let t = draft("1,2,1", &[10, 20, 30], &[(2, 2, 21)]);
        let rows = t.rows(5);
        let (path, bonus) = t.accept_path(&rows, &[10, 21, 66, 77, 88]);
        assert_eq!(path, vec![1, 4]);
        assert_eq!(bonus, 88);
    }

    #[test]
    fn accept_reject_all_gives_root_bonus() {
        let t = draft("2,1", &[10, 20], &[(1, 2, 11)]);
        let rows = t.rows(5);
        let (path, bonus) = t.accept_path(&rows, &[42, 0, 0, 0]);
        assert!(path.is_empty());
        assert_eq!(bonus, 42);
    }
}
