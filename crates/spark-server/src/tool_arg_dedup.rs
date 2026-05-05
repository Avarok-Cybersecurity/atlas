// SPDX-License-Identifier: AGPL-3.0-only

//! F5 (2026-04-26): tool-call argument hash dedup.
//!
//! Catches the agentic-failure mode where the model emits the *same*
//! tool call (same name + same arguments, modulo whitespace) over
//! and over within a single response — distinct from sentence-level
//! prose loops because the model may correctly vary its narration
//! while still re-issuing identical `Bash(mkdir -p /tmp/...)` calls.
//!
//! Algorithm:
//!
//! - Hash `(name, canonical_json(args))` to a `u64` via the standard
//!   library `DefaultHasher`. Whitespace and key-order in the
//!   arguments JSON are normalised before hashing so semantically
//!   identical calls collide.
//! - Maintain a ring buffer of the last `cap` hashes.
//! - Trip on:
//!   - **`threshold_consec`** consecutive identical hashes at the
//!     tail of the ring (fast path — catches rapid-fire dupes), OR
//!   - **`threshold_window`-of-`cap`** identical hashes anywhere in
//!     the ring (slow path — catches dupes interleaved with one-off
//!     other calls).
//!
//! Defaults: `cap=8`, `threshold_consec=5`, `threshold_window=6`.
//!
//! `check` is `~50 ns` per call.

use std::collections::VecDeque;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

const DEFAULT_CAP: usize = 8;
const DEFAULT_CONSEC: u32 = 5;
const DEFAULT_WINDOW: u32 = 6;

/// Tool-call argument hash dedup guard.
#[derive(Debug)]
pub struct ToolArgDedup {
    recent: VecDeque<u64>,
    cap: usize,
    threshold_consec: u32,
    threshold_window: u32,
}

impl Default for ToolArgDedup {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolArgDedup {
    /// Construct with default parameters.
    pub fn new() -> Self {
        Self::with_params(DEFAULT_CAP, DEFAULT_CONSEC, DEFAULT_WINDOW)
    }

    /// Construct with explicit ring capacity and trip thresholds.
    pub fn with_params(cap: usize, threshold_consec: u32, threshold_window: u32) -> Self {
        Self {
            recent: VecDeque::with_capacity(cap.max(1)),
            cap: cap.max(1),
            threshold_consec,
            threshold_window,
        }
    }

    /// Check whether `(name, args_json)` triggers the dedup trip.
    /// Always pushes the new hash onto the ring (after the check).
    /// Returns `true` if the consec or window threshold is reached.
    pub fn check(&mut self, name: &str, args_json: &str) -> bool {
        let h = hash_call(name, args_json);
        let trip = self.would_trip(h);
        if self.recent.len() >= self.cap {
            self.recent.pop_front();
        }
        self.recent.push_back(h);
        trip
    }

    /// Predict whether pushing `h` *would* trip a threshold,
    /// counting `h` as the most-recent occurrence.
    fn would_trip(&self, h: u64) -> bool {
        // Consec: count how many of the most-recent hashes already
        // in the ring equal `h`. +1 for the new push.
        let mut consec = 1u32;
        for &p in self.recent.iter().rev() {
            if p == h {
                consec += 1;
            } else {
                break;
            }
        }
        if consec >= self.threshold_consec {
            return true;
        }
        // Window: count total occurrences of `h` in the post-push
        // ring (existing + new) against `threshold_window`.
        let occ = self.recent.iter().filter(|&&p| p == h).count() as u32 + 1;
        occ >= self.threshold_window
    }

    /// Drop all stored hashes.
    pub fn reset(&mut self) {
        self.recent.clear();
    }

    /// Number of hashes currently stored.
    pub fn len(&self) -> usize {
        self.recent.len()
    }
}

/// Hash `(name, canonical_args)` to a u64. The JSON args are parsed
/// and re-serialised in canonical (sorted-key, no-whitespace) form
/// so semantically identical calls collide regardless of upstream
/// formatting. Falls back to raw-string hashing on parse error.
fn hash_call(name: &str, args_json: &str) -> u64 {
    let mut h = DefaultHasher::new();
    h.write(name.as_bytes());
    h.write_u8(0); // separator so "ab" + "c" != "a" + "bc"
    let canonical = canonicalize_json(args_json).unwrap_or_else(|| args_json.to_string());
    h.write(canonical.as_bytes());
    h.finish()
}

/// Re-serialise a JSON value with sorted object keys and no
/// whitespace. Returns `None` if `s` is not valid JSON.
fn canonicalize_json(s: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    Some(canonical_value_to_string(&v))
}

fn canonical_value_to_string(v: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical(&mut out, v);
    out
}

fn write_canonical(out: &mut String, v: &serde_json::Value) {
    use std::fmt::Write;
    match v {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => {
            let _ = write!(out, "{}", n);
        }
        serde_json::Value::String(s) => {
            let _ = write!(out, "{}", serde_json::Value::String(s.clone()));
        }
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, x) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(out, x);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            out.push('{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let _ = write!(out, "{}", serde_json::Value::String(k.clone()));
                out.push(':');
                write_canonical(out, &map[k]);
            }
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_consecutive_identical_calls_trip() {
        let mut g = ToolArgDedup::new();
        for i in 0..4 {
            assert!(
                !g.check("Bash", r#"{"command":"mkdir -p /tmp/foo"}"#),
                "iteration {i} must not trip"
            );
        }
        assert!(
            g.check("Bash", r#"{"command":"mkdir -p /tmp/foo"}"#),
            "5th identical call must trip"
        );
    }

    #[test]
    fn whitespace_and_key_order_normalised() {
        let mut g = ToolArgDedup::new();
        // These three SHOULD all hash identically.
        assert!(!g.check("Write", r#"{"path":"/tmp/x","content":"hi"}"#));
        assert!(!g.check("Write", r#"{ "path" : "/tmp/x" , "content" : "hi" }"#));
        assert!(!g.check("Write", r#"{"content":"hi","path":"/tmp/x"}"#));
        assert!(!g.check("Write", r#"{"path":"/tmp/x","content":"hi"}"#));
        assert!(
            g.check("Write", r#"{"path":"/tmp/x","content":"hi"}"#),
            "5th identical (modulo whitespace+key-order) call must trip"
        );
    }

    #[test]
    fn distinct_calls_do_not_trip() {
        let mut g = ToolArgDedup::new();
        for i in 0..8 {
            let args = format!(r#"{{"command":"mkdir -p /tmp/test-{i}"}}"#);
            assert!(
                !g.check("Bash", &args),
                "8 distinct calls must not trip (i={i})"
            );
        }
    }

    #[test]
    fn six_of_eight_window_trips() {
        let mut g = ToolArgDedup::new();
        // Pattern: 6 dupes interleaved with 2 distinct calls within
        // the 8-slot window. Total 6 dupes meets threshold_window.
        let dup = r#"{"command":"ls"}"#;
        let other_a = r#"{"command":"pwd"}"#;
        let other_b = r#"{"command":"whoami"}"#;
        // Place: dup, dup, other_a, dup, dup, other_b, dup, dup
        // Each dup pushed-in: after 1, after 2, [skipped: 3rd is
        // not consec], after 3 (consec=3), after 4 (consec=4),
        // [break consec], after 5, after 6 (window match).
        let mut tripped_at = None;
        for (i, args) in [dup, dup, other_a, dup, dup, other_b, dup, dup]
            .iter()
            .enumerate()
        {
            if g.check("Bash", args) {
                tripped_at = Some(i);
                break;
            }
        }
        assert!(
            tripped_at.is_some(),
            "must trip on 6/8 window pattern; ring={:?}",
            g.recent
        );
    }

    #[test]
    fn name_difference_breaks_dedup() {
        let mut g = ToolArgDedup::new();
        let args = r#"{"x":1}"#;
        // Same args, different tool names — must NOT trip even
        // after 5 calls because the per-tool hashes differ.
        for name in ["A", "B", "C", "D", "E"] {
            assert!(!g.check(name, args), "name {name} must not trip");
        }
    }

    #[test]
    fn invalid_json_falls_back_to_raw_string() {
        let mut g = ToolArgDedup::new();
        // Malformed JSON — fallback hashes the raw string. Identical
        // raw strings still trip the consec threshold.
        for _ in 0..4 {
            assert!(!g.check("Bash", "not valid {{ json"));
        }
        assert!(g.check("Bash", "not valid {{ json"));
    }

    #[test]
    fn reset_clears_state() {
        let mut g = ToolArgDedup::new();
        for _ in 0..4 {
            g.check("Bash", r#"{"x":1}"#);
        }
        assert_eq!(g.len(), 4);
        g.reset();
        assert_eq!(g.len(), 0);
        // After reset, the same call is "fresh" again — not yet at
        // the consec threshold.
        for _ in 0..4 {
            assert!(!g.check("Bash", r#"{"x":1}"#));
        }
    }
}
