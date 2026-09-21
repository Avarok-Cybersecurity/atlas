// SPDX-License-Identifier: AGPL-3.0-only

//! The probe corpus — two prompts built so that contamination is *detectable*,
//! not merely possible.
//!
//! # Why the prompts look like this
//!
//! * **Each prompt carries a canary and orders the model to echo it.** The
//!   canary therefore exists in BOTH places state can leak from: the prompt
//!   tokens (a KV/prefix-cache collision replays the other request's context)
//!   and the generated tokens (a bled-through decode state emits the other
//!   request's output). Either leak surfaces the foreign canary lexically,
//!   which is what [`super::transcript::Transcript::carries_foreign_canary`]
//!   detects — contamination on its own evidence, no reference diff needed.
//! * **The canaries share no words, no digit groups, no substrings.** They are
//!   uppercase-hyphen-digit codes that greedy decoding will never produce from
//!   unrelated context; the only way one appears in the other probe's reply is
//!   verbatim transport out of the other request's state.
//! * **The prompts are IDENTICAL up to the canary.** Deliberate: the shared
//!   preamble is a live prefix-cache collision surface, and the first token
//!   after it is inside the canary sentence. A cache that resumes the WRONG
//!   continuation at the shared-prefix boundary — the subtlest off-by-one in
//!   prefix matching — emits the foreign canary in the reply's first line,
//!   turning the quietest fault into the loudest signal.
//! * **The topics are disjoint** (lighthouse vs. water mill), so even a leak
//!   that mangles rather than transplants the canary still drags in vocabulary
//!   the solo reference never used, and the stream diff catches it as
//!   `Diverged`.
//! * **Deterministic by construction.** The driver sends temperature 0; the
//!   prompts ask for a fixed shape (code, exactly three sentences, code) so a
//!   healthy engine has one greedy continuation, and the three sentences keep
//!   the reply comfortably above the completion-token floor that guards
//!   against "equal because both replies were empty".

/// One probe: a prompt and the marker that must never appear in any OTHER
/// probe's reply.
pub struct Probe {
    /// Short label for the report table.
    pub name: &'static str,
    /// The distinctive lexical marker owned by this probe.
    pub canary: &'static str,
    /// The probe-specific instruction; [`Probe::prompt`] prefixes the shared
    /// preamble. Split so the preamble is written exactly once.
    tail: &'static str,
}

impl Probe {
    /// The full prompt sent to the endpoint: shared preamble + this probe's
    /// tail. The preamble lives in one const, so the "identical up to the
    /// canary" property holds by construction rather than by proofreading.
    pub fn prompt(&self) -> String {
        format!("{PREAMBLE}{}", self.tail)
    }
}

/// Shared preamble — identical across probes ON PURPOSE; see the module docs.
const PREAMBLE: &str = "You are a precise assistant taking part in a determinism audit. \
                        Follow the instructions exactly and do not add anything else. ";

pub const PROBES: [Probe; 2] = [
    Probe {
        name: "A",
        canary: "XK-AZURE-HERON-41",
        tail: "Your reference code is XK-AZURE-HERON-41. Begin your reply with the \
               reference code on its own line, then explain in exactly three short \
               sentences how a lighthouse warns ships at night, then end with the \
               reference code again on its own line.",
    },
    Probe {
        name: "B",
        canary: "QJ-CRIMSON-OTTER-77",
        tail: "Your reference code is QJ-CRIMSON-OTTER-77. Begin your reply with the \
               reference code on its own line, then explain in exactly three short \
               sentences how a water mill grinds grain into flour, then end with the \
               reference code again on its own line.",
    },
];

/// The canaries in probe order — the shape [`super::score::Legs`] consumes.
pub fn canaries() -> Vec<String> {
    PROBES.iter().map(|p| p.canary.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::contamination::transcript::Transcript;

    /// ★ The property the whole detector rests on: a canary belongs to exactly
    /// one prompt. Its own prompt carries it (positive); no other prompt
    /// contains it, or any substring confusable with it (negative).
    #[test]
    fn every_canary_is_unique_to_its_probe() {
        for (i, a) in PROBES.iter().enumerate() {
            assert!(
                a.prompt().contains(a.canary),
                "probe {} does not carry its own canary",
                a.name
            );
            for (j, b) in PROBES.iter().enumerate() {
                if i == j {
                    continue;
                }
                assert!(
                    !b.prompt().contains(a.canary),
                    "probe {}'s prompt contains probe {}'s canary — a clean run \
                     would read as contaminated",
                    b.name,
                    a.name
                );
                assert!(
                    !a.canary.contains(b.canary) && !b.canary.contains(a.canary),
                    "canaries {} / {} overlap as substrings",
                    a.canary,
                    b.canary
                );
                let a_parts: std::collections::BTreeSet<_> = a.canary.split('-').collect();
                let b_parts: std::collections::BTreeSet<_> = b.canary.split('-').collect();
                assert!(
                    a_parts.is_disjoint(&b_parts),
                    "canaries {} / {} share lexical components: {:?}",
                    a.canary,
                    b.canary,
                    a_parts.intersection(&b_parts).collect::<Vec<_>>()
                );
            }
        }
    }

    /// The shared-prefix collision surface is real: the prompts agree up to
    /// the preamble boundary and each canary sits in the probe-specific tail.
    #[test]
    fn prompts_share_the_preamble_and_diverge_at_the_canary() {
        let full: Vec<String> = PROBES.iter().map(Probe::prompt).collect();
        let canary_offsets: Vec<usize> = PROBES
            .iter()
            .zip(&full)
            .map(|(probe, prompt)| prompt.find(probe.canary).expect("canary in own prompt"))
            .collect();
        assert_eq!(canary_offsets[0], canary_offsets[1]);
        let boundary = canary_offsets[0];
        assert_eq!(&full[0][..boundary], &full[1][..boundary]);
        assert_eq!(
            &full[0][..boundary],
            format!("{PREAMBLE}Your reference code is ")
        );
        for p in &PROBES {
            assert!(
                p.prompt().starts_with(PREAMBLE),
                "probe {} lost the shared preamble",
                p.name
            );
            assert!(
                !PREAMBLE.contains(p.canary) && p.tail.contains(p.canary),
                "probe {}'s canary must live in the divergent tail, not the \
                 shared span",
                p.name
            );
        }
    }

    #[test]
    fn names_are_distinct_and_canaries_ride_in_probe_order() {
        let mut names = std::collections::BTreeSet::new();
        for p in &PROBES {
            assert!(names.insert(p.name), "duplicate probe name {}", p.name);
        }
        assert_eq!(
            canaries(),
            PROBES
                .iter()
                .map(|p| p.canary.to_string())
                .collect::<Vec<_>>(),
            "Legs indexes canaries by prompt position; order is load-bearing"
        );
    }

    /// Bridge to the core: a reply shaped like probe A's honest output does
    /// NOT read as contaminated, and the same reply with B's canary spliced in
    /// DOES — attributed to B.
    #[test]
    fn the_core_detector_fires_on_these_canaries() {
        let cans = canaries();
        let all: Vec<&str> = cans.iter().map(String::as_str).collect();
        let clean = Transcript {
            text: format!(
                "{}\nA lighthouse shines a rotating beam. Ships see it from afar. \
                 They steer clear of the rocks.\n{}",
                PROBES[0].canary, PROBES[0].canary
            ),
            ..Default::default()
        };
        assert_eq!(
            clean.carries_foreign_canary(PROBES[0].canary, &all),
            None,
            "a probe's own canary must not read as leakage"
        );
        let leaked = Transcript {
            text: format!("{}\nA lighthouse shines...", PROBES[1].canary),
            ..Default::default()
        };
        assert_eq!(
            leaked.carries_foreign_canary(PROBES[0].canary, &all),
            Some(PROBES[1].canary),
            "B's canary in A's reply must be attributed to B"
        );
    }
}
