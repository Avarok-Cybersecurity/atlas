// SPDX-License-Identifier: AGPL-3.0-only

//! Stop-string matching for `/completion`.
//!
//! llama.cpp's `stop` strings end generation when they appear anywhere in the
//! output, and the matched string is excluded from `content`. The OpenAI
//! completions route only honors single-token stops inside the scheduler and
//! trims a multi-token stop after the fact, so generation runs on past it.
//! Here every stop string is matched on the decoded text as it streams, text
//! that could still turn into a stop string is held back, and the caller
//! cancels the request the moment one matches.

/// Incremental matcher over the decoded text stream.
#[derive(Debug, Clone, Default)]
pub(crate) struct StopMatcher {
    stops: Vec<String>,
    /// Text received but not yet released because it might begin a stop string.
    held: String,
    hit: Option<String>,
}

impl StopMatcher {
    /// Empty stop strings are dropped, as llama.cpp does.
    pub(crate) fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            held: String::new(),
            hit: None,
        }
    }

    /// The stop string that ended the output, if one did.
    pub(crate) fn hit(&self) -> Option<&str> {
        self.hit.as_deref()
    }

    /// Feed newly decoded text; returns the text that is safe to release.
    ///
    /// On a match the text before the stop string is returned, the matcher
    /// closes, and every later call returns an empty string.
    pub(crate) fn push(&mut self, delta: &str) -> String {
        if self.hit.is_some() {
            return String::new();
        }
        self.held.push_str(delta);
        if self.stops.is_empty() {
            return std::mem::take(&mut self.held);
        }
        // Earliest match wins; at the same position the longest stop wins.
        let found = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()).map(|p| (p, s)))
            .min_by(|(pa, sa), (pb, sb)| pa.cmp(pb).then(sb.len().cmp(&sa.len())));
        if let Some((pos, stop)) = found {
            let stop = stop.clone();
            let out = self.held[..pos].to_string();
            self.held.clear();
            self.hit = Some(stop);
            return out;
        }
        // No match yet: keep only the longest tail that is still a proper
        // prefix of some stop string.
        let keep = self.partial_tail_len();
        let release = self.held.len() - keep;
        let out = self.held[..release].to_string();
        self.held.drain(..release);
        out
    }

    /// End of stream with no stop string seen: release whatever was held.
    pub(crate) fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    fn partial_tail_len(&self) -> usize {
        let max = self
            .stops
            .iter()
            .map(|s| s.len().saturating_sub(1))
            .max()
            .unwrap_or(0)
            .min(self.held.len());
        for k in (1..=max).rev() {
            let start = self.held.len() - k;
            if !self.held.is_char_boundary(start) {
                continue;
            }
            let tail = &self.held[start..];
            if self
                .stops
                .iter()
                .any(|s| s.len() > k && s.is_char_boundary(k) && s.starts_with(tail))
            {
                return k;
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], chunks: &[&str]) -> (String, Option<String>) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut m = StopMatcher::new(&stops);
        let mut out = String::new();
        for c in chunks {
            out.push_str(&m.push(c));
        }
        if m.hit().is_none() {
            out.push_str(&m.finish());
        }
        (out, m.hit().map(str::to_string))
    }

    #[test]
    fn no_stops_releases_everything_at_once() {
        let mut m = StopMatcher::new(&[]);
        assert_eq!(m.push("abc"), "abc");
        assert_eq!(m.finish(), "");
    }

    #[test]
    fn a_stop_inside_one_chunk_cuts_before_it() {
        assert_eq!(
            run(&["END"], &["hello END world"]),
            ("hello ".to_string(), Some("END".to_string()))
        );
    }

    #[test]
    fn a_stop_split_across_chunks_is_still_found_and_never_leaks() {
        let (text, hit) = run(&["END"], &["hello E", "N", "D tail"]);
        assert_eq!(text, "hello ");
        assert_eq!(hit.as_deref(), Some("END"));
    }

    #[test]
    fn a_false_start_is_released_once_it_cannot_complete() {
        let (text, hit) = run(&["END"], &["a E", "NX b"]);
        assert_eq!(text, "a ENX b");
        assert_eq!(hit, None);
    }

    #[test]
    fn partial_stop_at_end_of_stream_is_flushed() {
        let (text, hit) = run(&["END"], &["tail EN"]);
        assert_eq!(text, "tail EN");
        assert_eq!(hit, None);
    }

    #[test]
    fn text_is_released_as_soon_as_it_is_safe() {
        let mut m = StopMatcher::new(&["STOP".to_string()]);
        assert_eq!(m.push("ab"), "ab", "nothing can start a stop yet");
        assert_eq!(m.push("cS"), "c", "only the possible start is held");
        assert_eq!(m.push("T"), "");
        assert_eq!(m.push("OP!"), "");
        assert_eq!(m.hit(), Some("STOP"));
    }

    #[test]
    fn earliest_stop_wins_and_longest_at_a_tie() {
        assert_eq!(
            run(&["bc", "ab"], &["xabcx"]),
            ("x".to_string(), Some("ab".to_string()))
        );
        assert_eq!(
            run(&["a", "ab"], &["xaby"]),
            ("x".to_string(), Some("ab".to_string()))
        );
    }

    #[test]
    fn closes_after_a_hit() {
        let mut m = StopMatcher::new(&["x".to_string()]);
        assert_eq!(m.push("ax"), "a");
        assert_eq!(m.push("more"), "");
        assert_eq!(m.finish(), "");
        assert_eq!(m.hit(), Some("x"));
    }

    #[test]
    fn multibyte_text_and_stops_do_not_split_characters() {
        let (text, hit) = run(&["\u{20AC}\u{20AC}"], &["a\u{20AC}", "\u{20AC}b"]);
        assert_eq!(text, "a");
        assert_eq!(hit.as_deref(), Some("\u{20AC}\u{20AC}"));
        let (text, hit) = run(&["\u{20AC}\u{20AC}"], &["\u{20AC}b"]);
        assert_eq!(text, "\u{20AC}b");
        assert_eq!(hit, None);
    }

    #[test]
    fn empty_stop_strings_are_ignored() {
        let (text, hit) = run(&[""], &["abc"]);
        assert_eq!((text.as_str(), hit), ("abc", None));
    }

    #[test]
    fn newline_stop_works_like_any_other() {
        let (text, hit) = run(&["\n"], &["one", " two\nthree"]);
        assert_eq!(text, "one two");
        assert_eq!(hit.as_deref(), Some("\n"));
    }
}
