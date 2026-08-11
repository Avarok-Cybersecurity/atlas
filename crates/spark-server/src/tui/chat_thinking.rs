// SPDX-License-Identifier: AGPL-3.0-only

//! Thinking (reasoning) preferences for the Chat pane.
//!
//! Two orthogonal controls, on purpose:
//!
//! * [`ThinkingRequest`] — what the CLIENT ASKS FOR. It changes the request
//!   body, so it decides whether the model reasons at all.
//! * [`ThinkingView`] — how an arriving reasoning trace is DRAWN. It changes
//!   nothing on the wire.
//!
//! Collapsing a block you asked for and never asking for it are different
//! intents, and a single control that did both would make one of them
//! unreachable.

use std::time::Instant;

/// What the request asks the server to do about thinking.
///
/// [`ThinkingRequest::Auto`] OMITS `chat_template_kwargs.enable_thinking` from
/// the body entirely, so the server resolves the model's own default
/// (`[behavior].thinking_default` in its `MODEL.toml`). Sending the value the
/// client BELIEVES that default to be would freeze today's answer into the
/// dashboard and diverge, silently and plausibly, the first time a model ships
/// a different one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThinkingRequest {
    /// Send nothing; the model decides.
    #[default]
    Auto,
    /// `chat_template_kwargs: {"enable_thinking": false}`.
    Off,
    /// `chat_template_kwargs: {"enable_thinking": true}`.
    On,
}

impl ThinkingRequest {
    /// Auto → Off → On → Auto.
    ///
    /// Off sits one press from the default because "stop thinking at me" is
    /// the reason anyone reaches for this key; On is two presses because a
    /// model whose default is already thinking needs no help being asked.
    pub fn next(self) -> Self {
        match self {
            Self::Auto => Self::Off,
            Self::Off => Self::On,
            Self::On => Self::Auto,
        }
    }

    /// The value to put in `chat_template_kwargs.enable_thinking`, or `None`
    /// when the key must not appear at all.
    ///
    /// ★ The key is `enable_thinking`, nested under `chat_template_kwargs`.
    /// A bare `{"thinking": false}` deserializes into the request's Anthropic
    /// `thinking` object, contributes no directive, and is then IGNORED — it
    /// does not error, so it reads exactly like a toggle that works while the
    /// model goes on reasoning.
    pub fn enable_thinking(self) -> Option<bool> {
        match self {
            Self::Auto => None,
            Self::Off => Some(false),
            Self::On => Some(true),
        }
    }

    /// The state chip for the chat pane.
    ///
    /// In `Auto` the client does not know what the model chose until deltas
    /// arrive, so nothing is claimed until `observed` says what actually
    /// happened on the last completed reply. `observed` is an OBSERVATION,
    /// never a prediction: `None` renders a bare `auto`.
    pub fn chip(self, observed: Option<bool>, wide: bool) -> String {
        let resolved = match observed {
            Some(true) => " (thinking)",
            Some(false) => " (no thinking)",
            None => "",
        };
        match self {
            Self::Auto if wide => format!("thinking auto{resolved}"),
            Self::Auto => "think auto".to_string(),
            Self::Off if wide => "thinking off".to_string(),
            Self::Off => "think off".to_string(),
            Self::On if wide => "thinking on".to_string(),
            Self::On => "think on".to_string(),
        }
    }
}

/// How a reasoning trace is rendered once it arrives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThinkingView {
    /// Live while it streams, one quiet summary line once the answer lands.
    #[default]
    Collapsed,
    /// The whole trace stays on screen after the reply completes.
    Expanded,
    /// Not drawn at all — no header, no summary, no wasted row.
    Hidden,
}

impl ThinkingView {
    pub fn next(self) -> Self {
        match self {
            Self::Collapsed => Self::Expanded,
            Self::Expanded => Self::Hidden,
            Self::Hidden => Self::Collapsed,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Collapsed => "collapsed",
            Self::Expanded => "expanded",
            Self::Hidden => "hidden",
        }
    }
}

/// The reasoning half of one model reply.
#[derive(Default)]
pub struct Reasoning {
    pub text: String,
    /// Reasoning deltas seen. Counted separately from answer tokens because
    /// they are the same decode work and a footer that hid them made a
    /// healthy 77.6 ms/token look like a stall.
    pub tokens: usize,
    /// First reasoning delta → first answer delta (or end of stream), measured
    /// on the streaming task's clock. `None` until the reply finishes.
    pub think_ms: Option<f64>,
    /// When the first reasoning delta reached the TUI, for the live timer in
    /// the streaming header. Separate from `think_ms` on purpose: one is a
    /// measurement of the reply, the other drives an animation.
    started: Option<Instant>,
}

impl Reasoning {
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Note that reasoning has begun. Idempotent — only the first delta starts
    /// the clock.
    pub fn begin(&mut self) {
        self.started.get_or_insert_with(Instant::now);
    }

    /// Freeze the live timer at the moment the answer starts.
    ///
    /// Reasoning that keeps arriving after the answer began is normal, so
    /// without this the summary line goes on counting up underneath a reply
    /// the user is already reading. The authoritative span arrives with the
    /// terminal delta and replaces this estimate.
    pub fn seal(&mut self) {
        if self.think_ms.is_none()
            && let Some(t) = self.started
        {
            self.think_ms = Some(t.elapsed().as_secs_f64() * 1000.0);
        }
    }

    /// Seconds to show beside the header: the measured span once the reply has
    /// finished, otherwise the live elapsed. `None` before the first delta.
    pub fn seconds(&self) -> Option<f64> {
        self.think_ms
            .map(|ms| ms / 1000.0)
            .or_else(|| self.started.map(|t| t.elapsed().as_secs_f64()))
    }
}

/// `18.2s`, `412ms`, `1m 04s` — one compact duration format for the whole pane.
pub fn dur(secs: f64) -> String {
    if secs < 1.0 {
        format!("{:.0}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        format!("{}m {:02.0}s", (secs / 60.0).floor(), secs % 60.0)
    }
}

#[cfg(test)]
#[path = "chat_thinking_more_tests.rs"]
mod chip_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_omits_the_key_entirely() {
        // The whole point: a client that guesses the model's default hardcodes
        // it, and diverges the moment the model changes.
        assert_eq!(ThinkingRequest::Auto.enable_thinking(), None);
        assert_eq!(ThinkingRequest::Off.enable_thinking(), Some(false));
        assert_eq!(ThinkingRequest::On.enable_thinking(), Some(true));
    }

    #[test]
    fn both_cycles_return_home_in_three() {
        let mut r = ThinkingRequest::default();
        for _ in 0..3 {
            r = r.next();
        }
        assert_eq!(r, ThinkingRequest::Auto);
        assert_eq!(ThinkingRequest::Auto.next(), ThinkingRequest::Off);
        let mut v = ThinkingView::default();
        for _ in 0..3 {
            v = v.next();
        }
        assert_eq!(v, ThinkingView::Collapsed);
    }

    #[test]
    fn auto_claims_nothing_until_it_has_been_observed() {
        assert_eq!(ThinkingRequest::Auto.chip(None, true), "thinking auto");
        assert_eq!(
            ThinkingRequest::Auto.chip(Some(true), true),
            "thinking auto (thinking)"
        );
        assert_eq!(
            ThinkingRequest::Auto.chip(Some(false), true),
            "thinking auto (no thinking)"
        );
        // An explicit state is not an observation, so it never grows a suffix.
        assert_eq!(ThinkingRequest::Off.chip(Some(true), true), "thinking off");
    }

    #[test]
    fn durations_stay_short_enough_for_eighty_columns() {
        assert_eq!(dur(0.412), "412ms");
        assert_eq!(dur(18.24), "18.2s");
        assert_eq!(dur(64.0), "1m 04s");
    }

    #[test]
    fn a_reasoning_clock_starts_once_and_prefers_the_measurement() {
        let mut r = Reasoning::default();
        assert!(r.seconds().is_none(), "nothing to time yet");
        r.begin();
        let first = r.started;
        r.begin();
        assert_eq!(first, r.started, "the second delta does not restart it");
        assert!(r.seconds().is_some());
        r.think_ms = Some(18_200.0);
        assert_eq!(r.seconds(), Some(18.2), "the measurement wins");
    }
}
