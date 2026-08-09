// SPDX-License-Identifier: AGPL-3.0-only

//! Chat tab state: the transcript, the view preferences, and the reducer that
//! turns streamed deltas into it. The HTTP/SSE half lives in `chat_stream`.
//!
//! The request runs on the tokio runtime (Handle captured at TUI start); the
//! stream's deltas cross to the TUI thread over a std mpsc that the event
//! loop drains each tick.

use std::sync::mpsc::{Receiver, channel};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::chat_thinking::{Reasoning, ThinkingRequest, ThinkingView};

#[derive(Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Model,
}

pub struct ChatMessage {
    pub role: Role,
    pub text: String,
    /// The reasoning trace, if the model emitted one. Empty for every user
    /// message and for any model that did not think.
    pub reasoning: Reasoning,
    /// Time to the first delta of ANY kind — the honest TTFT.
    pub ttft_ms: Option<f64>,
    /// Time to the first delta the user can READ. A genuinely different
    /// number when the model thinks first, and the one that explains an
    /// apparently blank pane.
    pub answer_ttft_ms: Option<f64>,
    pub tok_per_s: Option<f64>,
    pub tokens: usize,
    /// The reply has finished, one way or another. Distinguishes "no answer
    /// arrived" from "the answer has not started yet", which look identical
    /// in the transcript and must not read the same on screen.
    pub done: bool,
}

impl ChatMessage {
    pub fn new(role: Role, text: String) -> Self {
        Self {
            role,
            text,
            reasoning: Reasoning::default(),
            ttft_ms: None,
            answer_ttft_ms: None,
            tok_per_s: None,
            tokens: 0,
            done: false,
        }
    }

    /// A completed model reply that produced no readable answer. Observed with
    /// `response_format` + thinking on: 2 of 4 requests returned reasoning and
    /// nothing else.
    pub fn is_answerless(&self) -> bool {
        self.role == Role::Model && self.done && self.text.is_empty()
    }
}

pub enum ChatDelta {
    Token(String),
    Reasoning(String),
    Done {
        ttft_ms: Option<f64>,
        answer_ttft_ms: Option<f64>,
        think_ms: Option<f64>,
        tok_per_s: Option<f64>,
        tokens: usize,
        reasoning_tokens: usize,
    },
    Error(String),
}

impl ChatDelta {
    /// A terminal delta carrying no measurement — what a cancelled stream
    /// reports. It must NOT be an `Error`: that would overwrite the partial
    /// reply already on screen with the word "cancelled".
    pub(super) fn cancelled() -> Self {
        Self::Done {
            ttft_ms: None,
            answer_ttft_ms: None,
            think_ms: None,
            tok_per_s: None,
            tokens: 0,
            reasoning_tokens: 0,
        }
    }
}

#[derive(Default)]
pub struct ChatState {
    pub transcript: Vec<ChatMessage>,
    pub input: String,
    pub streaming: bool,
    /// What the next message ASKS the model to do about thinking. A session
    /// preference: it persists until changed and applies to the next send,
    /// never retroactively.
    pub think_req: ThinkingRequest,
    /// How an arriving reasoning trace is drawn. Also a session preference —
    /// a view choice, not a property of one reply.
    pub think_view: ThinkingView,
    /// Whether the LAST completed reply actually thought. An observation, so
    /// `Auto` can report what the model resolved to instead of guessing it.
    /// Cleared whenever the request state changes, because the old answer no
    /// longer describes what the next request will ask for.
    pub observed_thinking: Option<bool>,
    /// Transcript viewport, in WRAPPED rows above the bottom. `None` follows the
    /// streaming tip; `Some(n)` holds station n rows up. Same contract as the Main
    /// log pane's `log_scroll`, so both panes answer to the same keys.
    pub scroll: Option<usize>,
    rx: Option<Receiver<ChatDelta>>,
    cancel: Option<tokio::sync::oneshot::Sender<()>>,
    runtime: Option<tokio::runtime::Handle>,
}

impl ChatState {
    pub fn set_runtime(&mut self, handle: tokio::runtime::Handle) {
        self.runtime = Some(handle);
    }

    /// Scroll the transcript by `rows` (positive = back toward older turns).
    /// Landing at or past the bottom restores follow, so a stream that is running
    /// keeps painting its tip without a second keypress.
    pub fn scroll_by(&mut self, rows: i32) {
        let cur = self.scroll.unwrap_or(0) as i32;
        let next = cur + rows;
        self.scroll = if next <= 0 { None } else { Some(next as usize) };
    }

    /// Snap back to the live tip.
    pub fn follow(&mut self) {
        self.scroll = None;
    }

    /// Ask for more, less, or the model's own choice of thinking.
    pub fn cycle_request(&mut self) -> ThinkingRequest {
        self.think_req = self.think_req.next();
        // The observation described the PREVIOUS request. Keeping it would
        // caption the new state with the old model's behavior.
        self.observed_thinking = None;
        self.think_req
    }

    /// Collapse / expand / hide the reasoning block. Display only — the wire
    /// is untouched.
    pub fn cycle_view(&mut self) -> ThinkingView {
        self.think_view = self.think_view.next();
        self.think_view
    }

    /// Chat keys that work whether or not the input box has focus.
    ///
    /// `typing` is true when a bare letter belongs to the input buffer, so
    /// only the chorded forms count there — `t` must still type a `t`.
    /// Returns what to say about it, if anything.
    pub fn on_view_key(&mut self, key: KeyEvent, typing: bool) -> Option<String> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            // Alt first: `Alt+t` still reports `Char('t')`, so an unguarded
            // `t` arm would swallow it and the display toggle would be dead.
            KeyCode::Char('t') if alt => {
                Some(format!("chat: reasoning {}", self.cycle_view().label()))
            }
            KeyCode::Char('T') if !typing => {
                Some(format!("chat: reasoning {}", self.cycle_view().label()))
            }
            KeyCode::Char('t') if ctrl || !typing => {
                let chip = self.cycle_request().chip(None, true);
                Some(format!("chat: {chip} — applies to the next message"))
            }
            _ => None,
        }
    }

    /// Transcript navigation when the pane, not the input box, has focus.
    pub fn on_content_key(&mut self, key: KeyEvent) -> Option<String> {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.scroll_by(1),
            KeyCode::Down | KeyCode::Char('j') => self.scroll_by(-1),
            KeyCode::PageUp => self.scroll_by(10),
            KeyCode::PageDown => self.scroll_by(-10),
            KeyCode::Char('G') | KeyCode::End => self.follow(),
            _ => return self.on_view_key(key, false),
        }
        None
    }

    /// Send the current input as a user message and start streaming a reply.
    pub fn send(&mut self, port: u16) {
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() || self.streaming {
            return;
        }
        self.input.clear();
        // Sending is an explicit "show me the new reply" — resume follow.
        self.follow();
        self.transcript.push(ChatMessage::new(Role::User, prompt));
        self.transcript
            .push(ChatMessage::new(Role::Model, String::new()));
        let Some(rt) = self.runtime.clone() else {
            if let Some(last) = self.transcript.last_mut() {
                last.text = "(chat unavailable: no runtime handle)".into();
                last.done = true;
            }
            return;
        };
        let (tx, rx) = channel();
        let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
        self.rx = Some(rx);
        self.cancel = Some(cancel_tx);
        self.streaming = true;
        let thinking = self.think_req;
        // History for multi-turn, excluding the empty model placeholder that
        // `send` just pushed.
        //
        // ★ Filter by POSITION, not by property. The placeholder is always the
        // LAST element; an EARLIER model turn can legitimately be empty — every
        // cancelled reply is one, and so is the documented `response_format` +
        // thinking case. Dropping those left two consecutive `user` messages on
        // the wire, which some chat templates reject outright. Keeping the
        // empty assistant turn preserves the alternation the templates need,
        // and it is truthful: the model really did answer nothing.
        let history = match self.transcript.last() {
            Some(m) if m.role == Role::Model && m.text.is_empty() => {
                &self.transcript[..self.transcript.len() - 1]
            }
            _ => &self.transcript[..],
        };
        let messages: Vec<(String, String)> = history
            .iter()
            .map(|m| {
                (
                    match m.role {
                        Role::User => "user".to_string(),
                        Role::Model => "assistant".to_string(),
                    },
                    m.text.clone(),
                )
            })
            .collect();
        rt.spawn(async move {
            tokio::select! {
                _ = super::chat_stream::stream_chat(port, messages, thinking, tx.clone()) => {}
                _ = cancel_rx => {
                    let _ = tx.send(ChatDelta::cancelled());
                }
            }
        });
    }

    pub fn cancel(&mut self) {
        if let Some(c) = self.cancel.take() {
            let _ = c.send(());
        }
    }

    /// End the stream, whatever ended it.
    fn settle(&mut self) {
        if let Some(m) = self.transcript.last_mut() {
            m.done = true;
            // What Auto resolved to, read off what actually arrived rather
            // than off what the client hoped for. A reply that produced
            // NOTHING — cancelled, or refused — observed nothing, and
            // recording `false` there would caption Auto with a claim the
            // model never got the chance to make.
            if m.tokens > 0 || !m.reasoning.is_empty() {
                self.observed_thinking = Some(!m.reasoning.is_empty());
            }
        }
        self.streaming = false;
        self.rx = None;
        self.cancel = None;
    }

    /// Drain pending deltas into the transcript (event-loop tick).
    pub fn pump(&mut self) {
        let Some(rx) = &self.rx else { return };
        let mut deltas: Vec<ChatDelta> = rx.try_iter().collect();
        // `try_iter` stops on Empty and Disconnected alike, so a sender that
        // died without a terminal delta — a panic in the streaming task — is
        // indistinguishable from "nothing yet". Left undetected it pins
        // `streaming` true, and `send` refuses while that is set: the chat
        // pane locks up for the rest of the process with no indication why.
        // The recipe fetch has handled this case since it was written; this
        // did not.
        //
        // ONE `try_recv`, and its value is kept. A delta can arrive between
        // `try_iter` ending and this call, and discarding it to learn whether
        // the channel is alive would drop a token off the reply.
        let mut disconnected = false;
        if deltas.is_empty() && self.streaming {
            match rx.try_recv() {
                Ok(d) => deltas.push(d),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => disconnected = true,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if disconnected {
            if let Some(m) = self.transcript.last_mut()
                && m.text.is_empty()
                && m.reasoning.is_empty()
            {
                m.text = "(the reply ended without finishing)".into();
            }
            self.settle();
            return;
        }
        for d in deltas {
            match d {
                ChatDelta::Reasoning(t) => {
                    if let Some(m) = self.transcript.last_mut() {
                        m.reasoning.begin();
                        m.reasoning.text.push_str(&t);
                        m.reasoning.tokens += 1;
                    }
                }
                ChatDelta::Token(t) => {
                    if let Some(m) = self.transcript.last_mut() {
                        if m.text.is_empty() {
                            // The answer has started: stop the thinking clock.
                            m.reasoning.seal();
                        }
                        m.text.push_str(&t);
                        m.tokens += 1;
                    }
                }
                ChatDelta::Done {
                    ttft_ms,
                    answer_ttft_ms,
                    think_ms,
                    tok_per_s,
                    tokens,
                    reasoning_tokens,
                } => {
                    if let Some(m) = self.transcript.last_mut() {
                        m.ttft_ms = ttft_ms;
                        m.answer_ttft_ms = answer_ttft_ms;
                        m.tok_per_s = tok_per_s;
                        // Only a real measurement replaces the sealed
                        // estimate — a cancel reports `None` for everything,
                        // and taking it would restart the live timer on a
                        // reply that has stopped.
                        if think_ms.is_some() {
                            m.reasoning.think_ms = think_ms;
                        }
                        if tokens > 0 {
                            m.tokens = tokens;
                        }
                        if reasoning_tokens > 0 {
                            m.reasoning.tokens = reasoning_tokens;
                        }
                    }
                    self.settle();
                    return;
                }
                ChatDelta::Error(e) => {
                    if let Some(m) = self.transcript.last_mut() {
                        m.text = format!("(error: {e})");
                    }
                    self.settle();
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "chat_tests.rs"]
mod pump_tests;

#[cfg(test)]
#[path = "chat_more_tests.rs"]
mod turn_tests;

#[cfg(test)]
#[path = "chat_history_tests.rs"]
mod chat_history_tests;
