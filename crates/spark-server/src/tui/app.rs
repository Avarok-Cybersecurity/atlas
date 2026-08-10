// SPDX-License-Identifier: AGPL-3.0-only

//! App state + input reducer. Rendering lives in `render/`; this file owns
//! what IS, and how key/mouse events change it.

use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

pub use super::section::Section;

use super::bench_state::BenchState;
use super::chat::ChatState;
use super::data::kernels::KernelTableModel;
use super::data::library::LibraryEntry;
use super::data::metrics_poll::StatsModel;
use super::progress::ProgressModel;

#[derive(Clone, Copy, PartialEq)]
pub enum MainSub {
    Overview,
    Kernels,
}

#[derive(Clone, Copy, PartialEq)]
pub enum BenchSub {
    Suite,
    History,
}

#[derive(Clone, Copy, PartialEq)]
pub enum TermSub {
    Ops,
    Chat,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Focus {
    Sidebar,
    Content,
    Input,
}

pub struct Toast {
    pub text: String,
    pub error: bool,
    pub at: Instant,
}

/// Ops REPL state.
#[derive(Default)]
pub struct OpsState {
    pub input: String,
    pub history: Vec<String>,
    pub history_pos: Option<usize>,
    pub output: Vec<String>,
    pub scroll_up: usize,
}

pub struct App {
    pub args: crate::cli::ServeArgs,
    /// The model cell, so the Library can start or replace a model.
    ///
    /// `None` only in tests, which build an App without a server behind it.
    pub host: Option<std::sync::Arc<crate::main_modules::model_host::ModelHost>>,
    /// Ask the event loop for a full repaint on the next frame.
    ///
    /// ratatui diffs its new buffer against the LAST BUFFER IT DREW, never
    /// against the terminal. Once the two diverge — a foreign write, a pane
    /// resize tmux reflowed — the diff emits nothing for those cells and the
    /// stale glyphs are permanent: a Library hint stayed on screen through a
    /// model load and a section change, unreachable by any amount of redrawing.
    /// Only a clear re-establishes the baseline, so it is requested at the
    /// moments the layout changes wholesale rather than every frame.
    pub repaint: bool,
    /// No model loaded, so the startup checklist is not tracking anything.
    /// Cleared the moment a load starts, including one begun from the Library.
    pub awaiting_model: bool,
    pub section: Section,
    pub main_sub: MainSub,
    pub bench_sub: BenchSub,
    pub term_sub: TermSub,
    pub focus: Focus,
    pub progress: ProgressModel,
    pub stats: StatsModel,
    pub started: Instant,
    /// None = follow newest; Some(n) = scrolled up by n lines.
    pub log_scroll: Option<usize>,
    pub log_filter: String,
    pub log_filter_editing: bool,
    pub kernels: Option<KernelTableModel>,
    /// Which model the kernel table describes.
    ///
    /// The table is built from the audit a LOAD populates. Keyed on
    /// `kernels.is_none()` alone it was built once, and since this branch made
    /// the no-model boot report `progress.ready` (the listener is up, not a
    /// model), that once was at boot with an empty audit — every module
    /// unresolved, a spurious "N kernel lookup(s) unresolved" toast, and the
    /// pane frozen that way through every model loaded afterwards.
    pub kernels_for: Option<String>,
    pub kernel_scroll: usize,
    pub kernel_filter: String,
    pub library: Vec<LibraryEntry>,
    /// The local cache needs (re)scanning.
    ///
    /// Was a `let mut library_scanned = false` local to the event loop, which
    /// made "scan once, lazily" easy and "scan again after a download"
    /// impossible. Set by a finished or cancelled download, and by a manual
    /// refresh; cleared when a scan is started.
    pub library_dirty: bool,
    /// The Library section: joined recipes + local weights, and the edit form.
    pub lib: crate::tui::lib_state::LibState,
    /// Model downloads and update checks.
    pub download: crate::tui::download_state::DownloadState,
    /// Scroll ceilings, published by the renderer. See `app_scroll`.
    pub log_scroll_max: std::cell::Cell<usize>,
    pub kernel_scroll_max: std::cell::Cell<usize>,
    pub chat_scroll_max: std::cell::Cell<usize>,
    /// An in-progress or just-finished mouse selection, in cell coordinates.
    ///
    /// Lives on `App` rather than in the event loop because the RENDERER needs
    /// it — the highlight is drawn from here — and because a selection must
    /// survive the ticks between button-down and button-up.
    pub selection: Option<crate::tui::selection::Selection>,
    pub network_selected: usize,
    pub network_detail: bool,
    pub ops: OpsState,
    pub chat: ChatState,
    pub bench: BenchState,
    /// The running scheduler's levers, published by `serve` once the run
    /// starts. `None` until then — the ops commands that toggle a lever say
    /// so rather than silently doing nothing.
    pub run: Option<crate::tui::RunHandles>,
    pub toasts: Vec<Toast>,
    pub help_open: bool,
    /// A `q` that would have destroyed work in flight, waiting to be answered.
    /// Set only when [`App::work_in_flight`] named something; an idle
    /// dashboard still quits on the first press.
    pub confirm_quit: bool,
    pub tick: u64,
    pub should_quit: bool,
    pub detach: bool,
}

impl App {
    pub fn new(args: crate::cli::ServeArgs) -> Self {
        // With no model there is nothing for Main to show — its checklist would
        // sit at 0/12 and its status pill would read LOADING for a load that is
        // not happening. The Library is the only screen that can move the user
        // forward, so it is where a no-argument boot lands.
        let awaiting_model = args.model.is_none() && args.model_from_path.is_none();
        Self {
            awaiting_model,
            repaint: false,
            args,
            host: None,
            section: if awaiting_model {
                Section::Library
            } else {
                Section::Main
            },
            main_sub: MainSub::Overview,
            bench_sub: BenchSub::Suite,
            run: None,
            term_sub: TermSub::Ops,
            focus: Focus::Content,
            progress: ProgressModel::default(),
            stats: StatsModel::default(),
            started: Instant::now(),
            log_scroll: None,
            log_filter: String::new(),
            log_filter_editing: false,
            kernels: None,
            kernels_for: None,
            kernel_scroll: 0,
            kernel_filter: String::new(),
            library: Vec::new(),
            library_dirty: true,
            download: Default::default(),
            selection: None,
            log_scroll_max: std::cell::Cell::new(0),
            kernel_scroll_max: std::cell::Cell::new(0),
            chat_scroll_max: std::cell::Cell::new(0),
            lib: Default::default(),
            network_selected: 0,
            network_detail: false,
            ops: OpsState::default(),
            chat: ChatState::default(),
            bench: BenchState::default(),
            toasts: Vec::new(),
            help_open: false,
            confirm_quit: false,
            tick: 0,
            should_quit: false,
            detach: false,
        }
    }

    pub fn toast(&mut self, text: impl Into<String>, error: bool) {
        self.toasts.push(Toast {
            text: text.into(),
            error,
            at: Instant::now(),
        });
        if self.toasts.len() > 3 {
            self.toasts.remove(0);
        }
    }

    pub fn on_tick(&mut self) {
        self.tick += 1;
        self.progress.ease_tick();
        // Info toasts auto-dismiss after 5s; errors persist.
        self.toasts
            .retain(|t| t.error || t.at.elapsed().as_secs() < 5);
        // A launch that failed after its thread started must not leave the
        // dashboard showing a load. The checklist was reset and the pill set
        // to LOADING the moment the thread spawned; if the swap then refused
        // — no compiled kernels for that model, say — nothing else would ever
        // correct them.
        if let Some(err) = self.lib.poll_launch() {
            let live = self.host.as_ref().and_then(|h| h.live_model());
            self.awaiting_model = live.is_none();
            self.progress.reset();
            self.repaint = true;
            self.toast(super::lib_state::problem_line(&err), true);
        }

        // Keep the benchmark target on the model that is actually serving.
        if let Some(name) = self
            .host
            .as_ref()
            .and_then(|h| h.live_model())
            .or_else(|| self.args.model_name.clone())
            .or_else(|| self.args.model.clone())
        {
            self.bench.follow_live_model(&name);
        }
        // Rebuild the kernel table when a DIFFERENT model finishes loading —
        // including after a swap, which no `is_none()` guard would notice.
        let live = self
            .host
            .as_ref()
            .and_then(|h| h.live_model())
            .or_else(|| self.args.model_name.clone())
            .or_else(|| self.args.model.clone());
        if self.progress.ready && !self.awaiting_model && live.is_some() && self.kernels_for != live
        {
            let model = super::data::kernels::build();
            // ONLY the actionable class toasts: an alarm that is almost always
            // noise is an alarm nobody reads (see `kernel_audit::FailureSplit`).
            let n = model.missing_required.len();
            if n > 0 {
                let msg = format!("{n} kernel lookup(s) unresolved — Main ▸ Kernels");
                self.toast(msg, false);
            }
            self.kernels = Some(model);
            self.kernels_for = live;
        }
    }

    /// True when a text input owns the keyboard.
    fn in_input(&self) -> bool {
        self.log_filter_editing
            || (self.section == Section::Library && self.lib.is_editing())
            || (self.section == Section::Benchmarks && self.bench.is_editing())
            || (self.section == Section::Terminal && self.focus == Focus::Input)
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        // Any keystroke ends a selection. Its coordinates are SCREEN CELLS, so
        // the moment the screen changes underneath it — a section switch, a
        // scroll, a filter — it is highlighting different text than the user
        // chose, and on another section it highlights nothing meaningful at
        // all. Reported as a stale highlight following the user between
        // screens.
        self.selection = None;
        // Ctrl+C always requests clean shutdown (raw mode swallows SIGINT).
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            super::shutdown::request("Ctrl+C");
            self.should_quit = true;
            return;
        }
        // A pending confirmation owns the keyboard until it is answered.
        if self.confirm_quit && self.answer_quit_prompt(key) {
            return;
        }
        if self.help_open {
            self.help_open = false;
            return;
        }
        if self.in_input() {
            self.on_input_key(key);
            return;
        }
        match key.code {
            KeyCode::Char('q') => self.on_quit_key(),
            KeyCode::Char('?') => self.help_open = true,
            KeyCode::Char('1') => self.jump(Section::Main),
            KeyCode::Char('2') => self.jump(Section::Stats),
            KeyCode::Char('3') => self.jump(Section::Network),
            KeyCode::Char('4') => self.jump(Section::Library),
            KeyCode::Char('5') => self.jump(Section::Benchmarks),
            KeyCode::Char('6') => self.jump(Section::Terminal),
            KeyCode::Tab => self.cycle_section(1),
            KeyCode::BackTab => self.cycle_section(-1),
            KeyCode::Char('f') if self.section == Section::Main => {
                self.log_filter_editing = true;
            }
            KeyCode::Char('i') | KeyCode::Enter
                if self.section == Section::Terminal && self.focus != Focus::Input =>
            {
                self.focus = Focus::Input;
            }
            // Benchmarks and Library own everything the global bindings above
            // did not claim — Esc included, since there it means "back one
            // step" and not "drop focus".
            _ if self.section == Section::Benchmarks => self.on_section_key(key),
            _ if self.section == Section::Library => self.on_library_key(key),
            KeyCode::Esc => {
                self.focus = Focus::Content;
                self.log_scroll = None;
            }
            _ => self.on_section_key(key),
        }
    }

    /// Which subsection of `s` is active, as an index into [`Section::subs`].
    pub fn sub_index(&self, s: Section) -> usize {
        match s {
            Section::Main => (self.main_sub == MainSub::Kernels) as usize,
            Section::Benchmarks => (self.bench_sub == BenchSub::History) as usize,
            Section::Terminal => (self.term_sub == TermSub::Chat) as usize,
            _ => 0,
        }
    }

    fn set_sub(&mut self, s: Section, i: usize) {
        match s {
            Section::Main => {
                self.main_sub = if i == 0 {
                    MainSub::Overview
                } else {
                    MainSub::Kernels
                }
            }
            Section::Benchmarks => {
                self.bench_sub = if i == 0 {
                    BenchSub::Suite
                } else {
                    BenchSub::History
                }
            }
            Section::Terminal => self.term_sub = if i == 0 { TermSub::Ops } else { TermSub::Chat },
            _ => {}
        }
    }

    /// Every navigable sidebar row, flattened in the order the sidebar draws them:
    /// one entry per subsection, or a single entry for a section that has none.
    fn nav_rows() -> Vec<(Section, usize)> {
        Section::ALL
            .iter()
            .flat_map(|s| (0..s.subs().len().max(1)).map(move |i| (*s, i)))
            .collect()
    }

    pub(super) fn jump(&mut self, s: Section) {
        if self.section != s {
            self.repaint = true;
        }
        if self.section == s {
            // Repeat-press cycles this section's subsections.
            let n = s.subs().len();
            if n > 1 {
                self.set_sub(s, (self.sub_index(s) + 1) % n);
            }
        }
        self.section = s;
        self.focus = Focus::Content;
    }

    /// `⇥` / `⇧⇥` walk the sidebar exactly as drawn — subsection rows included.
    /// Previously they stepped over top-level sections only, so Main ▸ Kernels was
    /// reachable solely by pressing `1` a second time, which nothing on screen said.
    fn cycle_section(&mut self, dir: i32) {
        let rows = Self::nav_rows();
        let cur = rows
            .iter()
            .position(|(s, i)| *s == self.section && *i == self.sub_index(*s))
            .unwrap_or(0) as i32;
        let (s, i) = rows[((cur + dir).rem_euclid(rows.len() as i32)) as usize];
        self.section = s;
        self.set_sub(s, i);
        self.focus = Focus::Content;
    }

    fn on_section_key(&mut self, key: KeyEvent) {
        let down = matches!(key.code, KeyCode::Down | KeyCode::Char('j'));
        let up = matches!(key.code, KeyCode::Up | KeyCode::Char('k'));
        match self.section {
            // Both panes move through `scroll`, the same entry point the wheel
            // uses. A second copy of "what scrolling means here" is how the
            // keyboard came to have no ceiling while the wheel had one: `k`
            // past the oldest line blanked the pane, and coming back cost as
            // many presses as had been spent going up.
            Section::Main => match self.main_sub {
                MainSub::Overview => {
                    if up {
                        self.scroll(-1);
                    } else if down {
                        self.scroll(1);
                    } else if matches!(key.code, KeyCode::Char('G') | KeyCode::End) {
                        self.log_scroll = None;
                    }
                }
                MainSub::Kernels => {
                    if down {
                        self.scroll(1);
                    } else if up {
                        self.scroll(-1);
                    } else if matches!(key.code, KeyCode::Char('g')) {
                        self.kernel_scroll = 0;
                    }
                }
            },
            // The Library owns its own navigation in `lib_keys`; routing it
            // here too would give it two reducers disagreeing about selection.
            Section::Library => {}
            Section::Network => {
                let n = self.args.world_size.max(1);
                if matches!(key.code, KeyCode::Right | KeyCode::Char('l')) && n > 1 {
                    self.network_selected = (self.network_selected + 1).min(n - 1);
                } else if matches!(key.code, KeyCode::Left | KeyCode::Char('h')) {
                    self.network_selected = self.network_selected.saturating_sub(1);
                } else if key.code == KeyCode::Enter {
                    self.network_detail = !self.network_detail;
                }
            }
            // Chat owns its own keys in `app_input`, where the input-focused
            // half of the same map already lives.
            Section::Terminal if self.term_sub == TermSub::Chat => self.on_chat_content_key(key),
            Section::Benchmarks => self.on_bench_key(key),
            Section::Terminal | Section::Stats => {}
        }
    }

    /// Route into the Library reducer and surface whatever it wants said.
    pub(super) fn on_library_key(&mut self, key: KeyEvent) {
        match self.lib.on_key(key) {
            super::lib_keys::Outcome::Toast { text, error } => self.toast(text, error),
            super::lib_keys::Outcome::Launch => self.launch_selected_recipe(),
            super::lib_keys::Outcome::Download => self.download_selected_model(),
            super::lib_keys::Outcome::CancelDownload => match self.download.cancel() {
                Some((text, error)) => self.toast(text, error),
                None => self.toast("nothing is downloading".to_string(), false),
            },
            super::lib_keys::Outcome::CheckFresh => self.check_selected_model(),
            super::lib_keys::Outcome::None => {}
        }
    }

    /// Route into the Benchmarks reducer and surface whatever it wants said.
    pub(super) fn on_bench_key(&mut self, key: KeyEvent) {
        if let super::bench_keys::Outcome::Toast { text, error } =
            self.bench.on_key(key, self.bench_sub)
        {
            self.toast(text, error);
        }
    }

    pub fn sidebar_click(&mut self, row_in_sidebar: usize) {
        // Rows are laid out by render/mod.rs: one section per visual row,
        // in Section::ALL order (subsection rows are handled there).
        if let Some(s) = Section::ALL.get(row_in_sidebar) {
            self.jump(*s);
        }
    }

    /// Click on one of the ACTIVE section's subsection rows — the only ones the
    /// sidebar draws. Selects it outright rather than cycling, because a click
    /// names the row it landed on.
    pub fn sidebar_sub_click(&mut self, sub: usize) {
        if sub < self.section.subs().len() {
            self.set_sub(self.section, sub);
            self.focus = Focus::Content;
        }
    }
}

// Nav-order tests live in their own mount to keep this file under the 500 LoC cap.
#[cfg(test)]
#[path = "app_tests.rs"]
mod tests;

// The key-by-key drive of the global bindings, same reason.
#[cfg(test)]
#[path = "app_keys_tests.rs"]
mod key_tests;
