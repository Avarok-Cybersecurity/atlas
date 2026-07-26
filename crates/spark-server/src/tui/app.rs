// SPDX-License-Identifier: AGPL-3.0-only

//! App state + input reducer. Rendering lives in `render/`; this file owns
//! what IS, and how key/mouse events change it.

use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::chat::ChatState;
use super::data::kernels::KernelTableModel;
use super::data::library::LibraryEntry;
use super::data::metrics_poll::StatsModel;
use super::progress::ProgressModel;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Section {
    Main,
    Stats,
    Network,
    Library,
    Terminal,
}

impl Section {
    pub const ALL: [Section; 5] = [
        Section::Main,
        Section::Stats,
        Section::Network,
        Section::Library,
        Section::Terminal,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Section::Main => "Main",
            Section::Stats => "Stats",
            Section::Network => "Network",
            Section::Library => "Library",
            Section::Terminal => "Terminal",
        }
    }
    pub fn icon(self) -> &'static str {
        match self {
            Section::Main => "◆",
            Section::Stats => "∿",
            Section::Network => "⬡",
            Section::Library => "▤",
            Section::Terminal => "❯",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum MainSub {
    Overview,
    Kernels,
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
    pub section: Section,
    pub main_sub: MainSub,
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
    pub kernel_scroll: usize,
    pub kernel_filter: String,
    pub library: Vec<LibraryEntry>,
    pub lib_selected: usize,
    pub lib_filter: String,
    pub lib_filter_editing: bool,
    pub network_selected: usize,
    pub network_detail: bool,
    pub ops: OpsState,
    pub chat: ChatState,
    pub toasts: Vec<Toast>,
    pub help_open: bool,
    pub tick: u64,
    pub should_quit: bool,
    pub detach: bool,
}

impl App {
    pub fn new(args: crate::cli::ServeArgs) -> Self {
        Self {
            args,
            section: Section::Main,
            main_sub: MainSub::Overview,
            term_sub: TermSub::Ops,
            focus: Focus::Content,
            progress: ProgressModel::default(),
            stats: StatsModel::default(),
            started: Instant::now(),
            log_scroll: None,
            log_filter: String::new(),
            log_filter_editing: false,
            kernels: None,
            kernel_scroll: 0,
            kernel_filter: String::new(),
            library: Vec::new(),
            lib_selected: 0,
            lib_filter: String::new(),
            lib_filter_editing: false,
            network_selected: 0,
            network_detail: false,
            ops: OpsState::default(),
            chat: ChatState::default(),
            toasts: Vec::new(),
            help_open: false,
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
        // Refresh the kernel table once when startup completes.
        if self.progress.ready && self.kernels.is_none() {
            let model = super::data::kernels::build();
            if !model.missing.is_empty() {
                let n = model.missing.len();
                self.toast(
                    format!("{n} kernel lookup(s) unresolved — Main ▸ Kernels"),
                    false,
                );
            }
            self.kernels = Some(model);
        }
    }

    /// True when a text input owns the keyboard.
    fn in_input(&self) -> bool {
        self.log_filter_editing
            || self.lib_filter_editing
            || (self.section == Section::Terminal && self.focus == Focus::Input)
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        // Ctrl+C always requests clean shutdown (raw mode swallows SIGINT).
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            super::shutdown::request("Ctrl+C");
            self.should_quit = true;
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
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('?') => self.help_open = true,
            KeyCode::Char('1') => self.jump(Section::Main),
            KeyCode::Char('2') => self.jump(Section::Stats),
            KeyCode::Char('3') => self.jump(Section::Network),
            KeyCode::Char('4') => self.jump(Section::Library),
            KeyCode::Char('5') => self.jump(Section::Terminal),
            KeyCode::Tab => self.cycle_section(1),
            KeyCode::BackTab => self.cycle_section(-1),
            KeyCode::Char('f') if self.section == Section::Main => {
                self.log_filter_editing = true;
            }
            KeyCode::Char('/') if self.section == Section::Library => {
                self.lib_filter_editing = true;
            }
            KeyCode::Char('i') | KeyCode::Enter
                if self.section == Section::Terminal && self.focus != Focus::Input =>
            {
                self.focus = Focus::Input;
            }
            KeyCode::Esc => {
                self.focus = Focus::Content;
                self.log_scroll = None;
            }
            _ => self.on_section_key(key),
        }
    }

    fn jump(&mut self, s: Section) {
        if self.section == s {
            // Repeat-press toggles the section's subsections.
            match s {
                Section::Main => {
                    self.main_sub = if self.main_sub == MainSub::Overview {
                        MainSub::Kernels
                    } else {
                        MainSub::Overview
                    };
                }
                Section::Terminal => {
                    self.term_sub = if self.term_sub == TermSub::Ops {
                        TermSub::Chat
                    } else {
                        TermSub::Ops
                    };
                }
                _ => {}
            }
        }
        self.section = s;
        self.focus = Focus::Content;
    }

    fn cycle_section(&mut self, dir: i32) {
        let i = Section::ALL
            .iter()
            .position(|s| *s == self.section)
            .unwrap_or(0) as i32;
        let n = Section::ALL.len() as i32;
        self.section = Section::ALL[((i + dir).rem_euclid(n)) as usize];
    }

    fn on_section_key(&mut self, key: KeyEvent) {
        let down = matches!(key.code, KeyCode::Down | KeyCode::Char('j'));
        let up = matches!(key.code, KeyCode::Up | KeyCode::Char('k'));
        match self.section {
            Section::Main => match self.main_sub {
                MainSub::Overview => {
                    if up {
                        let cur = self.log_scroll.unwrap_or(0);
                        self.log_scroll = Some(cur + 1);
                    } else if down {
                        match self.log_scroll {
                            Some(1) | None => self.log_scroll = None,
                            Some(n) => self.log_scroll = Some(n - 1),
                        }
                    } else if matches!(key.code, KeyCode::Char('G') | KeyCode::End) {
                        self.log_scroll = None;
                    }
                }
                MainSub::Kernels => {
                    if down {
                        self.kernel_scroll = self.kernel_scroll.saturating_add(1);
                    } else if up {
                        self.kernel_scroll = self.kernel_scroll.saturating_sub(1);
                    } else if matches!(key.code, KeyCode::Char('g')) {
                        self.kernel_scroll = 0;
                    }
                }
            },
            Section::Library => {
                let len = self.filtered_library().len();
                if down && len > 0 {
                    self.lib_selected = (self.lib_selected + 1).min(len - 1);
                } else if up {
                    self.lib_selected = self.lib_selected.saturating_sub(1);
                }
            }
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
            Section::Terminal | Section::Stats => {}
        }
    }

    fn on_input_key(&mut self, key: KeyEvent) {
        // Which buffer?
        if self.log_filter_editing {
            edit_line(&mut self.log_filter, key, &mut self.log_filter_editing);
            return;
        }
        if self.lib_filter_editing {
            edit_line(&mut self.lib_filter, key, &mut self.lib_filter_editing);
            self.lib_selected = 0;
            return;
        }
        // Terminal input.
        match self.term_sub {
            TermSub::Ops => match key.code {
                KeyCode::Esc => self.focus = Focus::Content,
                KeyCode::Enter => {
                    let line = std::mem::take(&mut self.ops.input);
                    if !line.trim().is_empty() {
                        self.ops.history.push(line.clone());
                        self.ops.history_pos = None;
                        super::commands::execute(&line, self);
                    }
                }
                KeyCode::Up => {
                    let h = &self.ops.history;
                    if !h.is_empty() {
                        let pos = match self.ops.history_pos {
                            None => h.len() - 1,
                            Some(p) => p.saturating_sub(1),
                        };
                        self.ops.history_pos = Some(pos);
                        self.ops.input = h[pos].clone();
                    }
                }
                KeyCode::Backspace => {
                    self.ops.input.pop();
                }
                KeyCode::Char(c) => self.ops.input.push(c),
                _ => {}
            },
            TermSub::Chat => match key.code {
                KeyCode::Esc => {
                    self.chat.cancel();
                    self.focus = Focus::Content;
                }
                // Enter sends; a trailing backslash continues onto a new
                // line (Ctrl+Enter is indistinguishable from Enter in legacy
                // terminal protocols, so it cannot be the only send chord).
                KeyCode::Enter => {
                    if let Some(stripped) = self.chat.input.strip_suffix('\\') {
                        self.chat.input = format!("{stripped}\n");
                    } else {
                        self.chat.send(self.args.port);
                    }
                }
                KeyCode::Backspace => {
                    self.chat.input.pop();
                }
                KeyCode::Char(c) => self.chat.input.push(c),
                _ => {}
            },
        }
    }

    pub fn filtered_library(&self) -> Vec<&LibraryEntry> {
        let f = self.lib_filter.to_lowercase();
        self.library
            .iter()
            .filter(|e| f.is_empty() || e.id.to_lowercase().contains(&f))
            .collect()
    }

    pub fn sidebar_click(&mut self, row_in_sidebar: usize) {
        // Rows are laid out by render/mod.rs: one section per visual row,
        // in Section::ALL order (subsection rows are handled there).
        if let Some(s) = Section::ALL.get(row_in_sidebar) {
            self.jump(*s);
        }
    }
}

/// Minimal single-line editor for the two filter boxes.
fn edit_line(buf: &mut String, key: KeyEvent, editing: &mut bool) {
    match key.code {
        KeyCode::Esc => {
            buf.clear();
            *editing = false;
        }
        KeyCode::Enter => *editing = false,
        KeyCode::Backspace => {
            buf.pop();
        }
        KeyCode::Char(c) => buf.push(c),
        _ => {}
    }
}
