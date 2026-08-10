// SPDX-License-Identifier: AGPL-3.0-only

//! [`Section`] — the sidebar's navigation model, and the SSOT three things read.
//!
//! The sidebar draws rows from it, Tab steps through those rows, and the mouse
//! handler maps a clicked row back to a section. Those three used to carry
//! their own hardcoded copies of the order and the subsection labels, which is
//! exactly how Tab came to skip past rows the sidebar was drawing. Adding a
//! section means editing this file and nothing else.

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Section {
    Main,
    Stats,
    Network,
    Library,
    Benchmarks,
    Terminal,
}

impl Section {
    pub const ALL: [Section; 6] = [
        Section::Main,
        Section::Stats,
        Section::Network,
        Section::Library,
        Section::Benchmarks,
        Section::Terminal,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Section::Main => "Main",
            Section::Stats => "Stats",
            Section::Network => "Network",
            Section::Library => "Library",
            Section::Benchmarks => "Benchmarks",
            Section::Terminal => "Terminal",
        }
    }
    pub fn icon(self) -> &'static str {
        match self {
            Section::Main => "◆",
            Section::Stats => "∿",
            Section::Network => "⬡",
            Section::Library => "▤",
            Section::Benchmarks => "▰",
            Section::Terminal => "❯",
        }
    }
    /// Subsection labels, in sidebar order. SSOT for three things that must agree:
    /// what the sidebar draws, what a repeat section-key press cycles, and what
    /// `⇥` stops on. They were three separate hardcoded lists, so `⇥` skipped
    /// straight past the subsection rows the sidebar was drawing.
    pub fn subs(self) -> &'static [&'static str] {
        match self {
            Section::Main => &["Overview", "Kernels"],
            Section::Benchmarks => &["Suite", "History"],
            Section::Terminal => &["Ops", "Chat"],
            Section::Stats | Section::Network | Section::Library => &[],
        }
    }
}

#[cfg(test)]
#[path = "section_tests.rs"]
mod tests;
