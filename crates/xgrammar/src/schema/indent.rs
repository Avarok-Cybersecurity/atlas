// SPDX-License-Identifier: AGPL-3.0-only
//
// IndentManager — port of `class IndentManager` from
// `cpp/json_schema_converter.cc`.
//
// Tracks the current indentation depth and produces the EBNF literal
// fragments that separate JSON elements (start/middle/end separators).

/// Manages indent depth and emits separator fragments for the EBNF
/// generation.
#[derive(Debug, Clone)]
pub struct IndentManager {
    any_whitespace: bool,
    enable_newline: bool,
    indent: i64,
    separator: String,
    total_indent: i64,
    /// One bool per nesting level: whether the next element is the
    /// first at that level.
    is_first: Vec<bool>,
    max_whitespace_cnt: Option<i32>,
}

impl IndentManager {
    /// Construct an indent manager. `indent == None` disables
    /// newlines. Port of the C++ constructor.
    pub fn new(
        indent: Option<i32>,
        separator: &str,
        any_whitespace: bool,
        max_whitespace_cnt: Option<i32>,
    ) -> Self {
        IndentManager {
            any_whitespace,
            enable_newline: indent.is_some(),
            indent: indent.unwrap_or(0) as i64,
            separator: separator.to_string(),
            total_indent: 0,
            is_first: vec![true],
            max_whitespace_cnt,
        }
    }

    /// The whitespace fragment for `any_whitespace` mode.
    fn whitespace_part(&self) -> String {
        match self.max_whitespace_cnt {
            None => "[ \\n\\t]*".to_string(),
            Some(n) => format!("[ \\n\\t]{{0,{n}}}"),
        }
    }

    /// Enter a deeper indentation level. Port of `StartIndent`.
    pub fn start_indent(&mut self) {
        self.total_indent += self.indent;
        self.is_first.push(true);
    }

    /// Leave the current indentation level. Port of `EndIndent`.
    pub fn end_indent(&mut self) {
        self.total_indent -= self.indent;
        self.is_first.pop();
    }

    /// Separator before the first element of a container.
    pub fn start_separator(&self) -> String {
        if self.any_whitespace {
            return self.whitespace_part();
        }
        if !self.enable_newline {
            return "\"\"".to_string();
        }
        format!("\"\\n{}\"", " ".repeat(self.total_indent.max(0) as usize))
    }

    /// Separator between elements of a container.
    pub fn middle_separator(&self) -> String {
        if self.any_whitespace {
            let ws = self.whitespace_part();
            return format!("{ws} \"{}\" {ws}", self.separator);
        }
        if !self.enable_newline {
            return format!("\"{}\"", self.separator);
        }
        format!(
            "\"{}\\n{}\"",
            self.separator,
            " ".repeat(self.total_indent.max(0) as usize)
        )
    }

    /// Separator after the last element of a container.
    pub fn end_separator(&self) -> String {
        if self.any_whitespace {
            return self.whitespace_part();
        }
        if !self.enable_newline {
            return "\"\"".to_string();
        }
        format!(
            "\"\\n{}\"",
            " ".repeat((self.total_indent - self.indent).max(0) as usize)
        )
    }

    /// Separator for an empty container.
    pub fn empty_separator(&self) -> String {
        if self.any_whitespace {
            return self.whitespace_part();
        }
        "\"\"".to_string()
    }

    /// The next contextual separator, advancing the `is_first` state.
    /// Port of `NextSeparator`.
    pub fn next_separator(&mut self, is_end: bool) -> String {
        if self.any_whitespace {
            let first = *self.is_first.last().unwrap_or(&true);
            if first || is_end {
                if let Some(last) = self.is_first.last_mut() {
                    *last = false;
                }
                return self.whitespace_part();
            }
            let ws = self.whitespace_part();
            return format!("{ws} \"{}\" {ws}", self.separator);
        }

        let mut res = String::new();
        let first = *self.is_first.last().unwrap_or(&true);
        if !first && !is_end {
            res.push_str(&self.separator);
        }
        if let Some(last) = self.is_first.last_mut() {
            *last = false;
        }
        if self.enable_newline {
            res.push_str("\\n");
        }
        let pad = if is_end {
            self.total_indent - self.indent
        } else {
            self.total_indent
        };
        res.push_str(&" ".repeat(pad.max(0) as usize));
        format!("\"{res}\"")
    }
}
