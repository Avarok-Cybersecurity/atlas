// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNFParser — the `TagDispatch` macro. Split out of `parser_macro.rs`
// to keep each file under the 250-line cap. Port of
// `EBNFParser::ParseTagDispatch` from xgrammar `cpp/grammar_parser.cc`.

use super::macros::MacroValue;
use super::{EbnfParser, ParseError};
use crate::grammar::builder::TagDispatchSpec;

impl EbnfParser {
    /// Parse a `TagDispatch(...)` macro call into a `TagDispatch` expr.
    pub(super) fn parse_tag_dispatch(&mut self) -> Result<i32, ParseError> {
        self.consume(1); // consume `TagDispatch`
        let start = self.cur as i32;
        let args = self.parse_macro_arguments()?;
        let delta = start - self.cur as i32;

        let mut spec = TagDispatchSpec::default();
        for arg in &args.arguments {
            let (tag, rule_name) = self.tag_pair(arg, delta)?;
            let rule_id = self.builder.get_rule_id(&rule_name);
            if rule_id == -1 {
                return Err(
                    self.parse_error(&format!("Rule \"{rule_name}\" is not defined"), delta)
                );
            }
            spec.tag_rule_pairs.push((tag, rule_id));
        }

        spec.stop_eos = true;
        spec.loop_after_dispatch = true;
        for (name, value) in &args.named_arguments {
            match name.as_str() {
                "stop_eos" => {
                    spec.stop_eos = expect_bool(value).ok_or_else(|| {
                        self.parse_error("stop_eos must be a boolean literal", delta)
                    })?;
                }
                "loop_after_dispatch" => {
                    spec.loop_after_dispatch = expect_bool(value).ok_or_else(|| {
                        self.parse_error("loop_after_dispatch must be a boolean literal", delta)
                    })?;
                }
                "stop_str" => {
                    spec.stop_str = self.tuple_of_strings(value, delta, "Stop strings")?;
                }
                "excludes" => {
                    spec.excluded_str = self.tuple_of_strings(value, delta, "excluded strings")?;
                }
                _ => {}
            }
        }

        if !spec.stop_eos && spec.stop_str.is_empty() {
            return Err(self.parse_error(
                "The TagDispatch must have stop_eos=true or stop_str is not empty",
                delta,
            ));
        }
        for exclude in &spec.excluded_str {
            if spec.stop_str.contains(exclude) {
                return Err(self.parse_error(
                    &format!(
                        "The TagDispatch should not have a common stop_str and exclude_str: {exclude}"
                    ),
                    delta,
                ));
            }
        }
        Ok(self.builder.add_tag_dispatch(&spec))
    }

    /// Decode one `("tag", rule_name)` positional tag-dispatch argument.
    fn tag_pair(&self, arg: &MacroValue, delta: i32) -> Result<(String, String), ParseError> {
        let elems = match arg {
            MacroValue::Tuple(e) => e,
            _ => {
                return Err(self.parse_error("Each tag dispatch element must be a tuple", delta));
            }
        };
        if elems.len() != 2 {
            return Err(self.parse_error(
                "Each tag dispatch element must be a pair (tag, rule)",
                delta,
            ));
        }
        let tag = match &elems[0] {
            MacroValue::Str(s) if !s.is_empty() => s.clone(),
            _ => {
                return Err(self.parse_error("Tag must be a non-empty string literal", delta));
            }
        };
        let rule_name = match &elems[1] {
            MacroValue::Ident(s) => s.clone(),
            _ => {
                return Err(self.parse_error("Rule reference must be an identifier", delta));
            }
        };
        Ok((tag, rule_name))
    }

    /// Decode a tuple-of-non-empty-strings macro value.
    fn tuple_of_strings(
        &self,
        value: &MacroValue,
        delta: i32,
        what: &str,
    ) -> Result<Vec<String>, ParseError> {
        let elems = match value {
            MacroValue::Tuple(e) => e,
            _ => return Err(self.parse_error(&format!("{what} must be a tuple"), delta)),
        };
        let mut out: Vec<String> = Vec::with_capacity(elems.len());
        for e in elems {
            match e {
                MacroValue::Str(s) if !s.is_empty() => out.push(s.clone()),
                _ => {
                    return Err(
                        self.parse_error("Stop string must be a non-empty string literal", delta)
                    );
                }
            }
        }
        Ok(out)
    }
}

/// Extract a boolean from a macro value, or `None`.
fn expect_bool(v: &MacroValue) -> Option<bool> {
    match v {
        MacroValue::Bool(b) => Some(*b),
        _ => None,
    }
}
