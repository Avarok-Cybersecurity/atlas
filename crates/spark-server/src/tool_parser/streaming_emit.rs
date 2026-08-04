// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

//! Live argument-fragment emission for the streaming tool-call detector.
//!
//! Split out of `streaming_impl.rs` (≤500 LoC cap): the two helpers that turn
//! newly-completed `<parameter>` blocks / JSON arg bytes into incremental
//! `ToolCallArgsFragment`s. `process()`/`flush()` in `streaming_impl.rs` call
//! these via `self` (hence `pub(super)`).

use super::*;

impl StreamingToolDetector {
    /// Coerce a single XML `<parameter=KEY>VALUE</parameter>` pair to its
    /// canonical schema-normalised key and a JSON-encoded value string. Reuses
    /// the SSOT `normalize_param_name` + `coerce_all` so a live fragment is
    /// byte-identical to what the buffered close-time path would have produced
    /// for that field. On any failure the value falls back to the quoted raw
    /// string (still valid JSON). Returns `(norm_key, json_value_string)`.
    pub(super) fn coerce_kv(&self, raw_key: &str, raw_value: &str) -> (String, String) {
        let name = self.current_tc_name.clone().unwrap_or_default();
        // Echo-attractor salvage (2026-07-09): re-split `parameter=filePath>…`
        // shapes where the real key leaked into the value — same SSOT helper
        // as the buffered pipeline (validation.rs step 2.5). Guarded on
        // `emitted_keys` so a properly-streamed key is never duplicated in
        // the fragment JSON.
        let salvaged =
            crate::tool_parser::salvage_echoed_param(&self.tools, &name, raw_key, raw_value)
                .filter(|(real_key, _)| !self.emitted_keys.contains(real_key));
        let (raw_key, raw_value) = match &salvaged {
            Some((real_key, real_val)) => (real_key.as_str(), real_val.as_str()),
            None => (raw_key, raw_value),
        };
        let norm_key = crate::tool_parser::normalize_param_name(&self.tools, &name, raw_key);
        let fallback = || serde_json::to_string(raw_value).unwrap_or_else(|_| "\"\"".to_string());
        let single = serde_json::to_string(&serde_json::json!({ norm_key.clone(): raw_value }));
        let Ok(single_args) = single else {
            return (norm_key, fallback());
        };
        let mut tc = ToolCall {
            id: String::new(),
            call_type: "function".into(),
            function: FunctionCall {
                name,
                arguments: single_args,
            },
        };
        coerce_all(std::slice::from_mut(&mut tc), &self.tools);
        let json_value_string = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
            .ok()
            .and_then(|v| v.get(&norm_key).cloned())
            .and_then(|v| serde_json::to_string(&v).ok())
            .unwrap_or_else(fallback);
        (norm_key, json_value_string)
    }

    /// Emit any NEWLY-COMPLETE argument fragments for the in-progress tool call
    /// seen in `self.buffer[..limit]`, advancing `self.current_tc_emitted` past
    /// what was streamed. Only runs on the live (`!buffer_args`) path.
    ///
    /// XML path (`<parameter=K>V</parameter>`): each complete param becomes a
    /// coerced `"key":value` fragment (with a leading `{` or `,`). On
    /// `final_close` it backfills missing required string params and emits the
    /// closing `}`. JSON path (`"arguments": {...}`, Hermes etc.): forwards the
    /// raw JSON bytes verbatim (no coercion — these formats are already typed).
    pub(super) fn stream_ready_fragments(
        &mut self,
        limit: usize,
        final_close: bool,
    ) -> Vec<DetectorOutput> {
        let mut outputs = Vec::new();
        let idx = self.call_counter as usize;
        let scan = &self.buffer[..limit.min(self.buffer.len())];

        if scan.contains("<parameter=") {
            // ── XML path ──
            loop {
                let from = self.current_tc_emitted;
                let Some(rel_open) = self.buffer[from..limit].find("<parameter=") else {
                    break;
                };
                let open_at = from + rel_open;
                let key_region = open_at + "<parameter=".len();
                let Some(rel_gt) = self.buffer[key_region..limit].find('>') else {
                    break; // key not finished yet
                };
                let gt_at = key_region + rel_gt;
                let value_region = gt_at + 1;
                // P0-2 (2026-07-09): VIRTUAL close on the garbled
                // close-reopen signature. At depth the model can drop the
                // `>` of a close, emitting `</parameter<parameter=KEY>` —
                // grammar-legal value content the exact-literal scan would
                // swallow into the previous value (real key+path lost, then
                // backfilled as ""). Close the value at the garble and
                // resume scanning at the reopen so the next param parses.
                let exact = self.buffer[value_region..limit].find("</parameter>");
                let garbled = self.buffer[value_region..limit].find("</parameter<parameter=");
                let (rel_close, advance) = match (exact, garbled) {
                    (Some(e), Some(g)) if g < e => (g, "</parameter".len()),
                    (Some(e), _) => (e, "</parameter>".len()),
                    (None, Some(g)) => (g, "</parameter".len()),
                    (None, None) => break, // value not closed yet — not complete
                };
                let close_at = value_region + rel_close;
                // Mirror parse_single_b.rs:79-105: key + value are both `.trim()`.
                let key = self.buffer[key_region..gt_at].trim().to_string();
                let raw_value = self.buffer[value_region..close_at].trim();
                let (norm_key, json_value_string) = self.coerce_kv(&key, raw_value);
                let prefix = if !self.args_open {
                    self.args_open = true;
                    "{"
                } else {
                    ","
                };
                let quoted_key =
                    serde_json::to_string(&norm_key).unwrap_or_else(|_| "\"\"".to_string());
                let fragment = format!("{prefix}{quoted_key}:{json_value_string}");
                outputs.push(DetectorOutput::ToolCallArgsFragment { fragment, idx });
                self.incremental_emitted = true;
                self.emitted_keys.push(norm_key);
                self.current_tc_emitted = close_at + advance;
            }

            if final_close {
                // Backfill required string params the model never emitted, then
                // close the args object. Mirrors validation.rs:124-135.
                if let Some(name) = self.current_tc_name.clone()
                    && let Some(tool_def) = self.tools.iter().find(|t| t.function.name == name)
                    && let Some(params) = tool_def.function.parameters.as_ref()
                {
                    let properties = params.get("properties").and_then(|p| p.as_object());
                    let required: Vec<String> = params
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    for req in &required {
                        if self.emitted_keys.iter().any(|k| k == req) {
                            continue;
                        }
                        let is_string = properties
                            .and_then(|p| p.get(req))
                            .and_then(|v| v.get("type"))
                            .and_then(|t| t.as_str())
                            .is_none_or(|t| t == "string");
                        if !is_string {
                            continue;
                        }
                        let prefix = if !self.args_open {
                            self.args_open = true;
                            "{"
                        } else {
                            ","
                        };
                        let quoted_key =
                            serde_json::to_string(req).unwrap_or_else(|_| "\"\"".to_string());
                        outputs.push(DetectorOutput::ToolCallArgsFragment {
                            fragment: format!("{prefix}{quoted_key}:\"\""),
                            idx,
                        });
                        self.incremental_emitted = true;
                        self.emitted_keys.push(req.clone());
                    }
                }
                // Closing brace. If no params at all were streamed, emit `{}`.
                let closing = if !self.args_open {
                    self.args_open = true;
                    "{}".to_string()
                } else {
                    "}".to_string()
                };
                outputs.push(DetectorOutput::ToolCallArgsFragment {
                    fragment: closing,
                    idx,
                });
                self.incremental_emitted = true;
            }
        } else if scan.contains("\"arguments\"") {
            // ── JSON path (Hermes etc.) — forward raw bytes verbatim ──
            let args_start = super::streaming::find_args_start(&self.buffer);
            if args_start >= limit {
                return outputs;
            }
            // #192: anchor on the args object's `{`, skipping the whitespace
            // models emit after the colon (`"arguments": {`). Without this,
            // `find_balanced_json_end` (which requires a leading `{`) never
            // balances, and the "no close yet" fallback streamed EVERYTHING
            // buffered — the hermes envelope's outer `}` and the (partial)
            // `</tool_call>` tag leaked into the client's function.arguments.
            let mut args_start = args_start;
            while args_start < limit && self.buffer.as_bytes()[args_start].is_ascii_whitespace() {
                args_start += 1;
            }
            let body = &self.buffer[args_start..limit];
            let settled = if let Some(e) = find_balanced_json_end(body) {
                e
            } else {
                // No balanced close yet — stream up to a UTF-8-safe boundary.
                let mut s = body.len();
                while s > 0 && !body.is_char_boundary(s) {
                    s -= 1;
                }
                s
            };
            let already = self.current_tc_emitted.min(settled);
            let new = &body[already..settled];
            if !new.is_empty() {
                outputs.push(DetectorOutput::ToolCallArgsFragment {
                    fragment: new.to_string(),
                    idx,
                });
                self.current_tc_emitted = settled;
                self.incremental_emitted = true;
            }
        }
        outputs
    }
}

impl StreamingToolDetector {
    /// Handle a bare `<function...` block (no `<tool_call>` wrapper).
    ///
    /// Returns `true` when the caller's scan loop should `continue` (a complete
    /// block was consumed), `false` when it should `break` and wait for tokens.
    ///
    /// Split out of `streaming_impl.rs::process` to stay under the 500 LoC cap.
    ///
    /// ## Why this streams now
    /// The bare `<function=NAME>` shape is what the `qwen3_xml` parser — the
    /// shipped GB10 config — actually receives. It used to fall straight through
    /// to `break`, so a client saw NOTHING of a tool call until the whole block
    /// had been generated and parsed, while the `<tool_call>`-wrapped shape had
    /// been streaming its header and each completed parameter all along. This
    /// closes that asymmetry with the SAME machinery rather than a second copy.
    ///
    /// Measured caveat: this moves FIRST-BYTE latency only. A `stream=False` A/B
    /// showed total request latency is unchanged (+1.4 ms), so it improves
    /// responsiveness and reported TTFT, NOT wall.
    pub(super) fn process_bare_function(&mut self, outputs: &mut Vec<DetectorOutput>) -> bool {
        let Some(func_pos) = self.buffer.find("<function") else {
            return false;
        };
        if func_pos > 0 {
            let before = self.buffer[..func_pos].to_string();
            self.buffer = self.buffer[func_pos..].to_string();
            outputs.push(DetectorOutput::Content(before));
        }

        if let Some(end) = bare_function_end(&self.buffer) {
            // If this call was already streamed incrementally below, its header
            // and completed params are ALREADY on the wire. Re-emitting a whole
            // `ToolCall` here would deliver the call TWICE, so close it the way
            // the `<tool_call>` arm does: residual fragments + `ToolCallEnd`.
            // These two paths MUST stay mutually exclusive.
            if !self.buffer_args && self.incremental_emitted {
                let idx = self.call_counter as usize;
                let frags = self.stream_ready_fragments(end, true);
                outputs.extend(frags);
                outputs.push(DetectorOutput::ToolCallEnd { idx });
                self.call_counter += 1;
                self.emitted_tool_calls = true;
                self.buffer = self.buffer[end..].to_string();
                self.reset_call_state();
                return true;
            }
            let block = self.buffer[..end].to_string();
            self.buffer = self.buffer[end..].to_string();
            let (_, calls) = parse_bare_function_calls(&block);
            for tc in calls {
                let idx = self.call_counter as usize;
                self.call_counter += 1;
                self.emitted_tool_calls = true;
                outputs.push(DetectorOutput::ToolCall(tc, idx));
            }
            self.reset_call_state();
            return true;
        }

        // Block not closed yet — emit the header as soon as the name is known,
        // then any newly-complete `<parameter=...>` fragments.
        //
        // `stream_ready_fragments` scans from `current_tc_emitted` for
        // `<parameter=` / `</parameter>` pairs, so the leading `<function=NAME>`
        // still sitting in the buffer is simply skipped — no stripping needed.
        if self.current_tc_name.is_none()
            && let Some(name) = extract_streaming_name(&self.buffer)
        {
            let id = next_tool_call_id();
            let idx = self.call_counter as usize;
            outputs.push(DetectorOutput::ToolCallStart {
                id: id.clone(),
                name: name.clone(),
                idx,
            });
            self.current_tc_name = Some(name);
            self.current_tc_id = Some(id);
        }
        if !self.buffer_args && self.current_tc_name.is_some() {
            let frags = self.stream_ready_fragments(self.buffer.len(), false);
            outputs.extend(frags);
        }
        false
    }
}
