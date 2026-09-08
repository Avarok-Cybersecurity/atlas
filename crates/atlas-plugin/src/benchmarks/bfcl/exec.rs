// SPDX-License-Identifier: AGPL-3.0-only

//! The two phases that talk to the outside world: issuing one sample's
//! request, and handing the collected responses to `score.py`.
//!
//! Split out of `mod.rs` to keep that file under the repository's 500-LoC
//! cap.

use super::*;

impl Bfcl {
    pub(super) async fn generate_one(&mut self) -> Result<()> {
        let handle = self.handle()?.clone();
        let sample = self.samples[self.cursor].clone();
        let target = handle.target();
        let body = json!({
            "model": target.model,
            "stream": true,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "messages": sample.messages,
            "tools": sample.tools,
            "tool_choice": sample.tool_choice,
        });
        let outcome = http::chat_stream(target, &body, self.request_timeout).await;
        let (tool_calls, has_tool_calls) = match &outcome {
            Ok(o) => (
                o.tool_calls
                    .iter()
                    .map(|c| json!({"name": c.name, "arguments": c.arguments}))
                    .collect::<Vec<_>>(),
                !o.tool_calls.is_empty(),
            ),
            Err(e) => {
                // A transport failure is scored as "no call", which is the
                // honest reading: the endpoint produced nothing. It is also
                // logged, so a run degraded by errors is visible rather than
                // showing up only as a mysteriously low score.
                //
                // ★ AND COUNTED, because a log line does not survive into a
                // record. Serially a degraded run shows up as warnings a human
                // reads; across four shards on three boxes the degraded shard
                // merges into the aggregate invisibly, scoring its failures as
                // "made no call" — which is the CORRECT answer for most of the
                // irrelevance subsets. A shard can therefore fail its way to a
                // better number. `metrics()` publishes this so the group can
                // refuse it.
                self.transport_errors += 1;
                handle.warn(one_line(format!("sample {}: {e:#}", sample.sample_id)));
                (Vec::new(), false)
            }
        };
        if has_tool_calls {
            self.tool_call_samples += 1;
        }
        self.responses.push(json!({
            "sample_id": sample.sample_id,
            "subset": sample.subset,
            "has_tool_calls": has_tool_calls,
            "tool_calls": tool_calls,
        }));
        self.cursor += 1;
        Ok(())
    }

    pub(super) async fn score(&mut self) -> Result<Scores> {
        let artifacts = self
            .artifacts
            .clone()
            .context("artifacts were not provisioned")?;
        let path = artifacts.dir.join("responses.jsonl");
        let mut text = String::new();
        for r in &self.responses {
            text.push_str(&serde_json::to_string(r)?);
            text.push('\n');
        }
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        self.responses_path = Some(path.clone());

        let out = crate::python::run(
            &artifacts.python,
            &[
                artifacts.scorer.to_str().context("scorer path")?,
                "--dataset",
                artifacts.dataset.to_str().context("dataset path")?,
                "--responses",
                path.to_str().context("responses path")?,
            ],
            Some(&artifacts.dir),
        )
        .await
        .context("scoring failed — responses.jsonl is kept, so this can be rescored")?;
        serde_json::from_str(out.stdout.trim())
            .with_context(|| format!("scorer printed unexpected output: {}", out.stdout))
    }
}
