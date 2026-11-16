// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. A plain OpenAI-style chat client for the routing gate; no upstream code.
//! Responsibility: collect the 4B's greedy replies from a running Atlas server (`/v1/chat/completions`, temperature 0,
//! thinking off by default), resumable: finished prompts are skipped and failed ones are retried. This is the only
//! network code in the study and it talks only to the server it is given.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde_json::{Value, json};

use super::{VERSION, max_tokens};

/// Arguments of `collect`.
pub struct CollectArgs {
    pub prompts: PathBuf,
    pub url: String,
    pub model: String,
    pub out: PathBuf,
    pub concurrency: usize,
    pub thinking: bool,
    pub limit: usize,
    pub timeout: f64,
    pub max_tokens_scale: f64,
}

/// Reads a `.jsonl` file of objects.
pub(crate) fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    std::fs::read_to_string(path)
        .with_context(|| format!("read {}", path.display()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(Into::into))
        .collect()
}

/// Finished records keyed by id (a later line replaces an earlier one).
pub(crate) fn load_done(path: &Path) -> Result<HashMap<String, Value>> {
    let mut done = HashMap::new();
    if path.exists() {
        for r in read_jsonl(path)? {
            if let Some(id) = r["id"].as_str() {
                done.insert(id.to_string(), r);
            }
        }
    }
    Ok(done)
}

/// One chat request; returns `{text, finish_reason, usage, latency_s}`.
pub fn post_chat(
    agent: &ureq::Agent,
    url: &str,
    model: &str,
    messages: &Value,
    max_tokens: usize,
    thinking: bool,
) -> Result<Value> {
    let body = json!({
        "model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    });
    let t0 = Instant::now();
    let resp = agent
        .post(&format!(
            "{}/v1/chat/completions",
            url.trim_end_matches('/')
        ))
        .send_json(&body)?;
    let (_, mut resp_body) = resp.into_parts();
    let d: Value = resp_body.read_json()?;
    let ch = &d["choices"][0];
    Ok(json!({
        "text": ch["message"]["content"].as_str().unwrap_or_default(),
        "finish_reason": ch["finish_reason"], "usage": d.get("usage").cloned().unwrap_or_else(|| json!({})),
        "latency_s": t0.elapsed().as_secs_f64(),
    }))
}

fn agent(timeout: f64) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs_f64(timeout)))
            .build(),
    )
}

/// `collect`: returns the exit code (1 when any request ended in an error).
pub fn cmd_collect(a: &CollectArgs) -> Result<i32> {
    let mut items = read_jsonl(&a.prompts)?;
    if a.limit > 0 {
        items.truncate(a.limit);
    }
    let done = load_done(&a.out)?;
    let todo: Vec<&Value> = items
        .iter()
        .filter(|it| {
            let id = it["id"].as_str().unwrap_or_default();
            done.get(id).is_none_or(|d| !d["error"].is_null())
        })
        .collect();
    println!(
        "prep collect: {} prompts, {} already done, {} to run, concurrency {}, thinking {}",
        items.len(),
        items.len() - todo.len(),
        todo.len(),
        a.concurrency,
        if a.thinking { "on" } else { "off" }
    );
    let base = a.url.trim_end_matches('/').to_string();
    let mut props = Value::Null;
    for ep in ["/props", "/v1/models"] {
        let probe = agent(10.0).get(&format!("{base}{ep}")).call();
        if let Ok(resp) = probe
            && let Ok(v) = resp.into_parts().1.read_json::<Value>()
        {
            props = v;
            break;
        }
    }
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    let meta = json!({
        "tool": VERSION, "url": a.url, "model": a.model, "thinking": a.thinking,
        "max_tokens_scale": a.max_tokens_scale, "started_unix": started, "props": props,
    });
    let mut meta_path = a.out.clone().into_os_string();
    meta_path.push(".meta.json");
    std::fs::write(meta_path, serde_json::to_string_pretty(&meta)?)?;

    let out = Mutex::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&a.out)?,
    );
    let (count, errors) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let client = agent(a.timeout);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a.concurrency.max(1))
        .build()?;
    pool.install(|| {
        todo.par_iter().for_each(|it| {
            let mt = ((max_tokens(it["kind"].as_str().unwrap_or_default()) as f64
                * a.max_tokens_scale) as usize)
                .max(16);
            let mut rec = json!({"id": it["id"], "max_tokens": mt, "error": Value::Null});
            for attempt in 0..3u64 {
                match post_chat(&client, &a.url, &a.model, &it["messages"], mt, a.thinking) {
                    Ok(Value::Object(fields)) => {
                        for (k, v) in fields {
                            rec[k] = v;
                        }
                        rec["error"] = Value::Null;
                        break;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        let mut msg = format!("{e}");
                        msg.truncate(300);
                        rec["error"] = Value::String(msg);
                        std::thread::sleep(Duration::from_secs(2 * (attempt + 1)));
                    }
                }
            }
            let failed = !rec["error"].is_null();
            if let Ok(mut f) = out.lock() {
                let _ = writeln!(f, "{}", serde_json::to_string(&rec).unwrap_or_default());
                let _ = f.flush();
            }
            let c = count.fetch_add(1, Ordering::SeqCst) + 1;
            let e = errors.fetch_add(usize::from(failed), Ordering::SeqCst) + usize::from(failed);
            if c.is_multiple_of(20) {
                println!("  {c}/{} done, {e} errors", todo.len());
            }
        });
    });
    let (c, e) = (count.load(Ordering::SeqCst), errors.load(Ordering::SeqCst));
    println!(
        "prep collect: wrote {c} replies to {}, {e} with errors",
        a.out.display()
    );
    Ok(i32::from(e > 0))
}
