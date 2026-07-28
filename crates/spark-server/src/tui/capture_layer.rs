// SPDX-License-Identifier: AGPL-3.0-only

//! Decodes `atlas::tui::progress` events into typed [`ProgressEvent`]s.
//!
//! Attached with an always-on per-layer filter for exactly that target, so
//! progress flows regardless of `RUST_LOG` (the events are `debug!` level and
//! invisible to the fmt layer at the default `info` filter — the plain-log
//! grep contract is untouched).

use std::sync::mpsc::Sender;

use tracing::field::{Field, Visit};

/// Typed startup-progress event. Mirrors the schema in
/// `spark_runtime::progress` (the `ev` field discriminates).
#[derive(Clone, Debug, PartialEq)]
pub enum ProgressEvent {
    Phase {
        phase: u8,
        name: String,
    },
    Preflight {
        disk_gb: f64,
        free_gb: f64,
    },
    ShardStart {
        shard: u64,
        total: u64,
        name: String,
    },
    ShardDone {
        shard: u64,
        total: u64,
        used_gb: f64,
        free_gb: f64,
    },
    Layer {
        layer: u64,
        total: u64,
    },
    Ready {
        port: u16,
    },
}

/// Field bag filled by the visitor; `decode` shapes it by `ev`.
#[derive(Default)]
struct Fields {
    ev: String,
    name: String,
    phase: u64,
    shard: u64,
    total: u64,
    layer: u64,
    port: u64,
    disk_gb: f64,
    free_gb: f64,
    used_gb: f64,
}

impl Visit for Fields {
    fn record_u64(&mut self, field: &Field, value: u64) {
        match field.name() {
            "phase" => self.phase = value,
            "shard" => self.shard = value,
            "total" => self.total = value,
            "layer" => self.layer = value,
            "port" => self.port = value,
            _ => {}
        }
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_u64(field, value.max(0) as u64);
    }
    fn record_f64(&mut self, field: &Field, value: f64) {
        match field.name() {
            "disk_gb" => self.disk_gb = value,
            "free_gb" => self.free_gb = value,
            "used_gb" => self.used_gb = value,
            _ => {}
        }
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "ev" => self.ev = value.to_string(),
            "name" => self.name = value.to_string(),
            _ => {}
        }
    }
    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

fn decode(f: Fields) -> Option<ProgressEvent> {
    Some(match f.ev.as_str() {
        "phase" => ProgressEvent::Phase {
            phase: f.phase.min(u8::MAX as u64) as u8,
            name: f.name,
        },
        "preflight" => ProgressEvent::Preflight {
            disk_gb: f.disk_gb,
            free_gb: f.free_gb,
        },
        "shard_start" => ProgressEvent::ShardStart {
            shard: f.shard,
            total: f.total,
            name: f.name,
        },
        "shard_done" => ProgressEvent::ShardDone {
            shard: f.shard,
            total: f.total,
            used_gb: f.used_gb,
            free_gb: f.free_gb,
        },
        "layer" => ProgressEvent::Layer {
            layer: f.layer,
            total: f.total,
        },
        "ready" => ProgressEvent::Ready {
            port: f.port.min(u16::MAX as u64) as u16,
        },
        _ => return None,
    })
}

/// The layer. Send end of a channel drained by the TUI thread each tick.
/// Sends never block; if the TUI is gone the error is ignored (server keeps
/// serving without a dashboard).
pub struct ProgressCaptureLayer {
    tx: Sender<ProgressEvent>,
}

impl ProgressCaptureLayer {
    pub fn new(tx: Sender<ProgressEvent>) -> Self {
        Self { tx }
    }
}

impl<S> tracing_subscriber::Layer<S> for ProgressCaptureLayer
where
    S: tracing::Subscriber,
{
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        if event.metadata().target() != spark_runtime::progress::TARGET {
            return;
        }
        let mut f = Fields::default();
        event.record(&mut f);
        if let Some(ev) = decode(f) {
            let _ = self.tx.send(ev);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    #[test]
    fn decodes_progress_events_and_ignores_others() {
        let (tx, rx) = std::sync::mpsc::channel();
        let sub = tracing_subscriber::registry().with(ProgressCaptureLayer::new(tx));
        tracing::subscriber::with_default(sub, || {
            spark_runtime::progress::phase(3, "gpu init");
            spark_runtime::progress::shard_start(2, 26, "model-00002.safetensors");
            tracing::info!("an ordinary log line");
            spark_runtime::progress::ready(8888);
        });
        let got: Vec<ProgressEvent> = rx.try_iter().collect();
        assert_eq!(
            got,
            vec![
                ProgressEvent::Phase {
                    phase: 3,
                    name: "gpu init".into()
                },
                ProgressEvent::ShardStart {
                    shard: 2,
                    total: 26,
                    name: "model-00002.safetensors".into()
                },
                ProgressEvent::Ready { port: 8888 },
            ]
        );
    }
}
