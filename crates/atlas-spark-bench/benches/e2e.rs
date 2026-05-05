// SPDX-License-Identifier: AGPL-3.0-only

use std::time::Duration;

use atlas_spark_bench::{require_server, send_blocking, send_streaming, short_prompt};
use criterion::{Criterion, criterion_group, criterion_main};

fn e2e_latency(c: &mut Criterion) {
    let url = require_server();
    let mut group = c.benchmark_group("e2e_latency");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function("blocking_150tok", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let r = send_blocking(&url, short_prompt(), 150).expect("blocking request failed");
                total += r.elapsed;
            }
            total
        });
    });

    group.bench_function("streaming_150tok", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let r =
                    send_streaming(&url, short_prompt(), 150).expect("streaming request failed");
                total += r.total_duration;
            }
            total
        });
    });

    group.finish();
}

criterion_group!(benches, e2e_latency);
criterion_main!(benches);
