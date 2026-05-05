// SPDX-License-Identifier: AGPL-3.0-only

use std::time::Duration;

use atlas_spark_bench::{require_server, send_concurrent_streaming};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn concurrent_throughput(c: &mut Criterion) {
    let url = require_server();
    let mut group = c.benchmark_group("concurrent_throughput");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    let prompt = "Explain the theory of general relativity in detail.";
    let max_tokens: usize = 150;

    for concurrency in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements((concurrency * max_tokens) as u64));
        group.bench_with_input(
            BenchmarkId::new("C", concurrency),
            &concurrency,
            |b, &c_level| {
                b.iter_custom(|iters| {
                    let mut total_wall = Duration::ZERO;
                    for _ in 0..iters {
                        let results = send_concurrent_streaming(&url, prompt, max_tokens, c_level);
                        let wall = results
                            .iter()
                            .filter_map(|r| r.as_ref().ok())
                            .map(|r| r.total_duration)
                            .max()
                            .unwrap_or_default();
                        total_wall += wall;
                    }
                    total_wall
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, concurrent_throughput);
criterion_main!(benches);
