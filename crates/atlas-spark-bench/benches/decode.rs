// SPDX-License-Identifier: AGPL-3.0-only

use std::time::Duration;

use atlas_spark_bench::{require_server, send_streaming};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn decode_throughput(c: &mut Criterion) {
    let url = require_server();
    let mut group = c.benchmark_group("decode_throughput");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    let prompt = "Explain the theory of general relativity in detail.";

    for max_tokens in [50, 100, 150, 200] {
        group.throughput(Throughput::Elements(max_tokens as u64));
        group.bench_with_input(
            BenchmarkId::new("tokens", max_tokens),
            &max_tokens,
            |b, &mt| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let r = send_streaming(&url, prompt, mt).expect("streaming request failed");
                        // Use decode_duration only if model generated enough tokens
                        if r.token_count >= 2 {
                            total += r.decode_duration;
                        } else {
                            total += r.total_duration;
                        }
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, decode_throughput);
criterion_main!(benches);
